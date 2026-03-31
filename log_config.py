"""
Centralized logging configuration for the Exness trading stack.

All modules should import LOG_DIR (or helpers) from here to ensure
consistent log paths under the project root (./Logs by default).
"""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final, Iterable, Optional, Union

__all__ = [
    "LOG_DIR",
    "ARTIFACTS_DIR",
    "get_log_path",
    "build_logger",
    "get_artifact_path",
    "get_artifact_dir",
    "log_dir_stats",
    "configure_logging",
    "attach_global_handler_to_loggers",
]

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
_BASE_DIR: Final[Path] = Path(__file__).resolve().parent

LOG_DIR: Final[Path] = (_BASE_DIR / "Logs").resolve()
ARTIFACTS_DIR: Final[Path] = (_BASE_DIR / "Artifacts").resolve()

LOG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Internal helpers (safe + idempotent)
# -----------------------------------------------------------------------------
_CFG_LOCK = threading.Lock()


def _stdout_stream():
    # ensures console output even if sys.stdout is redirected
    return getattr(sys, "__stdout__", sys.stdout)


def _fmt() -> logging.Formatter:
    return logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _as_level(level: Union[str, int]) -> Union[str, int]:
    # Keep compatibility with logging.setLevel accepting str/int.
    # But validate unknown strings early (prevents silent misconfig).
    if isinstance(level, int):
        return int(level)
    s = str(level).upper().strip()
    if s.isdigit():
        return int(s)
    if s in logging._nameToLevel:  # type: ignore[attr-defined]
        return s
    raise ValueError(f"Invalid log level: {level!r}")


def _find_file_handler(root: logging.Logger, base_filename: str) -> Optional[RotatingFileHandler]:
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == base_filename:
            return h
    return None


def _find_console_handler(root: logging.Logger) -> Optional[logging.StreamHandler]:
    # CRITICAL FIX: FileHandler is also a StreamHandler -> must exclude FileHandler
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            return h
    return None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def configure_logging(
    *,
    level: Union[str, int] = "INFO",
    system_log_name: Optional[str] = None,
    console: bool = True,
    max_bytes: int = 15 * 1024 * 1024,
    backup_count: int = 7,
) -> Optional[RotatingFileHandler]:
    """
    Global logger bootstrap (idempotent + production-safe):
    - Root logger configured to `level`
    - Optional rotating file handler (disabled when system_log_name is None/empty)
    - Optional console handler (stdout)
    - Captures warnings via logging.captureWarnings(True)

    Returns the rotating file handler instance if enabled, else None.
    """
    lvl = _as_level(level)

    with _CFG_LOCK:
        root = logging.getLogger()
        root.setLevel(lvl)

        file_handler: Optional[RotatingFileHandler] = None
        if system_log_name:
            log_path = get_log_path(system_log_name)
            base_fn = str(log_path)

            existing = _find_file_handler(root, base_fn)
            if existing is not None:
                existing.setLevel(lvl)
                existing.setFormatter(_fmt())
                file_handler = existing
            else:
                fh = RotatingFileHandler(
                    base_fn,
                    maxBytes=int(max_bytes),
                    backupCount=int(backup_count),
                    encoding="utf-8",
                    delay=True,
                )
                fh.setLevel(lvl)
                fh.setFormatter(_fmt())
                root.addHandler(fh)
                file_handler = fh

        if console:
            ch = _find_console_handler(root)
            if ch is None:
                ch = logging.StreamHandler(stream=_stdout_stream())
                root.addHandler(ch)
            ch.setLevel(lvl)
            ch.setFormatter(_fmt())

        logging.captureWarnings(True)

        # Attach file handler to non-propagating loggers only when file handler exists.
        attach_global_handler_to_loggers(file_handler)

        return file_handler


def attach_global_handler_to_loggers(
    handler: Optional[logging.Handler],
    names: Optional[Iterable[str]] = None,
) -> None:
    """
    Attach a global handler to existing loggers with propagate=False.
    If names provided, matches exact names or name-prefixes (name or name.*).
    """
    if handler is None:
        return

    prefixes = tuple(names or ())
    for name, obj in logging.root.manager.loggerDict.items():
        if not isinstance(obj, logging.Logger):
            continue

        if prefixes:
            ok = False
            for p in prefixes:
                if name == p or name.startswith(f"{p}."):
                    ok = True
                    break
            if not ok:
                continue

        if obj.propagate:
            continue
        if handler in obj.handlers:
            continue
        obj.addHandler(handler)


def get_log_path(*parts: str) -> Path:
    """Return a path inside LOG_DIR, creating parent directories if needed."""
    path = LOG_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_logger(name: str, filename: str, level: int = logging.INFO) -> logging.Logger:
    """Create or reuse a rotating file logger under the shared Logs directory."""
    logger = logging.getLogger(name)
    logger.setLevel(int(level))
    logger.propagate = False

    log_path = str(get_log_path(filename))
    existing = None
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", "") == log_path:
            existing = handler
            break

    if existing is None:
        handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
            delay=True,
        )
        handler.setLevel(int(level))
        handler.setFormatter(_fmt())
        logger.addHandler(handler)
    else:
        existing.setLevel(int(level))
        existing.setFormatter(_fmt())

    return logger


def get_artifact_dir(*parts: str) -> Path:
    """Return a directory inside ARTIFACTS_DIR, creating it if needed."""
    path = ARTIFACTS_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_artifact_path(*parts: str) -> Path:
    """Return a file path inside ARTIFACTS_DIR, creating parent dirs if needed."""
    path = ARTIFACTS_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def log_dir_stats() -> tuple[int, int]:
    """
    Returns (total_bytes, file_count) for LOG_DIR.
    Best-effort and safe for monitoring.
    """
    total = 0
    count = 0
    try:
        for p in LOG_DIR.rglob("*"):
            try:
                if p.is_file():
                    st = p.stat()
                    count += 1
                    total += int(st.st_size)
            except Exception:
                continue
    except Exception:
        pass
    return total, count
