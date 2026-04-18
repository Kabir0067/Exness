"""
Centralized logging configuration for the Exness trading stack.

All modules should import LOG_DIR (or helpers) to ensure
consistent log paths under the project root (./Logs by default).
"""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final, Iterable, Optional, Union

# =============================================================================
# Paths
# =============================================================================
_BASE_DIR: Final[Path] = Path(__file__).resolve().parent

LOG_DIR: Final[Path] = (_BASE_DIR / "Logs").resolve()
ARTIFACTS_DIR: Final[Path] = (_BASE_DIR / "Artifacts").resolve()

# Ensure required directories exist; never fail during import
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass


# =============================================================================
# Global Constants
# =============================================================================
_CFG_LOCK = threading.Lock()


# =============================================================================
# Public API
# =============================================================================
def configure_logging(
    *,
    level: Union[str, int] = "INFO",
    system_log_name: Optional[str] = None,
    console: bool = True,
    max_bytes: int = 15 * 1024 * 1024,
    backup_count: int = 7,
) -> Optional[RotatingFileHandler]:
    """
    Initialize global logging configuration.

    Idempotent, thread-safe, and production-hardened.
    """
    lvl = _as_level(level)

    with _CFG_LOCK:
        root = logging.getLogger()

        try:
            root.setLevel(lvl)
        except Exception:
            root.setLevel(logging.INFO)

        file_handler: Optional[RotatingFileHandler] = None

        if system_log_name:
            try:
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

            except Exception as exc:
                logging.getLogger(__name__).error(
                    "Failed to configure file handler: %s", exc
                )

        if console:
            try:
                ch = _find_console_handler(root)
                if ch is None:
                    ch = logging.StreamHandler(stream=_stdout_stream())
                    root.addHandler(ch)

                ch.setLevel(lvl)
                ch.setFormatter(_fmt())

            except Exception as exc:
                logging.getLogger(__name__).error(
                    "Failed to configure console handler: %s", exc
                )

        try:
            logging.captureWarnings(True)
        except Exception:
            pass

        try:
            attach_global_handler_to_loggers(file_handler)
        except Exception as exc:
            logging.getLogger(__name__).error(
                "attach_global_handler_to_loggers failed: %s", exc
            )

        return file_handler


def attach_global_handler_to_loggers(
    handler: Optional[logging.Handler],
    names: Optional[Iterable[str]] = None,
) -> None:
    """Attach a handler to existing loggers with propagate=False."""
    if handler is None:
        return

    prefixes = tuple(names or ())

    for name, obj in logging.root.manager.loggerDict.items():
        try:
            if not isinstance(obj, logging.Logger):
                continue

            if prefixes:
                matched = False
                for p in prefixes:
                    if name == p or name.startswith(f"{p}."):
                        matched = True
                        break
                if not matched:
                    continue

            if obj.propagate:
                continue

            if handler in obj.handlers:
                continue

            obj.addHandler(handler)

        except Exception:
            continue


def get_log_path(*parts: str) -> Path:
    """Return a file path within LOG_DIR."""
    try:
        path = LOG_DIR.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return LOG_DIR / "fallback.log"


def build_logger(name: str, filename: str, level: int = logging.INFO) -> logging.Logger:
    """Create or reuse a rotating file logger instance."""
    logger = logging.getLogger(name)

    try:
        logger.setLevel(int(level))
    except Exception:
        logger.setLevel(logging.INFO)

    logger.propagate = False

    log_path = str(get_log_path(filename))
    existing = None

    for handler in logger.handlers:
        try:
            if (
                isinstance(handler, RotatingFileHandler)
                and getattr(handler, "baseFilename", "") == log_path
            ):
                existing = handler
                break
        except Exception:
            continue

    if existing is None:
        try:
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
        except Exception as exc:
            logging.getLogger(__name__).error(
                "Failed to create logger handler: %s", exc
            )
    else:
        try:
            existing.setLevel(int(level))
            existing.setFormatter(_fmt())
        except Exception:
            pass

    return logger


def get_artifact_dir(*parts: str) -> Path:
    """Return a directory within ARTIFACTS_DIR."""
    try:
        path = ARTIFACTS_DIR.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return ARTIFACTS_DIR


def get_artifact_path(*parts: str) -> Path:
    """Return a file path within ARTIFACTS_DIR."""
    try:
        path = ARTIFACTS_DIR.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return ARTIFACTS_DIR / "fallback.dat"


def log_dir_stats() -> tuple[int, int]:
    """Compute total size (bytes) and file count within LOG_DIR."""
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
        return 0, 0

    return total, count


# =============================================================================
# Private Helpers
# =============================================================================
def _stdout_stream():
    """Guarantee a valid stdout stream fallback."""
    try:
        return getattr(sys, "__stdout__", sys.stdout)
    except Exception:
        return sys.stdout


def _fmt() -> logging.Formatter:
    """Return the standard log formatter."""
    return logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _as_level(level: Union[str, int]) -> Union[str, int]:
    """Normalize and validate log level defensively."""
    try:
        if isinstance(level, int):
            return int(level)

        s = str(level).upper().strip()
        if s.isdigit():
            return int(s)

        if s in logging._nameToLevel:  # type: ignore[attr-defined]
            return s

        raise ValueError(f"Invalid log level: {level!r}")
    except Exception as exc:
        logging.getLogger(__name__).error("Invalid log level: %s", exc)
        return "INFO"


def _find_file_handler(
    root: logging.Logger, base_filename: str
) -> Optional[RotatingFileHandler]:
    """Locate existing rotating file handler by filename."""
    for handler in root.handlers:
        try:
            if (
                isinstance(handler, RotatingFileHandler)
                and getattr(handler, "baseFilename", "") == base_filename
            ):
                return handler
        except Exception:
            continue
    return None


def _find_console_handler(
    root: logging.Logger,
) -> Optional[logging.StreamHandler]:
    """Identify console handler while excluding file handlers."""
    for handler in root.handlers:
        try:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                return handler
        except Exception:
            continue
    return None


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "ARTIFACTS_DIR",
    "LOG_DIR",
    "attach_global_handler_to_loggers",
    "build_logger",
    "configure_logging",
    "get_artifact_dir",
    "get_artifact_path",
    "get_log_path",
    "log_dir_stats",
]
