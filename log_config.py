"""Centralized logging configuration for the Exness trading stack.

All modules should import LOG_DIR (or helpers) from here to ensure
consistent log paths under the project root (./Logs by default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable, Optional, Union

import logging
import sys
from logging.handlers import RotatingFileHandler

__all__ = [
    "LOG_DIR",
    "ARTIFACTS_DIR",
    "get_log_path",
    "get_artifact_path",
    "get_artifact_dir",
    "log_dir_stats",
    "configure_logging",
    "attach_global_handler_to_loggers",
]

_BASE_DIR: Final[Path] = Path(__file__).resolve().parent

LOG_DIR = (_BASE_DIR / "Logs").resolve()
ARTIFACTS_DIR = (_BASE_DIR / "Artifacts").resolve()

LOG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def configure_logging(
    *,
    level: Union[str, int] = "INFO",
    system_log_name: str = "system.log",
    console: bool = True,
    max_bytes: int = 15 * 1024 * 1024,
    backup_count: int = 7,
) -> RotatingFileHandler:
    """
    Global system logger:
    - Root logger at INFO level
    - Rotating file handler at Logs/system.log
    - Optional console handler
    - Captures warnings
    """
    root = logging.getLogger()
    root.setLevel(level)

    log_path = get_log_path(system_log_name)
    handler = RotatingFileHandler(
        str(log_path),
        maxBytes=int(max_bytes),
        backupCount=int(backup_count),
        encoding="utf-8",
        delay=True,
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    # Avoid duplicate handlers
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == handler.baseFilename for h in root.handlers):
        root.addHandler(handler)

    if console:
        if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
            ch = logging.StreamHandler(stream=getattr(sys, "__stdout__", sys.stdout))
            ch.setLevel(level)
            ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            root.addHandler(ch)

    logging.captureWarnings(True)

    # Attach to existing non-propagating loggers so they still show in system.log
    attach_global_handler_to_loggers(handler)
    return handler


def attach_global_handler_to_loggers(
    handler: logging.Handler,
    names: Optional[Iterable[str]] = None,
) -> None:
    """
    Attach a global handler to existing loggers with propagate=False.
    If names provided, only matches exact names or name-prefixes.
    """
    if handler is None:
        return
    prefixes = list(names or [])
    for name, obj in logging.root.manager.loggerDict.items():
        if not isinstance(obj, logging.Logger):
            continue
        if prefixes:
            match = False
            for p in prefixes:
                if name == p or name.startswith(f"{p}."):
                    match = True
                    break
            if not match:
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
                    count += 1
                    total += int(p.stat().st_size)
            except Exception:
                continue
    except Exception:
        pass
    return total, count
