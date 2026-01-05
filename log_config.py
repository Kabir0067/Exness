"""Centralized logging configuration for the Exness trading stack.

All modules should import LOG_DIR (or helpers) from here to ensure
consistent log paths under the project root (./Logs by default).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

__all__ = ["LOG_DIR", "get_log_path"]

_BASE_DIR: Final[Path] = Path(__file__).resolve().parent

# Allow overriding via environment variable, otherwise default to ./Logs
_LOG_DIR_ENV = os.getenv("EXNESS_LOG_DIR")
if _LOG_DIR_ENV:
    LOG_DIR: Final[Path] = Path(_LOG_DIR_ENV).expanduser().resolve()
else:
    LOG_DIR = (_BASE_DIR / "Logs").resolve()

LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_log_path(*parts: str) -> Path:
    """Return a path inside LOG_DIR, creating parent directories if needed."""

    path = LOG_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
