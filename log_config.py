"""Centralized logging configuration for the Exness trading stack.

All modules should import LOG_DIR (or helpers) from here to ensure
consistent log paths under the project root (./Logs by default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = ["LOG_DIR", "get_log_path", "log_dir_stats"]

_BASE_DIR: Final[Path] = Path(__file__).resolve().parent

LOG_DIR = (_BASE_DIR / "Logs").resolve()

LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_log_path(*parts: str) -> Path:
    """Return a path inside LOG_DIR, creating parent directories if needed."""

    path = LOG_DIR.joinpath(*parts)
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
