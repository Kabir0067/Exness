from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np


def _project_root(max_levels: int = 6) -> Path:
    """Resolve project root so every module writes into the same Logs/ directory.

    Strategy:
    - Walk up a few parents.
    - If we see typical marker files (config.py/mt5_client.py) or dirs (Bot/Strategies/etc),
      treat that folder as project root.
    - Otherwise fallback to current working directory.
    """
    here = Path(__file__).resolve()

    marker_files = ("config.py", "mt5_client.py")
    marker_dirs = ("Bot", "Strategies", "ExnessAPI", "DataFeed")

    parents = [here.parent] + list(here.parents)
    for p in parents[: max(1, int(max_levels))]:
        try:
            if any((p / mf).is_file() for mf in marker_files):
                return p
            if any((p / md).is_dir() for md in marker_dirs):
                return p
        except Exception:
            continue

    return Path.cwd()


# ------------------------------------------------------------
# Logging (error-only + rotation)
# ------------------------------------------------------------
LOG_DIR = _project_root() / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

UTILS_LOG_PATH = LOG_DIR / "utils.log"


def _ensure_rotating_file_handler(
    logger: logging.Logger,
    path: Path,
    level: int,
    fmt: str,
    *,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> None:
    """Attach a rotating file handler once (idempotent)."""
    logger.setLevel(level)
    logger.propagate = False

    target = path.resolve()
    for h in list(logger.handlers):
        if isinstance(h, (logging.FileHandler, RotatingFileHandler)):
            try:
                base = Path(getattr(h, "baseFilename", "")).resolve()
                if base == target:
                    h.setLevel(level)
                    return
            except Exception:
                continue

    rh = RotatingFileHandler(
        str(path),
        maxBytes=int(max_bytes),
        backupCount=int(backup_count),
        encoding="utf-8",
        delay=True,
    )
    rh.setLevel(level)
    rh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(rh)


log_utils = logging.getLogger("utils")
_ensure_rotating_file_handler(
    log_utils,
    UTILS_LOG_PATH,
    logging.ERROR,
    "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s",
)


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def safe_last(x: Any, default: float = 0.0) -> float:
    """Return last finite value from an array-like; otherwise default."""
    try:
        if x is None:
            return float(default)

        arr = np.asarray(x)
        if arr.size == 0:
            return float(default)

        v = float(arr[-1])
        if np.isfinite(v):
            return v

        log_utils.error("safe_last non-finite | v=%s", v)
        return float(default)
    except Exception as exc:
        log_utils.error("safe_last error: %s", exc)
        return float(default)


def percentile_rank(series: Any, value: float) -> float:
    """Percentile rank of `value` within finite values of `series`.

    Returns 50.0 on empty/error.
    """
    try:
        arr = np.asarray(series, dtype=np.float64)
        if arr.size == 0:
            return 50.0

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 50.0

        v = float(value)
        less_eq = float(np.sum(finite <= v))
        return float(100.0 * less_eq / float(finite.size))
    except Exception as exc:
        log_utils.error("percentile_rank error: %s", exc)
        return 50.0


def format_usdt(amount: float) -> str:
    """Format money-like value with 2 decimals; safe for NaN/inf."""
    try:
        a = float(amount)
        if not np.isfinite(a):
            log_utils.error("format_usdt non-finite | amount=%s", amount)
            return "$0.00"
        return f"${a:,.2f}"
    except Exception as exc:
        log_utils.error("format_usdt error: %s", exc)
        return "$0.00"


__all__ = ["safe_last", "percentile_rank", "format_usdt"]
