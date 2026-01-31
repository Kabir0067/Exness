from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from log_config import LOG_DIR as LOG_ROOT, get_log_path

LOG_DIR = LOG_ROOT

# Diagnostics toggles
_DIAG_ENABLED = bool(int(os.getenv("PORTFOLIO_DIAG_ENABLED", "1") or "1"))
_DIAG_EVERY_SEC = float(os.getenv("PORTFOLIO_DIAG_EVERY_SEC", "60.0") or "60.0")

# Loggers
log_health = logging.getLogger("portfolio.engine.health")
log_err = logging.getLogger("portfolio.engine.err")
log_diag = logging.getLogger("portfolio.engine.diag")

log_health.setLevel(logging.INFO)
log_err.setLevel(logging.ERROR)
log_diag.setLevel(logging.INFO)

log_health.propagate = False
log_err.propagate = False
log_diag.propagate = False

_FMT = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _add_handler(logger: logging.Logger, filename: str, level: int) -> None:
    if logger.handlers:
        return
    try:
        path = str(get_log_path(filename))
    except Exception:
        # fallback
        os.makedirs(str(LOG_DIR), exist_ok=True)
        path = os.path.join(str(LOG_DIR), filename)

    h = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=5, encoding="utf-8", delay=True)
    h.setLevel(level)
    h.setFormatter(_FMT)
    logger.addHandler(h)


_add_handler(log_health, "portfolio_engine_health.log", logging.INFO)
_add_handler(log_err, "portfolio_engine_error.log", logging.ERROR)
if _DIAG_ENABLED:
    _add_handler(log_diag, "portfolio_engine_diag.jsonl", logging.INFO)
