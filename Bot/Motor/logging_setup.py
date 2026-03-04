from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from log_config import LOG_DIR as LOG_ROOT, get_log_path

LOG_DIR = LOG_ROOT

# Diagnostics toggles
_DIAG_ENABLED : bool = True
_DIAG_EVERY_SEC : float = 60.0

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

# Route core risk logs to health file
log_core_risk = logging.getLogger("core.portfolio_risk")
log_core_risk.setLevel(logging.INFO)
log_core_risk.propagate = False
log_core_feature = logging.getLogger("core.feature_engine")
log_core_feature.setLevel(logging.ERROR)
log_core_feature.propagate = False
log_core_signal = logging.getLogger("core.signal_engine")
log_core_signal.setLevel(logging.ERROR)
log_core_signal.propagate = False
log_core_risk_engine = logging.getLogger("core.risk_engine")
log_core_risk_engine.setLevel(logging.ERROR)
log_core_risk_engine.propagate = False

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
_add_handler(log_core_risk, "portfolio_engine_health.log", logging.INFO)
_add_handler(log_core_feature, "portfolio_engine_error.log", logging.ERROR)
_add_handler(log_core_signal, "portfolio_engine_error.log", logging.ERROR)
_add_handler(log_core_risk_engine, "portfolio_engine_error.log", logging.ERROR)
if _DIAG_ENABLED:
    _add_handler(log_diag, "portfolio_engine_diag.jsonl", logging.INFO)
