# _btc_indicators/logging_.py — unchanged (OK)
from __future__ import annotations

import logging

from log_config import LOG_DIR as LOG_ROOT, get_log_path

LOG_DIR = LOG_ROOT

logger = logging.getLogger("indicators_btc")
logger.setLevel(logging.ERROR)
logger.propagate = False

if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler(
        str(get_log_path("feature_engine_btc.log")),
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    logger.addHandler(fh)
