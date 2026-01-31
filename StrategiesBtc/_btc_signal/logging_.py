# _btc_signal/logging_.py — from StrategiesBtc btc_signal_engine (verified)
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from log_config import get_log_path

log = logging.getLogger("signal_btc")
log.setLevel(logging.ERROR)
log.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in log.handlers):
    fh = RotatingFileHandler(
        filename=str(get_log_path("signal_engine_btc.log")),
        maxBytes=int(5_242_880),
        backupCount=int(5),
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    log.addHandler(fh)
