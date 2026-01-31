from __future__ import annotations

import logging

from log_config import LOG_DIR as LOG_ROOT, get_log_path

# ============================================================
# ERROR-only logging (isolated file, no collision)
# ============================================================

LOG_DIR = LOG_ROOT

log = logging.getLogger("signal_xau")
log.setLevel(logging.ERROR)
log.propagate = False

if not log.handlers:
    fh = logging.FileHandler(
        str(get_log_path("signal_engine_xau.log")),
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
        )
    )
    log.addHandler(fh)
