# StrategiesBtc/btc_signal_engine.py  (REFactored / API-compatible wrapper)
from __future__ import annotations

from ._btc_signal.models import SignalResult
from ._btc_signal.signal_engine import SignalEngine

__all__ = ["SignalEngine", "SignalResult"]
