# StrategiesXau/xau_signal_engine.py  (REFactored / API-compatible wrapper)
from __future__ import annotations

from ._xau_signal.models import SignalResult
from ._xau_signal.signal_engine import SignalEngine

__all__ = ["SignalEngine", "SignalResult"]
