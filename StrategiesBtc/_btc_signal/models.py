# _btc_signal/models.py — from StrategiesBtc btc_signal_engine (verified)
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True, slots=True)
class SignalResult:
    symbol: str
    signal: str  # "Buy" | "Sell" | "Neutral"
    confidence: int  # 0..100
    regime: Optional[str]
    reasons: List[str]
    spread_bps: Optional[float]
    latency_ms: float
    timestamp: str
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    lot: Optional[float] = None
    signal_id: str = ""
    trade_blocked: bool = False
