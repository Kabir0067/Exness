from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SignalResult:
    symbol: str
    signal: str  # "Buy" | "Sell" | "Neutral"
    confidence: int  # 0..100
    regime: Optional[str]
    reasons: List[str] = field(default_factory=list)
    spread_bps: Optional[float] = None
    latency_ms: float = 0.0
    timestamp: str = ""

    # plan (NOT execution)
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    lot: Optional[float] = None

    # routing / control
    signal_id: str = ""
    trade_blocked: bool = False
