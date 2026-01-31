# _btc_risk/models.py — from StrategiesBtc btc_risk_management (verified)
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from .utils import _side_norm


@dataclass(slots=True)
class RiskDecision:
    allowed: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class AccountCache:
    ts: float = 0.0
    balance: float = 0.0
    equity: float = 0.0
    margin_free: float = 0.0


@dataclass
class SignalThrottle:
    daily_count: int = 0
    hour_window_start_ts: float = field(default_factory=lambda: time.time())
    hour_window_count: int = 0


@dataclass
class MT5Throttle:
    ts: float = 0.0
    ok: bool = False


@dataclass
class SymbolMeta:
    ts: float = 0.0
    digits: int = 2
    point: float = 0.01
    vol_min: float = 0.01
    vol_max: float = 100.0
    vol_step: float = 0.01
    stops_level_points: int = 0
    freeze_level_points: int = 0


@dataclass(slots=True)
class SignalSurvivalState:
    order_id: str
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    entry_time: float
    confidence: float
    mfe: float = 0.0
    mae: float = 0.0
    bars_held: int = 0
    hit_sl: bool = False
    hit_tp: bool = False
    last_price: float = 0.0
    last_update_ts: float = 0.0

    def update(self, *, current_price: float) -> None:
        d = _side_norm(self.direction)
        entry = float(self.entry_price)
        px = float(current_price)
        self.last_price = px
        if d == "Buy":
            self.mfe = float(max(self.mfe, px - entry))
            self.mae = float(min(self.mae, px - entry))
        else:
            self.mfe = float(max(self.mfe, entry - px))
            self.mae = float(min(self.mae, entry - px))
        self.bars_held = int(self.bars_held) + 1
