# models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .utils import _side_norm


@dataclass(slots=True)
class RiskDecision:
    allowed: bool
    reasons: List[str] = field(default_factory=list)


@dataclass(slots=True)
class PlanOrderResult:
    ok: bool = False
    reason: str = ""

    # filled when ok=True
    side: str = ""
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    lot: Optional[float] = None
    n_orders: int = 0

    # diagnostics / extra info
    meta: Dict[str, Any] = field(default_factory=dict)


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
