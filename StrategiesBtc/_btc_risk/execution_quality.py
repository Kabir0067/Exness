# _btc_risk/execution_quality.py — Execution quality monitor (breaker input) for BTC
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np


@dataclass(slots=True)
class ExecSample:
    """Single execution sample; slippage in points (ABS for robust breakers)."""
    ts: float
    side: str
    expected_price: float
    filled_price: float
    point: float
    spread_points: float
    latency_total_ms: float

    @property
    def slippage_points(self) -> float:
        """ABS slippage in points (always >= 0) for robust breakers."""
        try:
            pt = float(self.point)
            if pt <= 0:
                return 0.0
            return float(abs(float(self.filled_price) - float(self.expected_price)) / pt)
        except Exception:
            return 0.0


class ExecutionQualityMonitor:
    """Rolling monitor for execution + quote anomalies (BTC thresholds)."""

    def __init__(self, *, window: int = 300) -> None:
        self.window = int(window)
        self.samples: Deque[ExecSample] = deque(maxlen=self.window)
        self.spreads: Deque[float] = deque(maxlen=self.window)
        self.mid_moves: Deque[float] = deque(maxlen=self.window)
        self._last_mid: Optional[float] = None
        self.ewma_slip: Optional[float] = None
        self.ewma_lat: Optional[float] = None
        self.ewma_spread: Optional[float] = None

    @staticmethod
    def _ewma(prev: Optional[float], x: float, alpha: float) -> float:
        return x if prev is None else (alpha * x + (1.0 - alpha) * prev)

    def on_quote(self, bid: float, ask: float, point: float) -> None:
        try:
            pt = float(point)
            bid = float(bid)
            ask = float(ask)
            if pt <= 0 or bid <= 0 or ask <= 0 or ask < bid:
                return
            mid = (bid + ask) / 2.0
            spread_points = (ask - bid) / pt

            self.spreads.append(float(spread_points))
            self.ewma_spread = self._ewma(self.ewma_spread, float(spread_points), alpha=0.08)

            if self._last_mid is not None:
                move_points = abs(mid - self._last_mid) / pt
                self.mid_moves.append(float(move_points))

            self._last_mid = mid
        except Exception:
            return

    def on_execution(self, sample: ExecSample) -> None:
        self.samples.append(sample)
        self.ewma_slip = self._ewma(self.ewma_slip, float(sample.slippage_points), alpha=0.10)
        self.ewma_lat = self._ewma(self.ewma_lat, float(sample.latency_total_ms), alpha=0.10)

    def snapshot(self) -> Dict[str, float]:
        slip = [s.slippage_points for s in self.samples]
        lat = [float(s.latency_total_ms) for s in self.samples]
        spr = list(self.spreads)

        def p95(x: List[float]) -> float:
            if not x:
                return 0.0
            return float(np.percentile(np.array(x, dtype=np.float64), 95))

        return {
            "n": float(len(self.samples)),
            "ewma_slippage_points": float(self.ewma_slip or 0.0),
            "ewma_latency_ms": float(self.ewma_lat or 0.0),
            "ewma_spread_points": float(self.ewma_spread or 0.0),
            "p95_slippage_points": p95(slip),
            "p95_latency_ms": p95(lat),
            "p95_spread_points": p95(spr) if spr else 0.0,
            "max_spread_points": float(max(spr)) if spr else 0.0,
            "p95_mid_move_points": float(np.percentile(np.array(self.mid_moves, dtype=np.float64), 95))
            if self.mid_moves else 0.0,
        }

    def anomaly_reasons(self, *, cfg: Any) -> List[str]:
        """Thresholds from cfg; BTC defaults: 550ms latency, 20 pts slippage, 100 spread, 15 ewma slip."""
        s = self.snapshot()
        max_p95_lat = float(getattr(cfg, "exec_max_p95_latency_ms", 550.0))
        max_p95_slip = float(getattr(cfg, "exec_max_p95_slippage_points", 20.0))
        max_spread = float(getattr(cfg, "exec_max_spread_points", 100.0))
        max_ewma_slip = float(getattr(cfg, "exec_max_ewma_slippage_points", 15.0))

        reasons: List[str] = []
        if s["p95_latency_ms"] > max_p95_lat:
            reasons.append("exec:latency_p95")
        if s["p95_slippage_points"] > max_p95_slip:
            reasons.append("exec:slippage_p95")
        if s["ewma_slippage_points"] > max_ewma_slip:
            reasons.append("exec:slippage_ewma")
        if s["max_spread_points"] > max_spread:
            reasons.append("mkt:spread_shock")
        return reasons
