from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CostBreakdown:
    spread: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0
    funding: float = 0.0

    @property
    def total(self) -> float:
        return float(self.spread + self.slippage + self.commission + self.funding)


class TransactionCostModel:
    """
    Deterministic transaction cost model with optional random slippage jitter.
    """

    def __init__(
        self,
        *,
        commission_bps: float = 0.0,
        funding_rate_annual: float = 0.0,
    ) -> None:
        self.commission_bps = float(max(0.0, commission_bps))
        self.funding_rate_annual = float(max(0.0, funding_rate_annual))

    def estimate(
        self,
        *,
        notional: float,
        spread_bps: float,
        slippage_bps: float,
        volatility: float,
        hold_minutes: float,
        rng: Optional[np.random.Generator] = None,
    ) -> CostBreakdown:
        notional_f = float(max(0.0, notional))
        spread_cost = notional_f * (float(max(0.0, spread_bps)) / 10_000.0)

        slip_mult = 1.0 + min(max(float(volatility), 0.0) / 0.001, 3.0)
        slippage_cost = notional_f * (float(max(0.0, slippage_bps)) / 10_000.0) * slip_mult
        if rng is not None:
            slippage_cost *= float(rng.uniform(0.8, 1.2))

        # Round-turn commission (entry + exit).
        commission = notional_f * (self.commission_bps / 10_000.0) * 2.0

        hold_minutes_f = float(max(0.0, hold_minutes))
        minutes_per_year = 365.25 * 24.0 * 60.0
        funding = notional_f * self.funding_rate_annual * (hold_minutes_f / minutes_per_year)

        return CostBreakdown(
            spread=float(spread_cost),
            slippage=float(slippage_cost),
            commission=float(commission),
            funding=float(funding),
        )

