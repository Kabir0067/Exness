"""
core/portfolio_risk.py — Portfolio-level risk management and transaction costs.

Provides cross-asset correlation checks, total exposure limits, portfolio
drawdown guards, and a deterministic transaction cost model. This module
acts as the final gatekeeper before any order reaches the execution layer.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import risk_manager as _risk_manager

log = logging.getLogger("core.risk_engine")

for _risk_export in _risk_manager.__all__:
    globals()[_risk_export] = getattr(_risk_manager, _risk_export)


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class AssetExposure:
    """Live exposure snapshot for a single asset."""

    symbol: str = ""
    side: str = ""  # "Buy" or "Sell"
    volume: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0
    position_count: int = 0
    open_risk: float = 0.0


@dataclass(frozen=True)
class CostBreakdown:
    """Immutable breakdown of estimated transaction costs."""

    spread: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0
    funding: float = 0.0

    @property
    def total(self) -> float:
        """Sum of all cost components."""
        return float(self.spread + self.slippage + self.commission + self.funding)


# =============================================================================
# Classes
# =============================================================================
class PortfolioRiskManager:
    """
    Institutional-grade portfolio-level risk management.

    Responsibilities:
      1. Max Daily Drawdown — hard stop ALL assets if combined PnL < -X%
      2. Correlation Guard — reduce sizing when BTC + XAU leverage same direction
      3. Total Exposure Cap — sum of all positions must not exceed Nx equity
      4. Position-level risk — enforce per-trade risk limits

    This is called by the engine BEFORE any order is dispatched to execution.
    """

    def __init__(
        self,
        *,
        max_daily_drawdown_pct: float = 0.05,
        hard_stop_buffer_pct: float = 0.0001,
        max_total_exposure_factor: float = 3.0,
        correlation_reduction: float = 0.50,
        max_risk_per_trade_pct: float = 0.015,
        max_concurrent_positions: int = 6,
        max_total_drawdown_pct: float = 0.12,
        max_asset_exposure_factor: float = 1.5,
        max_asset_risk_pct: float = 0.03,
    ) -> None:
        self.max_daily_dd_pct = max_daily_drawdown_pct
        self.hard_stop_buffer_pct = max(0.0, float(hard_stop_buffer_pct or 0.0))
        self.max_exposure_factor = max_total_exposure_factor
        self.correlation_reduction = correlation_reduction
        self.max_risk_per_trade = max_risk_per_trade_pct
        self.max_concurrent = max_concurrent_positions
        self.max_total_drawdown_pct = max(0.0, float(max_total_drawdown_pct or 0.0))
        self.max_asset_exposure_factor = max(
            0.0, float(max_asset_exposure_factor or 0.0)
        )
        self.max_asset_risk_pct = max(0.0, float(max_asset_risk_pct or 0.0))

        # State
        self._daily_start_equity: float = 0.0
        self._daily_date: str = ""
        self._hard_stopped: bool = False
        self._hard_stop_reason: str = ""
        self._exposures: Dict[str, AssetExposure] = {}
        self._trade_log: List[float] = []
        self._last_equity: float = 0.0
        self._last_daily_drawdown_pct: float = 0.0
        self._peak_equity: float = 0.0
        self._last_peak_drawdown_pct: float = 0.0
        self._lock = threading.RLock()

    # ---- Public API ---------------------------------------------------------

    def set_daily_start_equity(self, equity: float, date_str: str) -> None:
        """Called at start of each trading day to set baseline."""
        with self._lock:
            eq = float(equity or 0.0)
            if date_str != self._daily_date:
                self._daily_date = date_str
                self._daily_start_equity = eq
                self._hard_stopped = False
                self._hard_stop_reason = ""
                self._trade_log.clear()
                log.info(
                    "PORTFOLIO_DAILY_RESET | date=%s equity=%.2f max_dd=%.2f%%",
                    date_str,
                    eq,
                    self.max_daily_dd_pct * 100,
                )
            if eq > 0.0 and eq > self._peak_equity:
                self._peak_equity = eq

    def update_exposure(self, symbol: str, exposure: AssetExposure) -> None:
        """Update live exposure for an asset."""
        with self._lock:
            self._exposures[symbol] = exposure

    def record_trade_pnl(self, pnl: float) -> None:
        """Record a closed trade PnL for daily tracking."""
        with self._lock:
            self._trade_log.append(float(pnl))

    def is_portfolio_hard_stopped(self) -> Tuple[bool, str]:
        """Check if portfolio-level hard stop is active."""
        with self._lock:
            return self._hard_stopped, self._hard_stop_reason

    def evaluate_equity(self, current_equity: float) -> Tuple[bool, str]:
        """
        Continuously enforce the portfolio daily drawdown rule.

        This is intended for runtime loop checks, not only pre-order gating.
        """
        with self._lock:
            equity = float(current_equity or 0.0)
            self._last_equity = equity
            if equity > 0.0 and equity > self._peak_equity:
                self._peak_equity = equity
            dd_blocked, dd_reason = self._check_daily_drawdown(equity)
            peak_blocked, peak_reason = self._check_peak_drawdown(equity)
            reason = dd_reason if dd_blocked else peak_reason
            if (dd_blocked or peak_blocked) and not self._hard_stopped:
                self._hard_stopped = True
                self._hard_stop_reason = reason
                log.critical("PORTFOLIO_HARD_STOP | %s", reason)
            return self._hard_stopped, self._hard_stop_reason

    def check_before_order(
        self,
        *,
        asset: str,
        side: str,
        equity: float,
        proposed_lot: float,
        proposed_margin: float,
        proposed_risk_amount: float = 0.0,
        asset_exposure_factor: float = 0.0,
        asset_risk_factor: float = 0.0,
    ) -> Tuple[bool, float, str]:
        """
        Pre-order portfolio risk check.

        Returns:
            (allowed, adjusted_lot, reason)
            - allowed: True if trade passes all portfolio checks.
            - adjusted_lot: Possibly reduced lot (if correlation guard triggers).
            - reason: Empty string if allowed, otherwise the block reason.
        """
        with self._lock:
            # 1. Hard stop check
            if self._hard_stopped:
                return False, 0.0, f"PORTFOLIO_HARD_STOP: {self._hard_stop_reason}"

            # 2. Daily drawdown check
            dd_blocked, dd_reason = self.evaluate_equity(equity)
            if dd_blocked:
                return False, 0.0, dd_reason

            # 3. Max concurrent positions
            total_positions = sum(e.position_count for e in self._exposures.values())
            if total_positions >= self.max_concurrent:
                return (
                    False,
                    0.0,
                    f"MAX_CONCURRENT_POSITIONS: {total_positions}/{self.max_concurrent}",
                )

            # 4. Total exposure cap
            total_margin = (
                sum(e.margin_used for e in self._exposures.values()) + proposed_margin
            )
            if equity > 0 and total_margin / equity > self.max_exposure_factor:
                return (
                    False,
                    0.0,
                    (
                        f"EXPOSURE_CAP: total_margin={total_margin:.2f} "
                        f"exceeds {self.max_exposure_factor:.1f}x equity={equity:.2f}"
                    ),
                )

            asset_key = str(asset or "").upper().strip()
            asset_exposure = self._exposures.get(
                asset_key, AssetExposure(symbol=asset_key)
            )
            asset_margin = float(asset_exposure.margin_used) + float(
                proposed_margin or 0.0
            )
            asset_exposure_cap = float(
                asset_exposure_factor or self.max_asset_exposure_factor
            )
            if (
                asset_exposure_cap > 0.0
                and equity > 0.0
                and (asset_margin / equity) > asset_exposure_cap
            ):
                return (
                    False,
                    0.0,
                    (
                        f"ASSET_EXPOSURE_CAP: {asset_key} margin={asset_margin:.2f} "
                        f"exceeds {asset_exposure_cap:.2f}x equity={equity:.2f}"
                    ),
                )

            asset_risk_limit = float(asset_risk_factor or self.max_asset_risk_pct)
            proposed_risk = max(0.0, float(proposed_risk_amount or 0.0))
            combined_asset_risk = float(asset_exposure.open_risk) + proposed_risk
            if (
                asset_risk_limit > 0.0
                and equity > 0.0
                and (combined_asset_risk / equity) > asset_risk_limit
            ):
                return (
                    False,
                    0.0,
                    (
                        f"ASSET_RISK_CAP: {asset_key} risk={combined_asset_risk / equity:.2%} "
                        f"> {asset_risk_limit:.2%}"
                    ),
                )

            # 5. Correlation guard — reduce sizing when both assets face the same direction
            adjusted_lot = proposed_lot
            correlated = self._check_correlation(asset, side)
            if correlated:
                adjusted_lot = round(proposed_lot * self.correlation_reduction, 2)
                adjusted_lot = max(0.01, adjusted_lot)
                log.info(
                    "CORRELATION_GUARD | %s lot %.2f→%.2f (both assets same-side=%s)",
                    asset,
                    proposed_lot,
                    adjusted_lot,
                    side,
                )

            return True, adjusted_lot, ""

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary dict for health logging."""
        with self._lock:
            total_pnl = sum(e.unrealized_pnl for e in self._exposures.values())
            total_margin = sum(e.margin_used for e in self._exposures.values())
            total_positions = sum(e.position_count for e in self._exposures.values())
            daily_closed_pnl = sum(self._trade_log)

            return {
                "daily_start_equity": self._daily_start_equity,
                "peak_equity": self._peak_equity,
                "unrealized_pnl": total_pnl,
                "daily_closed_pnl": daily_closed_pnl,
                "last_equity": self._last_equity,
                "daily_drawdown_pct": self._last_daily_drawdown_pct,
                "peak_drawdown_pct": self._last_peak_drawdown_pct,
                "total_margin": total_margin,
                "total_positions": total_positions,
                "hard_stopped": self._hard_stopped,
                "hard_stop_reason": self._hard_stop_reason,
                "exposures": {
                    sym: {
                        "side": exp.side,
                        "volume": exp.volume,
                        "pnl": exp.unrealized_pnl,
                        "positions": exp.position_count,
                        "open_risk": exp.open_risk,
                    }
                    for sym, exp in self._exposures.items()
                },
            }

    # ---- Private Helpers ----------------------------------------------------

    def _check_daily_drawdown(self, current_equity: float) -> Tuple[bool, str]:
        """Check if daily drawdown exceeds limit."""
        if self._daily_start_equity <= 0:
            self._last_daily_drawdown_pct = 0.0
            return False, ""
        dd = (self._daily_start_equity - current_equity) / self._daily_start_equity
        self._last_daily_drawdown_pct = float(max(0.0, dd))
        trip_limit = max(
            0.0, float(self.max_daily_dd_pct) - float(self.hard_stop_buffer_pct)
        )
        if dd >= trip_limit:
            reason = (
                f"DAILY_DRAWDOWN_GUARD: {dd:.2%} >= guard {trip_limit:.2%} "
                f"(limit={self.max_daily_dd_pct:.2%}, buffer={self.hard_stop_buffer_pct:.2%}) "
                f"(start={self._daily_start_equity:.2f}, now={current_equity:.2f})"
            )
            return True, reason
        return False, ""

    def _check_peak_drawdown(self, current_equity: float) -> Tuple[bool, str]:
        """Check if total drawdown from equity peak exceeds the hard stop."""
        if self._peak_equity <= 0.0:
            self._last_peak_drawdown_pct = 0.0
            return False, ""
        dd = (self._peak_equity - current_equity) / self._peak_equity
        self._last_peak_drawdown_pct = float(max(0.0, dd))
        if self.max_total_drawdown_pct > 0.0 and dd >= self.max_total_drawdown_pct:
            reason = (
                f"MAX_DRAWDOWN_STOP: {dd:.2%} >= {self.max_total_drawdown_pct:.2%} "
                f"(peak={self._peak_equity:.2f}, now={current_equity:.2f})"
            )
            return True, reason
        return False, ""

    def _check_correlation(self, new_asset: str, new_side: str) -> bool:
        """
        Detect correlated exposure when adding a position.

        BTC and XAU both tend to be 'risk-off' assets — when both have
        positions in the same direction, total risk is correlated.
        """
        for sym, exp in self._exposures.items():
            if sym == new_asset:
                continue
            if exp.position_count > 0 and exp.side == new_side:
                return True
        return False


class TransactionCostModel:
    """Deterministic transaction cost model with optional random slippage jitter."""

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
        """Estimate round-trip transaction costs for a proposed trade."""
        notional_f = float(max(0.0, notional))
        spread_cost = notional_f * (float(max(0.0, spread_bps)) / 10_000.0)

        slip_mult = 1.0 + min(max(float(volatility), 0.0) / 0.001, 3.0)
        slippage_cost = (
            notional_f * (float(max(0.0, slippage_bps)) / 10_000.0) * slip_mult
        )
        if rng is not None:
            slippage_cost *= float(rng.uniform(0.8, 1.2))

        # Round-turn commission (entry + exit).
        commission = notional_f * (self.commission_bps / 10_000.0) * 2.0

        hold_minutes_f = float(max(0.0, hold_minutes))
        minutes_per_year = 365.25 * 24.0 * 60.0
        funding = (
            notional_f * self.funding_rate_annual * (hold_minutes_f / minutes_per_year)
        )

        return CostBreakdown(
            spread=float(spread_cost),
            slippage=float(slippage_cost),
            commission=float(commission),
            funding=float(funding),
        )


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "AssetExposure",
    "CostBreakdown",
    "PortfolioRiskManager",
    "TransactionCostModel",
]

__all__ += list(_risk_manager.__all__)
__all__ = tuple(__all__)
