# core/models.py — Shared dataclasses for the trading system.
# Extracted from duplicated BTC/XAU risk managers.
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class KillSwitchState:
    """
    Auto-kill switch state machine.
    Tracks expectancy and win-rate to halt trading when performance degrades.
    """
    status: str = "ACTIVE"
    cooling_until_ts: float = 0.0
    last_expectancy: float = 0.0
    last_winrate: float = 0.0
    last_trades: int = 0

    def update(
        self,
        profits: List[float],
        now: float,
        min_trades: int,
        kill_expectancy: float,
        kill_winrate: float,
        cooling_expectancy: float,
        cooling_sec: float,
    ) -> None:
        """
        Evaluate kill switch conditions:
          - KILLED: expectancy < kill_expectancy AND winrate < kill_winrate
          - COOLING: expectancy < cooling_expectancy (temporary pause)
          - ACTIVE: otherwise
        """
        n = len(profits)
        self.last_trades = n
        if n < min_trades:
            self.status = "ACTIVE"
            return

        wins = sum(1 for p in profits if p > 0)
        wr = wins / n if n > 0 else 0.0
        exp = sum(profits) / n if n > 0 else 0.0

        self.last_expectancy = exp
        self.last_winrate = wr

        if now < self.cooling_until_ts:
            self.status = "COOLING"
            return

        if exp < kill_expectancy and wr < kill_winrate:
            self.status = "KILLED"
            return

        if exp < cooling_expectancy:
            self.cooling_until_ts = now + cooling_sec
            self.status = "COOLING"
            return

        self.status = "ACTIVE"


@dataclass
class AccountCache:
    """Cached account info to prevent 0.0 glitches from MT5."""
    ts: float = 0.0
    balance: float = 0.0
    equity: float = 0.0
    margin_free: float = 0.0


@dataclass
class SignalThrottle:
    """Per-asset signal rate limiter."""
    daily_count: int = 0
    hour_window_start_ts: float = field(default_factory=time.time)
    hour_window_count: int = 0

    def register(self, max_per_hour: int = 20) -> bool:
        """Returns True if signal is allowed, False if throttled."""
        now = time.time()
        # Reset hour window
        if now - self.hour_window_start_ts > 3600:
            self.hour_window_start_ts = now
            self.hour_window_count = 0
        if self.hour_window_count >= max_per_hour:
            return False
        self.hour_window_count += 1
        self.daily_count += 1
        return True


@dataclass
class SignalResult:
    """Unified signal output from SignalEngine."""
    signal: str = "Neutral"
    symbol: str = ""
    confidence: int = 0
    spread_pct: float = 0.0
    regime: str = "normal"
    signal_id: str = ""
    bar_key: str = "no_bar"
    reasons: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    trade_blocked: bool = False
    # Execution plan fields (filled when execute=True)
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lot: float = 0.0
    timeframe: str = ""


@dataclass
class AssetCandidate:
    """A validated trading candidate ready for execution decision."""
    asset: str = ""
    signal: str = "Neutral"
    confidence: int = 0
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lot: float = 0.0
    signal_id: str = ""
    bar_key: str = "no_bar"
    reasons: List[str] = field(default_factory=list)
    spread_pct: float = 0.0
    regime: str = ""
    timeframe: str = ""
    trade_blocked: bool = False


@dataclass
class OrderIntent:
    """Intent to place an order, queued for ExecutionWorker."""
    symbol: str = ""
    signal: str = ""
    lot: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    price: float = 0.0
    order_id: str = ""
    signal_id: str = ""
    enqueue_time: float = 0.0
    cfg: object = None
    risk_manager: object = None


@dataclass
class ExecutionResult:
    """Result of an order execution attempt."""
    order_id: str = ""
    signal_id: str = ""
    ok: bool = False
    reason: str = ""
    sent_ts: float = 0.0
    fill_ts: float = 0.0
    req_price: float = 0.0
    exec_price: float = 0.0
    volume: float = 0.0
    slippage: float = 0.0
    retcode: int = 0


@dataclass
class PortfolioStatus:
    """Snapshot of portfolio state for health monitoring."""
    timestamp: float = 0.0
    total_equity: float = 0.0
    total_margin_used: float = 0.0
    open_positions_btc: int = 0
    open_positions_xau: int = 0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    phase_btc: str = "A"
    phase_xau: str = "A"
    mt5_connected: bool = False
    heartbeat_status: str = "ALIVE"
