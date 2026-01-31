from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class AssetCandidate:
    asset: str  # "XAU" / "BTC"
    symbol: str
    signal: str  # "Buy" / "Sell" / "Neutral"
    confidence: float  # 0..1
    lot: float
    sl: float
    tp: float
    latency_ms: float
    blocked: bool
    reasons: Tuple[str, ...]
    signal_id: str
    raw_result: Any


@dataclass(frozen=True)
class PortfolioStatus:
    connected: bool
    trading: bool
    manual_stop: bool
    active_asset: str
    balance: float
    equity: float
    dd_pct: float
    today_pnl: float
    open_trades_total: int
    open_trades_xau: int
    open_trades_btc: int
    last_signal_xau: str
    last_signal_btc: str
    last_selected_asset: str
    exec_queue_size: int
    last_reconcile_ts: float


@dataclass(frozen=True)
class OrderIntent:
    asset: str
    symbol: str
    signal: str
    confidence: float
    lot: float
    sl: float
    tp: float
    price: float
    enqueue_time: float
    order_id: str
    signal_id: str
    idempotency_key: str
    risk_manager: Any
    cfg: Any


@dataclass(frozen=True)
class ExecutionResult:
    order_id: str
    signal_id: str
    ok: bool
    reason: str
    sent_ts: float
    fill_ts: float
    req_price: float
    exec_price: float
    volume: float
    slippage: float
    retcode: int = 0
