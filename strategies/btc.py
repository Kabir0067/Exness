"""
strategies/btc.py — BTC strategy stack assembly.

Wires BTCEngineConfig, RiskManager, FeatureEngine, and SignalEngine
into a frozen dataclass for downstream consumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from core.config import BTCEngineConfig, BTCSymbolParams
from core.feature_engine import FeatureEngine
from core.risk_manager import RiskManager
from core.signal_engine import SignalEngine

log = logging.getLogger("strategies.btc")


# =============================================================================
# Protocols
# =============================================================================
class MarketFeedLike(Protocol):
    """Minimal contract for a market data provider."""

    def get(self, *args: Any, **kwargs: Any) -> Any: ...


# =============================================================================
# Data Structures
# =============================================================================
@dataclass(frozen=True)
class BTCStack:
    """Immutable container for a fully-wired BTC trading stack."""

    cfg: BTCEngineConfig
    sp: BTCSymbolParams
    risk_manager: RiskManager
    feature_engine: FeatureEngine
    signal_engine: SignalEngine


# =============================================================================
# Factory
# =============================================================================
def create_btc_stack(
    *,
    market_feed: MarketFeedLike,
    login: int = 0,
    password: str = "",
    server: str = "",
    telegram_token: str = "",
    admin_id: int = 0,
) -> BTCStack:
    """Assemble and return a complete BTC trading stack."""
    sp = BTCSymbolParams()
    cfg = BTCEngineConfig(
        login=int(login or 0),
        password=str(password or ""),
        server=str(server or ""),
        telegram_token=str(telegram_token or ""),
        admin_id=int(admin_id or 0),
        symbol_params=sp,
    )

    rm = RiskManager(cfg, sp)
    fe = FeatureEngine(cfg)
    se = SignalEngine(cfg, sp, market_feed, fe, rm)

    log.info("BTC stack created | symbol=%s magic=%d", sp.symbol, cfg.magic)
    return BTCStack(
        cfg=cfg, sp=sp, risk_manager=rm, feature_engine=fe, signal_engine=se
    )
