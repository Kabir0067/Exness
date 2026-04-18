"""
strategies/xau.py — XAU strategy stack assembly.

Wires XAUEngineConfig, RiskManager, FeatureEngine, and SignalEngine
into a frozen dataclass for downstream consumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from core.config import XAUEngineConfig, XAUSymbolParams
from core.feature_engine import FeatureEngine
from core.risk_manager import RiskManager
from core.signal_engine import SignalEngine

log = logging.getLogger("strategies.xau")


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
class XAUStack:
    """Immutable container for a fully-wired XAU trading stack."""

    cfg: XAUEngineConfig
    sp: XAUSymbolParams
    risk_manager: RiskManager
    feature_engine: FeatureEngine
    signal_engine: SignalEngine


# =============================================================================
# Factory
# =============================================================================
def create_xau_stack(
    *,
    market_feed: MarketFeedLike,
    login: int = 0,
    password: str = "",
    server: str = "",
    telegram_token: str = "",
    admin_id: int = 0,
) -> XAUStack:
    """Assemble and return a complete XAU trading stack."""
    sp = XAUSymbolParams()
    cfg = XAUEngineConfig(
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

    log.info("XAU stack created | symbol=%s magic=%d", sp.symbol, cfg.magic)
    return XAUStack(
        cfg=cfg, sp=sp, risk_manager=rm, feature_engine=fe, signal_engine=se
    )
