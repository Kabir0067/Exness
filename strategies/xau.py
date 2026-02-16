# strategies/xau.py — XAU strategy adapter (thin layer over core/).
# Creates XAU-specific config and instantiates unified core components.
from __future__ import annotations

import logging
from typing import Any

from core.config import XAUEngineConfig, XAUSymbolParams
from core.feature_engine import FeatureEngine
from core.risk_engine import RiskManager
from core.signal_engine import SignalEngine

log = logging.getLogger("strategies.xau")


def create_xau_stack(
    *,
    market_feed: Any,
    login: int = 0,
    password: str = "",
    server: str = "",
    telegram_token: str = "",
    admin_id: int = 0,
) -> dict:
    """
    Create the full XAU trading component stack.

    Returns:
        {
            "cfg": XAUEngineConfig,
            "sp": XAUSymbolParams,
            "risk_manager": RiskManager,
            "feature_engine": FeatureEngine,
            "signal_engine": SignalEngine,
        }
    """
    sp = XAUSymbolParams()
    cfg = XAUEngineConfig(
        login=login,
        password=password,
        server=server,
        telegram_token=telegram_token,
        admin_id=admin_id,
        symbol_params=sp,
    )

    rm = RiskManager(cfg, sp)
    fe = FeatureEngine(cfg)
    se = SignalEngine(cfg, sp, market_feed, fe, rm)

    log.info("XAU stack created | symbol=%s magic=%d", sp.symbol, cfg.magic)

    return {
        "cfg": cfg,
        "sp": sp,
        "risk_manager": rm,
        "feature_engine": fe,
        "signal_engine": se,
    }
