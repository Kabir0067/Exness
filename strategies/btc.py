# strategies/btc.py — BTC strategy adapter (thin layer over core/).
# Creates BTC-specific config and instantiates unified core components.
from __future__ import annotations

import logging
from typing import Any

from core.config import BTCEngineConfig, BTCSymbolParams
from core.feature_engine import FeatureEngine
from core.risk_engine import RiskManager
from core.signal_engine import SignalEngine

log = logging.getLogger("strategies.btc")


def create_btc_stack(
    *,
    market_feed: Any,
    login: int = 0,
    password: str = "",
    server: str = "",
    telegram_token: str = "",
    admin_id: int = 0,
) -> dict:
    """
    Create the full BTC trading component stack.

    Returns:
        {
            "cfg": BTCEngineConfig,
            "sp": BTCSymbolParams,
            "risk_manager": RiskManager,
            "feature_engine": FeatureEngine,
            "signal_engine": SignalEngine,
        }
    """
    sp = BTCSymbolParams()
    cfg = BTCEngineConfig(
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

    log.info("BTC stack created | symbol=%s magic=%d", sp.symbol, cfg.magic)

    return {
        "cfg": cfg,
        "sp": sp,
        "risk_manager": rm,
        "feature_engine": fe,
        "signal_engine": se,
    }
