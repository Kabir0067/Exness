# Bot/engine.py  (PORTFOLIO WRAPPER, PRODUCTION)
"""Compatibility wrapper.

Your project originally had a single-asset engine in Bot/engine.py.
For the multi-asset system (XAU + BTC) we keep the same import path:

    from Bot.engine import engine

…and re-export the portfolio engine instance from Bot/portfolio_engine.py.
"""

from __future__ import annotations

from .portfolio_engine import (  # type: ignore
    engine,
    MultiAssetTradingEngine,
    PortfolioStatus,
    AssetCandidate,
)

__all__ = [
    "engine",
    "MultiAssetTradingEngine",
    "PortfolioStatus",
    "AssetCandidate",
]
