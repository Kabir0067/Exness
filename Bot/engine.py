"""Bot/engine.py

Unified engine export for portfolio trading (XAU + BTC).

The production runner uses the portfolio engine that operates both XAU and BTC,
but executes trades for only one asset at a time (single-asset lock).

Public API:
- engine.start() - Start portfolio engine
- engine.stop() - Stop portfolio engine
- engine.status() - Get PortfolioStatus
- engine.request_manual_stop() - Manual stop via Telegram
- engine.clear_manual_stop() - Clear manual stop
- engine.manual_stop_active() - Check manual stop status

Exported classes:
- MultiAssetTradingEngine - Main portfolio engine class
- engine - Global singleton instance
"""

from __future__ import annotations

from .portfolio_engine import MultiAssetTradingEngine, engine

# Export all public API
__all__ = [
    "MultiAssetTradingEngine",
    "engine",
]
