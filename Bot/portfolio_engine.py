"""
Bot/portfolio_engine.py — Backward-compatible re-export facade.

This file is maintained for backward compatibility.
Public imports remain stable:
    from portfolio_engine import engine, MultiAssetTradingEngine
"""

from __future__ import annotations

from .Motor.engine import MultiAssetTradingEngine, engine  # noqa: F401
from .Motor.models import (  # noqa: F401
    AssetCandidate,
    ExecutionResult,
    OrderIntent,
    PortfolioStatus,
)
from .Motor.pipeline import UTCScheduler  # noqa: F401

__all__ = [
    "AssetCandidate",
    "ExecutionResult",
    "MultiAssetTradingEngine",
    "OrderIntent",
    "PortfolioStatus",
    "UTCScheduler",
    "engine",
]
