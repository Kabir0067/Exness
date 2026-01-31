from __future__ import annotations

"""Portfolio engine (refactored into modules).

This file is kept for backward compatibility.
Public imports remain valid:
- from portfolio_engine import engine, MultiAssetTradingEngine
"""

from .Motor.engine import MultiAssetTradingEngine, engine  # noqa: F401
from .Motor.models import AssetCandidate, ExecutionResult, OrderIntent, PortfolioStatus  # noqa: F401
from .Motor.scheduler import UTCScheduler  # noqa: F401

__all__ = [
    "UTCScheduler",
    "AssetCandidate",
    "PortfolioStatus",
    "OrderIntent",
    "ExecutionResult",
    "MultiAssetTradingEngine",
    "engine",
]
