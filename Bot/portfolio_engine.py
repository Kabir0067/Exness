"""
Backward-compatible facade for the portfolio engine exports.

Keeps public imports stable while re-exporting the motor entry points
and core runtime types.
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
