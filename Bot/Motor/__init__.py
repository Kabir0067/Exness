"""
Bot/Motor/__init__.py - Module initialization for portfolio multi-asset engine.

Ин модул интерфейсҳои асосии моторро берун меорад.
"""

from __future__ import annotations

import sys

from . import execution as _execution
from . import fsm as _fsm
from . import inference as _inference
from . import models as _models
from . import pipeline as _pipeline

# =============================================================================
# Imports
# =============================================================================
from .engine import MultiAssetTradingEngine, engine
from .execution import ExecutionManager, ExecutionWorker, OrderSyncManager
from .fsm import DeterministicFSMRunner, EngineFSMStageServices, IFSMEnginePorts
from .inference import InferenceEngine
from .models import (
    AssetCandidate,
    EngineCycleContext,
    EngineState,
    ExecutionResult,
    OrderIntent,
    PortfolioStatus,
    StepDecision,
)
from .pipeline import UTCScheduler

# =============================================================================
# Global Setup
# =============================================================================
_ALIASES = {
    "logging_setup": _models,
    "utils": _models,
    "fsm_types": _models,
    "scheduler": _pipeline,
    "inference_engine": _inference,
    "execution_manager": _execution,
    "order_sync_manager": _execution,
    "fsm_ports": _fsm,
    "fsm_runtime": _fsm,
    "fsm_services": _fsm,
}

for _name, _module in _ALIASES.items():
    sys.modules.setdefault(f"{__name__}.{_name}", _module)

# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "AssetCandidate",
    "DeterministicFSMRunner",
    "EngineCycleContext",
    "EngineFSMStageServices",
    "EngineState",
    "ExecutionManager",
    "ExecutionResult",
    "ExecutionWorker",
    "IFSMEnginePorts",
    "InferenceEngine",
    "MultiAssetTradingEngine",
    "OrderIntent",
    "OrderSyncManager",
    "PortfolioStatus",
    "StepDecision",
    "UTCScheduler",
    "engine",
]
