from __future__ import annotations

import sys

from .engine import MultiAssetTradingEngine, engine  # noqa: F401
from .models import AssetCandidate, PortfolioStatus, OrderIntent, ExecutionResult  # noqa: F401
from .models import EngineCycleContext, EngineState, StepDecision  # noqa: F401
from .pipeline import UTCScheduler  # noqa: F401
from .inference import InferenceEngine  # noqa: F401
from .execution import ExecutionManager, ExecutionWorker, OrderSyncManager  # noqa: F401
from .fsm import DeterministicFSMRunner, EngineFSMStageServices, IFSMEnginePorts  # noqa: F401

from . import execution as _execution
from . import fsm as _fsm
from . import inference as _inference
from . import models as _models
from . import pipeline as _pipeline

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

