from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from core.ml_router import MLSignal

from .models import AssetCandidate


class EngineState(Enum):
    BOOT = "BOOT"
    DATA_SYNC = "DATA_SYNC"
    ML_INFERENCE = "ML_INFERENCE"
    RISK_CALC = "RISK_CALC"
    EXECUTION_QUEUE = "EXECUTION_QUEUE"
    VERIFICATION = "VERIFICATION"
    HALT = "HALT"


@dataclass
class EngineCycleContext:
    payloads: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = field(default_factory=dict)
    ml_signals: Dict[str, MLSignal] = field(default_factory=dict)
    candidates: list[AssetCandidate] = field(default_factory=list)
    halt_reason: str = ""


@dataclass(frozen=True)
class StepDecision:
    next_state: EngineState
    ctx: EngineCycleContext
    skip_sleep: bool = False
