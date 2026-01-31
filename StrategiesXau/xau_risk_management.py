# StrategiesXau/xau_risk_management.py  (REFactored / API-compatible wrapper)
from __future__ import annotations

from ._xau_risk.execution_quality import ExecSample, ExecutionQualityMonitor
from ._xau_risk.logging_ import LOG_DIR, _FILE_LOCK, log_risk
from ._xau_risk.models import PlanOrderResult, RiskDecision, SignalSurvivalState
from ._xau_risk.risk_manager import RiskManager
from ._xau_risk.utils import percentile_rank

__all__ = [
    "RiskManager",
    "RiskDecision",
    "ExecutionQualityMonitor",
    "ExecSample",
    "percentile_rank",
    "SignalSurvivalState",
]
