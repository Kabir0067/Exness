# StrategiesXau/xau_indicators.py  (REFactored / API-compatible wrapper)
from __future__ import annotations

from ._xau_indicators.backend import _Indicators
from ._xau_indicators.feature_engine import AnomalyResult, Classic_FeatureEngine
from ._xau_indicators.logging_ import LOG_DIR, logger
from ._xau_indicators.utils import safe_last

# Alias (same as original StrategiesXau)
ClassicFeatureEngine = Classic_FeatureEngine

__all__ = [
    "Classic_FeatureEngine",
    "ClassicFeatureEngine",
    "AnomalyResult",
    "safe_last",
]
