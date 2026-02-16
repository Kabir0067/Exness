from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterable, Optional

MAX_MODEL_AGE_HOURS = {
    "BTC": 24,  # Crypto changes fast
    "XAU": 48,  # Forex more stable
}


def _normalize_asset(asset: str) -> str:
    s = str(asset or "").upper().strip()
    if s.startswith("BTC"):
        return "BTC"
    if s.startswith("XAU"):
        return "XAU"
    return s


class ModelAgeChecker:
    def __init__(self, model_state_path: Path) -> None:
        self.model_path = Path(model_state_path)
        self._lock = threading.Lock()

    def get_model_age_hours(self) -> Optional[float]:
        """Returns model age in hours, or None if model doesn't exist."""
        with self._lock:
            if not self.model_path.exists():
                return None
            try:
                mtime = float(self.model_path.stat().st_mtime)
            except Exception:
                return None
            age_seconds = max(0.0, time.time() - mtime)
            return age_seconds / 3600.0

    def needs_retraining(self, assets: Iterable[str]) -> bool:
        """Check if any asset needs retraining based on min threshold."""
        age = self.get_model_age_hours()
        if age is None:
            return True

        thresholds: list[float] = []
        for asset in assets:
            key = _normalize_asset(asset)
            thresholds.append(float(MAX_MODEL_AGE_HOURS.get(key, 48)))

        min_threshold = min(thresholds) if thresholds else min(MAX_MODEL_AGE_HOURS.values())
        return age > float(min_threshold)
