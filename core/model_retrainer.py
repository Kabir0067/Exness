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
        """Check if any asset needs retraining based on per-asset age thresholds."""
        for asset in assets:
            key = _normalize_asset(asset)
            threshold = float(MAX_MODEL_AGE_HOURS.get(key, 48))
            # Check per-asset state file first
            asset_state = self.model_path.parent / f"model_state_{key}.pkl"
            if asset_state.exists():
                try:
                    age_sec = max(0.0, time.time() - float(asset_state.stat().st_mtime))
                    if (age_sec / 3600.0) > threshold:
                        return True
                except Exception:
                    return True
            else:
                # Missing per-asset state file = needs training
                return True
        return False
