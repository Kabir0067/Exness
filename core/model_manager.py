import os
import pickle
import json
import logging
from datetime import datetime
from glob import glob
from dataclasses import dataclass
from typing import Any, Optional, List

log = logging.getLogger("core.model_manager")

from log_config import get_artifact_dir

@dataclass
class ModelMetadata:
    version: str
    timestamp: str
    sharpe: float
    win_rate: float
    author: str = "quantum_trainer"
    status: str = "PENDING"  # PENDING, VERIFIED, REJECTED
    backtest_sharpe: float = 0.0
    backtest_win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    real_backtest: bool = False
    training_features: Optional[List[str]] = None
    source: str = "model_train"

class ModelManager:
    """
    The 'Holy Trinity' Gatekeeper.
    Manages Model Training -> Backtest Verification -> Live Deployment.
    """
    def __init__(self, models_dir: str = ""):
        if not models_dir:
            models_dir = str(get_artifact_dir("models"))
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def save_model(self, model: Any, metadata: ModelMetadata) -> str:
        """Save a new model with metadata."""
        version = metadata.version
        base_path = os.path.join(self.models_dir, f"v{version}")
        
        # Save model pickle
        model_path = f"{base_path}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        # Save metadata json
        meta_path = f"{base_path}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata.__dict__, f, indent=4)
            
        log.info(f"Model saved: {version} | Sharpe: {metadata.sharpe}")
        return base_path

    def load_latest_verified_model(self) -> Optional[Any]:
        """Load the most recent model that passed backtest verification."""
        # Find all .json metadata files
        meta_files = glob(os.path.join(self.models_dir, "v*.json"))
        best_model_path = None
        best_ts = ""

        for meta_file in meta_files:
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                
                # STRICT GATE: Only load VERIFIED models
                if meta.get("status") != "VERIFIED":
                    continue
                    
                ts = meta.get("timestamp", "")
                if ts > best_ts:
                    best_ts = ts
                    best_model_path = meta_file.replace(".json", ".pkl")
            except Exception:
                continue
                
        if best_model_path and os.path.exists(best_model_path):
            try:
                with open(best_model_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                log.error(f"Failed to load model {best_model_path}: {e}")
                return None
                
        log.warning("No VERIFIED models found in registry.")
        return None

    def verify_model(self, version: str, sharpe: float, win_rate: float) -> bool:
        """
        Called by Backtest Engine.
        Updates model status to VERIFIED or REJECTED based on performance.
        """
        meta_path = os.path.join(self.models_dir, f"v{version}.json")
        if not os.path.exists(meta_path):
            log.error(f"Model metadata not found: {version}")
            return False
            
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                
            # QUANTUM GATING LOGIC
            # STRICT REQUIREMENT: Sharpe >= 1.5
            is_good = (sharpe >= 1.5) and (win_rate >= 0.55)
            meta["status"] = "VERIFIED" if is_good else "REJECTED"
            meta["backtest_sharpe"] = sharpe
            meta["backtest_win_rate"] = win_rate
            meta["real_backtest"] = True
            meta["verified_at_utc"] = datetime.utcnow().isoformat()
            
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
                
            if is_good:
                log.info(f"Model {version} VERIFIED (Sharpe={sharpe:.2f} >= 1.5)")
            else:
                log.warning(f"Model {version} REJECTED (Sharpe={sharpe:.2f} < 1.5)")
             
            return is_good
        except Exception as e:
            log.error(f"Verification update failed: {e}")
            return False

# Singleton
model_manager = ModelManager()
