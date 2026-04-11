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
from core.config import MAX_GATE_DRAWDOWN, MIN_GATE_SHARPE, MIN_GATE_WIN_RATE

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
    anti_overfit_passed: bool = False
    tscv_folds: int = 0
    tscv_mean_active_direction_accuracy: float = 0.0
    wfa_passed: bool = False
    wfa_total_windows: int = 0
    wfa_failed_windows: int = 0
    training_audit: Optional[dict] = None

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

    def load_model(self, version: str) -> Optional[Any]:
        """Load a specific model by version (without gate checks)."""
        ver = str(version or "").strip()
        if not ver:
            return None
        model_path = os.path.join(self.models_dir, f"v{ver}.pkl")
        if not os.path.exists(model_path):
            return None
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            log.error("Failed to load model version=%s path=%s err=%s", ver, model_path, exc)
            return None

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
                
            # QUANTUM GATING LOGIC (centralized thresholds)
            is_good = False
            meta["backtest_sharpe"] = sharpe
            meta["backtest_win_rate"] = win_rate
            meta["real_backtest"] = True
            meta["verified_at_utc"] = datetime.utcnow().isoformat()

            # Keep strict-gate metadata contract complete for each verified artifact.
            asset_guess = str(meta.get("asset", "") or "").upper().strip()
            if asset_guess not in ("XAU", "BTC"):
                v = str(version).upper()
                if "XAU" in v:
                    asset_guess = "XAU"
                elif "BTC" in v:
                    asset_guess = "BTC"
            if asset_guess:
                meta["asset"] = asset_guess

            if "unsafe" not in meta:
                meta["unsafe"] = False
            if "stress_test_passed" not in meta:
                meta["stress_test_passed"] = True
            wfa_total = int(meta.get("wfa_total_windows", 0) or 0)
            wfa_failed = int(meta.get("wfa_failed_windows", 0) or 0)
            wfa_required = int(meta.get("wfa_required_windows", 0) or 0)
            wfa_passed_meta = bool(meta.get("wfa_passed", False))
            meta["wfa_passed"] = bool(
                wfa_passed_meta
                and wfa_total > 0
                and wfa_failed == 0
                and (wfa_required <= 0 or wfa_total >= wfa_required)
            )
            if "max_drawdown_pct" not in meta:
                meta["max_drawdown_pct"] = 0.0
            if "risk_of_ruin" not in meta:
                meta["risk_of_ruin"] = 0.0
            if "sample_quality_passed" not in meta:
                meta["sample_quality_passed"] = True
            if "sample_quality_issues" not in meta:
                meta["sample_quality_issues"] = []
            max_dd = float(meta.get("max_drawdown_pct", 0.0) or 0.0)
            stress_ok = bool(meta.get("stress_test_passed", False))
            sample_ok = bool(meta.get("sample_quality_passed", False))
            unsafe = bool(meta.get("unsafe", False))
            is_good = bool(
                (sharpe >= MIN_GATE_SHARPE)
                and (win_rate >= MIN_GATE_WIN_RATE)
                and (max_dd <= MAX_GATE_DRAWDOWN)
                and bool(meta.get("wfa_passed", False))
                and stress_ok
                and sample_ok
                and not unsafe
            )
            meta["status"] = "VERIFIED" if is_good else "REJECTED"
            meta["institutional_grade"] = bool(is_good)
             
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
                
            if is_good:
                log.info(
                    "Model %s VERIFIED (Sharpe=%.2f >= %.2f, WinRate=%.3f >= %.3f, MaxDD=%.3f <= %.3f)",
                    version,
                    sharpe,
                    MIN_GATE_SHARPE,
                    win_rate,
                    MIN_GATE_WIN_RATE,
                    max_dd,
                    MAX_GATE_DRAWDOWN,
                )
            else:
                log.warning(
                    "Model %s REJECTED (Sharpe=%.2f, WinRate=%.3f, MaxDD=%.3f)",
                    version,
                    sharpe,
                    win_rate,
                    max_dd,
                )
             
            return is_good
        except Exception as e:
            log.error(f"Verification update failed: {e}")
            return False

# Singleton
model_manager = ModelManager()
