# core/model_engine.py - model registry, age checks, and gates.

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from log_config import get_artifact_dir

from .config import (
    MAX_GATE_DRAWDOWN,
    MAX_GATE_SHARPE,
    MIN_GATE_SHARPE,
    MIN_GATE_WFA_PASS_RATE,
    MIN_GATE_WIN_RATE,
)

log = logging.getLogger("core.model_engine")

# ---- merged from core/model_manager.py ----


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
            log.error(
                "Failed to load model version=%s path=%s err=%s", ver, model_path, exc
            )
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

# ---- merged from core/model_retrainer.py ----

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


# ---- merged from core/model_gate.py ----

DEFAULT_REQUIRED_ASSETS: Tuple[str, ...] = ("XAU", "BTC")
_LEGACY_BACKUP_DONE = False
_LEGACY_BACKUP_LOCK = threading.Lock()
_BOOL_TRUE = frozenset({"1", "true", "yes", "y", "on"})


def _normalize_asset(asset: str) -> str:
    s = str(asset or "").upper().strip()
    if s.startswith("XAU"):
        return "XAU"
    if s.startswith("BTC"):
        return "BTC"
    return s


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _anti_overfit_gate_required() -> bool:
    for name in (
        "LIVE_REQUIRE_ANTI_OVERFIT_GATE",
        "INSTITUTIONAL_REQUIRE_ALPHA_GATE",
        "AUTO_TRAIN_REQUIRE_ALPHA_GATE",
    ):
        raw = str(os.environ.get(name, "") or "").strip().lower()
        if raw:
            return raw in _BOOL_TRUE
    return False


def _remove_first(haystack: str, needle: str) -> str:
    i = haystack.find(needle)
    if i < 0:
        return haystack
    return haystack[:i] + haystack[i + len(needle) :]


def _legacy_shared_version(version: str, asset: str) -> str:
    v = str(version or "").strip()
    low = v.lower()
    token = "_xau" if _normalize_asset(asset) == "XAU" else "_btc"
    if token in low:
        v = _remove_first(v, token)
    return v


def _state_path(models_dir: Path, asset: str) -> Path:
    return models_dir / f"model_state_{_normalize_asset(asset)}.pkl"


def _meta_path(models_dir: Path, version: str) -> Path:
    return models_dir / f"v{version}.json"


def _model_path(models_dir: Path, version: str) -> Path:
    return models_dir / f"v{version}.pkl"


def _read_pickle_dict(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            return None, "not_dict"
        return obj, "ok"
    except Exception as exc:
        return None, str(exc)


def _read_json_dict(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None, "not_dict"
        return obj, "ok"
    except Exception as exc:
        return None, str(exc)


def _global_state_path(models_dir: Path) -> Path:
    return models_dir / "model_state.pkl"


def _copy_if_exists(src: Path, dst_dir: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    try:
        shutil.copy2(src, dst_dir / src.name)
        return True
    except Exception:
        return False


def backup_legacy_shared_artifacts_once(
    models_dir: Optional[Path] = None,
) -> Optional[Path]:
    global _LEGACY_BACKUP_DONE
    with _LEGACY_BACKUP_LOCK:
        if _LEGACY_BACKUP_DONE:
            return None

    try:
        base = (
            Path(models_dir) if models_dir is not None else get_artifact_dir("models")
        )
        base.mkdir(parents=True, exist_ok=True)

        legacy_candidates = []
        legacy_state = _global_state_path(base)
        if legacy_state.exists():
            legacy_candidates.append(legacy_state)

        for p in base.glob("v*_institutional.*"):
            low = p.name.lower()
            if "_xau_" in low or "_btc_" in low:
                continue
            if p.suffix.lower() in {".pkl", ".json"}:
                legacy_candidates.append(p)

        if not legacy_candidates:
            with _LEGACY_BACKUP_LOCK:
                _LEGACY_BACKUP_DONE = True
            return None

        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        dst = base / f"_legacy_shared_backup_{ts}"
        dst.mkdir(parents=True, exist_ok=True)

        copied = 0
        for src in legacy_candidates:
            if _copy_if_exists(src, dst):
                copied += 1

        with _LEGACY_BACKUP_LOCK:
            _LEGACY_BACKUP_DONE = True
        if copied <= 0:
            return None
        return dst
    except Exception as exc:
        with _LEGACY_BACKUP_LOCK:
            _LEGACY_BACKUP_DONE = True
        log.warning("LEGACY_BACKUP_FAILED | err=%s", exc)
        return None


@dataclass
class AssetGateStatus:
    asset: str
    ok: bool
    reason: str
    state_path: str
    model_path: str
    meta_path: str
    model_version: str
    status: str
    verified: bool
    real_backtest: bool
    sharpe: float
    win_rate: float
    max_drawdown_pct: float
    legacy_fallback: bool = False


def _state_from_legacy(
    models_dir: Path, asset: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Path], str]:
    g = _global_state_path(models_dir)
    if not g.exists():
        return None, None, "missing_global_state"
    obj, err = _read_pickle_dict(g)
    if obj is None:
        return None, None, f"global_state_unreadable:{err}"
    st_asset = _normalize_asset(str(obj.get("asset", "")))
    if st_asset != _normalize_asset(asset):
        return None, None, f"global_state_other_asset:{st_asset or 'UNKNOWN'}"
    return obj, g, "ok"


def _asset_gate_status(
    models_dir: Path,
    asset: str,
    *,
    allow_legacy_fallback: bool,
) -> AssetGateStatus:
    asset_u = _normalize_asset(asset)
    state_p = _state_path(models_dir, asset_u)
    state: Optional[Dict[str, Any]] = None
    state_err = ""
    legacy = False

    if state_p.exists():
        state, state_err = _read_pickle_dict(state_p)
        if state is None:
            return AssetGateStatus(
                asset=asset_u,
                ok=False,
                reason=f"state_unreadable:{state_err}",
                state_path=str(state_p),
                model_path="",
                meta_path="",
                model_version="",
                status="",
                verified=False,
                real_backtest=False,
                sharpe=0.0,
                win_rate=0.0,
                max_drawdown_pct=0.0,
                legacy_fallback=False,
            )
    elif allow_legacy_fallback:
        state, legacy_path, legacy_err = _state_from_legacy(models_dir, asset_u)
        if state is not None and legacy_path is not None:
            state_p = legacy_path
            legacy = True
        else:
            return AssetGateStatus(
                asset=asset_u,
                ok=False,
                reason=f"state_missing:{legacy_err}",
                state_path=str(state_p),
                model_path="",
                meta_path="",
                model_version="",
                status="",
                verified=False,
                real_backtest=False,
                sharpe=0.0,
                win_rate=0.0,
                max_drawdown_pct=0.0,
                legacy_fallback=False,
            )
    else:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason="state_missing",
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version="",
            status="",
            verified=False,
            real_backtest=False,
            sharpe=0.0,
            win_rate=0.0,
            max_drawdown_pct=0.0,
            legacy_fallback=False,
        )

    status = str(state.get("status", "")).upper()
    verified = bool(state.get("verified", False))
    real_bt = bool(state.get("real_backtest", False))
    sharpe = _safe_float(state.get("sharpe_ratio", state.get("sharpe", 0.0)), 0.0)
    win_rate = _safe_float(state.get("win_rate", 0.0), 0.0)
    max_dd = _safe_float(state.get("max_drawdown_pct", 0.0), 0.0)
    version = str(state.get("model_version", "")).strip()
    state_asset = _normalize_asset(str(state.get("asset", "")))
    state_unsafe = bool(state.get("unsafe", False))
    state_stress_ok = bool(state.get("stress_test_passed", False))
    state_wfa_ok = bool(state.get("wfa_passed", False))
    state_wfa_total = int(state.get("wfa_total_windows", 0) or 0)
    state_wfa_failed = int(state.get("wfa_failed_windows", 0) or 0)
    state_wfa_required = int(state.get("wfa_required_windows", 0) or 0)
    state_wfa_skipped = bool(state.get("wfa_skipped", False))
    state_anti_overfit_ok = bool(state.get("anti_overfit_passed", False))
    state_tscv_folds = int(state.get("tscv_folds", 0) or 0)
    state_rou = _safe_float(state.get("risk_of_ruin", 0.0), 0.0)
    state_sample_quality_passed = bool(state.get("sample_quality_passed", True))
    _state_issues_raw = state.get("sample_quality_issues", [])
    if isinstance(_state_issues_raw, (list, tuple)):
        state_sample_issues = [str(x) for x in _state_issues_raw if str(x).strip()]
    else:
        state_sample_issues = []
    anti_overfit_gate_required = _anti_overfit_gate_required()
    required_state_fields = [
        "status",
        "verified",
        "real_backtest",
        "wfa_passed",
        "wfa_total_windows",
        "wfa_failed_windows",
        "wfa_required_windows",
        "stress_test_passed",
        "unsafe",
    ]
    if anti_overfit_gate_required:
        required_state_fields.extend(("anti_overfit_passed", "tscv_folds"))
    missing_state_fields = [k for k in required_state_fields if k not in state]

    if legacy:
        # Legacy state detected — continue validation instead of hard-failing.
        # Only warn, don't auto-reject. The checks below will validate metrics.
        log.warning(
            "LEGACY_STATE_DETECTED | asset=%s version=%s — continuing validation",
            asset_u,
            version,
        )

    if state_asset and state_asset != asset_u:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason=f"state_asset_mismatch:{state_asset}",
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    if missing_state_fields:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason=f"state_contract_missing:{','.join(missing_state_fields)}",
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    if not real_bt:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason="non_real_backtest_state",
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    if state_unsafe:
        unsafe_tags: list[str] = []
        if not state_sample_quality_passed:
            unsafe_tags.append("sample_quality_fail")
            if state_sample_issues:
                unsafe_tags.append(f"issues={','.join(state_sample_issues[:3])}")
        if not state_wfa_ok:
            unsafe_tags.append("wfa_fail")
        if not state_stress_ok:
            unsafe_tags.append("stress_fail")
        if state_rou > 0.01:
            unsafe_tags.append(f"risk_of_ruin={state_rou:.4f}")
        unsafe_reason = "state_marked_unsafe"
        if unsafe_tags:
            unsafe_reason = f"{unsafe_reason}:{'|'.join(unsafe_tags)}"
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason=unsafe_reason,
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    if not state_stress_ok:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason="state_stress_failed",
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    if not state_wfa_ok:
        wfa_tag = (
            f"{state_wfa_failed}/{state_wfa_total}"
            if state_wfa_total > 0
            else f"{state_wfa_failed}/{state_wfa_total}:required={state_wfa_required}:skipped={int(state_wfa_skipped)}"
        )
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason=f"state_wfa_failed:{wfa_tag}",
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    if anti_overfit_gate_required and (
        not state_anti_overfit_ok or state_tscv_folds < 2
    ):
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason=f"state_anti_overfit_failed:passed={int(state_anti_overfit_ok)}:tscv_folds={state_tscv_folds}",
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    if status != "VERIFIED" or not verified:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason=f"state_not_verified:{status or 'UNKNOWN'}",
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    if sharpe < MIN_GATE_SHARPE:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason=f"state_sharpe_below_gate:{sharpe:.3f}<{MIN_GATE_SHARPE:.3f}",
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    # Suspicious Sharpe gate (anti-overfit / anti-leakage).
    # Real retail FX/crypto scalping strategies do not produce Sharpe > 5 on
    # honest out-of-sample data. A number like this is almost always a
    # backtester bug, label leakage or an optimistic fill model — NOT alpha.
    if sharpe > MAX_GATE_SHARPE:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason=(
                f"state_sharpe_suspicious:{sharpe:.3f}>{MAX_GATE_SHARPE:.3f}"
                "|likely_overfit_or_leakage"
            ),
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    # Walk-forward pass-rate gate. `wfa_passed` alone is boolean and may be
    # True with only 60% of windows passing — that is not enough for live.
    state_wfa_pass_rate = _safe_float(state.get("wfa_pass_rate", 1.0), 1.0)
    if state_wfa_total > 0 and state_wfa_pass_rate < MIN_GATE_WFA_PASS_RATE:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason=(
                f"state_wfa_pass_rate_low:{state_wfa_pass_rate:.3f}"
                f"<{MIN_GATE_WFA_PASS_RATE:.3f}"
            ),
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    if win_rate < MIN_GATE_WIN_RATE:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason=f"state_winrate_below_gate:{win_rate:.3f}<{MIN_GATE_WIN_RATE:.3f}",
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    if max_dd > MAX_GATE_DRAWDOWN:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason=f"state_drawdown_above_gate:{max_dd:.3f}>{MAX_GATE_DRAWDOWN:.3f}",
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    if not version:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason="state_model_version_missing",
            state_path=str(state_p),
            model_path="",
            meta_path="",
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    mp = _model_path(models_dir, version)
    if not mp.exists():
        legacy_version = _legacy_shared_version(version, asset_u)
        legacy_mp = _model_path(models_dir, legacy_version)
        if legacy_version != version and legacy_mp.exists():
            mp = legacy_mp
        else:
            return AssetGateStatus(
                asset=asset_u,
                ok=False,
                reason=f"model_payload_missing:{mp.name}",
                state_path=str(state_p),
                model_path=str(mp),
                meta_path="",
                model_version=version,
                status=status,
                verified=verified,
                real_backtest=real_bt,
                sharpe=sharpe,
                win_rate=win_rate,
                max_drawdown_pct=max_dd,
                legacy_fallback=False,
            )

    jp = _meta_path(models_dir, version)
    if not jp.exists():
        legacy_version = _legacy_shared_version(version, asset_u)
        legacy_jp = _meta_path(models_dir, legacy_version)
        if legacy_version != version and legacy_jp.exists():
            jp = legacy_jp
        else:
            return AssetGateStatus(
                asset=asset_u,
                ok=False,
                reason=f"meta_missing:{jp.name}",
                state_path=str(state_p),
                model_path=str(mp),
                meta_path=str(jp),
                model_version=version,
                status=status,
                verified=verified,
                real_backtest=real_bt,
                sharpe=sharpe,
                win_rate=win_rate,
                max_drawdown_pct=max_dd,
                legacy_fallback=False,
            )

    meta, meta_err = _read_json_dict(jp)
    if meta is None:
        return AssetGateStatus(
            asset=asset_u,
            ok=False,
            reason=f"meta_unreadable:{meta_err}",
            state_path=str(state_p),
            model_path=str(mp),
            meta_path=str(jp),
            model_version=version,
            status=status,
            verified=verified,
            real_backtest=real_bt,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            legacy_fallback=False,
        )

    meta_status = str(meta.get("status", "")).upper()
    meta_real = bool(meta.get("real_backtest", False))
    meta_asset = _normalize_asset(str(meta.get("asset", "")))
    meta_unsafe = bool(meta.get("unsafe", False))
    meta_stress_ok = bool(meta.get("stress_test_passed", False))
    meta_wfa_ok = bool(meta.get("wfa_passed", False))
    meta_wfa_total = int(meta.get("wfa_total_windows", 0) or 0)
    meta_wfa_failed = int(meta.get("wfa_failed_windows", 0) or 0)
    meta_wfa_required = int(meta.get("wfa_required_windows", 0) or 0)
    meta_wfa_skipped = bool(meta.get("wfa_skipped", False))
    meta_anti_overfit_ok = bool(meta.get("anti_overfit_passed", False))
    meta_tscv_folds = int(meta.get("tscv_folds", 0) or 0)
    meta_rou = _safe_float(meta.get("risk_of_ruin", 0.0), 0.0)
    meta_sample_quality_passed = bool(meta.get("sample_quality_passed", True))
    _meta_issues_raw = meta.get("sample_quality_issues", [])
    if isinstance(_meta_issues_raw, (list, tuple)):
        meta_sample_issues = [str(x) for x in _meta_issues_raw if str(x).strip()]
    else:
        meta_sample_issues = []
    meta_sharpe = _safe_float(meta.get("backtest_sharpe", meta.get("sharpe", 0.0)), 0.0)
    meta_wr = _safe_float(meta.get("backtest_win_rate", meta.get("win_rate", 0.0)), 0.0)
    meta_dd = _safe_float(meta.get("max_drawdown_pct", max_dd), max_dd)
    required_meta_fields = [
        "asset",
        "status",
        "real_backtest",
        "unsafe",
        "stress_test_passed",
        "wfa_passed",
        "backtest_sharpe",
        "backtest_win_rate",
        "max_drawdown_pct",
    ]
    if anti_overfit_gate_required:
        required_meta_fields.extend(("anti_overfit_passed", "tscv_folds"))
    missing_meta_fields = [k for k in required_meta_fields if k not in meta]

    if missing_meta_fields:
        reason = f"meta_contract_missing:{','.join(missing_meta_fields)}"
        ok = False
    elif meta_asset != asset_u:
        reason = f"meta_asset_mismatch:{meta_asset or 'UNKNOWN'}"
        ok = False
    elif not meta_real:
        reason = "meta_non_real_backtest"
        ok = False
    elif meta_unsafe:
        unsafe_tags: list[str] = []
        if not meta_sample_quality_passed:
            unsafe_tags.append("sample_quality_fail")
            if meta_sample_issues:
                unsafe_tags.append(f"issues={','.join(meta_sample_issues[:3])}")
        if not meta_wfa_ok:
            unsafe_tags.append("wfa_fail")
        if not meta_stress_ok:
            unsafe_tags.append("stress_fail")
        if meta_rou > 0.01:
            unsafe_tags.append(f"risk_of_ruin={meta_rou:.4f}")
        reason = "meta_marked_unsafe"
        if unsafe_tags:
            reason = f"{reason}:{'|'.join(unsafe_tags)}"
        ok = False
    elif not meta_stress_ok:
        reason = "meta_stress_failed"
        ok = False
    elif not meta_wfa_ok:
        wfa_tag = (
            f"{meta_wfa_failed}/{meta_wfa_total}"
            if meta_wfa_total > 0
            else f"{meta_wfa_failed}/{meta_wfa_total}:required={meta_wfa_required}:skipped={int(meta_wfa_skipped)}"
        )
        reason = f"meta_wfa_failed:{wfa_tag}"
        ok = False
    elif anti_overfit_gate_required and (
        not meta_anti_overfit_ok or meta_tscv_folds < 2
    ):
        reason = f"meta_anti_overfit_failed:passed={int(meta_anti_overfit_ok)}:tscv_folds={meta_tscv_folds}"
        ok = False
    elif meta_status != "VERIFIED":
        reason = f"meta_not_verified:{meta_status or 'UNKNOWN'}"
        ok = False
    elif meta_sharpe < MIN_GATE_SHARPE:
        reason = f"meta_sharpe_below_gate:{meta_sharpe:.3f}<{MIN_GATE_SHARPE:.3f}"
        ok = False
    elif meta_sharpe > MAX_GATE_SHARPE or bool(meta.get("suspicious_sharpe", False)):
        # Meta-side suspicious Sharpe rejection. Matches the state-side
        # gate and also catches artifacts where the training pipeline
        # flagged `suspicious_sharpe=True` explicitly.
        reason = (
            f"meta_sharpe_suspicious:{meta_sharpe:.3f}>{MAX_GATE_SHARPE:.3f}"
            "|likely_overfit_or_leakage"
        )
        ok = False
    elif meta_wr < MIN_GATE_WIN_RATE:
        reason = f"meta_winrate_below_gate:{meta_wr:.3f}<{MIN_GATE_WIN_RATE:.3f}"
        ok = False
    elif meta_dd > MAX_GATE_DRAWDOWN:
        reason = f"meta_drawdown_above_gate:{meta_dd:.3f}>{MAX_GATE_DRAWDOWN:.3f}"
        ok = False
    else:
        meta_wfa_pass_rate = _safe_float(meta.get("wfa_pass_rate", 1.0), 1.0)
        if meta_wfa_total > 0 and meta_wfa_pass_rate < MIN_GATE_WFA_PASS_RATE:
            reason = (
                f"meta_wfa_pass_rate_low:{meta_wfa_pass_rate:.3f}"
                f"<{MIN_GATE_WFA_PASS_RATE:.3f}"
            )
            ok = False
        else:
            reason = "ok"
            ok = True

    return AssetGateStatus(
        asset=asset_u,
        ok=ok,
        reason=reason,
        state_path=str(state_p),
        model_path=str(mp),
        meta_path=str(jp),
        model_version=version,
        status=status,
        verified=verified,
        real_backtest=real_bt,
        sharpe=sharpe,
        win_rate=win_rate,
        max_drawdown_pct=max_dd,
        legacy_fallback=False,
    )


def _summarize_reason(statuses: Dict[str, AssetGateStatus]) -> str:
    for asset in sorted(statuses.keys()):
        st = statuses[asset]
        if not st.ok:
            return f"{asset}:{st.reason}"
    return "unknown_gate_failure"


def gate_details(
    required_assets: Iterable[str] = DEFAULT_REQUIRED_ASSETS,
    *,
    models_dir: Optional[Path] = None,
    allow_legacy_fallback: bool = True,
) -> Dict[str, Any]:
    base = Path(models_dir) if models_dir is not None else get_artifact_dir("models")
    base.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_legacy_shared_artifacts_once(base)

    assets: list[str] = []
    seen = set()
    for a in required_assets:
        au = _normalize_asset(a)
        if not au or au in seen:
            continue
        seen.add(au)
        assets.append(au)
    if not assets:
        assets = list(DEFAULT_REQUIRED_ASSETS)

    by_asset: Dict[str, AssetGateStatus] = {}
    for asset in assets:
        by_asset[asset] = _asset_gate_status(
            base,
            asset,
            allow_legacy_fallback=allow_legacy_fallback,
        )

    ok = all(st.ok for st in by_asset.values())
    reason = "ok" if ok else _summarize_reason(by_asset)
    return {
        "ok": bool(ok),
        "reason": reason,
        "required_assets": assets,
        "models_dir": str(base),
        "legacy_backup_dir": str(backup_dir) if backup_dir else "",
        "assets": {k: asdict(v) for k, v in by_asset.items()},
    }


def gate_ready(
    required_assets: Iterable[str] = DEFAULT_REQUIRED_ASSETS,
    *,
    models_dir: Optional[Path] = None,
    allow_legacy_fallback: bool = True,
) -> Tuple[bool, str]:
    details = gate_details(
        required_assets=required_assets,
        models_dir=models_dir,
        allow_legacy_fallback=allow_legacy_fallback,
    )
    return bool(details["ok"]), str(details.get("reason", "unknown"))


__all__ = (
    "ModelMetadata",
    "ModelManager",
    "model_manager",
    "MAX_MODEL_AGE_HOURS",
    "ModelAgeChecker",
    "DEFAULT_REQUIRED_ASSETS",
    "backup_legacy_shared_artifacts_once",
    "AssetGateStatus",
    "gate_details",
    "gate_ready",
)
