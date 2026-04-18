"""
runmain/gate.py — System readiness and model gating logic.

This module controls the live-trading go/no-go mechanism by verifying model
staleness, backtest performance, and payload health. It coordinates partial
asset gating and automated retraining workflows.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from core.config import MAX_GATE_DRAWDOWN, MIN_GATE_SHARPE, MIN_GATE_WIN_RATE
from core.model_engine import DEFAULT_REQUIRED_ASSETS, gate_details
from log_config import get_artifact_dir, get_artifact_path

from .bootstrap import (
    env_float,
    env_truthy,
    log,
    setup_named_logger,
    state,
)

_log_gate = setup_named_logger(
    "gate", file_name="gate.log", level=logging.INFO, backups=3
)


# =============================================================================
# Global Constants & State
# =============================================================================
_GATE_STATUS_LOG_TTL_SEC: float = 60.0
_gate_status_lock = __import__("threading").Lock()
_gate_status_last_sig: str = ""
_gate_status_last_ts: float = 0.0

_RETRAIN_SESSION_COUNT: int = 0
_RETRAIN_SESSION_MAX: int = int(os.environ.get("RETRAIN_SESSION_MAX", "3") or 3)


# =============================================================================
# Environment & Configuration Rules
# =============================================================================
def monitoring_only_mode() -> bool:
    """Legacy monitoring-only startup mode is disabled in production."""
    return False


def auto_retrain_enabled() -> bool:
    """Check if automated retraining is permitted in the current environment."""
    if monitoring_only_mode():
        return False
    return env_truthy("AUTO_RETRAIN_ENABLED", "1")


def required_gate_assets() -> tuple[str, ...]:
    """Retrieve the tuple of assets required to pass the gate."""
    raw = str(os.environ.get("REQUIRED_GATE_ASSETS", "") or "").strip()
    if not raw:
        raw = ",".join(DEFAULT_REQUIRED_ASSETS)
    seen: set[str] = set()
    out: list[str] = []
    for item in raw.split(","):
        a = item.upper().strip()
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return tuple(out) if out else tuple(DEFAULT_REQUIRED_ASSETS)


def log_monitoring_only_profile() -> None:
    """Log the active monitoring-only profile constraints."""
    models_dir = get_artifact_dir("models")
    model_state = get_artifact_path("models", "model_state.pkl")
    log.warning(
        "MONITORING_ONLY_PROFILE | trading_disabled=True auto_train=False "
        "auto_retrain=False | artifact_freeze_required=True "
        "| models_dir=%s | model_state=%s",
        models_dir,
        model_state,
    )


def _partial_gate_enabled(required: Optional[tuple[str, ...]] = None) -> bool:
    """Check if a subset of required assets is allowed to trade."""
    assets = required or required_gate_assets()
    if not env_truthy("PARTIAL_GATE_MODE", "0"):
        return False
    if len(assets) > 1 and env_truthy("STRICT_DUAL_ASSET_MODE", "1"):
        return False
    return True


# =============================================================================
# Private Core Logic
# =============================================================================
def _resolve_training_assets(assets_spec: str = "") -> list[str]:
    """Normalize comma-separated asset spec and guarantee fallback inclusions."""
    default = ",".join(required_gate_assets())
    assets_spec = str(assets_spec or default).strip()
    assets: list[str] = [a.strip().upper() for a in assets_spec.split(",") if a.strip()]
    if not assets:
        assets = list(required_gate_assets())
    req = list(required_gate_assets())
    for a in req:
        if a not in assets:
            assets.append(a)
    return assets


def _passes_soft_fallback_metrics(st: dict[str, Any]) -> bool:
    """Validate core historical performance margins for soft fallback paths."""
    if not isinstance(st, dict):
        return False
    if not bool(st.get("real_backtest", False)):
        return False
    sharpe = float(st.get("sharpe", 0.0) or 0.0)
    win_rate = float(st.get("win_rate", 0.0) or 0.0)
    max_dd = float(st.get("max_drawdown_pct") or 1.0)
    version = str(st.get("model_version", "") or "").strip()
    return (
        sharpe >= MIN_GATE_SHARPE
        and win_rate >= MIN_GATE_WIN_RATE
        and max_dd <= MAX_GATE_DRAWDOWN
        and bool(version)
    )


def _gate_assets_tuple(
    details: Any,
) -> list[tuple[str, bool, str, str, float, float, float, bool]]:
    """Convert payload dictionary to uniform tuple format for iterating constraints."""
    assets: dict[str, Any] = (
        details.get("assets", {}) if isinstance(details, dict) else {}
    )
    items: list[tuple[str, bool, str, str, float, float, float, bool]] = []
    if not isinstance(assets, dict):
        return items
    for asset, st in sorted(assets.items(), key=lambda kv: str(kv[0])):
        if not isinstance(st, dict):
            continue
        try:
            items.append(
                (
                    str(asset),
                    bool(st.get("ok", False)),
                    str(st.get("reason", "unknown")),
                    str(st.get("model_version", "")),
                    float(st.get("sharpe", 0.0) or 0.0),
                    float(st.get("win_rate", 0.0) or 0.0),
                    float(st.get("max_drawdown_pct", 0.0) or 0.0),
                    bool(st.get("legacy_fallback", False)),
                )
            )
        except Exception:
            continue
    return items


def _model_gate_ready() -> tuple[bool, str]:
    """Retrieve full gate readiness state natively without fallback logic."""
    global _gate_status_last_sig, _gate_status_last_ts
    details = gate_details(
        required_assets=required_gate_assets(), allow_legacy_fallback=True
    )
    items = _gate_assets_tuple(details)
    sig = "|".join(
        f"{a}:{int(ok)}:{r}:{v}:{s:.3f}:{w:.3f}:{d:.3f}:{int(lg)}"
        for a, ok, r, v, s, w, d, lg in items
    )
    now = time.time()
    should_log = False
    with _gate_status_lock:
        if (
            sig != _gate_status_last_sig
            or (now - _gate_status_last_ts) >= _GATE_STATUS_LOG_TTL_SEC
        ):
            _gate_status_last_sig = sig
            _gate_status_last_ts = now
            should_log = True
    if should_log:
        for asset, ok, reason, version, sharpe, win_rate, max_dd, legacy in items:
            try:
                _log_gate.info(
                    "ASSET_GATE_STATUS | asset=%s ok=%s reason=%s version=%s "
                    "sharpe=%.3f win_rate=%.3f max_dd=%.3f legacy=%s",
                    asset,
                    ok,
                    reason,
                    version,
                    sharpe,
                    win_rate,
                    max_dd,
                    legacy,
                )
            except Exception:
                continue
    if isinstance(details, dict):
        return bool(details.get("ok", False)), str(details.get("reason", "unknown"))
    return False, "unknown"


def _single_asset_gate_status(asset: str) -> tuple[bool, str]:
    """Check readiness strictly for one asset symbol."""
    asset_u = asset.upper().strip()
    details = gate_details(required_assets=(asset_u,), allow_legacy_fallback=True)
    st = (details.get("assets", {}) or {}).get(asset_u, {})
    if isinstance(st, dict):
        return bool(st.get("ok", False)), str(
            st.get("reason", details.get("reason", "unknown"))
        )
    return bool(details.get("ok", False)), str(details.get("reason", "unknown"))


def _partial_gate_assets() -> list[str]:
    """Return all assets currently passing independent tests."""
    passing: list[str] = []
    for asset in required_gate_assets():
        try:
            det = gate_details(required_assets=(asset,), allow_legacy_fallback=True)
            ast = (det.get("assets", {}) or {}).get(asset, {})
            if isinstance(ast, dict) and bool(ast.get("ok", False)):
                passing.append(asset)
        except Exception as exc:
            log.error("PARTIAL_GATE_ASSET_CHECK_FAILED | asset=%s | err=%s", asset, exc)
            continue
    return passing


def _soft_wfa_fallback_assets(required: Optional[tuple[str, ...]] = None) -> list[str]:
    """Retrieve assets blocked in gate solely due to WFA stress test thresholds."""
    if not env_truthy("PARTIAL_GATE_ALLOW_WFA_FALLBACK", "1"):
        return []
    assets_req = required or required_gate_assets()
    if not assets_req:
        return []
    try:
        det = gate_details(required_assets=assets_req, allow_legacy_fallback=True)
        amap = det.get("assets", {}) if isinstance(det, dict) else {}
        out: list[str] = []
        for asset in assets_req:
            st = amap.get(asset, {}) if isinstance(amap, dict) else {}
            if not isinstance(st, dict):
                continue
            reason = str(st.get("reason", ""))
            if not (
                reason.startswith("state_wfa_failed")
                or reason.startswith("meta_wfa_failed")
            ):
                continue
            if _passes_soft_fallback_metrics(st):
                out.append(asset)
        return out
    except Exception as exc:
        log.error("SOFT_WFA_FALLBACK_FAILED | err=%s", exc)
        return []


def _soft_sample_quality_fallback_assets(
    required: Optional[tuple[str, ...]] = None,
) -> list[str]:
    """Retrieve assets mapped as unsafe solely from poor historic sample limits."""
    if not env_truthy("PARTIAL_GATE_ALLOW_SAMPLE_QUALITY_FALLBACK", "1"):
        return []
    assets_req = required or required_gate_assets()
    if not assets_req:
        return []
    try:
        det = gate_details(required_assets=assets_req, allow_legacy_fallback=True)
        amap = det.get("assets", {}) if isinstance(det, dict) else {}
        out: list[str] = []
        for asset in assets_req:
            st = amap.get(asset, {}) if isinstance(amap, dict) else {}
            if not isinstance(st, dict):
                continue
            reason = str(st.get("reason", "") or "")
            is_sample_only_unsafe = (
                (
                    reason.startswith("state_marked_unsafe:")
                    or reason.startswith("meta_marked_unsafe:")
                )
                and "sample_quality_fail" in reason
                and "wfa_fail" not in reason
                and "stress_fail" not in reason
                and "risk_of_ruin=" not in reason
            )
            if not is_sample_only_unsafe:
                continue
            if _passes_soft_fallback_metrics(st):
                out.append(asset)
        return out
    except Exception as exc:
        log.error("SOFT_SAMPLE_QUALITY_FALLBACK_FAILED | err=%s", exc)
        return []


def _backup_model_artifacts(models_dir: Path) -> Optional[Path]:
    """Backup models cleanly before retraining tasks start."""
    try:
        if not models_dir.exists():
            return None
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        backup_dir = models_dir / f"_backup_{ts}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for p in models_dir.iterdir():
            if p.is_file() and not p.name.startswith("_backup_"):
                if p.name == "model_state.pkl" or p.suffix in {".pkl", ".json"}:
                    shutil.copy2(p, backup_dir / p.name)
        return backup_dir
    except Exception as exc:
        log.warning("Model backup failed: %s", exc)
        return None


def _restore_model_artifacts(backup_dir: Optional[Path], models_dir: Path) -> None:
    """Revert failed retrains back to stable model artifacts."""
    if backup_dir is None:
        return
    try:
        for p in backup_dir.iterdir():
            if p.is_file():
                shutil.copy2(p, models_dir / p.name)
    except Exception as exc:
        log.warning("Model restore failed: %s", exc)


def _safe_retrain_models(notifier: Any) -> bool:
    """Wrap training logic dynamically inside a clean backup-ready scope."""
    try:
        from Backtest.engine import run_institutional_backtest
    except Exception as exc:
        log.error("Retraining import failed: %s", exc)
        return False
    models_dir = get_artifact_dir("models")
    backup_dir = _backup_model_artifacts(models_dir)
    required = list(required_gate_assets())
    try:
        assets = _resolve_training_assets("XAU,BTC")
        passed: list[str] = []
        failed: list[str] = []
        for asset in assets:
            try:
                det = gate_details(required_assets=(asset,), allow_legacy_fallback=True)
                ast = (det.get("assets", {}) or {}).get(asset, {})
                if isinstance(ast, dict) and bool(ast.get("ok", False)):
                    log.info("RETRAIN_SKIP_ALREADY_PASS | asset=%s", asset)
                    passed.append(asset)
                    continue
            except Exception:
                pass
            log.info("Retraining %s model...", asset)
            metrics = run_institutional_backtest(asset)
            gate_ok, gate_reason = _single_asset_gate_status(asset)
            if gate_ok:
                passed.append(asset)
                log.info(
                    "Retraining gate passed | asset=%s reason=%s "
                    "sharpe=%.3f win_rate=%.3f max_dd=%.5f",
                    asset,
                    gate_reason,
                    float(getattr(metrics, "sharpe_ratio", 0.0) or 0.0),
                    float(getattr(metrics, "win_rate", 0.0) or 0.0),
                    float(getattr(metrics, "max_drawdown_pct", 0.0) or 0.0),
                )
            else:
                failed.append(asset)
                log.error(
                    "Retraining gate failed | asset=%s reason=%s "
                    "sharpe=%.3f win_rate=%.3f max_dd=%.5f",
                    asset,
                    gate_reason,
                    float(getattr(metrics, "sharpe_ratio", 0.0) or 0.0),
                    float(getattr(metrics, "win_rate", 0.0) or 0.0),
                    float(getattr(metrics, "max_drawdown_pct", 0.0) or 0.0),
                )
        gate_ok, gate_reason, gate_active = model_gate_ready_effective()
        if gate_ok:
            req_now = set(required)
            active_set = set(gate_active)
            if not req_now.issubset(active_set):
                log.warning(
                    "RETRAIN_POSTCHECK_PARTIAL | active=%s blocked=%s reason=%s",
                    ",".join(gate_active) or "-",
                    ",".join(sorted(req_now - active_set)) or "-",
                    gate_reason,
                )
            return True
        log.error(
            "RETRAIN_POSTCHECK_FAILED | reason=%s | passed=%s failed=%s required=%s",
            gate_reason,
            ",".join(passed) or "-",
            ",".join(failed) or "-",
            ",".join(required),
        )
        if not passed:
            _restore_model_artifacts(backup_dir, models_dir)
        return False
    except Exception as exc:
        log.error("Retraining exception: %s", exc, exc_info=True)
        _restore_model_artifacts(backup_dir, models_dir)
        return False


# =============================================================================
# Public Application Logic
# =============================================================================
def model_gate_ready_effective() -> tuple[bool, str, list[str]]:
    """Determine holistic system pass tracking acceptable asset subsets."""
    required = list(required_gate_assets())
    gate_ok, gate_reason = _model_gate_ready()
    if gate_ok:
        return True, gate_reason, required
    if not _partial_gate_enabled(tuple(required)):
        return False, gate_reason, []

    partial = [a for a in _partial_gate_assets() if a in required]
    if partial:
        blocked = [a for a in required if a not in partial]
        reason = (
            f"partial_gate:{gate_reason}:blocked={','.join(blocked)}"
            if blocked
            else f"partial_gate:{gate_reason}"
        )
        return True, reason, partial

    wfa = [a for a in _soft_wfa_fallback_assets(tuple(required)) if a in required]
    if wfa:
        blocked = [a for a in required if a not in wfa]
        reason = (
            f"partial_wfa_fallback:{gate_reason}:blocked={','.join(blocked)}"
            if blocked
            else f"partial_wfa_fallback:{gate_reason}"
        )
        return True, reason, wfa

    sq = [
        a
        for a in _soft_sample_quality_fallback_assets(tuple(required))
        if a in required
    ]
    if sq:
        blocked = [a for a in required if a not in sq]
        reason = (
            f"partial_sample_quality_fallback:{gate_reason}:blocked={','.join(blocked)}"
            if blocked
            else f"partial_sample_quality_fallback:{gate_reason}"
        )
        return True, reason, sq

    return False, gate_reason, []


def models_ready() -> tuple[bool, str]:
    """Verify that payload binaries actively verify matching requirements."""
    try:
        ready, reason, active_assets = model_gate_ready_effective()
        if not ready:
            return False, reason
        det = gate_details(
            required_assets=required_gate_assets(), allow_legacy_fallback=True
        )
        amap = det.get("assets", {}) if isinstance(det, dict) else {}
        for asset in active_assets:
            st = amap.get(asset, {}) if isinstance(amap, dict) else {}
            if not isinstance(st, dict):
                return False, f"{asset}:missing_gate_details"
            version = str(st.get("model_version", "") or "").strip()
            model_path = str(st.get("model_path", "") or "")
            if not (model_path and os.path.exists(model_path)) and version:
                model_path = str(get_artifact_path("models", f"v{version}.pkl"))
            if not (model_path and os.path.exists(model_path)):
                return False, f"{asset}:missing_model_payload"
            meta_path = str(st.get("meta_path", "") or "")
            if not (meta_path and os.path.exists(meta_path)) and version:
                meta_path = str(get_artifact_path("models", f"v{version}.json"))
            if not (meta_path and os.path.exists(meta_path)):
                return False, f"{asset}:missing_model_meta"
        return True, reason
    except Exception as exc:
        log.error("MODELS_READY_CHECK_FAILED | err=%s", exc)
        return False, f"error:{exc}"


def prime_engine_model_health() -> None:
    """Pre-flight internal engine payloads cleanly passing errors downward."""
    engine = state.engine
    if engine is None:
        return
    try:
        engine._check_model_health()
    except Exception as exc:
        log.error(
            "ENGINE_MODEL_HEALTH_PREFLIGHT_FAILED | err=%s | tb=%s",
            exc,
            traceback.format_exc(),
        )
        try:
            reason_now = (
                str(getattr(engine, "_gate_last_reason", "") or "").strip().lower()
            )
            if not reason_now or reason_now == "unknown":
                engine._gate_last_reason = f"health_check_error:{type(exc).__name__}"
        except Exception:
            pass


def auto_train_models_strict() -> bool:  # noqa: C901
    """Trigger strict retraining pipeline with file-backed cooldown memory."""
    _log = logging.getLogger("main")
    try:
        from Backtest.engine import run_institutional_backtest
    except Exception as exc:
        _log.error("Auto-train(strict) import failed: %s", exc)
        return False
    assets_default = ",".join(required_gate_assets())
    assets_env = str(
        os.environ.get("AUTO_TRAIN_ASSETS", assets_default) or assets_default
    )
    assets: list[str] = _resolve_training_assets(assets_env)
    assets_set = set(assets)
    retry_hours = max(0.0, env_float("AUTO_TRAIN_RETRY_HOURS", 3.0))
    failure_cooldown_hours = max(
        0.0, env_float("AUTO_TRAIN_FAILURE_COOLDOWN_HOURS", max(6.0, retry_hours))
    )
    force_retry_on_recent_fail = env_truthy(
        "AUTO_TRAIN_FORCE_RETRY_WHEN_RECENT_GATE_FAIL", "0"
    )
    failure_registry_path = get_artifact_path("models", "auto_train_failures.json")

    def _load_failures() -> dict[str, dict[str, Any]]:
        try:
            if not failure_registry_path.exists():
                return {}
            with failure_registry_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if not isinstance(raw, dict):
                return {}
            return {
                ku: dict(v)
                for k, v in raw.items()
                if (ku := k.upper().strip())
                and ku in assets_set
                and isinstance(v, dict)
            }
        except Exception:
            return {}

    def _save_failures(data: dict[str, dict[str, Any]]) -> None:
        try:
            failure_registry_path.parent.mkdir(parents=True, exist_ok=True)
            clean = {
                ku: {
                    "ts": float(v.get("ts", 0.0) or 0.0),
                    "reason": str(v.get("reason", "unknown")),
                    "count": int(v.get("count", 1) or 1),
                }
                for k, v in data.items()
                if (ku := k.upper().strip())
                and ku in assets_set
                and isinstance(v, dict)
            }
            with failure_registry_path.open("w", encoding="utf-8") as fh:
                json.dump(clean, fh, indent=2)
        except Exception as exc:
            _log.warning(
                "Auto-train(strict) could not persist failure registry: %s", exc
            )

    failures = _load_failures()

    auto_fast = env_truthy("AUTO_TRAIN_FAST_MODE", "1")
    fast_defaults: dict[str, str] = {}
    if auto_fast:
        fast_defaults = {
            "INSTITUTIONAL_FAST_MODE": "1",
            "INSTITUTIONAL_REQUIRE_ALPHA_GATE": os.environ.get(
                "AUTO_TRAIN_REQUIRE_ALPHA_GATE", "0"
            )
            or "0",
            "TRAIN_MAX_BARS": os.environ.get("AUTO_TRAIN_MAX_BARS", "20000") or "20000",
            "INSTITUTIONAL_MAX_ITERS": os.environ.get("AUTO_TRAIN_MAX_ITERS", "700")
            or "700",
            "INSTITUTIONAL_EARLY_STOPPING_ROUNDS": os.environ.get(
                "AUTO_TRAIN_EARLY_STOPPING_ROUNDS", "120"
            )
            or "120",
            "INSTITUTIONAL_CV_MAX_ITERS": os.environ.get(
                "AUTO_TRAIN_CV_MAX_ITERS", "180"
            )
            or "180",
            "INSTITUTIONAL_CV_MAX_SAMPLES": os.environ.get(
                "AUTO_TRAIN_CV_MAX_SAMPLES", "18000"
            )
            or "18000",
            "INSTITUTIONAL_TSCV_FOLDS": os.environ.get("AUTO_TRAIN_TSCV_FOLDS", "3")
            or "3",
            "INSTITUTIONAL_WFA_WINDOWS": os.environ.get("AUTO_TRAIN_WFA_WINDOWS", "3")
            or "3",
        }
        _log.info(
            "Auto-train(strict) FAST_PROFILE | iters=%s cv_iters=%s bars=%s",
            fast_defaults["INSTITUTIONAL_MAX_ITERS"],
            fast_defaults["INSTITUTIONAL_CV_MAX_ITERS"],
            fast_defaults["TRAIN_MAX_BARS"],
        )

    passed_assets: list[str] = []
    failed_assets: list[str] = []
    for asset in assets:
        if failure_cooldown_hours > 0:
            rec = failures.get(asset, {})
            ts = float(rec.get("ts", 0.0) or 0.0) if isinstance(rec, dict) else 0.0
            if ts > 0.0:
                age_h = (time.time() - ts) / 3600.0
                if age_h < failure_cooldown_hours:
                    remain_h = max(0.0, failure_cooldown_hours - age_h)
                    _log.warning(
                        "Auto-train(strict) SKIP | asset=%s reason=recent_failure age=%.2fh < %.2fh remain=%.2fh last=%s",
                        asset,
                        age_h,
                        failure_cooldown_hours,
                        remain_h,
                        (
                            str(rec.get("reason", "unknown"))
                            if isinstance(rec, dict)
                            else "unknown"
                        ),
                    )
                    failed_assets.append(asset)
                    continue

        if retry_hours > 0:
            try:
                st_path = get_artifact_path("models", f"model_state_{asset}.pkl")
                if st_path.exists():
                    age_h = (time.time() - st_path.stat().st_mtime) / 3600.0
                    if age_h < retry_hours:
                        gate_ok_now, gate_reason_now = _single_asset_gate_status(asset)
                        if gate_ok_now or not force_retry_on_recent_fail:
                            _log.info(
                                "Auto-train(strict) SKIP | asset=%s reason=recent_state age=%.2fh < %.2fh gate_ok=%s gate_reason=%s",
                                asset,
                                age_h,
                                retry_hours,
                                gate_ok_now,
                                gate_reason_now,
                            )
                            (passed_assets if gate_ok_now else failed_assets).append(
                                asset
                            )
                            continue
                        _log.warning(
                            "Auto-train(strict) FORCE_RETRAIN | asset=%s reason=recent_state_but_gate_failed age=%.2fh gate_reason=%s",
                            asset,
                            age_h,
                            gate_reason_now,
                        )
            except Exception:
                pass

        try:
            det = gate_details(required_assets=(asset,), allow_legacy_fallback=True)
            ast = (det.get("assets", {}) or {}).get(asset, {})
            if isinstance(ast, dict) and bool(ast.get("ok", False)):
                _log.info(
                    "Auto-train(strict) SKIP | asset=%s reason=gate_already_ok sharpe=%.3f",
                    asset,
                    float(ast.get("sharpe", 0.0) or 0.0),
                )
                passed_assets.append(asset)
                continue
        except Exception:
            pass

        force_retrain = False
        force_retrain_reason = ""
        try:
            gate_ok_pre, gate_reason_pre = _single_asset_gate_status(asset)
            r = str(gate_reason_pre or "")
            wfa_no_evidence = r.startswith(
                ("state_wfa_failed", "meta_wfa_failed")
            ) and (":0/0" in r or "unknown" in r)
            anti_overfit_or_contract = r.startswith(
                (
                    "state_anti_overfit_failed",
                    "meta_anti_overfit_failed",
                    "state_contract_missing",
                    "meta_contract_missing",
                )
            )
            force_retrain = bool(
                (not gate_ok_pre) and (wfa_no_evidence or anti_overfit_or_contract)
            )
            force_retrain_reason = r
        except Exception:
            pass

        prev_force = os.environ.get("BACKTEST_FORCE_RETRAIN")
        if force_retrain:
            os.environ["BACKTEST_FORCE_RETRAIN"] = "1"
            _log.warning(
                "Auto-train(strict) FORCE_MODEL_RETRAIN | asset=%s reason=%s",
                asset,
                force_retrain_reason or "gate_failed",
            )
        prev_env: dict[str, Optional[str]] = {}
        if auto_fast:
            for k, v in fast_defaults.items():
                prev_env[k] = os.environ.get(k)
                os.environ[k] = v

        try:
            metrics = run_institutional_backtest(asset)
            gate_ok_asset, gate_reason_asset = _single_asset_gate_status(asset)
            if gate_ok_asset:
                passed_assets.append(asset)
                failures.pop(asset, None)
                _log.info(
                    "Auto-train(strict) passed | asset=%s sharpe=%.3f win_rate=%.3f max_dd=%.3f gate_reason=%s",
                    asset,
                    float(getattr(metrics, "sharpe_ratio", 0.0) or 0.0),
                    float(getattr(metrics, "win_rate", 0.0) or 0.0),
                    float(getattr(metrics, "max_drawdown_pct", 0.0) or 0.0),
                    gate_reason_asset,
                )
            else:
                failed_assets.append(asset)
                failures[asset] = {
                    "ts": time.time(),
                    "reason": str(gate_reason_asset or "gate_failed"),
                    "count": int(failures.get(asset, {}).get("count", 0) or 0) + 1,
                }
                _log.warning(
                    "Auto-train(strict) failed gate | asset=%s reason=%s sharpe=%.3f win_rate=%.3f max_dd=%.3f",
                    asset,
                    gate_reason_asset,
                    float(getattr(metrics, "sharpe_ratio", 0.0) or 0.0),
                    float(getattr(metrics, "win_rate", 0.0) or 0.0),
                    float(getattr(metrics, "max_drawdown_pct", 0.0) or 0.0),
                )
        except Exception as exc:
            _log.error("Auto-train(strict) failed | asset=%s err=%s", asset, exc)
            failed_assets.append(asset)
            failures[asset] = {
                "ts": time.time(),
                "reason": f"exception:{exc}",
                "count": int(failures.get(asset, {}).get("count", 0) or 0) + 1,
            }
        finally:
            if force_retrain:
                if prev_force is None:
                    os.environ.pop("BACKTEST_FORCE_RETRAIN", None)
                else:
                    os.environ["BACKTEST_FORCE_RETRAIN"] = prev_force
            if auto_fast:
                for k, old in prev_env.items():
                    if old is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = old

    _save_failures(failures)
    if failed_assets:
        _log.error(
            "AUTO_TRAIN_QUALITY_FAILED | failed_assets=%s", ",".join(failed_assets)
        )
    post_ok, post_reason, post_active = model_gate_ready_effective()
    if post_ok:
        required_now = set(required_gate_assets())
        active_set = set(post_active)
        if required_now.issubset(active_set):
            return True
        if not _partial_gate_enabled(tuple(sorted(required_now))):
            _log.error(
                "AUTO_TRAIN_STRICT_DUAL_ASSET_BLOCK | active=%s blocked=%s reason=%s",
                ",".join(post_active) or "-",
                ",".join(sorted(required_now - active_set)) or "-",
                post_reason,
            )
            return False
        _log.warning(
            "AUTO_TRAIN_PARTIAL_GATE | active=%s blocked=%s reason=%s",
            ",".join(post_active) or "-",
            ",".join(sorted(required_now - active_set)) or "-",
            post_reason,
        )
        return True
    if not set(required_gate_assets()).issubset(set(passed_assets)):
        missing = sorted(set(required_gate_assets()) - set(passed_assets))
        _log.error(
            "AUTO_TRAIN_REQUIRED_ASSET_NOT_PASSED | missing=%s passed=%s",
            ",".join(missing) or "-",
            ",".join(sorted(set(passed_assets))) or "-",
        )
    return False


def run_retraining_cycle(notifier: Any, *, reason: str) -> bool:
    """Pause engine safely, trigger model retrain cycle, verify, and reload."""
    global _RETRAIN_SESSION_COUNT
    engine = state.engine
    if engine is None:
        log.error("run_retraining_cycle | engine not wired")
        return False
    _RETRAIN_SESSION_COUNT += 1
    if _RETRAIN_SESSION_COUNT > _RETRAIN_SESSION_MAX:
        log.warning(
            "RETRAIN_SESSION_LIMIT_REACHED | attempts=%d max=%d reason=%s | Continuing smoothly.",
            _RETRAIN_SESSION_COUNT,
            _RETRAIN_SESSION_MAX,
            reason,
        )
        return False
    log.warning(
        "Retraining cycle started | reason=%s attempt=%d/%d",
        reason,
        _RETRAIN_SESSION_COUNT,
        _RETRAIN_SESSION_MAX,
    )
    try:
        try:
            engine._retraining_mode = True
            log.info("Trading paused during retraining")
        except Exception:
            pass
        if not _safe_retrain_models(notifier):
            log.warning("Retraining failed. Using old model.")
            notifier.notify("⚠️ Бозомӯзӣ ноком шуд.\nМодели пешина истифода мешавад.")
            return False
        log.info("Retraining complete. Resuming trading.")
        notifier.notify("✅ Бозомӯзӣ анҷом ёфт.\nМодели нав бор шуд.")
        try:
            engine.reload_model()
            post_ok, post_reason, post_active = model_gate_ready_effective()
            if not post_ok:
                log.error("RETRAIN_POSTCHECK_FAILED | reason=%s", post_reason)
                notifier.notify(f"⚠️ Санҷиши пас аз бозомӯзӣ нагузашт: {post_reason}")
                return False
            req_now = set(required_gate_assets())
            active_set = set(post_active)
            if not req_now.issubset(active_set):
                log.warning(
                    "RETRAIN_POSTCHECK_PARTIAL | active=%s blocked=%s reason=%s",
                    ",".join(post_active) or "-",
                    ",".join(sorted(req_now - active_set)) or "-",
                    post_reason,
                )
            try:
                engine.clear_manual_stop()
            except Exception:
                pass
            return True
        except Exception as exc:
            log.error("Model reload failed: %s", exc)
            notifier.notify(
                f"⚠️ Санҷиши пас аз бозомӯзӣ нагузашт: model_reload_failed:{exc}"
            )
            return False
    finally:
        try:
            engine._retraining_mode = False
        except Exception:
            pass


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "auto_retrain_enabled",
    "auto_train_models_strict",
    "log_monitoring_only_profile",
    "model_gate_ready_effective",
    "models_ready",
    "monitoring_only_mode",
    "prime_engine_model_health",
    "required_gate_assets",
    "run_retraining_cycle",
]
