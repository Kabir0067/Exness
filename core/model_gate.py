from __future__ import annotations

import json
import logging
import pickle
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from log_config import get_artifact_dir
from core.config import MAX_GATE_DRAWDOWN, MIN_GATE_SHARPE, MIN_GATE_WIN_RATE

log = logging.getLogger("core.model_gate")

DEFAULT_REQUIRED_ASSETS: Tuple[str, ...] = ("XAU", "BTC")
_LEGACY_BACKUP_DONE = False
_LEGACY_BACKUP_LOCK = threading.Lock()


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


def backup_legacy_shared_artifacts_once(models_dir: Optional[Path] = None) -> Optional[Path]:
    global _LEGACY_BACKUP_DONE
    with _LEGACY_BACKUP_LOCK:
        if _LEGACY_BACKUP_DONE:
            return None

    try:
        base = Path(models_dir) if models_dir is not None else get_artifact_dir("models")
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


def _state_from_legacy(models_dir: Path, asset: str) -> Tuple[Optional[Dict[str, Any]], Optional[Path], str]:
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
    state_rou = _safe_float(state.get("risk_of_ruin", 0.0), 0.0)
    state_sample_quality_passed = bool(state.get("sample_quality_passed", True))
    _state_issues_raw = state.get("sample_quality_issues", [])
    if isinstance(_state_issues_raw, (list, tuple)):
        state_sample_issues = [str(x) for x in _state_issues_raw if str(x).strip()]
    else:
        state_sample_issues = []
    required_state_fields = (
        "status",
        "verified",
        "real_backtest",
        "wfa_passed",
        "wfa_total_windows",
        "wfa_failed_windows",
        "wfa_required_windows",
        "stress_test_passed",
        "unsafe",
    )
    missing_state_fields = [k for k in required_state_fields if k not in state]

    if legacy:
        # Legacy state detected — continue validation instead of hard-failing.
        # Only warn, don't auto-reject. The checks below will validate metrics.
        log.warning(
            "LEGACY_STATE_DETECTED | asset=%s version=%s — continuing validation",
            asset_u, version,
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
            else "unknown"
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
    required_meta_fields = (
        "asset",
        "status",
        "real_backtest",
        "unsafe",
        "stress_test_passed",
        "wfa_passed",
        "backtest_sharpe",
        "backtest_win_rate",
        "max_drawdown_pct",
    )
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
            else "unknown"
        )
        reason = f"meta_wfa_failed:{wfa_tag}"
        ok = False
    elif meta_status != "VERIFIED":
        reason = f"meta_not_verified:{meta_status or 'UNKNOWN'}"
        ok = False
    elif meta_sharpe < MIN_GATE_SHARPE:
        reason = f"meta_sharpe_below_gate:{meta_sharpe:.3f}<{MIN_GATE_SHARPE:.3f}"
        ok = False
    elif meta_wr < MIN_GATE_WIN_RATE:
        reason = f"meta_winrate_below_gate:{meta_wr:.3f}<{MIN_GATE_WIN_RATE:.3f}"
        ok = False
    elif meta_dd > MAX_GATE_DRAWDOWN:
        reason = f"meta_drawdown_above_gate:{meta_dd:.3f}>{MAX_GATE_DRAWDOWN:.3f}"
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
