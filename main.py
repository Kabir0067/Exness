from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import shutil
import signal
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, Optional, Protocol

import urllib3

from log_config import (
    LOG_DIR as LOG_ROOT,
    attach_global_handler_to_loggers,
    configure_logging,
    get_artifact_dir,
    get_artifact_path,
    get_log_path,
    log_dir_stats,
)
from core.config import MAX_GATE_DRAWDOWN, MIN_GATE_SHARPE, MIN_GATE_WIN_RATE
from core.model_gate import DEFAULT_REQUIRED_ASSETS, gate_details

# =============================================================================
# Globals / runtime wiring
# =============================================================================
TG_HEALTH_NOTIFY = False

bot: Any = None
ADMIN: int = 0
bot_commands: Optional[Callable[[], None]] = None
send_signal_notification: Optional[Callable[[str, Any], None]] = None
engine: Any = None

_tg_available: bool = True

# Disable global aggregate file log. Each module writes to its own log file.
_system_log_handler = configure_logging(level="INFO", system_log_name=None, console=True)

_GATE_STATUS_LOG_TTL_SEC: float = 60.0
_gate_status_log_lock = Lock()
_gate_status_last_sig: str = ""
_gate_status_last_ts: float = 0.0

LOG_DIR = LOG_ROOT


# =============================================================================
# Logger helpers (no duplicates, no recursion, production-safe)
# =============================================================================
_FMT = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _stdout_stream():
    return getattr(sys, "__stdout__", sys.stdout)


def _setup_named_logger(
    name: str,
    *,
    file_name: str,
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backups: int = 5,
) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.propagate = False

    # Avoid duplicates (per-handler type+target)
    has_file = False
    has_console = False
    for h in list(lg.handlers):
        if isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == str(get_log_path(file_name)):
            has_file = True
            h.setLevel(level)
            h.setFormatter(_FMT)
        elif isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            has_console = True
            h.setLevel(level)
            h.setFormatter(_FMT)

    if not has_file:
        fh = RotatingFileHandler(
            str(get_log_path(file_name)),
            maxBytes=int(max_bytes),
            backupCount=int(backups),
            encoding="utf-8",
            delay=True,
        )
        fh.setLevel(level)
        fh.setFormatter(_FMT)
        lg.addHandler(fh)

    if not has_console:
        ch = logging.StreamHandler(_stdout_stream())
        ch.setLevel(level)
        ch.setFormatter(_FMT)
        lg.addHandler(ch)

    return lg


log = _setup_named_logger("main", file_name="main.log", level=logging.INFO, backups=5)
log_super = _setup_named_logger("telegram.supervisor", file_name="telegram.log", level=logging.INFO, backups=3)


# =============================================================================
# StdIO + exception hooks (no-crash)
# =============================================================================
class _StdToLogger:
    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = int(level)

    def write(self, buf: str) -> None:
        s = str(buf)
        if not s:
            return
        for line in s.rstrip().splitlines():
            if line:
                self._logger.log(self._level, line)

    def flush(self) -> None:
        return


def _setup_exception_hooks() -> None:
    def _handle_exception(exc_type, exc, tb) -> None:
        try:
            tb_txt = "".join(traceback.format_tb(tb)) if tb else ""
            logging.getLogger("main").error("UNCAUGHT_EXCEPTION | %s | tb=%s", exc, tb_txt)
        finally:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _handle_exception

    if hasattr(__import__("threading"), "excepthook"):
        import threading as _threading

        def _thread_excepthook(args):  # type: ignore[no-redef]
            tb_txt = ""
            try:
                tb_txt = "".join(traceback.format_tb(getattr(args, "exc_traceback", None)))
            except Exception:
                tb_txt = ""
            logging.getLogger("main").error(
                "THREAD_EXCEPTION | thread=%s exc=%s | tb=%s",
                getattr(args, "thread", None),
                getattr(args, "exc_value", None),
                tb_txt,
            )

        _threading.excepthook = _thread_excepthook


_setup_exception_hooks()

# Redirect prints to logger (handlers use sys.__stdout__ so no recursion)
sys.stdout = _StdToLogger(logging.getLogger("stdout"), logging.INFO)
sys.stderr = _StdToLogger(logging.getLogger("stderr"), logging.ERROR)


# =============================================================================
# Utils
# =============================================================================
def _env_truthy(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name, default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def sleep_interruptible(stop_event: Event, seconds: float) -> None:
    end = time.monotonic() + float(seconds)
    while not stop_event.is_set():
        left = end - time.monotonic()
        if left <= 0:
            return
        stop_event.wait(timeout=min(0.5, left))


class SingletonInstance:
    """
    Ensure only one instance runs by binding a TCP socket.
    If bind fails, another instance is assumed to be running.
    """

    def __init__(self, port: int = 12345):
        self._port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._locked = False

    def __enter__(self):
        try:
            self._sock.bind(("127.0.0.1", self._port))
            # keep it bound for entire process lifetime
            self._sock.listen(1)
            self._locked = True
            return self
        except OSError as exc:
            self._locked = False
            raise RuntimeError(f"Another instance is already running on port {self._port}") from exc

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._locked:
            try:
                self._sock.close()
            except Exception:
                pass


@dataclass(frozen=True)
class Backoff:
    base: float = 1.0
    factor: float = 2.0
    max_delay: float = 60.0

    def delay(self, attempt: int) -> float:
        a = int(attempt)
        if a <= 1:
            return min(self.max_delay, self.base)
        try:
            return min(self.max_delay, self.base * (self.factor ** (a - 1)))
        except Exception:
            return float(self.max_delay)


class RateLimiter:
    """Allow an action at most once per interval seconds (per key)."""

    def __init__(self, interval_sec: float) -> None:
        self.interval = float(interval_sec)
        self._lock = Lock()
        self._last: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            last = float(self._last.get(key, 0.0) or 0.0)
            if (now - last) >= self.interval:
                self._last[key] = now
                return True
            return False


# =============================================================================
# Network exceptions
# =============================================================================
try:
    import requests  # type: ignore
    from requests.exceptions import (  # type: ignore
        ChunkedEncodingError,
        ConnectTimeout,
        ConnectionError as RequestsConnectionError,
        ReadTimeout,
        RequestException,
    )
except Exception:  # pragma: no cover
    RequestException = Exception  # type: ignore
    ReadTimeout = Exception  # type: ignore
    ConnectTimeout = Exception  # type: ignore
    RequestsConnectionError = Exception  # type: ignore
    ChunkedEncodingError = Exception  # type: ignore

NETWORK_EXC = (
    RequestException,
    ConnectionError,
    TimeoutError,
    socket.gaierror,
    socket.timeout,
    OSError,
)

_NET_ERRS_TG = (
    ReadTimeout,
    ConnectTimeout,
    RequestException,
    urllib3.exceptions.ReadTimeoutError,
    urllib3.exceptions.ProtocolError,
    ConnectionError,
    RequestsConnectionError,
    ChunkedEncodingError,
    http.client.RemoteDisconnected,
)


# =============================================================================
# Model gate (compact + behavior-preserving)
# =============================================================================
def _required_gate_assets() -> tuple[str, ...]:
    raw = str(os.environ.get("REQUIRED_GATE_ASSETS", "") or "").strip()
    if not raw:
        raw = ",".join(DEFAULT_REQUIRED_ASSETS)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        a = str(item or "").upper().strip()
        if not a or a in seen:
            continue
        seen.add(a)
        out.append(a)
    return tuple(out or list(DEFAULT_REQUIRED_ASSETS))


def _gate_assets_tuple(details: Any) -> list[tuple[str, bool, str, str, float, float, float, bool]]:
    assets = details.get("assets", {}) if isinstance(details, dict) else {}
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
    details = gate_details(required_assets=_required_gate_assets(), allow_legacy_fallback=True)
    items = _gate_assets_tuple(details)

    sig = "|".join(
        f"{a}:{int(ok)}:{r}:{v}:{s:.3f}:{w:.3f}:{d:.3f}:{int(lg)}" for a, ok, r, v, s, w, d, lg in items
    )

    now = time.time()
    should_log = False
    global _gate_status_last_sig, _gate_status_last_ts
    with _gate_status_log_lock:
        if (sig != _gate_status_last_sig) or ((now - _gate_status_last_ts) >= _GATE_STATUS_LOG_TTL_SEC):
            _gate_status_last_sig = sig
            _gate_status_last_ts = now
            should_log = True

    if should_log:
        for asset, ok, reason, version, sharpe, win_rate, max_dd, legacy in items:
            try:
                log.info(
                    "ASSET_GATE_STATUS | asset=%s ok=%s reason=%s version=%s sharpe=%.3f win_rate=%.3f max_dd=%.3f legacy=%s",
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
    asset_u = str(asset or "").upper().strip()
    details = gate_details(required_assets=(asset_u,), allow_legacy_fallback=True)
    st = (details.get("assets", {}) or {}).get(asset_u, {})
    if isinstance(st, dict):
        return bool(st.get("ok", False)), str(st.get("reason", details.get("reason", "unknown")))
    return bool(details.get("ok", False)), str(details.get("reason", "unknown"))


def _partial_gate_assets() -> list[str]:
    passing: list[str] = []
    for asset in _required_gate_assets():
        try:
            det = gate_details(required_assets=(asset,), allow_legacy_fallback=True)
            ast = (det.get("assets", {}) or {}).get(asset, {})
            if isinstance(ast, dict) and bool(ast.get("ok", False)):
                passing.append(asset)
        except Exception:
            continue
    return passing


def _soft_wfa_fallback_assets(required_assets: Optional[tuple[str, ...]] = None) -> list[str]:
    if not _env_truthy("PARTIAL_GATE_ALLOW_WFA_FALLBACK", "1"):
        return []
    assets_req = tuple(required_assets or _required_gate_assets())
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
            if not (reason.startswith("state_wfa_failed") or reason.startswith("meta_wfa_failed")):
                continue
            if not bool(st.get("real_backtest", False)):
                continue
            sharpe = float(st.get("sharpe", 0.0) or 0.0)
            win_rate = float(st.get("win_rate", 0.0) or 0.0)
            max_dd_raw = st.get("max_drawdown_pct", 1.0)
            max_dd = float(1.0 if max_dd_raw is None else max_dd_raw)
            version = str(st.get("model_version", "") or "").strip()
            if sharpe >= MIN_GATE_SHARPE and win_rate >= MIN_GATE_WIN_RATE and max_dd <= MAX_GATE_DRAWDOWN and version:
                out.append(asset)
        return out
    except Exception:
        return []


def _soft_sample_quality_fallback_assets(required_assets: Optional[tuple[str, ...]] = None) -> list[str]:
    if not _env_truthy("PARTIAL_GATE_ALLOW_SAMPLE_QUALITY_FALLBACK", "1"):
        return []
    assets_req = tuple(required_assets or _required_gate_assets())
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
                (reason.startswith("state_marked_unsafe:") or reason.startswith("meta_marked_unsafe:"))
                and ("sample_quality_fail" in reason)
                and ("wfa_fail" not in reason)
                and ("stress_fail" not in reason)
                and ("risk_of_ruin=" not in reason)
            )
            if not is_sample_only_unsafe:
                continue
            if not bool(st.get("real_backtest", False)):
                continue
            sharpe = float(st.get("sharpe", 0.0) or 0.0)
            win_rate = float(st.get("win_rate", 0.0) or 0.0)
            max_dd_raw = st.get("max_drawdown_pct", 1.0)
            max_dd = float(1.0 if max_dd_raw is None else max_dd_raw)
            version = str(st.get("model_version", "") or "").strip()
            if sharpe >= MIN_GATE_SHARPE and win_rate >= MIN_GATE_WIN_RATE and max_dd <= MAX_GATE_DRAWDOWN and version:
                out.append(asset)
        return out
    except Exception:
        return []


def _model_gate_ready_effective() -> tuple[bool, str, list[str]]:
    required = list(_required_gate_assets())
    gate_ok, gate_reason = _model_gate_ready()
    if gate_ok:
        return True, gate_reason, required

    if not _env_truthy("PARTIAL_GATE_MODE", "1"):
        return False, gate_reason, []

    partial_assets = [a for a in _partial_gate_assets() if a in required]
    if partial_assets:
        blocked = [a for a in required if a not in partial_assets]
        reason = (
            f"partial_gate:{gate_reason}:blocked={','.join(blocked)}"
            if blocked
            else f"partial_gate:{gate_reason}"
        )
        return True, reason, partial_assets

    soft_wfa_assets = [a for a in _soft_wfa_fallback_assets(tuple(required)) if a in required]
    if soft_wfa_assets:
        blocked = [a for a in required if a not in soft_wfa_assets]
        reason = (
            f"partial_wfa_fallback:{gate_reason}:blocked={','.join(blocked)}"
            if blocked
            else f"partial_wfa_fallback:{gate_reason}"
        )
        return True, reason, soft_wfa_assets

    soft_sample_assets = [a for a in _soft_sample_quality_fallback_assets(tuple(required)) if a in required]
    if soft_sample_assets:
        blocked = [a for a in required if a not in soft_sample_assets]
        reason = (
            f"partial_sample_quality_fallback:{gate_reason}:blocked={','.join(blocked)}"
            if blocked
            else f"partial_sample_quality_fallback:{gate_reason}"
        )
        return True, reason, soft_sample_assets

    return False, gate_reason, []


def _models_ready() -> tuple[bool, str]:
    try:
        ready, reason, active_assets = _model_gate_ready_effective()
        if not ready:
            return False, reason

        det = gate_details(required_assets=_required_gate_assets(), allow_legacy_fallback=True)
        amap = det.get("assets", {}) if isinstance(det, dict) else {}

        for asset in active_assets:
            st = amap.get(asset, {}) if isinstance(amap, dict) else {}
            if not isinstance(st, dict):
                return False, f"{asset}:missing_gate_details"

            version = str(st.get("model_version", "") or "").strip()

            model_path = str(st.get("model_path", "") or "")
            if (not model_path or not os.path.exists(model_path)) and version:
                model_path = str(get_artifact_path("models", f"v{version}.pkl"))
            if not model_path or not os.path.exists(model_path):
                return False, f"{asset}:missing_model_payload"

            meta_path = str(st.get("meta_path", "") or "")
            if (not meta_path or not os.path.exists(meta_path)) and version:
                meta_path = str(get_artifact_path("models", f"v{version}.json"))
            if not meta_path or not os.path.exists(meta_path):
                return False, f"{asset}:missing_model_meta"

        return True, reason
    except Exception as exc:
        return False, f"error:{exc}"


# =============================================================================
# Auto-train (behavior preserved)
# =============================================================================
def _auto_train_models_strict() -> bool:
    log_local = logging.getLogger("main")
    try:
        from Backtest.engine import run_institutional_backtest
    except Exception as exc:
        log_local.error("Auto-train(strict) import failed: %s", exc)
        return False

    assets_default = ",".join(_required_gate_assets())
    assets_env = str(os.environ.get("AUTO_TRAIN_ASSETS", assets_default) or assets_default)
    assets = [a.strip().upper() for a in assets_env.split(",") if a.strip()]
    retry_hours = max(0.0, _env_float("AUTO_TRAIN_RETRY_HOURS", 3.0))
    failure_cooldown_hours = max(0.0, _env_float("AUTO_TRAIN_FAILURE_COOLDOWN_HOURS", max(6.0, retry_hours)))
    force_retry_recent_failed_state = _env_truthy("AUTO_TRAIN_FORCE_RETRY_WHEN_RECENT_GATE_FAIL", "0")
    failure_registry_path = get_artifact_path("models", "auto_train_failures.json")
    if not assets:
        assets = list(_required_gate_assets())
    for a in _required_gate_assets():
        if a not in assets:
            assets.append(a)

    passed_assets: list[str] = []
    failed_assets: list[str] = []
    assets_set = set(assets)

    def _load_failure_registry() -> dict[str, dict[str, Any]]:
        try:
            if not failure_registry_path.exists():
                return {}
            with failure_registry_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return {}
            out: dict[str, dict[str, Any]] = {}
            for k, v in raw.items():
                ku = str(k or "").upper().strip()
                if not ku or ku not in assets_set:
                    continue
                if isinstance(v, dict):
                    out[ku] = dict(v)
            return out
        except Exception:
            return {}

    def _save_failure_registry(data: dict[str, dict[str, Any]]) -> None:
        try:
            failure_registry_path.parent.mkdir(parents=True, exist_ok=True)
            clean: dict[str, dict[str, Any]] = {}
            for k, v in data.items():
                ku = str(k or "").upper().strip()
                if not ku or ku not in assets_set or not isinstance(v, dict):
                    continue
                clean[ku] = {
                    "ts": float(v.get("ts", 0.0) or 0.0),
                    "reason": str(v.get("reason", "unknown")),
                    "count": int(v.get("count", 1) or 1),
                }
            with failure_registry_path.open("w", encoding="utf-8") as f:
                json.dump(clean, f, indent=2)
        except Exception as exc:
            log_local.warning("Auto-train(strict) could not persist failure registry: %s", exc)

    failures = _load_failure_registry()

    auto_fast_mode = _env_truthy("AUTO_TRAIN_FAST_MODE", "1")
    fast_defaults: dict[str, str] = {}
    if auto_fast_mode:
        fast_defaults = {
            "INSTITUTIONAL_FAST_MODE": "1",
            "INSTITUTIONAL_REQUIRE_ALPHA_GATE": str(os.environ.get("AUTO_TRAIN_REQUIRE_ALPHA_GATE", "0") or "0"),
            "TRAIN_MAX_BARS": str(os.environ.get("AUTO_TRAIN_MAX_BARS", "20000") or "20000"),
            "INSTITUTIONAL_MAX_ITERS": str(os.environ.get("AUTO_TRAIN_MAX_ITERS", "700") or "700"),
            "INSTITUTIONAL_EARLY_STOPPING_ROUNDS": str(
                os.environ.get("AUTO_TRAIN_EARLY_STOPPING_ROUNDS", "120") or "120"
            ),
            "INSTITUTIONAL_CV_MAX_ITERS": str(os.environ.get("AUTO_TRAIN_CV_MAX_ITERS", "180") or "180"),
            "INSTITUTIONAL_CV_MAX_SAMPLES": str(os.environ.get("AUTO_TRAIN_CV_MAX_SAMPLES", "18000") or "18000"),
            "INSTITUTIONAL_TSCV_FOLDS": str(os.environ.get("AUTO_TRAIN_TSCV_FOLDS", "3") or "3"),
            "INSTITUTIONAL_WFA_WINDOWS": str(os.environ.get("AUTO_TRAIN_WFA_WINDOWS", "3") or "3"),
        }
        log_local.info(
            "Auto-train(strict) FAST_PROFILE | mode=on iters=%s cv_iters=%s bars=%s",
            fast_defaults["INSTITUTIONAL_MAX_ITERS"],
            fast_defaults["INSTITUTIONAL_CV_MAX_ITERS"],
            fast_defaults["TRAIN_MAX_BARS"],
        )

    for asset in assets:
        # Avoid repeated expensive training loops after recent hard failures.
        if failure_cooldown_hours > 0:
            rec = failures.get(asset, {})
            ts = float(rec.get("ts", 0.0) or 0.0) if isinstance(rec, dict) else 0.0
            if ts > 0.0:
                age_hours_fail = (time.time() - ts) / 3600.0
                if age_hours_fail < failure_cooldown_hours:
                    remain_h = max(0.0, failure_cooldown_hours - age_hours_fail)
                    reason_prev = str(rec.get("reason", "unknown")) if isinstance(rec, dict) else "unknown"
                    log_local.warning(
                        "Auto-train(strict) SKIP | asset=%s reason=recent_failure age=%.2fh < %.2fh remain=%.2fh last_reason=%s",
                        asset,
                        age_hours_fail,
                        failure_cooldown_hours,
                        remain_h,
                        reason_prev,
                    )
                    failed_assets.append(asset)
                    continue

        # Backoff guard: don't retrain same asset repeatedly on every restart.
        if retry_hours > 0:
            try:
                st_path = get_artifact_path("models", f"model_state_{asset}.pkl")
                if st_path.exists():
                    age_hours = (time.time() - st_path.stat().st_mtime) / 3600.0
                    if age_hours < retry_hours:
                        gate_ok_now, gate_reason_now = _single_asset_gate_status(asset)
                        if gate_ok_now or (not force_retry_recent_failed_state):
                            log_local.info(
                                "Auto-train(strict) SKIP | asset=%s reason=recent_state age=%.2fh < %.2fh gate_ok=%s gate_reason=%s",
                                asset,
                                age_hours,
                                retry_hours,
                                gate_ok_now,
                                gate_reason_now,
                            )
                            if gate_ok_now:
                                passed_assets.append(asset)
                            else:
                                failed_assets.append(asset)
                            continue
                        log_local.warning(
                            "Auto-train(strict) FORCE_RETRAIN | asset=%s reason=recent_state_but_gate_failed age=%.2fh < %.2fh gate_reason=%s",
                            asset,
                            age_hours,
                            retry_hours,
                            gate_reason_now,
                        )
            except Exception:
                pass

        # Smart retrain: skip assets that already pass gate
        try:
            det = gate_details(required_assets=(asset,), allow_legacy_fallback=True)
            ast = (det.get("assets", {}) or {}).get(asset, {})
            if isinstance(ast, dict) and bool(ast.get("ok", False)):
                log_local.info(
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
            gate_reason_s = str(gate_reason_pre or "")
            wfa_no_evidence = (
                gate_reason_s.startswith(("state_wfa_failed", "meta_wfa_failed"))
                and (":0/0" in gate_reason_s or "unknown" in gate_reason_s)
            )
            anti_overfit_or_contract = gate_reason_s.startswith(
                (
                    "state_anti_overfit_failed",
                    "meta_anti_overfit_failed",
                    "state_contract_missing",
                    "meta_contract_missing",
                )
            )
            force_retrain = bool((not gate_ok_pre) and (wfa_no_evidence or anti_overfit_or_contract))
            force_retrain_reason = gate_reason_s
        except Exception:
            force_retrain = False
            force_retrain_reason = ""

        prev_force_retrain = os.environ.get("BACKTEST_FORCE_RETRAIN")
        if force_retrain:
            os.environ["BACKTEST_FORCE_RETRAIN"] = "1"
            log_local.warning(
                "Auto-train(strict) FORCE_MODEL_RETRAIN | asset=%s reason=%s",
                asset,
                force_retrain_reason or "gate_failed",
            )

        prev_env: dict[str, Optional[str]] = {}
        if auto_fast_mode:
            for k, v in fast_defaults.items():
                prev_env[k] = os.environ.get(k)
                os.environ[k] = v

        try:
            metrics = run_institutional_backtest(asset)
            gate_ok_asset, gate_reason_asset = _single_asset_gate_status(asset)
            if gate_ok_asset:
                passed_assets.append(asset)
                if asset in failures:
                    failures.pop(asset, None)
                log_local.info(
                    "Auto-train(strict) passed | asset=%s sharpe=%.3f win_rate=%.3f max_dd=%.3f gate_reason=%s",
                    asset,
                    float(getattr(metrics, "sharpe_ratio", 0.0) or 0.0),
                    float(getattr(metrics, "win_rate", 0.0) or 0.0),
                    float(getattr(metrics, "max_drawdown_pct", 0.0) or 0.0),
                    gate_reason_asset,
                )
            else:
                failed_assets.append(asset)
                prev_count = int(failures.get(asset, {}).get("count", 0) or 0)
                failures[asset] = {
                    "ts": float(time.time()),
                    "reason": str(gate_reason_asset or "gate_failed"),
                    "count": prev_count + 1,
                }
                log_local.warning(
                    "Auto-train(strict) failed gate | asset=%s reason=%s sharpe=%.3f win_rate=%.3f max_dd=%.3f",
                    asset,
                    gate_reason_asset,
                    float(getattr(metrics, "sharpe_ratio", 0.0) or 0.0),
                    float(getattr(metrics, "win_rate", 0.0) or 0.0),
                    float(getattr(metrics, "max_drawdown_pct", 0.0) or 0.0),
                )
        except Exception as exc:
            log_local.error("Auto-train(strict) failed | asset=%s err=%s", asset, exc)
            failed_assets.append(asset)
            prev_count = int(failures.get(asset, {}).get("count", 0) or 0)
            failures[asset] = {
                "ts": float(time.time()),
                "reason": f"exception:{exc}",
                "count": prev_count + 1,
            }
        finally:
            if force_retrain:
                if prev_force_retrain is None:
                    os.environ.pop("BACKTEST_FORCE_RETRAIN", None)
                else:
                    os.environ["BACKTEST_FORCE_RETRAIN"] = prev_force_retrain
            if auto_fast_mode:
                for k, old_v in prev_env.items():
                    if old_v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = old_v

    _save_failure_registry(failures)

    required_assets = set(_required_gate_assets())
    passed_required_assets = required_assets.issubset(set(passed_assets))
    if failed_assets:
        log_local.error("AUTO_TRAIN_QUALITY_FAILED | failed_assets=%s", ",".join(failed_assets))

    post_ok, post_reason, post_active_assets = _model_gate_ready_effective()
    if post_ok:
        required_now = set(_required_gate_assets())
        active_set = set(post_active_assets)
        if required_now.issubset(active_set):
            return True
        log_local.warning(
            "AUTO_TRAIN_PARTIAL_GATE | active=%s blocked=%s reason=%s",
            ",".join(post_active_assets) if post_active_assets else "-",
            ",".join(sorted(required_now.difference(active_set))) if required_now.difference(active_set) else "-",
            post_reason,
        )
        return True

    if not passed_required_assets:
        missing = sorted(required_assets.difference(set(passed_assets)))
        log_local.error(
            "AUTO_TRAIN_REQUIRED_ASSET_NOT_PASSED | missing=%s passed=%s",
            ",".join(missing) if missing else "-",
            ",".join(sorted(set(passed_assets))) if passed_assets else "-",
        )
    return False


# =============================================================================
# Runtime bootstrap (lazy imports; behavior preserved)
# =============================================================================
def _bootstrap_runtime() -> bool:
    global bot, ADMIN, bot_commands, send_signal_notification, engine, _tg_available

    # already loaded?
    if engine is not None and (bot is not None or not _tg_available):
        return True

    # env preflight
    try:
        from core.config import preflight_env

        ok, missing, msg = preflight_env()
        if not ok:
            missing = list(missing or [])
            missing_exness = any(m in ("EXNESS_LOGIN", "EXNESS_PASSWORD", "EXNESS_SERVER") for m in missing)
            missing_tg = any(m in ("TG_TOKEN", "TG_ADMIN_ID") for m in missing)

            auto_dry = _env_truthy("AUTO_DRY_RUN_ON_MISSING_ENV", "1")
            allow_missing_tg = _env_truthy("ALLOW_MISSING_TG", "1")

            if missing_exness and auto_dry:
                os.environ["DRY_RUN"] = "1"
                log.warning("Missing creds -> auto dry-run enabled")
                ok = True

            if missing_tg and allow_missing_tg:
                _tg_available = False
                log.warning("Telegram credentials missing -> engine-only mode")
                ok = True

            if not ok:
                if msg:
                    print(msg)
                if missing:
                    print(f"Missing env vars: {', '.join(missing)}")
                return False
    except Exception as exc:
        print(f"Config preflight failed: {exc}")
        return False

    # imports
    try:
        from Bot.portfolio_engine import engine as _engine

        if _tg_available:
            token = os.environ.get("TG_TOKEN") or os.environ.get("BOT_TOKEN")
            admin = os.environ.get("TG_ADMIN_ID") or os.environ.get("ADMIN_ID")
            if not token or not admin:
                if _env_truthy("ALLOW_MISSING_TG", "1"):
                    _tg_available = False
                    log.warning("Telegram not configured -> engine-only mode")
                else:
                    raise RuntimeError("Telegram credentials missing")

        if _tg_available:
            from Bot.bot import (
                ADMIN as _admin,
                bot as _bot,
                bot_commands as _bot_commands,
                send_signal_notification as _send_signal_notification,
            )
        else:
            _bot = None
            _admin = 0
            _bot_commands = None
            _send_signal_notification = None

    except Exception as exc:
        log.warning("Runtime import failed (Telegram disabled): %s", exc)
        _tg_available = False
        _engine = None  # type: ignore[assignment]
        _bot = None
        _admin = 0
        _bot_commands = None
        _send_signal_notification = None

    if _engine is None:
        return False

    bot = _bot
    ADMIN = int(_admin or 0)
    bot_commands = _bot_commands
    send_signal_notification = _send_signal_notification
    engine = _engine

    if _env_truthy("DRY_RUN", "0"):
        try:
            engine.dry_run = True
        except Exception:
            pass
        log.info("ENGINE_MODE | dry_run=True")
    else:
        log.info("ENGINE_MODE | dry_run=False")

    log.info("STARTUP_MODE | telegram=%s", "enabled" if _tg_available else "disabled")

    try:
        attach_global_handler_to_loggers(_system_log_handler)
    except Exception:
        pass

    return True


# =============================================================================
# Shutdown
# =============================================================================
class GracefulShutdown:
    """
    Production shutdown:
    - SIGINT/SIGTERM -> stop_event
    - 2nd signal -> hard exit
    """

    def __init__(self) -> None:
        self.stop_event = Event()
        self._received = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame) -> None:
        if self._received:
            log.warning("Сигнали такрорӣ (%s). Баромади маҷбурӣ.", signum)
            raise SystemExit(2)
        self._received = True
        log.info("Сигнали қатъ (%s). Оромона қатъ мекунем...", signum)
        self.stop_event.set()

    def request_stop(self) -> None:
        self.stop_event.set()


# =============================================================================
# Notifier interface
# =============================================================================
class NotifierLike(Protocol):
    def notify(self, message: str) -> None: ...


class Notifier:
    """
    Telegram notifications without blocking trading threads.
    - bounded queue (drop if overload)
    - serialized bot API calls
    - backoff on network down
    - never raises to callers
    """

    def __init__(self, stop_event: Event, *, queue_max: int = 100) -> None:
        self.stop_event = stop_event
        self.q: "Queue[str]" = Queue(maxsize=int(queue_max))
        self._t: Optional[Thread] = None
        self._bot_lock = Lock()
        self._log_rl = RateLimiter(30.0)

    def start(self) -> None:
        if self._t and self._t.is_alive():
            return
        self._t = Thread(target=self._worker, name="notifier", daemon=True)
        self._t.start()

    def notify(self, message: str) -> None:
        msg = str(message)
        try:
            self.q.put_nowait(msg)
        except Full:
            if self._log_rl.allow("notify_drop"):
                log.warning("Notifier queue full -> drop notifications (throttled)")

    def _send_once(self, msg: str) -> bool:
        try:
            if bot is None:
                return False
            with self._bot_lock:
                bot.send_message(ADMIN, msg)
            return True
        except NETWORK_EXC as exc:
            if self._log_rl.allow("notify_net"):
                log.warning("Telegram notify network error (throttled): %s", exc)
            return False
        except Exception as exc:
            log.error("Telegram notify error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _worker(self) -> None:
        backoff = Backoff(base=1.5, factor=2.0, max_delay=60.0)
        pending: Optional[str] = None
        attempt = 0

        while not self.stop_event.is_set():
            try:
                if pending is None:
                    pending = self.q.get(timeout=0.5)
                    attempt = 0
            except Empty:
                continue

            if pending is None:
                continue

            attempt += 1
            if self._send_once(pending):
                pending = None
                continue

            sleep_interruptible(self.stop_event, backoff.delay(attempt))

        # best-effort drain
        try:
            while True:
                _ = self.q.get_nowait()
        except Exception:
            pass


class NullNotifier:
    def notify(self, message: str) -> None:
        try:
            log.info("NOTIFY_DISABLED | %s", message)
        except Exception:
            pass


# =============================================================================
# Log monitor
# =============================================================================
class LogMonitor:
    def __init__(self, stop_event: Event) -> None:
        self.stop_event = stop_event
        self._t: Optional[Thread] = None
        self._log_rl = RateLimiter(600.0)
        self.interval = 300.0
        self.max_mb = 512.0
        self.max_files = 2000

    def start(self) -> None:
        if self._t and self._t.is_alive():
            return
        self._t = Thread(target=self._worker, name="log.monitor", daemon=True)
        self._t.start()

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                total_bytes, file_count = log_dir_stats()
                total_mb = float(total_bytes) / (1024.0 * 1024.0)
                if total_mb > self.max_mb or int(file_count) > self.max_files:
                    if self._log_rl.allow("log_volume"):
                        log.warning(
                            "Log volume high | size=%.1fMB files=%s thresholds=(%.1fMB,%s)",
                            total_mb,
                            file_count,
                            self.max_mb,
                            self.max_files,
                        )
            except Exception as exc:
                if self._log_rl.allow("log_monitor_err"):
                    log.warning("Log monitor error: %s", exc)

            sleep_interruptible(self.stop_event, self.interval)


# =============================================================================
# Retraining (safe)
# =============================================================================
def _backup_model_artifacts(models_dir):
    try:
        if not models_dir.exists():
            return None
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        backup_dir = models_dir / f"_backup_{ts}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for p in models_dir.iterdir():
            if not p.is_file():
                continue
            if p.name.startswith("_backup_"):
                continue
            if p.name == "model_state.pkl" or p.suffix in {".pkl", ".json"}:
                shutil.copy2(p, backup_dir / p.name)
        return backup_dir
    except Exception as exc:
        log.warning("Model backup failed: %s", exc)
        return None


def _restore_model_artifacts(backup_dir, models_dir) -> None:
    if backup_dir is None:
        return
    try:
        for p in backup_dir.iterdir():
            if p.is_file():
                shutil.copy2(p, models_dir / p.name)
    except Exception as exc:
        log.warning("Model restore failed: %s", exc)


def _safe_retrain_models(notifier: NotifierLike) -> bool:
    try:
        from Backtest.engine import run_institutional_backtest
    except Exception as exc:
        log.error("Retraining import failed: %s", exc)
        return False

    models_dir = get_artifact_dir("models")
    backup_dir = _backup_model_artifacts(models_dir)
    required_assets = list(_required_gate_assets())

    try:
        assets_env: str = "XAU,BTC"
        assets = [a.strip().upper() for a in assets_env.split(",") if a.strip()] or list(required_assets)
        for a in required_assets:
            if a not in assets:
                assets.append(a)

        passed_assets: list[str] = []
        failed_assets: list[str] = []

        for asset in assets:
            # Smart retrain: skip already passing assets
            try:
                det = gate_details(required_assets=(asset,), allow_legacy_fallback=True)
                ast = (det.get("assets", {}) or {}).get(asset, {})
                if isinstance(ast, dict) and bool(ast.get("ok", False)):
                    log.info("RETRAIN_SKIP_ALREADY_PASS | asset=%s reason=gate_already_ok", asset)
                    passed_assets.append(asset)
                    continue
            except Exception:
                pass

            log.info("Retraining %s model...", asset)
            metrics = run_institutional_backtest(asset)
            gate_ok_asset, gate_reason_asset = _single_asset_gate_status(asset)
            if gate_ok_asset:
                passed_assets.append(asset)
                log.info(
                    "Retraining gate passed | asset=%s reason=%s sharpe=%.3f win_rate=%.3f max_dd=%.5f",
                    asset,
                    gate_reason_asset,
                    float(getattr(metrics, "sharpe_ratio", 0.0) or 0.0),
                    float(getattr(metrics, "win_rate", 0.0) or 0.0),
                    float(getattr(metrics, "max_drawdown_pct", 0.0) or 0.0),
                )
            else:
                failed_assets.append(asset)
                log.error(
                    "Retraining gate failed | asset=%s reason=%s sharpe=%.3f win_rate=%.3f max_dd=%.5f",
                    asset,
                    gate_reason_asset,
                    float(getattr(metrics, "sharpe_ratio", 0.0) or 0.0),
                    float(getattr(metrics, "win_rate", 0.0) or 0.0),
                    float(getattr(metrics, "max_drawdown_pct", 0.0) or 0.0),
                )

        gate_ok, gate_reason, gate_active_assets = _model_gate_ready_effective()
        if gate_ok:
            required_now = set(required_assets)
            active_set = set(gate_active_assets)
            if not required_now.issubset(active_set):
                log.warning(
                    "RETRAIN_POSTCHECK_PARTIAL | active=%s blocked=%s reason=%s",
                    ",".join(gate_active_assets) if gate_active_assets else "-",
                    ",".join(sorted(required_now.difference(active_set))) if required_now.difference(active_set) else "-",
                    gate_reason,
                )
            return True

        log.error(
            "RETRAIN_POSTCHECK_FAILED | reason=%s | passed_assets=%s failed_assets=%s required_assets=%s",
            gate_reason,
            ",".join(passed_assets) if passed_assets else "-",
            ",".join(failed_assets) if failed_assets else "-",
            ",".join(required_assets),
        )

        # Only restore backup if NO assets passed (preserve partial success)
        if not passed_assets:
            _restore_model_artifacts(backup_dir, models_dir)
        return False

    except Exception as exc:
        log.error("Retraining exception: %s", exc, exc_info=True)
        _restore_model_artifacts(backup_dir, models_dir)
        return False


def _run_retraining_cycle(notifier: NotifierLike, *, reason: str) -> bool:
    log.warning("Retraining cycle started | reason=%s", reason)
    success = False
    try:
        try:
            engine._retraining_mode = True
            log.info("Trading paused during retraining")
        except Exception:
            pass

        retrain_ok = _safe_retrain_models(notifier)
        if retrain_ok:
            log.info("Retraining complete. Resuming trading.")
            notifier.notify("✅ Бозомӯзӣ анҷом ёфт.\nМодели нав бор шуд.")
            try:
                engine.reload_model()
                post_ok, post_reason, post_active_assets = _model_gate_ready_effective()
                if not post_ok:
                    log.error("RETRAIN_POSTCHECK_FAILED | reason=%s", post_reason)
                    notifier.notify(f"⚠️ Санҷиши пас аз бозомӯзӣ нагузашт: {post_reason}")
                    success = False
                else:
                    required_now = set(_required_gate_assets())
                    active_set = set(post_active_assets)
                    if not required_now.issubset(active_set):
                        log.warning(
                            "RETRAIN_POSTCHECK_PARTIAL | active=%s blocked=%s reason=%s",
                            ",".join(post_active_assets) if post_active_assets else "-",
                            ",".join(sorted(required_now.difference(active_set))) if required_now.difference(active_set) else "-",
                            post_reason,
                        )
                    success = True
                try:
                    engine.clear_manual_stop()
                except Exception:
                    pass
            except Exception as exc:
                log.error("Model reload failed: %s", exc)
                log.error("RETRAIN_POSTCHECK_FAILED | reason=model_reload_failed:%s", exc)
                notifier.notify(f"⚠️ Санҷиши пас аз бозомӯзӣ нагузашт: model_reload_failed:{exc}")
                success = False
        else:
            log.warning("Retraining failed. Using old model.")
            notifier.notify("⚠️ Бозомӯзӣ ноком шуд.\nМодели пешина истифода мешавад.")
            success = False
        return bool(success)
    finally:
        try:
            engine._retraining_mode = False
        except Exception:
            pass


# =============================================================================
# Engine supervisor
# =============================================================================
def run_engine_supervisor(stop_event: Event, notifier: NotifierLike) -> None:
    backoff = Backoff(base=2.0, factor=2.0, max_delay=60.0)
    restart_guard = RateLimiter(20.0)
    manual_stop_rl = RateLimiter(60.0)

    started_once = False
    attempt = 0

    from core.model_retrainer import MAX_MODEL_AGE_HOURS, ModelAgeChecker

    RETRAIN_CHECK_INTERVAL = 3600.0
    GATE_RETRAIN_COOLDOWN_SEC = max(30.0, _env_float("GATE_RETRAIN_COOLDOWN_SEC", 300.0))
    retrain_assets = list(_required_gate_assets())
    last_retrain_check = 0.0

    dry_run_mode = bool(getattr(engine, "dry_run", False))
    if dry_run_mode:
        gate_ready_at_start, gate_reason_at_start, gate_active_assets_at_start = True, "dry_run_bypass", list(_required_gate_assets())
    else:
        gate_ready_at_start, gate_reason_at_start, gate_active_assets_at_start = _model_gate_ready_effective()

    last_gate_retrain_attempt = 0.0 if gate_ready_at_start else time.time()
    retraining_in_progress = False
    gate_block_rl = RateLimiter(30.0)

    # Preserve original state file path (as in your code)
    model_checker = ModelAgeChecker(get_artifact_path("models", "model_state.pkl"))

    notifier.notify("🧠 Нозири мотор оғоз шуд.")
    if not gate_ready_at_start:
        log.warning(
            "Engine gate initially blocked | reason=%s | delaying next retrain by %.0fs",
            gate_reason_at_start,
            GATE_RETRAIN_COOLDOWN_SEC,
        )
    else:
        required_now = set(_required_gate_assets())
        active_now = set(gate_active_assets_at_start)
        if not required_now.issubset(active_now):
            log.warning(
                "Engine gate started in partial mode | active=%s blocked=%s reason=%s",
                ",".join(gate_active_assets_at_start) if gate_active_assets_at_start else "-",
                ",".join(sorted(required_now.difference(active_now))) if required_now.difference(active_now) else "-",
                gate_reason_at_start,
            )

    while not stop_event.is_set():
        # Cheap status probe first
        ok_connected = True
        ok_trading = True
        manual_stop = False
        try:
            st = engine.status()
            ok_connected = bool(getattr(st, "connected", True))
            ok_trading = bool(getattr(st, "trading", True))
            manual_stop = bool(getattr(st, "manual_stop", False))
        except Exception:
            pass

        # Hourly retraining check (only if not dry-run)
        if not retraining_in_progress and not dry_run_mode:
            now = time.time()
            if (now - last_retrain_check) > RETRAIN_CHECK_INTERVAL:
                last_retrain_check = now
                if model_checker.needs_retraining(retrain_assets):
                    age = model_checker.get_model_age_hours()
                    age_txt = f"{age:.1f}h" if age is not None else "unknown"
                    log.warning(
                        "Model expired (age: %s). Starting retraining... thresholds=%s",
                        age_txt,
                        MAX_MODEL_AGE_HOURS,
                    )
                    notifier.notify(f"🔄 Бозомӯзии модел оғоз шуд.\nСинни модел: {age_txt}")
                    retraining_in_progress = True
                    try:
                        _run_retraining_cycle(notifier, reason=f"age_expired:{age_txt}")
                    finally:
                        retraining_in_progress = False

        # Re-probe
        try:
            st = engine.status()
            ok_connected = bool(getattr(st, "connected", True))
            ok_trading = bool(getattr(st, "trading", True))
            manual_stop = bool(getattr(st, "manual_stop", False))
        except Exception:
            pass

        # Gate blocked -> controlled retraining (avoid start spam)
        if not ok_trading and not retraining_in_progress and not dry_run_mode:
            gate_ok, gate_reason, _active_assets = _model_gate_ready_effective()
            if not gate_ok:
                now = time.time()
                elapsed = now - last_gate_retrain_attempt
                if elapsed >= GATE_RETRAIN_COOLDOWN_SEC:
                    last_gate_retrain_attempt = now
                    retraining_in_progress = True
                    try:
                        log.warning("Engine gate blocked | reason=%s | triggering retrain", gate_reason)
                        notifier.notify(f"⛔ Дарвозаи модел баста аст: {gate_reason}\n🔁 Бозомӯзӣ оғоз мешавад...")
                        _run_retraining_cycle(notifier, reason=f"gate_blocked:{gate_reason}")
                    finally:
                        retraining_in_progress = False
                else:
                    remain = max(1.0, GATE_RETRAIN_COOLDOWN_SEC - elapsed)
                    if gate_block_rl.allow("engine_gate_blocked_wait"):
                        log.warning("Engine gate blocked | reason=%s | next retrain in %.0fs", gate_reason, remain)
                    sleep_interruptible(stop_event, min(5.0, remain))
                continue

        if manual_stop:
            if manual_stop_rl.allow("engine_manual_stop"):
                log.info("Engine idle (manual stop active); supervisor waiting")
            sleep_interruptible(stop_event, 1.0)
            continue

        # Ensure engine running
        if not ok_trading:
            try:
                attempt += 1
                started = bool(engine.start())
                if not started:
                    try:
                        st_after = engine.status()
                        if bool(getattr(st_after, "manual_stop", False)):
                            attempt = 0
                            if manual_stop_rl.allow("engine_start_blocked_manual"):
                                log.warning("Engine start blocked; switched to monitoring/manual-stop mode")
                            sleep_interruptible(stop_event, 1.0)
                            continue
                    except Exception:
                        pass
                    raise RuntimeError("engine.start returned False")

                if not started_once:
                    started_once = True
                    notifier.notify("🟢 Мотори тиҷорат оғоз шуд.")
                attempt = 0

            except Exception as exc:
                delay = backoff.delay(attempt)
                if attempt == 1 or attempt % 5 == 0:
                    log.error("Engine start failed: %s | retry in %.1fs", exc, delay)
                    notifier.notify(f"⚠️ Оғози мотор ноком шуд: {exc}\n⏳ Кӯшиши навбатӣ пас аз {delay:.0f}с")
                sleep_interruptible(stop_event, delay)
                continue

        # Connected health
        if not ok_connected:
            if restart_guard.allow("engine_unhealthy"):
                log.warning("Engine unhealthy (connected=%s trading=%s) -> waiting/recovering", ok_connected, ok_trading)
                if TG_HEALTH_NOTIFY:
                    notifier.notify("🟠 Ҳолати мотор ноустувор аст, интизори барқароршавӣ...")
            sleep_interruptible(stop_event, 2.0)
            continue

        sleep_interruptible(stop_event, 1.0)

    # Shutdown
    try:
        engine.stop()
    except Exception as exc:
        log.error("Engine stop error: %s | tb=%s", exc, traceback.format_exc())
    notifier.notify("🔴 Мотори тиҷорат қатъ шуд.")


# =============================================================================
# Telegram supervisor
# =============================================================================
def run_telegram_supervisor(stop_event: Event, notifier: NotifierLike) -> None:
    try:
        if callable(bot_commands):
            bot_commands()
    except NETWORK_EXC as exc:
        log.warning("bot_commands network error: %s (ignored)", exc)
    except Exception as exc:
        log.error("bot_commands error: %s | tb=%s", exc, traceback.format_exc())

    notifier.notify("🚀 Боти Telegram оғоз шуд.")

    backoff = Backoff(base=1.0, factor=2.0, max_delay=60.0)
    log_rl = RateLimiter(20.0)
    attempt = 0

    while not stop_event.is_set():
        try:
            attempt += 1
            log_super.info("Starting Telegram bot polling (attempt %s)...", attempt)
            if bot is None:
                sleep_interruptible(stop_event, 1.0)
                continue

            bot.infinity_polling(
                timeout=75,
                long_polling_timeout=75,
                restart_on_change=False,
                skip_pending=True,
            )
            attempt = 0

        except _NET_ERRS_TG as exc:
            delay = backoff.delay(attempt)
            if log_rl.allow("tg_net"):
                log_super.warning("Telegram network unstable: %s | retry in %.1fs", exc, delay)
            sleep_interruptible(stop_event, delay)

        except Exception as exc:
            delay = backoff.delay(attempt)
            log.error("Telegram polling error: %s | retry in %.1fs | tb=%s", exc, delay, traceback.format_exc())
            sleep_interruptible(stop_event, delay)

    try:
        if bot is not None:
            bot.stop_polling()
    except Exception:
        pass

    notifier.notify("⏹️ Боти Telegram қатъ шуд.")


# =============================================================================
# Engine notify worker
# =============================================================================
def run_engine_notify_worker(stop_event: Event) -> None:
    from Bot.bot_utils import get_notify_queue

    q = get_notify_queue()
    backoff = Backoff(base=1.0, factor=2.0, max_delay=30.0)
    log_rl = RateLimiter(30.0)

    pending = None
    attempt = 0

    while not stop_event.is_set():
        try:
            if pending is None:
                pending = q.get(timeout=0.5)
                attempt = 0
        except Empty:
            continue

        if pending is None:
            continue

        chat_id, msg = pending
        attempt += 1

        try:
            if bot is None:
                pending = None
                continue
            bot.send_message(chat_id, msg, parse_mode="HTML")
            pending = None
        except NETWORK_EXC as exc:
            if log_rl.allow("notify_net"):
                log.warning("Engine notify network error: %s", exc)
            sleep_interruptible(stop_event, backoff.delay(attempt))
        except Exception as exc:
            log.error("Engine notify error: %s", exc)
            pending = None

    # best-effort drain
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass


# =============================================================================
# Main
# =============================================================================
def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAUUSDm Scalping System (Exness MT5) - Production Runner")
    p.add_argument("--headless", action="store_true", help="Оғоз бе Telegram (ҳолати VPS)")
    p.add_argument("--engine-only", action="store_true", help="Фақат мотор, бе бот")
    p.add_argument("--dry-run", action="store_true", help="Run in simulation mode (Mock MT5). Default is LIVE.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    try:
        with SingletonInstance(port=12345):
            return _main_inner(argv)
    except RuntimeError as e:
        print(f"FATAL: {e}")
        return 1
    except Exception as e:
        print(f"FATAL: Singleton check failed: {e}")
        return 1


def _main_inner(argv: Optional[list[str]] = None) -> int:
    if not _bootstrap_runtime():
        return 1

    shutdown = GracefulShutdown()
    args = _parse_args(argv)

    # Default to LIVE (Production)
    if args.dry_run:
        try:
            engine.dry_run = True
        except Exception:
            pass
        try:
            engine._check_model_health()
        except Exception:
            pass
    else:
        # Preflight: ensure trained models exist (auto-train if missing)
        ready, reason = _models_ready()
        if not ready:
            log.warning("MODELS_MISSING | reason=%s", reason)
            if _env_truthy("AUTO_TRAIN_ON_STARTUP", "1"):
                ok = _auto_train_models_strict()
                if ok:
                    log.info("Auto-training completed")
                else:
                    log.error("Auto-training failed; continuing startup")
            else:
                log.warning("AUTO_TRAIN_ON_STARTUP=0 | skipping startup retraining")

            try:
                engine._check_model_health()
            except Exception:
                pass
    if args.dry_run:
        log.info("DRY_RUN_STARTUP | skipping strict model preflight")

    # Print System Matrix (Quantum Status)
    try:
        engine.print_startup_matrix()
    except Exception:
        pass

    # Wiring: Engine -> Bot
    if send_signal_notification is not None:
        try:
            engine.set_signal_notifier(send_signal_notification)
        except Exception:
            pass
    log.info("SIGNAL_NOTIFIER_WIRED | Engine -> Bot connected")

    if _tg_available:
        notifier: NotifierLike = Notifier(shutdown.stop_event, queue_max=200)
        try:
            notifier.start()  # type: ignore[attr-defined]
        except Exception:
            pass
    else:
        notifier = NullNotifier()

    LogMonitor(shutdown.stop_event).start()

    notify_worker_thread: Optional[Thread] = None
    if _tg_available:
        notify_worker_thread = Thread(
            target=run_engine_notify_worker,
            args=(shutdown.stop_event,),
            name="engine.notify_worker",
            daemon=True,
        )
        notify_worker_thread.start()

    engine_thread = Thread(
        target=run_engine_supervisor,
        args=(shutdown.stop_event, notifier),
        name="engine.supervisor",
        daemon=False,
    )

    bot_thread: Optional[Thread] = None

    try:
        engine_thread.start()

        if _tg_available and not (args.headless or args.engine_only):
            bot_thread = Thread(
                target=run_telegram_supervisor,
                args=(shutdown.stop_event, notifier),
                name="telegram.supervisor",
                daemon=False,
            )
            bot_thread.start()
        elif not _tg_available:
            log.info("Telegram disabled -> running engine-only")

        while not shutdown.stop_event.is_set():
            shutdown.stop_event.wait(timeout=1.0)

        return 0

    except KeyboardInterrupt:
        log.info("Ctrl+C -> stop")
        shutdown.request_stop()
        return 0

    except Exception as exc:
        log.error("Fatal main error: %s | tb=%s", exc, traceback.format_exc())
        notifier.notify(f"🛑 Хатои ҷиддии система: {exc}")
        shutdown.request_stop()
        return 1

    finally:
        shutdown.request_stop()

        try:
            if bot is not None:
                bot.stop_polling()
        except Exception:
            pass

        try:
            if bot_thread and bot_thread.is_alive():
                bot_thread.join(timeout=20.0)
        except Exception:
            pass

        try:
            if notify_worker_thread and notify_worker_thread.is_alive():
                notify_worker_thread.join(timeout=5.0)
        except Exception:
            pass

        try:
            if engine_thread.is_alive():
                engine_thread.join(timeout=30.0)
        except Exception:
            pass

        notifier.notify("⏹️ Система қатъ шуд.")
        sleep_interruptible(shutdown.stop_event, 0.2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
