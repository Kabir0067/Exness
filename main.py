from __future__ import annotations

import argparse
import http.client
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
from queue import Queue, Full, Empty
import threading
from threading import Event, Lock, Thread
from typing import Any, Callable, Optional

import urllib3

TG_HEALTH_NOTIFY = False

from log_config import (
    LOG_DIR as LOG_ROOT,
    get_log_path,
    log_dir_stats,
    configure_logging,
    attach_global_handler_to_loggers,
    get_artifact_dir,
    get_artifact_path,
)
from core.config import MAX_GATE_DRAWDOWN, MIN_GATE_SHARPE, MIN_GATE_WIN_RATE
from core.model_gate import DEFAULT_REQUIRED_ASSETS, gate_details

bot: Any = None
ADMIN: int = 0
bot_commands: Optional[Callable[[], None]] = None
send_signal_notification: Optional[Callable[[str, Any], None]] = None
engine: Any = None
_tg_available: bool = True
# Disable global aggregate file log. Each module writes to its own log file.
_system_log_handler = configure_logging(level="INFO", system_log_name=None, console=True)
_GATE_STATUS_LOG_TTL_SEC: float = float(os.getenv("GATE_STATUS_LOG_TTL_SEC", "60") or "60")
_gate_status_log_lock = Lock()
_gate_status_last_sig: str = ""
_gate_status_last_ts: float = 0.0


def _env_truthy(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name, default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _required_gate_assets() -> tuple[str, ...]:
    raw = str(os.getenv("GATE_REQUIRED_ASSETS", ",".join(DEFAULT_REQUIRED_ASSETS)) or ",".join(DEFAULT_REQUIRED_ASSETS))
    out: list[str] = []
    seen = set()
    for item in raw.split(","):
        a = str(item or "").upper().strip()
        if not a or a in seen:
            continue
        seen.add(a)
        out.append(a)
    if not out:
        return tuple(DEFAULT_REQUIRED_ASSETS)
    return tuple(out)


def _model_gate_ready() -> tuple[bool, str]:
    details = gate_details(required_assets=_required_gate_assets(), allow_legacy_fallback=True)
    assets = details.get("assets", {}) if isinstance(details, dict) else {}
    items = []
    if isinstance(assets, dict):
        for asset, st in sorted(assets.items(), key=lambda kv: str(kv[0])):
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

    sig = "|".join(
        f"{a}:{int(ok)}:{r}:{v}:{s:.3f}:{w:.3f}:{d:.3f}:{int(lg)}"
        for a, ok, r, v, s, w, d, lg in items
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
    return bool(details.get("ok", False)), str(details.get("reason", "unknown"))


def _single_asset_gate_status(asset: str) -> tuple[bool, str]:
    asset_u = str(asset or "").upper().strip()
    details = gate_details(required_assets=(asset_u,), allow_legacy_fallback=True)
    st = (details.get("assets", {}) or {}).get(asset_u, {})
    if isinstance(st, dict):
        return bool(st.get("ok", False)), str(st.get("reason", details.get("reason", "unknown")))
    return bool(details.get("ok", False)), str(details.get("reason", "unknown"))


def _partial_gate_assets() -> list[str]:
    """Return list of assets that pass gate individually (partial mode)."""
    passing: list[str] = []
    for asset in _required_gate_assets():
        try:
            details = gate_details(required_assets=(asset,), allow_legacy_fallback=True)
            asset_st = (details.get("assets", {}) or {}).get(asset, {})
            if isinstance(asset_st, dict) and bool(asset_st.get("ok", False)):
                passing.append(asset)
        except Exception:
            continue
    return passing


def _soft_wfa_fallback_assets(required_assets: Optional[tuple[str, ...]] = None) -> list[str]:
    """
    Return assets that only fail WFA but satisfy base safety metrics.
    Used as last-resort degraded mode when partial gate has zero fully passing assets.
    """
    if not _env_truthy("PARTIAL_GATE_ALLOW_WFA_FALLBACK", "1"):
        return []
    assets_req = tuple(required_assets or _required_gate_assets())
    if not assets_req:
        return []
    try:
        details = gate_details(required_assets=assets_req, allow_legacy_fallback=True)
        asset_map = details.get("assets", {}) if isinstance(details, dict) else {}
        out: list[str] = []
        for asset in assets_req:
            st = asset_map.get(asset, {}) if isinstance(asset_map, dict) else {}
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
            if (
                sharpe >= MIN_GATE_SHARPE
                and win_rate >= MIN_GATE_WIN_RATE
                and max_dd <= MAX_GATE_DRAWDOWN
                and version
            ):
                out.append(asset)
        return out
    except Exception:
        return []


def _soft_sample_quality_fallback_assets(required_assets: Optional[tuple[str, ...]] = None) -> list[str]:
    """
    Return assets that only fail sample-quality checks while base risk metrics remain strong.
    This is a degraded-mode fallback for operational continuity.
    """
    if not _env_truthy("PARTIAL_GATE_ALLOW_SAMPLE_QUALITY_FALLBACK", "1"):
        return []
    assets_req = tuple(required_assets or _required_gate_assets())
    if not assets_req:
        return []
    try:
        details = gate_details(required_assets=assets_req, allow_legacy_fallback=True)
        asset_map = details.get("assets", {}) if isinstance(details, dict) else {}
        out: list[str] = []
        for asset in assets_req:
            st = asset_map.get(asset, {}) if isinstance(asset_map, dict) else {}
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
            if (
                sharpe >= MIN_GATE_SHARPE
                and win_rate >= MIN_GATE_WIN_RATE
                and max_dd <= MAX_GATE_DRAWDOWN
                and version
            ):
                out.append(asset)
        return out
    except Exception:
        return []


def _model_gate_ready_effective() -> tuple[bool, str, list[str]]:
    """
    Effective readiness policy:
    1) strict gate pass
    2) partial pass (PARTIAL_GATE_MODE=1)
    3) soft WFA fallback (PARTIAL_GATE_ALLOW_WFA_FALLBACK=1)
    """
    gate_ok, gate_reason = _model_gate_ready()
    required_assets = list(_required_gate_assets())
    if gate_ok:
        return True, gate_reason, required_assets

    if not _env_truthy("PARTIAL_GATE_MODE", "1"):
        return False, gate_reason, []

    passing = _partial_gate_assets()
    if passing:
        blocked = sorted(set(required_assets).difference(set(passing)))
        reason = (
            f"partial_ok:passing={','.join(passing)}"
            f" blocked={','.join(blocked) if blocked else '-'}"
            f" base_reason={gate_reason}"
        )
        return True, reason, passing

    soft_wfa_assets = _soft_wfa_fallback_assets(tuple(required_assets))
    soft_sample_assets = _soft_sample_quality_fallback_assets(tuple(required_assets))
    soft_assets = sorted(set(soft_wfa_assets).union(set(soft_sample_assets)))
    if soft_assets:
        blocked = sorted(set(required_assets).difference(set(soft_assets)))
        mode_parts: list[str] = []
        if soft_wfa_assets:
            mode_parts.append(f"wfa={','.join(soft_wfa_assets)}")
        if soft_sample_assets:
            mode_parts.append(f"sample_quality={','.join(soft_sample_assets)}")
        mode_tag = ";".join(mode_parts) if mode_parts else "mixed"
        reason = (
            f"soft_fallback[{mode_tag}]:passing={','.join(soft_assets)}"
            f" blocked={','.join(blocked) if blocked else '-'}"
            f" base_reason={gate_reason}"
        )
        return True, reason, soft_assets

    return False, gate_reason, []


def _models_ready() -> tuple[bool, str]:
    try:
        ready, reason, active_assets = _model_gate_ready_effective()
        if not ready:
            return False, reason

        details = gate_details(required_assets=_required_gate_assets(), allow_legacy_fallback=True)
        assets = details.get("assets", {}) if isinstance(details, dict) else {}
        for asset in active_assets:
            st = assets.get(asset, {}) if isinstance(assets, dict) else {}
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


def _auto_train_models() -> bool:
    log_local = logging.getLogger("main")
    log_local.warning("Models not found. Starting automatic initial training...")

    # Write directly to REAL console (bypass _StdToLogger wrapper)
    _real_out = getattr(sys, "__stdout__", None) or sys.stdout

    def _con(msg: str) -> None:
        try:
            _real_out.write(msg + "\n")
            _real_out.flush()
        except Exception:
            pass

    _con("\n" + "=" * 60)
    _con("🔄 Auto-training оғоз шуд — метавонад чанд дақиқа гирад.")
    _con("   You will see iteration-by-iteration progress below.")
    _con("=" * 60 + "\n")

    try:
        from Backtest.engine import run_institutional_backtest
    except Exception as exc:
        log_local.error("Auto-train import failed: %s", exc)
        return False

    ok = True
    for asset in ("XAU", "BTC"):
        # Skip assets that already pass gate AND have fresh models (< 24h)
        try:
            from core.model_gate import gate_details as _gate_check
            _det = _gate_check(required_assets=(asset,), allow_legacy_fallback=True)
            _ast = (_det.get("assets", {}) or {}).get(asset, {})
            if isinstance(_ast, dict) and bool(_ast.get("ok", False)):
                _sp = str(_ast.get("state_path", ""))
                age_hours = 0.0
                if _sp:
                    import pathlib as _pl
                    _p = _pl.Path(_sp)
                    if _p.exists():
                        age_hours = (time.time() - _p.stat().st_mtime) / 3600.0
                if age_hours < 24.0:
                    log_local.info("Auto-train SKIP | asset=%s reason=gate_ok+fresh (age=%.1fh < 24h)", asset, age_hours)
                    continue
        except Exception:
            pass
        
        try:
            run_institutional_backtest(asset)
        except Exception as exc:
            log_local.error("Auto-train failed | asset=%s err=%s", asset, exc)
            ok = False

    if ok:
        artifacts_dir = str(get_artifact_dir("models"))
        _con("\n" + "=" * 60)
        _con(f"✅ Training анҷом ёфт. Модел дар ин роҳ сабт шуд: {artifacts_dir}")
        _con("   Starting Trading Engine...")
        _con("=" * 60 + "\n")
    else:
        _con("\n⚠️ Training бо хато анҷом ёфт. Логҳоро санҷед.")
    return ok


def _auto_train_models_strict() -> bool:
    """
    Strict auto-train:
    - Retrain all required assets (default strict: XAU + BTC)
    - Success only after post-training gate check passes
    """
    log_local = logging.getLogger("main")
    try:
        from Backtest.engine import run_institutional_backtest
    except Exception as exc:
        log_local.error("Auto-train(strict) import failed: %s", exc)
        return False

    assets_default = ",".join(_required_gate_assets())
    assets_env = str(os.getenv("AUTO_TRAIN_ASSETS", assets_default) or assets_default)
    assets = [a.strip().upper() for a in assets_env.split(",") if a.strip()]
    retry_hours = float(os.getenv("AUTO_TRAIN_RETRY_HOURS", "3") or "3")
    if not assets:
        assets = list(_required_gate_assets())
    for asset in _required_gate_assets():
        if asset not in assets:
            assets.append(asset)

    passed_assets: list[str] = []
    failed_assets: list[str] = []
    for asset in assets:
        # Backoff guard: don't retrain same asset repeatedly on every restart.
        if retry_hours > 0:
            try:
                st_path = get_artifact_path("models", f"model_state_{asset}.pkl")
                if st_path.exists():
                    age_hours = (time.time() - st_path.stat().st_mtime) / 3600.0
                    if age_hours < retry_hours:
                        gate_ok_now, gate_reason_now = _single_asset_gate_status(asset)
                        if gate_ok_now:
                            log_local.info(
                                "Auto-train(strict) SKIP | asset=%s reason=recent_state age=%.2fh < %.2fh gate_ok=%s gate_reason=%s",
                                asset,
                                age_hours,
                                retry_hours,
                                gate_ok_now,
                                gate_reason_now,
                            )
                            passed_assets.append(asset)
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
            details = gate_details(required_assets=(asset,), allow_legacy_fallback=True)
            asset_st = (details.get("assets", {}) or {}).get(asset, {})
            if isinstance(asset_st, dict) and bool(asset_st.get("ok", False)):
                log_local.info(
                    "Auto-train(strict) SKIP | asset=%s reason=gate_already_ok sharpe=%.3f",
                    asset, float(asset_st.get("sharpe", 0.0) or 0.0),
                )
                passed_assets.append(asset)
                continue
        except Exception:
            pass

        try:
            metrics = run_institutional_backtest(asset)
            gate_ok_asset, gate_reason_asset = _single_asset_gate_status(asset)
            if gate_ok_asset:
                passed_assets.append(asset)
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
            continue

    required_assets = set(_required_gate_assets())
    passed_required_assets = required_assets.issubset(set(passed_assets))
    if failed_assets:
        log_local.error(
            "AUTO_TRAIN_QUALITY_FAILED | failed_assets=%s",
            ",".join(failed_assets),
        )

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


def _setup_exception_hooks() -> None:
    def _handle_exception(exc_type, exc, tb) -> None:
        try:
            tb_txt = "".join(traceback.format_tb(tb)) if tb else ""
            logging.getLogger("main").error(
                "UNCAUGHT_EXCEPTION | %s | tb=%s",
                exc,
                tb_txt,
            )
        finally:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _handle_exception

    # Thread exceptions (Py3.8+)
    if hasattr(threading, "excepthook"):
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
        threading.excepthook = _thread_excepthook


class _StdToLogger:
    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level

    def write(self, buf: str) -> None:
        for line in str(buf).rstrip().splitlines():
            if line:
                self._logger.log(self._level, line)

    def flush(self) -> None:
        return


_setup_exception_hooks()
sys.stdout = _StdToLogger(logging.getLogger("stdout"), logging.INFO)
sys.stderr = _StdToLogger(logging.getLogger("stderr"), logging.ERROR)


def _bootstrap_runtime() -> bool:
    """
    Load env+runtime deps lazily to avoid import-time crashes on missing .env.
    """
    global bot, ADMIN, bot_commands, send_signal_notification, engine, _tg_available

    if bot is not None and engine is not None:
        return True

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

    try:
        from Bot.portfolio_engine import engine as _engine

        # Decide if TG is available (token + admin id)
        if _tg_available:
            token = os.environ.get("TG_TOKEN") or os.environ.get("BOT_TOKEN")
            admin = os.environ.get("TG_ADMIN_ID") or os.environ.get("ADMIN_ID")
            if not token or not admin:
                allow_missing_tg = _env_truthy("ALLOW_MISSING_TG", "1")
                if allow_missing_tg:
                    _tg_available = False
                    log.warning("Telegram not configured -> engine-only mode")
                else:
                    raise RuntimeError("Telegram credentials missing")

        if _tg_available:
            from Bot.bot import (
                bot as _bot,
                ADMIN as _admin,
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
        _bot = None
        _admin = 0
        _bot_commands = None
        _send_signal_notification = None

    bot = _bot
    ADMIN = int(_admin or 0)
    bot_commands = _bot_commands
    send_signal_notification = _send_signal_notification
    engine = _engine

    # Enforce dry-run on engine if requested
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

# ==========================
# Logging (production safe)
# ==========================
LOG_DIR = LOG_ROOT

log = logging.getLogger("main")
log.setLevel(logging.INFO)
log.propagate = False

if not log.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = RotatingFileHandler(
        str(get_log_path("main.log")),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler(getattr(sys, "__stdout__", sys.stdout))
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    log.addHandler(fh)
    log.addHandler(ch)

log_super = logging.getLogger("telegram.supervisor")
log_super.setLevel(logging.INFO)
log_super.propagate = False

if not log_super.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = RotatingFileHandler(
        str(get_log_path("telegram.log")),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
        delay=True,
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler(getattr(sys, "__stdout__", sys.stdout))
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    log_super.addHandler(fh)
    log_super.addHandler(ch)


def sleep_interruptible(stop_event: Event, seconds: float) -> None:
    """Sleep, but exit early if stop_event set."""
    end = time.monotonic() + float(seconds)
    while not stop_event.is_set():
        left = end - time.monotonic()
        if left <= 0:
            return
        stop_event.wait(timeout=min(0.5, left))



class SingletonInstance:
    """
    Ensure only one instance of the bot runs by binding a TCP socket.
    If the port is already in use, we assume another instance is running and exit.
    """
    def __init__(self, port: int = 12345):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._port = int(port)
        self._lock_success = False

    def __enter__(self):
        try:
            # Bind to localhost on the specific port
            self._socket.bind(("127.0.0.1", self._port))
            self._lock_success = True
            return self
        except socket.error:
            # Port is already in use
            self._lock_success = False
            raise RuntimeError(f"Another instance is already running on port {self._port}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._lock_success:
            try:
                self._socket.close()
            except Exception:
                pass


@dataclass(frozen=True)
class Backoff:
    base: float = 1.0
    factor: float = 2.0
    max_delay: float = 60.0

    def delay(self, attempt: int) -> float:
        if attempt <= 1:
            return min(self.max_delay, self.base)
        try:
            return min(self.max_delay, self.base * (self.factor ** (attempt - 1)))
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


# Network-ish exceptions (Telegram / HTTP / DNS / sockets)
try:
    import requests  # type: ignore
    from requests.exceptions import (  # type: ignore
        RequestException,
        ReadTimeout,
        ConnectTimeout,
        ConnectionError as RequestsConnectionError,
        ChunkedEncodingError,
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


# ==========================
# Graceful shutdown
# ==========================
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


# ==========================
# Notification Dispatcher (non-blocking)
# ==========================
class Notifier:
    """
    Sends Telegram notifications without blocking trading threads.
    - Bounded queue (drop if overload)
    - Serialized bot API calls
    - Backoff on network down
    - Never raises to callers
    """

    def __init__(self, stop_event: Event, *, queue_max: int = 100) -> None:
        self.stop_event = stop_event
        self.q: "Queue[str]" = Queue(maxsize=int(queue_max))
        self._t: Optional[Thread] = None
        self._bot_lock = Lock()
        self._log_rl = RateLimiter(30.0)  # prevent log spam on internet down

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
            ok = self._send_once(pending)
            if ok:
                pending = None
                continue

            delay = backoff.delay(attempt)
            sleep_interruptible(self.stop_event, delay)

        # best-effort drain (no blocking)
        try:
            while True:
                _ = self.q.get_nowait()
        except Exception:
            pass


# ==========================
# Null Notifier (engine-only)
# ==========================
class NullNotifier:
    """No-op notifier for engine-only mode."""
    def notify(self, message: str) -> None:
        try:
            log.info("NOTIFY_DISABLED | %s", message)
        except Exception:
            pass

# ==========================
# Log volume monitor
# ==========================
class LogMonitor:
    """
    Periodically checks log directory size and file count.
    Emits warnings when thresholds are exceeded.
    """

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


# ==========================
# Engine Supervisor (never crash)
# ==========================
def _run_retraining_cycle(notifier: Notifier, *, reason: str) -> bool:
    """Run one retraining cycle and hot-reload on success."""
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


def run_engine_supervisor(stop_event: Event, notifier: Notifier) -> None:
    """
    High-reliability engine supervisor:
    - Start engine with backoff if MT5 not ready
    - Monitor loop; if engine stops/fails -> guarded restart
    - Never raises; never restart-storms
    """
    backoff = Backoff(base=2.0, factor=2.0, max_delay=60.0)
    restart_guard = RateLimiter(20.0)
    manual_stop_rl = RateLimiter(60.0)

    started_once = False
    attempt = 0
    from core.model_retrainer import ModelAgeChecker, MAX_MODEL_AGE_HOURS

    RETRAIN_CHECK_INTERVAL = 3600.0  # 1 hour
    GATE_RETRAIN_COOLDOWN_SEC = float(os.getenv("GATE_RETRAIN_COOLDOWN_SEC", "300") or "300")
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
        # Cheap status probe first (avoid engine.start() spam)
        ok_connected = True
        ok_trading = True
        manual_stop = False
        try:
            st = engine.status()
            ok_connected = bool(getattr(st, "connected", True))
            ok_trading = bool(getattr(st, "trading", True))
            manual_stop = bool(getattr(st, "manual_stop", False))
        except Exception:
            ok_connected = True
            ok_trading = True
            manual_stop = False

        # HOURLY RETRAINING CHECK
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

        # Re-probe state after potential retraining/reload.
        try:
            st = engine.status()
            ok_connected = bool(getattr(st, "connected", True))
            ok_trading = bool(getattr(st, "trading", True))
            manual_stop = bool(getattr(st, "manual_stop", False))
        except Exception:
            ok_connected = True
            ok_trading = True
            manual_stop = False

        # If gate is blocked, avoid engine.start() spam and trigger controlled retraining.
        if not ok_trading and not retraining_in_progress and not dry_run_mode:
            gate_ok, gate_reason, _gate_active_assets = _model_gate_ready_effective()
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
                        log.warning(
                            "Engine gate blocked | reason=%s | next retrain in %.0fs",
                            gate_reason,
                            remain,
                        )
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

        # Monitor connected health
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


def _safe_retrain_models(notifier: Notifier) -> bool:
    """
    Safely retrain models with exception handling.
    Returns True if successful, False otherwise.
    """
    try:
        from Backtest.engine import run_institutional_backtest
    except Exception as exc:
        log.error("Retraining import failed: %s", exc)
        return False

    models_dir = get_artifact_dir("models")
    backup_dir = _backup_model_artifacts(models_dir)

    required_assets = list(_required_gate_assets())

    try:
        assets_env = str(os.getenv("RETRAIN_ASSETS", "XAU,BTC") or "XAU,BTC")
        assets = [a.strip().upper() for a in assets_env.split(",") if a.strip()]
        if not assets:
            assets = list(required_assets)
        for a in required_assets:
            if a not in assets:
                assets.append(a)

        passed_assets: list[str] = []
        failed_assets: list[str] = []

        for asset in assets:
            # Bug #17: Smart retrain — skip already-passing assets
            details = gate_details(required_assets=(asset,), allow_legacy_fallback=True)
            asset_status = (details.get("assets", {}) or {}).get(asset, {})
            if isinstance(asset_status, dict) and bool(asset_status.get("ok", False)):
                log.info("RETRAIN_SKIP_ALREADY_PASS | asset=%s reason=gate_already_ok", asset)
                passed_assets.append(asset)
                continue
            
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

        # Only restore backup if NO required assets passed (preserve partial success)
        if not passed_assets:
            _restore_model_artifacts(backup_dir, models_dir)
        return False
    except Exception as exc:
        log.error("Retraining exception: %s", exc, exc_info=True)
        _restore_model_artifacts(backup_dir, models_dir)
        return False


def _check_backtest_passed(metrics) -> bool:
    """Check if backtest meets minimum quality thresholds."""
    try:
        sharpe = float(getattr(metrics, "sharpe_ratio", 0.0) or 0.0)
        win_rate = float(getattr(metrics, "win_rate", 0.0) or 0.0)
        max_dd = float(getattr(metrics, "max_drawdown_pct", 0.0) or 0.0)
    except Exception:
        return False
    return (
        sharpe >= MIN_GATE_SHARPE and
        win_rate >= MIN_GATE_WIN_RATE and
        max_dd <= MAX_GATE_DRAWDOWN
    )


# ==========================
# Telegram Supervisor (waits when internet down)
# ==========================
def run_telegram_supervisor(stop_event: Event, notifier: Notifier) -> None:
    """
    Production Telegram polling:
    - Infinite retry with bounded backoff
    - On internet loss: WAIT silently + throttle logs (no crash)
    - stop_event stops polling (best-effort)
    """
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

            # skip_pending speeds up startup and avoids old backlog
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


# ==========================
# Engine Notify Worker (non-blocking queue consumer)
# ==========================
def run_engine_notify_worker(stop_event: Event) -> None:
    """
    Consumes engine notification queue without blocking trading loop.
    Messages from engine notifiers are delivered asynchronously.
    """
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
            pending = None  # Success - clear pending
        except NETWORK_EXC as exc:
            if log_rl.allow("notify_net"):
                log.warning("Engine notify network error: %s", exc)
            delay = backoff.delay(attempt)
            sleep_interruptible(stop_event, delay)
        except Exception as exc:
            log.error("Engine notify error: %s", exc)
            pending = None  # Drop on non-network errors
    
    # Best-effort drain on shutdown
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass


# ==========================
# Main
# ==========================
def main(argv: Optional[list[str]] = None) -> int:
    try:
        # Use a context manager to hold the socket lock for the duration of main()
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

    parser = argparse.ArgumentParser(description="XAUUSDm Scalping System (Exness MT5) - Production Runner")
    parser.add_argument("--headless", action="store_true", help="Оғоз бе Telegram (ҳолати VPS)")
    parser.add_argument("--engine-only", action="store_true", help="Фақат мотор, бе бот")
    parser.add_argument("--dry-run", action="store_true", help="Run in simulation mode (Mock MT5). Default is LIVE.")
    args = parser.parse_args(argv)

    # Default to LIVE (Production)
    if args.dry_run:
        engine.dry_run = True
        try:
            engine._check_model_health()
        except Exception:
            pass

    if not args.dry_run:
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
                engine._check_model_health()  # refresh gatekeeper after training
            except Exception:
                pass
    else:
        log.info("DRY_RUN_STARTUP | skipping strict model preflight")

    # Print System Matrix (Quantum Status)
    engine.print_startup_matrix()

    # ==========================
    # WIRING: Signal Flow
    # ==========================
    # Explicitly connect Engine -> Bot
    # This ensures signals generated in engine.py reach Bot/bot.py -> Telegram
    if send_signal_notification is not None:
        engine.set_signal_notifier(send_signal_notification)
    log.info("SIGNAL_NOTIFIER_WIRED | Engine -> Bot connected")

    if _tg_available:
        notifier = Notifier(shutdown.stop_event, queue_max=200)
        notifier.start()
    else:
        notifier = NullNotifier()

    log_monitor = LogMonitor(shutdown.stop_event)
    log_monitor.start()

    # Engine notification worker (fire-and-forget queue consumer)
    notify_worker_thread = None
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
