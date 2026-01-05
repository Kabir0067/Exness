# mt5_client.py  (SENIOR / PRODUCTION-GRADE)
from __future__ import annotations

import glob
import logging
import os
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Optional, Tuple

import MetaTrader5 as mt5
from pywinauto import Application  


from config_xau import EngineConfig, get_config_from_env
from log_config import LOG_DIR as LOG_ROOT, get_log_path

# =============================================================================
# ERROR-only logging (rotated)
# =============================================================================
LOG_DIR = LOG_ROOT

logger = logging.getLogger("mt5")
logger.setLevel(logging.ERROR)
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(
        filename=str(get_log_path("mt5.log")),
        maxBytes=int(os.getenv("MT5_LOG_MAX_BYTES", "5242880")),  # 5MB
        backupCount=int(os.getenv("MT5_LOG_BACKUPS", "5")),
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    logger.addHandler(fh)

# =============================================================================
# Locks & global state
# =============================================================================
MT5_LOCK = threading.RLock()   # protect every mt5.* call
INIT_LOCK = threading.Lock()   # protect init/shutdown sequence

_initialized: bool = False
_config: Optional[EngineConfig] = None
_mt5_process: Optional[subprocess.Popen] = None  # best-effort (Windows)
_last_health_reason: str = "not_initialized"

# UI auto-enable (OFF by default)
_UI_ENABLE_ALGO = os.getenv("MT5_UI_ENABLE_ALGO", "0").strip() == "1"
_UI_ENABLE_ALGO_ONCE = False
_UI_LOCK = threading.Lock()


# =============================================================================
# Internal models
# =============================================================================
@dataclass(frozen=True)
class Health:
    ok: bool
    reason: str


# =============================================================================
# Helpers
# =============================================================================
def _is_windows() -> bool:
    return os.name == "nt"


def _last_error_str() -> str:
    try:
        return str(mt5.last_error())
    except Exception:
        return "unknown_mt5_error"


def _sleep_backoff(base: float, attempt: int, cap: float = 30.0) -> None:
    delay = min(float(cap), float(base) * (2.0 ** int(attempt)))
    time.sleep(delay)


def _validate_cfg(cfg: EngineConfig) -> None:
    if cfg is None:
        raise RuntimeError("EngineConfig is None")

    if getattr(cfg, "login", None) in (None, "", 0):
        raise RuntimeError("MT5 login is missing")
    if getattr(cfg, "password", None) in (None, ""):
        raise RuntimeError("MT5 password is missing")
    if getattr(cfg, "server", None) in (None, ""):
        raise RuntimeError("MT5 server is missing")

    mt5_path = str(getattr(cfg, "mt5_path", "") or "").strip()
    if mt5_path and not os.path.exists(mt5_path):
        raise RuntimeError(f"MT5_PATH does not exist: {mt5_path}")


def _resolve_symbol(cfg: EngineConfig) -> str:
    sp = getattr(cfg, "symbol_params", None)
    if sp is None:
        return ""
    return str(getattr(sp, "resolved", "") or getattr(sp, "base", "") or "").strip()


def _resolve_mt5_path_windows(cfg: EngineConfig) -> Optional[str]:
    """
    Resolve terminal64.exe path on Windows.
    Priority:
      1) cfg.mt5_path if exists
      2) common install locations
      3) APPDATA / LOCALAPPDATA MetaQuotes Terminal instances
    """
    p = str(getattr(cfg, "mt5_path", "") or "").strip()
    if p and os.path.exists(p):
        return p

    candidates: list[str] = []
    for base in (r"C:\Program Files", r"C:\Program Files (x86)"):
        candidates += glob.glob(base + r"\MetaTrader 5*\terminal64.exe")
        candidates += glob.glob(base + r"\MetaQuotes*\terminal64.exe")
        candidates += glob.glob(base + r"\Exness*\terminal64.exe")
        candidates += glob.glob(base + r"\*\terminal64.exe")

    for env_key in ("APPDATA", "LOCALAPPDATA"):
        root = os.getenv(env_key, "")
        if root:
            candidates += glob.glob(os.path.join(root, r"MetaQuotes\Terminal\*\terminal64.exe"))

    for x in candidates:
        if x and os.path.exists(x):
            return x

    return None


def _is_mt5_terminal_running_windows() -> bool:
    try:
        out = subprocess.check_output(["tasklist"], text=True, errors="ignore").lower()
        return "terminal64.exe" in out
    except Exception:
        return False


def _start_mt5_terminal_windows(path: str, portable: bool) -> bool:
    """
    Best-effort start MT5 terminal if not running.
    """
    global _mt5_process

    if not path or not os.path.exists(path):
        return False

    try:
        if _mt5_process and _mt5_process.poll() is None:
            return True

        cmd = [path]
        if portable:
            cmd.append("/portable")

        creationflags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW

        _mt5_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        time.sleep(2.0)
        return _mt5_process.poll() is None
    except Exception as exc:
        logger.error("MT5 terminal autostart failed: %s | tb=%s", exc, traceback.format_exc())
        return False


def _enable_algo_trading_ui_once() -> bool:
    """
    UI automation to click AlgoTrading in MT5 terminal.
    Runs only if:
      - Windows
      - MT5_UI_ENABLE_ALGO=1
      - executed once per process
    """
    global _UI_ENABLE_ALGO_ONCE

    if not (_is_windows() and _UI_ENABLE_ALGO):
        return False

    with _UI_LOCK:
        if _UI_ENABLE_ALGO_ONCE:
            return False
        _UI_ENABLE_ALGO_ONCE = True

    try:
        # lazy import so project does not crash if not installed

        app = Application(backend="uia").connect(path="terminal64.exe")
        main = app.window(title_re=".*MetaTrader 5.*")

        # best-effort selector: MT5 versions differ
        # Try Button first, then MenuItem
        btn = main.child_window(title_re=".*Algo.*Trading.*", control_type="Button")
        if btn.exists(timeout=1.0):
            btn.click_input()
            return True

        mi = main.child_window(title_re=".*Algo.*Trading.*", control_type="MenuItem")
        if mi.exists(timeout=1.0):
            mi.click_input()
            return True

        return False
    except Exception as exc:
        logger.error("AlgoTrading UI enable failed: %s | tb=%s", exc, traceback.format_exc())
        return False


def _mt5_health_check(cfg: EngineConfig) -> Health:
    """
    Minimal fast health check. Must be called under MT5_LOCK.
    """
    try:
        ti = mt5.terminal_info()
        if ti is None:
            return Health(False, "terminal_info_none")

        if not bool(getattr(ti, "connected", False)):
            return Health(False, "terminal_not_connected")

        # Terminal AlgoTrading switch
        if not bool(getattr(ti, "trade_allowed", False)):
            _enable_algo_trading_ui_once()
            ti2 = mt5.terminal_info()
            if not ti2 or not bool(getattr(ti2, "trade_allowed", False)):
                return Health(False, "algo_trading_disabled_in_terminal")

        acc = mt5.account_info()
        if acc is None:
            return Health(False, "account_info_none")

        if not bool(getattr(acc, "trade_allowed", False)):
            return Health(False, "trading_disabled_for_account")

        if int(getattr(acc, "login", -1)) != int(cfg.login):
            return Health(False, "wrong_account")

        sym = _resolve_symbol(cfg)
        if sym:
            info = mt5.symbol_info(sym)
            if info is None:
                return Health(False, f"symbol_not_found:{sym}")

            if hasattr(info, "visible") and (not bool(info.visible)):
                if not mt5.symbol_select(sym, True):
                    return Health(False, f"symbol_select_failed:{sym}")

            tick = mt5.symbol_info_tick(sym)
            if tick is None:
                return Health(False, f"no_tick:{sym}")

            bid = float(getattr(tick, "bid", 0.0) or 0.0)
            ask = float(getattr(tick, "ask", 0.0) or 0.0)
            if bid <= 0.0 and ask <= 0.0:
                return Health(False, f"bad_tick_price:{sym}")

        return Health(True, "ok")
    except Exception as exc:
        logger.error("MT5 health check exception: %s | tb=%s", exc, traceback.format_exc())
        return Health(False, "health_check_exception")


def _initialize_and_login(cfg: EngineConfig, timeout_ms: int) -> None:
    """
    Initialize + login. Raises RuntimeError on failure.
    """
    _validate_cfg(cfg)

    portable = bool(getattr(cfg, "mt5_portable", False))

    mt5_path = str(getattr(cfg, "mt5_path", "") or "").strip()
    if _is_windows() and not mt5_path:
        mt5_path = _resolve_mt5_path_windows(cfg) or ""

    init_kwargs: dict = {"timeout": int(timeout_ms), "portable": bool(portable)}
    if mt5_path:
        init_kwargs["path"] = str(mt5_path)

    # reset state before init (important after crash)
    try:
        mt5.shutdown()
    except Exception:
        pass
    time.sleep(0.15)

    ok = mt5.initialize(
        login=int(cfg.login),
        password=str(cfg.password),
        server=str(cfg.server),
        **init_kwargs,
    )
    if ok:
        return

    err1 = _last_error_str()

    # fallback: initialize() then login()
    try:
        mt5.shutdown()
    except Exception:
        pass
    time.sleep(0.20)

    ok2 = mt5.initialize(**init_kwargs)
    if not ok2:
        raise RuntimeError(f"mt5.initialize failed: {_last_error_str()} | first_try={err1}")

    logged = mt5.login(
        login=int(cfg.login),
        password=str(cfg.password),
        server=str(cfg.server),
        timeout=int(timeout_ms),
    )
    if not logged:
        err2 = _last_error_str()
        try:
            mt5.shutdown()
        except Exception:
            pass
        raise RuntimeError(f"mt5.login failed: {err2} | first_try={err1}")


# =============================================================================
# Public API
# =============================================================================
def ensure_mt5(max_retries: int = 6, retry_delay: float = 1.5) -> mt5:
    """
    Contract:
      - returns mt5 on success
      - raises RuntimeError on failure
      - never returns None
      - thread-safe
    """
    global _initialized, _config, _last_health_reason

    # fast path
    with MT5_LOCK:
        if _initialized and _config is not None:
            h = _mt5_health_check(_config)
            _last_health_reason = h.reason
            if h.ok:
                return mt5

            logger.error("Fast path health failed: %s", h.reason)
            try:
                mt5.shutdown()
            except Exception:
                pass
            _initialized = False

    with INIT_LOCK:
        if _config is None:
            _config = get_config_from_env()
        _validate_cfg(_config)

        # double-check health under MT5_LOCK
        with MT5_LOCK:
            if _initialized:
                h2 = _mt5_health_check(_config)
                _last_health_reason = h2.reason
                if h2.ok:
                    return mt5

                logger.error("Double-check health failed: %s", h2.reason)
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                _initialized = False

        portable = bool(getattr(_config, "mt5_portable", False))
        autostart = bool(getattr(_config, "mt5_autostart", True))
        timeout_ms = int(getattr(_config, "mt5_timeout_ms", 10_000) or 10_000)

        last_exc: Optional[Exception] = None

        for attempt in range(int(max_retries)):
            try:
                if _is_windows() and autostart:
                    if not _is_mt5_terminal_running_windows():
                        p = _resolve_mt5_path_windows(_config)
                        if p:
                            _start_mt5_terminal_windows(str(p), portable=portable)
                        else:
                            logger.error("Could not resolve MT5 terminal path for autostart")

                with MT5_LOCK:
                    _initialize_and_login(_config, timeout_ms=timeout_ms)
                    h = _mt5_health_check(_config)
                    _last_health_reason = h.reason
                    if not h.ok:
                        try:
                            mt5.shutdown()
                        except Exception:
                            pass
                        raise RuntimeError(f"mt5_health_failed:{h.reason}")

                    _initialized = True
                    return mt5

            except Exception as exc:
                last_exc = exc
                with MT5_LOCK:
                    try:
                        mt5.shutdown()
                    except Exception:
                        pass
                    _initialized = False

                logger.error(
                    "ensure_mt5 attempt %d/%d failed: %s | tb=%s",
                    attempt + 1,
                    int(max_retries),
                    exc,
                    traceback.format_exc(),
                )

                if attempt < int(max_retries) - 1:
                    _sleep_backoff(float(retry_delay), attempt, cap=30.0)

        raise RuntimeError(f"MT5 initialization failed after {max_retries} attempts: {last_exc}")


def shutdown_mt5() -> None:
    """
    Safe shutdown (idempotent).
    """
    global _initialized
    with INIT_LOCK:
        with MT5_LOCK:
            if not _initialized:
                return
            try:
                mt5.shutdown()
            except Exception as exc:
                logger.error("MT5 shutdown error: %s | tb=%s", exc, traceback.format_exc())
            finally:
                _initialized = False


def mt5_status() -> Tuple[bool, str]:
    """
    Non-throwing status (for health logs / telegram).
    """
    global _initialized, _config, _last_health_reason
    with MT5_LOCK:
        if not _initialized or _config is None:
            return False, "not_initialized"
        return True, _last_health_reason


__all__ = ["ensure_mt5", "shutdown_mt5", "mt5_status", "MT5_LOCK", "mt5"]
