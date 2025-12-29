# mt5_client.py
from __future__ import annotations

import glob
import logging
import os
import subprocess
import threading
import time
from typing import Optional, Tuple

import MetaTrader5 as mt5

from config import EngineConfig, get_config_from_env

# ============================================================
# ERROR-only logging
# ============================================================
os.makedirs("Logs", exist_ok=True)

logger = logging.getLogger("mt5")
logger.setLevel(logging.ERROR)
logger.propagate = False

if not logger.handlers:
    fh = logging.FileHandler("Logs/mt5.log", encoding="utf-8")
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(fh)

# ============================================================
# Locks & global state
# ============================================================
MT5_LOCK = threading.RLock()   # protect every mt5.* call
INIT_LOCK = threading.Lock()   # protect init/shutdown sequence

_initialized: bool = False
_config: Optional[EngineConfig] = None
_mt5_process: Optional[subprocess.Popen] = None  # best-effort (Windows)

# ============================================================
# Helpers
# ============================================================
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

    # Production rule: if mt5_path was provided -> it MUST exist (fail fast)
    mt5_path = str(getattr(cfg, "mt5_path", "") or "").strip()
    if mt5_path and not os.path.exists(mt5_path):
        raise RuntimeError(f"MT5_PATH does not exist: {mt5_path}")


def _resolve_symbol(cfg: EngineConfig) -> str:
    sp = getattr(cfg, "symbol_params", None)
    if sp is None:
        return ""
    return getattr(sp, "resolved", "") or getattr(sp, "base", "") or ""


def _resolve_mt5_path_windows(cfg: EngineConfig) -> Optional[str]:
    """
    Resolve terminal64.exe path on Windows.
    Priority:
      1) cfg.mt5_path if exists
      2) common install locations (Program Files / Exness-named folders)
      3) APPDATA MetaQuotes Terminal instances
    """
    p = str(getattr(cfg, "mt5_path", "") or "").strip()
    if p:
        return p if os.path.exists(p) else None

    # Common installs
    candidates: list[str] = []
    for base in (r"C:\Program Files", r"C:\Program Files (x86)"):
        candidates += glob.glob(base + r"\MetaTrader 5*\terminal64.exe")
        candidates += glob.glob(base + r"\*\terminal64.exe")

    for x in candidates:
        if x and os.path.exists(x):
            return x

    # APPDATA terminals
    appdata = os.getenv("APPDATA", "")
    if appdata:
        for x in glob.glob(os.path.join(appdata, r"MetaQuotes\Terminal\*\terminal64.exe")):
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
    Best-effort: start terminal if not running.
    Returns True if process looks alive.
    """
    global _mt5_process
    try:
        if not path or not os.path.exists(path):
            return False

        # already started by this module
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

        time.sleep(2.5)
        return _mt5_process.poll() is None
    except Exception as exc:
        logger.error("MT5 terminal autostart failed: %s", exc)
        return False


def _mt5_health_check(cfg: EngineConfig) -> Tuple[bool, str]:
    """
    Minimal fast health check. Must be called under MT5_LOCK.
    """
    try:
        ti = mt5.terminal_info()
        if ti is None:
            return False, "terminal_info_none"

        # connected to trade server
        if not bool(getattr(ti, "connected", False)):
            return False, "terminal_not_connected"

        # autotrading in terminal
        if not bool(getattr(ti, "trade_allowed", False)):
            return False, "trade_not_allowed"

        acc = mt5.account_info()
        if acc is None:
            return False, "account_info_none"

        if int(getattr(acc, "login", -1)) != int(cfg.login):
            return False, "wrong_account"

        sym = _resolve_symbol(cfg)
        if sym:
            info = mt5.symbol_info(sym)
            if info is None:
                return False, f"symbol_not_found:{sym}"

            # ensure in MarketWatch
            if hasattr(info, "visible") and (not bool(info.visible)):
                if not mt5.symbol_select(sym, True):
                    return False, f"symbol_select_failed:{sym}"

            tick = mt5.symbol_info_tick(sym)
            if tick is None:
                return False, f"no_tick:{sym}"

            bid = float(getattr(tick, "bid", 0.0) or 0.0)
            ask = float(getattr(tick, "ask", 0.0) or 0.0)
            if bid <= 0.0 and ask <= 0.0:
                return False, f"bad_tick_price:{sym}"

        return True, "ok"
    except Exception as exc:
        logger.error("MT5 health check exception: %s", exc)
        return False, "health_check_exception"


def _initialize_and_login(cfg: EngineConfig, timeout_ms: int) -> None:
    """
    Initialize + login:
      1) initialize(login,password,server, path?, portable?, timeout)
      2) fallback initialize(path?, portable?, timeout) then login()
    Raises RuntimeError on failure.
    """
    _validate_cfg(cfg)

    portable = bool(getattr(cfg, "mt5_portable", False))

    mt5_path = str(getattr(cfg, "mt5_path", "") or "").strip()
    if _is_windows() and not mt5_path:
        mt5_path = _resolve_mt5_path_windows(cfg) or ""

    init_kwargs = {"timeout": int(timeout_ms), "portable": bool(portable)}
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


# ============================================================
# Public API
# ============================================================
def ensure_mt5(max_retries: int = 6, retry_delay: float = 1.5) -> mt5:
    global _initialized, _config

    # fast path
    with MT5_LOCK:
        if _initialized and _config is not None:
            ok, reason = _mt5_health_check(_config)
            if ok:
                return mt5

            # HEALTH FAIL -> reset and continue to re-init (NO None returns)
            logger.error("Fast path health check failed: %s", reason)
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
                ok, reason = _mt5_health_check(_config)
                if ok:
                    return mt5

                logger.error("Double-check health failed: %s", reason)
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                _initialized = False

        last_exc: Optional[Exception] = None
        portable = bool(getattr(_config, "mt5_portable", False))
        autostart = bool(getattr(_config, "mt5_autostart", True))

        for attempt in range(int(max_retries)):
            try:
                if _is_windows() and autostart:
                    try:
                        if not _is_mt5_terminal_running_windows():
                            p = _resolve_mt5_path_windows(_config)
                            if p:
                                _start_mt5_terminal_windows(str(p), portable=portable)
                            else:
                                logger.error("Could not resolve MT5 terminal path for autostart")
                    except Exception as e:
                        logger.error("Autostart failed: %s", e)

                with MT5_LOCK:
                    _initialize_and_login(_config, timeout_ms=10_000)
                    ok, reason = _mt5_health_check(_config)
                    if not ok:
                        try:
                            mt5.shutdown()
                        except Exception:
                            pass
                        raise RuntimeError(f"mt5_health_failed:{reason}")

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

                logger.error("ensure_mt5 attempt %d/%d failed: %s", attempt + 1, int(max_retries), exc)

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
                logger.error("MT5 shutdown error: %s", exc)
            finally:
                _initialized = False


__all__ = ["ensure_mt5", "shutdown_mt5", "MT5_LOCK", "mt5"]
