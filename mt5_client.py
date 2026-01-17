# mt5_client.py (PRODUCTION / IPC-TIMEOUT HARDENED / ENV-COMPAT / SINGLE-INSTANCE SAFE)
from __future__ import annotations

import atexit
import glob
import logging
import os
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Tuple

import MetaTrader5 as mt5

from log_config import LOG_DIR as LOG_ROOT, get_log_path


# =============================================================================
# Logging (ERROR-only by default)
# =============================================================================
_LOG_LEVEL = (os.getenv("MT5_LOG_LEVEL", "ERROR") or "ERROR").strip().upper()
logger = logging.getLogger("mt5")
logger.setLevel(getattr(logging, _LOG_LEVEL, logging.ERROR))
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(
        filename=str(get_log_path("mt5.log")),
        maxBytes=int(os.getenv("MT5_LOG_MAX_BYTES", "5242880")),  # 5MB
        backupCount=int(os.getenv("MT5_LOG_BACKUPS", "5")),
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(getattr(logging, _LOG_LEVEL, logging.ERROR))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    logger.addHandler(fh)


# =============================================================================
# Locks & state
# =============================================================================
MT5_LOCK = threading.RLock()   # protect every mt5.* call
INIT_LOCK = threading.Lock()   # protect init/shutdown sequence

_initialized: bool = False
_last_health_reason: str = "not_initialized"
_last_health_log_ts: float = 0.0


# =============================================================================
# ENV flags
# =============================================================================
def _env_bool(key: str, default: bool) -> bool:
    raw = (os.getenv(key, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


ENV_SINGLE_INSTANCE = _env_bool("MT5_SINGLE_INSTANCE_LOCK", True)
ENV_AUTOSTART = _env_bool("MT5_AUTOSTART", True)
ENV_TASKKILL_ON_IPC = _env_bool("MT5_TASKKILL_ON_IPC_TIMEOUT", True)
ENV_REQUIRE_PATH = _env_bool("MT5_REQUIRE_PATH", True)

ENV_TIMEOUT_MS = int(os.getenv("MT5_TIMEOUT_MS", "90000"))
ENV_START_WAIT_SEC = float(os.getenv("MT5_START_WAIT_SEC", "10"))
ENV_READY_TIMEOUT_SEC = float(os.getenv("MT5_READY_TIMEOUT_SEC", "60"))

ENV_MT5_PATH = (os.getenv("MT5_PATH", "") or "").strip()


# =============================================================================
# Models
# =============================================================================
@dataclass(frozen=True)
class MT5LoginCfg:
    login: int
    password: str
    server: str


@dataclass(frozen=True)
class Health:
    ok: bool
    reason: str


# =============================================================================
# Helpers
# =============================================================================
def _is_windows() -> bool:
    return os.name == "nt"


def _last_error() -> Tuple[int, str]:
    try:
        code, msg = mt5.last_error()
        return int(code), str(msg)
    except Exception:
        return -99999, "unknown_mt5_error"


def _sleep_backoff(base: float, attempt: int, cap: float = 30.0) -> None:
    delay = min(float(cap), float(base) * (2.0 ** int(attempt)))
    time.sleep(delay)


def _throttled_health_log(reason: str) -> None:
    global _last_health_log_ts
    now = time.time()
    if now - _last_health_log_ts >= 15.0:
        logger.error("Fast path health failed: %s", reason)
        _last_health_log_ts = now


# =============================================================================
# Single-instance lock (prevents 2 python processes calling MT5)
# =============================================================================
_lock_guard = threading.Lock()
_lock_fp = None
_lock_acquired = False

def _lock_file_path() -> Path:
    base = Path(str(LOG_ROOT)) if str(LOG_ROOT) else Path.cwd() / "Logs"
    base.mkdir(parents=True, exist_ok=True)
    return base / "mt5_single_instance.lock"

def _read_lock_file(p: Path) -> str:
    try:
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        pass
    return ""

def _acquire_single_instance_lock() -> None:
    global _lock_fp, _lock_acquired
    if not ENV_SINGLE_INSTANCE:
        return
    with _lock_guard:
        if _lock_acquired and _lock_fp:
            return
        p = _lock_file_path()
        try:
            _lock_fp = open(p, "a+", encoding="utf-8")
            _lock_fp.seek(0)
            try:
                _lock_fp.truncate(0)
                _lock_fp.write(f"pid={os.getpid()} ts={int(time.time())}\n")
                _lock_fp.flush()
            except Exception:
                pass

            if _is_windows():
                import msvcrt  # type: ignore
                try:
                    _lock_fp.seek(0)
                    msvcrt.locking(_lock_fp.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    other = _read_lock_file(p)
                    raise RuntimeError(f"Another engine process is running (lock busy). Stop it first. lock_info={other}")
            else:
                import fcntl  # type: ignore
                try:
                    fcntl.flock(_lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    other = _read_lock_file(p)
                    raise RuntimeError(f"Another engine process is running (lock busy). Stop it first. lock_info={other}")

            _lock_acquired = True
        except Exception as exc:
            try:
                if _lock_fp:
                    _lock_fp.close()
            except Exception:
                pass
            _lock_fp = None
            _lock_acquired = False
            raise RuntimeError(f"Failed to acquire single-instance lock: {exc}") from exc

def _release_single_instance_lock() -> None:
    global _lock_fp, _lock_acquired
    if not ENV_SINGLE_INSTANCE:
        return
    with _lock_guard:
        if not _lock_acquired or not _lock_fp:
            return
        try:
            if _is_windows():
                import msvcrt  # type: ignore
                try:
                    _lock_fp.seek(0)
                    msvcrt.locking(_lock_fp.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            else:
                import fcntl  # type: ignore
                try:
                    fcntl.flock(_lock_fp.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        finally:
            try:
                _lock_fp.close()
            except Exception:
                pass
            _lock_fp = None
            _lock_acquired = False


# =============================================================================
# Config extraction: env-first, then config_xau (if exists)
# =============================================================================
def _get_login_cfg() -> MT5LoginCfg:
    # Accept both MT5_* and EXNESS_*
    def _get_int(keys: list[str]) -> int:
        for k in keys:
            v = (os.getenv(k, "") or "").strip()
            if v:
                try:
                    return int(v)
                except Exception:
                    pass
        return 0

    def _get_str(keys: list[str]) -> str:
        for k in keys:
            v = (os.getenv(k, "") or "").strip()
            if v:
                return v
        return ""

    login = _get_int(["MT5_LOGIN", "EXNESS_LOGIN"])
    password = _get_str(["MT5_PASSWORD", "EXNESS_PASSWORD"])
    server = _get_str(["MT5_SERVER", "EXNESS_SERVER"])

    if login and password and server:
        return MT5LoginCfg(login=login, password=password, server=server)

    # fallback to config_xau.get_config_from_env if present
    try:
        from config_xau import get_config_from_env  # type: ignore
        cfg = get_config_from_env()
        login2 = int(getattr(cfg, "login", 0) or 0)
        password2 = str(getattr(cfg, "password", "") or "")
        server2 = str(getattr(cfg, "server", "") or "")
        if login2 and password2 and server2:
            return MT5LoginCfg(login=login2, password=password2, server=server2)
    except Exception:
        pass

    raise RuntimeError("MT5 credentials missing. Set MT5_LOGIN/MT5_PASSWORD/MT5_SERVER (or EXNESS_*) in .env")


def _resolve_mt5_path_windows() -> str:
    if ENV_MT5_PATH and os.path.exists(ENV_MT5_PATH):
        return ENV_MT5_PATH

    candidates: list[str] = []
    for base in (r"C:\Program Files", r"C:\Program Files (x86)"):
        candidates += glob.glob(base + r"\MetaTrader 5*\terminal64.exe")
        candidates += glob.glob(base + r"\MetaQuotes*\terminal64.exe")
        candidates += glob.glob(base + r"\Exness*\terminal64.exe")
    for env_key in ("APPDATA", "LOCALAPPDATA"):
        root = os.getenv(env_key, "")
        if root:
            candidates += glob.glob(os.path.join(root, r"MetaQuotes\Terminal\*\terminal64.exe"))

    for x in candidates:
        if x and os.path.exists(x):
            return x
    return ""


# =============================================================================
# Windows process helpers (kill/restart)
# =============================================================================
def _is_terminal_running_windows() -> bool:
    try:
        out = subprocess.check_output(["tasklist"], text=True, errors="ignore").lower()
        return "terminal64.exe" in out
    except Exception:
        return False


def _taskkill_terminal_windows() -> None:
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", "terminal64.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass
    time.sleep(1.5)


def _start_terminal_windows(path: str, portable: bool) -> None:
    if not path or not os.path.exists(path):
        raise RuntimeError(f"MT5_PATH invalid: {path}")

    cmd = [path]
    if portable:
        cmd.append("/portable")

    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)
    time.sleep(max(2.0, float(ENV_START_WAIT_SEC)))


# =============================================================================
# Health check
# =============================================================================
def _health(login_cfg: MT5LoginCfg) -> Health:
    try:
        ti = mt5.terminal_info()
        if ti is None:
            return Health(False, "terminal_info_none")
        if not bool(getattr(ti, "connected", False)):
            return Health(False, "terminal_not_connected")
        if not bool(getattr(ti, "trade_allowed", False)):
            return Health(False, "algo_trading_disabled_in_terminal")

        ai = mt5.account_info()
        if ai is None:
            return Health(False, "account_info_none")
        if int(getattr(ai, "login", -1)) != int(login_cfg.login):
            return Health(False, "wrong_account")
        if not bool(getattr(ai, "trade_allowed", False)):
            return Health(False, "trading_disabled_for_account")

        return Health(True, "ok")
    except Exception:
        logger.error("health exception | tb=%s", traceback.format_exc())
        return Health(False, "health_check_exception")


def _wait_ready(login_cfg: MT5LoginCfg, timeout_sec: float) -> None:
    deadline = time.time() + float(timeout_sec)
    last = "waiting"
    while time.time() < deadline:
        h = _health(login_cfg)
        last = h.reason
        if h.ok:
            return
        time.sleep(0.6)
    raise RuntimeError(f"mt5_health_failed:{last}")


# =============================================================================
# Initialize + login (multi-fallback)
# =============================================================================
def _init_and_login(login_cfg: MT5LoginCfg, mt5_path: str, portable: bool, timeout_ms: int) -> None:
    # reset
    try:
        mt5.shutdown()
    except Exception:
        pass
    time.sleep(0.35)

    init_kwargs = {
        "path": str(mt5_path) if mt5_path else None,
        "portable": bool(portable),
        "timeout": int(timeout_ms),
    }

    # Attempt #1: initialize with creds (often more stable)
    ok1 = mt5.initialize(
        login=int(login_cfg.login),
        password=str(login_cfg.password),
        server=str(login_cfg.server),
        **{k: v for k, v in init_kwargs.items() if v is not None},
    )
    if ok1:
        return

    e1 = _last_error()

    # Attempt #2: initialize then login
    try:
        mt5.shutdown()
    except Exception:
        pass
    time.sleep(0.35)

    ok2 = mt5.initialize(**{k: v for k, v in init_kwargs.items() if v is not None})
    if not ok2:
        e2 = _last_error()
        raise RuntimeError(f"mt5.initialize failed: {e2} | first_try={e1}")

    logged = mt5.login(
        login=int(login_cfg.login),
        password=str(login_cfg.password),
        server=str(login_cfg.server),
        timeout=int(timeout_ms),
    )
    if not logged:
        e3 = _last_error()
        try:
            mt5.shutdown()
        except Exception:
            pass
        raise RuntimeError(f"mt5.login failed: {e3} | first_try={e1}")


# =============================================================================
# Public API
# =============================================================================
def ensure_mt5(max_retries: int = 6, retry_delay: float = 1.5) -> mt5:
    global _initialized, _last_health_reason

    _acquire_single_instance_lock()

    login_cfg = _get_login_cfg()

    portable = _env_bool("MT5_PORTABLE", False)
    autostart = ENV_AUTOSTART

    timeout_base = max(int(ENV_TIMEOUT_MS), 60_000)
    timeouts = [timeout_base, max(timeout_base, 180_000)]

    mt5_path = ""
    if _is_windows():
        mt5_path = _resolve_mt5_path_windows()
        if ENV_REQUIRE_PATH and (not mt5_path or not os.path.exists(mt5_path)):
            raise RuntimeError("MT5_PATH is required on Windows. Set MT5_PATH to terminal64.exe full path")
    else:
        mt5_path = ENV_MT5_PATH

    # fast path
    with MT5_LOCK:
        if _initialized:
            h = _health(login_cfg)
            _last_health_reason = h.reason
            if h.ok:
                return mt5
            _throttled_health_log(h.reason)
            try:
                mt5.shutdown()
            except Exception:
                pass
            _initialized = False

    last_exc: Optional[Exception] = None

    with INIT_LOCK:
        for attempt in range(int(max_retries)):
            try:
                # ensure terminal running (Windows)
                if _is_windows() and autostart:
                    if not _is_terminal_running_windows():
                        _start_terminal_windows(mt5_path, portable=portable)
                    else:
                        # give terminal time to become IPC-ready
                        if attempt == 0:
                            time.sleep(min(6.0, float(ENV_START_WAIT_SEC)))

                with MT5_LOCK:
                    _init_and_login(login_cfg, mt5_path, portable, timeouts[min(attempt, len(timeouts) - 1)])
                    _wait_ready(login_cfg, timeout_sec=float(ENV_READY_TIMEOUT_SEC))
                    _initialized = True
                    _last_health_reason = "ok"
                    return mt5

            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()

                fatal_ipc = ("ipc timeout" in msg) or ("(-10005" in msg) or ("-10005" in msg)

                if _is_windows() and ENV_TASKKILL_ON_IPC and fatal_ipc:
                    logger.error("IPC timeout -> taskkill terminal64.exe + retry | err=%s", exc)
                    try:
                        mt5.shutdown()
                    except Exception:
                        pass
                    _taskkill_terminal_windows()
                    if autostart:
                        try:
                            _start_terminal_windows(mt5_path, portable=portable)
                        except Exception as exc2:
                            logger.error("MT5 restart failed: %s | tb=%s", exc2, traceback.format_exc())

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
    global _initialized
    with INIT_LOCK:
        with MT5_LOCK:
            try:
                mt5.shutdown()
            except Exception as exc:
                logger.error("MT5 shutdown error: %s | tb=%s", exc, traceback.format_exc())
            _initialized = False
    _release_single_instance_lock()


def mt5_status() -> Tuple[bool, str]:
    with MT5_LOCK:
        return bool(_initialized), str(_last_health_reason)


atexit.register(shutdown_mt5)

__all__ = ["ensure_mt5", "shutdown_mt5", "mt5_status", "MT5_LOCK", "mt5"]
