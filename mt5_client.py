# mt5_client.py (SENIOR / PRODUCTION / ENV-CREDS ONLY / IPC-HARDENED / SINGLE-INSTANCE SAFE)
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
# Public lock: protect EVERY mt5.* call
# =============================================================================
MT5_LOCK = threading.RLock()

# Protect init/shutdown sequence (MT5 is not thread-safe there)
_INIT_LOCK = threading.Lock()

# Internal state
_initialized: bool = False
_last_health_reason: str = "not_initialized"
_last_health_log_ts_mono: float = 0.0  # monotonic

# Single-instance file lock state
_lock_guard = threading.Lock()
_lock_fp = None
_lock_acquired = False


# =============================================================================
# Configuration (ENV for credentials only)
# =============================================================================
@dataclass(frozen=True)
class MT5Credentials:
    login: int
    password: str
    server: str


@dataclass(frozen=True)
class MT5ClientConfig:
    # Credentials (required)
    creds: MT5Credentials

    # Terminal start/attach options
    mt5_path: Optional[str] = None  # full path to terminal64.exe (recommended on Windows)
    portable: bool = False
    autostart: bool = True
    require_path_on_windows: bool = True

    # Robustness
    single_instance_lock: bool = True
    taskkill_on_ipc_timeout: bool = True
    timeout_ms: int = 90_000
    start_wait_sec: float = 3.0
    ready_timeout_sec: float = 30.0
    max_retries: int = 6
    retry_backoff_base_sec: float = 1.2
    backoff_cap_sec: float = 12.0

    # Logging
    log_level: int = logging.ERROR
    log_max_bytes: int = 5_242_880  # 5MB
    log_backups: int = 5
    health_log_throttle_sec: float = 15.0

    # IPC timeout detection markers
    ipc_markers: Tuple[str, ...] = ("ipc timeout", "-10005", "(-10005")

    # Safety: if True, validate account login must match creds.login
    require_correct_account: bool = True


# =============================================================================
# Logging (ERROR-only by default)
# =============================================================================
logger = logging.getLogger("mt5")
logger.setLevel(logging.ERROR)
logger.propagate = False


def _setup_logger(cfg: MT5ClientConfig) -> None:
    logger.setLevel(int(cfg.log_level))
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return

    fh = RotatingFileHandler(
        filename=str(get_log_path("mt5.log")),
        maxBytes=int(cfg.log_max_bytes),
        backupCount=int(cfg.log_backups),
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(int(cfg.log_level))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    logger.addHandler(fh)


# =============================================================================
# Helpers
# =============================================================================
def _is_windows() -> bool:
    return os.name == "nt"


def _mono() -> float:
    return time.monotonic()


def _last_error() -> Tuple[int, str]:
    # mt5.last_error() is a mt5.* call -> protect
    with MT5_LOCK:
        try:
            code, msg = mt5.last_error()
            return int(code), str(msg)
        except Exception:
            return -99999, "unknown_mt5_error"


def _sleep_backoff(base: float, attempt: int, cap: float) -> None:
    """
    Deterministic exponential backoff with a tiny deterministic additive term.
    No randomness (good for reproducible behavior in production logs).
    """
    a = int(attempt)
    delay = float(base) * (2.0 ** a)
    delay = min(float(cap), delay)
    delay = delay + min(0.25, 0.03 * a)  # deterministic micro-add
    time.sleep(delay)


def _throttled_health_log(cfg: MT5ClientConfig, reason: str) -> None:
    global _last_health_log_ts_mono
    now = _mono()
    if now - _last_health_log_ts_mono >= float(cfg.health_log_throttle_sec):
        logger.error("Fast path health failed: %s", reason)
        _last_health_log_ts_mono = now


def _validate_cfg(cfg: MT5ClientConfig) -> None:
    # Strict validation (fail fast, deterministic)
    if cfg is None:
        raise RuntimeError("MT5ClientConfig is None")

    login = int(getattr(cfg.creds, "login", 0) or 0)
    password = str(getattr(cfg.creds, "password", "") or "")
    server = str(getattr(cfg.creds, "server", "") or "")

    if login <= 0:
        raise RuntimeError("MT5 credentials invalid: login must be > 0")
    if not password:
        raise RuntimeError("MT5 credentials invalid: password is empty")
    if not server:
        raise RuntimeError("MT5 credentials invalid: server is empty")

    if int(cfg.timeout_ms) <= 0:
        raise RuntimeError("timeout_ms must be > 0")
    if float(cfg.ready_timeout_sec) <= 0:
        raise RuntimeError("ready_timeout_sec must be > 0")
    if int(cfg.max_retries) <= 0:
        raise RuntimeError("max_retries must be > 0")


# =============================================================================
# Single-instance lock (prevents running 2 python engines controlling MT5)
# =============================================================================
def _lock_file_path() -> Path:
    base = Path(str(LOG_ROOT)) if str(LOG_ROOT) else (Path.cwd() / "Logs")
    base.mkdir(parents=True, exist_ok=True)
    return base / "mt5_single_instance.lock"


def _read_lock_file(p: Path) -> str:
    try:
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        pass
    return ""


def _write_lock_info(p: Path) -> None:
    # Called only after lock is acquired
    try:
        info = f"pid={os.getpid()} ts={int(time.time())}\n"
        # Keep file content minimal and deterministic
        _lock_fp.seek(0)
        _lock_fp.truncate(0)
        _lock_fp.write(info)
        _lock_fp.flush()
        try:
            os.fsync(_lock_fp.fileno())
        except Exception:
            pass
    except Exception:
        pass


def _acquire_single_instance_lock(cfg: MT5ClientConfig) -> None:
    """
    Correctness fix:
    - NEVER truncate/write before acquiring OS-level lock (otherwise you destroy the other process info).
    """
    global _lock_fp, _lock_acquired
    if not cfg.single_instance_lock:
        return

    with _lock_guard:
        if _lock_acquired and _lock_fp:
            return

        p = _lock_file_path()
        try:
            _lock_fp = open(p, "a+", encoding="utf-8")
            _lock_fp.seek(0)

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
            _write_lock_info(p)

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
# Windows path resolution (NO ENV)
# =============================================================================
def _resolve_mt5_path_windows(mt5_path: Optional[str]) -> str:
    if mt5_path and os.path.exists(mt5_path):
        return mt5_path

    candidates: list[str] = []

    # Program Files candidates
    for base in (r"C:\Program Files", r"C:\Program Files (x86)"):
        candidates += glob.glob(base + r"\MetaTrader 5*\terminal64.exe")
        candidates += glob.glob(base + r"\MetaQuotes*\terminal64.exe")
        candidates += glob.glob(base + r"\Exness*\terminal64.exe")

    # AppData candidates without os.getenv
    user_home = Path.home()
    appdata_paths = [
        user_home / "AppData" / "Roaming",
        user_home / "AppData" / "Local",
    ]

    for root in appdata_paths:
        if root.exists():
            search_pattern = str(root / "MetaQuotes" / "Terminal" / "*" / "terminal64.exe")
            candidates += glob.glob(search_pattern)

    for x in candidates:
        if x and os.path.exists(x):
            return x

    return ""


# =============================================================================
# Windows process helpers
# =============================================================================
def _is_terminal_running_windows() -> bool:
    # Faster than scanning the whole tasklist output
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq terminal64.exe"],
            text=True,
            errors="ignore",
        ).lower()
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


def _start_terminal_windows(path: str, portable: bool) -> None:
    if not path or not os.path.exists(path):
        raise RuntimeError(f"MT5 path invalid: {path}")

    cmd = [path]
    if portable:
        cmd.append("/portable")

    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW

    # Important: set cwd to terminal directory (more stable for terminals installed under roaming paths)
    cwd = str(Path(path).resolve().parent)
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        cwd=cwd,
    )


# =============================================================================
# Health check
# =============================================================================
@dataclass(frozen=True)
class Health:
    ok: bool
    reason: str


def _health(cfg: MT5ClientConfig) -> Health:
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

        if cfg.require_correct_account:
            if int(getattr(ai, "login", -1)) != int(cfg.creds.login):
                return Health(False, "wrong_account")

        if not bool(getattr(ai, "trade_allowed", False)):
            return Health(False, "trading_disabled_for_account")

        return Health(True, "ok")
    except Exception:
        logger.error("health exception | tb=%s", traceback.format_exc())
        return Health(False, "health_check_exception")


def _wait_ready(cfg: MT5ClientConfig, timeout_sec: float) -> None:
    deadline = _mono() + float(timeout_sec)
    last = "waiting"
    # Fast readiness loop (no MT5_LOCK while sleeping)
    while _mono() < deadline:
        with MT5_LOCK:
            h = _health(cfg)
        last = h.reason
        if h.ok:
            return
        time.sleep(0.20)
    raise RuntimeError(f"mt5_health_failed:{last}")


# =============================================================================
# Initialize + login (multi-fallback)
# =============================================================================
def _init_and_login(cfg: MT5ClientConfig, mt5_path: str) -> None:
    """
    Two-stage init strategy:
      1) mt5.initialize(login, password, server, ...)
      2) mt5.initialize(...) then mt5.login(...)
    """
    # Ensure clean session (do not hold MT5_LOCK while sleeping)
    with MT5_LOCK:
        try:
            mt5.shutdown()
        except Exception:
            pass
    time.sleep(0.10)

    init_kwargs = {
        "path": str(mt5_path) if mt5_path else None,
        "portable": bool(cfg.portable),
        "timeout": int(cfg.timeout_ms),
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    # Attempt #1: initialize with creds
    with MT5_LOCK:
        ok1 = mt5.initialize(
            login=int(cfg.creds.login),
            password=str(cfg.creds.password),
            server=str(cfg.creds.server),
            **init_kwargs,
        )
    if ok1:
        return

    e1 = _last_error()

    # Attempt #2: initialize then login
    with MT5_LOCK:
        try:
            mt5.shutdown()
        except Exception:
            pass
    time.sleep(0.10)

    with MT5_LOCK:
        ok2 = mt5.initialize(**init_kwargs)
    if not ok2:
        e2 = _last_error()
        raise RuntimeError(f"mt5.initialize failed: {e2} | first_try={e1}")

    with MT5_LOCK:
        logged = mt5.login(
            login=int(cfg.creds.login),
            password=str(cfg.creds.password),
            server=str(cfg.creds.server),
            timeout=int(cfg.timeout_ms),
        )
    if not logged:
        e3 = _last_error()
        with MT5_LOCK:
            try:
                mt5.shutdown()
            except Exception:
                pass
        raise RuntimeError(f"mt5.login failed: {e3} | first_try={e1}")


def _is_ipc_timeout(cfg: MT5ClientConfig, exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(m in msg for m in cfg.ipc_markers):
        return True
    code, last = _last_error()
    if code == -10005:
        return True
    if any(m in str(last).lower() for m in cfg.ipc_markers):
        return True
    return False


# =============================================================================
# Default config loader (ENV for credentials only)
# =============================================================================
def _default_config_from_env() -> MT5ClientConfig:
    try:
        from config_xau import get_config_from_env as _get_cfg  # lazy import
        cfg = _get_cfg()
    except Exception:
        try:
            from config_btc import get_config_from_env as _get_cfg  # lazy import
            cfg = _get_cfg()
        except Exception as exc:
            raise RuntimeError("Unable to build MT5 config from env-backed config files") from exc

    creds = MT5Credentials(
        login=int(getattr(cfg, "login", 0) or 0),
        password=str(getattr(cfg, "password", "") or ""),
        server=str(getattr(cfg, "server", "") or ""),
    )

    return MT5ClientConfig(
        creds=creds,
        mt5_path=getattr(cfg, "mt5_path", None),
        portable=bool(getattr(cfg, "mt5_portable", False)),
        autostart=bool(getattr(cfg, "mt5_autostart", True)),
        timeout_ms=int(getattr(cfg, "mt5_timeout_ms", 90_000) or 90_000),
        require_path_on_windows=False,
    )


# =============================================================================
# Public API
# =============================================================================
def ensure_mt5(cfg: Optional[MT5ClientConfig] = None) -> mt5:
    """
    Deterministic MT5 initializer.
    - Single-instance safe.
    - IPC-timeout hardened (optional taskkill on Windows).
    - All mt5.* calls are protected by MT5_LOCK.
    """
    global _initialized, _last_health_reason

    if cfg is None:
        cfg = _default_config_from_env()

    _validate_cfg(cfg)
    _setup_logger(cfg)
    _acquire_single_instance_lock(cfg)

    # Resolve terminal path
    mt5_path = ""
    if _is_windows():
        mt5_path = _resolve_mt5_path_windows(cfg.mt5_path)
        if cfg.require_path_on_windows and (not mt5_path or not os.path.exists(mt5_path)):
            raise RuntimeError("MT5 terminal64.exe path required on Windows. Provide MT5ClientConfig.mt5_path.")
    else:
        mt5_path = cfg.mt5_path or ""

    # Fast path: already initialized + healthy
    with MT5_LOCK:
        if _initialized:
            h = _health(cfg)
            _last_health_reason = h.reason
            if h.ok:
                return mt5
            _throttled_health_log(cfg, h.reason)
            try:
                mt5.shutdown()
            except Exception:
                pass
            _initialized = False

    last_exc: Optional[Exception] = None

    with _INIT_LOCK:
        for attempt in range(int(cfg.max_retries)):
            try:
                # Autostart terminal on Windows (outside MT5_LOCK)
                if _is_windows() and cfg.autostart:
                    if not _is_terminal_running_windows():
                        _start_terminal_windows(mt5_path, portable=cfg.portable)
                        if float(cfg.start_wait_sec) > 0:
                            time.sleep(float(cfg.start_wait_sec))
                    else:
                        # minimal settle time on first attempt
                        if attempt == 0 and float(cfg.start_wait_sec) > 0:
                            time.sleep(min(0.75, float(cfg.start_wait_sec)))

                _init_and_login(cfg, mt5_path)
                _wait_ready(cfg, timeout_sec=float(cfg.ready_timeout_sec))

                with MT5_LOCK:
                    _initialized = True
                    _last_health_reason = "ok"

                return mt5

            except Exception as exc:
                last_exc = exc

                # IPC timeout recovery on Windows
                is_ipc = _is_windows() and bool(cfg.taskkill_on_ipc_timeout) and _is_ipc_timeout(cfg, exc)
                if is_ipc:
                    logger.error("IPC timeout -> taskkill terminal64.exe + retry | err=%s | last_error=%s", exc, _last_error())
                    with MT5_LOCK:
                        try:
                            mt5.shutdown()
                        except Exception:
                            pass
                    _taskkill_terminal_windows()
                    if cfg.autostart:
                        try:
                            _start_terminal_windows(mt5_path, portable=cfg.portable)
                            if float(cfg.start_wait_sec) > 0:
                                time.sleep(float(cfg.start_wait_sec))
                        except Exception as exc2:
                            logger.error("MT5 restart failed: %s | tb=%s", exc2, traceback.format_exc())

                with MT5_LOCK:
                    try:
                        mt5.shutdown()
                    except Exception:
                        pass
                    _initialized = False
                    _last_health_reason = f"init_failed:{type(exc).__name__}"

                logger.error(
                    "ensure_mt5 attempt %d/%d failed: %s | last_error=%s | tb=%s",
                    attempt + 1,
                    int(cfg.max_retries),
                    exc,
                    _last_error(),
                    traceback.format_exc(),
                )

                if attempt < int(cfg.max_retries) - 1:
                    _sleep_backoff(float(cfg.retry_backoff_base_sec), attempt, cap=float(cfg.backoff_cap_sec))

        raise RuntimeError(f"MT5 initialization failed after {cfg.max_retries} attempts: {last_exc}")


def shutdown_mt5() -> None:
    global _initialized, _last_health_reason
    with _INIT_LOCK:
        with MT5_LOCK:
            try:
                mt5.shutdown()
            except Exception as exc:
                logger.error("MT5 shutdown error: %s | tb=%s", exc, traceback.format_exc())
            _initialized = False
            _last_health_reason = "shutdown"
    _release_single_instance_lock()


def mt5_status() -> Tuple[bool, str]:
    with MT5_LOCK:
        return bool(_initialized), str(_last_health_reason)


atexit.register(shutdown_mt5)

__all__ = ["ensure_mt5", "shutdown_mt5", "mt5_status", "MT5_LOCK", "MT5ClientConfig", "MT5Credentials", "mt5"]
