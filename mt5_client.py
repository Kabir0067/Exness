# mt5_client.py (SENIOR / PRODUCTION / ENV-CREDS ONLY / IPC-HARDENED / SINGLE-INSTANCE SAFE)
from __future__ import annotations

import atexit
import concurrent.futures
import glob
import logging
import os
import queue
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

# Protect init/shutdown sequence
_INIT_LOCK = threading.Lock()

# Internal state
_initialized: bool = False
_ever_initialized: bool = False
_last_health_reason: str = "not_initialized"
_last_health_log_ts_mono: float = 0.0
_last_mt5_repair_ts_mono: float = 0.0

# MT5 async call queue (single-thread dispatcher).
_mt5_async_queue: "queue.Queue[Optional[tuple[str, tuple, dict, concurrent.futures.Future]]]" = queue.Queue(maxsize=2048)
_mt5_async_thread: Optional[threading.Thread] = None
_mt5_async_stop = threading.Event()
_mt5_async_boot_lock = threading.Lock()

# Single-instance file lock state
_lock_guard = threading.Lock()
_lock_fp = None
_lock_acquired = False

# Auth failure throttling
_auth_block_until_mono: float = 0.0
_auth_block_reason: str = ""

# Non-retriable runtime throttling
_non_retriable_block_until_mono: float = 0.0
_non_retriable_block_reason: str = ""


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
    creds: MT5Credentials

    mt5_path: Optional[str] = None
    portable: bool = False
    autostart: bool = True
    require_path_on_windows: bool = True

    single_instance_lock: bool = True
    taskkill_on_ipc_timeout: bool = True
    timeout_ms: int = 300_000
    start_wait_sec: float = 5.0
    ready_timeout_sec: float = 120.0
    max_retries: int = 6
    retry_backoff_base_sec: float = 1.2
    backoff_cap_sec: float = 12.0

    auth_fail_cooldown_sec: float = 600.0
    non_retriable_cooldown_sec: float = 20.0

    # NOTE: by default we still keep errors minimal, but allow WARNING for health/autorepair hints.
    log_level: int = logging.ERROR
    log_max_bytes: int = 5_242_880
    log_backups: int = 5
    health_log_throttle_sec: float = 15.0

    ipc_markers: Tuple[str, ...] = ("ipc timeout", "-10005", "(-10005")
    require_correct_account: bool = True


class MT5AuthError(RuntimeError):
    """Hard error: wrong login/password/server (do not retry in a tight loop)."""


# =============================================================================
# Logging
# =============================================================================
logger = logging.getLogger("mt5")
logger.propagate = False


def _setup_logger(cfg: MT5ClientConfig) -> None:
    # CRITICAL FIX: allow warnings even if cfg.log_level is ERROR (health/autorepair messages)
    eff_level = min(int(cfg.log_level), logging.WARNING)
    logger.setLevel(eff_level)

    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return

    fh = RotatingFileHandler(
        filename=str(get_log_path("mt5.log")),
        maxBytes=int(cfg.log_max_bytes),
        backupCount=int(cfg.log_backups),
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(eff_level)
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
    with MT5_LOCK:
        try:
            code, msg = mt5.last_error()
            return int(code), str(msg)
        except Exception:
            return -99999, "unknown_mt5_error"


def _normalize_env_str(s: str) -> str:
    x = (s or "").strip()
    if len(x) >= 2 and ((x[0] == x[-1] == '"') or (x[0] == x[-1] == "'")):
        x = x[1:-1].strip()
    return x.replace("\r", "").replace("\n", "")


def _sleep_backoff(base: float, attempt: int, cap: float) -> None:
    a = int(attempt)
    delay = float(base) * (2.0 ** a)
    delay = min(float(cap), delay) + min(0.25, 0.03 * a)
    time.sleep(delay)


def _throttled_health_log(cfg: MT5ClientConfig, reason: str) -> None:
    global _last_health_log_ts_mono
    now = _mono()
    if now - _last_health_log_ts_mono >= float(cfg.health_log_throttle_sec):
        logger.warning("Fast path health failed (attempting auto-heal): %s", reason)
        _last_health_log_ts_mono = now


def _validate_cfg(cfg: MT5ClientConfig) -> None:
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
    if float(cfg.non_retriable_cooldown_sec) < 0:
        raise RuntimeError("non_retriable_cooldown_sec must be >= 0")


def _is_auth_failed(err: Tuple[int, str]) -> bool:
    code, msg = int(err[0]), str(err[1]).lower()
    return bool(code == -6 or "authorization failed" in msg or "invalid account" in msg)


def _mt5_shutdown_silent() -> None:
    with MT5_LOCK:
        try:
            mt5.shutdown()
        except Exception:
            pass


def _mt5_direct_call(method_name: str, *args, **kwargs):
    fn = getattr(mt5, str(method_name), None)
    if not callable(fn):
        raise AttributeError(f"mt5.{method_name} not callable")
    with MT5_LOCK:
        return fn(*args, **kwargs)


def _looks_like_transport_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    if "mt5_async_queue_full" in msg:
        return True
    if "timeout" in msg:
        return True
    if "ipc" in msg or "-10005" in msg:
        return True
    if "no connection" in msg or "connection" in msg:
        return True
    return False


def _repair_mt5_once(throttle_sec: float = 0.5) -> bool:
    global _last_mt5_repair_ts_mono
    now = _mono()
    if (now - _last_mt5_repair_ts_mono) < max(0.0, float(throttle_sec)):
        return False
    _last_mt5_repair_ts_mono = now
    try:
        ensure_mt5()
        return True
    except Exception:
        return False


def _mt5_async_loop() -> None:
    while not _mt5_async_stop.is_set():
        try:
            item = _mt5_async_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            if item is None:
                break

            method_name, args, kwargs, fut = item
            if fut.cancelled():
                continue

            fn = getattr(mt5, str(method_name), None)
            if not callable(fn):
                fut.set_exception(AttributeError(f"mt5.{method_name} not callable"))
                continue

            try:
                with MT5_LOCK:
                    out = fn(*args, **kwargs)
                fut.set_result(out)
            except Exception as exc:
                fut.set_exception(exc)
        finally:
            try:
                _mt5_async_queue.task_done()
            except Exception:
                pass


def _ensure_mt5_async_thread() -> None:
    global _mt5_async_thread
    if _mt5_async_thread and _mt5_async_thread.is_alive():
        return

    with _mt5_async_boot_lock:
        if _mt5_async_thread and _mt5_async_thread.is_alive():
            return
        _mt5_async_stop.clear()
        _mt5_async_thread = threading.Thread(
            target=_mt5_async_loop,
            name="mt5-async-dispatch",
            daemon=True,
        )
        _mt5_async_thread.start()


def _stop_mt5_async_thread() -> None:
    global _mt5_async_thread
    try:
        _mt5_async_stop.set()
        try:
            _mt5_async_queue.put_nowait(None)
        except Exception:
            pass
        th = _mt5_async_thread
        if th and th.is_alive():
            th.join(timeout=1.0)
    finally:
        _mt5_async_thread = None
        _mt5_async_stop.clear()


def mt5_async_submit(
    method_name: str,
    *args,
    ensure_ready: bool = False,
    **kwargs,
) -> concurrent.futures.Future:
    """
    Submit an MT5 API call to the dedicated async dispatcher thread.

    This preserves MT5 thread safety while removing lock contention from caller
    threads that only need queued execution.
    """
    if ensure_ready:
        ensure_mt5()

    # Re-entrant safety: if called from dispatcher thread, execute inline.
    cur = threading.current_thread()
    if _mt5_async_thread is not None and cur is _mt5_async_thread:
        fut_inline: concurrent.futures.Future = concurrent.futures.Future()
        try:
            fn = getattr(mt5, str(method_name), None)
            if not callable(fn):
                raise AttributeError(f"mt5.{method_name} not callable")
            with MT5_LOCK:
                fut_inline.set_result(fn(*args, **kwargs))
        except Exception as exc:
            fut_inline.set_exception(exc)
        return fut_inline

    _ensure_mt5_async_thread()

    fut: concurrent.futures.Future = concurrent.futures.Future()
    item = (str(method_name), tuple(args), dict(kwargs), fut)
    try:
        _mt5_async_queue.put_nowait(item)
    except queue.Full:
        # Fallback path: do direct guarded call instead of dropping request.
        try:
            fut.set_result(_mt5_direct_call(str(method_name), *args, **kwargs))
        except Exception as exc:
            fut.set_exception(exc)
    return fut


def mt5_async_call(
    method_name: str,
    *args,
    timeout: float = 1.0,
    default=None,
    ensure_ready: bool = False,
    raise_on_error: bool = False,
    retries: int = 1,
    repair_on_transport_error: bool = True,
    direct_fallback: bool = True,
    **kwargs,
):
    """
    Synchronous wait helper over `mt5_async_submit`.

    Returns `default` on timeout/error unless `raise_on_error=True`.
    """
    wait_s = max(0.01, float(timeout))
    attempts = max(1, int(retries) + 1)
    last_exc: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            fut = mt5_async_submit(
                method_name,
                *args,
                ensure_ready=ensure_ready,
                **kwargs,
            )
            return fut.result(timeout=wait_s)
        except concurrent.futures.TimeoutError as exc:
            last_exc = exc
            if repair_on_transport_error:
                _repair_mt5_once()
            if attempt < (attempts - 1):
                continue
            if direct_fallback:
                try:
                    return _mt5_direct_call(method_name, *args, **kwargs)
                except Exception as exc2:
                    last_exc = exc2
        except Exception as exc:
            last_exc = exc
            if repair_on_transport_error and _looks_like_transport_error(exc):
                _repair_mt5_once()
            if attempt < (attempts - 1):
                continue
            if direct_fallback:
                try:
                    return _mt5_direct_call(method_name, *args, **kwargs)
                except Exception as exc2:
                    last_exc = exc2

    if raise_on_error and last_exc is not None:
        raise last_exc
    return default


# =============================================================================
# Single-instance lock
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
    try:
        if not _lock_fp:
            return
        info = f"pid={os.getpid()} ts={int(time.time())}\n"
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
# Windows helpers (NO ENV)
# =============================================================================
def _resolve_mt5_path_windows(mt5_path: Optional[str]) -> str:
    if mt5_path and os.path.exists(mt5_path):
        return mt5_path

    candidates: list[str] = []
    for base in (r"C:\Program Files", r"C:\Program Files (x86)"):
        candidates += glob.glob(base + r"\MetaTrader 5*\terminal64.exe")
        candidates += glob.glob(base + r"\MetaQuotes*\terminal64.exe")
        candidates += glob.glob(base + r"\Exness*\terminal64.exe")

    user_home = Path.home()
    for root in (user_home / "AppData" / "Roaming", user_home / "AppData" / "Local"):
        if root.exists():
            candidates += glob.glob(str(root / "MetaQuotes" / "Terminal" / "*" / "terminal64.exe"))

    for x in candidates:
        if x and os.path.exists(x):
            return x
    return ""


def _is_terminal_running_windows() -> bool:
    try:
        out = subprocess.check_output(["tasklist", "/FI", "IMAGENAME eq terminal64.exe"], text=True, errors="ignore").lower()
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

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    cwd = str(Path(path).resolve().parent)

    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        cwd=cwd,
    )


# =============================================================================
# Health check (LOCKED)
# =============================================================================
@dataclass(frozen=True)
class Health:
    ok: bool
    reason: str


_NON_RETRIABLE_HEALTH_REASONS = {
    "algo_trading_disabled_in_terminal",
    "trading_disabled_for_account",
    "wrong_account",
}


def _health(cfg: MT5ClientConfig) -> Health:
    # CRITICAL FIX: always locked (even if called from elsewhere)
    with MT5_LOCK:
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

            if cfg.require_correct_account and int(getattr(ai, "login", -1)) != int(cfg.creds.login):
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

    while _mono() < deadline:
        h = _health(cfg)
        last = h.reason
        if h.ok:
            return
        if last in _NON_RETRIABLE_HEALTH_REASONS:
            break
        time.sleep(0.20)

    hint = ""
    if last == "algo_trading_disabled_in_terminal":
        hint = " Enable Algo Trading in MT5: Tools -> Options -> Expert Advisors -> Allow Algo Trading (or press AutoTrading button on toolbar)."
    raise RuntimeError(f"mt5_health_failed:{last}{hint}")


def _is_non_retriable_runtime_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return bool("mt5_health_failed:" in msg and any(r in msg for r in _NON_RETRIABLE_HEALTH_REASONS))


# =============================================================================
# Initialize + login (multi-fallback)
# =============================================================================
def _init_and_login(cfg: MT5ClientConfig, mt5_path: str) -> None:
    _mt5_shutdown_silent()
    time.sleep(0.10)

    init_kwargs = {
        "portable": bool(cfg.portable),
        "timeout": int(cfg.timeout_ms),
    }
    if mt5_path:
        init_kwargs["path"] = str(mt5_path)

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

    # Attempt #2: initialize without creds then login
    _mt5_shutdown_silent()
    time.sleep(0.10)

    with MT5_LOCK:
        ok2 = mt5.initialize(**init_kwargs)
    if not ok2:
        e2 = _last_error()
        if _is_auth_failed(e1) or _is_auth_failed(e2):
            raise MT5AuthError(
                f"authorization_failed: login={cfg.creds.login} server={cfg.creds.server} path={mt5_path} | "
                f"init_with_creds={e1} init_no_creds={e2}"
            )
        raise RuntimeError(f"mt5.initialize failed: {e2} | first_try={e1}")

    with MT5_LOCK:
        logged = mt5.login(
            login=int(cfg.creds.login),
            password=str(cfg.creds.password),
            server=str(cfg.creds.server),
            timeout=int(cfg.timeout_ms),
        )
    if logged:
        return

    e3 = _last_error()
    _mt5_shutdown_silent()
    if _is_auth_failed(e1) or _is_auth_failed(e3):
        raise MT5AuthError(
            f"authorization_failed: login={cfg.creds.login} server={cfg.creds.server} path={mt5_path} | init_with_creds={e1} login={e3}"
        )
    raise RuntimeError(f"mt5.login failed: {e3} | first_try={e1}")


def _is_ipc_timeout(cfg: MT5ClientConfig, exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(m in msg for m in cfg.ipc_markers):
        return True
    code, last = _last_error()
    if code == -10005:
        return True
    last_s = str(last).lower()
    return any(m in last_s for m in cfg.ipc_markers)


# =============================================================================
# Default config loader (ENV for credentials only)
# =============================================================================
def _default_config_from_env() -> MT5ClientConfig:
    # CRITICAL FIX: support both module layouts
    try:
        from core.config import get_config_from_env as _get_cfg  # type: ignore
    except Exception:
        try:
            from config import get_config_from_env as _get_cfg  # type: ignore
        except Exception as exc:
            raise RuntimeError("Unable to build MT5 config from env-backed config files") from exc

    cfg = _get_cfg()

    creds = MT5Credentials(
        login=int(getattr(cfg, "login", 0) or 0),
        password=_normalize_env_str(str(getattr(cfg, "password", "") or "")),
        server=_normalize_env_str(str(getattr(cfg, "server", "") or "")),
    )

    return MT5ClientConfig(
        creds=creds,
        mt5_path=getattr(cfg, "mt5_path", None),
        portable=bool(getattr(cfg, "mt5_portable", False)),
        autostart=bool(getattr(cfg, "mt5_autostart", True)),
        timeout_ms=int(getattr(cfg, "mt5_timeout_ms", 300_000) or 300_000),
        require_path_on_windows=False,
        auth_fail_cooldown_sec=float(getattr(cfg, "mt5_auth_cooldown_sec", 600.0) or 600.0),
        non_retriable_cooldown_sec=float(getattr(cfg, "mt5_non_retriable_cooldown_sec", 20.0) or 20.0),
        ready_timeout_sec=float(getattr(cfg, "mt5_ready_timeout_sec", 120.0) or 120.0),
    )


# =============================================================================
# Public API
# =============================================================================
def ensure_mt5(cfg: Optional[MT5ClientConfig] = None) -> mt5:
    global _initialized, _ever_initialized, _last_health_reason
    global _auth_block_until_mono, _auth_block_reason
    global _non_retriable_block_until_mono, _non_retriable_block_reason

    if cfg is None:
        cfg = _default_config_from_env()

    _validate_cfg(cfg)
    _setup_logger(cfg)

    now = _mono()
    if now < _auth_block_until_mono:
        raise MT5AuthError(f"auth_blocked:{_auth_block_reason}")

    if now < _non_retriable_block_until_mono:
        cached = str(_non_retriable_block_reason or "mt5_health_failed:non_retriable_cached")
        with MT5_LOCK:
            _last_health_reason = f"blocked_cooldown:{cached}"
        raise RuntimeError(cached)

    _acquire_single_instance_lock(cfg)

    mt5_path = ""
    if _is_windows():
        mt5_path = _resolve_mt5_path_windows(cfg.mt5_path)
        if cfg.require_path_on_windows and (not mt5_path or not os.path.exists(mt5_path)):
            raise RuntimeError("MT5 terminal64.exe path required on Windows. Provide MT5ClientConfig.mt5_path.")
    else:
        mt5_path = cfg.mt5_path or ""

    # Fast path: already initialized and healthy
    with MT5_LOCK:
        if _initialized:
            h = _health(cfg)
            _last_health_reason = h.reason
            if h.ok:
                return mt5

    if _ever_initialized and _last_health_reason not in ("ok", "not_initialized"):
        _throttled_health_log(cfg, _last_health_reason)

    # If terminal is zombie (IPC dead), kill it (Windows) before retrying.
    if _is_windows() and cfg.taskkill_on_ipc_timeout and _ever_initialized and _is_terminal_running_windows():
        h = _health(cfg)
        if h.reason == "terminal_info_none":
            logger.warning("Detecting zombie terminal (terminal_info_none). Forcing taskkill before restart.")
            _mt5_shutdown_silent()
            _taskkill_terminal_windows()
        else:
            _mt5_shutdown_silent()
    else:
        _mt5_shutdown_silent()

    with MT5_LOCK:
        _initialized = False

    last_exc: Optional[Exception] = None

    with _INIT_LOCK:
        for attempt in range(int(cfg.max_retries)):
            try:
                if _is_windows() and cfg.autostart:
                    if not _is_terminal_running_windows():
                        _start_terminal_windows(mt5_path, portable=cfg.portable)
                        if float(cfg.start_wait_sec) > 0:
                            time.sleep(float(cfg.start_wait_sec))
                    elif attempt == 0 and float(cfg.start_wait_sec) > 0:
                        time.sleep(min(0.75, float(cfg.start_wait_sec)))

                _init_and_login(cfg, mt5_path)
                _wait_ready(cfg, timeout_sec=float(cfg.ready_timeout_sec))

                with MT5_LOCK:
                    _initialized = True
                    _last_health_reason = "ok"
                _ever_initialized = True

                _non_retriable_block_until_mono = 0.0
                _non_retriable_block_reason = ""
                return mt5

            except MT5AuthError as exc:
                last_exc = exc
                _auth_block_reason = str(exc)
                _auth_block_until_mono = _mono() + float(cfg.auth_fail_cooldown_sec)

                _mt5_shutdown_silent()
                with MT5_LOCK:
                    _initialized = False
                    _last_health_reason = "auth_failed"

                logger.error("ensure_mt5 AUTH FAILED (cooldown %.0fs) | err=%s", float(cfg.auth_fail_cooldown_sec), exc)
                raise

            except Exception as exc:
                last_exc = exc

                is_ipc = _is_windows() and bool(cfg.taskkill_on_ipc_timeout) and _is_ipc_timeout(cfg, exc)
                if is_ipc:
                    logger.error("IPC timeout -> taskkill terminal64.exe + retry | err=%s | last_error=%s", exc, _last_error())
                    _mt5_shutdown_silent()
                    _taskkill_terminal_windows()
                    if cfg.autostart:
                        try:
                            _start_terminal_windows(mt5_path, portable=cfg.portable)
                            if float(cfg.start_wait_sec) > 0:
                                time.sleep(float(cfg.start_wait_sec))
                        except Exception as exc2:
                            logger.error("MT5 restart failed: %s | tb=%s", exc2, traceback.format_exc())

                _mt5_shutdown_silent()

                with MT5_LOCK:
                    _initialized = False
                    exc_txt = str(exc).replace("\n", " ").strip()
                    _last_health_reason = f"init_failed:{(exc_txt[:240] if exc_txt else type(exc).__name__)}"

                if _is_non_retriable_runtime_error(exc):
                    _non_retriable_block_reason = str(exc)
                    _non_retriable_block_until_mono = _mono() + float(cfg.non_retriable_cooldown_sec)
                    logger.warning("ensure_mt5 non-retriable failure: %s", exc)
                    raise

                logger.error(
                    "ensure_mt5 attempt %d/%d failed: %s | tb=%s",
                    attempt + 1,
                    int(cfg.max_retries),
                    exc,
                    traceback.format_exc(),
                )

                if attempt < int(cfg.max_retries) - 1:
                    _sleep_backoff(float(cfg.retry_backoff_base_sec), attempt, cap=float(cfg.backoff_cap_sec))

        raise RuntimeError(f"MT5 initialization failed after {cfg.max_retries} attempts: {last_exc}")


def shutdown_mt5() -> None:
    global _initialized, _last_health_reason
    global _non_retriable_block_until_mono, _non_retriable_block_reason

    with _INIT_LOCK:
        _stop_mt5_async_thread()
        _mt5_shutdown_silent()
        with MT5_LOCK:
            _initialized = False
            _last_health_reason = "shutdown"
            _non_retriable_block_until_mono = 0.0
            _non_retriable_block_reason = ""

    _release_single_instance_lock()


def mt5_status() -> Tuple[bool, str]:
    with MT5_LOCK:
        return bool(_initialized), str(_last_health_reason)


atexit.register(shutdown_mt5)

__all__ = [
    "ensure_mt5",
    "shutdown_mt5",
    "mt5_status",
    "mt5_async_submit",
    "mt5_async_call",
    "MT5_LOCK",
    "MT5ClientConfig",
    "MT5Credentials",
    "MT5AuthError",
    "mt5",
]
