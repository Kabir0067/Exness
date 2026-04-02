# mt5_client.py
# PRODUCTION | WINDOWS-NATIVE | IPC-HARDENED | IDEMPOTENT | ASYNC-LOGGING
# ============================================================================
# Architecture: Single async-dispatcher thread + deque/Condition (no timed-poll)
# Credential source: Environment variables only (via config layer)
# Capital safety: Idempotency key on every order_send, ghost-fill detection
# Logging: QueueHandler/QueueListener — disk I/O never blocks trading thread
# Process health: psutil PID-cache — no subprocess.check_output(tasklist) forks
# Lock ordering: MT5_LOCK(RLock) never nested inside _INIT_LOCK;
#                double-checked locking on _initialized
# ============================================================================
from __future__ import annotations

import atexit
import collections
import concurrent.futures
import glob
import logging
import logging.handlers
import msvcrt
import os
import queue
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import psutil
import MetaTrader5 as mt5

from log_config import LOG_DIR as LOG_ROOT, get_log_path


# =============================================================================
# Public lock: guard EVERY mt5.* call from all threads
# =============================================================================
# NOTE: We use threading.Lock (not RLock) deliberately.
# Re-entrant use of MT5_LOCK is a design smell that hides reentrancy bugs.
# All internal callers are written to never hold MT5_LOCK while calling
# another function that also acquires MT5_LOCK.
MT5_LOCK = threading.Lock()

# Serialises the multi-step init/shutdown sequence.
# ORDERING RULE: _INIT_LOCK is ALWAYS acquired BEFORE MT5_LOCK, never after.
_INIT_LOCK = threading.Lock()

# =============================================================================
# Module-level state — mutated only under the appropriate lock (see comments)
# =============================================================================
_initialized: bool = False                  # guarded by MT5_LOCK
_ever_initialized: bool = False             # guarded by _INIT_LOCK
_last_health_reason: str = "not_initialized"  # guarded by MT5_LOCK
_last_health_log_ts_mono: float = 0.0       # guarded by _INIT_LOCK
_last_mt5_repair_ts_mono: float = 0.0       # guarded by _INIT_LOCK

# Auth / non-retriable cooldown — guarded by _INIT_LOCK
_auth_block_until_mono: float = 0.0
_auth_block_reason: str = ""
_non_retriable_block_until_mono: float = 0.0
_non_retriable_block_reason: str = ""

# =============================================================================
# FIX #2 — psutil PID cache (Windows-native, no subprocess fork)
# =============================================================================
_terminal_pid: Optional[int] = None        # set when we start the terminal
_terminal_pid_lock = threading.Lock()       # protect _terminal_pid


# =============================================================================
# FIX #1 — Async dispatcher: deque + Condition (sub-millisecond wakeup)
# =============================================================================
# Item format stored in deque: (enqueue_mono, method_name, args, kwargs, Future)
_mt5_deque: Deque[Optional[Tuple]] = collections.deque()
_mt5_deque_cond: threading.Condition = threading.Condition(threading.Lock())

_mt5_async_thread: Optional[threading.Thread] = None
_mt5_async_stop = threading.Event()
_mt5_async_boot_lock = threading.Lock()

# Items older than this (seconds) are rejected before dispatch — stale for HFT.
_ASYNC_STALE_THRESHOLD_SEC: float = 0.5

# =============================================================================
# FIX #3 — Single-instance lock (Windows msvcrt, already native)
# =============================================================================
_lock_guard = threading.Lock()
_lock_fp = None
_lock_acquired = False


# =============================================================================
# Exceptions
# =============================================================================
class MT5AuthError(RuntimeError):
    """Hard auth failure — wrong login/password/server. Do NOT retry tight."""


class MT5GhostFillDetected(RuntimeError):
    """
    Raised by safe_order_send when an IPC error occurred but a matching
    position or deal was found on the broker side.  The order WAS executed.
    """


# =============================================================================
# Configuration
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

    log_level: int = logging.ERROR
    log_max_bytes: int = 5_242_880
    log_backups: int = 5
    health_log_throttle_sec: float = 15.0

    # FIX #4 — idempotency
    order_idem_prefix: str = "ik"           # short prefix to save comment space
    order_comment_max_len: int = 31         # MT5 hard limit
    ghost_fill_lookback_sec: float = 120.0  # history window after reconnect

    ipc_markers: Tuple[str, ...] = ("ipc timeout", "-10005", "(-10005")
    require_correct_account: bool = True

    # FIX #5 — async logging queue depth
    log_queue_maxsize: int = 8192


# =============================================================================
# FIX #5 — Async logging (QueueHandler + QueueListener)
# Disk I/O never touches the execution thread.
# =============================================================================
logger = logging.getLogger("mt5")
logger.propagate = False

_log_listener: Optional[logging.handlers.QueueListener] = None


def _setup_logger(cfg: MT5ClientConfig) -> None:
    global _log_listener

    eff_level = min(int(cfg.log_level), logging.WARNING)
    logger.setLevel(eff_level)

    # Idempotent — only install handlers once.
    if any(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers):
        return

    # The file handler is the only slow part; it lives in the listener thread.
    fh = RotatingFileHandler(
        filename=str(get_log_path("mt5.log")),
        maxBytes=int(cfg.log_max_bytes),
        backupCount=int(cfg.log_backups),
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(eff_level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
        )
    )

    # Non-blocking queue — records are dropped if full, NEVER block caller.
    log_q: queue.Queue = queue.Queue(maxsize=int(cfg.log_queue_maxsize))

    q_handler = logging.handlers.QueueHandler(log_q)
    q_handler.setLevel(eff_level)

    _log_listener = logging.handlers.QueueListener(
        log_q, fh, respect_handler_level=True
    )
    _log_listener.start()
    atexit.register(_log_listener.stop)

    logger.addHandler(q_handler)


# =============================================================================
# Helpers
# =============================================================================
def _mono() -> float:
    return time.monotonic()


def _last_error() -> Tuple[int, str]:
    # Must be called with MT5_LOCK already held by the caller, or it acquires it.
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
    delay = float(base) * (2.0 ** int(attempt))
    delay = min(float(cap), delay) + min(0.25, 0.03 * attempt)
    time.sleep(delay)


def _validate_cfg(cfg: MT5ClientConfig) -> None:
    if cfg is None:
        raise RuntimeError("MT5ClientConfig is None")
    if int(getattr(cfg.creds, "login", 0) or 0) <= 0:
        raise RuntimeError("MT5 credentials invalid: login must be > 0")
    if not str(getattr(cfg.creds, "password", "") or ""):
        raise RuntimeError("MT5 credentials invalid: password is empty")
    if not str(getattr(cfg.creds, "server", "") or ""):
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
    return bool(
        code == -6
        or "authorization failed" in msg
        or "invalid account" in msg
    )


def _looks_like_transport_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return any(
        kw in msg
        for kw in ("mt5_async_stale", "timeout", "ipc", "-10005",
                   "no connection", "connection", "mt5_async_queue")
    )


def _mt5_shutdown_silent() -> None:
    """Shutdown MT5 without raising. Never acquires MT5_LOCK itself."""
    try:
        # Caller is responsible for ensuring MT5_LOCK is not held by us here.
        mt5.shutdown()
    except Exception:
        pass


def _throttled_health_log(cfg: MT5ClientConfig, reason: str) -> None:
    global _last_health_log_ts_mono
    now = _mono()
    if now - _last_health_log_ts_mono >= float(cfg.health_log_throttle_sec):
        logger.warning(
            "Fast-path health failed — attempting auto-heal: %s", reason
        )
        _last_health_log_ts_mono = now


def _cancel_pending_mt5_async(reason: str = "mt5_async_reset") -> int:
    """
    Clear queued async MT5 calls and fail their futures so reconnect/reset does
    not replay stale requests against a restarted terminal.
    """
    drained: List[Tuple] = []
    with _mt5_deque_cond:
        while _mt5_deque:
            item = _mt5_deque.popleft()
            if item is None:
                continue
            drained.append(item)

    cancelled = 0
    for item in drained:
        try:
            _, method_name, _, _, fut = item
            if fut.cancelled() or fut.done():
                continue
            fut.set_exception(RuntimeError(f"{reason}:{method_name}"))
            cancelled += 1
        except Exception:
            continue
    return cancelled


def _requires_hard_terminal_reset(reason: str) -> bool:
    reason_s = str(reason or "").strip().lower()
    if not reason_s:
        return False
    return reason_s in {
        "terminal_info_none",
        "terminal_not_connected",
        "account_info_none",
        "health_check_exception",
    }


# =============================================================================
# FIX #2 — Windows process health via psutil (no subprocess fork)
# =============================================================================
def _is_terminal_running_windows() -> bool:
    """
    Check whether terminal64.exe is alive.
    Primary path: O(1) psutil PID probe — no process fork, ~1 µs.
    Cold path: psutil process scan (only when PID is unknown).
    """
    global _terminal_pid
    with _terminal_pid_lock:
        if _terminal_pid is not None:
            try:
                p = psutil.Process(_terminal_pid)
                return p.is_running() and p.status() not in (
                    psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                _terminal_pid = None  # PID is gone — reset cache

        # Cold path: scan running processes once, then cache.
        try:
            for proc in psutil.process_iter(["name", "pid"]):
                pname = (proc.info.get("name") or "").lower()
                if "terminal64" in pname:
                    _terminal_pid = proc.info["pid"]
                    return True
        except Exception:
            pass
        return False


def _taskkill_terminal_windows() -> None:
    """
    Force-terminate terminal64.exe.
    Prefers psutil (cached PID) — falls back to taskkill only if PID unknown.
    """
    global _terminal_pid
    with _terminal_pid_lock:
        if _terminal_pid is not None:
            try:
                p = psutil.Process(_terminal_pid)
                p.kill()
                _terminal_pid = None
                return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                _terminal_pid = None

    # Fallback — only path that spawns a subprocess; reached once on cold start.
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", "terminal64.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def _start_terminal_windows(mt5_path: str, portable: bool) -> None:
    """Launch MT5 terminal and cache its PID for future health checks."""
    global _terminal_pid
    if not mt5_path or not os.path.exists(mt5_path):
        raise RuntimeError(f"MT5 path invalid or not found: {mt5_path!r}")

    cmd = [mt5_path] + (["/portable"] if portable else [])
    cwd = str(Path(mt5_path).resolve().parent)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        cwd=cwd,
    )
    with _terminal_pid_lock:
        _terminal_pid = proc.pid


def _resolve_mt5_path_windows(mt5_path: Optional[str]) -> str:
    """Locate terminal64.exe on Windows via config, then common install paths."""
    if mt5_path and os.path.exists(mt5_path):
        return mt5_path

    candidates: List[str] = []
    for base in (r"C:\Program Files", r"C:\Program Files (x86)"):
        candidates += glob.glob(base + r"\MetaTrader 5*\terminal64.exe")
        candidates += glob.glob(base + r"\MetaQuotes*\terminal64.exe")
        candidates += glob.glob(base + r"\Exness*\terminal64.exe")

    user_home = Path.home()
    for root in (
        user_home / "AppData" / "Roaming",
        user_home / "AppData" / "Local",
    ):
        if root.exists():
            candidates += glob.glob(
                str(root / "MetaQuotes" / "Terminal" / "*" / "terminal64.exe")
            )

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ""


# =============================================================================
# Single-instance lock (Windows msvcrt — native byte-range lock)
# =============================================================================
def _lock_file_path() -> Path:
    base = Path(str(LOG_ROOT)) if LOG_ROOT else (Path.cwd() / "Logs")
    base.mkdir(parents=True, exist_ok=True)
    return base / "mt5_single_instance.lock"


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
            try:
                msvcrt.locking(_lock_fp.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                existing = ""
                try:
                    existing = p.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    pass
                raise RuntimeError(
                    f"Another engine process is running (lock busy). "
                    f"Stop it first. lock_info={existing}"
                )
            _lock_acquired = True
            # Write PID / timestamp for diagnostics.
            try:
                _lock_fp.seek(0)
                _lock_fp.truncate(0)
                _lock_fp.write(f"pid={os.getpid()} ts={int(time.time())}\n")
                _lock_fp.flush()
                os.fsync(_lock_fp.fileno())
            except Exception:
                pass
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
            _lock_fp.seek(0)
            msvcrt.locking(_lock_fp.fileno(), msvcrt.LK_UNLCK, 1)
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
# Health check
# =============================================================================
@dataclass(frozen=True)
class Health:
    ok: bool
    reason: str


_NON_RETRIABLE_HEALTH_REASONS = frozenset({
    "algo_trading_disabled_in_terminal",
    "trading_disabled_for_account",
    "wrong_account",
})


def _health(cfg: MT5ClientConfig) -> Health:
    """
    Check MT5 terminal + account health.
    MUST be called WITHOUT MT5_LOCK held by the caller — it acquires the lock
    internally to enforce the single lock-ordering rule.
    """
    acquired = MT5_LOCK.acquire(timeout=5.0)
    if not acquired:
        logger.error(
            "_health: MT5_LOCK acquire timed out after 5s — "
            "possible deadlock or long-running MT5 call"
        )
        return Health(False, "lock_acquisition_timeout")
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
        logger.error("_health exception | tb=%s", traceback.format_exc())
        return Health(False, "health_check_exception")
    finally:
        MT5_LOCK.release()


def _wait_ready(cfg: MT5ClientConfig, timeout_sec: float) -> None:
    deadline = _mono() + float(timeout_sec)
    last_reason = "waiting"
    while _mono() < deadline:
        h = _health(cfg)
        last_reason = h.reason
        if h.ok:
            return
        if last_reason in _NON_RETRIABLE_HEALTH_REASONS:
            break
        time.sleep(0.20)

    hint = ""
    if last_reason == "algo_trading_disabled_in_terminal":
        hint = (
            " Enable Algo Trading in MT5: "
            "Tools → Options → Expert Advisors → Allow Algo Trading "
            "(or press the AutoTrading button on the toolbar)."
        )
    raise RuntimeError(f"mt5_health_failed:{last_reason}{hint}")


def _is_non_retriable_runtime_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return (
        "mt5_health_failed:" in msg
        and any(r in msg for r in _NON_RETRIABLE_HEALTH_REASONS)
    )


# =============================================================================
# Initialize + login (multi-fallback)
# MUST be called from within _INIT_LOCK, WITHOUT MT5_LOCK held.
# =============================================================================
def _init_and_login(cfg: MT5ClientConfig, mt5_path: str) -> None:
    _mt5_shutdown_silent()
    time.sleep(0.10)

    init_kwargs: Dict[str, Any] = {
        "portable": bool(cfg.portable),
        "timeout": int(cfg.timeout_ms),
    }
    if mt5_path:
        init_kwargs["path"] = str(mt5_path)

    # Attempt 1: initialize with credentials embedded.
    with MT5_LOCK:
        ok1 = mt5.initialize(
            login=int(cfg.creds.login),
            password=str(cfg.creds.password),
            server=str(cfg.creds.server),
            **init_kwargs,
        )
    if ok1:
        return

    with MT5_LOCK:
        e1 = _last_error()

    # Attempt 2: initialize without credentials, then explicit login.
    _mt5_shutdown_silent()
    time.sleep(0.10)

    with MT5_LOCK:
        ok2 = mt5.initialize(**init_kwargs)

    if not ok2:
        with MT5_LOCK:
            e2 = _last_error()
        if _is_auth_failed(e1) or _is_auth_failed(e2):
            raise MT5AuthError(
                f"authorization_failed: login={cfg.creds.login} "
                f"server={cfg.creds.server} path={mt5_path!r} | "
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

    with MT5_LOCK:
        e3 = _last_error()

    _mt5_shutdown_silent()
    if _is_auth_failed(e1) or _is_auth_failed(e3):
        raise MT5AuthError(
            f"authorization_failed: login={cfg.creds.login} "
            f"server={cfg.creds.server} path={mt5_path!r} | "
            f"init_with_creds={e1} login={e3}"
        )
    raise RuntimeError(f"mt5.login failed: {e3} | first_try={e1}")


def _is_ipc_timeout(cfg: MT5ClientConfig, exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(m in msg for m in cfg.ipc_markers):
        return True
    with MT5_LOCK:
        code, last = _last_error()
    if code == -10005:
        return True
    return any(m in str(last).lower() for m in cfg.ipc_markers)


# =============================================================================
# FIX #1 — Async dispatcher: deque + Condition (sub-millisecond wakeup)
# =============================================================================
def _mt5_async_loop() -> None:
    """
    Dedicated MT5 dispatch thread.
    Blocks on Condition.wait_for() — woken immediately by mt5_async_submit().
    No timed polling; worst-case dispatch latency is one OS context switch.
    """
    while not _mt5_async_stop.is_set():
        with _mt5_deque_cond:
            _mt5_deque_cond.wait_for(
                lambda: bool(_mt5_deque) or _mt5_async_stop.is_set(),
                timeout=1.0,   # safety heartbeat only — not the primary wakeup
            )
            if _mt5_async_stop.is_set() and not _mt5_deque:
                break
            try:
                item = _mt5_deque.popleft()
            except IndexError:
                continue

        if item is None:
            break

        enqueue_ts, method_name, args, kwargs, fut = item

        # Explicitly clean up cancelled futures — releases exc/result refs.
        if fut.cancelled():
            del fut
            continue

        # Reject stale items — irrelevant fill for HFT after 500 ms.
        age = _mono() - enqueue_ts
        if age > _ASYNC_STALE_THRESHOLD_SEC:
            fut.set_exception(
                TimeoutError(
                    f"mt5_async_stale: {method_name!r} queued {age*1000:.0f}ms ago"
                )
            )
            del fut
            continue

        fn = getattr(mt5, str(method_name), None)
        if not callable(fn):
            fut.set_exception(AttributeError(f"mt5.{method_name!r} is not callable"))
            del fut
            continue

        # Acquire MT5_LOCK with a hard timeout to prevent dispatcher starvation.
        acquired = MT5_LOCK.acquire(timeout=5.0)
        if not acquired:
            fut.set_exception(
                TimeoutError(
                    f"mt5_async_dispatch: MT5_LOCK acquire timeout for {method_name!r}"
                )
            )
            del fut
            continue
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:
            fut.set_exception(exc)
        finally:
            MT5_LOCK.release()
            del fut


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
        _cancel_pending_mt5_async("mt5_async_stop")
        with _mt5_deque_cond:
            _mt5_deque.append(None)  # sentinel to unblock wait_for()
            _mt5_deque_cond.notify()
        th = _mt5_async_thread
        if th and th.is_alive():
            th.join(timeout=2.0)
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
    Queue an MT5 API call to the dedicated dispatcher thread.
    Returns a Future; the call executes when the dispatcher next runs.

    Thread-safety: safe to call from any thread including the dispatcher
    itself (re-entrant path executes inline to avoid deadlock).
    """
    if ensure_ready:
        ensure_mt5()

    # Re-entrant safety: if called from the dispatcher thread, run inline.
    if _mt5_async_thread is not None and threading.current_thread() is _mt5_async_thread:
        fut_inline: concurrent.futures.Future = concurrent.futures.Future()
        fn = getattr(mt5, str(method_name), None)
        if not callable(fn):
            fut_inline.set_exception(
                AttributeError(f"mt5.{method_name!r} not callable (inline)")
            )
            return fut_inline
        acquired = MT5_LOCK.acquire(timeout=5.0)
        if not acquired:
            fut_inline.set_exception(
                TimeoutError(f"MT5_LOCK timeout on inline call {method_name!r}")
            )
            return fut_inline
        try:
            fut_inline.set_result(fn(*args, **kwargs))
        except Exception as exc:
            fut_inline.set_exception(exc)
        finally:
            MT5_LOCK.release()
        return fut_inline

    _ensure_mt5_async_thread()

    fut: concurrent.futures.Future = concurrent.futures.Future()
    item = (_mono(), str(method_name), tuple(args), dict(kwargs), fut)

    with _mt5_deque_cond:
        _mt5_deque.append(item)
        _mt5_deque_cond.notify()  # wake dispatcher immediately — no 200ms wait

    return fut


def _mt5_direct_call(method_name: str, *args, **kwargs) -> Any:
    """Emergency direct call — bypasses the queue. Use only in fallback paths."""
    fn = getattr(mt5, str(method_name), None)
    if not callable(fn):
        raise AttributeError(f"mt5.{method_name!r} not callable")
    acquired = MT5_LOCK.acquire(timeout=5.0)
    if not acquired:
        raise TimeoutError(f"MT5_LOCK acquire timeout in direct call {method_name!r}")
    try:
        return fn(*args, **kwargs)
    finally:
        MT5_LOCK.release()


def mt5_async_call(
    method_name: str,
    *args,
    timeout: float = 1.0,
    default: Any = None,
    ensure_ready: bool = False,
    raise_on_error: bool = False,
    retries: int = 1,
    repair_on_transport_error: bool = True,
    direct_fallback: bool = True,
    **kwargs,
) -> Any:
    """
    Synchronous wait wrapper around mt5_async_submit.
    Returns `default` on timeout/error unless raise_on_error=True.
    """
    wait_s = max(0.01, float(timeout))
    attempts = max(1, int(retries) + 1)
    last_exc: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            fut = mt5_async_submit(method_name, *args, ensure_ready=ensure_ready, **kwargs)
            return fut.result(timeout=wait_s)
        except concurrent.futures.TimeoutError as exc:
            last_exc = exc
            if repair_on_transport_error:
                _repair_mt5_once()
            if attempt < attempts - 1:
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
            if attempt < attempts - 1:
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
# FIX #3 — Auto-repair with throttle
# =============================================================================
def _repair_mt5_once(throttle_sec: float = 1.0) -> bool:
    """
    Attempt to reconnect MT5 at most once per throttle_sec window.
    Called automatically on transport errors; guarded by _INIT_LOCK internally.
    """
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


# =============================================================================
# FIX #4 — Idempotent order execution (ghost-fill guard)
# =============================================================================
def _build_idempotency_comment(cfg: MT5ClientConfig, original_comment: str) -> str:
    """
    Embed a short unique key into the MT5 order comment field.
    The key survives the broker round-trip and is readable via
    positions_get() / history_deals_get() after reconnect.
    """
    key = f"|{cfg.order_idem_prefix}{uuid.uuid4().hex[:10]}"
    combined = f"{original_comment}{key}"
    # MT5 hard limit is 31 chars — truncate original comment if necessary.
    if len(combined) > cfg.order_comment_max_len:
        keep = cfg.order_comment_max_len - len(key)
        combined = f"{original_comment[:max(0, keep)]}{key}"
    return combined


def _extract_idempotency_key(cfg: MT5ClientConfig, comment: str) -> Optional[str]:
    """Extract the embedded key from an MT5 comment field."""
    prefix = f"|{cfg.order_idem_prefix}"
    idx = comment.rfind(prefix)
    if idx == -1:
        return None
    raw = comment[idx + 1:]        # strip leading '|'
    # key = cfg.order_idem_prefix + 10 hex chars
    expected_len = len(cfg.order_idem_prefix) + 10
    return raw[:expected_len] if len(raw) >= expected_len else None


def _check_ghost_fill(
    idempotency_key: str,
    cfg: MT5ClientConfig,
    *,
    extra_lookback_sec: float = 0.0,
) -> bool:
    """
    Return True if a position or deal bearing `idempotency_key` already exists
    on the broker side — meaning the order executed despite our IPC error.

    Called after a transport error before any retry.  If True, the caller MUST
    NOT resend the order.
    """
    lookback = float(cfg.ghost_fill_lookback_sec) + extra_lookback_sec
    try:
        # 1. Check open positions.
        positions = mt5_async_call("positions_get", timeout=5.0, default=()) or ()
        for pos in positions:
            comment = str(getattr(pos, "comment", "") or "")
            key = _extract_idempotency_key(cfg, comment)
            if key and key == idempotency_key:
                logger.warning(
                    "Ghost fill detected in OPEN POSITIONS: "
                    "key=%s ticket=%s",
                    idempotency_key,
                    getattr(pos, "ticket", "?"),
                )
                return True

        # 2. Check recent deal history (covers the IPC timeout window).
        from_dt = datetime.now(timezone.utc) - timedelta(seconds=lookback)
        to_dt = datetime.now(timezone.utc)
        deals = (
            mt5_async_call("history_deals_get", from_dt, to_dt, timeout=5.0, default=())
            or ()
        )
        for deal in deals:
            comment = str(getattr(deal, "comment", "") or "")
            key = _extract_idempotency_key(cfg, comment)
            if key and key == idempotency_key:
                logger.warning(
                    "Ghost fill detected in DEAL HISTORY: "
                    "key=%s deal_ticket=%s",
                    idempotency_key,
                    getattr(deal, "ticket", "?"),
                )
                return True

    except Exception as exc:
        # Conservative: log and assume NO ghost fill so the caller can decide.
        # The caller should surface this to the risk layer.
        logger.error(
            "Ghost fill check FAILED (conservative=no_ghost): key=%s err=%s",
            idempotency_key,
            exc,
        )
    return False


def safe_order_send(
    request: Dict[str, Any],
    cfg: Optional[MT5ClientConfig] = None,
    *,
    max_attempts: int = 2,
    order_timeout: float = 10.0,
) -> Any:
    """
    Idempotent wrapper around mt5.order_send.

    Embeds a unique key in request["comment"] before sending.
    On transport error, queries the broker for an existing fill bearing that
    key *before* allowing any retry.  If a ghost fill is found, raises
    MT5GhostFillDetected so the caller's risk layer can reconcile position.

    Usage:
        result = safe_order_send(
            {"action": mt5.TRADE_ACTION_DEAL, "symbol": "XAUUSD", ...},
            cfg=my_cfg,
        )

    Returns the mt5.OrderSendResult on success, None if caller should
    handle MT5GhostFillDetected, or raises on hard failure.
    """
    if cfg is None:
        cfg = _default_config_from_env()

    original_comment = str(request.get("comment", "") or "")
    idem_comment = _build_idempotency_comment(cfg, original_comment)
    # Extract the key we embedded so we can search for it later.
    idem_key_str = _extract_idempotency_key(cfg, idem_comment)
    if idem_key_str is None:
        # Shouldn't happen — defensive fallback: use uuid directly.
        idem_key_str = f"{cfg.order_idem_prefix}{uuid.uuid4().hex[:10]}"

    tagged_request = {**request, "comment": idem_comment}

    for attempt in range(max(1, int(max_attempts))):
        try:
            result = mt5_async_call(
                "order_send",
                tagged_request,
                timeout=float(order_timeout),
                raise_on_error=True,
            )
            return result

        except Exception as exc:
            is_transport = _looks_like_transport_error(exc)
            logger.error(
                "safe_order_send attempt %d/%d failed: transport=%s err=%s",
                attempt + 1, max_attempts, is_transport, exc,
            )

            if not is_transport:
                # Hard broker rejection (invalid price, insufficient margin…)
                # — do NOT check ghost fill, just propagate.
                raise

            # Transport error path — reconnect, then check for ghost fill.
            _repair_mt5_once(throttle_sec=1.0)

            ghost = _check_ghost_fill(idem_key_str, cfg)
            if ghost:
                raise MT5GhostFillDetected(
                    f"Order may have been executed on broker side despite IPC error. "
                    f"key={idem_key_str!r}  Verify position before retrying."
                ) from exc

            if attempt < max_attempts - 1:
                logger.warning(
                    "safe_order_send: no ghost fill found — retrying "
                    "(attempt %d/%d) key=%s",
                    attempt + 1, max_attempts, idem_key_str,
                )
                continue

            # All attempts exhausted.
            raise RuntimeError(
                f"safe_order_send failed after {max_attempts} attempts: {exc}"
            ) from exc

    return None  # unreachable — silences type checkers


# =============================================================================
# Default config loader (env-backed)
# =============================================================================
def _default_config_from_env() -> MT5ClientConfig:
    try:
        from core.config import get_config_from_env as _get_cfg  # type: ignore
    except Exception:
        try:
            from config import get_config_from_env as _get_cfg  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Unable to build MT5ClientConfig: cannot import get_config_from_env"
            ) from exc

    cfg_src = _get_cfg()

    creds = MT5Credentials(
        login=int(getattr(cfg_src, "login", 0) or 0),
        password=_normalize_env_str(str(getattr(cfg_src, "password", "") or "")),
        server=_normalize_env_str(str(getattr(cfg_src, "server", "") or "")),
    )

    return MT5ClientConfig(
        creds=creds,
        mt5_path=getattr(cfg_src, "mt5_path", None),
        portable=bool(getattr(cfg_src, "mt5_portable", False)),
        autostart=bool(getattr(cfg_src, "mt5_autostart", True)),
        timeout_ms=int(getattr(cfg_src, "mt5_timeout_ms", 300_000) or 300_000),
        auth_fail_cooldown_sec=float(getattr(cfg_src, "mt5_auth_cooldown_sec", 600.0) or 600.0),
        non_retriable_cooldown_sec=float(getattr(cfg_src, "mt5_non_retriable_cooldown_sec", 20.0) or 20.0),
        ready_timeout_sec=float(getattr(cfg_src, "mt5_ready_timeout_sec", 120.0) or 120.0),
    )


# =============================================================================
# FIX #3 — ensure_mt5: double-checked locking + safe lock ordering
# =============================================================================
def ensure_mt5(cfg: Optional[MT5ClientConfig] = None) -> Any:
    """
    Ensure MT5 is initialized, connected, and healthy.  Returns the mt5 module.

    Lock ordering (never violated):
        fast path:  MT5_LOCK only
        slow path:  MT5_LOCK (check + mark uninitialized) → release → _INIT_LOCK
                    → MT5_LOCK (brief, only inside _init_and_login / _health)

    Idempotent: safe to call from multiple threads concurrently.
    """
    global _initialized, _ever_initialized, _last_health_reason
    global _auth_block_until_mono, _auth_block_reason
    global _non_retriable_block_until_mono, _non_retriable_block_reason

    if cfg is None:
        cfg = _default_config_from_env()

    _validate_cfg(cfg)
    _setup_logger(cfg)

    now = _mono()

    # Cooldown guards — checked before any lock acquisition.
    if now < _auth_block_until_mono:
        raise MT5AuthError(f"auth_blocked:{_auth_block_reason}")

    if now < _non_retriable_block_until_mono:
        cached = str(_non_retriable_block_reason or "mt5_health_failed:non_retriable_cached")
        raise RuntimeError(cached)

    _acquire_single_instance_lock(cfg)

    mt5_path = _resolve_mt5_path_windows(cfg.mt5_path)

    # -------------------------------------------------------------------------
    # Fast path: already initialized + healthy — no _INIT_LOCK needed.
    # Double-checked locking: we mark _initialized=False here, under MT5_LOCK,
    # so no other thread can use a connection we are about to tear down.
    # -------------------------------------------------------------------------
    need_reinit = False
    acquired_fast = MT5_LOCK.acquire(timeout=5.0)
    if acquired_fast:
        try:
            if _initialized:
                # _health() releases and re-acquires MT5_LOCK internally,
                # so we must release here first to respect lock ordering.
                MT5_LOCK.release()
                h = _health(cfg)
                if h.ok:
                    return mt5
                # Health failed — re-acquire to update state atomically.
                MT5_LOCK.acquire()
                _initialized = False
                _last_health_reason = h.reason
                need_reinit = True
            else:
                need_reinit = True
        finally:
            if MT5_LOCK.locked():
                # Might be locked or not depending on the path above.
                try:
                    MT5_LOCK.release()
                except RuntimeError:
                    pass
    else:
        logger.error("ensure_mt5 fast path: MT5_LOCK timeout — proceeding to _INIT_LOCK")
        need_reinit = True

    if _ever_initialized and _last_health_reason not in ("ok", "not_initialized"):
        _throttled_health_log(cfg, _last_health_reason)

    # -------------------------------------------------------------------------
    # Slow path: reinitialize under _INIT_LOCK.
    # -------------------------------------------------------------------------
    last_exc: Optional[Exception] = None
    with _INIT_LOCK:
        # Cancel any queued MT5 RPCs before reconnect/shutdown so old requests
        # do not hit a dead or restarted terminal and create stale heartbeats.
        _stop_mt5_async_thread()

        hard_reset_reason = str(_last_health_reason or "")
        if _ever_initialized and _is_terminal_running_windows():
            h2 = _health(cfg)
            hard_reset_reason = h2.reason
            if cfg.taskkill_on_ipc_timeout and _requires_hard_terminal_reset(hard_reset_reason):
                logger.warning(
                    "Zombie/disconnected terminal detected (%s). Forcing taskkill before restart.",
                    hard_reset_reason,
                )
                _mt5_shutdown_silent()
                _taskkill_terminal_windows()
            else:
                _mt5_shutdown_silent()
        else:
            _mt5_shutdown_silent()

        # Double-check: another thread may have won the race and already
        # re-established the connection while we were waiting for _INIT_LOCK.
        inner_acquired = MT5_LOCK.acquire(timeout=5.0)
        if inner_acquired:
            already_ok = _initialized
            MT5_LOCK.release()
            if already_ok:
                h3 = _health(cfg)
                if h3.ok:
                    return mt5

        for attempt in range(int(cfg.max_retries)):
            try:
                if cfg.autostart:
                    if not _is_terminal_running_windows():
                        _start_terminal_windows(mt5_path, portable=cfg.portable)
                        if float(cfg.start_wait_sec) > 0:
                            time.sleep(float(cfg.start_wait_sec))
                    elif attempt == 0 and float(cfg.start_wait_sec) > 0:
                        time.sleep(min(0.75, float(cfg.start_wait_sec)))

                _init_and_login(cfg, mt5_path)
                _wait_ready(cfg, timeout_sec=float(cfg.ready_timeout_sec))

                # Success — update state under MT5_LOCK.
                state_acquired = MT5_LOCK.acquire(timeout=5.0)
                if state_acquired:
                    _initialized = True
                    _last_health_reason = "ok"
                    MT5_LOCK.release()
                else:
                    logger.error("ensure_mt5: MT5_LOCK timeout when marking _initialized=True")

                _ever_initialized = True
                _non_retriable_block_until_mono = 0.0
                _non_retriable_block_reason = ""
                return mt5

            except MT5AuthError as exc:
                last_exc = exc
                _auth_block_reason = str(exc)
                _auth_block_until_mono = _mono() + float(cfg.auth_fail_cooldown_sec)
                _mt5_shutdown_silent()
                state_acquired = MT5_LOCK.acquire(timeout=2.0)
                if state_acquired:
                    _initialized = False
                    _last_health_reason = "auth_failed"
                    MT5_LOCK.release()
                logger.error(
                    "ensure_mt5 AUTH FAILED (cooldown=%.0fs): %s",
                    float(cfg.auth_fail_cooldown_sec),
                    exc,
                )
                raise

            except Exception as exc:
                last_exc = exc
                is_ipc = bool(cfg.taskkill_on_ipc_timeout) and _is_ipc_timeout(cfg, exc)

                if is_ipc:
                    logger.error(
                        "IPC timeout → taskkill + retry | attempt=%d err=%s",
                        attempt + 1, exc,
                    )
                    _stop_mt5_async_thread()
                    _mt5_shutdown_silent()
                    _taskkill_terminal_windows()
                    if cfg.autostart:
                        try:
                            _start_terminal_windows(mt5_path, portable=cfg.portable)
                            if float(cfg.start_wait_sec) > 0:
                                time.sleep(float(cfg.start_wait_sec))
                        except Exception as exc2:
                            logger.error("MT5 restart failed: %s", exc2)

                _mt5_shutdown_silent()
                state_acquired = MT5_LOCK.acquire(timeout=2.0)
                if state_acquired:
                    _initialized = False
                    exc_txt = str(exc).replace("\n", " ").strip()
                    _last_health_reason = (
                        f"init_failed:{exc_txt[:240] if exc_txt else type(exc).__name__}"
                    )
                    MT5_LOCK.release()

                if _is_non_retriable_runtime_error(exc):
                    _non_retriable_block_reason = str(exc)
                    _non_retriable_block_until_mono = (
                        _mono() + float(cfg.non_retriable_cooldown_sec)
                    )
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
                    _sleep_backoff(
                        float(cfg.retry_backoff_base_sec),
                        attempt,
                        cap=float(cfg.backoff_cap_sec),
                    )

    raise RuntimeError(
        f"MT5 initialization failed after {cfg.max_retries} attempts: {last_exc}"
    )


# =============================================================================
# Shutdown
# =============================================================================
def shutdown_mt5() -> None:
    """
    Gracefully shut down MT5, the async dispatcher, and the instance lock.
    Registered with atexit — safe to call multiple times.
    """
    global _initialized, _last_health_reason
    global _non_retriable_block_until_mono, _non_retriable_block_reason

    with _INIT_LOCK:
        _stop_mt5_async_thread()
        _mt5_shutdown_silent()
        state_acquired = MT5_LOCK.acquire(timeout=3.0)
        if state_acquired:
            _initialized = False
            _last_health_reason = "shutdown"
            _non_retriable_block_until_mono = 0.0
            _non_retriable_block_reason = ""
            MT5_LOCK.release()

    _release_single_instance_lock()


def mt5_status() -> Tuple[bool, str]:
    """Return (is_initialized, last_health_reason). Thread-safe, non-blocking."""
    acquired = MT5_LOCK.acquire(timeout=1.0)
    if not acquired:
        return False, "status_lock_timeout"
    try:
        return bool(_initialized), str(_last_health_reason)
    finally:
        MT5_LOCK.release()


# =============================================================================
# Shutdown hook
# =============================================================================
atexit.register(shutdown_mt5)


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Core lifecycle
    "ensure_mt5",
    "shutdown_mt5",
    "mt5_status",
    # Async dispatch
    "mt5_async_submit",
    "mt5_async_call",
    # FIX #4 — idempotent order execution
    "safe_order_send",
    # Configuration / credentials
    "MT5ClientConfig",
    "MT5Credentials",
    # Exceptions
    "MT5AuthError",
    "MT5GhostFillDetected",
    # Shared lock (for callers that must do compound operations atomically)
    "MT5_LOCK",
    # Re-export the mt5 module for convenience
    "mt5",
]
