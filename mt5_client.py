"""
Terminal management and thread-safe MT5 dispatch helpers.

Wraps MetaTrader 5 connectivity, async dispatch, recovery logic,
and idempotent order submission for the live runtime.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import glob
import heapq
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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore

from log_config import LOG_DIR as LOG_ROOT
from log_config import get_log_path

# =============================================================================
# Global Constants & Meta Definitions
# =============================================================================
def _env_bool(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str = "") -> str:
    raw = str(os.getenv(name, default) or default).strip()
    return _normalize_env_str(raw)


def _env_float(
    name: str,
    default: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    try:
        value = float(os.getenv(name, str(default)) or default)
    except Exception:
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return float(value)


def _env_int(
    name: str,
    default: int,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    try:
        value = int(float(os.getenv(name, str(default)) or default))
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    return int(value)


# Async dispatcher pressure controls
_ASYNC_STALE_THRESHOLD_SEC: float = _env_float(
    "MT5_ASYNC_STALE_THRESHOLD_SEC",
    1.5,
    minimum=0.05,
    maximum=30.0,
)
_MT5_ASYNC_QUEUE_MAXSIZE: int = _env_int(
    "MT5_ASYNC_QUEUE_MAXSIZE",
    256,
    minimum=16,
    maximum=4096,
)
_MT5_ASYNC_LOCK_TIMEOUT_SEC: float = _env_float(
    "MT5_ASYNC_LOCK_TIMEOUT_SEC",
    8.0,
    minimum=0.25,
    maximum=60.0,
)
_MT5_ASYNC_DIRECT_FALLBACK_MAX_DEPTH: int = _env_int(
    "MT5_ASYNC_DIRECT_FALLBACK_MAX_DEPTH",
    4,
    minimum=0,
    maximum=1024,
)
_MT5_ASYNC_PREEMPT_WRITES: bool = _env_bool("MT5_ASYNC_PREEMPT_WRITES", "1")
_MT5_ASYNC_WRITE_DIRECT_FALLBACK_UNDER_LOAD: bool = _env_bool(
    "MT5_ASYNC_WRITE_DIRECT_FALLBACK_UNDER_LOAD", "1"
)

_MT5_DISPATCH_PRIORITY_CRITICAL = 0
_MT5_DISPATCH_PRIORITY_HIGH = 10
_MT5_DISPATCH_PRIORITY_NORMAL = 20
_MT5_DISPATCH_PRIORITY_LOW = 30

# =============================================================================
# Global Mutex and State Variables
# =============================================================================
# MT5_LOCK: guards EVERY mt5.* call from all threads.
MT5_LOCK = threading.Lock()

# _INIT_LOCK: Serialises the multi-step init/shutdown sequence.
# ORDERING RULE: _INIT_LOCK is ALWAYS acquired BEFORE MT5_LOCK, never after.
_INIT_LOCK = threading.Lock()

_initialized: bool = False  # guarded by MT5_LOCK
_ever_initialized: bool = False  # guarded by _INIT_LOCK
_last_health_reason: str = "not_initialized"  # guarded by MT5_LOCK
_last_health_log_ts_mono: float = 0.0  # guarded by _INIT_LOCK
_last_mt5_repair_ts_mono: float = 0.0  # guarded by _INIT_LOCK
_last_hard_reset_probe_bucket: str = ""
_last_hard_reset_probe_count: int = 0
_last_hard_reset_probe_ts_mono: float = 0.0

_auth_block_until_mono: float = 0.0
_auth_block_reason: str = ""
_non_retriable_block_until_mono: float = 0.0
_non_retriable_block_reason: str = ""
_transport_state_lock = threading.Lock()
_transport_error_streak: int = 0
_transport_timeout_scale: float = 1.0

_terminal_pid: Optional[int] = None
_terminal_pid_lock = threading.Lock()

# Async dispatcher State
_mt5_dispatch_heap: List[Tuple[int, int, "_AsyncDispatchItem"]] = []
_mt5_dispatch_cond = threading.Condition(threading.Lock())
_mt5_dispatch_seq: int = 0
_mt5_dispatch_metrics_lock = threading.Lock()
_mt5_dispatch_metrics: Dict[str, Any] = {
    "enqueued_total": 0,
    "completed_total": 0,
    "failed_total": 0,
    "stale_total": 0,
    "overflow_total": 0,
    "preempted_total": 0,
    "cancelled_total": 0,
    "lock_timeout_total": 0,
    "inline_total": 0,
    "direct_fallback_total": 0,
    "direct_fallback_blocked_total": 0,
    "high_watermark": 0,
    "last_queue_depth": 0,
    "last_enqueue_ts_mono": 0.0,
    "last_complete_ts_mono": 0.0,
    "last_lock_wait_ms": 0.0,
    "lock_wait_ema_ms": 0.0,
    "last_call_ms": 0.0,
    "call_ema_ms": 0.0,
}

_mt5_async_thread: Optional[threading.Thread] = None
_mt5_async_stop = threading.Event()
_mt5_async_boot_lock = threading.Lock()

# Single-instance lock state
_lock_guard = threading.Lock()
_lock_fp: Any = None
_lock_acquired: bool = False


# =============================================================================
# Custom Exceptions
# =============================================================================
class MT5AuthError(RuntimeError):
    """Hard auth failure — wrong login/password/server. Do NOT retry tight."""


class MT5GhostFillDetected(RuntimeError):
    """
    Raised by safe_order_send when an IPC error occurred but a matching
    position or deal was found on the broker side. The order WAS executed.
    """


# =============================================================================
# Data Structures
# =============================================================================
@dataclass(frozen=True)
class MT5Credentials:
    """Immutable terminal credentials."""

    login: int
    password: str
    server: str


@dataclass(frozen=True)
class MT5ClientConfig:
    """Client configuration map for deployment."""

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
    hard_reset_after_failures: int = 2
    hard_reset_debounce_sec: float = 8.0

    order_idem_prefix: str = "ik"
    order_comment_max_len: int = 31
    ghost_fill_lookback_sec: float = 120.0

    ipc_markers: Tuple[str, ...] = ("ipc timeout", "-10005", "(-10005")
    require_correct_account: bool = True

    log_queue_maxsize: int = 8192


@dataclass(frozen=True)
class Health:
    """Immutable snapshot of terminal operational health."""

    ok: bool
    reason: str


@dataclass(frozen=True)
class _AsyncDispatchItem:
    """Internal MT5 dispatch item held in the bounded priority queue."""

    enqueue_ts: float
    method_name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    fut: concurrent.futures.Future
    priority: int
    seq: int


# =============================================================================
# Logging Setup
# =============================================================================
logger = logging.getLogger("mt5")
logger.propagate = False

_log_listener: Optional[logging.handlers.QueueListener] = None


def _setup_logger(cfg: MT5ClientConfig) -> None:
    global _log_listener

    eff_level = min(int(cfg.log_level), logging.WARNING)
    logger.setLevel(eff_level)

    if any(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers):
        return

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
# Private Flow & Utility Helpers
# =============================================================================
def _mono() -> float:
    return time.monotonic()


def _last_error() -> Tuple[int, str]:
    try:
        if mt5 is None:
            return -1, "mt5 library not imported"
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


def _ema(prev: float, sample: float, alpha: float = 0.20) -> float:
    sample_v = max(0.0, float(sample))
    prev_v = max(0.0, float(prev))
    if prev_v <= 0.0:
        return sample_v
    return (prev_v * (1.0 - alpha)) + (sample_v * alpha)


def _mt5_dispatch_priority(method_name: str, explicit: Optional[int] = None) -> int:
    if explicit is not None:
        return int(explicit)

    method = str(method_name or "").strip().lower()
    if method in {"order_send", "order_check"}:
        return _MT5_DISPATCH_PRIORITY_CRITICAL
    if method in {
        "positions_get",
        "positions_total",
        "orders_get",
        "orders_total",
        "account_info",
        "symbol_info",
        "symbol_info_tick",
        "copy_rates_from_pos",
        "copy_ticks_from",
        "copy_ticks_range",
        "copy_rates_range",
        "copy_rates_from",
    }:
        return _MT5_DISPATCH_PRIORITY_HIGH
    if method.startswith("history_"):
        return _MT5_DISPATCH_PRIORITY_LOW
    return _MT5_DISPATCH_PRIORITY_NORMAL


def _mt5_dispatch_metrics_inc(name: str, inc: int = 1) -> None:
    with _mt5_dispatch_metrics_lock:
        _mt5_dispatch_metrics[name] = int(_mt5_dispatch_metrics.get(name, 0) or 0) + int(
            inc
        )


def _mt5_dispatch_metrics_note_queue_depth(depth: int) -> None:
    with _mt5_dispatch_metrics_lock:
        _mt5_dispatch_metrics["last_queue_depth"] = int(depth)
        _mt5_dispatch_metrics["high_watermark"] = max(
            int(_mt5_dispatch_metrics.get("high_watermark", 0) or 0),
            int(depth),
        )


def _mt5_dispatch_metrics_note_enqueue(depth: int) -> None:
    now = _mono()
    with _mt5_dispatch_metrics_lock:
        _mt5_dispatch_metrics["enqueued_total"] = int(
            _mt5_dispatch_metrics.get("enqueued_total", 0) or 0
        ) + 1
        _mt5_dispatch_metrics["last_queue_depth"] = int(depth)
        _mt5_dispatch_metrics["high_watermark"] = max(
            int(_mt5_dispatch_metrics.get("high_watermark", 0) or 0),
            int(depth),
        )
        _mt5_dispatch_metrics["last_enqueue_ts_mono"] = float(now)


def _mt5_dispatch_metrics_note_completion(
    *,
    call_ms: float,
    lock_wait_ms: float,
    failed: bool = False,
) -> None:
    now = _mono()
    with _mt5_dispatch_metrics_lock:
        _mt5_dispatch_metrics["completed_total"] = int(
            _mt5_dispatch_metrics.get("completed_total", 0) or 0
        ) + 1
        if failed:
            _mt5_dispatch_metrics["failed_total"] = int(
                _mt5_dispatch_metrics.get("failed_total", 0) or 0
            ) + 1
        _mt5_dispatch_metrics["last_complete_ts_mono"] = float(now)
        _mt5_dispatch_metrics["last_lock_wait_ms"] = float(lock_wait_ms)
        _mt5_dispatch_metrics["lock_wait_ema_ms"] = _ema(
            float(_mt5_dispatch_metrics.get("lock_wait_ema_ms", 0.0) or 0.0),
            lock_wait_ms,
        )
        _mt5_dispatch_metrics["last_call_ms"] = float(call_ms)
        _mt5_dispatch_metrics["call_ema_ms"] = _ema(
            float(_mt5_dispatch_metrics.get("call_ema_ms", 0.0) or 0.0),
            call_ms,
        )


def _mt5_dispatch_reset_runtime_state() -> None:
    global _mt5_dispatch_seq
    with _mt5_dispatch_cond:
        _mt5_dispatch_heap.clear()
        _mt5_dispatch_seq = 0
    with _mt5_dispatch_metrics_lock:
        _mt5_dispatch_metrics.clear()
        _mt5_dispatch_metrics.update(
            {
                "enqueued_total": 0,
                "completed_total": 0,
                "failed_total": 0,
                "stale_total": 0,
                "overflow_total": 0,
                "preempted_total": 0,
                "cancelled_total": 0,
                "lock_timeout_total": 0,
                "inline_total": 0,
                "direct_fallback_total": 0,
                "direct_fallback_blocked_total": 0,
                "high_watermark": 0,
                "last_queue_depth": 0,
                "last_enqueue_ts_mono": 0.0,
                "last_complete_ts_mono": 0.0,
                "last_lock_wait_ms": 0.0,
                "lock_wait_ema_ms": 0.0,
                "last_call_ms": 0.0,
                "call_ema_ms": 0.0,
            }
        )


def _mt5_dispatch_queue_depth() -> int:
    with _mt5_dispatch_cond:
        return int(len(_mt5_dispatch_heap))


def _mt5_should_allow_direct_fallback(method_name: str, requested: bool) -> bool:
    if not requested:
        return False

    depth = _mt5_dispatch_queue_depth()
    if depth <= _MT5_ASYNC_DIRECT_FALLBACK_MAX_DEPTH:
        return True

    priority = _mt5_dispatch_priority(method_name)
    allow_under_load = bool(
        priority <= _MT5_DISPATCH_PRIORITY_CRITICAL
        and _MT5_ASYNC_WRITE_DIRECT_FALLBACK_UNDER_LOAD
    )
    if not allow_under_load:
        _mt5_dispatch_metrics_inc("direct_fallback_blocked_total")
    return allow_under_load


def _mt5_dispatch_evict_candidate(
    new_priority: int,
) -> Optional[_AsyncDispatchItem]:
    if not _mt5_dispatch_heap:
        return None

    victim_idx: Optional[int] = None
    victim_key: Tuple[int, int] = (-1, -1)

    for idx, (priority, seq, item) in enumerate(_mt5_dispatch_heap):
        if priority <= new_priority:
            continue
        key = (priority, seq)
        if key > victim_key:
            victim_idx = idx
            victim_key = key

    if victim_idx is None:
        return None

    _, _, victim = _mt5_dispatch_heap.pop(victim_idx)
    heapq.heapify(_mt5_dispatch_heap)
    return victim


def _mt5_dispatch_submit_item(
    method_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    fut: concurrent.futures.Future,
    *,
    priority: Optional[int] = None,
) -> None:
    global _mt5_dispatch_seq

    enqueue_ts = _mono()
    dispatch_priority = _mt5_dispatch_priority(method_name, explicit=priority)
    victim: Optional[_AsyncDispatchItem] = None

    with _mt5_dispatch_cond:
        if len(_mt5_dispatch_heap) >= _MT5_ASYNC_QUEUE_MAXSIZE:
            if _MT5_ASYNC_PREEMPT_WRITES:
                victim = _mt5_dispatch_evict_candidate(dispatch_priority)
            if victim is None and len(_mt5_dispatch_heap) >= _MT5_ASYNC_QUEUE_MAXSIZE:
                _mt5_dispatch_metrics_inc("overflow_total")
                _mt5_dispatch_metrics_note_queue_depth(len(_mt5_dispatch_heap))
                fut.set_exception(
                    RuntimeError(
                        f"mt5_async_queue_full:{method_name}:{len(_mt5_dispatch_heap)}/"
                        f"{_MT5_ASYNC_QUEUE_MAXSIZE}"
                    )
                )
                return

        item = _AsyncDispatchItem(
            enqueue_ts=enqueue_ts,
            method_name=str(method_name),
            args=tuple(args),
            kwargs=dict(kwargs),
            fut=fut,
            priority=int(dispatch_priority),
            seq=int(_mt5_dispatch_seq),
        )
        _mt5_dispatch_seq += 1
        heapq.heappush(_mt5_dispatch_heap, (item.priority, item.seq, item))
        depth = len(_mt5_dispatch_heap)
        _mt5_dispatch_metrics_note_enqueue(depth)
        _mt5_dispatch_cond.notify()

    if victim is not None:
        _mt5_dispatch_metrics_inc("preempted_total")
        _mt5_dispatch_metrics_inc("overflow_total")
        _set_future_exception_safe(
            victim.fut,
            RuntimeError(
                f"mt5_async_queue_preempted:{victim.method_name}->{method_name}"
            ),
            victim.method_name,
        )


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
    return bool(code == -6 or "authorization failed" in msg or "invalid account" in msg)


def _looks_like_transport_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return any(
        kw in msg
        for kw in (
            "mt5_async_stale",
            "timeout",
            "ipc",
            "-10005",
            "no connection",
            "connection",
            "mt5_async_queue",
        )
    )


def _should_repair_transport_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    if any(
        noisy in msg
        for noisy in (
            "mt5_async_queue",
            "mt5_async_stale",
            "queue_preempted",
            "queue_full",
            "lock acquire timeout",
            "status_lock_timeout",
        )
    ):
        return False
    return any(
        kw in msg
        for kw in (
            "ipc",
            "-10005",
            "no connection",
            "authorization failed",
            "invalid account",
        )
    )


def _update_transport_health(
    *,
    ok: bool,
    exc: Optional[Exception] = None,
) -> None:
    """Track recent transport instability to adapt timeouts under jitter."""
    global _transport_error_streak, _transport_timeout_scale

    with _transport_state_lock:
        if ok:
            _transport_error_streak = max(0, _transport_error_streak - 1)
            _transport_timeout_scale = max(1.0, _transport_timeout_scale * 0.85)
            return

        if exc is not None and not _looks_like_transport_error(exc):
            return

        _transport_error_streak = min(8, _transport_error_streak + 1)
        _transport_timeout_scale = min(3.0, 1.0 + (0.30 * _transport_error_streak))


def _adaptive_transport_timeout(timeout: float, attempt: int) -> float:
    """Expand request timeouts while the transport layer remains unstable."""
    with _transport_state_lock:
        scale = max(1.0, float(_transport_timeout_scale))

    effective = max(0.01, float(timeout)) * scale * (1.0 + (0.35 * max(0, attempt)))
    return min(max(effective, 0.05), 60.0)


def _adaptive_retry_pause(attempt: int) -> None:
    """Small pacing delay to absorb bursty latency spikes before retrying."""
    with _transport_state_lock:
        scale = max(1.0, float(_transport_timeout_scale))

    delay = min(3.0, (0.15 * scale) + (0.20 * max(0, attempt)))
    time.sleep(delay)


def _mt5_shutdown_silent() -> None:
    if mt5 is None:
        return
    try:
        mt5.shutdown()
    except Exception:
        pass


def _throttled_health_log(cfg: MT5ClientConfig, reason: str) -> None:
    global _last_health_log_ts_mono
    now = _mono()
    if now - _last_health_log_ts_mono >= float(cfg.health_log_throttle_sec):
        logger.warning("Fast-path health failed — attempting auto-heal: %s", reason)
        _last_health_log_ts_mono = now


def _cancel_pending_mt5_async(reason: str = "mt5_async_reset") -> int:
    drained: List[_AsyncDispatchItem] = []
    with _mt5_dispatch_cond:
        while _mt5_dispatch_heap:
            _, _, item = heapq.heappop(_mt5_dispatch_heap)
            drained.append(item)
        _mt5_dispatch_metrics_note_queue_depth(0)

    cancelled = 0
    for item in drained:
        try:
            method_name = str(item.method_name)
            fut = item.fut
            if fut.cancelled() or fut.done():
                continue
            fut.set_exception(RuntimeError(f"{reason}:{method_name}"))
            cancelled += 1
        except Exception:
            continue
    if cancelled > 0:
        _mt5_dispatch_metrics_inc("cancelled_total", cancelled)
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


def _reset_hard_reset_probe() -> None:
    global _last_hard_reset_probe_bucket
    global _last_hard_reset_probe_count
    global _last_hard_reset_probe_ts_mono

    _last_hard_reset_probe_bucket = ""
    _last_hard_reset_probe_count = 0
    _last_hard_reset_probe_ts_mono = 0.0


def _should_force_terminal_reset(
    cfg: MT5ClientConfig, reason: str
) -> Tuple[bool, int, int]:
    global _last_hard_reset_probe_bucket
    global _last_hard_reset_probe_count
    global _last_hard_reset_probe_ts_mono

    threshold = max(1, int(cfg.hard_reset_after_failures))
    if not _requires_hard_terminal_reset(reason):
        _reset_hard_reset_probe()
        return False, 0, threshold

    now = _mono()
    bucket = "hard_terminal_reset"
    debounce_sec = max(0.5, float(cfg.hard_reset_debounce_sec))

    if (
        _last_hard_reset_probe_bucket == bucket
        and (now - _last_hard_reset_probe_ts_mono) <= debounce_sec
    ):
        _last_hard_reset_probe_count += 1
    else:
        _last_hard_reset_probe_bucket = bucket
        _last_hard_reset_probe_count = 1

    _last_hard_reset_probe_ts_mono = now
    return (
        _last_hard_reset_probe_count >= threshold,
        _last_hard_reset_probe_count,
        threshold,
    )


def _is_terminal_running_windows() -> bool:
    global _terminal_pid
    with _terminal_pid_lock:
        if _terminal_pid is not None:
            try:
                p = psutil.Process(_terminal_pid)
                return p.is_running() and p.status() not in (
                    psutil.STATUS_ZOMBIE,
                    psutil.STATUS_DEAD,
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                _terminal_pid = None

        try:
            for proc in psutil.process_iter(["name", "pid"]):
                pname = (proc.info.get("name") or "").lower()
                if "terminal64" in pname or pname == "terminal.exe":
                    _terminal_pid = proc.info["pid"]
                    return True
        except Exception:
            pass
        return False


def _taskkill_terminal_windows() -> None:
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

    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", "terminal64.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", "terminal.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def _start_terminal_windows(mt5_path: str, portable: bool) -> None:
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
    if mt5_path and os.path.exists(mt5_path):
        return str(mt5_path)
    if mt5_path:
        try:
            logger.warning("MT5_CONFIG_PATH_NOT_FOUND | path=%r", str(mt5_path))
        except Exception:
            pass

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
            return str(candidate)
    return ""


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
            raise RuntimeError(
                f"Failed to acquire single-instance lock: {exc}"
            ) from exc


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
            try:
                p = _lock_file_path()
                if p.exists():
                    p.unlink()
            except Exception:
                pass
            _lock_fp = None
            _lock_acquired = False


_NON_RETRIABLE_HEALTH_REASONS = frozenset(
    {
        "algo_trading_disabled_in_terminal",
        "trading_disabled_for_account",
        "wrong_account",
    }
)


def _health(cfg: MT5ClientConfig) -> Health:
    if mt5 is None:
        return Health(False, "no mt5 available")
    acquired = MT5_LOCK.acquire(timeout=5.0)
    if not acquired:
        logger.error("_health: MT5_LOCK acquire timed out")
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
        if cfg.require_correct_account and int(getattr(ai, "login", -1)) != int(
            cfg.creds.login
        ):
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
            "Tools → Options → Expert Advisors → Allow Algo Trading."
        )
    raise RuntimeError(f"mt5_health_failed:{last_reason}{hint}")


def _is_non_retriable_runtime_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return "mt5_health_failed:" in msg and any(
        r in msg for r in _NON_RETRIABLE_HEALTH_REASONS
    )


def _init_and_login(cfg: MT5ClientConfig, mt5_path: str) -> None:
    if mt5 is None:
        raise RuntimeError("mt5 not imported")
    _mt5_shutdown_silent()
    time.sleep(0.10)

    init_kwargs: Dict[str, Any] = {
        "portable": bool(cfg.portable),
        "timeout": int(cfg.timeout_ms),
    }
    if mt5_path:
        init_kwargs["path"] = str(mt5_path)

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


def _set_future_result_safe(
    fut: concurrent.futures.Future, result: Any, method_name: str
) -> None:
    try:
        fut.set_result(result)
    except concurrent.futures.InvalidStateError:
        logger.debug("_set_future_result_safe: future cancelled (%r)", method_name)


def _set_future_exception_safe(
    fut: concurrent.futures.Future, exc: Exception, method_name: str
) -> None:
    try:
        fut.set_exception(exc)
    except concurrent.futures.InvalidStateError:
        logger.debug("_set_future_exception_safe: future cancelled (%r)", method_name)


def _mt5_async_loop() -> None:
    while True:
        with _mt5_dispatch_cond:
            _mt5_dispatch_cond.wait_for(
                lambda: bool(_mt5_dispatch_heap) or _mt5_async_stop.is_set(),
                timeout=1.0,
            )
            if _mt5_async_stop.is_set() and not _mt5_dispatch_heap:
                break
            if not _mt5_dispatch_heap:
                continue
            _, _, item = heapq.heappop(_mt5_dispatch_heap)
            _mt5_dispatch_metrics_note_queue_depth(len(_mt5_dispatch_heap))

        enqueue_ts = float(item.enqueue_ts)
        method_name = str(item.method_name)
        args = item.args
        kwargs = item.kwargs
        fut = item.fut

        if not fut.set_running_or_notify_cancel():
            _mt5_dispatch_metrics_inc("cancelled_total")
            del fut
            continue

        age = _mono() - enqueue_ts
        if age > _ASYNC_STALE_THRESHOLD_SEC:
            _set_future_exception_safe(
                fut,
                TimeoutError(
                    f"mt5_async_stale: {method_name!r} queued {age * 1000:.0f}ms ago"
                ),
                method_name,
            )
            _mt5_dispatch_metrics_inc("stale_total")
            del fut
            continue

        if mt5 is None:
            _set_future_exception_safe(fut, AttributeError("mt5 is None"), method_name)
            del fut
            continue

        fn = getattr(mt5, str(method_name), None)
        if not callable(fn):
            _set_future_exception_safe(
                fut, AttributeError(f"mt5.{method_name!r} is not callable"), method_name
            )
            del fut
            continue

        lock_wait_started = _mono()
        acquired = MT5_LOCK.acquire(timeout=_MT5_ASYNC_LOCK_TIMEOUT_SEC)
        lock_wait_ms = (_mono() - lock_wait_started) * 1000.0
        if not acquired:
            _set_future_exception_safe(
                fut,
                TimeoutError(
                    f"mt5_async_dispatch: MT5_LOCK acquire timeout for {method_name!r}"
                ),
                method_name,
            )
            _mt5_dispatch_metrics_inc("lock_timeout_total")
            del fut
            continue

        call_started = _mono()
        try:
            call_result = fn(*args, **kwargs)
            call_exc: Optional[Exception] = None
        except Exception as exc:
            call_result = None
            call_exc = exc
        finally:
            MT5_LOCK.release()
        call_ms = (_mono() - call_started) * 1000.0
        _mt5_dispatch_metrics_note_completion(
            call_ms=call_ms,
            lock_wait_ms=lock_wait_ms,
            failed=call_exc is not None,
        )

        if call_exc is not None:
            _set_future_exception_safe(fut, call_exc, method_name)
        else:
            _set_future_result_safe(fut, call_result, method_name)

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
        with _mt5_dispatch_cond:
            _mt5_dispatch_cond.notify_all()
        th = _mt5_async_thread
        if th and th.is_alive():
            th.join(timeout=2.0)
    finally:
        _mt5_async_thread = None
        _mt5_async_stop.clear()


def _mt5_direct_call(method_name: str, *args, **kwargs) -> Any:
    if mt5 is None:
        raise AttributeError("mt5 not valid")
    fn = getattr(mt5, str(method_name), None)
    if not callable(fn):
        raise AttributeError(f"mt5.{method_name!r} not callable")
    lock_wait_started = _mono()
    acquired = MT5_LOCK.acquire(timeout=_MT5_ASYNC_LOCK_TIMEOUT_SEC)
    lock_wait_ms = (_mono() - lock_wait_started) * 1000.0
    if not acquired:
        _mt5_dispatch_metrics_inc("lock_timeout_total")
        raise TimeoutError(f"MT5_LOCK acquire timeout in direct call {method_name!r}")
    call_started = _mono()
    try:
        result = fn(*args, **kwargs)
        call_exc: Optional[Exception] = None
        return result
    except Exception as exc:
        call_exc = exc
        raise
    finally:
        call_ms = (_mono() - call_started) * 1000.0
        _mt5_dispatch_metrics_note_completion(
            call_ms=call_ms,
            lock_wait_ms=lock_wait_ms,
            failed=call_exc is not None,
        )
        MT5_LOCK.release()


def _repair_mt5_once(throttle_sec: float = 1.0) -> bool:
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


def _build_idempotency_comment(cfg: MT5ClientConfig, original_comment: str) -> str:
    key = f"|{cfg.order_idem_prefix}{uuid.uuid4().hex[:10]}"
    combined = f"{original_comment}{key}"
    if len(combined) > cfg.order_comment_max_len:
        keep = cfg.order_comment_max_len - len(key)
        combined = f"{original_comment[:max(0, keep)]}{key}"
    return combined


def _extract_idempotency_key(cfg: MT5ClientConfig, comment: str) -> Optional[str]:
    prefix = f"|{cfg.order_idem_prefix}"
    idx = comment.rfind(prefix)
    if idx == -1:
        return None
    raw = comment[idx + 1 :]
    expected_len = len(cfg.order_idem_prefix) + 10
    return raw[:expected_len] if len(raw) >= expected_len else None


def _check_ghost_fill(
    idempotency_key: str,
    cfg: MT5ClientConfig,
    *,
    extra_lookback_sec: float = 0.0,
) -> bool:
    lookback = float(cfg.ghost_fill_lookback_sec) + extra_lookback_sec
    try:
        positions = mt5_async_call("positions_get", timeout=5.0, default=()) or ()
        for pos in positions:
            comment = str(getattr(pos, "comment", "") or "")
            key = _extract_idempotency_key(cfg, comment)
            if key and key == idempotency_key:
                logger.warning(
                    "Ghost fill detected in OPEN POSITIONS: key=%s ticket=%s",
                    idempotency_key,
                    getattr(pos, "ticket", "?"),
                )
                return True

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
                    "Ghost fill detected in DEAL HISTORY: key=%s deal_ticket=%s",
                    idempotency_key,
                    getattr(deal, "ticket", "?"),
                )
                return True
    except Exception as exc:
        logger.error(
            "Ghost fill check FAILED (conservative=no_ghost): key=%s err=%s",
            idempotency_key,
            exc,
        )
    return False


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
    cfg_path = _env_str(
        "MT5_PATH",
        _env_str(
            "MT5_TERMINAL_PATH",
            _env_str(
                "EXNESS_MT5_PATH",
                str(getattr(cfg_src, "mt5_path", "") or ""),
            ),
        ),
    )
    cfg_portable_default = "1" if bool(getattr(cfg_src, "mt5_portable", False)) else "0"
    cfg_autostart_default = "1" if bool(getattr(cfg_src, "mt5_autostart", True)) else "0"
    cfg_taskkill_default = (
        "1" if bool(getattr(cfg_src, "mt5_taskkill_on_ipc_timeout", True)) else "0"
    )
    cfg_single_lock_default = (
        "1" if bool(getattr(cfg_src, "mt5_single_instance_lock", True)) else "0"
    )

    return MT5ClientConfig(
        creds=creds,
        mt5_path=cfg_path or None,
        portable=_env_bool("MT5_PORTABLE", cfg_portable_default),
        autostart=_env_bool("MT5_AUTOSTART", cfg_autostart_default),
        single_instance_lock=_env_bool(
            "MT5_SINGLE_INSTANCE_LOCK", cfg_single_lock_default
        ),
        taskkill_on_ipc_timeout=_env_bool(
            "MT5_TASKKILL_ON_IPC_TIMEOUT", cfg_taskkill_default
        ),
        timeout_ms=_env_int(
            "MT5_TIMEOUT_MS",
            int(getattr(cfg_src, "mt5_timeout_ms", 300_000) or 300_000),
            minimum=1_000,
            maximum=900_000,
        ),
        start_wait_sec=_env_float(
            "MT5_START_WAIT_SEC",
            float(getattr(cfg_src, "mt5_start_wait_sec", 5.0) or 5.0),
            minimum=0.0,
            maximum=60.0,
        ),
        max_retries=_env_int(
            "MT5_MAX_RETRIES",
            int(getattr(cfg_src, "mt5_max_retries", 6) or 6),
            minimum=1,
            maximum=30,
        ),
        retry_backoff_base_sec=_env_float(
            "MT5_RETRY_BACKOFF_BASE_SEC",
            float(getattr(cfg_src, "mt5_retry_backoff_base_sec", 1.2) or 1.2),
            minimum=0.1,
            maximum=30.0,
        ),
        backoff_cap_sec=_env_float(
            "MT5_BACKOFF_CAP_SEC",
            float(getattr(cfg_src, "mt5_backoff_cap_sec", 12.0) or 12.0),
            minimum=0.1,
            maximum=120.0,
        ),
        auth_fail_cooldown_sec=float(
            getattr(cfg_src, "mt5_auth_cooldown_sec", 600.0) or 600.0
        ),
        non_retriable_cooldown_sec=float(
            getattr(cfg_src, "mt5_non_retriable_cooldown_sec", 20.0) or 20.0
        ),
        ready_timeout_sec=_env_float(
            "MT5_READY_TIMEOUT_SEC",
            float(getattr(cfg_src, "mt5_ready_timeout_sec", 120.0) or 120.0),
            minimum=1.0,
            maximum=600.0,
        ),
        hard_reset_after_failures=_env_int(
            "MT5_HARD_RESET_AFTER_FAILURES",
            int(getattr(cfg_src, "mt5_hard_reset_after_failures", 2) or 2),
            minimum=1,
            maximum=20,
        ),
        hard_reset_debounce_sec=_env_float(
            "MT5_HARD_RESET_DEBOUNCE_SEC",
            float(getattr(cfg_src, "mt5_hard_reset_debounce_sec", 8.0) or 8.0),
            minimum=0.0,
            maximum=120.0,
        ),
    )


def _set_mt5_state(initialized: bool, reason: str, lock_timeout: float = 2.0) -> bool:
    global _initialized, _last_health_reason
    state_acquired = MT5_LOCK.acquire(timeout=lock_timeout)
    if state_acquired:
        _initialized = initialized
        _last_health_reason = reason
        MT5_LOCK.release()
    else:
        logger.error("_set_mt5_state: MT5_LOCK timeout (%.1fs)", lock_timeout)
    return state_acquired


# =============================================================================
# Public API
# =============================================================================
def mt5_async_submit(
    method_name: str,
    *args,
    ensure_ready: bool = False,
    dispatch_priority: Optional[int] = None,
    **kwargs,
) -> concurrent.futures.Future:
    """Queue an MT5 API call to the dedicated dispatcher thread safely."""
    if ensure_ready:
        ensure_mt5()

    if (
        _mt5_async_thread is not None
        and threading.current_thread() is _mt5_async_thread
    ):
        fut_inline: concurrent.futures.Future = concurrent.futures.Future()
        if mt5 is None:
            fut_inline.set_exception(AttributeError("mt5 is None"))
            return fut_inline

        fn = getattr(mt5, str(method_name), None)
        if not callable(fn):
            fut_inline.set_exception(
                AttributeError(f"mt5.{method_name!r} not callable (inline)")
            )
            return fut_inline

        lock_wait_started = _mono()
        acquired = MT5_LOCK.acquire(timeout=_MT5_ASYNC_LOCK_TIMEOUT_SEC)
        lock_wait_ms = (_mono() - lock_wait_started) * 1000.0
        if not acquired:
            _mt5_dispatch_metrics_inc("lock_timeout_total")
            fut_inline.set_exception(
                TimeoutError(f"MT5_LOCK timeout on inline call {method_name!r}")
            )
            return fut_inline
        call_started = _mono()
        try:
            result = fn(*args, **kwargs)
            call_exc: Optional[Exception] = None
            fut_inline.set_result(result)
        except Exception as exc:
            call_exc = exc
            try:
                fut_inline.set_exception(exc)
            except concurrent.futures.InvalidStateError:
                pass
        finally:
            call_ms = (_mono() - call_started) * 1000.0
            _mt5_dispatch_metrics_inc("inline_total")
            _mt5_dispatch_metrics_note_completion(
                call_ms=call_ms,
                lock_wait_ms=lock_wait_ms,
                failed=call_exc is not None,
            )
            MT5_LOCK.release()
        return fut_inline

    _ensure_mt5_async_thread()

    fut: concurrent.futures.Future = concurrent.futures.Future()
    _mt5_dispatch_submit_item(
        method_name=str(method_name),
        args=tuple(args),
        kwargs=dict(kwargs),
        fut=fut,
        priority=dispatch_priority,
    )

    return fut


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
    dispatch_priority: Optional[int] = None,
    **kwargs,
) -> Any:
    """Synchronous wait wrapper around mt5_async_submit."""
    attempts = max(1, int(retries) + 1)
    last_exc: Optional[Exception] = None

    for attempt in range(attempts):
        wait_s = _adaptive_transport_timeout(timeout, attempt)
        fut: Optional[concurrent.futures.Future] = None
        try:
            fut = mt5_async_submit(
                method_name,
                *args,
                ensure_ready=ensure_ready,
                dispatch_priority=dispatch_priority,
                **kwargs,
            )
            result = fut.result(timeout=wait_s)
            _update_transport_health(ok=True)
            return result
        except concurrent.futures.TimeoutError as exc:
            last_exc = exc
            cancelled = bool(fut is not None and fut.cancel())
            if cancelled:
                _mt5_dispatch_metrics_inc("cancelled_total")
            _update_transport_health(ok=False, exc=exc)
            if repair_on_transport_error and _should_repair_transport_error(exc):
                _repair_mt5_once()
            if attempt < attempts - 1:
                if not cancelled:
                    last_exc = TimeoutError(
                        f"mt5_async_inflight_timeout:{method_name!r}:retry_blocked"
                    )
                    break
                _adaptive_retry_pause(attempt)
                continue
            if cancelled and _mt5_should_allow_direct_fallback(
                method_name, direct_fallback
            ):
                try:
                    _mt5_dispatch_metrics_inc("direct_fallback_total")
                    result = _mt5_direct_call(method_name, *args, **kwargs)
                    _update_transport_health(ok=True)
                    return result
                except Exception as exc2:
                    last_exc = exc2
            elif direct_fallback and not cancelled:
                last_exc = TimeoutError(
                    f"mt5_async_inflight_timeout:{method_name!r}:direct_fallback_blocked"
                )
        except Exception as exc:
            last_exc = exc
            if repair_on_transport_error and _should_repair_transport_error(exc):
                _update_transport_health(ok=False, exc=exc)
                _repair_mt5_once()
            elif _looks_like_transport_error(exc):
                _update_transport_health(ok=False, exc=exc)
            if attempt < attempts - 1:
                if _looks_like_transport_error(exc):
                    _adaptive_retry_pause(attempt)
                continue
            if _mt5_should_allow_direct_fallback(method_name, direct_fallback):
                try:
                    _mt5_dispatch_metrics_inc("direct_fallback_total")
                    result = _mt5_direct_call(method_name, *args, **kwargs)
                    _update_transport_health(ok=True)
                    return result
                except Exception as exc2:
                    last_exc = exc2

    if raise_on_error and last_exc is not None:
        raise last_exc
    return default


def safe_order_send(
    request: Dict[str, Any],
    cfg: Optional[MT5ClientConfig] = None,
    *,
    max_attempts: int = 2,
    order_timeout: float = 10.0,
) -> Any:
    """Idempotent wrapper around mt5.order_send with ghost-fill protection."""
    if cfg is None:
        cfg = _default_config_from_env()

    original_comment = str(request.get("comment", "") or "")
    idem_comment = _build_idempotency_comment(cfg, original_comment)
    idem_key_str = _extract_idempotency_key(cfg, idem_comment)
    if idem_key_str is None:
        idem_key_str = f"{cfg.order_idem_prefix}{uuid.uuid4().hex[:10]}"

    tagged_request = {**request, "comment": idem_comment}

    for attempt in range(max(1, int(max_attempts))):
        try:
            effective_timeout = _adaptive_transport_timeout(order_timeout, attempt)
            result = mt5_async_call(
                "order_send",
                tagged_request,
                timeout=effective_timeout,
                raise_on_error=True,
            )
            return result

        except Exception as exc:
            is_transport = _looks_like_transport_error(exc)
            logger.error(
                "safe_order_send attempt %d/%d failed: transport=%s err=%s",
                attempt + 1,
                max_attempts,
                is_transport,
                exc,
            )

            if not is_transport:
                raise

            if _should_repair_transport_error(exc):
                _repair_mt5_once(throttle_sec=min(2.5, 1.0 + (0.40 * attempt)))
            ghost = _check_ghost_fill(idem_key_str, cfg)
            if ghost:
                raise MT5GhostFillDetected(
                    f"Order executed on broker side despite IPC error. "
                    f"key={idem_key_str!r} Verify position before retrying."
                ) from exc

            if attempt < max_attempts - 1:
                logger.warning(
                    "safe_order_send: no ghost fill found — retrying key=%s",
                    idem_key_str,
                )
                _adaptive_retry_pause(attempt)
                continue

            raise RuntimeError(
                f"safe_order_send failed after {max_attempts} attempts: {exc}"
            ) from exc

    return None


def ensure_mt5(cfg: Optional[MT5ClientConfig] = None) -> Any:
    """Ensure MT5 is initialized, connected, and healthy. Resilient connection matrix."""
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
        cached = str(
            _non_retriable_block_reason or "mt5_health_failed:non_retriable_cached"
        )
        raise RuntimeError(cached)

    _acquire_single_instance_lock(cfg)
    mt5_path = _resolve_mt5_path_windows(cfg.mt5_path)

    acquired_fast = MT5_LOCK.acquire(timeout=5.0)
    if acquired_fast:
        _fast_lock_held = True
        try:
            if _initialized:
                MT5_LOCK.release()
                _fast_lock_held = False
                h = _health(cfg)
                if h.ok:
                    return mt5
                MT5_LOCK.acquire()
                _fast_lock_held = True
                _initialized = False
                _last_health_reason = h.reason
        finally:
            if _fast_lock_held:
                try:
                    MT5_LOCK.release()
                except RuntimeError:
                    logger.error(
                        "ensure_mt5 fast path: RuntimeError releasing MT5_LOCK"
                    )

    if _ever_initialized and _last_health_reason not in ("ok", "not_initialized"):
        _throttled_health_log(cfg, _last_health_reason)

    last_exc: Optional[Exception] = None
    with _INIT_LOCK:
        _stop_mt5_async_thread()

        hard_reset_reason = str(_last_health_reason or "")
        if _ever_initialized and _is_terminal_running_windows():
            h2 = _health(cfg)
            hard_reset_reason = h2.reason
            if cfg.taskkill_on_ipc_timeout and _requires_hard_terminal_reset(
                hard_reset_reason
            ):
                (
                    should_kill,
                    failure_count,
                    failure_threshold,
                ) = _should_force_terminal_reset(cfg, hard_reset_reason)
                if should_kill:
                    logger.warning(
                        "Zombie terminal (%s). Forcing taskkill after %d/%d consecutive health failures.",
                        hard_reset_reason,
                        failure_count,
                        failure_threshold,
                    )
                    _mt5_shutdown_silent()
                    _taskkill_terminal_windows()
                else:
                    logger.warning(
                        "Terminal unhealthy (%s). Soft reset %d/%d before taskkill.",
                        hard_reset_reason,
                        failure_count,
                        failure_threshold,
                    )
                    _mt5_shutdown_silent()
            else:
                _reset_hard_reset_probe()
                _mt5_shutdown_silent()
        else:
            _reset_hard_reset_probe()
            _mt5_shutdown_silent()

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
                    if mt5_path and not _is_terminal_running_windows():
                        _start_terminal_windows(mt5_path, portable=cfg.portable)
                        if float(cfg.start_wait_sec) > 0:
                            time.sleep(float(cfg.start_wait_sec))
                    elif not mt5_path and attempt == 0:
                        logger.warning(
                            "MT5_PATH_UNRESOLVED | action=initialize_without_explicit_path"
                        )
                    elif attempt == 0 and float(cfg.start_wait_sec) > 0:
                        time.sleep(min(0.75, float(cfg.start_wait_sec)))

                _init_and_login(cfg, mt5_path)
                _wait_ready(cfg, timeout_sec=float(cfg.ready_timeout_sec))

                if not _set_mt5_state(True, "ok", lock_timeout=5.0):
                    logger.error(
                        "ensure_mt5: MT5_LOCK timeout when marking _initialized=True"
                    )

                _ever_initialized = True
                _reset_hard_reset_probe()
                _non_retriable_block_until_mono = 0.0
                _non_retriable_block_reason = ""
                return mt5

            except MT5AuthError as exc:
                last_exc = exc
                _auth_block_reason = str(exc)
                _auth_block_until_mono = _mono() + float(cfg.auth_fail_cooldown_sec)
                _mt5_shutdown_silent()
                _set_mt5_state(False, "auth_failed", lock_timeout=2.0)
                logger.error("ensure_mt5 AUTH FAILED: %s", exc)
                raise

            except Exception as exc:
                last_exc = exc
                is_ipc = bool(cfg.taskkill_on_ipc_timeout) and _is_ipc_timeout(cfg, exc)

                if is_ipc:
                    logger.error("IPC timeout → taskkill + retry err=%s", exc)
                    _stop_mt5_async_thread()
                    _mt5_shutdown_silent()
                    _taskkill_terminal_windows()
                    if cfg.autostart and mt5_path:
                        try:
                            _start_terminal_windows(mt5_path, portable=cfg.portable)
                            if float(cfg.start_wait_sec) > 0:
                                time.sleep(float(cfg.start_wait_sec))
                        except Exception as exc2:
                            logger.error("MT5 restart failed: %s", exc2)

                _mt5_shutdown_silent()

                exc_txt = str(exc).replace("\n", " ").strip()
                init_fail_reason = (
                    f"init_failed:{exc_txt[:240] if exc_txt else type(exc).__name__}"
                )
                _set_mt5_state(False, init_fail_reason, lock_timeout=2.0)

                if _is_non_retriable_runtime_error(exc):
                    _non_retriable_block_reason = str(exc)
                    _non_retriable_block_until_mono = _mono() + float(
                        cfg.non_retriable_cooldown_sec
                    )
                    logger.warning("ensure_mt5 non-retriable failure: %s", exc)
                    raise

                logger.error(
                    "ensure_mt5 attempt %d/%d failed: %s",
                    attempt + 1,
                    int(cfg.max_retries),
                    exc,
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


def shutdown_mt5() -> None:
    """Graceful MT5 shutdown process."""
    global _initialized, _last_health_reason
    global _non_retriable_block_until_mono, _non_retriable_block_reason

    with _INIT_LOCK:
        _stop_mt5_async_thread()
        _mt5_shutdown_silent()
        _set_mt5_state(False, "shutdown", lock_timeout=3.0)
        _reset_hard_reset_probe()
        _non_retriable_block_until_mono = 0.0
        _non_retriable_block_reason = ""
        _mt5_dispatch_reset_runtime_state()

    _release_single_instance_lock()


def mt5_status() -> Tuple[bool, str]:
    """Retrieve fast-path readiness metric."""
    acquired = MT5_LOCK.acquire(timeout=1.0)
    if not acquired:
        return False, "status_lock_timeout"
    try:
        return bool(_initialized), str(_last_health_reason)
    finally:
        MT5_LOCK.release()


def mt5_dispatch_status() -> Dict[str, Any]:
    """Return bounded-dispatch health and pressure telemetry for diagnostics."""
    with _mt5_dispatch_cond:
        pending = [item for _, _, item in _mt5_dispatch_heap]

    queue_depth = int(len(pending))
    now_mono = _mono()
    oldest_age_ms = max(
        (
            max(0.0, (now_mono - float(item.enqueue_ts)) * 1000.0)
            for item in pending
        ),
        default=0.0,
    )
    queued_critical = sum(
        1 for item in pending if int(item.priority) <= _MT5_DISPATCH_PRIORITY_CRITICAL
    )
    queued_high = sum(
        1
        for item in pending
        if int(item.priority) == _MT5_DISPATCH_PRIORITY_HIGH
    )
    queued_normal = sum(
        1
        for item in pending
        if int(item.priority) == _MT5_DISPATCH_PRIORITY_NORMAL
    )
    queued_low = sum(
        1 for item in pending if int(item.priority) >= _MT5_DISPATCH_PRIORITY_LOW
    )

    with _mt5_dispatch_metrics_lock:
        metrics = dict(_mt5_dispatch_metrics)
    with _transport_state_lock:
        transport_error_streak = int(_transport_error_streak)
        transport_timeout_scale = float(_transport_timeout_scale)

    utilization = (
        float(queue_depth) / float(_MT5_ASYNC_QUEUE_MAXSIZE)
        if _MT5_ASYNC_QUEUE_MAXSIZE > 0
        else 0.0
    )
    worker_alive = bool(_mt5_async_thread and _mt5_async_thread.is_alive())
    return {
        "worker_alive": worker_alive,
        "stop_requested": bool(_mt5_async_stop.is_set()),
        "queue_depth": queue_depth,
        "queue_maxsize": int(_MT5_ASYNC_QUEUE_MAXSIZE),
        "queue_utilization": round(utilization, 4),
        "pressure": (
            "HIGH"
            if utilization >= 0.75
            else ("MEDIUM" if utilization >= 0.40 else "LOW")
        ),
        "oldest_age_ms": round(oldest_age_ms, 1),
        "stale_threshold_sec": float(_ASYNC_STALE_THRESHOLD_SEC),
        "lock_timeout_sec": float(_MT5_ASYNC_LOCK_TIMEOUT_SEC),
        "direct_fallback_depth_limit": int(_MT5_ASYNC_DIRECT_FALLBACK_MAX_DEPTH),
        "queued_critical": queued_critical,
        "queued_high": queued_high,
        "queued_normal": queued_normal,
        "queued_low": queued_low,
        "transport_error_streak": transport_error_streak,
        "transport_timeout_scale": round(transport_timeout_scale, 3),
        "enqueued_total": int(metrics.get("enqueued_total", 0) or 0),
        "completed_total": int(metrics.get("completed_total", 0) or 0),
        "failed_total": int(metrics.get("failed_total", 0) or 0),
        "stale_total": int(metrics.get("stale_total", 0) or 0),
        "overflow_total": int(metrics.get("overflow_total", 0) or 0),
        "preempted_total": int(metrics.get("preempted_total", 0) or 0),
        "cancelled_total": int(metrics.get("cancelled_total", 0) or 0),
        "lock_timeout_total": int(metrics.get("lock_timeout_total", 0) or 0),
        "inline_total": int(metrics.get("inline_total", 0) or 0),
        "direct_fallback_total": int(metrics.get("direct_fallback_total", 0) or 0),
        "direct_fallback_blocked_total": int(
            metrics.get("direct_fallback_blocked_total", 0) or 0
        ),
        "high_watermark": int(metrics.get("high_watermark", 0) or 0),
        "last_queue_depth": int(metrics.get("last_queue_depth", 0) or 0),
        "last_lock_wait_ms": round(
            float(metrics.get("last_lock_wait_ms", 0.0) or 0.0), 3
        ),
        "lock_wait_ema_ms": round(
            float(metrics.get("lock_wait_ema_ms", 0.0) or 0.0), 3
        ),
        "last_call_ms": round(float(metrics.get("last_call_ms", 0.0) or 0.0), 3),
        "call_ema_ms": round(float(metrics.get("call_ema_ms", 0.0) or 0.0), 3),
    }


atexit.register(shutdown_mt5)


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "MT5AuthError",
    "MT5ClientConfig",
    "MT5Credentials",
    "MT5GhostFillDetected",
    "MT5_LOCK",
    "ensure_mt5",
    "mt5",
    "mt5_async_call",
    "mt5_async_submit",
    "mt5_dispatch_status",
    "mt5_status",
    "safe_order_send",
    "shutdown_mt5",
]
