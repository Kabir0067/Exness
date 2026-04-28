"""
Runtime bootstrap, logging, and shared application state helpers.

Initializes process-level telemetry, stdout redirection, exception hooks,
and lazy runtime wiring for the live controller stack.
"""

from __future__ import annotations

import http.client
import logging
import os
import signal
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from threading import Event, Lock, Thread
from typing import Any, Callable, Optional

import urllib3

from log_config import (
    LOG_DIR as LOG_ROOT,
    attach_global_handler_to_loggers,
    configure_logging,
    configure_module_logs,
    get_log_path,
    log_dir_stats,
)

# Prevent logging internals from dumping recursive diagnostics to stderr.
logging.raiseExceptions = False


# =============================================================================
# Global State & Streams
# =============================================================================
LOG_DIR = LOG_ROOT

_REAL_STDOUT = getattr(sys, "__stdout__", sys.stdout)
_REAL_STDERR = getattr(sys, "__stderr__", sys.stderr)

_FMT = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

_system_log_handler = configure_logging(
    level="INFO", system_log_name=None, console=True
)

# Attach dedicated file handlers for modules that keep separate logs.
try:
    _n_attached = configure_module_logs()
    logging.getLogger("main").info(
        "MODULE_LOGS_CONFIGURED | attached=%d", int(_n_attached)
    )
except Exception as _mod_log_exc:
    # Keep boot alive even if per-module log wiring fails.
    logging.getLogger("main").error(
        "MODULE_LOGS_FAILED | err=%s", _mod_log_exc
    )

# Network exception groups (used by supervisors and notifier)
try:
    from requests.exceptions import (  # type: ignore
        ChunkedEncodingError,
        ConnectTimeout,
        ReadTimeout,
        RequestException,
    )
    from requests.exceptions import (
        ConnectionError as RequestsConnectionError,
    )
except Exception:
    RequestException = Exception  # type: ignore
    ReadTimeout = Exception  # type: ignore
    ConnectTimeout = Exception  # type: ignore
    RequestsConnectionError = Exception  # type: ignore
    ChunkedEncodingError = Exception  # type: ignore

NETWORK_EXC: tuple[type[Exception], ...] = (
    RequestException,  # type: ignore[misc]
    ConnectionError,
    TimeoutError,
    socket.gaierror,
    socket.timeout,
    OSError,
)

NET_ERRS_TG: tuple[type[Exception], ...] = (
    ReadTimeout,  # type: ignore[misc]
    ConnectTimeout,  # type: ignore[misc]
    RequestException,  # type: ignore[misc]
    urllib3.exceptions.ReadTimeoutError,
    urllib3.exceptions.ProtocolError,
    ConnectionError,
    RequestsConnectionError,  # type: ignore[misc]
    ChunkedEncodingError,  # type: ignore[misc]
    http.client.RemoteDisconnected,
)


# =============================================================================
# Data Structures
# =============================================================================
class AppState:
    """
    Single mutable container for all lazily-resolved runtime references.
    Attribute assignment is intentionally unrestricted so that
    ``bootstrap_runtime`` can wire references after import.
    """

    bot: Any = None
    ADMIN: int = 0
    bot_commands: Optional[Callable[[], None]] = None
    send_signal_notification: Optional[Callable[[str, Any], None]] = None
    engine: Any = None
    tg_available: bool = True

    # Sentinel: becomes True after bootstrap_runtime succeeds.
    _ready: bool = False

    def reset(self) -> None:  # useful in tests
        self.bot = None
        self.ADMIN = 0
        self.bot_commands = None
        self.send_signal_notification = None
        self.engine = None
        self.tg_available = True
        self._ready = False


state = AppState()


@dataclass(frozen=True)
class Backoff:
    """Exponential back-off with ceiling for connection attempts."""

    base: float = 1.0
    factor: float = 2.0
    max_delay: float = 60.0

    def delay(self, attempt: int) -> float:
        a = max(1, int(attempt))
        if a == 1:
            return min(self.max_delay, self.base)
        try:
            return min(self.max_delay, self.base * (self.factor ** (a - 1)))
        except Exception:
            return float(self.max_delay)


# =============================================================================
# Concurrency Utilities
# =============================================================================
class SingletonInstance:
    """
    Process-level singleton guard using a local TCP socket.
    Raises ``RuntimeError`` if another instance is already bound to *port*.
    """

    def __init__(self, port: int = 12345) -> None:
        self._port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._locked = False

    def __enter__(self) -> "SingletonInstance":
        try:
            self._sock.bind(("127.0.0.1", self._port))
            self._sock.listen(1)
            self._locked = True
            return self
        except OSError as exc:
            self._locked = False
            raise RuntimeError(
                f"Another instance is already running on port {self._port}"
            ) from exc

    def __exit__(self, *_: Any) -> None:
        if self._locked:
            try:
                self._sock.close()
            except Exception:
                pass


class RateLimiter:
    """Allow an action at most once per *interval_sec* per string key."""

    def __init__(self, interval_sec: float) -> None:
        self._interval = float(interval_sec)
        self._lock = Lock()
        self._last: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if (now - self._last.get(key, 0.0)) >= self._interval:
                self._last[key] = now
                return True
            return False


class GracefulShutdown:
    """
    Install SIGINT/SIGTERM handlers.
    First signal -> sets stop_event.
    Second signal -> hard SystemExit(2).
    """

    def __init__(self) -> None:
        self.stop_event = Event()
        self._received = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, frame: Any) -> None:
        if self._received:
            log.warning("Сигнали такрорӣ (%s). Баромад маҷбурӣ.", signum)
            raise SystemExit(2)
        self._received = True
        log.info("Сигнали қатъ (%s). Оромона қатъ мекунад...", signum)
        self.stop_event.set()

    def request_stop(self) -> None:
        self.stop_event.set()


def sleep_interruptible(stop_event: Event, seconds: float) -> None:
    """Sleep up to *seconds*, waking immediately if *stop_event* is set."""
    end = time.monotonic() + float(seconds)
    while not stop_event.is_set():
        left = end - time.monotonic()
        if left <= 0:
            return
        stop_event.wait(timeout=min(0.5, left))


# =============================================================================
# Telemetry Hooks & Daemons
# =============================================================================
class _StdToLogger:
    """
    Write-compatible shim that redirects stdout lines into a Python logger.
    Thread-safe; guards against recursive re-entry from logging internals.
    """

    def __init__(
        self, logger: logging.Logger, level: int, fallback_stream: Any
    ) -> None:
        self._logger = logger
        self._level = int(level)
        self._fallback = fallback_stream
        self._local = threading.local()

    def write(self, buf: str) -> int:
        s = str(buf)
        if not s:
            return 0
        if getattr(self._local, "busy", False):
            try:
                self._fallback.write(s)
                self._fallback.flush()
            except Exception:
                pass
            return len(s)
        self._local.busy = True
        try:
            for line in s.splitlines():
                line = line.rstrip()
                if line:
                    self._logger.log(self._level, line)
        except Exception:
            try:
                self._fallback.write(s)
                self._fallback.flush()
            except Exception:
                pass
        finally:
            self._local.busy = False
        return len(s)

    def flush(self) -> None:
        try:
            self._fallback.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return bool(self._fallback.isatty())
        except Exception:
            return False

    @property
    def encoding(self) -> str:
        return getattr(self._fallback, "encoding", "utf-8")

    def fileno(self) -> int:
        try:
            return int(self._fallback.fileno())
        except Exception:
            raise OSError("No file descriptor available")


class LogMonitor:
    """Background daemon thread that warns when log directory grows too large."""

    def __init__(
        self,
        stop_event: Event,
        *,
        interval: float = 300.0,
        max_mb: float = 512.0,
        max_files: int = 2000,
    ) -> None:
        self.stop_event = stop_event
        self.interval = interval
        self.max_mb = max_mb
        self.max_files = max_files
        self._t: Optional[Thread] = None
        self._rl = RateLimiter(600.0)

    def start(self) -> None:
        if self._t and self._t.is_alive():
            return
        self._t = Thread(target=self._worker, name="log.monitor", daemon=True)
        self._t.start()

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                total_bytes, file_count = log_dir_stats()
                total_mb = total_bytes / (1024.0 * 1024.0)
                if total_mb > self.max_mb or file_count > self.max_files:
                    if self._rl.allow("log_volume"):
                        log.warning(
                            "Log volume high | size=%.1fMB files=%s thresholds=(%.1fMB, %s)",
                            total_mb,
                            file_count,
                            self.max_mb,
                            self.max_files,
                        )
            except Exception as exc:
                if self._rl.allow("log_monitor_err"):
                    log.warning("Log monitor error: %s", exc)
            sleep_interruptible(self.stop_event, self.interval)


def _setup_exception_hooks() -> None:
    def _handle_exception(exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type in (KeyboardInterrupt, SystemExit):
            try:
                logging.getLogger("main").warning(
                    "PROCESS_EXIT_SIGNAL | type=%s code=%s",
                    getattr(exc_type, "__name__", str(exc_type)),
                    getattr(exc, "code", ""),
                )
            except Exception:
                pass
            cur_stderr = sys.stderr
            try:
                sys.stderr = _REAL_STDERR
                sys.__excepthook__(exc_type, exc, tb)
            finally:
                sys.stderr = cur_stderr
            return

        try:
            tb_txt = "".join(traceback.format_tb(tb)) if tb else ""
            logging.getLogger("main").error(
                "UNCAUGHT_EXCEPTION | %s | tb=%s", exc, tb_txt
            )
        except Exception:
            pass
        cur_stderr = sys.stderr
        try:
            sys.stderr = _REAL_STDERR
            sys.__excepthook__(exc_type, exc, tb)
        finally:
            sys.stderr = cur_stderr

    sys.excepthook = _handle_exception

    if hasattr(threading, "excepthook"):

        def _thread_excepthook(args: Any) -> None:
            tb_txt = ""
            try:
                tb_txt = "".join(
                    traceback.format_tb(getattr(args, "exc_traceback", None))
                )
            except Exception:
                pass
            try:
                logging.getLogger("main").error(
                    "THREAD_EXCEPTION | thread=%s exc=%s | tb=%s",
                    getattr(args, "thread", None),
                    getattr(args, "exc_value", None),
                    tb_txt,
                )
            except Exception:
                pass

        threading.excepthook = _thread_excepthook


# =============================================================================
# Setup Helpers (Public)
# =============================================================================
def setup_named_logger(
    name: str,
    *,
    file_name: str,
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backups: int = 5,
) -> logging.Logger:
    """
    Create (or retrieve) a named logger with a rotating file handler
    and a console (stdout) handler. Idempotent — safe to call multiple times.
    """
    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.propagate = False

    target_file = str(get_log_path(file_name))
    has_file = has_console = False

    for h in list(lg.handlers):
        if (
            isinstance(h, RotatingFileHandler)
            and getattr(h, "baseFilename", "") == target_file
        ):
            has_file = True
            h.setLevel(level)
            h.setFormatter(_FMT)
        elif isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.FileHandler
        ):
            has_console = True
            h.setLevel(level)
            h.setFormatter(_FMT)

    if not has_file:
        fh = RotatingFileHandler(
            target_file,
            maxBytes=int(max_bytes),
            backupCount=int(backups),
            encoding="utf-8",
            delay=True,
        )
        fh.setLevel(level)
        fh.setFormatter(_FMT)
        lg.addHandler(fh)

    if not has_console:
        ch = logging.StreamHandler(_REAL_STDOUT)
        ch.setLevel(level)
        ch.setFormatter(_FMT)
        lg.addHandler(ch)

    return lg


# Module-level loggers eagerly initialized
log = setup_named_logger("main", file_name="main.log", level=logging.INFO, backups=5)
log_super = setup_named_logger(
    "telegram.supervisor", file_name="telegram.log", level=logging.INFO, backups=3
)
log_stdout = setup_named_logger(
    "stdout", file_name="stdout.log", level=logging.INFO, backups=3
)

# Set up global overrides
_setup_exception_hooks()
sys.stdout = _StdToLogger(log_stdout, logging.INFO, _REAL_STDOUT)
sys.stderr = _REAL_STDERR


def env_truthy(name: str, default: str = "0") -> bool:
    """Safe boolean resolution for environment variables."""
    return str(os.environ.get(name, default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def env_float(name: str, default: float) -> float:
    """Safe float resolution for environment variables."""
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


# =============================================================================
# Runtime Bootstrap (Lazy Eval)
# =============================================================================
def _preflight_config() -> bool:
    """Strict live preflight for the single supported startup mode."""
    try:
        from core.config import preflight_env

        ok, missing, msg = preflight_env()
        missing = list(missing or [])
        if not ok:
            if msg:
                print(msg)
            if missing:
                print(f"Missing env vars: {', '.join(missing)}")
            return False
        return True
    except Exception as exc:
        print(f"Config preflight failed: {exc}")
        return False


def _safe_import_engine() -> Any:
    """Isolated engine import with fallback."""
    try:
        from Bot.portfolio_engine import engine as _engine

        return _engine
    except Exception as exc:
        log.error("Engine import failed: %s", exc)
        return None


def bootstrap_runtime(*, allow_telegram: bool = True) -> bool:
    """
    Lazily import and wire all runtime components into `state`.
    Returns True on success, False on unrecoverable failure.
    Idempotent: safe to call multiple times.
    """
    if not allow_telegram:
        state.tg_available = False

    if state._ready and (state.bot is not None or not state.tg_available):
        return True

    if not _preflight_config():
        return False

    try:
        _engine = _safe_import_engine()
        if _engine is None:
            return False

        if state.tg_available:
            from core.config import get_config_from_env as _get_runtime_config

            runtime_cfg = _get_runtime_config("XAU")
            token = str(getattr(runtime_cfg, "telegram_token", "") or "").strip()
            admin = int(getattr(runtime_cfg, "admin_id", 0) or 0)
            if not (token and admin):
                raise RuntimeError("Telegram credentials missing")

        if state.tg_available:
            from Bot.bot import (
                ADMIN as _admin,
            )
            from Bot.bot import (
                bot as _bot,
            )
            from Bot.bot import (
                bot_commands as _bot_commands,
            )
            from Bot.bot import (
                send_signal_notification as _send_signal_notification,
            )

            state.bot = _bot
            state.ADMIN = int(_admin or 0)
            state.bot_commands = _bot_commands
            state.send_signal_notification = _send_signal_notification
        else:
            state.bot = None
            state.ADMIN = 0
            state.bot_commands = None
            state.send_signal_notification = None

    except Exception as exc:
        log.error("Runtime import failed; full-stack startup aborted: %s", exc)
        return False

    state.engine = _engine

    try:
        state.engine.dry_run = False
    except Exception:
        pass

    log.info("ENGINE_MODE | dry_run=False")

    log.info(
        "STARTUP_MODE | mode=full_live_stack telegram=%s",
        "enabled" if state.tg_available else "disabled",
    )

    try:
        attach_global_handler_to_loggers(_system_log_handler)
    except Exception:
        pass

    state._ready = True
    return True


def controller_boot_report() -> dict[str, Any]:
    """Return a structured startup snapshot from the engine controller."""
    engine = state.engine
    if engine is None:
        return {}
    for attr in ("controller_snapshot", "runtime_watchdog_snapshot"):
        fn = getattr(engine, attr, None)
        if callable(fn):
            try:
                snap = fn()
                if isinstance(snap, dict):
                    return snap if attr == "controller_snapshot" else {"runtime": snap}
            except Exception:
                pass
    return {}


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "AppState",
    "Backoff",
    "GracefulShutdown",
    "LOG_DIR",
    "LogMonitor",
    "NET_ERRS_TG",
    "NETWORK_EXC",
    "RateLimiter",
    "SingletonInstance",
    "bootstrap_runtime",
    "controller_boot_report",
    "env_float",
    "env_truthy",
    "log",
    "log_stdout",
    "log_super",
    "setup_named_logger",
    "sleep_interruptible",
    "state",
]
