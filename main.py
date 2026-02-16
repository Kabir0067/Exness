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

bot: Any = None
ADMIN: int = 0
bot_commands: Optional[Callable[[], None]] = None
send_signal_notification: Optional[Callable[[str, Any], None]] = None
engine: Any = None
_tg_available: bool = True
_system_log_handler = configure_logging(level="INFO", system_log_name="system.log", console=True)


def _env_truthy(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name, default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _models_ready() -> tuple[bool, str]:
    try:
        models_dir = get_artifact_dir("models")
        state_path = get_artifact_path("models", "model_state.pkl")
        if not state_path.exists():
            return False, f"missing_state:{state_path}"
        try:
            import pickle
            with state_path.open("rb") as f:
                state = pickle.load(f)
            if not isinstance(state, dict):
                return False, "state_invalid"
            if not bool(state.get("real_backtest", False)):
                return False, "non_real_backtest_artifact"
        except Exception as exc:
            return False, f"state_unreadable:{exc}"
        has_pkl = any(models_dir.glob("v*_institutional.pkl"))
        has_json = any(models_dir.glob("v*_institutional.json"))
        if not has_pkl or not has_json:
            return False, "missing_model_files"
        return True, "ok"
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
    _con("🔄 AUTO-TRAINING STARTED — This may take several minutes.")
    _con("   You will see iteration-by-iteration progress below.")
    _con("=" * 60 + "\n")

    try:
        from Backtest.engine import run_institutional_backtest
    except Exception as exc:
        log_local.error("Auto-train import failed: %s", exc)
        return False

    ok = True
    for asset in ("XAU", "BTC"):
        try:
            run_institutional_backtest(asset)
        except Exception as exc:
            log_local.error("Auto-train failed | asset=%s err=%s", asset, exc)
            ok = False

    if ok:
        artifacts_dir = str(get_artifact_dir("models"))
        _con("\n" + "=" * 60)
        _con(f"✅ TRAINING COMPLETE. Model saved to {artifacts_dir}")
        _con("   Starting Trading Engine...")
        _con("=" * 60 + "\n")
    else:
        _con("\n⚠️  TRAINING FINISHED WITH ERRORS. Check logs above.")
    return ok


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
            log.warning("Сигнал такрорӣ (%s). Баромади маҷбурӣ.", signum)
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
    retrain_assets = ["XAU", "BTC"]
    last_retrain_check = 0.0
    retraining_in_progress = False
    model_checker = ModelAgeChecker(get_artifact_path("models", "model_state.pkl"))

    notifier.notify("🧠 Engine supervisor started")

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

        if manual_stop:
            if manual_stop_rl.allow("engine_manual_stop"):
                log.info("Engine idle (manual stop active); supervisor waiting")
            sleep_interruptible(stop_event, 1.0)
            continue

        # HOURLY RETRAINING CHECK
        if not retraining_in_progress:
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
                    notifier.notify(f"🔄 Model Retraining Started\nAge: {age_txt}")

                    retraining_in_progress = True
                    try:
                        try:
                            engine._retraining_mode = True
                            log.info("Trading paused during retraining")
                        except Exception:
                            pass

                        success = _safe_retrain_models(notifier)

                        if success:
                            log.info("Retraining complete. Resuming trading.")
                            notifier.notify("✅ Retraining Complete\nNew model loaded")
                            try:
                                engine.reload_model()
                            except Exception as exc:
                                log.error("Model reload failed: %s", exc)
                        else:
                            log.warning("Retraining failed. Using old model.")
                            notifier.notify("⚠️ Retraining Failed\nContinuing with old model")
                    finally:
                        try:
                            engine._retraining_mode = False
                        except Exception:
                            pass
                        retraining_in_progress = False

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
                    notifier.notify("🟢 Мотори тиҷорат оғоз шуд")
                attempt = 0
            except Exception as exc:
                delay = backoff.delay(attempt)
                if attempt == 1 or attempt % 5 == 0:
                    log.error("Engine start failed: %s | retry in %.1fs", exc, delay)
                    notifier.notify(f"⚠️ Engine start failed: {exc} | retry in {delay:.0f}s")
                sleep_interruptible(stop_event, delay)
                continue

        # Monitor connected health
        if not ok_connected:
            if restart_guard.allow("engine_unhealthy"):
                log.warning("Engine unhealthy (connected=%s trading=%s) -> waiting/recovering", ok_connected, ok_trading)
                if TG_HEALTH_NOTIFY:
                    notifier.notify("🟠 Engine unhealthy -> waiting for reconnect")
            sleep_interruptible(stop_event, 2.0)
            continue

        sleep_interruptible(stop_event, 1.0)

    # Shutdown
    try:
        engine.stop()
    except Exception as exc:
        log.error("Engine stop error: %s | tb=%s", exc, traceback.format_exc())
    notifier.notify("🔴 Мотори тиҷорат қатъ шуд")


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

    try:
        # Retrain XAU
        log.info("Retraining XAU model...")
        xau_result = run_institutional_backtest("XAU")
        if not _check_backtest_passed(xau_result):
            log.error("XAU retraining failed quality checks")
            _restore_model_artifacts(backup_dir, models_dir)
            return False

        # Retrain BTC
        log.info("Retraining BTC model...")
        btc_result = run_institutional_backtest("BTC")
        if not _check_backtest_passed(btc_result):
            log.error("BTC retraining failed quality checks")
            _restore_model_artifacts(backup_dir, models_dir)
            return False

        return True
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
        sharpe >= 1.5 and
        win_rate >= 0.55 and
        max_dd <= 0.25
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

    notifier.notify("🚀 Боти Telegram ОҒОЗ ШУД")

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

    notifier.notify("⏹️ Telegram бот қатъ шуд")


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

    # Preflight: ensure trained models exist (auto-train if missing)
    ready, reason = _models_ready()
    if not ready:
        log.warning("MODELS_MISSING | reason=%s", reason)
        ok = _auto_train_models()
        if ok:
            log.info("Auto-training completed")
        else:
            log.error("Auto-training failed; continuing startup")

        try:
            engine._check_model_health()  # refresh gatekeeper after training
        except Exception:
            pass

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
        notifier.notify(f"🛑 Fatal main error: {exc}")
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

        notifier.notify("⏹️ System stopped")
        sleep_interruptible(shutdown.stop_event, 0.2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
