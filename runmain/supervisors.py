"""
runmain/supervisors.py — Thread supervisors and notification dispatch.

Provides the engine supervisor (monitoring runtime health and model gates),
the Telegram supervisor (polling), and the asynchronous notification worker.
These loops are robust to network failures with exponential backoffs.
"""

from __future__ import annotations

import logging
import time
import traceback
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Optional, Protocol, runtime_checkable

from .bootstrap import (
    NET_ERRS_TG,
    NETWORK_EXC,
    Backoff,
    RateLimiter,
    env_float,
    env_truthy,
    log,
    log_super,
    setup_named_logger,
    sleep_interruptible,
    state,
)
from .gate import (
    auto_retrain_enabled,
    model_gate_ready_effective,
    monitoring_only_mode,
    required_gate_assets,
    run_retraining_cycle,
)

_log_sv = setup_named_logger(
    "supervisors", file_name="supervisors.log", level=logging.INFO, backups=3
)


# =============================================================================
# Global Constants
# =============================================================================
TG_HEALTH_NOTIFY: bool = env_truthy("TG_HEALTH_NOTIFY", "0")


# =============================================================================
# Protocols
# =============================================================================
@runtime_checkable
class NotifierLike(Protocol):
    """Abstract interface for system notifications."""

    def notify(self, message: str) -> None: ...


# =============================================================================
# Classes
# =============================================================================
class Notifier:
    """
    Thread-safe Telegram notifier with:
    - bounded queue (drops on overflow; never blocks callers)
    - serialised bot.send_message calls (one lock)
    - exponential back-off on network errors
    - graceful drain on shutdown
    """

    def __init__(self, stop_event: Event, *, queue_max: int = 200) -> None:
        self._stop = stop_event
        self._q: Queue[str] = Queue(maxsize=int(queue_max))
        self._bot_lock = Lock()
        self._log_rl = RateLimiter(30.0)
        self._t: Optional[Thread] = None

    def start(self) -> None:
        """Start the background worker thread."""
        if self._t and self._t.is_alive():
            return
        self._t = Thread(target=self._worker, name="notifier", daemon=True)
        self._t.start()

    def notify(self, message: str) -> None:
        """Enqueue a message for delivery."""
        try:
            self._q.put_nowait(str(message))
        except Full:
            if self._log_rl.allow("notify_drop"):
                log.warning("Notifier queue full → notifications dropped (throttled)")

    def _send_once(self, msg: str) -> bool:
        """Attempt to deliver a single message."""
        bot = state.bot
        if bot is None:
            return False

        try:
            with self._bot_lock:
                bot.send_message(state.ADMIN, msg)
            return True

        except NETWORK_EXC as exc:
            if self._log_rl.allow("notify_net"):
                log.warning("Telegram notify network error (throttled): %s", exc)
            return False

        except Exception as exc:
            log.error(
                "Telegram notify error: %s | tb=%s",
                exc,
                traceback.format_exc(),
            )
            return False

    def _worker(self) -> None:
        """Background loop consuming the notification queue."""
        backoff = Backoff(base=1.5, factor=2.0, max_delay=60.0)
        pending: Optional[str] = None
        attempt = 0

        while not self._stop.is_set():
            try:
                if pending is None:
                    pending = self._q.get(timeout=0.5)
                    attempt = 0
            except Empty:
                continue

            if not pending:
                continue

            attempt += 1

            if self._send_once(pending):
                pending = None
                continue

            sleep_interruptible(self._stop, backoff.delay(attempt))

        # Drain remaining messages (best-effort)
        while True:
            try:
                self._q.get_nowait()
            except Empty:
                break


class NullNotifier:
    """No-op notifier for engine-only / headless mode."""

    def notify(self, message: str) -> None:
        """Log the notification output locally."""
        try:
            log.info("NOTIFY_DISABLED | %s", message)
        except Exception:
            pass


# =============================================================================
# Global Functions
# =============================================================================
def run_engine_supervisor(
    stop_event: Event, notifier: NotifierLike
) -> None:  # noqa: C901
    """Run the engine's main polling loop, checking health and retrain statuses."""
    engine = state.engine
    backoff = Backoff(base=2.0, factor=2.0, max_delay=60.0)
    restart_guard = RateLimiter(20.0)
    manual_stop_rl = RateLimiter(60.0)
    gate_block_rl = RateLimiter(30.0)

    runtime_recover_after = max(6.0, env_float("ENGINE_RUNTIME_RECOVER_SEC", 12.0))

    from core.model_engine import ModelAgeChecker
    from log_config import get_artifact_path

    RETRAIN_CHECK_INTERVAL = 3600.0
    GATE_RETRAIN_COOLDOWN = max(30.0, env_float("GATE_RETRAIN_COOLDOWN_SEC", 300.0))

    retrain_assets = list(required_gate_assets())
    model_checker = ModelAgeChecker(get_artifact_path("models", "model_state.pkl"))

    mon_only = monitoring_only_mode()
    auto_retrain = auto_retrain_enabled()
    dry_run = bool(getattr(engine, "dry_run", False))

    try:
        if dry_run:
            gate_ok_start, gate_reason_start, _ = (
                True,
                "dry_run_bypass",
                list(required_gate_assets()),
            )
        else:
            gate_ok_start, gate_reason_start, _ = model_gate_ready_effective()
    except Exception:
        gate_ok_start, gate_reason_start, _ = False, "gate_error", []

    last_retrain_check = (
        0.0 if env_truthy("AUTO_RETRAIN_ON_SUPERVISOR_STARTUP", "0") else time.time()
    )
    last_gate_retrain_ts = 0.0 if gate_ok_start else time.time()

    retraining = False
    started_once = False
    attempt = 0

    notifier.notify("🧠 Нозири мотор оғоз шуд.")

    if mon_only:
        log.warning("MONITORING_ONLY_SUPERVISOR | retraining disabled")
    elif not auto_retrain:
        log.warning("AUTO_RETRAIN_ENABLED=0 | retraining disabled")

    if not gate_ok_start:
        log.warning("Engine gate initially blocked | reason=%s", gate_reason_start)

    def _read_engine_status() -> tuple[bool, bool, bool, dict[str, Any]]:
        try:
            st = engine.status()
            connected = bool(getattr(st, "connected", True))
            trading = bool(getattr(st, "trading", True))
            manual = bool(getattr(st, "manual_stop", False))
        except Exception:
            connected, trading, manual = True, True, False

        snap: dict[str, Any] = {}
        try:
            fn = getattr(engine, "runtime_watchdog_snapshot", None)
            if callable(fn):
                raw = fn()
                if isinstance(raw, dict):
                    snap = raw
        except Exception:
            pass
        return connected, trading, manual, snap

    while not stop_event.is_set():
        connected, trading, manual_stop, snap = _read_engine_status()

        # ---- Age-based retraining ------------------------------------------
        try:
            if (
                auto_retrain
                and not retraining
                and not dry_run
                and trading
                and not manual_stop
            ):
                now = time.time()
                if (now - last_retrain_check) > RETRAIN_CHECK_INTERVAL:
                    last_retrain_check = now
                    if model_checker.needs_retraining(retrain_assets):
                        age = model_checker.get_model_age_hours()
                        age_txt = f"{age:.1f}h" if age is not None else "unknown"

                        log.warning("Model expired (%s) → retraining", age_txt)
                        notifier.notify(f"🔄 Бозомӯзии модел\nСинни модел: {age_txt}")

                        retraining = True
                        try:
                            run_retraining_cycle(
                                notifier, reason=f"age_expired:{age_txt}"
                            )
                        finally:
                            retraining = False

                        connected, trading, manual_stop, snap = _read_engine_status()
        except Exception as exc:
            log.error("Age retraining block error: %s", exc)

        # ---- Runtime health ------------------------------------------------
        try:
            thread_dead = bool(snap.get("thread_dead", False))
            stale = bool(snap.get("stale", False))
            hb_age = float(snap.get("heartbeat_age_sec", 0.0) or 0.0)
        except Exception:
            thread_dead = stale = False
            hb_age = 0.0

        if (thread_dead or stale) and not manual_stop:
            if restart_guard.allow("engine_runtime_watchdog"):
                log.error(
                    "Engine unhealthy | dead=%s stale=%s hb=%.1fs",
                    thread_dead,
                    stale,
                    hb_age,
                )

            recover_fn = getattr(engine, "_recover_all", None)
            try:
                if callable(recover_fn) and (
                    thread_dead or hb_age >= runtime_recover_after
                ):
                    recover_fn()
                    attempt = 0
            except Exception as exc:
                log.error("Recover failed: %s", exc)

            sleep_interruptible(stop_event, 1.0)
            continue

        # ---- Gate retraining -----------------------------------------------
        try:
            if (
                auto_retrain
                and not trading
                and not retraining
                and not dry_run
                and not manual_stop
            ):
                gate_ok, gate_reason, _ = model_gate_ready_effective()
                if not gate_ok:
                    elapsed = time.time() - last_gate_retrain_ts
                    if elapsed >= GATE_RETRAIN_COOLDOWN:
                        last_gate_retrain_ts = time.time()
                        retraining = True
                        try:
                            log.warning("Gate blocked → retrain | %s", gate_reason)
                            notifier.notify(
                                f"⛔ Дарвоза баста аст: {gate_reason}\n🔃 Бозомӯзӣ"
                            )
                            run_retraining_cycle(
                                notifier, reason=f"gate_blocked:{gate_reason}"
                            )
                        finally:
                            retraining = False
                    else:
                        remain = max(1.0, GATE_RETRAIN_COOLDOWN - elapsed)
                        if gate_block_rl.allow("gate_wait"):
                            log.warning("Gate blocked | retry in %.0fs", remain)
                        sleep_interruptible(stop_event, min(5.0, remain))
                        continue
        except Exception as exc:
            log.error("Gate block handler error: %s", exc)

        # ---- Manual stop ---------------------------------------------------
        if manual_stop:
            try:
                resume_fn = getattr(engine, "_system_resume_trading_if_safe", None)
                if callable(resume_fn) and bool(resume_fn("engine_supervisor")):
                    sleep_interruptible(stop_event, 0.5)
                    continue
            except Exception as exc:
                log.error("SYSTEM_AUTO_RESUME_CHECK_FAILED | err=%s", exc)
            try:
                if manual_stop_rl.allow("manual_idle"):
                    log.info("Engine in manual-stop mode")
            except Exception:
                pass
            sleep_interruptible(stop_event, 1.0)
            continue

        # ---- Start engine --------------------------------------------------
        if not trading:
            try:
                attempt += 1
                started = bool(engine.start())
                if not started:
                    raise RuntimeError("engine.start returned False")
                if not started_once:
                    started_once = True
                    notifier.notify("🟢 Мотор оғоз шуд")
                attempt = 0
                connected, trading, manual_stop, snap = _read_engine_status()
            except Exception as exc:
                delay = backoff.delay(attempt)
                if attempt == 1 or attempt % 5 == 0:
                    log.error("Start failed: %s", exc)
                sleep_interruptible(stop_event, delay)
                continue

        # ---- Connectivity --------------------------------------------------
        if not connected:
            try:
                if restart_guard.allow("engine_unhealthy"):
                    log.warning("Engine disconnected")
                recover_fn = getattr(engine, "_recover_all", None)
                if callable(recover_fn):
                    recover_fn()
            except Exception as exc:
                log.error("Connectivity recover failed: %s", exc)
            sleep_interruptible(stop_event, 2.0)
            continue

        sleep_interruptible(stop_event, 1.0)

    try:
        engine.stop()
    except Exception as exc:
        log.error("Engine stop error: %s", exc)

    notifier.notify("🔴 Мотор қатъ шуд")


def run_telegram_supervisor(stop_event: Event, notifier: NotifierLike) -> None:
    """Supervise and restart the Telegram bot listener."""
    try:
        if callable(state.bot_commands):
            state.bot_commands()
    except Exception:
        pass

    notifier.notify("🚀 Telegram бот оғоз шуд")

    backoff = Backoff(base=1.0, factor=2.0, max_delay=60.0)
    log_rl = RateLimiter(20.0)
    attempt = 0

    while not stop_event.is_set():
        bot = state.bot
        if bot is None:
            sleep_interruptible(stop_event, 1.0)
            continue

        try:
            attempt += 1
            log_super.info("Polling attempt %s", attempt)

            bot.infinity_polling(
                timeout=75,
                long_polling_timeout=75,
                restart_on_change=False,
                skip_pending=True,
            )
            attempt = 0

        except NET_ERRS_TG as exc:
            delay = backoff.delay(attempt)
            if log_rl.allow("tg_net"):
                log_super.warning("Telegram network: %s", exc)
            sleep_interruptible(stop_event, delay)

        except Exception as exc:
            delay = backoff.delay(attempt)
            log.error("Telegram error: %s", exc)
            sleep_interruptible(stop_event, delay)

    try:
        if state.bot:
            state.bot.stop_polling()
    except Exception:
        pass

    notifier.notify("⏹️ Telegram бот қатъ шуд")


def run_engine_notify_worker(stop_event: Event) -> None:
    """Consume the global message queue and send Telegram notifications."""
    from Bot.bot_utils import get_notify_queue

    q = get_notify_queue()
    backoff = Backoff(base=1.0, factor=2.0, max_delay=30.0)
    log_rl = RateLimiter(30.0)

    pending: Optional[tuple[Any, str]] = None
    attempt = 0

    while not stop_event.is_set():
        try:
            if pending is None:
                pending = q.get(timeout=0.5)
                attempt = 0
        except Empty:
            continue

        if not pending:
            continue

        chat_id, msg = pending
        attempt += 1

        bot = state.bot
        if bot is None:
            pending = None
            continue

        try:
            bot.send_message(chat_id, msg, parse_mode="HTML")
            pending = None
        except NETWORK_EXC as exc:
            if log_rl.allow("notify_net"):
                log.warning("Engine notify network error: %s", exc)
            sleep_interruptible(stop_event, backoff.delay(attempt))
        except Exception as exc:
            log.error("Engine notify error: %s", exc)
            pending = None

    while True:
        try:
            q.get_nowait()
        except Empty:
            break


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "Notifier",
    "NotifierLike",
    "NullNotifier",
    "run_engine_notify_worker",
    "run_engine_supervisor",
    "run_telegram_supervisor",
]
