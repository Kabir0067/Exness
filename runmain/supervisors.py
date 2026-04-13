"""
supervisors.py - Runtime supervisors and notification infrastructure.

Components
----------
NotifierLike     : structural Protocol for notifier duck-typing
Notifier         : async-queue Telegram notifier (bounded, back-pressure-aware)
NullNotifier     : no-op notifier for headless / engine-only mode
run_engine_supervisor()         - watches engine health, gates, retraining
run_telegram_supervisor()       - keeps bot polling alive with back-off
run_engine_notify_worker()      - drains the engine-level notify queue
"""

from __future__ import annotations

import logging
import time
import traceback
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Optional, Protocol, runtime_checkable

from .bootstrap import (
    Backoff,
    NET_ERRS_TG,
    NETWORK_EXC,
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

# Opt-in: send health notifications to Telegram (disabled by default to reduce noise).
TG_HEALTH_NOTIFY: bool = env_truthy("TG_HEALTH_NOTIFY", "0")


# ===========================================================================
# Notifier protocol + implementations
# ===========================================================================
@runtime_checkable
class NotifierLike(Protocol):
    def notify(self, message: str) -> None: ...


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
        if self._t and self._t.is_alive():
            return
        self._t = Thread(target=self._worker, name="notifier", daemon=True)
        self._t.start()

    def notify(self, message: str) -> None:
        try:
            self._q.put_nowait(str(message))
        except Full:
            if self._log_rl.allow("notify_drop"):
                log.warning("Notifier queue full в†’ notifications dropped (throttled)")

    # ------------------------------------------------------------------
    def _send_once(self, msg: str) -> bool:
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
                "Telegram notify error: %s | tb=%s", exc, traceback.format_exc()
            )
            return False

    def _worker(self) -> None:
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

            if pending is None:
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
        try:
            log.info("NOTIFY_DISABLED | %s", message)
        except Exception:
            pass


# ===========================================================================
# Engine supervisor
# ===========================================================================
def run_engine_supervisor(stop_event: Event, notifier: NotifierLike) -> None:  # noqa: C901
    """
    Main engine watchdog loop.

    Responsibilities:
    - Starts / restarts the trading engine after failures
    - Monitors runtime thread health and triggers ``_recover_all``
    - Schedules model-age retraining
    - Triggers gate-blocked retraining with cooldown
    - Respects manual-stop state
    """
    engine = state.engine

    backoff = Backoff(base=2.0, factor=2.0, max_delay=60.0)
    restart_guard = RateLimiter(20.0)
    manual_stop_rl = RateLimiter(60.0)
    gate_block_rl = RateLimiter(30.0)

    runtime_recover_after = max(6.0, env_float("ENGINE_RUNTIME_RECOVER_SEC", 12.0))
    connect_grace = max(
        3.0, env_float("ENGINE_CONNECT_GRACE_SEC", max(8.0, runtime_recover_after))
    )

    from core.model_engine import MAX_MODEL_AGE_HOURS, ModelAgeChecker
    from log_config import get_artifact_path

    RETRAIN_CHECK_INTERVAL = 3600.0
    GATE_RETRAIN_COOLDOWN = max(30.0, env_float("GATE_RETRAIN_COOLDOWN_SEC", 300.0))
    retrain_assets = list(required_gate_assets())
    model_checker = ModelAgeChecker(get_artifact_path("models", "model_state.pkl"))

    mon_only = monitoring_only_mode()
    auto_retrain = auto_retrain_enabled()
    dry_run = bool(getattr(engine, "dry_run", False))

    if dry_run:
        gate_ok_start, gate_reason_start, gate_active_start = (
            True, "dry_run_bypass", list(required_gate_assets())
        )
    else:
        gate_ok_start, gate_reason_start, gate_active_start = model_gate_ready_effective()

    # Avoid slowing startup with an immediate age-based retrain. Startup already
    # does strict model preflight; recurring age checks can wait for a healthy
    # trading loop unless explicitly requested.
    last_retrain_check = (
        0.0 if env_truthy("AUTO_RETRAIN_ON_SUPERVISOR_STARTUP", "0") else time.time()
    )
    last_gate_retrain_ts = 0.0 if gate_ok_start else time.time()
    retraining = False
    started_once = False
    attempt = 0

    notifier.notify("🧠 Нозири мотор оғоз шуд.")

    if mon_only:
        log.warning("MONITORING_ONLY_SUPERVISOR | retraining disabled by profile")
    elif not auto_retrain:
        log.warning("AUTO_RETRAIN_ENABLED=0 | supervisor retraining disabled")

    if not gate_ok_start:
        log.warning(
            "Engine gate initially blocked | reason=%s | next retrain after %.0fs",
            gate_reason_start, GATE_RETRAIN_COOLDOWN,
        )
    else:
        req = set(required_gate_assets())
        active = set(gate_active_start)
        if not req.issubset(active):
            log.warning(
                "Engine gate started in partial mode | active=%s blocked=%s reason=%s",
                ",".join(gate_active_start) or "-",
                ",".join(sorted(req - active)) or "-",
                gate_reason_start,
            )

    # ------------------------------------------------------------------
    def _read_engine_status() -> tuple[bool, bool, bool, dict[str, Any]]:
        """Returns (connected, trading, manual_stop, runtime_snapshot)."""
        try:
            st = engine.status()
            connected = bool(getattr(st, "connected", True))
            trading = bool(getattr(st, "trading", True))
            manual = bool(getattr(st, "manual_stop", False))
        except Exception:
            connected = trading = True
            manual = False
        snap: dict[str, Any] = {}
        try:
            fn = getattr(engine, "runtime_watchdog_snapshot", None)
            if callable(fn):
                raw = fn()
                if isinstance(raw, dict):
                    snap = dict(raw)
        except Exception:
            pass
        return connected, trading, manual, snap

    # ------------------------------------------------------------------
    while not stop_event.is_set():
        connected, trading, manual_stop, snap = _read_engine_status()

        # ---- age-based retraining ----------------------------------------
        if auto_retrain and not retraining and not dry_run and trading and not manual_stop:
            now = time.time()
            if (now - last_retrain_check) > RETRAIN_CHECK_INTERVAL:
                last_retrain_check = now
                if model_checker.needs_retraining(retrain_assets):
                    age = model_checker.get_model_age_hours()
                    age_txt = f"{age:.1f}h" if age is not None else "unknown"
                    log.warning(
                        "Model expired (age: %s). Starting retraining...", age_txt
                    )
                    notifier.notify(
                        f"🔄 Бозомӯзии модел оғоз шуд.\nСинни модел: {age_txt}"
                    )
                    retraining = True
                    try:
                        run_retraining_cycle(notifier, reason=f"age_expired:{age_txt}")
                    finally:
                        retraining = False
                    connected, trading, manual_stop, snap = _read_engine_status()

        # ---- runtime thread health ----------------------------------------
        thread_dead = bool(snap.get("thread_dead", False))
        stale = bool(snap.get("stale", False))
        hb_age = float(snap.get("heartbeat_age_sec", 0.0) or 0.0)

        if (thread_dead or stale) and not manual_stop:
            if restart_guard.allow("engine_runtime_watchdog"):
                log.error(
                    "Engine runtime unhealthy (alive=%s stale=%s hb_age=%.1fs) -> recover",
                    snap.get("loop_alive", False), stale, hb_age,
                )
            recover_fn = getattr(engine, "_recover_all", None)
            if (thread_dead or hb_age >= runtime_recover_after) and callable(recover_fn):
                try:
                    recover_fn()
                    attempt = 0
                except Exception as exc:
                    log.error(
                        "Engine recover failed: %s | tb=%s", exc, traceback.format_exc()
                    )
            sleep_interruptible(stop_event, 1.0)
            continue

        # ---- gate-blocked retraining --------------------------------------
        if auto_retrain and not trading and not retraining and not dry_run and not manual_stop:
            gate_ok, gate_reason, _ = model_gate_ready_effective()
            if not gate_ok:
                elapsed = time.time() - last_gate_retrain_ts
                if elapsed >= GATE_RETRAIN_COOLDOWN:
                    last_gate_retrain_ts = time.time()
                    retraining = True
                    try:
                        log.warning(
                            "Engine gate blocked | reason=%s | triggering retrain",
                            gate_reason,
                        )
                        notifier.notify(
                            f"⛔ Дарвозаи модел баста аст: {gate_reason}\n"
                            f"🔃 Бозомӯзӣ оғоз мешавад..."
                        )
                        run_retraining_cycle(notifier, reason=f"gate_blocked:{gate_reason}")
                    finally:
                        retraining = False
                else:
                    remain = max(1.0, GATE_RETRAIN_COOLDOWN - elapsed)
                    if gate_block_rl.allow("gate_blocked_wait"):
                        log.warning(
                            "Engine gate blocked | reason=%s | next retrain in %.0fs",
                            gate_reason, remain,
                        )
                    sleep_interruptible(stop_event, min(5.0, remain))
                continue

        # ---- manual-stop idle --------------------------------------------
        if manual_stop:
            if (not trading) or thread_dead or stale:
                if restart_guard.allow("engine_monitoring_watchdog"):
                    log.warning(
                        "Engine monitoring runtime inactive (trading=%s alive=%s stale=%s hb_age=%.1fs) -> restart monitoring loop",
                        trading,
                        snap.get("loop_alive", False),
                        stale,
                        hb_age,
                    )
                try:
                    if (thread_dead or stale) and callable(getattr(engine, "_recover_all", None)):
                        engine._recover_all()
                    engine.start()
                    attempt = 0
                except Exception as exc:
                    log.error(
                        "Engine monitoring restart failed: %s | tb=%s",
                        exc,
                        traceback.format_exc(),
                    )
                sleep_interruptible(stop_event, 1.0)
                continue
            if manual_stop_rl.allow("engine_manual_stop"):
                unsafe_reason = ""
                fn = getattr(engine, "unsafe_account_state_snapshot", None)
                if callable(fn):
                    try:
                        unsafe_reason = str((fn() or {}).get("reason", "") or "")
                    except Exception:
                        pass
                if unsafe_reason:
                    log.warning(
                        "Engine idle (manual stop); unsafe_state=%s", unsafe_reason
                    )
                else:
                    log.info("Engine idle (manual stop active); supervisor waiting")
            sleep_interruptible(stop_event, 1.0)
            continue

        # ---- start engine if not trading ----------------------------------
        if not trading:
            try:
                attempt += 1
                started = bool(engine.start())
                if not started:
                    # Check if manual stop kicked in during start
                    try:
                        st_after = engine.status()
                        if bool(getattr(st_after, "manual_stop", False)):
                            attempt = 0
                            if manual_stop_rl.allow("engine_start_blocked_manual"):
                                log.warning(
                                    "Engine start blocked; switched to manual-stop mode"
                                )
                            sleep_interruptible(stop_event, 1.0)
                            continue
                    except Exception:
                        pass
                    raise RuntimeError("engine.start returned False")

                if not started_once:
                    started_once = True
                    notifier.notify("🟢 Мотори тиҷорат оғоз шуд.")
                attempt = 0
                connected, trading, manual_stop, snap = _read_engine_status()

            except Exception as exc:
                delay = backoff.delay(attempt)
                if attempt == 1 or attempt % 5 == 0:
                    log.error(
                        "Engine start failed: %s | retry in %.1fs", exc, delay
                    )
                    notifier.notify(
                        f"⚠️ Оғози мотор ноком шуд: {exc}\n"
                        f"⏳ Кӯшиши навбатӣ пас аз {delay:.0f}с"
                    )
                sleep_interruptible(stop_event, delay)
                continue

        # ---- connectivity grace / recover ---------------------------------
        if not connected:
            starting = bool(snap.get("starting", False))
            bootstrapping = bool(snap.get("bootstrapping", False))
            loop_ts = float(snap.get("loop_started_ts", 0.0) or 0.0)
            start_age = max(0.0, time.time() - loop_ts) if loop_ts > 0.0 else 0.0
            in_grace = starting or bootstrapping or (
                loop_ts > 0.0 and start_age < connect_grace
            )
            if in_grace:
                sleep_interruptible(stop_event, 1.0)
                continue
            if restart_guard.allow("engine_unhealthy"):
                log.warning(
                    "Engine unhealthy (connected=%s trading=%s) -> recovering",
                    connected, trading,
                )
                if TG_HEALTH_NOTIFY:
                    notifier.notify("🟠 Ҳолати мотор ноустувор, интизори барқароршавӣ...")
            fn = getattr(engine, "_recover_all", None)
            if callable(fn):
                try:
                    fn()
                except Exception as exc:
                    log.error(
                        "Connectivity recover failed: %s | tb=%s",
                        exc, traceback.format_exc(),
                    )
            sleep_interruptible(stop_event, 2.0)
            continue

        sleep_interruptible(stop_event, 1.0)

    # ---- graceful engine stop --------------------------------------------
    try:
        engine.stop()
    except Exception as exc:
        log.error("Engine stop error: %s | tb=%s", exc, traceback.format_exc())
    notifier.notify("🔴 Мотори тиҷорат қатъ шуд.")


# ===========================================================================
# Telegram supervisor
# ===========================================================================
def run_telegram_supervisor(stop_event: Event, notifier: NotifierLike) -> None:
    """
    Keeps bot.infinity_polling running, restarting after network errors
    with exponential back-off.
    """
    try:
        if callable(state.bot_commands):
            state.bot_commands()
    except NET_ERRS_TG as exc:  # type: ignore[misc]
        log.warning("bot_commands network error: %s (ignored)", exc)
    except Exception as exc:
        log.error("bot_commands error: %s | tb=%s", exc, traceback.format_exc())

    notifier.notify("🚀 Боти Telegram оғоз шуд.")

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
            log_super.info("Starting Telegram bot polling (attempt %s)...", attempt)
            bot.infinity_polling(
                timeout=75,
                long_polling_timeout=75,
                restart_on_change=False,
                skip_pending=True,
            )
            attempt = 0  # clean exit -> reset
        except NET_ERRS_TG as exc:  # type: ignore[misc]
            delay = backoff.delay(attempt)
            if log_rl.allow("tg_net"):
                log_super.warning(
                    "Telegram network unstable: %s | retry in %.1fs", exc, delay
                )
            sleep_interruptible(stop_event, delay)
        except Exception as exc:
            delay = backoff.delay(attempt)
            log.error(
                "Telegram polling error: %s | retry in %.1fs | tb=%s",
                exc, delay, traceback.format_exc(),
            )
            sleep_interruptible(stop_event, delay)

    try:
        bot = state.bot
        if bot is not None:
            bot.stop_polling()
    except Exception:
        pass

    notifier.notify("⏹️ Боти Telegram қатъ шуд.")


# ===========================================================================
# Engine notify worker
# ===========================================================================
def run_engine_notify_worker(stop_event: Event) -> None:
    """
    Drains the engine-side notify queue (chat_id, msg) pairs and delivers
    them via bot.send_message with back-off on network errors.
    """
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

        if pending is None:
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

    # Drain remaining items
    while True:
        try:
            q.get_nowait()
        except Empty:
            break
