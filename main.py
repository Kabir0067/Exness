"""
Production entry point for the institutional trading runtime.

Coordinates startup checks, runtime bootstrap, thread supervision,
and graceful shutdown for the live stack.

Refactored for:
- Zero redundancy in thread restart logic (single helper, state-driven)
- Memory-efficient thread tracking via slots-based manager
- Unbreakable shutdown with guaranteed joins and fallbacks
- Enhanced type safety and professional documentation
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import traceback
from dataclasses import dataclass
from threading import Thread
from typing import Any, Callable, Optional

# Bootstrap must be imported first. It redirects stdout and installs hooks.
from runmain.bootstrap import (
    _REAL_STDERR,
    _REAL_STDOUT,
    GracefulShutdown,
    LogMonitor,
    SingletonInstance,
    bootstrap_runtime,
    controller_boot_report,
    log,
    sleep_interruptible,
    state,
)
from runmain.gate import (
    auto_train_models_strict,
    model_gate_ready_effective,
    models_ready,
    prime_engine_model_health,
    required_gate_assets,
)
from runmain.supervisors import (
    Notifier,
    NotifierLike,
    NullNotifier,
    run_engine_notify_worker,
    run_engine_supervisor,
    run_telegram_supervisor,
)

STARTUP_AUTO_TRAIN_ENABLED = True


# =============================================================================
# Classes (memory-optimized with slots)
# =============================================================================
@dataclass(slots=True)
class _ThreadState:
    """Lightweight container for supervised thread lifecycle."""

    thread: Optional[Thread] = None
    last_restart_ts: float = 0.0


class _SafeArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that safely writes encoded output."""

    def _print_message(self, message: str, file: Optional[object] = None) -> None:
        if not message:
            return
        target = file if file is not None else sys.stderr
        try:
            target.write(message)  # type: ignore[union-attr]
        except UnicodeEncodeError:
            enc = getattr(target, "encoding", None) or "utf-8"
            target.write(  # type: ignore[union-attr]
                message.encode(enc, errors="replace").decode(enc, errors="replace")
            )


# =============================================================================
# Public Entry Point
# =============================================================================
def main(argv: Optional[list[str]] = None) -> int:
    """Top-level entry point: parse args, acquire singleton lock, run main loop."""
    args = _parse_args(argv)
    try:
        with SingletonInstance(port=12345):
            return _main_inner(args)
    except RuntimeError as exc:
        print(f"FATAL: {exc}", file=_REAL_STDERR)
        return 1
    except Exception as exc:
        print(f"FATAL: Singleton check failed: {exc}", file=_REAL_STDERR)
        return 1


# =============================================================================
# Argument Parsing
# =============================================================================
def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    """Parse CLI arguments using real stdio to avoid redirected streams."""
    p = _SafeArgumentParser(
        description="XAUUSDm/BTCUSDm Institutional Trading System — Production Runner"
    )
    with contextlib.redirect_stdout(_REAL_STDOUT), contextlib.redirect_stderr(
        _REAL_STDERR
    ):
        return p.parse_args(argv)


# =============================================================================
# Thread Factories
# =============================================================================
def _spawn_engine_thread(stop_event: Any, notifier: NotifierLike) -> Thread:
    """Create (but do not start) the engine supervisor thread."""
    return Thread(
        target=run_engine_supervisor,
        args=(stop_event, notifier),
        name="engine.supervisor",
        daemon=False,
    )


def _spawn_bot_thread(stop_event: Any, notifier: NotifierLike) -> Thread:
    """Create (but do not start) the Telegram supervisor thread."""
    return Thread(
        target=run_telegram_supervisor,
        args=(stop_event, notifier),
        name="telegram.supervisor",
        daemon=False,
    )


def _spawn_notify_worker(stop_event: Any) -> Thread:
    """Create (but do not start) the engine notification worker thread."""
    return Thread(
        target=run_engine_notify_worker,
        args=(stop_event,),
        name="engine.notify_worker",
        daemon=True,
    )


# =============================================================================
# Thread Management Helpers (zero redundancy, slots-backed)
# =============================================================================
def _restart_thread_if_dead(
    thread_state: _ThreadState,
    spawn_func: Callable[[Any, NotifierLike], Thread],
    stop_event: Any,
    notifier: NotifierLike,
    now: float,
    cooldown: float,
    log_msg: str,
    notify_text: str,
) -> _ThreadState:
    """Restart a dead thread safely with cooldown enforcement (single source of truth)."""
    if thread_state.thread is not None and thread_state.thread.is_alive():
        return thread_state

    if (now - thread_state.last_restart_ts) < cooldown:
        return thread_state

    thread_state.last_restart_ts = now
    log.error(log_msg)

    try:
        notifier.notify(notify_text)
    except Exception as exc:
        log.warning("NOTIFIER_FAILED_DURING_RESTART | err=%s", exc)

    try:
        new_thread = spawn_func(stop_event, notifier)
        new_thread.start()
        thread_state.thread = new_thread
        return thread_state
    except Exception as exc:
        log.error("THREAD_SPAWN_FAILED | err=%s", exc)
        return thread_state


def _safe_join_thread(
    thread: Optional[Thread],
    timeout: float,
    label: str,
) -> None:
    """Join thread with timeout and structured logging."""
    if not thread or not thread.is_alive():
        return
    try:
        thread.join(timeout=timeout)
        if not thread.is_alive():
            log.info("THREAD_JOINED_CLEANLY | %s", label)
        else:
            log.warning("THREAD_JOIN_TIMEOUT | %s", label)
    except Exception as exc:
        log.warning("THREAD_JOIN_FAILED | %s | err=%s", label, exc)


# =============================================================================
# Startup: Model Checks + Auto-Training
# =============================================================================
def _run_startup_model_checks() -> None:
    """Execute model readiness checks and conditional auto-training."""
    ready, reason = models_ready()
    if not ready:
        log.warning("MODELS_MISSING | reason=%s", reason)
        if not STARTUP_AUTO_TRAIN_ENABLED:
            log.warning("STARTUP_AUTO_TRAIN_DISABLED | reason=%s", reason)
            prime_engine_model_health()
            return
        ok = auto_train_models_strict()
        if ok:
            post_ok, post_reason, active_assets = model_gate_ready_effective()
            missing = sorted(set(required_gate_assets()) - set(active_assets))
            if post_ok and missing:
                log.warning(
                    "Auto-training completed with partial gate | active=%s blocked=%s reason=%s",
                    ",".join(active_assets) or "-",
                    ",".join(missing) or "-",
                    post_reason,
                )
            else:
                log.info("Auto-training completed successfully")
        else:
            log.error("Auto-training failed; continuing with existing models")
    prime_engine_model_health()


# =============================================================================
# Main Orchestration Loop
# =============================================================================
def _main_inner(args: argparse.Namespace) -> int:
    """Core orchestration: bootstrap, wire components, supervise threads."""
    if not bootstrap_runtime(allow_telegram=True):
        return 1

    engine = state.engine
    shutdown = GracefulShutdown()

    _run_startup_model_checks()

    try:
        engine.print_startup_matrix()
    except Exception as exc:
        log.warning("STARTUP_MATRIX_PRINT_FAILED | err=%s", exc)

    notifier_wired = False
    if state.send_signal_notification is not None:
        try:
            engine.set_signal_notifier(state.send_signal_notification)
            notifier_wired = True
        except Exception as exc:
            log.warning("SIGNAL_NOTIFIER_WIRING_FAILED | err=%s", exc)

    log.info(
        "SIGNAL_NOTIFIER_WIRED | %s",
        "Bot connected" if notifier_wired else "disabled",
    )

    boot = controller_boot_report()
    if boot:
        log.info(
            "PRODUCTION_BOOT | controller=%s gate_reason=%s blocked=%s chaos=%s",
            str(boot.get("controller_state", "-") or "-"),
            str(boot.get("gate_reason", "") or ""),
            ",".join(boot.get("blocked_assets", ())) or "-",
            str(boot.get("chaos_state", "-") or "-"),
        )

    if state.tg_available:
        notifier: NotifierLike = Notifier(shutdown.stop_event, queue_max=200)
        notifier.start()  # type: ignore[attr-defined]
    else:
        notifier = NullNotifier()

    LogMonitor(shutdown.stop_event).start()

    notify_worker: Optional[Thread] = None
    if state.tg_available:
        notify_worker = _spawn_notify_worker(shutdown.stop_event)
        notify_worker.start()

    # Slots-backed state for zero-redundancy restarts
    engine_state = _ThreadState()
    bot_state = _ThreadState()

    RESTART_COOLDOWN = 2.0
    last_probe_ts = 0.0
    PROBE_INTERVAL = 30.0
    use_bot = bool(state.tg_available)

    try:
        engine_state.thread = _spawn_engine_thread(shutdown.stop_event, notifier)
        engine_state.thread.start()

        if use_bot:
            bot_state.thread = _spawn_bot_thread(shutdown.stop_event, notifier)
            bot_state.thread.start()
        else:
            log.info(
                "Telegram unavailable -> engine continues without bot"
                if not state.tg_available
                else "Telegram supervisor unavailable"
            )

        while not shutdown.stop_event.is_set():
            now = time.time()

            if (now - last_probe_ts) >= PROBE_INTERVAL:
                last_probe_ts = now
                try:
                    fn = getattr(engine, "manage_runtime_housekeeping", None)
                    if callable(fn):
                        fn()
                except Exception as exc:
                    log.warning("RUNTIME_HOUSEKEEPING_FAILED | err=%s", exc)

                try:
                    chaos_fn = getattr(engine, "run_chaos_audit", None)
                    if callable(chaos_fn):
                        chaos = chaos_fn()
                        if isinstance(chaos, dict) and not bool(
                            chaos.get("overall_passed", True)
                        ):
                            failed_scenarios = [
                                name
                                for name, payload in dict(
                                    chaos.get("scenarios", {})
                                ).items()
                                if isinstance(payload, dict)
                                and not bool(payload.get("passed", False))
                            ]
                            if failed_scenarios:
                                log.warning(
                                    "CHAOS_AUDIT_WARN | failed=%s",
                                    ",".join(failed_scenarios[:6]),
                                )
                except Exception as exc:
                    log.warning("CHAOS_AUDIT_FAILED | err=%s", exc)

            engine_state = _restart_thread_if_dead(
                engine_state,
                _spawn_engine_thread,
                shutdown.stop_event,
                notifier,
                now,
                RESTART_COOLDOWN,
                "ENGINE_SUPERVISOR_THREAD_EXITED | restarting",
                "⚠️ Нозири мотор қатъ шуд. Аз нав оғоз мешавад.",
            )

            if use_bot:
                bot_state = _restart_thread_if_dead(
                    bot_state,
                    _spawn_bot_thread,
                    shutdown.stop_event,
                    notifier,
                    now,
                    RESTART_COOLDOWN,
                    "TELEGRAM_SUPERVISOR_THREAD_EXITED | restarting",
                    "⚠️ Telegram supervisor қатъ шуд. Аз нав оғоз мешавад.",
                )

            shutdown.stop_event.wait(timeout=1.0)

        return 0

    except KeyboardInterrupt:
        log.info("Ctrl+C -> graceful shutdown")
        shutdown.request_stop()
        return 0

    except Exception as exc:
        log.error("Fatal main error: %s | tb=%s", exc, traceback.format_exc())
        try:
            notifier.notify(f"🛑 Хатои ҷиддии система: {exc}")
        except Exception:
            pass
        shutdown.request_stop()
        return 1

    finally:
        shutdown.request_stop()
        try:
            bot = state.bot
            if bot is not None:
                bot.stop_polling()
        except Exception as exc:
            log.warning("BOT_STOP_POLLING_FAILED | err=%s", exc)

        _safe_join_thread(bot_state.thread, 20.0, "telegram.supervisor")
        _safe_join_thread(notify_worker, 5.0, "engine.notify_worker")
        _safe_join_thread(engine_state.thread, 30.0, "engine.supervisor")

        try:
            notifier.notify("⏹️ Система қатъ шуд.")
        except Exception:
            pass

        sleep_interruptible(shutdown.stop_event, 0.2)


# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))