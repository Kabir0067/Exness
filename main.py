"""
main.py — Production entry point for the XAUUSDm Institutional Trading System.

Responsibilities
----------------
1. Parse CLI arguments
2. Enforce single-instance lock
3. Call bootstrap_runtime() to wire all components
4. Run startup model checks / auto-training
5. Spawn and supervise engine + Telegram threads
6. Handle graceful shutdown

Usage
-----
    python main.py  # full live stack: engine + analytics + Telegram
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from threading import Thread
from typing import Optional

# Bootstrap must be imported first — it redirects stdout and installs exception hooks.
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
    models_ready,
    prime_engine_model_health,
)
from runmain.supervisors import (
    Notifier,
    NotifierLike,
    NullNotifier,
    run_engine_notify_worker,
    run_engine_supervisor,
    run_telegram_supervisor,
)


# =============================================================================
# Classes
# =============================================================================
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
        print(
            f"FATAL: Singleton check failed: {exc}",
            file=_REAL_STDERR,
        )
        return 1


# =============================================================================
# Argument Parsing
# =============================================================================
def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    """Parse CLI arguments using real stdio to avoid redirected streams."""
    p = _SafeArgumentParser(
        description="XAUUSDm Institutional Scalping System — Production Runner"
    )

    # Use real stdio to avoid redirected streams
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = _REAL_STDOUT
        sys.stderr = _REAL_STDERR
        return p.parse_args(argv)
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
# =============================================================================
# Thread Factories
# =============================================================================
def _spawn_engine_thread(stop_event, notifier: NotifierLike) -> Thread:
    """Create (but do not start) the engine supervisor thread."""
    return Thread(
        target=run_engine_supervisor,
        args=(stop_event, notifier),
        name="engine.supervisor",
        daemon=False,
    )


def _spawn_bot_thread(stop_event, notifier: NotifierLike) -> Thread:
    """Create (but do not start) the Telegram supervisor thread."""
    return Thread(
        target=run_telegram_supervisor,
        args=(stop_event, notifier),
        name="telegram.supervisor",
        daemon=False,
    )


def _spawn_notify_worker(stop_event) -> Thread:
    """Create (but do not start) the engine notification worker thread."""
    return Thread(
        target=run_engine_notify_worker,
        args=(stop_event,),
        name="engine.notify_worker",
        daemon=True,
    )


# =============================================================================
# Thread Management Helpers
# =============================================================================
def _restart_thread_if_dead(
    thread: Optional[Thread],
    spawn_func: callable,
    stop_event,
    notifier: NotifierLike,
    last_restart_ts: float,
    now: float,
    cooldown: float,
    log_msg: str,
    notify_text: str,
) -> tuple[Optional[Thread], float]:
    """Restart a dead thread safely with cooldown enforcement."""
    if thread is not None and thread.is_alive():
        return thread, last_restart_ts

    if (now - last_restart_ts) < cooldown:
        return thread, last_restart_ts

    last_restart_ts = now
    log.error(log_msg)

    try:
        notifier.notify(notify_text)
    except Exception as exc:
        log.warning("NOTIFIER_FAILED_DURING_RESTART | err=%s", exc)

    try:
        new_thread = spawn_func(stop_event, notifier)
        new_thread.start()
        return new_thread, last_restart_ts
    except Exception as exc:
        log.error("THREAD_SPAWN_FAILED | err=%s", exc)
        return thread, last_restart_ts


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
        ok = auto_train_models_strict()

        if ok:
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

    engine_thread = _spawn_engine_thread(shutdown.stop_event, notifier)
    bot_thread: Optional[Thread] = None

    RESTART_COOLDOWN = 2.0
    last_engine_restart = 0.0
    last_bot_restart = 0.0
    last_probe_ts = 0.0
    PROBE_INTERVAL = 30.0

    use_bot = bool(state.tg_available)

    try:
        engine_thread.start()

        if use_bot:
            bot_thread = _spawn_bot_thread(shutdown.stop_event, notifier)
            bot_thread.start()
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

            engine_thread, last_engine_restart = _restart_thread_if_dead(
                engine_thread,
                _spawn_engine_thread,
                shutdown.stop_event,
                notifier,
                last_engine_restart,
                now,
                RESTART_COOLDOWN,
                "ENGINE_SUPERVISOR_THREAD_EXITED | restarting",
                "⚠️ Нозири мотор қатъ шуд. Аз нав оғоз мешавад.",
            )

            if use_bot:
                bot_thread, last_bot_restart = _restart_thread_if_dead(
                    bot_thread,
                    _spawn_bot_thread,
                    shutdown.stop_event,
                    notifier,
                    last_bot_restart,
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

        _safe_join_thread(bot_thread, 20.0, "telegram.supervisor")
        _safe_join_thread(notify_worker, 5.0, "engine.notify_worker")
        _safe_join_thread(engine_thread, 30.0, "engine.supervisor")

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
