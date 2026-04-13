"""
main.py вЂ” Production entry point for the XAUUSDm Institutional Trading System.

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
    python main.py                  # full live stack: engine + analytics + Telegram
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from threading import Thread
from typing import Optional

# Bootstrap must be imported first вЂ” it redirects stdout and installs exception hooks.
from runmain.bootstrap import (
    GracefulShutdown,
    LogMonitor,
    SingletonInstance,
    _REAL_STDERR,
    _REAL_STDOUT,
    controller_boot_report,
    bootstrap_runtime,
    env_truthy,
    log,
    sleep_interruptible,
    state,
)
from runmain.gate import (
    auto_train_models_strict,
    log_monitoring_only_profile,
    models_ready,
    monitoring_only_mode,
    prime_engine_model_health,
)
from runmain.supervisors import (
    NullNotifier,
    Notifier,
    NotifierLike,
    run_engine_notify_worker,
    run_engine_supervisor,
    run_telegram_supervisor,
)


# ===========================================================================
# Argument parsing
# ===========================================================================
class _SafeArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that encodes messages safely before writing them."""

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


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    p = _SafeArgumentParser(
        description="XAUUSDm Institutional Scalping System вЂ” Production Runner"
    )
    # Legacy flags are accepted for old shortcuts, but production startup is
    # intentionally one path now: python main.py starts the full stack.
    p.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--engine-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)

    # Parse against the real stdio to avoid our redirected stdout/stderr.
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = _REAL_STDOUT
        sys.stderr = _REAL_STDERR
        return p.parse_args(argv)
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


def _telegram_disabled(args: argparse.Namespace) -> bool:
    return bool(
        env_truthy("DISABLE_TELEGRAM", "0")
        or env_truthy("HEADLESS", "0")
        or env_truthy("ENGINE_ONLY", "0")
    )


def _effective_dry_run(args: argparse.Namespace) -> bool:
    return bool(
        env_truthy("DRY_RUN", "0")
        or bool(getattr(state.engine, "dry_run", False))
    )


def _legacy_cli_mode_requested(args: argparse.Namespace) -> list[str]:
    modes: list[str] = []
    if getattr(args, "dry_run", False):
        modes.append("--dry-run")
    if getattr(args, "headless", False):
        modes.append("--headless")
    if getattr(args, "engine_only", False):
        modes.append("--engine-only")
    return modes


# ===========================================================================
# Thread factories
# ===========================================================================
def _spawn_engine_thread(stop_event, notifier: NotifierLike) -> Thread:
    return Thread(
        target=run_engine_supervisor,
        args=(stop_event, notifier),
        name="engine.supervisor",
        daemon=False,
    )


def _spawn_bot_thread(stop_event, notifier: NotifierLike) -> Thread:
    return Thread(
        target=run_telegram_supervisor,
        args=(stop_event, notifier),
        name="telegram.supervisor",
        daemon=False,
    )


def _spawn_notify_worker(stop_event) -> Thread:
    return Thread(
        target=run_engine_notify_worker,
        args=(stop_event,),
        name="engine.notify_worker",
        daemon=True,
    )


# ===========================================================================
# Startup: model checks + auto-training
# ===========================================================================
def _run_startup_model_checks(*, dry_run: bool) -> None:
    engine = state.engine
    if dry_run:
        try:
            engine.dry_run = True
        except Exception:
            pass
        prime_engine_model_health()
        log.info("DRY_RUN_STARTUP | skipping strict model preflight")
        return

    ready, reason = models_ready()
    if not ready:
        log.warning("MODELS_MISSING | reason=%s", reason)
        if monitoring_only_mode():
            log.warning("MONITORING_ONLY_PROFILE | skipping startup auto-training")
        elif env_truthy("AUTO_TRAIN_ON_STARTUP", "1"):
            ok = auto_train_models_strict()
            if ok:
                log.info("Auto-training completed successfully")
            else:
                log.error("Auto-training failed; continuing startup with existing models")
        else:
            log.warning("AUTO_TRAIN_ON_STARTUP=0 | skipping startup retraining")

    prime_engine_model_health()


# ===========================================================================
# Main orchestration loop
# ===========================================================================
def _main_inner(args: argparse.Namespace) -> int:
    if not bootstrap_runtime(allow_telegram=not _telegram_disabled(args)):
        return 1

    engine = state.engine
    shutdown = GracefulShutdown()
    dry_run = _effective_dry_run(args)
    mon_only = monitoring_only_mode()
    legacy_modes = _legacy_cli_mode_requested(args)
    if legacy_modes:
        log.warning(
            "LEGACY_CLI_MODE_IGNORED | flags=%s | python main.py always starts the full stack; use env overrides for diagnostics",
            ",".join(legacy_modes),
        )

    # Apply monitoring-only profile
    if mon_only:
        try:
            engine.request_manual_stop()
        except Exception:
            pass
        log_monitoring_only_profile()

    # Model preflight
    _run_startup_model_checks(dry_run=dry_run)

    # Print human-readable startup matrix
    try:
        engine.print_startup_matrix()
    except Exception:
        pass

    # Wire signal notifier into engine
    notifier_wired = False
    if state.send_signal_notification is not None:
        try:
            engine.set_signal_notifier(state.send_signal_notification)
            notifier_wired = True
        except Exception:
            pass
    log.info(
        "SIGNAL_NOTIFIER_WIRED | %s",
        "Bot connected" if notifier_wired else "disabled",
    )

    # Log controller boot snapshot
    boot = controller_boot_report()
    if boot:
        log.info(
            "PRODUCTION_BOOT | controller=%s gate_reason=%s blocked=%s chaos=%s",
            str(boot.get("controller_state", "-") or "-"),
            str(boot.get("gate_reason", "") or ""),
            ",".join(boot.get("blocked_assets", ())) or "-",
            str(boot.get("chaos_state", "-") or "-"),
        )

    # --- Notifier -----------------------------------------------------------
    if state.tg_available:
        notifier: NotifierLike = Notifier(shutdown.stop_event, queue_max=200)
        notifier.start()  # type: ignore[attr-defined]
    else:
        notifier = NullNotifier()

    # --- Background monitors ------------------------------------------------
    LogMonitor(shutdown.stop_event).start()

    notify_worker: Optional[Thread] = None
    if state.tg_available:
        notify_worker = _spawn_notify_worker(shutdown.stop_event)
        notify_worker.start()

    # --- Engine + bot threads -----------------------------------------------
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
                "Telegram unavailable -> engine and analytics continue without bot polling"
                if not state.tg_available
                else "Telegram supervisor unavailable"
            )

        # Main watchdog loop
        while not shutdown.stop_event.is_set():
            now = time.time()

            # Periodic engine housekeeping + chaos audit
            if (now - last_probe_ts) >= PROBE_INTERVAL:
                last_probe_ts = now
                try:
                    fn = getattr(engine, "manage_runtime_housekeeping", None)
                    if callable(fn):
                        fn()
                except Exception:
                    pass
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
                except Exception:
                    pass

            # Engine thread watchdog
            if (
                not engine_thread.is_alive()
                and (now - last_engine_restart) >= RESTART_COOLDOWN
            ):
                last_engine_restart = now
                log.error("ENGINE_SUPERVISOR_THREAD_EXITED | restarting")
                try:
                    notifier.notify("⚠️ Нозири мотор қатъ шуд. Аз нав оғоз мешавад.")
                except Exception:
                    pass
                engine_thread = _spawn_engine_thread(shutdown.stop_event, notifier)
                engine_thread.start()

            # Bot thread watchdog
            if use_bot:
                if (
                    bot_thread is None or not bot_thread.is_alive()
                ) and (now - last_bot_restart) >= RESTART_COOLDOWN:
                    last_bot_restart = now
                    log.error("TELEGRAM_SUPERVISOR_THREAD_EXITED | restarting")
                    try:
                        notifier.notify(
                            "⚠️ Telegram supervisor қатъ шуд. Аз нав оғоз мешавад."
                        )
                    except Exception:
                        pass
                    bot_thread = _spawn_bot_thread(shutdown.stop_event, notifier)
                    bot_thread.start()

            shutdown.stop_event.wait(timeout=1.0)

        return 0

    except KeyboardInterrupt:
        log.info("Ctrl+C -> initiating graceful shutdown")
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

        # Stop bot polling
        try:
            bot = state.bot
            if bot is not None:
                bot.stop_polling()
        except Exception:
            pass

        # Join threads with timeouts
        for thread, timeout in [
            (bot_thread, 20.0),
            (notify_worker, 5.0),
            (engine_thread, 30.0),
        ]:
            if thread and thread.is_alive():
                try:
                    thread.join(timeout=timeout)
                except Exception:
                    pass

        try:
            notifier.notify("⏹️ Система қатъ шуд.")
        except Exception:
            pass
        sleep_interruptible(shutdown.stop_event, 0.2)


# ===========================================================================
# Public entry point
# ===========================================================================
def main(argv: Optional[list[str]] = None) -> int:
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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

