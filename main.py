from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from queue import Queue, Full, Empty
from threading import Event, Lock, Thread
from typing import Optional
    
from Bot.bot import ADMIN, bot, bot_commands
from Bot.engine import engine

# ==========================
# Logging (production safe)
# ==========================
LOG_DIR = "Logs"
os.makedirs(LOG_DIR, exist_ok=True)

log = logging.getLogger("main")
log.setLevel(logging.INFO)
log.propagate = False

if not log.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, "main.log"),
        maxBytes=int(os.getenv("MAIN_LOG_MAX_BYTES", "5242880")),  # 5MB
        backupCount=int(os.getenv("MAIN_LOG_BACKUPS", "5")),
        encoding="utf-8",
        delay=True,
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    log.addHandler(fh)
    log.addHandler(ch)


# ==========================
# Helpers
# ==========================
def sleep_interruptible(stop_event: Event, seconds: float) -> None:
    """Sleep, but exit early if stop_event set."""
    end = time.time() + float(seconds)
    while not stop_event.is_set():
        left = end - time.time()
        if left <= 0:
            return
        stop_event.wait(timeout=min(0.5, left))


@dataclass(frozen=True)
class Backoff:
    base: float = 1.0
    factor: float = 2.0
    max_delay: float = 60.0

    def delay(self, attempt: int) -> float:
        if attempt <= 1:
            return min(self.max_delay, self.base)
        return min(self.max_delay, self.base * (self.factor ** (attempt - 1)))


class RateLimiter:
    """Allow an action at most once per interval seconds (per key)."""

    def __init__(self, interval_sec: float) -> None:
        self.interval = float(interval_sec)
        self._lock = Lock()
        self._last: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            last = self._last.get(key, 0.0)
            if (now - last) >= self.interval:
                self._last[key] = now
                return True
            return False


# Network-ish exceptions (Telegram / HTTP / DNS / sockets)
try:
    import requests  # type: ignore
    from requests.exceptions import RequestException  # type: ignore
except Exception:  # pragma: no cover
    RequestException = Exception  # type: ignore

NETWORK_EXC = (
    RequestException,
    ConnectionError,
    TimeoutError,
    socket.gaierror,
    socket.timeout,
    OSError,
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
# Engine Supervisor (never crash)
# ==========================
def run_engine_supervisor(stop_event: Event, notifier: Notifier) -> None:
    """
    High-reliability engine supervisor:
    - Start engine with backoff if MT5/internet not ready
    - Monitor loop; if engine stops/fails -> restart with guarded backoff
    - Never raises; never restart-storms
    """
    backoff = Backoff(base=2.0, factor=2.0, max_delay=60.0)
    restart_guard = RateLimiter(20.0)  # at most one restart attempt per 20s
    manual_stop_rl = RateLimiter(60.0)  # throttle manual-stop logs
    started_once = False
    attempt = 0

    notifier.notify("🧠 Engine supervisor started")

    while not stop_event.is_set():
        # Ensure engine running
        try:
            attempt += 1
            engine.start()
            if not started_once:
                started_once = True
                notifier.notify("🟢 Мотори тиҷорат оғоз шуд")
            attempt = 0  # reset on successful start
        except Exception as exc:
            delay = backoff.delay(attempt)
            if attempt == 1 or attempt % 5 == 0:
                log.error("Engine start failed: %s | retry in %.1fs", exc, delay)
                notifier.notify(f"⚠️ Engine start failed: {exc} | retry in {delay:.0f}s")
            sleep_interruptible(stop_event, delay)
            continue

        # Monitor phase
        try:
            # Prefer status() if available; never crash if status has issues
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

            if not ok_connected:
                # restart guarded
                if restart_guard.allow("engine_restart"):
                    log.warning("Engine unhealthy (connected=%s trading=%s) -> restarting", ok_connected, ok_trading)
                    notifier.notify("🟠 Engine unhealthy -> restart")
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    sleep_interruptible(stop_event, 2.0)
                else:
                    sleep_interruptible(stop_event, 2.0)
                continue

            if not ok_trading:
                if manual_stop:
                    if manual_stop_rl.allow("engine_manual_stop"):
                        log.info("Engine idle (manual stop active); supervisor waiting")
                    sleep_interruptible(stop_event, 2.0)
                    continue

                if restart_guard.allow("engine_restart"):
                    log.warning("Engine unhealthy (connected=%s trading=%s) -> restarting", ok_connected, ok_trading)
                    notifier.notify("🟠 Engine unhealthy -> restart")
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    sleep_interruptible(stop_event, 2.0)
                else:
                    sleep_interruptible(stop_event, 2.0)
                continue

            sleep_interruptible(stop_event, 1.0)

        except Exception as exc:
            # Engine monitor must never crash
            log.error("Engine supervisor error: %s | tb=%s", exc, traceback.format_exc())
            sleep_interruptible(stop_event, 2.0)

    # Shutdown
    try:
        engine.stop()
    except Exception as exc:
        log.error("Engine stop error: %s | tb=%s", exc, traceback.format_exc())
    notifier.notify("🔴 Мотори тиҷорат қатъ шуд")


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
    # Best-effort set commands (do not block startup)
    try:
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
        attempt += 1
        try:
            # If this blocks forever, stop_event will call stop_polling() in finally of main.
            bot.infinity_polling(timeout=20, long_polling_timeout=20)
            attempt = 0  # reset if returned cleanly
        except NETWORK_EXC as exc:
            delay = backoff.delay(attempt)
            if log_rl.allow("tg_net"):
                log.warning("Telegram network down (throttled): %s | retry in %.1fs", exc, delay)
            sleep_interruptible(stop_event, delay)
        except Exception as exc:
            delay = backoff.delay(attempt)
            log.error("Telegram polling error: %s | retry in %.1fs | tb=%s", exc, delay, traceback.format_exc())
            sleep_interruptible(stop_event, delay)

    # best-effort stop
    try:
        bot.stop_polling()
    except Exception:
        pass

    notifier.notify("⏹️ Telegram бот қатъ шуд")


# ==========================
# Main
# ==========================
def main(argv: Optional[list[str]] = None) -> int:
    shutdown = GracefulShutdown()

    parser = argparse.ArgumentParser(description="XAUUSDm Scalping System (Exness MT5) - Production Runner")
    parser.add_argument("--headless", action="store_true", help="Оғоз бе Telegram (ҳолати VPS)")
    parser.add_argument("--engine-only", action="store_true", help="Фақат мотор, бе бот")
    args = parser.parse_args(argv)

    notifier = Notifier(shutdown.stop_event, queue_max=200)
    notifier.start()

    notifier.notify("✅ System boot")

    engine_thread = Thread(
        target=run_engine_supervisor,
        args=(shutdown.stop_event, notifier),
        name="engine.supervisor",
        daemon=False,
    )

    bot_thread: Optional[Thread] = None

    try:
        engine_thread.start()

        if not (args.headless or args.engine_only):
            bot_thread = Thread(
                target=run_telegram_supervisor,
                args=(shutdown.stop_event, notifier),
                name="telegram.supervisor",
                daemon=False,
            )
            bot_thread.start()

        # Main wait loop (signals handled here)
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

        # Stop Telegram polling best-effort (prevents hang)
        try:
            bot.stop_polling()
        except Exception:
            pass

        # Join threads with timeouts (never hang forever)
        try:
            if bot_thread and bot_thread.is_alive():
                bot_thread.join(timeout=20.0)
        except Exception:
            pass

        try:
            if engine_thread.is_alive():
                engine_thread.join(timeout=30.0)
        except Exception:
            pass

        notifier.notify("⏹️ System stopped")
        sleep_interruptible(shutdown.stop_event, 0.5)  # give notifier a moment


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
