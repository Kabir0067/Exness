"""
Autonomous operations watchtower for the live trading stack.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("bot.guardian")


HEARTBEAT_INTERVAL_SEC = max(
    300,
    int(os.getenv("GUARDIAN_HEARTBEAT_SEC", str(4 * 3600)) or (4 * 3600)),
)
ALERT_MIN_INTERVAL_SEC = max(
    60,
    int(os.getenv("GUARDIAN_ALERT_MIN_INTERVAL_SEC", "600") or 600),
)
DISCONNECT_GRACE_SEC = max(
    10,
    int(os.getenv("GUARDIAN_DISCONNECT_GRACE_SEC", "60") or 60),
)
SLIPPAGE_THRESHOLD_POINTS = float(
    os.getenv("GUARDIAN_SLIPPAGE_THRESHOLD_POINTS", "20") or 20.0
)
POLL_INTERVAL_SEC = max(
    5,
    int(os.getenv("GUARDIAN_POLL_INTERVAL_SEC", "30") or 30),
)


@dataclass
class _AlertThrottle:
    """Per-key alert throttle."""

    last_fire: Dict[str, float] = field(default_factory=dict)
    min_interval: int = ALERT_MIN_INTERVAL_SEC

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        last = float(self.last_fire.get(key, 0.0) or 0.0)
        if (now - last) < float(self.min_interval):
            return False
        self.last_fire[key] = now
        return True


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _format_duration(seconds: float) -> str:
    total = int(max(0.0, float(seconds)))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _position_profit(symbol: str, position_ticket: int) -> Optional[float]:
    """Return live open-position profit when the broker still exposes it."""
    if int(position_ticket or 0) <= 0:
        return None

    try:
        import MetaTrader5 as mt5

        positions = mt5.positions_get(symbol=str(symbol or "")) or []
    except Exception:
        return None

    for position in positions:
        try:
            if int(getattr(position, "ticket", 0) or 0) != int(position_ticket):
                continue
            return float(getattr(position, "profit", 0.0) or 0.0)
        except Exception:
            continue

    return None


def _format_heartbeat(snapshot: Dict[str, Any]) -> str:
    """Build the periodic heartbeat message."""
    status = str(snapshot.get("status", "unknown")).lower()
    icon = {"healthy": "OK", "warming": "WAIT", "problem": "ALERT"}.get(
        status,
        "INFO",
    )
    session = snapshot.get("session") or {}
    xau = session.get("xau") or {}
    btc = session.get("btc") or {}

    lines = [
        f"<b>GUARDIAN HEARTBEAT</b> [{icon}]",
        f"Uptime: <b>{snapshot.get('uptime_human', '-')}</b>",
        f"Controller: <b>{snapshot.get('controller_state', '-')}</b>",
        f"MT5: <b>{'connected' if snapshot.get('mt5_connected') else 'disconnected'}</b>",
        f"Trading: <b>{'active' if snapshot.get('trading') else 'paused'}</b>",
        f"Equity: <b>{_safe_float(snapshot.get('equity', 0.0)):,.2f}$</b>",
        f"Day P/L: <b>{_safe_float(snapshot.get('today_pnl', 0.0)):+,.2f}$</b>",
        f"Drawdown: <b>{_safe_float(snapshot.get('dd_pct', 0.0)) * 100:.2f}%</b>",
        f"Open positions: XAU=<b>{int(snapshot.get('open_xau', 0) or 0)}</b> | BTC=<b>{int(snapshot.get('open_btc', 0) or 0)}</b>",
    ]

    if xau or btc:
        lines.append(
            "Sessions: "
            f"XAU={'open' if xau.get('is_open') else 'closed'}"
            + (" (cooldown)" if xau.get("is_post_gap_cooldown") else "")
            + f" | BTC={'open' if btc.get('is_open') else 'closed'}"
        )

    gate_reason = str(snapshot.get("gate_reason", "") or "").strip()
    if gate_reason and gate_reason not in {"ok", "-"}:
        lines.append(f"Gate: <code>{gate_reason}</code>")

    return "\n".join(lines)


def _format_trade_alert(
    intent: Any,
    result: Any,
    *,
    day_pnl: float = 0.0,
    open_pnl: Optional[float] = None,
) -> str:
    """Build the execution alert."""
    try:
        symbol = str(getattr(intent, "symbol", "") or "-")
        side = str(getattr(intent, "signal", "") or "-")
        lot = _safe_float(getattr(intent, "lot", 0.0))
        price = _safe_float(getattr(result, "exec_price", 0.0))
        slippage = _safe_float(getattr(result, "slippage", 0.0))
        ok = bool(getattr(result, "ok", False))
        position_ticket = int(getattr(result, "position_ticket", 0) or 0)
    except Exception:
        return ""

    status = "EXECUTED" if ok else "FAILED"
    lines = [
        f"<b>TRADE {status}</b>",
        f"Asset: <b>{symbol}</b>",
        f"Side: <b>{side}</b>",
        f"Size: <b>{lot:.2f}</b>",
        f"Price: <b>{price:.5f}</b>",
        f"Slippage: <b>{slippage:.2f}</b> points",
        f"Day P/L: <b>{day_pnl:+.2f}$</b>",
    ]

    if open_pnl is not None:
        lines.append(f"Open P/L: <b>{open_pnl:+.2f}$</b>")
    if position_ticket > 0:
        lines.append(f"Ticket: <code>{position_ticket}</code>")

    return "\n".join(lines)


class Guardian:
    """Background watchtower thread."""

    def __init__(
        self,
        *,
        engine_provider: Callable[[], Any],
        notify: Callable[[str], None],
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._engine_provider = engine_provider
        self._notify = notify
        self._stop = stop_event or threading.Event()
        self._throttle = _AlertThrottle()
        self._last_heartbeat_ts = 0.0
        self._last_mt5_ok_ts = time.monotonic()
        self._last_session_snapshot: Dict[str, Any] = {}
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="guardian.watchtower",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "GUARDIAN_STARTED | heartbeat=%ds poll=%ds",
            HEARTBEAT_INTERVAL_SEC,
            POLL_INTERVAL_SEC,
        )

    def request_stop(self) -> None:
        self._stop.set()

    def on_trade_result(self, intent: Any, result: Any) -> None:
        """Send trade and slippage alerts."""
        try:
            day_pnl = 0.0
            engine = self._safe_engine()
            if engine is not None:
                try:
                    status = engine.status()
                    day_pnl = _safe_float(getattr(status, "today_pnl", 0.0))
                except Exception:
                    day_pnl = 0.0

            symbol = str(getattr(intent, "symbol", "") or "")
            position_ticket = int(getattr(result, "position_ticket", 0) or 0)
            open_pnl = _position_profit(symbol, position_ticket)

            text = _format_trade_alert(
                intent,
                result,
                day_pnl=day_pnl,
                open_pnl=open_pnl,
            )
            if text:
                self._safe_notify(text)

            slippage = _safe_float(getattr(result, "slippage", 0.0))
            if (
                abs(slippage) > float(SLIPPAGE_THRESHOLD_POINTS)
                and self._throttle.allow("slippage")
            ):
                self._safe_notify(
                    "\n".join(
                        [
                            "<b>SLIPPAGE ALERT</b>",
                            f"Asset: <b>{symbol or '-'}</b>",
                            f"Observed: <b>{slippage:+.2f}</b> points",
                            f"Limit: <b>{SLIPPAGE_THRESHOLD_POINTS:.1f}</b> points",
                        ]
                    )
                )
        except Exception as exc:
            log.warning("GUARDIAN_TRADE_ALERT_FAILED | err=%s", exc)

    def build_heartbeat(self) -> Dict[str, Any]:
        """Assemble the snapshot used by the heartbeat message."""
        snapshot: Dict[str, Any] = {
            "status": "warming",
            "uptime_human": "-",
            "controller_state": "-",
            "mt5_connected": False,
            "trading": False,
            "equity": 0.0,
            "today_pnl": 0.0,
            "dd_pct": 0.0,
            "open_xau": 0,
            "open_btc": 0,
            "gate_reason": "",
            "session": {},
        }

        engine = self._safe_engine()
        if engine is None:
            snapshot["status"] = "problem"
            return snapshot

        try:
            status = engine.status()
            snapshot["mt5_connected"] = bool(getattr(status, "connected", False))
            snapshot["trading"] = bool(getattr(status, "trading", False))
            snapshot["controller_state"] = str(
                getattr(status, "controller_state", "-") or "-"
            )
            snapshot["open_xau"] = int(getattr(status, "open_trades_xau", 0) or 0)
            snapshot["open_btc"] = int(getattr(status, "open_trades_btc", 0) or 0)
            snapshot["today_pnl"] = _safe_float(getattr(status, "today_pnl", 0.0))
            snapshot["dd_pct"] = _safe_float(getattr(status, "dd_pct", 0.0))
            snapshot["gate_reason"] = str(getattr(status, "gate_reason", "") or "")
        except Exception as exc:
            snapshot["gate_reason"] = f"status_error:{exc}"

        try:
            boot_ts = float(getattr(engine, "_boot_ts", 0.0) or 0.0)
            if boot_ts > 0.0:
                snapshot["uptime_human"] = _format_duration(time.time() - boot_ts)
        except Exception:
            pass

        try:
            from ExnessAPI.functions import get_account_info

            account = get_account_info() or {}
            snapshot["equity"] = _safe_float(account.get("equity", 0.0))
        except Exception:
            pass

        try:
            from core.session_manager import session_snapshot

            snapshot["session"] = session_snapshot()
        except Exception:
            pass

        if snapshot["mt5_connected"] and snapshot["trading"] and not snapshot["gate_reason"]:
            snapshot["status"] = "healthy"
        elif not snapshot["mt5_connected"]:
            snapshot["status"] = "problem"
        else:
            snapshot["status"] = "warming"

        return snapshot

    def _run(self) -> None:
        self._safe_notify(
            "\n".join(
                [
                    "<b>GUARDIAN ONLINE</b>",
                    f"Heartbeat interval: <b>{HEARTBEAT_INTERVAL_SEC // 3600}h</b>",
                    f"Alert cooldown: <b>{ALERT_MIN_INTERVAL_SEC // 60}m</b>",
                ]
            )
        )

        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                log.error("GUARDIAN_TICK_FAILED | err=%s", exc)
            self._stop.wait(timeout=POLL_INTERVAL_SEC)

    def _tick(self) -> None:
        now = time.monotonic()

        if (now - self._last_heartbeat_ts) >= float(HEARTBEAT_INTERVAL_SEC):
            self._last_heartbeat_ts = now
            self._safe_notify(_format_heartbeat(self.build_heartbeat()))

        self._check_mt5_health(now)
        self._check_session_transitions()

    def _check_mt5_health(self, now: float) -> None:
        engine = self._safe_engine()
        if engine is None:
            return

        connected = False
        try:
            status = engine.status()
            connected = bool(getattr(status, "connected", False))
        except Exception:
            connected = False

        if connected:
            self._last_mt5_ok_ts = now
            return

        stale_for = now - float(self._last_mt5_ok_ts or now)
        if stale_for < float(DISCONNECT_GRACE_SEC):
            return
        if not self._throttle.allow("mt5_disconnect"):
            return

        self._safe_notify(
            "\n".join(
                [
                    "<b>MT5 DISCONNECT ALERT</b>",
                    f"Broker link unhealthy for <b>{stale_for:.0f}s</b>.",
                    "Auto-recovery remains active.",
                ]
            )
        )

    def _check_session_transitions(self) -> None:
        try:
            from core.session_manager import session_snapshot

            snapshot = session_snapshot()
        except Exception:
            return

        previous = self._last_session_snapshot or {}
        for asset_key in ("xau", "btc"):
            current = snapshot.get(asset_key) or {}
            old = previous.get(asset_key) or {}
            if not current:
                continue

            current_open = bool(current.get("is_open"))
            old_open = bool(old.get("is_open", current_open))
            if current_open == old_open:
                continue

            if not self._throttle.allow(f"session_{asset_key}"):
                continue

            lines = [
                f"<b>SESSION {'REOPENED' if current_open else 'CLOSED'}</b>",
                f"Asset: <b>{asset_key.upper()}</b>",
            ]
            if current_open and bool(current.get("is_post_gap_cooldown")):
                lines.append("Post-gap cooldown is active. New trades remain delayed.")
            if not current_open:
                mins = int(float(current.get("seconds_until_open", 0.0) or 0.0) // 60)
                lines.append(f"Next open in about <b>{mins}m</b>.")
            self._safe_notify("\n".join(lines))

        self._last_session_snapshot = snapshot

    def _safe_engine(self) -> Any:
        try:
            return self._engine_provider()
        except Exception:
            return None

    def _safe_notify(self, text: str) -> None:
        try:
            self._notify(text)
        except Exception as exc:
            log.warning("GUARDIAN_NOTIFY_FAILED | err=%s", exc)


_DEFAULT_GUARDIAN: Optional[Guardian] = None
_DEFAULT_GUARDIAN_LOCK = threading.Lock()


def install_guardian(
    *,
    engine_provider: Callable[[], Any],
    notify: Callable[[str], None],
    stop_event: Optional[threading.Event] = None,
) -> Guardian:
    """Install the process-wide Guardian singleton."""
    global _DEFAULT_GUARDIAN
    with _DEFAULT_GUARDIAN_LOCK:
        if _DEFAULT_GUARDIAN is not None:
            return _DEFAULT_GUARDIAN
        guardian = Guardian(
            engine_provider=engine_provider,
            notify=notify,
            stop_event=stop_event,
        )
        guardian.start()
        _DEFAULT_GUARDIAN = guardian
        return guardian


def get_guardian() -> Optional[Guardian]:
    """Return the current Guardian singleton."""
    return _DEFAULT_GUARDIAN


__all__ = [
    "Guardian",
    "get_guardian",
    "install_guardian",
]
