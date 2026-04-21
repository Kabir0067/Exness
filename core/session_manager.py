"""
Market session transition and post-gap safety helpers.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from log_config import get_log_path

    _STATE_DEFAULT = str(get_log_path("session_state.json"))
except Exception:
    _STATE_DEFAULT = os.path.join(os.getcwd(), "Logs", "session_state.json")


XAU_FRIDAY_CLOSE_MIN_UTC = int(
    os.getenv("SESSION_XAU_FRIDAY_CLOSE_MIN", "1260") or 1260
)
XAU_SUNDAY_OPEN_MIN_UTC = int(
    os.getenv("SESSION_XAU_SUNDAY_OPEN_MIN", "1320") or 1320
)
POST_GAP_COOLDOWN_SEC = max(
    60,
    int(os.getenv("SESSION_POST_GAP_COOLDOWN_SEC", "1200") or 1200),
)
STATE_FILE = os.getenv("SESSION_STATE_PATH", "") or _STATE_DEFAULT


@dataclass(frozen=True)
class SessionState:
    """Immutable session snapshot."""

    asset: str
    is_open: bool
    is_post_gap_cooldown: bool
    seconds_since_open: float
    seconds_until_open: float
    reason: str = ""

    @property
    def can_trade(self) -> bool:
        return bool(self.is_open and not self.is_post_gap_cooldown)

    def as_dict(self) -> Dict[str, object]:
        return {
            "asset": self.asset,
            "is_open": self.is_open,
            "is_post_gap_cooldown": self.is_post_gap_cooldown,
            "seconds_since_open": round(float(self.seconds_since_open), 2),
            "seconds_until_open": round(float(self.seconds_until_open), 2),
            "reason": self.reason,
            "can_trade": self.can_trade,
        }


@dataclass
class _PersistedMarkers:
    """Crash-safe transition markers."""

    last_open_ts: Dict[str, float] = field(default_factory=dict)
    last_close_ts: Dict[str, float] = field(default_factory=dict)


class SessionManager:
    """Thread-safe, crash-safe session tracker."""

    def __init__(
        self,
        *,
        state_path: Optional[str] = None,
        cooldown_sec: Optional[int] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._state_path = str(state_path or STATE_FILE)
        self._cooldown_sec = int(cooldown_sec or POST_GAP_COOLDOWN_SEC)
        self._markers = _PersistedMarkers()
        self._load()

    def session_state(self, asset: str) -> SessionState:
        """Return the current market session state for the asset."""
        asset_name = _normalize_asset(asset)
        now_utc = _utcnow()
        now_ts = now_utc.timestamp()

        is_open, next_open_ts, reason = self._compute_open(asset_name, now_utc)

        with self._lock:
            last_open = float(self._markers.last_open_ts.get(asset_name, 0.0) or 0.0)
            last_close = float(
                self._markers.last_close_ts.get(asset_name, 0.0) or 0.0
            )

            if is_open:
                if last_open <= 0.0:
                    self._markers.last_open_ts[asset_name] = now_ts
                    last_open = now_ts
                    self._save_locked()
                elif last_close > last_open:
                    self._markers.last_open_ts[asset_name] = now_ts
                    last_open = now_ts
                    self._save_locked()
            else:
                if last_close <= 0.0 or last_open > last_close:
                    self._markers.last_close_ts[asset_name] = now_ts
                    last_close = now_ts
                    self._save_locked()

        seconds_since_open = max(0.0, now_ts - last_open) if last_open > 0.0 else 0.0
        seconds_until_open = (
            max(0.0, next_open_ts - now_ts) if next_open_ts > 0.0 else 0.0
        )
        in_cooldown = bool(
            is_open
            and last_open > 0.0
            and last_close > 0.0
            and last_close < last_open
            and seconds_since_open < float(self._cooldown_sec)
        )

        return SessionState(
            asset=asset_name,
            is_open=is_open,
            is_post_gap_cooldown=in_cooldown,
            seconds_since_open=seconds_since_open,
            seconds_until_open=seconds_until_open,
            reason=reason,
        )

    def can_trade(self, asset: str) -> Tuple[bool, str]:
        """Return a trade gate decision for the asset."""
        state = self.session_state(asset)
        if not state.is_open:
            minutes = int(state.seconds_until_open // 60)
            return False, f"market_closed:reopen_in_{minutes}m:{state.reason}"
        if state.is_post_gap_cooldown:
            left = int(max(0.0, float(self._cooldown_sec) - state.seconds_since_open))
            return False, f"post_gap_cooldown:{left}s_remaining"
        return True, "session_open"

    def mark_open_now(self, asset: str) -> None:
        """Force-stamp an open marker."""
        with self._lock:
            self._markers.last_open_ts[_normalize_asset(asset)] = _utcnow().timestamp()
            self._save_locked()

    def snapshot(self) -> Dict[str, object]:
        """Return a compact multi-asset snapshot."""
        return {
            "cooldown_sec": int(self._cooldown_sec),
            "state_path": self._state_path,
            "xau": self.session_state("XAU").as_dict(),
            "btc": self.session_state("BTC").as_dict(),
        }

    def _compute_open(
        self,
        asset: str,
        now_utc: datetime,
    ) -> Tuple[bool, float, str]:
        if asset == "BTC":
            return True, 0.0, "btc_24x7"

        weekday = int(now_utc.weekday())
        minute_of_day = (now_utc.hour * 60) + now_utc.minute

        if weekday == 5:
            return False, self._next_sunday_open_ts(now_utc), "saturday"
        if weekday == 4 and minute_of_day >= XAU_FRIDAY_CLOSE_MIN_UTC:
            return False, self._next_sunday_open_ts(now_utc), "friday_after_close"
        if weekday == 6 and minute_of_day < XAU_SUNDAY_OPEN_MIN_UTC:
            return False, self._next_sunday_open_ts(now_utc), "sunday_before_open"
        return True, 0.0, "xau_weekday"

    @staticmethod
    def _next_sunday_open_ts(now_utc: datetime) -> float:
        weekday = int(now_utc.weekday())
        days_ahead = (6 - weekday) % 7
        if weekday == 6:
            minute_of_day = (now_utc.hour * 60) + now_utc.minute
            if minute_of_day >= XAU_SUNDAY_OPEN_MIN_UTC:
                days_ahead = 7

        base = now_utc.replace(
            hour=XAU_SUNDAY_OPEN_MIN_UTC // 60,
            minute=XAU_SUNDAY_OPEN_MIN_UTC % 60,
            second=0,
            microsecond=0,
        )
        return float(base.timestamp() + (days_ahead * 86400.0))

    def _load(self) -> None:
        try:
            path = Path(self._state_path)
            if not path.exists():
                return

            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                return

            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return

            self._markers.last_open_ts = {
                str(key).upper(): float(value)
                for key, value in (payload.get("last_open_ts") or {}).items()
                if isinstance(value, (int, float))
            }
            self._markers.last_close_ts = {
                str(key).upper(): float(value)
                for key, value in (payload.get("last_close_ts") or {}).items()
                if isinstance(value, (int, float))
            }
        except Exception:
            self._markers = _PersistedMarkers()

    def _save_locked(self) -> None:
        try:
            path = Path(self._state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(path.suffix + ".tmp")
            payload = {
                "last_open_ts": dict(self._markers.last_open_ts),
                "last_close_ts": dict(self._markers.last_close_ts),
                "saved_at_utc": _utcnow().isoformat(),
            }
            temp_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temp_path, path)
        except Exception:
            pass


_DEFAULT: Optional[SessionManager] = None
_DEFAULT_LOCK = threading.Lock()


def get_session_manager() -> SessionManager:
    """Return the process-wide SessionManager singleton."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = SessionManager()
    return _DEFAULT


def can_trade(asset: str) -> Tuple[bool, str]:
    """Return whether the asset is tradable right now."""
    return get_session_manager().can_trade(asset)


def session_state(asset: str) -> SessionState:
    """Return the current session state."""
    return get_session_manager().session_state(asset)


def session_snapshot() -> Dict[str, object]:
    """Return a multi-asset session snapshot."""
    return get_session_manager().snapshot()


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _normalize_asset(asset: str) -> str:
    asset_name = str(asset or "").upper().strip()
    if asset_name in {"XAUUSD", "XAUUSDM", "XAUUSDM."}:
        return "XAU"
    if asset_name in {"BTCUSD", "BTCUSDM", "BTCUSDM."}:
        return "BTC"
    return asset_name


__all__ = [
    "POST_GAP_COOLDOWN_SEC",
    "SessionManager",
    "SessionState",
    "can_trade",
    "get_session_manager",
    "session_snapshot",
    "session_state",
]
