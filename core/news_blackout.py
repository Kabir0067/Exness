"""
Scheduled news blackout checks for high-impact economic events.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from log_config import get_artifact_path  # type: ignore

    _DEFAULT_CAL = str(get_artifact_path("news_calendar.json"))
except Exception:
    _DEFAULT_CAL = os.path.join(os.getcwd(), "Artifacts", "news_calendar.json")


# =============================================================================
# Global Constants
# =============================================================================
_HIGH_IMPACT_NAMES = frozenset({"NFP", "CPI", "FOMC"})


# =============================================================================
# Logging Setup
# =============================================================================
log_news_blackout = logging.getLogger("core.news_blackout")


# =============================================================================
# Private Helpers
# =============================================================================
def _env_int(name: str, default: int) -> int:
    """Return an integer environment value with fallback."""
    try:
        return int(os.getenv(name, str(default)) or str(default))
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    """Return a boolean environment value with fallback."""
    raw_value = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    if not raw_value:
        return bool(default)
    return raw_value in ("1", "true", "yes", "y", "on")


def _parse_iso_utc(value: str) -> Optional[datetime]:
    """Parse a strict UTC ISO-8601 timestamp."""
    if not value:
        return None

    normalized_value = str(value).strip()
    if normalized_value.endswith("Z"):
        normalized_value = normalized_value[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized_value)
    except Exception:
        return None

    if parsed.tzinfo is None:
        return None

    return parsed.astimezone(timezone.utc)


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class CalendarEvent:
    name: str
    ts_utc: datetime
    pre_min: int
    post_min: int

    def blackout_window(self) -> Tuple[datetime, datetime]:
        """Return the active blackout window for the event."""
        return (
            self.ts_utc - timedelta(minutes=int(self.pre_min)),
            self.ts_utc + timedelta(minutes=int(self.post_min)),
        )


@dataclass
class BlackoutDecision:
    blocked: bool
    reason: str
    event_name: str = ""
    event_ts_utc: str = ""
    seconds_to_event: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable blackout decision."""
        return {
            "blocked": bool(self.blocked),
            "reason": self.reason,
            "event_name": self.event_name,
            "event_ts_utc": self.event_ts_utc,
            "seconds_to_event": round(float(self.seconds_to_event), 3),
        }


# =============================================================================
# Public API
# =============================================================================
class NewsBlackout:
    """Deterministic event blackout gate backed by a JSON calendar."""

    def __init__(self, calendar_path: Optional[str] = None) -> None:
        self._path = str(
            calendar_path or os.getenv("NEWS_BLACKOUT_CALENDAR_PATH", "") or _DEFAULT_CAL
        )
        self._lock = threading.RLock()
        self._mtime: float = -1.0
        self._events: List[CalendarEvent] = []
        self._default_pre = _env_int("NEWS_BLACKOUT_DEFAULT_PRE_MIN", 45)
        self._default_post = _env_int("NEWS_BLACKOUT_DEFAULT_POST_MIN", 45)
        self._enabled = _env_bool("NEWS_BLACKOUT_ENABLED", True)
        self._fail_open = _env_bool("NEWS_BLACKOUT_FAIL_OPEN", False)
        self._allow_missing_calendar = _env_bool(
            "NEWS_BLACKOUT_ALLOW_MISSING_CALENDAR",
            True,
        )
        self._calendar_exists = False
        self._missing_calendar_logged = False
        self._reload_locked()

    @property
    def path(self) -> str:
        return self._path

    def reload(self) -> None:
        """Force a calendar reload under the instance lock."""
        with self._lock:
            self._reload_locked()

    def check(
        self,
        now_utc: Optional[datetime] = None,
        *,
        restrict_to_high_impact: bool = True,
    ) -> BlackoutDecision:
        """Return the blackout decision for the given UTC timestamp."""
        if not self._enabled:
            return BlackoutDecision(blocked=False, reason="disabled_override")

        now = (
            now_utc.astimezone(timezone.utc)
            if now_utc is not None
            else datetime.now(timezone.utc)
        )

        with self._lock:
            try:
                self._reload_locked()
            except Exception as exc:
                log_news_blackout.error("news_calendar_reload_failed | err=%s", exc)
                if self._fail_open:
                    return BlackoutDecision(
                        blocked=False,
                        reason=f"reload_failed_fail_open:{exc}",
                    )
                return BlackoutDecision(
                        blocked=True,
                        reason=f"reload_failed_fail_closed:{exc}",
                    )

            if not self._calendar_exists:
                if self._allow_missing_calendar:
                    if not self._missing_calendar_logged:
                        log_news_blackout.warning(
                            "news_calendar_missing_allow_open | path=%s",
                            self._path,
                        )
                        self._missing_calendar_logged = True
                    self._bootstrap_missing_calendar_locked()
                    return BlackoutDecision(
                        blocked=False,
                        reason="calendar_missing_allow_open",
                    )

            if not self._events:
                if self._allow_missing_calendar:
                    if not self._missing_calendar_logged:
                        log_news_blackout.warning(
                            "news_calendar_empty_allow_open | path=%s",
                            self._path,
                        )
                        self._missing_calendar_logged = True
                    return BlackoutDecision(
                        blocked=False,
                        reason="empty_calendar_allow_open",
                    )

                if self._fail_open:
                    return BlackoutDecision(
                        blocked=False,
                        reason="no_calendar_fail_open",
                    )

                allow_empty = str(
                    os.getenv("NEWS_BLACKOUT_ALLOW_EMPTY_CALENDAR", "0") or "0"
                ).strip() in ("1", "true", "True")

                if allow_empty:
                    return BlackoutDecision(
                        blocked=False,
                        reason="empty_calendar_allowed",
                    )

                return BlackoutDecision(
                    blocked=True,
                    reason="no_calendar_fail_closed",
                )

            for event in self._events:
                if restrict_to_high_impact and event.name not in _HIGH_IMPACT_NAMES:
                    continue

                start_utc, end_utc = event.blackout_window()
                if start_utc <= now <= end_utc:
                    seconds_to_event = (event.ts_utc - now).total_seconds()
                    return BlackoutDecision(
                        blocked=True,
                        reason=f"{event.name}_blackout_window",
                        event_name=event.name,
                        event_ts_utc=event.ts_utc.isoformat(),
                        seconds_to_event=seconds_to_event,
                    )

        return BlackoutDecision(blocked=False, reason="clear")

    def _reload_locked(self) -> None:
        calendar_path = Path(self._path)
        if not calendar_path.exists():
            if self._events:
                log_news_blackout.warning(
                    "news_calendar_disappeared | path=%s",
                    self._path,
                )
            self._calendar_exists = False
            self._events = []
            self._mtime = -1.0
            return

        self._calendar_exists = True

        try:
            modified_time = float(calendar_path.stat().st_mtime)
        except Exception:
            modified_time = time.time()

        if modified_time == self._mtime and self._events:
            return

        try:
            with calendar_path.open("r", encoding="utf-8") as file_handle:
                raw_calendar = json.load(file_handle)
        except Exception as exc:
            log_news_blackout.error(
                "news_calendar_parse_error | path=%s err=%s",
                self._path,
                exc,
            )
            return

        if not isinstance(raw_calendar, dict):
            log_news_blackout.error(
                "news_calendar_bad_shape | path=%s",
                self._path,
            )
            return

        default_pre = int(
            raw_calendar.get("default_pre_min", self._default_pre)
            or self._default_pre
        )
        default_post = int(
            raw_calendar.get("default_post_min", self._default_post)
            or self._default_post
        )
        raw_events = raw_calendar.get("events") or []
        events: List[CalendarEvent] = []

        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue

            name = str(raw_event.get("name", "") or "").upper().strip()
            timestamp = _parse_iso_utc(str(raw_event.get("ts_utc", "") or ""))
            if not name or timestamp is None:
                continue

            pre_min = int(raw_event.get("pre_min", default_pre) or default_pre)
            post_min = int(raw_event.get("post_min", default_post) or default_post)
            if pre_min < 0 or post_min < 0:
                continue

            events.append(
                CalendarEvent(
                    name=name,
                    ts_utc=timestamp,
                    pre_min=pre_min,
                    post_min=post_min,
                )
            )

        events.sort(key=lambda event: event.ts_utc)
        self._events = events
        self._mtime = modified_time

        log_news_blackout.info(
            "news_calendar_loaded | path=%s events=%d default_pre=%d "
            "default_post=%d",
            self._path,
            len(self._events),
            default_pre,
            default_post,
        )

    def _bootstrap_missing_calendar_locked(self) -> None:
        path = Path(self._path)
        if path.exists():
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "default_pre_min": int(self._default_pre),
                "default_post_min": int(self._default_post),
                "events": [],
            }
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            log_news_blackout.warning(
                "news_calendar_bootstrap_failed | path=%s err=%s",
                self._path,
                exc,
            )


# =============================================================================
# Module State
# =============================================================================
_DEFAULT: Optional[NewsBlackout] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_blackout() -> NewsBlackout:
    """Return the module-level default blackout gate."""
    global _DEFAULT

    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = NewsBlackout()

    return _DEFAULT


def is_blocked(
    now_utc: Optional[datetime] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Return blackout status, reason, and details."""
    decision = get_default_blackout().check(now_utc=now_utc)
    return bool(decision.blocked), str(decision.reason), decision.to_dict()
