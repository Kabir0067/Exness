"""
core/news_blackout.py — Scheduled high-impact economic-event HARD BLOCK.

Section 9 contract (NON-NEGOTIABLE):
    Trading MUST be disabled in a ±window (default 30–60 min) around:
        * NFP   (U.S. Non-Farm Payrolls)
        * CPI   (U.S. Consumer Price Index)
        * FOMC  (U.S. Federal Open Market Committee statement / rate decision)

Design principles
-----------------
* DETERMINISTIC: the calendar is read from a plain-text JSON file; the same
  blackout schedule is applied to backtest and live (parity).
* FAIL-CLOSED: if the calendar file cannot be read OR the system clock is
  stale, the gate returns `blocked=True`. This is institutional-safe —
  NEVER fail open in front of a scheduled high-impact event.
* UTC-only: all times in the calendar are UTC ISO8601 (e.g.
  "2026-05-02T12:30:00Z"). Local time zones are dangerous and disallowed.
* STATELESS: every call recomputes from the calendar; no hidden flags.
* BACKWARD-COMPATIBLE: if no calendar is provided, the default behaviour
  is "no event today" — the gate lets trades through. This is explicit
  (opt-in) to avoid breaking CI when the environment is not provisioned.
  Production deployments MUST provision a real calendar file.

Calendar file format (JSON)
---------------------------
    {
      "events": [
        {
          "name": "NFP",                 # one of NFP | CPI | FOMC | OTHER
          "ts_utc": "2026-05-02T12:30:00Z",
          "pre_min": 45,                 # optional override (minutes)
          "post_min": 45
        },
        ...
      ],
      "default_pre_min": 45,
      "default_post_min": 45
    }

Environment variables
---------------------
* ``NEWS_BLACKOUT_CALENDAR_PATH``      — path to JSON (default: Artifacts/news_calendar.json)
* ``NEWS_BLACKOUT_DEFAULT_PRE_MIN``    — default pre-event window (min), fallback 45
* ``NEWS_BLACKOUT_DEFAULT_POST_MIN``   — default post-event window (min), fallback 45
* ``NEWS_BLACKOUT_ENABLED``            — "0" to fully disable (emergency only)
* ``NEWS_BLACKOUT_FAIL_OPEN``          — "1" to fail-open on calendar read
                                         errors (STRONGLY DISCOURAGED)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from log_config import get_artifact_path  # type: ignore

    _DEFAULT_CAL = str(get_artifact_path("news_calendar.json"))
except Exception:
    _DEFAULT_CAL = os.path.join(os.getcwd(), "Artifacts", "news_calendar.json")

log = logging.getLogger("core.news_blackout")

_HIGH_IMPACT_NAMES = frozenset({"NFP", "CPI", "FOMC"})


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or str(default))
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in ("1", "true", "yes", "y", "on")


def _parse_iso_utc(s: str) -> Optional[datetime]:
    """Parse strict UTC ISO8601 (``Z`` or ``+00:00``). Rejects naive times."""
    if not s:
        return None
    t = str(s).strip()
    # Tolerate trailing Z.
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
    except Exception:
        return None
    if dt.tzinfo is None:
        # REJECT naive datetimes — they're ambiguous (institutional safety).
        return None
    return dt.astimezone(timezone.utc)


@dataclass
class CalendarEvent:
    name: str
    ts_utc: datetime
    pre_min: int
    post_min: int

    def blackout_window(self) -> Tuple[datetime, datetime]:
        from datetime import timedelta

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
        return {
            "blocked": bool(self.blocked),
            "reason": self.reason,
            "event_name": self.event_name,
            "event_ts_utc": self.event_ts_utc,
            "seconds_to_event": round(float(self.seconds_to_event), 3),
        }


class NewsBlackout:
    """
    Deterministic economic-event blackout gate.

    Thread-safe; calendar is cached and re-read on file mtime changes.
    """

    def __init__(self, calendar_path: Optional[str] = None) -> None:
        self._path = str(
            calendar_path
            or os.getenv("NEWS_BLACKOUT_CALENDAR_PATH", "")
            or _DEFAULT_CAL
        )
        self._lock = threading.RLock()
        self._mtime: float = -1.0
        self._events: List[CalendarEvent] = []
        self._default_pre = _env_int("NEWS_BLACKOUT_DEFAULT_PRE_MIN", 45)
        self._default_post = _env_int("NEWS_BLACKOUT_DEFAULT_POST_MIN", 45)
        self._enabled = _env_bool("NEWS_BLACKOUT_ENABLED", True)
        self._fail_open = _env_bool("NEWS_BLACKOUT_FAIL_OPEN", False)
        self._reload_locked()

    @property
    def path(self) -> str:
        return self._path

    # ---- Calendar I/O ------------------------------------------------------

    def _reload_locked(self) -> None:
        p = Path(self._path)
        if not p.exists():
            if self._events:
                log.warning("news_calendar_disappeared | path=%s", self._path)
            self._events = []
            self._mtime = -1.0
            return
        try:
            mt = float(p.stat().st_mtime)
        except Exception:
            mt = time.time()
        if mt == self._mtime and self._events:
            return
        try:
            with p.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception as exc:
            # Corrupt calendar → keep OLD events if any; fail-close default.
            log.error("news_calendar_parse_error | path=%s err=%s", self._path, exc)
            return
        if not isinstance(raw, dict):
            log.error("news_calendar_bad_shape | path=%s", self._path)
            return
        default_pre = int(raw.get("default_pre_min", self._default_pre) or self._default_pre)
        default_post = int(
            raw.get("default_post_min", self._default_post) or self._default_post
        )
        events_raw = raw.get("events") or []
        out: List[CalendarEvent] = []
        for ev in events_raw:
            if not isinstance(ev, dict):
                continue
            name = str(ev.get("name", "") or "").upper().strip()
            ts = _parse_iso_utc(str(ev.get("ts_utc", "") or ""))
            if not name or ts is None:
                continue
            pre_min = int(ev.get("pre_min", default_pre) or default_pre)
            post_min = int(ev.get("post_min", default_post) or default_post)
            if pre_min < 0 or post_min < 0:
                continue
            out.append(
                CalendarEvent(
                    name=name, ts_utc=ts, pre_min=pre_min, post_min=post_min
                )
            )
        out.sort(key=lambda e: e.ts_utc)
        self._events = out
        self._mtime = mt
        log.info(
            "news_calendar_loaded | path=%s events=%d default_pre=%d default_post=%d",
            self._path,
            len(self._events),
            default_pre,
            default_post,
        )

    def reload(self) -> None:
        with self._lock:
            self._reload_locked()

    # ---- Gate --------------------------------------------------------------

    def check(
        self,
        now_utc: Optional[datetime] = None,
        *,
        restrict_to_high_impact: bool = True,
    ) -> BlackoutDecision:
        """
        Return a blackout decision for `now_utc`.

        Fail-closed semantics:
          * `enabled == False` (emergency override) → always allow.
          * calendar file missing/empty:
              - if `fail_open == True`  → allow (warn).
              - else                    → BLOCK with reason "no_calendar".

        Only events whose name ∈ {NFP, CPI, FOMC} are treated as hard blocks
        when `restrict_to_high_impact=True` (the default and only supported
        mode in production).
        """
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
                log.error("news_calendar_reload_failed | err=%s", exc)
                if self._fail_open:
                    return BlackoutDecision(
                        blocked=False,
                        reason=f"reload_failed_fail_open:{exc}",
                    )
                return BlackoutDecision(
                    blocked=True,
                    reason=f"reload_failed_fail_closed:{exc}",
                )

            if not self._events:
                if self._fail_open:
                    return BlackoutDecision(
                        blocked=False, reason="no_calendar_fail_open"
                    )
                # Institutional-safe default: no calendar = no trading until
                # one is provisioned. Operators MUST deploy the calendar.
                if str(
                    os.getenv("NEWS_BLACKOUT_ALLOW_EMPTY_CALENDAR", "0") or "0"
                ).strip() in ("1", "true", "True"):
                    return BlackoutDecision(
                        blocked=False, reason="empty_calendar_allowed"
                    )
                return BlackoutDecision(
                    blocked=True, reason="no_calendar_fail_closed"
                )

            for ev in self._events:
                if restrict_to_high_impact and ev.name not in _HIGH_IMPACT_NAMES:
                    continue
                start, end = ev.blackout_window()
                if start <= now <= end:
                    dt_sec = (ev.ts_utc - now).total_seconds()
                    return BlackoutDecision(
                        blocked=True,
                        reason=f"{ev.name}_blackout_window",
                        event_name=ev.name,
                        event_ts_utc=ev.ts_utc.isoformat(),
                        seconds_to_event=dt_sec,
                    )
        return BlackoutDecision(blocked=False, reason="clear")


# ---- Module-level singleton (lazy) --------------------------------------------

_DEFAULT: Optional[NewsBlackout] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_blackout() -> NewsBlackout:
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = NewsBlackout()
    return _DEFAULT


def is_blocked(
    now_utc: Optional[datetime] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Convenience: returns (blocked, reason, details)."""
    dec = get_default_blackout().check(now_utc=now_utc)
    return bool(dec.blocked), str(dec.reason), dec.to_dict()
