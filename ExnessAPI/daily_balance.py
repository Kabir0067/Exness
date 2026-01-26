from __future__ import annotations

"""
Daily balance anchoring for Phase A/B/C (per-asset).

Goal:
- At the start of each new UTC day, capture the account balance using ExnessAPI.functions.get_balance()
  (fallback: provided current_balance), and persist it to a small CSV file.
- RiskManager reads this anchor as daily_start_balance and applies phase rules based on daily return.

This module is intentionally dependency-light and safe to import even if MT5 is temporarily unavailable.
"""

import csv
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

_LOCK = threading.Lock()


def _utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _safe_float(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _default_path(asset: str) -> Path:
    name = f"daily_balance_{asset.lower()}.csv"
    try:
        from log_config import get_log_path  # type: ignore

        return Path(get_log_path(name))
    except Exception:
        return Path(name)


@dataclass(frozen=True)
class DailyRow:
    date_utc: str
    start_balance: float
    peak_equity: float


def _read_last_row(path: Path) -> Optional[DailyRow]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        r = rows[-1]
        d = str(r.get("date_utc") or "")
        sb = _safe_float(r.get("start_balance"))
        pk = _safe_float(r.get("peak_equity"))
        if not d:
            return None
        return DailyRow(d, sb, pk if pk > 0 else sb)
    except Exception:
        return None


def _append_row(path: Path, row: DailyRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date_utc", "start_balance", "peak_equity"])
        if not file_exists:
            w.writeheader()
        w.writerow(
            {
                "date_utc": row.date_utc,
                "start_balance": f"{row.start_balance:.2f}",
                "peak_equity": f"{row.peak_equity:.2f}",
            }
        )


def initialize_daily_balance(
    current_balance: float,
    asset: str = "GEN",
    balance_provider: Optional[Callable[[], float]] = None,
) -> Tuple[float, str]:
    """
    Returns (daily_start_balance, saved_mode).

    saved_mode is kept for backward compatibility with older RiskManager code;
    the module always returns "A" and the RiskManager should compute actual phase
    using its own thresholds.
    """
    asset = (asset or "GEN").strip().upper()
    today = _utc_date()
    path = _default_path(asset)

    with _LOCK:
        last = _read_last_row(path)

        # If already initialized for today, reuse
        if last and last.date_utc == today and last.start_balance > 0:
            return float(last.start_balance), "A"

        # Otherwise create new anchor row
        start = 0.0
        if balance_provider is not None:
            try:
                start = _safe_float(balance_provider())
            except Exception:
                start = 0.0

        if start <= 0.0:
            start = _safe_float(current_balance)

        # If still invalid, persist a zero row (RiskManager will fallback)
        row = DailyRow(today, start, start)
        _append_row(path, row)
        return float(start), "A"


def get_peak_balance(
    current_equity: Optional[float] = None,
    previous_peak: Optional[float] = None,
    asset: str = "GEN",
) -> float:
    """
    Returns a peak equity value.

    - If current_equity & previous_peak are provided: returns max(previous_peak, current_equity).
    - If not provided: returns today's stored peak_equity from CSV (or 0.0).
    """
    if current_equity is not None or previous_peak is not None:
        ce = _safe_float(current_equity)
        pp = _safe_float(previous_peak)
        return float(max(ce, pp))

    asset = (asset or "GEN").strip().upper()
    path = _default_path(asset)
    with _LOCK:
        last = _read_last_row(path)
        if not last or last.date_utc != _utc_date():
            return 0.0
        return float(last.peak_equity)


def update_peak_equity(asset: str, current_equity: float) -> float:
    """
    Updates today's peak_equity in the CSV (best-effort). Returns the updated peak.
    """
    asset = (asset or "GEN").strip().upper()
    today = _utc_date()
    path = _default_path(asset)
    ce = _safe_float(current_equity)

    with _LOCK:
        # Load all rows (small file), update last row if today
        if not path.exists():
            # If no file yet, initialize with current_equity as both start+peak
            row = DailyRow(today, ce, ce)
            _append_row(path, row)
            return ce

        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []

        if not rows:
            row = DailyRow(today, ce, ce)
            # rewrite header+row
            with path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["date_utc", "start_balance", "peak_equity"])
                w.writeheader()
                w.writerow({"date_utc": today, "start_balance": f"{ce:.2f}", "peak_equity": f"{ce:.2f}"})
            return ce

        last = rows[-1]
        last_date = str(last.get("date_utc") or "")
        if last_date != today:
            # append a new day row (start uses current equity as fallback)
            row = DailyRow(today, ce, ce)
            _append_row(path, row)
            return ce

        # update peak
        start = _safe_float(last.get("start_balance"))
        peak = _safe_float(last.get("peak_equity"))
        new_peak = max(peak, ce, start)

        # rewrite (atomic-ish)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["date_utc", "start_balance", "peak_equity"])
                w.writeheader()
                for r in rows[:-1]:
                    w.writerow(r)
                w.writerow({"date_utc": today, "start_balance": f"{start:.2f}", "peak_equity": f"{new_peak:.2f}"})
            tmp.replace(path)
        except Exception:
            # best effort; ignore
            pass
        return float(new_peak)
