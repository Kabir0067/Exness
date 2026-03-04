from __future__ import annotations

"""
Daily balance anchoring for Phase A/B/C (per-asset).

Goal:
- At the start of each new UTC day, capture the account balance using balance_provider()
  (fallback: provided current_balance), and persist it to a small CSV file.
- RiskManager reads this anchor as daily_start_balance and applies phase rules based on daily return.

Design:
- dependency-light (safe if MT5 / APIs unavailable)
- single-process thread-safe
- atomic-ish writes on updates to avoid partial/corrupt CSV
"""

import csv
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

__all__ = [
    "DailyRow",
    "initialize_daily_balance",
    "get_peak_balance",
    "update_peak_equity",
]

_LOCK = threading.Lock()


def _utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _safe_float(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _normalize_asset(asset: str) -> str:
    a = (asset or "GEN").strip().upper()
    return a if a else "GEN"


def _default_path(asset: str) -> Path:
    name = f"daily_balance_{_normalize_asset(asset).lower()}.csv"
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


_FIELDS = ("date_utc", "start_balance", "peak_equity")


def _parse_row(r: dict) -> Optional[DailyRow]:
    try:
        d = str(r.get("date_utc") or "").strip()
        if not d:
            return None
        sb = _safe_float(r.get("start_balance"))
        pk = _safe_float(r.get("peak_equity"))
        if sb <= 0.0:
            return DailyRow(d, 0.0, 0.0)
        if pk <= 0.0:
            pk = sb
        return DailyRow(d, float(sb), float(pk))
    except Exception:
        return None


def _read_last_row(path: Path) -> Optional[DailyRow]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        return _parse_row(rows[-1])
    except Exception:
        return None


def _atomic_write(path: Path, rows: list[dict]) -> None:
    """
    Best-effort atomic-ish rewrite:
    write tmp -> replace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tmp.replace(path)


def _append_row(path: Path, row: DailyRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_FIELDS))
        if not file_exists:
            w.writeheader()
        w.writerow(
            {
                "date_utc": str(row.date_utc).strip(),
                "start_balance": f"{float(row.start_balance):.2f}",
                "peak_equity": f"{float(row.peak_equity):.2f}",
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
    this module always returns "A" and the RiskManager should compute actual phase.
    """
    asset_u = _normalize_asset(asset)
    today = _utc_date()
    path = _default_path(asset_u)

    with _LOCK:
        last = _read_last_row(path)

        # Already initialized for today
        if last and last.date_utc == today and last.start_balance > 0:
            return float(last.start_balance), "A"

        # Build new start anchor
        start = 0.0
        if balance_provider is not None:
            try:
                start = _safe_float(balance_provider())
            except Exception:
                start = 0.0

        if start <= 0.0:
            start = _safe_float(current_balance)

        row = DailyRow(today, float(start), float(start))
        _append_row(path, row)
        return float(start), "A"


def get_peak_balance(
    current_equity: Optional[float] = None,
    previous_peak: Optional[float] = None,
    asset: str = "GEN",
) -> float:
    """
    Returns peak equity.

    - If current_equity or previous_peak provided: returns max(previous_peak, current_equity).
    - Else: returns today's stored peak_equity from CSV (or 0.0).
    """
    if current_equity is not None or previous_peak is not None:
        ce = _safe_float(current_equity)
        pp = _safe_float(previous_peak)
        return float(max(ce, pp))

    asset_u = _normalize_asset(asset)
    path = _default_path(asset_u)
    with _LOCK:
        last = _read_last_row(path)
        if not last or last.date_utc != _utc_date():
            return 0.0
        return float(last.peak_equity)


def update_peak_equity(asset: str, current_equity: float) -> float:
    """
    Updates today's peak_equity in the CSV (best-effort). Returns updated peak.

    Critical properties:
    - Never raises (best-effort), always returns a float.
    - Avoids duplicate-day bug by trimming whitespace on date_utc.
    - Uses atomic-ish rewrite for the update path to prevent partial CSV corruption.
    """
    asset_u = _normalize_asset(asset)
    today = _utc_date()
    path = _default_path(asset_u)
    ce = _safe_float(current_equity)

    with _LOCK:
        if not path.exists():
            row = DailyRow(today, ce, ce)
            _append_row(path, row)
            return float(ce)

        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []

        if not rows:
            # Rewrite clean file
            _atomic_write(
                path,
                [{"date_utc": today, "start_balance": f"{ce:.2f}", "peak_equity": f"{ce:.2f}"}],
            )
            return float(ce)

        last = rows[-1]
        last_date = str(last.get("date_utc") or "").strip()

        if last_date != today:
            row = DailyRow(today, ce, ce)
            _append_row(path, row)
            return float(ce)

        start = _safe_float(last.get("start_balance"))
        peak = _safe_float(last.get("peak_equity"))
        new_peak = float(max(peak, ce, start))

        # Rewrite only with normalized values for last row
        out_rows = rows[:-1]
        out_rows.append(
            {
                "date_utc": today,
                "start_balance": f"{start:.2f}",
                "peak_equity": f"{new_peak:.2f}",
            }
        )

        try:
            _atomic_write(path, out_rows)
        except Exception:
            # best effort: ignore filesystem errors
            pass

        return float(new_peak)