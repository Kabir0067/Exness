"""
ExnessAPI/daily_balance.py — Daily balance anchoring for Phase A/B/C (per-asset).

Provides thread-safe CSV-backed storage for daily start balance and peak
equity tracking, enabling phase-based risk management decisions.
"""

from __future__ import annotations

import csv
import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

# =============================================================================
# Global Constants
# =============================================================================
_LOCK = threading.Lock()
_FIELDS = ("date_utc", "start_balance", "peak_equity")

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================
@dataclass(frozen=True)
class DailyRow:
    """Immutable representation of a single daily balance record."""

    date_utc: str
    start_balance: float
    peak_equity: float


# =============================================================================
# Public API
# =============================================================================
def initialize_daily_balance(
    current_balance: float,
    asset: str = "GEN",
    balance_provider: Optional[Callable[[], float]] = None,
) -> Tuple[float, str]:
    """
    Initialize or retrieve today's daily balance anchor.

    Returns (start_balance, phase) where phase is always 'A' at init time.
    """
    asset_u = _normalize_asset(asset)
    today = _utc_date()
    path = _default_path(asset_u)

    with _LOCK:
        last = _read_last_row(path)

        if last and last.date_utc == today and last.start_balance > 0:
            return float(last.start_balance), "A"

        start_balance = 0.0

        if balance_provider is not None:
            try:
                start_balance = _safe_float(balance_provider())
            except Exception as exc:
                logger.warning("balance_provider failed: %s", exc)
                start_balance = 0.0

        if start_balance <= 0.0:
            start_balance = _safe_float(current_balance)

        row = DailyRow(today, float(start_balance), float(start_balance))
        _append_row(path, row)

        return float(start_balance), "A"


def get_peak_balance(
    current_equity: Optional[float] = None,
    previous_peak: Optional[float] = None,
    asset: str = "GEN",
) -> float:
    """Return the highest equity seen today, optionally computing from arguments."""
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
    """Update today's peak equity if current_equity exceeds the stored value."""
    asset_u = _normalize_asset(asset)
    today = _utc_date()
    path = _default_path(asset_u)
    current_equity_safe = _safe_float(current_equity)

    with _LOCK:
        if not path.exists():
            row = DailyRow(today, current_equity_safe, current_equity_safe)
            _append_row(path, row)
            return float(current_equity_safe)

        try:
            with path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
        except Exception as exc:
            logger.error("Read failed, resetting file: %s", exc)
            rows = []

        if not rows:
            _atomic_write(
                path,
                [
                    {
                        "date_utc": today,
                        "start_balance": f"{current_equity_safe:.2f}",
                        "peak_equity": f"{current_equity_safe:.2f}",
                    }
                ],
            )
            return float(current_equity_safe)

        last_row = rows[-1]
        last_date = str(last_row.get("date_utc") or "").strip()

        if last_date != today:
            row = DailyRow(today, current_equity_safe, current_equity_safe)
            _append_row(path, row)
            return float(current_equity_safe)

        start_balance = _safe_float(last_row.get("start_balance"))
        peak_equity = _safe_float(last_row.get("peak_equity"))

        new_peak = float(max(peak_equity, current_equity_safe, start_balance))

        # Avoid unnecessary disk write if value is unchanged
        if new_peak == peak_equity:
            return float(new_peak)

        updated_rows = rows[:-1]
        updated_rows.append(
            {
                "date_utc": today,
                "start_balance": f"{start_balance:.2f}",
                "peak_equity": f"{new_peak:.2f}",
            }
        )

        try:
            _atomic_write(path, updated_rows)
        except Exception as exc:
            logger.error("Atomic update failed: %s", exc)

        return float(new_peak)


# =============================================================================
# Private Helpers
# =============================================================================
def _utc_date() -> str:
    """Return today's date in ISO format (UTC)."""
    return datetime.now(timezone.utc).date().isoformat()


def _safe_float(value: object) -> float:
    """Convert to float with strict validation; fallback to 0.0 on failure."""
    try:
        result = float(value)  # type: ignore[arg-type]
        return result if math.isfinite(result) else 0.0
    except Exception as exc:
        logger.debug("safe_float conversion failed: %s", exc)
        return 0.0


def _normalize_asset(asset: str) -> str:
    """Normalize asset identifier to stable uppercase form."""
    normalized = (asset or "GEN").strip().upper()
    return normalized if normalized else "GEN"


def _default_path(asset: str) -> Path:
    """Resolve storage path via centralized logging config when available."""
    filename = f"daily_balance_{_normalize_asset(asset).lower()}.csv"

    try:
        from log_config import get_log_path  # type: ignore

        return Path(get_log_path(filename))
    except Exception as exc:
        logger.warning("get_log_path failed, fallback to local path: %s", exc)
        return Path(filename)


def _parse_row(row: dict) -> Optional[DailyRow]:
    """Parse CSV row into typed structure with strict validation."""
    try:
        date = str(row.get("date_utc") or "").strip()
        if not date:
            return None

        start_balance = _safe_float(row.get("start_balance"))
        peak_equity = _safe_float(row.get("peak_equity"))

        # Enforce non-negative invariants for financial data integrity
        if start_balance <= 0.0:
            return DailyRow(date, 0.0, 0.0)

        if peak_equity <= 0.0:
            peak_equity = start_balance

        return DailyRow(date, float(start_balance), float(peak_equity))

    except Exception as exc:
        logger.error("Failed to parse row: %s", exc)
        return None


def _read_last_row(path: Path) -> Optional[DailyRow]:
    """Retrieve the last row without loading the entire file into memory."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            last_row = None
            for row in reader:
                last_row = row

        if not last_row:
            return None

        return _parse_row(last_row)

    except Exception as exc:
        logger.error("Failed reading CSV: %s", exc)
        return None


def _atomic_write(path: Path, rows: list[dict]) -> None:
    """Perform atomic-like file replacement to prevent partial writes."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        with tmp_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(_FIELDS))
            writer.writeheader()

            for row in rows:
                writer.writerow(row)

        tmp_path.replace(path)

    except Exception as exc:
        logger.error("Atomic write failed: %s", exc)


def _append_row(path: Path, row: DailyRow) -> None:
    """Append row with header initialization if file is new."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()

        with path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(_FIELDS))

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "date_utc": str(row.date_utc).strip(),
                    "start_balance": f"{float(row.start_balance):.2f}",
                    "peak_equity": f"{float(row.peak_equity):.2f}",
                }
            )

    except Exception as exc:
        logger.error("Append row failed: %s", exc)


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "DailyRow",
    "get_peak_balance",
    "initialize_daily_balance",
    "update_peak_equity",
]
