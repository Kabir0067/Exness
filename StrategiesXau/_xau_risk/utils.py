from __future__ import annotations

import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np

from .logging_ import log_risk

def _utcnow() -> datetime:
    """Naive UTC datetime (tzinfo=None) for safe JSON/csv writes and date compares."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def percentile_rank(series: np.ndarray, value: float) -> float:
    """Percentile rank for finite values only. Returns 50.0 on any failure."""
    try:
        if series is None or len(series) == 0:
            return 50.0
        x = series[np.isfinite(series)]
        if len(x) == 0:
            return 50.0
        return float(np.sum(x <= value) / len(x) * 100.0)
    except Exception as exc:
        log_risk.error("percentile_rank error: %s | tb=%s", exc, traceback.format_exc())
        return 50.0


def _is_finite(*xs: float) -> bool:
    try:
        return all(bool(np.isfinite(x)) for x in xs)
    except Exception:
        return False


_SIDE_MAP = {
    "buy": "Buy",
    "long": "Buy",
    "b": "Buy",
    "1": "Buy",
    "order_type_buy": "Buy",
    "op_buy": "Buy",
    "sell": "Sell",
    "short": "Sell",
    "s": "Sell",
    "-1": "Sell",
    "order_type_sell": "Sell",
    "op_sell": "Sell",
}


def _side_norm(side: str) -> str:
    s = str(side or "").strip().lower()
    if not s:
        return "Buy"
    if s in _SIDE_MAP:
        return _SIDE_MAP[s]
    if "buy" in s:
        return "Buy"
    if "sell" in s:
        return "Sell"
    return str(side)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# -------------------- deterministic ladder (keep your logic) --------------------
_BASE_LOT = 0.02
_BASE_TP_USD = 2.0
_STEP_BALANCE = 100.0
_STEP_LOT = 0.01
_STEP_TP_USD = 1.0
_STEP_START_BALANCE = 200.0


def gold_lot_and_takeprofit(balance: float) -> Tuple[float, float]:
    """
    Balance-based ladder:
      1..199   -> lot=0.02, tp=2
      200..299 -> lot=0.03, tp=3
      300..399 -> lot=0.04, tp=4
      ... (each +100 balance adds +0.01 lot and +1 USD TP)
    """
    try:
        bal = float(balance or 0.0)
    except Exception:
        return 0.0, 0.0

    if bal < 1.0:
        return 0.0, 0.0

    if bal < _STEP_START_BALANCE:
        step = 0
    else:
        step = int((bal - _STEP_START_BALANCE) // _STEP_BALANCE) + 1

    lot = _BASE_LOT + (step * _STEP_LOT)
    tp = _BASE_TP_USD + (step * _STEP_TP_USD)
    return round(float(lot), 2), round(float(tp), 2)


def _atr_fallback(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ATR fallback (no TA-Lib)."""
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

    p = max(1, int(period))
    alpha = 1.0 / float(p)
    out = np.empty_like(tr, dtype=np.float64)
    out[0] = tr[0]
    for i in range(1, len(tr)):
        out[i] = alpha * tr[i] + (1.0 - alpha) * out[i - 1]
    return out
