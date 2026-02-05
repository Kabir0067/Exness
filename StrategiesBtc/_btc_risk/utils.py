# _btc_risk/utils.py
from __future__ import annotations

import math
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


_BASE_LOT = 0.02
_BASE_TP_USD = 2.0
_STEP_BALANCE = 100.0
_STEP_LOT = 0.01
_STEP_TP_USD = 1.0
_STEP_START_BALANCE = 200.0


def btc_lot_and_tp_usd(balance: float) -> Tuple[float, float]:
    """
    Balance-based ladder for BTC:
      1..199   -> lot=0.02, tp=2
      200..299 -> lot=0.03, tp=3
      ...
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


def clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def tp_multiplier_from_conf(confidence: float, min_mult: float = 1.5, max_mult: float = 3.0) -> float:
    c = clamp01(confidence)
    lo = float(min_mult)
    hi = float(max_mult)
    if hi < lo:
        hi = lo
    return float(lo + (hi - lo) * c)


def atr_take_profit(
    entry: float,
    atr: float,
    side: str,
    confidence: float,
    min_mult: float = 1.5,
    max_mult: float = 3.0,
) -> float:
    mult = tp_multiplier_from_conf(confidence, min_mult=min_mult, max_mult=max_mult)
    s = str(side or "").strip().lower()
    sign = 1.0 if s in ("buy", "long", "b", "1", "order_type_buy", "op_buy") else -1.0
    return float(entry + sign * float(atr) * mult)


def adaptive_risk_money(
    equity: float,
    base_risk_pct: float,
    confidence: float,
    phase: str,
    drawdown: float,
    *,
    phase_factors: dict | None = None,
    dd_cut: float = 0.10,
    dd_mult: float = 0.5,
) -> float:
    if not math.isfinite(float(equity)) or equity <= 0.0:
        return 0.0
    br = float(base_risk_pct)
    if not math.isfinite(br) or br <= 0.0:
        return 0.0
    c = clamp01(confidence)
    pf = phase_factors or {"A": 1.2, "B": 0.8, "C": 0.5}
    ph = str(phase or "A").upper()
    phase_mult = float(pf.get(ph, 1.0))
    dd = float(drawdown)
    if not math.isfinite(dd):
        dd = 0.0
    dd_factor = float(dd_mult) if dd > float(dd_cut) else 1.0
    return float(equity * br * c * phase_mult * dd_factor)
