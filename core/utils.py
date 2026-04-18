"""
core/utils.py — Unified utility functions for the entire trading system.

Consolidates duplicated code from _btc_risk/utils.py and _xau_risk/utils.py.
Provides numeric helpers, ATR calculations, position sizing, trailing stops,
and timeframe conversions used system-wide.
"""

from __future__ import annotations

import logging
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Tuple

import numpy as np
import talib

log = logging.getLogger("core.utils")


# =============================================================================
# Global Constants
# =============================================================================
_SIDE_MAP = MappingProxyType(
    {
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
        "neutral": "Neutral",
        "hold": "Neutral",
        "flat": "Neutral",
        "0": "Neutral",
    }
)

_TF_SECONDS = MappingProxyType(
    {
        "M1": 60,
        "M2": 120,
        "M3": 180,
        "M5": 300,
        "M10": 600,
        "M15": 900,
        "M20": 1200,
        "M30": 1800,
        "H1": 3600,
        "H2": 7200,
        "H4": 14400,
        "H6": 21600,
        "H8": 28800,
        "D1": 86400,
        "W1": 604800,
    }
)

_BASE_LOT = 0.02
_BASE_TP_USD = 2.0
_STEP_BALANCE = 100.0
_STEP_LOT = 0.01
_STEP_TP_USD = 1.0
_STEP_START_BALANCE = 200.0


# =============================================================================
# Time Helpers
# =============================================================================
def _utcnow() -> datetime:
    """Naive UTC datetime (tzinfo=None) for safe JSON/csv writes and date compares."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# =============================================================================
# Numeric Helpers
# =============================================================================
def _is_finite(*xs: float) -> bool:
    """Check all values are finite floats."""
    try:
        return all(bool(np.isfinite(x)) for x in xs)
    except Exception:
        return False


def clamp01(x: float) -> float:
    """Clamp a value to [0.0, 1.0]."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


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
        log.error("percentile_rank error: %s | tb=%s", exc, traceback.format_exc())
        return 50.0


# =============================================================================
# Side Normalization
# =============================================================================
def _side_norm(side: str) -> str:
    """Normalize any side string to 'Buy', 'Sell', or 'Neutral'."""
    s = str(side or "").strip().lower()
    if not s:
        return "Neutral"
    if s in _SIDE_MAP:
        return _SIDE_MAP[s]
    if "buy" in s:
        return "Buy"
    if "sell" in s:
        return "Sell"
    return "Neutral"


def is_buy(side: str) -> bool:
    """Return True if the normalized side is 'Buy'."""
    return _side_norm(side) == "Buy"


# =============================================================================
# File I/O
# =============================================================================
def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text to a file via temp-rename pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# =============================================================================
# ATR Calculation (TA-Lib)
# =============================================================================
def _atr_fallback(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """ATR via TA-Lib (fast, vectorized)."""
    h = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    p = max(1, int(period))
    return talib.ATR(h, low_arr, c, timeperiod=p)


def _atr_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> float:
    """Return the latest ATR value as a scalar float."""
    arr = _atr_fallback(high, low, close, period)
    if len(arr) == 0:
        return 0.0
    v = float(arr[-1])
    return v if math.isfinite(v) else 0.0


# =============================================================================
# Fractal Efficiency & Volatility (KER / RVI)
# =============================================================================
def kaufman_efficiency_ratio(close: np.ndarray, period: int = 10) -> float:
    """
    Kaufman Efficiency Ratio (KER) in [0, 1].

    Measures trend efficiency: 1.0 = smooth trend, 0.0 = noisy chop.
    """
    try:
        c = np.asarray(close, dtype=np.float64)
        p = int(max(2, period))
        if len(c) < p + 1:
            return 0.0
        change = abs(float(c[-1] - c[-p - 1]))
        volatility = float(np.sum(np.abs(np.diff(c[-(p + 1) :]))))
        if volatility <= 0.0:
            return 0.0
        return clamp01(change / volatility)
    except Exception:
        return 0.0


def relative_volatility_index(close: np.ndarray, period: int = 14) -> float:
    """
    Relative Volatility Index (RVI) in [0, 1].

    Uses std-dev of up vs down moves; 0.5 is neutral.
    """
    try:
        c = np.asarray(close, dtype=np.float64)
        if len(c) < max(3, int(period)):
            return 0.5
        chg = np.diff(c)
        p = int(max(2, period))
        up = np.where(chg > 0.0, chg, 0.0)
        dn = np.where(chg < 0.0, -chg, 0.0)
        up_std = float(np.std(up[-p:], ddof=0)) if len(up) >= p else float(np.std(up))
        dn_std = float(np.std(dn[-p:], ddof=0)) if len(dn) >= p else float(np.std(dn))
        denom = up_std + dn_std
        if denom <= 0.0:
            return 0.5
        return clamp01(up_std / denom)
    except Exception:
        return 0.5


def fractal_volatility_stop(
    entry: float,
    atr: float,
    side: str,
    *,
    base_mult: float,
    rvi: float,
    ker: float,
    rvi_weight: float = 1.0,
    chaos_mult: float = 1.0,
) -> float:
    """
    Volatility-adjusted fractal stop distance.

    Dynamic_SL = ATR * (1 + RVI * rvi_weight) * base_mult * (1 + chaos_mult * (1 - KER))
    """
    if not _is_finite(entry, atr) or atr <= 0:
        return float(entry)
    rvi_n = clamp01(rvi)
    ker_n = clamp01(ker)
    base = max(float(base_mult), 0.01)
    rvi_w = max(float(rvi_weight), 0.0)
    chaos_w = max(float(chaos_mult), 0.0)

    dyn_atr = float(atr) * (1.0 + rvi_n * rvi_w)
    dyn_mult = base * (1.0 + chaos_w * (1.0 - ker_n))
    dist = dyn_atr * dyn_mult

    sign = -1.0 if is_buy(side) else 1.0
    return float(entry + sign * dist)


# =============================================================================
# Position Sizing — Dynamic (ATR-based + equity %)
# =============================================================================
def dynamic_position_size(
    equity: float,
    risk_pct: float,
    atr: float,
    point_value: float,
    lot_step: float = 0.01,
    lot_min: float = 0.01,
    lot_max: float = 100.0,
) -> float:
    """
    Institutional-grade position sizing.

    Lot = (equity × risk_pct) / (ATR_in_points × point_value)

    Ensures risk per trade is constant in dollar terms regardless of volatility.
    Higher ATR → smaller lot. Lower ATR → larger lot.

    Args:
        equity: Account equity in USD.
        risk_pct: Risk per trade as fraction (e.g. 0.015 = 1.5%).
        atr: Current ATR in price units.
        point_value: Dollar value per one point move for 1 lot.
        lot_step: Broker's lot step (typically 0.01).
        lot_min: Broker's minimum lot.
        lot_max: Broker's maximum lot.

    Returns:
        Normalized lot size floored to lot_step.
    """
    if not _is_finite(equity, risk_pct, atr, point_value):
        return lot_min
    if equity <= 0 or risk_pct <= 0 or atr <= 0 or point_value <= 0:
        return lot_min

    risk_usd = equity * risk_pct
    raw_lot = risk_usd / (atr * point_value)

    if lot_step > 0:
        raw_lot = math.floor(raw_lot / lot_step) * lot_step

    return max(lot_min, min(lot_max, round(raw_lot, 8)))


# =============================================================================
# Lot Ladder (legacy balance-based — kept for backward compatibility)
# =============================================================================
def lot_and_tp_usd(balance: float) -> Tuple[float, float]:
    """
    Balance-based lot/TP ladder (shared by BTC and XAU).

      1..199   -> lot=0.02, tp=2
      200..299 -> lot=0.03, tp=3
      300..399 -> lot=0.04, tp=4
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


# =============================================================================
# TP / Risk-Reward Helpers
# =============================================================================
def tp_multiplier_from_conf(
    confidence: float,
    min_mult: float = 1.5,
    max_mult: float = 3.0,
    regime: str = "normal",
) -> float:
    """
    Scale TP multiplier linearly with confidence, adapted by volatility regime.

    - explosive: extend targets (+50%)
    - compressed: contract targets (-30%)
    """
    c = clamp01(confidence)

    # Regime adjustments
    if regime == "explosive":
        max_mult *= 1.5
    elif regime == "compressed":
        max_mult *= 0.7
        min_mult = max(1.0, min_mult * 0.8)

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
    """ATR-based take-profit level, scaled by confidence."""
    mult = tp_multiplier_from_conf(confidence, min_mult=min_mult, max_mult=max_mult)
    sign = 1.0 if is_buy(side) else -1.0
    return float(entry + sign * float(atr) * mult)


def atr_stop_loss(
    entry: float,
    atr: float,
    side: str,
    multiplier: float = 2.5,
) -> float:
    """ATR-based stop-loss level. Default: 2.5x ATR from entry."""
    sign = -1.0 if is_buy(side) else 1.0
    return float(entry + sign * float(atr) * multiplier)


# =============================================================================
# Adaptive Risk Money (equity x confidence x phase x drawdown)
# =============================================================================
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
    """
    Dynamic risk money calculation for institutional-grade position sizing.

    risk_money = equity x base_risk_pct x confidence x phase_mult x dd_factor

    Phase factors:
      A (normal) -> 1.2x
      B (caution) -> 0.8x
      C (defensive) -> 0.5x

    Drawdown guard: if drawdown > dd_cut, apply dd_mult (halves risk).
    """
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


# =============================================================================
# Trailing Stop Helpers
# =============================================================================
def volatility_trailing_stop(
    entry: float,
    current_price: float,
    atr: float,
    side: str,
    trail_atr_mult: float = 1.5,
) -> float:
    """
    Smart trailing stop based on volatility (ATR).

    Trails behind price at trail_atr_mult x ATR distance.

    For BUY: trail_stop = current_price - (ATR x mult)
    For SELL: trail_stop = current_price + (ATR x mult)

    Returns the trailing stop price.
    """
    if not _is_finite(entry, current_price, atr) or atr <= 0:
        return entry
    dist = atr * trail_atr_mult
    if is_buy(side):
        return max(entry, current_price - dist)
    else:
        return min(entry, current_price + dist)


def breakeven_price(
    entry: float,
    tp: float,
    current_price: float,
    side: str,
    be_trigger_pct: float = 0.40,
) -> float | None:
    """
    Move stop to breakeven when price reaches be_trigger_pct of distance to TP.

    Returns entry (breakeven) if triggered, else None.
    """
    if not _is_finite(entry, tp, current_price):
        return None
    dist_to_tp = abs(tp - entry)
    if dist_to_tp <= 0:
        return None

    if is_buy(side):
        progress = (current_price - entry) / dist_to_tp
    else:
        progress = (entry - current_price) / dist_to_tp

    if progress >= be_trigger_pct:
        return entry  # Move SL to breakeven
    return None


# =============================================================================
# Timeframe Helpers
# =============================================================================
def tf_seconds(tf: str) -> int:
    """Convert timeframe string (M1, M5, H1, etc.) to seconds."""
    return _TF_SECONDS.get(str(tf or "").strip().upper(), 60)
