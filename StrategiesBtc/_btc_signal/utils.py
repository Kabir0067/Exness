# _btc_signal/utils.py — from StrategiesBtc btc_signal_engine (hardened)
from __future__ import annotations

import numpy as np

try:
    import talib  # type: ignore
except Exception:
    talib = None  # type: ignore


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _atr_np(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    n = int(len(c))
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    if n == 1:
        out[0] = float(h[0] - l[0])
        return out

    period = max(1, int(period))
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]

    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    if n < period:
        out[-1] = float(np.mean(tr))
        return out

    out[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    if talib is not None:
        try:
            return talib.ATR(h, l, c, int(period))  # type: ignore[attr-defined]
        except Exception:
            return _atr_np(h, l, c, int(period))
    return _atr_np(h, l, c, int(period))


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
