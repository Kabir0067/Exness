# _btc_indicators/backend.py — FIXED (production-grade)
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import talib  # type: ignore
except Exception:
    talib = None  # type: ignore

from .utils import _to_f64


class _Indicators:
    """
    TA-Lib backend with safe fallback to numpy (fast, deterministic).
    Fallback is used if talib is missing OR talib throws.
    """

    def __init__(self) -> None:
        self._has_talib = talib is not None

    # -------------------------- finite sanitizers --------------------------
    @staticmethod
    def _ffill_finite(x: np.ndarray, default: float = 0.0) -> np.ndarray:
        """
        Ensure output has ONLY finite values:
          - leading non-finite -> first finite
          - later non-finite -> forward-filled
          - if all non-finite -> filled with default
        """
        a = np.asarray(x, dtype=np.float64)
        n = int(a.size)
        if n == 0:
            return a

        m = np.isfinite(a)
        if bool(np.all(m)):
            return a

        out = a.copy()
        if not bool(np.any(m)):
            out.fill(float(default))
            return out

        first = int(np.argmax(m))
        out[:first] = out[first]

        src = out.copy()
        idx = np.where(np.isfinite(src), np.arange(n, dtype=np.int64), first).astype(np.int64)
        np.maximum.accumulate(idx, out=idx)
        return src[idx]

    @classmethod
    def _clean_in(cls, x: np.ndarray) -> np.ndarray:
        return cls._ffill_finite(_to_f64(x), default=0.0)

    @classmethod
    def _clean_out(cls, x: np.ndarray) -> np.ndarray:
        return cls._ffill_finite(_to_f64(x), default=0.0)

    # -------------------------- core math helpers --------------------------
    @staticmethod
    def _ema_alpha(span: int) -> float:
        s = int(max(1, span))
        return 2.0 / (float(s) + 1.0)

    @staticmethod
    def _ema_rec(x: np.ndarray, alpha: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = int(x.size)
        if n == 0:
            return x
        out = np.empty(n, dtype=np.float64)
        out[0] = float(x[0])
        a = float(alpha)
        ia = 1.0 - a
        for i in range(1, n):
            out[i] = a * float(x[i]) + ia * float(out[i - 1])
        return out

    @staticmethod
    def _wilder_rec(x: np.ndarray, period: int) -> np.ndarray:
        p = int(max(1, period))
        alpha = 1.0 / float(p)
        return _Indicators._ema_rec(x, alpha)

    @staticmethod
    def _sma_fast(x: np.ndarray, period: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = int(x.size)
        if n == 0:
            return x
        p = int(max(1, period))
        idx = np.arange(n, dtype=np.int64)
        start = idx - (p - 1)
        start = np.maximum(start, 0)
        cs = np.cumsum(x, dtype=np.float64)
        cs0 = np.empty(n + 1, dtype=np.float64)
        cs0[0] = 0.0
        cs0[1:] = cs
        s = cs0[idx + 1] - cs0[start]
        cnt = (idx - start + 1).astype(np.float64)
        return (s / cnt).astype(np.float64)

    @staticmethod
    def _std_fast(x: np.ndarray, period: int) -> np.ndarray:
        # FIXED: np.sqrt(var, dtype=...) is invalid. Use astype after sqrt.
        x = np.asarray(x, dtype=np.float64)
        n = int(x.size)
        if n == 0:
            return x
        p = int(max(1, period))
        idx = np.arange(n, dtype=np.int64)
        start = idx - (p - 1)
        start = np.maximum(start, 0)
        cs = np.cumsum(x, dtype=np.float64)
        css = np.cumsum(x * x, dtype=np.float64)
        cs0 = np.empty(n + 1, dtype=np.float64)
        css0 = np.empty(n + 1, dtype=np.float64)
        cs0[0] = css0[0] = 0.0
        cs0[1:], css0[1:] = cs, css
        s = cs0[idx + 1] - cs0[start]
        ss = css0[idx + 1] - css0[start]
        cnt = (idx - start + 1).astype(np.float64)
        mean = s / cnt
        var = np.maximum((ss / cnt) - (mean * mean), 0.0)
        return np.sqrt(var).astype(np.float64)

    # -------------------------- public indicator API --------------------------
    def EMA(self, x: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                y = talib.EMA(self._clean_in(x), int(period))  # type: ignore[attr-defined]
                return self._clean_out(y)
        except Exception:
            pass
        xx = self._clean_in(x)
        return self._ema_rec(xx, self._ema_alpha(int(period))).astype(np.float64)

    def SMA(self, x: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                y = talib.SMA(self._clean_in(x), int(period))  # type: ignore[attr-defined]
                return self._clean_out(y)
        except Exception:
            pass
        return self._sma_fast(self._clean_in(x), int(period))

    def STDDEV(self, x: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                y = talib.STDDEV(self._clean_in(x), int(period))  # type: ignore[attr-defined]
                return self._clean_out(y)
        except Exception:
            pass
        return self._std_fast(self._clean_in(x), int(period))

    def ATR(self, h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                y = talib.ATR(self._clean_in(h), self._clean_in(l), self._clean_in(c), int(period))  # type: ignore[attr-defined]
                return self._clean_out(y)
        except Exception:
            pass

        hh, ll, cc = self._clean_in(h), self._clean_in(l), self._clean_in(c)
        prev_c = np.roll(cc, 1)
        prev_c[0] = cc[0]
        tr = np.maximum(hh - ll, np.maximum(np.abs(hh - prev_c), np.abs(ll - prev_c)))
        atr = self._wilder_rec(tr, int(period))
        return self._clean_out(atr)

    def RSI(self, c: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                y = talib.RSI(self._clean_in(c), int(period))  # type: ignore[attr-defined]
                return self._clean_out(y)
        except Exception:
            pass

        cc = self._clean_in(c)
        delta = np.diff(cc, prepend=cc[0])
        gain = np.maximum(delta, 0.0)
        loss = np.maximum(-delta, 0.0)
        avg_gain = self._wilder_rec(gain, int(period))
        avg_loss = self._wilder_rec(loss, int(period))
        rs = avg_gain / np.maximum(1e-12, avg_loss)
        rsi = (100.0 - (100.0 / (1.0 + rs))).astype(np.float64)
        return self._clean_out(rsi)

    def ADX(self, h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                y = talib.ADX(self._clean_in(h), self._clean_in(l), self._clean_in(c), int(period))  # type: ignore[attr-defined]
                return self._clean_out(y)
        except Exception:
            pass

        hh, ll, cc = self._clean_in(h), self._clean_in(l), self._clean_in(c)
        prev_h, prev_l, prev_c = np.roll(hh, 1), np.roll(ll, 1), np.roll(cc, 1)
        prev_h[0], prev_l[0], prev_c[0] = hh[0], ll[0], cc[0]

        up_move = hh - prev_h
        down_move = prev_l - ll

        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

        tr = np.maximum(hh - ll, np.maximum(np.abs(hh - prev_c), np.abs(ll - prev_c)))
        atr = self._wilder_rec(tr, int(period))

        plus_di = 100.0 * (self._wilder_rec(plus_dm, int(period)) / np.maximum(1e-12, atr))
        minus_di = 100.0 * (self._wilder_rec(minus_dm, int(period)) / np.maximum(1e-12, atr))

        dx = 100.0 * (np.abs(plus_di - minus_di) / np.maximum(1e-12, (plus_di + minus_di)))
        adx = self._wilder_rec(dx, int(period)).astype(np.float64)
        return self._clean_out(adx)

    def MACD(
        self,
        c: np.ndarray,
        fastperiod: int,
        slowperiod: int,
        signalperiod: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            if self._has_talib:
                macd, sig, hist = talib.MACD(  # type: ignore[attr-defined]
                    self._clean_in(c),
                    fastperiod=int(fastperiod),
                    slowperiod=int(slowperiod),
                    signalperiod=int(signalperiod),
                )
                return self._clean_out(macd), self._clean_out(sig), self._clean_out(hist)
        except Exception:
            pass

        cc = self._clean_in(c)
        fast = self._ema_rec(cc, self._ema_alpha(int(fastperiod)))
        slow = self._ema_rec(cc, self._ema_alpha(int(slowperiod)))
        macd = fast - slow
        sig = self._ema_rec(macd, self._ema_alpha(int(signalperiod)))
        hist = macd - sig
        return self._clean_out(macd), self._clean_out(sig), self._clean_out(hist)

    def BBANDS(
        self,
        c: np.ndarray,
        timeperiod: int,
        nbdevup: float,
        nbdevdn: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            if self._has_talib:
                upper, mid, lower = talib.BBANDS(  # type: ignore[attr-defined]
                    self._clean_in(c),
                    timeperiod=int(timeperiod),
                    nbdevup=float(nbdevup),
                    nbdevdn=float(nbdevdn),
                    matype=0,
                )
                return self._clean_out(upper), self._clean_out(mid), self._clean_out(lower)
        except Exception:
            pass

        x = self._clean_in(c)
        mid = self._sma_fast(x, int(timeperiod))
        std = self._std_fast(x, int(timeperiod))
        upper = mid + float(nbdevup) * std
        lower = mid - float(nbdevdn) * std
        return self._clean_out(upper), self._clean_out(mid), self._clean_out(lower)
