from __future__ import annotations

from typing import Tuple
import numpy as np

try:
    import talib  # type: ignore
except Exception:  # TA-Lib may be missing in some environments
    talib = None  # type: ignore

from .utils import _to_f64


class _Indicators:
    """
    TA-Lib backend with fast NumPy fallback.
    Fallback is used if talib is missing OR talib throws.

    Goals:
      - scalping-speed (no pandas overhead)
      - stable for NaN/inf (failsafe clamps)
      - Wilder/EMA seeding is correct (near TA-Lib)
    """

    _EPS = 1e-12

    def __init__(self) -> None:
        self._has_talib = talib is not None

    @staticmethod
    def _clean(x: np.ndarray) -> np.ndarray:
        # keep NaNs for warmup math; but remove inf
        a = _to_f64(x)
        if a.size == 0:
            return a
        # avoid copy when already finite
        if np.isfinite(a).all():
            return a
        b = a.copy()
        b[~np.isfinite(b)] = np.nan
        return b

    @staticmethod
    def _first_finite(x: np.ndarray, fallback: float = 0.0) -> float:
        try:
            if x.size == 0:
                return float(fallback)
            idx = np.flatnonzero(np.isfinite(x))
            if idx.size == 0:
                return float(fallback)
            return float(x[int(idx[0])])
        except Exception:
            return float(fallback)

    @staticmethod
    def _ema_np(x: np.ndarray, period: int, *, wilder: bool = False) -> np.ndarray:
        """
        EMA with proper seeding:
          - out[:n-1] = NaN when len >= n
          - seed = mean(x[:n]) ignoring NaN
          - Wilder uses alpha = 1/n
        """
        x = _Indicators._clean(x)
        n = int(max(1, period))
        out = np.full_like(x, np.nan, dtype=np.float64)
        if x.size == 0:
            return out

        alpha = (1.0 / n) if wilder else (2.0 / (n + 1.0))

        if x.size < n:
            # progressive fallback (rare in your pipeline due to min_bars)
            prev = _Indicators._first_finite(x, fallback=0.0)
            out[0] = prev
            for i in range(1, x.size):
                xi = float(x[i]) if np.isfinite(x[i]) else prev
                prev = prev + alpha * (xi - prev)
                out[i] = prev
            return out

        start = n - 1

        # NaN-safe seed WITHOUT double nanmean calls + without warnings
        w = x[:n]
        m = np.nanmean(w) if np.isfinite(w).any() else np.nan
        seed = float(m) if np.isfinite(m) else _Indicators._first_finite(w, fallback=0.0)

        out[start] = seed
        prev = seed

        for i in range(start + 1, x.size):
            xi = float(x[i]) if np.isfinite(x[i]) else prev
            prev = prev + alpha * (xi - prev)
            out[i] = prev

        return out

    @staticmethod
    def _sma_np(x: np.ndarray, period: int) -> np.ndarray:
        x = _Indicators._clean(x)
        n = int(max(1, period))
        out = np.full_like(x, np.nan, dtype=np.float64)
        if x.size == 0:
            return out

        finite = np.isfinite(x)
        xc = np.where(finite, x, 0.0).astype(np.float64, copy=False)
        cc = finite.astype(np.float64, copy=False)

        if x.size < n:
            cs = np.cumsum(xc, dtype=np.float64)
            ccount = np.cumsum(cc, dtype=np.float64)
            denom = np.maximum(ccount, 1.0)
            m = cs / denom
            # if no finite yet -> NaN
            m[ccount < 1.0] = np.nan
            out[:] = m
            return out

        cs = np.cumsum(xc, dtype=np.float64)
        ccount = np.cumsum(cc, dtype=np.float64)

        cs0 = np.empty(cs.size + 1, dtype=np.float64)
        cc0 = np.empty(ccount.size + 1, dtype=np.float64)
        cs0[0] = 0.0
        cc0[0] = 0.0
        cs0[1:] = cs
        cc0[1:] = ccount

        window_sum = cs0[n:] - cs0[:-n]
        window_cnt = cc0[n:] - cc0[:-n]

        # strict: require full window to avoid biased/false signals
        ok = window_cnt >= float(n)
        m = np.full_like(window_sum, np.nan, dtype=np.float64)
        m[ok] = window_sum[ok] / float(n)
        out[n - 1 :] = m
        return out

    @staticmethod
    def _std_np(x: np.ndarray, period: int) -> np.ndarray:
        x = _Indicators._clean(x)
        n = int(max(1, period))
        out = np.full_like(x, np.nan, dtype=np.float64)
        if x.size == 0:
            return out

        finite = np.isfinite(x)
        xc = np.where(finite, x, 0.0).astype(np.float64, copy=False)
        cc = finite.astype(np.float64, copy=False)

        if x.size < n:
            cs = np.cumsum(xc, dtype=np.float64)
            cs2 = np.cumsum(xc * xc, dtype=np.float64)
            ccount = np.cumsum(cc, dtype=np.float64)

            denom = np.maximum(ccount, 1.0)
            mean = cs / denom
            var = (cs2 / denom) - (mean * mean)
            var = np.maximum(var, 0.0)

            s = np.sqrt(var)
            s[ccount < 2.0] = np.nan  # need at least 2 points
            out[:] = s
            return out

        cs = np.cumsum(xc, dtype=np.float64)
        cs2 = np.cumsum(xc * xc, dtype=np.float64)
        ccount = np.cumsum(cc, dtype=np.float64)

        cs0 = np.empty(cs.size + 1, dtype=np.float64)
        cs20 = np.empty(cs2.size + 1, dtype=np.float64)
        cc0 = np.empty(ccount.size + 1, dtype=np.float64)
        cs0[0] = 0.0
        cs20[0] = 0.0
        cc0[0] = 0.0
        cs0[1:] = cs
        cs20[1:] = cs2
        cc0[1:] = ccount

        s = cs0[n:] - cs0[:-n]
        s2 = cs20[n:] - cs20[:-n]
        cnt = cc0[n:] - cc0[:-n]

        mean = np.full_like(s, np.nan, dtype=np.float64)
        var = np.full_like(s, np.nan, dtype=np.float64)

        ok = cnt >= float(n)  # strict full window
        mean[ok] = s[ok] / float(n)
        var[ok] = (s2[ok] / float(n)) - (mean[ok] * mean[ok])
        var = np.maximum(var, 0.0)

        out[n - 1 :] = np.sqrt(var)
        return out

    def EMA(self, x: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.EMA(_to_f64(x), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass
        return self._ema_np(x, int(period), wilder=False)

    def SMA(self, x: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.SMA(_to_f64(x), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass
        return self._sma_np(x, int(period))

    def STDDEV(self, x: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.STDDEV(_to_f64(x), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass
        return self._std_np(x, int(period))

    def ATR(self, h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.ATR(_to_f64(h), _to_f64(l), _to_f64(c), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass

        h = self._clean(h)
        l = self._clean(l)
        c = self._clean(c)
        if c.size == 0:
            return np.asarray(c, dtype=np.float64)

        prev_c = np.empty_like(c)
        prev_c[0] = c[0]
        prev_c[1:] = c[:-1]

        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
        return self._ema_np(tr, int(period), wilder=True)

    def RSI(self, c: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.RSI(_to_f64(c), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass

        c = self._clean(c)
        if c.size == 0:
            return np.asarray(c, dtype=np.float64)

        delta = np.empty_like(c, dtype=np.float64)
        delta[0] = 0.0
        delta[1:] = c[1:] - c[:-1]

        gain = np.maximum(delta, 0.0)
        loss = np.maximum(-delta, 0.0)

        avg_gain = self._ema_np(gain, int(period), wilder=True)
        avg_loss = self._ema_np(loss, int(period), wilder=True)

        rsi = np.full_like(c, np.nan, dtype=np.float64)

        ag = avg_gain
        al = avg_loss
        fin = np.isfinite(ag) & np.isfinite(al)

        # Correct RSI behavior:
        # - if ag==0 and al==0 => 50 (flat)
        # - if al==0 and ag>0 => 100
        # - if ag==0 and al>0 => 0
        if fin.any():
            agv = ag[fin]
            alv = al[fin]

            flat = (agv <= self._EPS) & (alv <= self._EPS)
            up = (alv <= self._EPS) & (agv > self._EPS)
            dn = (agv <= self._EPS) & (alv > self._EPS)
            mid = ~(flat | up | dn)

            outv = np.empty_like(agv, dtype=np.float64)
            outv[flat] = 50.0
            outv[up] = 100.0
            outv[dn] = 0.0

            rs = np.empty_like(agv, dtype=np.float64)
            rs[mid] = agv[mid] / np.maximum(self._EPS, alv[mid])
            outv[mid] = 100.0 - (100.0 / (1.0 + rs[mid]))

            rsi[fin] = outv

        return rsi.astype(np.float64, copy=False)

    def ADX(self, h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.ADX(_to_f64(h), _to_f64(l), _to_f64(c), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass

        h = self._clean(h)
        l = self._clean(l)
        c = self._clean(c)
        n = c.size
        if n == 0:
            return np.asarray(c, dtype=np.float64)

        prev_h = np.empty_like(h)
        prev_l = np.empty_like(l)
        prev_c = np.empty_like(c)
        prev_h[0], prev_l[0], prev_c[0] = h[0], l[0], c[0]
        prev_h[1:], prev_l[1:], prev_c[1:] = h[:-1], l[:-1], c[:-1]

        up_move = h - prev_h
        down_move = prev_l - l

        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

        atr = self._ema_np(tr, int(period), wilder=True)
        plus_sm = self._ema_np(plus_dm, int(period), wilder=True)
        minus_sm = self._ema_np(minus_dm, int(period), wilder=True)

        atr_safe = np.where(np.isfinite(atr), np.maximum(atr, self._EPS), np.nan)
        plus_di = 100.0 * (plus_sm / atr_safe)
        minus_di = 100.0 * (minus_sm / atr_safe)

        denom = plus_di + minus_di
        dx = np.where(
            np.isfinite(denom) & (np.abs(denom) > self._EPS),
            100.0 * (np.abs(plus_di - minus_di) / np.maximum(self._EPS, denom)),
            0.0,
        )

        adx = self._ema_np(dx, int(period), wilder=True)
        return adx.astype(np.float64, copy=False)

    def MACD(
        self, c: np.ndarray, fastperiod: int, slowperiod: int, signalperiod: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            if self._has_talib:
                return talib.MACD(  # type: ignore[attr-defined]
                    _to_f64(c),
                    fastperiod=int(fastperiod),
                    slowperiod=int(slowperiod),
                    signalperiod=int(signalperiod),
                )
        except Exception:
            pass

        c = self._clean(c)
        fast = self._ema_np(c, int(fastperiod), wilder=False)
        slow = self._ema_np(c, int(slowperiod), wilder=False)
        macd = fast - slow
        signal = self._ema_np(macd, int(signalperiod), wilder=False)
        hist = macd - signal
        return (
            macd.astype(np.float64, copy=False),
            signal.astype(np.float64, copy=False),
            hist.astype(np.float64, copy=False),
        )

    def BBANDS(
        self, c: np.ndarray, timeperiod: int, nbdevup: float, nbdevdn: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            if self._has_talib:
                return talib.BBANDS(  # type: ignore[attr-defined]
                    _to_f64(c),
                    timeperiod=int(timeperiod),
                    nbdevup=float(nbdevup),
                    nbdevdn=float(nbdevdn),
                    matype=0,
                )
        except Exception:
            pass

        c = self._clean(c)
        mid = self._sma_np(c, int(timeperiod))
        std = self._std_np(c, int(timeperiod))
        upper = mid + float(nbdevup) * std
        lower = mid - float(nbdevdn) * std
        return (
            upper.astype(np.float64, copy=False),
            mid.astype(np.float64, copy=False),
            lower.astype(np.float64, copy=False),
        )
