# core/feature_engine.py — Unified FeatureEngine for all assets.
# Merges _btc_indicators/feature_engine.py (1106 lines)
#    and _xau_indicators/feature_engine.py (927 lines)
# into a single config-driven class.
from __future__ import annotations

import math
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib

from .config import BaseEngineConfig
from .utils import (
    kaufman_efficiency_ratio,
    relative_volatility_index,
)

log = logging.getLogger("core.feature_engine")


def safe_last(arr: np.ndarray, default: float = 0.0) -> float:
    """Extract last finite value from an array."""
    if arr is None or len(arr) == 0:
        return default
    v = float(arr[-1])
    return v if math.isfinite(v) else default


@dataclass
class AnomalyResult:
    score: float = 0.0
    reasons: List[str] = None
    blocked: bool = False
    level: str = "normal"

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class FeatureEngine:
    """
    Unified feature engine for indicator computation, pattern detection,
    and market microstructure analysis.

    Config-driven: all thresholds, lookback periods, and round-number
    levels are parameterized via BaseEngineConfig.

    Computes:
      - EMAs (8, 21, 50, 200), RSI, MACD, ADX, Bollinger Bands, ATR
      - CCI, MOM, ROC, STOCH (K/D), OBV
      - VWAP, Volume Z-Score
      - Smart-money: FVG, Liquidity Sweep, Order Blocks, Round Numbers
      - Trend/Regime analysis: linreg slope, volatility regime
      - Market anomalies: flash crash, volume spike, spread anomalies
      - Divergence detection: RSI divergence (bullish/bearish)
      - Forecast continuation: breakout + volume confirmation

    Usage:
        fe = FeatureEngine(cfg)
        indicators = fe.compute_indicators({"M1": df_m1, "M5": df_m5, "M15": df_m15})
    """

    def __init__(self, cfg: BaseEngineConfig) -> None:
        self.cfg = cfg
        ic = cfg.indicator

        # Indicator periods from config
        self._ema_short = ic.get("ema_short", 8)
        self._ema_medium = ic.get("ema_medium", 21)
        self._ema_slow = ic.get("ema_slow", 50)
        self._ema_anchor = ic.get("ema_anchor", 200)
        self._rsi_period = ic.get("rsi_period", 14)
        self._atr_period = ic.get("atr_period", 14)
        self._macd_fast = ic.get("macd_fast", 12)
        self._macd_slow = ic.get("macd_slow", 26)
        self._macd_signal = ic.get("macd_signal", 9)
        self._adx_period = ic.get("adx_period", 14)
        self._bb_period = ic.get("bb_period", 20)
        self._bb_std = cfg.bb_std
        self._cci_period = ic.get("cci_period", 20)
        self._mom_period = ic.get("mom_period", 10)
        self._roc_period = ic.get("roc_period", 10)
        self._stoch_k_period = ic.get("stoch_k_period", 14)
        self._stoch_slowk_period = ic.get("stoch_slowk_period", 3)
        self._stoch_slowd_period = ic.get("stoch_slowd_period", 3)
        self._stoch_matype = ic.get("stoch_matype", 0)
        self._ker_period = int(getattr(cfg, "ker_period", 10) or 10)
        self._rvi_period = int(getattr(cfg, "rvi_period", 14) or 14)

        # Round number levels — asset-specific
        symbol = cfg.symbol_params.base.upper()
        if "BTC" in symbol:
            self._round_levels = [500, 1000, 2500, 5000, 10000]
            self._round_atr_mult = 1.5
        else:
            self._round_levels = [5, 10, 25, 50, 100]
            self._round_atr_mult = 1.0

        # Anomaly thresholds
        self._anomaly_z_vol_thresh = 4.0
        self._anomaly_atr_spike_mult = 3.0
        self._anomaly_spread_z_thresh = 5.0

        self._validate_cfg()
        log.info("FeatureEngine initialized for %s", symbol)

    def _validate_cfg(self) -> None:
        for attr in ["_ema_short", "_ema_medium", "_rsi_period", "_atr_period", "_cci_period", "_mom_period", "_roc_period"]:
            v = getattr(self, attr)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"Invalid indicator config: {attr}={v}")

    # ═════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═════════════════════════════════════════════════════════════════

    def compute_indicators(
        self, df_dict: Dict[str, pd.DataFrame], shift: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute indicators for all timeframes.

        Args:
            df_dict: {timeframe_str: DataFrame} with OHLCV data
            shift: 1 = use prev bar (prevent look-ahead), 0 = use current

        Returns:
            {timeframe_str: {indicator_name: value}} for each TF
        """
        result: Dict[str, Dict[str, Any]] = {}

        for tf, df in df_dict.items():
            if df is None or len(df) < 10:
                result[tf] = {}
                continue
            try:
                result[tf] = self._compute_tf(tf=tf, df=df, shift=shift)
            except Exception as exc:
                log.error(
                    "compute_indicators(%s) error: %s\n%s",
                    tf, exc, traceback.format_exc(),
                )
                result[tf] = {}

        # MTF alignment
        if len(result) >= 2:
            self._check_mtf_alignment(result)

        return result

    # ═════════════════════════════════════════════════════════════════
    # TIMEFRAME COMPUTATION
    # ═════════════════════════════════════════════════════════════════

    def _compute_tf(
        self, *, tf: str, df: pd.DataFrame, shift: int,
    ) -> Dict[str, Any]:
        """Compute all indicators for a single timeframe."""
        df = self._ensure_volume(df)
        min_bars = self._min_bars(tf)
        if len(df) < min_bars:
            return {}

        # Normalize columns
        cols = {c.lower(): c for c in df.columns}
        h = self._as_1d_float(df[cols.get("high", "High")].to_numpy())
        l = self._as_1d_float(df[cols.get("low", "Low")].to_numpy())
        c = self._as_1d_float(df[cols.get("close", "Close")].to_numpy())
        o = self._as_1d_float(df[cols.get("open", "Open")].to_numpy())
        v = (
            self._as_1d_float(df[cols.get("tick_volume", cols.get("volume", "tick_volume"))].to_numpy())
            if any(k in cols for k in ("tick_volume", "volume"))
            else np.ones(len(c), dtype=np.float64)
        )

        # Defensive alignment in case duplicated columns produced odd shapes.
        n = int(min(len(h), len(l), len(c), len(o), len(v)))
        if n < min_bars:
            return {}
        if len(h) != n:
            h = h[-n:]
        if len(l) != n:
            l = l[-n:]
        if len(c) != n:
            c = c[-n:]
        if len(o) != n:
            o = o[-n:]
        if len(v) != n:
            v = v[-n:]

        # Fill NaN/inf
        c = self._ffill_finite(c)
        h = self._ffill_finite(h, default=c[0] if len(c) > 0 else 0.0)
        l = self._ffill_finite(l, default=c[0] if len(c) > 0 else 0.0)
        o = self._ffill_finite(o, default=c[0] if len(c) > 0 else 0.0)

        c0 = float(c[0]) if len(c) > 0 else 0.0

        # ── EMAs (TA-Lib) ──
        ema_s = self._nan_to_num(talib.EMA(c, timeperiod=self._ema_short), c0)
        ema_m = self._nan_to_num(talib.EMA(c, timeperiod=self._ema_medium), c0)
        ema_l = self._nan_to_num(talib.EMA(c, timeperiod=self._ema_slow), c0)
        if len(c) >= self._ema_anchor:
            ema_a = self._nan_to_num(talib.EMA(c, timeperiod=self._ema_anchor), c0)
        else:
            ema_a = ema_l

        # ── RSI (TA-Lib) ──
        rsi = self._nan_to_num(talib.RSI(c, timeperiod=self._rsi_period), 50.0)

        # ── MACD (TA-Lib) ──
        macd_line, macd_signal, macd_hist = talib.MACD(
            c,
            fastperiod=self._macd_fast,
            slowperiod=self._macd_slow,
            signalperiod=self._macd_signal,
        )
        macd_line = self._nan_to_num(macd_line, 0.0)
        macd_signal = self._nan_to_num(macd_signal, 0.0)
        macd_hist = self._nan_to_num(macd_hist, 0.0)

        # ── ATR (TA-Lib) ──
        atr = self._nan_to_num(talib.ATR(h, l, c, timeperiod=self._atr_period), 0.0)
        atr_last = safe_last(atr)
        close_last = float(c[-shift]) if shift < len(c) else float(c[-1])
        atr_pct = atr_last / close_last if close_last > 0 else 0.0

        # ── Fractal efficiency & volatility ──
        ker = kaufman_efficiency_ratio(c, period=self._ker_period)
        rvi = relative_volatility_index(c, period=self._rvi_period)

        # ── ADX (TA-Lib) ──
        adx = self._nan_to_num(talib.ADX(h, l, c, timeperiod=self._adx_period), 0.0)

        # ── Bollinger Bands (TA-Lib) ──
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            c,
            timeperiod=self._bb_period,
            nbdevup=self._bb_std,
            nbdevdn=self._bb_std,
            matype=0,
        )
        bb_upper = self._nan_to_num(bb_upper, c0)
        bb_middle = self._nan_to_num(bb_middle, c0)
        bb_lower = self._nan_to_num(bb_lower, c0)
        bb_width = (safe_last(bb_upper) - safe_last(bb_lower)) / safe_last(bb_middle) \
            if safe_last(bb_middle) > 0 else 0.0
        bb_pctb = (close_last - safe_last(bb_lower)) / (safe_last(bb_upper) - safe_last(bb_lower)) \
            if (safe_last(bb_upper) - safe_last(bb_lower)) > 0 else 0.5

        # ── Momentum & Volatility (TA-Lib) ──
        cci = self._nan_to_num(talib.CCI(h, l, c, timeperiod=self._cci_period), 0.0)
        mom = self._nan_to_num(talib.MOM(c, timeperiod=self._mom_period), 0.0)
        roc = self._nan_to_num(talib.ROC(c, timeperiod=self._roc_period), 0.0)
        stoch_k, stoch_d = talib.STOCH(
            h, l, c,
            fastk_period=self._stoch_k_period,
            slowk_period=self._stoch_slowk_period,
            slowk_matype=self._stoch_matype,
            slowd_period=self._stoch_slowd_period,
            slowd_matype=self._stoch_matype,
        )
        stoch_k = self._nan_to_num(stoch_k, 50.0)
        stoch_d = self._nan_to_num(stoch_d, 50.0)
        obv = self._nan_to_num(talib.OBV(c, v), 0.0)

        # ── VWAP ──
        vwap = self._vwap(df, window=20)

        # ── Volume Z-Score ──
        z_vol = self._z_score(v, period=50)
        z_vol_series = self._z_score_series(v, period=50)

        # ── Trend ──
        trend = self._determine_trend(c, ema_m, ema_l, ema_a, adx)

        # ── Linreg slope ──
        linreg_slope = self.linreg_trend_slope(c, period=20)

        # ── Volatility regime ──
        vol_regime, atr_ratio = self.volatility_regime(atr, period=50)

        # ── Divergence ──
        div = self._detect_divergence_swings(ind=rsi, price=c)

        # ── Forecast ──
        forecast = self.forecast_continuation(df, v)

        # ── Smart money patterns ──
        fvg = self._detect_fvg(df, atr)
        sweep = self._liquidity_sweep(df, atr_last=atr_last, z_volume=z_vol, adx=adx)
        ob = self._order_block(df, atr_last=atr_last, v=v, z_vol_series=z_vol_series)
        stop_hunt_side, stop_hunt_strength = self._stop_hunt_profile(
            df=df,
            atr_last=atr_last,
            z_volume=z_vol,
            sweep=sweep,
        )
        ob_touch_prox, ob_pretouch_bias = self._order_block_pretouch(df=df, atr_last=atr_last)
        near_rn = self._near_round_number(price=close_last, atr=atr_last)

        # ── Anomalies ──
        anomaly = self._detect_market_anomalies(
            dfp=df, atr_series=atr, z_vol_series=z_vol_series,
            adx=safe_last(adx),
        )

        # ── Confluence ──
        confl = self._calculate_confluence_increment(
            has_fvg=bool(fvg),
            has_sweep=bool(sweep),
            has_ob=bool(ob),
            has_near_rn=near_rn,
            has_div=bool(div),
            z_volume=z_vol,
            trend=trend,
            bb_width=bb_width,
            vwap_dist_atr=abs(close_last - vwap) / atr_last if atr_last > 0 else 0.0,
        )

        # Build output (shift-aware)
        idx = -shift if shift < len(c) else -1
        out = {
            # EMAs
            "ema_short": float(ema_s[idx]),
            "ema_medium": float(ema_m[idx]),
            "ema_slow": float(ema_l[idx]),
            "ema_anchor": float(ema_a[idx]) if len(ema_a) > abs(idx) else 0.0,
            # RSI
            "rsi": float(rsi[idx]),
            # MACD
            "macd": float(macd_line[idx]),
            "macd_signal": float(macd_signal[idx]),
            "macd_hist": float(macd_hist[idx]),
            # ATR
            "atr": atr_last,
            "atr_pct": atr_pct,
            # Fractal / volatility
            "ker": float(ker),
            "rvi": float(rvi),
            # ADX
            "adx": safe_last(adx),
            # Bollinger
            "bb_upper": safe_last(bb_upper),
            "bb_lower": safe_last(bb_lower),
            "bb_middle": safe_last(bb_middle),
            "bb_width": bb_width,
            "bb_pctb": bb_pctb,
            # Momentum / Volatility
            "cci": float(cci[idx]),
            "mom": float(mom[idx]),
            "roc": float(roc[idx]),
            "stoch_k": float(stoch_k[idx]),
            "stoch_d": float(stoch_d[idx]),
            "obv": float(obv[idx]),
            # VWAP
            "vwap": vwap,
            # Volume
            "z_volume": z_vol,
            # Trend / regime
            "trend": trend,
            "linreg_slope": linreg_slope,
            "regime": vol_regime,
            "atr_ratio": atr_ratio,
            "volatility_regime": vol_regime,
            # Patterns
            "fvg": fvg,
            "sweep": sweep,
            "order_block": ob,
            "stop_hunt_side": stop_hunt_side,
            "stop_hunt_strength": float(stop_hunt_strength),
            "ob_touch_proximity": float(ob_touch_prox),
            "ob_pretouch_bias": float(ob_pretouch_bias),
            "near_round": near_rn,
            "divergence": div,
            "forecast": forecast,
            # Anomaly
            "anomaly": anomaly,
            # Confluence
            "confluence": confl,
            # Close
            "close": close_last,
        }

        return out

    # ═════════════════════════════════════════════════════════════════
    # INDICATOR CALCULATIONS (TA-Lib)
    # ═════════════════════════════════════════════════════════════════

    @staticmethod
    def _ffill_finite(a: np.ndarray, default: float = 0.0) -> np.ndarray:
        """Forward-fill non-finite values."""
        out = a.copy()
        last_good = default
        for i in range(len(out)):
            if math.isfinite(out[i]):
                last_good = out[i]
            else:
                out[i] = last_good
        return out

    @staticmethod
    def _as_1d_float(values: Any) -> np.ndarray:
        """
        Normalize unknown input shape to 1D float array.
        Handles duplicated DataFrame columns where pandas may return 2D values.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            return np.array([float(arr)], dtype=np.float64)
        if arr.ndim == 1:
            return arr
        if arr.shape[0] == 0:
            return np.array([], dtype=np.float64)
        arr2 = arr.reshape(arr.shape[0], -1)
        return np.asarray(arr2[:, -1], dtype=np.float64).reshape(-1)

    @staticmethod
    def _nan_to_num(arr: np.ndarray, default: float = 0.0) -> np.ndarray:
        return np.nan_to_num(arr, nan=default, posinf=default, neginf=default)

    def _vwap(self, df: pd.DataFrame, window: int = 20) -> float:
        """Session VWAP approximation."""
        try:
            cols = {c.lower(): c for c in df.columns}
            c = df[cols.get("close", "Close")].values
            h = df[cols.get("high", "High")].values
            l = df[cols.get("low", "Low")].values

            vol_col = None
            for k in ("tick_volume", "volume"):
                if k in cols:
                    vol_col = cols[k]
                    break
            if vol_col is None:
                return float(c[-1]) if len(c) > 0 else 0.0
            v = df[vol_col].values.astype(np.float64)

            w = min(window, len(c))
            typical = (c[-w:] + h[-w:] + l[-w:]) / 3.0
            vol = v[-w:]
            vol_sum = np.sum(vol)
            if vol_sum <= 0:
                return float(np.mean(typical))
            return float(np.sum(typical * vol) / vol_sum)
        except Exception:
            return 0.0

    @staticmethod
    def _z_score(values: np.ndarray, period: int = 50) -> float:
        """Z-score of last value relative to recent window."""
        vals = FeatureEngine._as_1d_float(values)
        if len(vals) < 5:
            return 0.0
        p = max(2, int(period))
        w = vals[-p:]
        m = float(np.mean(w))
        s = float(np.std(w))
        if not np.isfinite(s) or s <= 1e-12:
            return 0.0
        last = float(vals[-1])
        if not np.isfinite(last) or not np.isfinite(m):
            return 0.0
        return float((last - m) / s)

    @staticmethod
    def _z_score_series(values: np.ndarray, period: int = 50) -> np.ndarray:
        """Rolling Z-score series."""
        vals = FeatureEngine._as_1d_float(values)
        out = np.zeros_like(vals, dtype=np.float64)
        p = max(2, int(period))
        for i in range(p, len(vals)):
            w = vals[i - p:i]
            m = float(np.mean(w))
            s = float(np.std(w))
            if np.isfinite(s) and s > 1e-12 and np.isfinite(m):
                out[i] = (vals[i] - m) / s
        return out

    # ═════════════════════════════════════════════════════════════════
    # PATTERN DETECTION
    # ═════════════════════════════════════════════════════════════════

    def _determine_trend(
        self, c: np.ndarray, ema21: np.ndarray, ema50: np.ndarray,
        ema200: np.ndarray, adx: np.ndarray,
    ) -> str:
        """Determine trend direction from EMA stack."""
        s = safe_last(ema21)
        m = safe_last(ema50)
        l = safe_last(ema200)
        if s > m > l > 0:
            return "bull"
        if s < m < l and l > 0:
            return "bear"
        return "flat"

    def _min_bars(self, timeframe: str) -> int:
        return max(self._ema_anchor + 10, 50)

    def _ensure_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}
        has_vol = any(k in cols for k in ("tick_volume", "volume"))
        if not has_vol:
            df = df.copy()
            df["tick_volume"] = 1.0
        return df

    def _detect_fvg(self, df: pd.DataFrame, atr: np.ndarray) -> str:
        """Detect Fair Value Gaps."""
        try:
            cols = {c.lower(): c for c in df.columns}
            h = df[cols.get("high", "High")].values
            l = df[cols.get("low", "Low")].values
            if len(h) < 3:
                return ""
            atr_last = safe_last(atr)
            if atr_last <= 0:
                return ""

            # Bullish FVG: gap between bar[-3] high and bar[-1] low
            gap = l[-1] - h[-3]
            if gap > atr_last * 0.5:
                return "bull_fvg"

            # Bearish FVG
            gap = l[-3] - h[-1]
            if gap > atr_last * 0.5:
                return "bear_fvg"
        except Exception:
            pass
        return ""

    def _liquidity_sweep(
        self, df: pd.DataFrame, *, atr_last: float, z_volume: float, adx: np.ndarray,
    ) -> str:
        """Detect liquidity sweeps (stop hunts)."""
        try:
            cols = {c.lower(): c for c in df.columns}
            h = df[cols.get("high", "High")].values
            l = df[cols.get("low", "Low")].values
            c = df[cols.get("close", "Close")].values
            if len(h) < 10 or atr_last <= 0:
                return ""

            # Lookback for recent high/low
            look = min(20, len(h) - 1)
            recent_high = float(np.max(h[-look:-1]))
            recent_low = float(np.min(l[-look:-1]))

            # Bullish sweep: price dips below recent low then closes above
            if l[-1] < recent_low and c[-1] > recent_low:
                if z_volume > 1.0:
                    return "bull_sweep"

            # Bearish sweep
            if h[-1] > recent_high and c[-1] < recent_high:
                if z_volume > 1.0:
                    return "bear_sweep"
        except Exception:
            pass
        return ""

    def _order_block(
        self, df: pd.DataFrame, *, atr_last: float, v: np.ndarray, z_vol_series: np.ndarray,
    ) -> str:
        """Detect order blocks (institutional candle patterns)."""
        try:
            cols = {c.lower(): c for c in df.columns}
            o = df[cols.get("open", "Open")].values
            c = df[cols.get("close", "Close")].values
            if len(o) < 5 or atr_last <= 0:
                return ""

            # Bullish OB: last down candle before a strong up move
            body_prev = c[-2] - o[-2]
            body_curr = c[-1] - o[-1]
            if body_prev < -atr_last * 0.5 and body_curr > atr_last:
                return "bull_ob"

            # Bearish OB
            if body_prev > atr_last * 0.5 and body_curr < -atr_last:
                return "bear_ob"
        except Exception:
            pass
        return ""

    def _stop_hunt_profile(
        self,
        *,
        df: pd.DataFrame,
        atr_last: float,
        z_volume: float,
        sweep: str,
    ) -> Tuple[str, float]:
        """
        Stop-hunt interpretation:
          - bull_sweep -> bear_trap (bullish response likely)
          - bear_sweep -> bull_trap (bearish response likely)
        Returns (side, signed_strength in [-1, +1]).
        """
        try:
            side = ""
            signed = 0.0
            if sweep == "bull_sweep":
                side = "bear_trap"
                signed = 1.0
            elif sweep == "bear_sweep":
                side = "bull_trap"
                signed = -1.0
            else:
                return "", 0.0

            cols = {c.lower(): c for c in df.columns}
            h = pd.to_numeric(df[cols.get("high", "High")], errors="coerce")
            l = pd.to_numeric(df[cols.get("low", "Low")], errors="coerce")
            o = pd.to_numeric(df[cols.get("open", "Open")], errors="coerce")
            c = pd.to_numeric(df[cols.get("close", "Close")], errors="coerce")
            if len(h) < 3:
                return side, float(signed * 0.30)
            rng = float((h.iloc[-1] - l.iloc[-1]) or 0.0)
            if rng <= 0.0:
                return side, float(signed * 0.30)
            lower_wick = float(max(0.0, min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]))
            upper_wick = float(max(0.0, h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])))
            wick_dom = (lower_wick - upper_wick) / max(rng, 1e-12)
            wick_mag = abs(wick_dom)
            atr_norm = min(1.0, rng / max(float(atr_last), 1e-12))
            vol_mag = min(1.0, max(0.0, float(z_volume)) / 3.0)
            strength = min(1.0, 0.40 * wick_mag + 0.35 * atr_norm + 0.25 * vol_mag)
            return side, float(np.clip(signed * strength, -1.0, 1.0))
        except Exception:
            return "", 0.0

    def _order_block_pretouch(
        self,
        *,
        df: pd.DataFrame,
        atr_last: float,
    ) -> Tuple[float, float]:
        """
        Order-block pre-touch anticipation:
          - proximity in [0, 1]: closeness to nearest recent OB anchor
          - bias in [-1, 1]: +bull / -bear pressure into that touch
        """
        try:
            cols = {c.lower(): c for c in df.columns}
            o = pd.to_numeric(df[cols.get("open", "Open")], errors="coerce")
            c = pd.to_numeric(df[cols.get("close", "Close")], errors="coerce")
            if len(o) < 30 or atr_last <= 0.0:
                return 0.0, 0.0
            body = c - o
            body_abs = body.abs()
            thr = float(max(1e-12, atr_last * 0.55))

            prev_o = o.shift(1)
            prev_body = body.shift(1)
            bull_ob = pd.Series(
                np.where((prev_body < 0.0) & (body > thr), prev_o, np.nan),
                index=df.index,
            ).ffill()
            bear_ob = pd.Series(
                np.where((prev_body > 0.0) & (body < -thr), prev_o, np.nan),
                index=df.index,
            ).ffill()

            cur = float(c.iloc[-1])
            b_anchor = float(bull_ob.iloc[-1]) if np.isfinite(bull_ob.iloc[-1]) else np.nan
            s_anchor = float(bear_ob.iloc[-1]) if np.isfinite(bear_ob.iloc[-1]) else np.nan
            dist_b = abs(cur - b_anchor) / atr_last if np.isfinite(b_anchor) else np.inf
            dist_s = abs(cur - s_anchor) / atr_last if np.isfinite(s_anchor) else np.inf
            nearest = float(min(dist_b, dist_s))
            prox = float(np.clip(1.0 - min(nearest, 3.0) / 3.0, 0.0, 1.0))

            if np.isfinite(dist_b) and np.isfinite(dist_s):
                bias = float(np.clip((dist_s - dist_b) / 3.0, -1.0, 1.0))
            elif np.isfinite(dist_b):
                bias = 0.5
            elif np.isfinite(dist_s):
                bias = -0.5
            else:
                bias = 0.0

            # Emphasize bias only when we are actually near a block.
            return prox, float(np.clip(bias * prox, -1.0, 1.0))
        except Exception:
            return 0.0, 0.0

    def _near_round_number(self, *, price: float, atr: float) -> bool:
        """Check if price is near a psychologically significant round number."""
        if price <= 0 or atr <= 0:
            return False
        threshold = atr * self._round_atr_mult
        for level in self._round_levels:
            remainder = price % level
            dist = min(remainder, level - remainder)
            if dist <= threshold:
                return True
        return False

    def _detect_divergence_swings(
        self, *, ind: np.ndarray, price: np.ndarray,
    ) -> str:
        """Detect bullish/bearish RSI divergence."""
        try:
            if len(ind) < 20 or len(price) < 20:
                return ""

            # Simple 2-point divergence on last 20 bars
            p = price[-20:]
            r = ind[-20:]

            # Find local lows
            p_low1 = float(np.min(p[:10]))
            p_low2 = float(np.min(p[10:]))
            r_low1 = float(np.min(r[:10]))
            r_low2 = float(np.min(r[10:]))

            # Bullish: price makes lower low, RSI makes higher low
            if p_low2 < p_low1 and r_low2 > r_low1:
                return "bull_div"

            # Bearish: price makes higher high, RSI makes lower high
            p_hi1 = float(np.max(p[:10]))
            p_hi2 = float(np.max(p[10:]))
            r_hi1 = float(np.max(r[:10]))
            r_hi2 = float(np.max(r[10:]))
            if p_hi2 > p_hi1 and r_hi2 < r_hi1:
                return "bear_div"
        except Exception:
            pass
        return ""

    def _detect_market_anomalies(
        self, *, dfp: pd.DataFrame, atr_series: np.ndarray,
        z_vol_series: np.ndarray, adx: float,
    ) -> AnomalyResult:
        """Detect market anomalies that should block or flag trading."""
        result = AnomalyResult()
        reasons = []

        try:
            cols = {c.lower(): c for c in dfp.columns}
            c = dfp[cols.get("close", "Close")].values

            # Flash crash detection: >3x ATR single-bar move
            if len(c) >= 2 and len(atr_series) >= 1:
                last_move = abs(c[-1] - c[-2])
                atr_last = safe_last(atr_series)
                if atr_last > 0 and last_move > atr_last * self._anomaly_atr_spike_mult:
                    result.score += 0.5
                    reasons.append(f"flash_move:{last_move/atr_last:.1f}xATR")

            # Volume spike
            z_vol_last = safe_last(z_vol_series)
            if z_vol_last > self._anomaly_z_vol_thresh:
                result.score += 0.3
                reasons.append(f"vol_spike:z={z_vol_last:.1f}")

            result.reasons = reasons
            if result.score > 0.7:
                result.blocked = True
                result.level = "critical"
            elif result.score > 0.3:
                result.level = "warning"
        except Exception:
            pass

        return result

    # ═════════════════════════════════════════════════════════════════
    # TREND ANALYSIS
    # ═════════════════════════════════════════════════════════════════

    def linreg_trend_slope(
        self, closes: np.ndarray, period: Optional[int] = None,
    ) -> float:
        """
        Linear regression slope normalized to [-1, +1].
        Used for H1/M15 trend direction detection.
        """
        if period is None:
            period = 20
        if len(closes) < period:
            return 0.0
        try:
            y = closes[-period:]
            x = np.arange(period, dtype=np.float64)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            denom = np.sum((x - x_mean) ** 2)
            if denom == 0:
                return 0.0
            slope = np.sum((x - x_mean) * (y - y_mean)) / denom

            # Normalize: slope relative to price makes it dimensionless
            norm = slope / y_mean if y_mean != 0 else 0.0
            # Scale to [-1, 1]
            return float(max(-1.0, min(1.0, norm * 1000)))
        except Exception:
            return 0.0

    def volatility_regime(
        self, atr_series: np.ndarray, period: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Detect volatility regime based on ATR ratio.
        Returns (regime, atr_ratio)
        """
        if period is None:
            period = 50
        if len(atr_series) < period + 1:
            return "normal", 1.0
        try:
            current = float(atr_series[-1])
            sma = float(np.mean(atr_series[-period:]))
            if sma <= 0:
                return "normal", 1.0
            ratio = current / sma
            if ratio > 1.5:
                return "explosive", ratio
            if ratio < 0.5:
                return "dead", ratio
            return "normal", ratio
        except Exception:
            return "normal", 1.0

    def forecast_continuation(
        self, df: pd.DataFrame, vol_series: np.ndarray,
    ) -> str:
        """Predict next candle continuation: breakout + volume confirmation."""
        try:
            cols = {c.lower(): c for c in df.columns}
            c = df[cols.get("close", "Close")].values
            h = df[cols.get("high", "High")].values
            l = df[cols.get("low", "Low")].values

            if len(c) < 3:
                return "none"

            avg_vol = float(np.mean(vol_series[-20:])) if len(vol_series) >= 20 else 1.0
            last_vol = float(vol_series[-1]) if len(vol_series) > 0 else 0.0
            vol_mult = last_vol / avg_vol if avg_vol > 0 else 0.0

            if c[-1] > h[-2] and vol_mult > 1.5:
                return "bull_continuation"
            if c[-1] < l[-2] and vol_mult > 1.5:
                return "bear_continuation"
        except Exception:
            pass
        return "none"

    # ═════════════════════════════════════════════════════════════════
    # CONFLUENCE & SIGNAL STRENGTH
    # ═════════════════════════════════════════════════════════════════

    def _calculate_confluence_increment(
        self,
        *,
        has_fvg: bool,
        has_sweep: bool,
        has_ob: bool,
        has_near_rn: bool,
        has_div: bool,
        z_volume: float,
        trend: str,
        bb_width: float,
        vwap_dist_atr: float,
    ) -> float:
        """Calculate confluence bonus score (0.0 to 1.0)."""
        score = 0.0
        if has_fvg:
            score += 0.15
        if has_sweep:
            score += 0.20
        if has_ob:
            score += 0.15
        if has_near_rn:
            score += 0.10
        if has_div:
            score += 0.15
        if z_volume > 1.5:
            score += 0.10
        if trend in ("bull", "bear"):
            score += 0.05
        if bb_width > 0.02:
            score += 0.05
        if vwap_dist_atr < 1.0:
            score += 0.05
        return min(1.0, score)

    def _determine_signal_strength(
        self, score: float, m1_data: Dict[str, Any],
    ) -> str:
        if score >= 70:
            return "strong"
        elif score >= 50:
            return "moderate"
        elif score >= 30:
            return "weak"
        return "none"

    def _check_mtf_alignment(self, out: Dict[str, Any]) -> None:
        """Check multi-timeframe trend alignment."""
        trends = [v.get("trend", "flat") for v in out.values() if isinstance(v, dict)]
        if all(t == "bull" for t in trends):
            for v in out.values():
                if isinstance(v, dict):
                    v["mtf_aligned"] = True
                    v["mtf_direction"] = "bull"
        elif all(t == "bear" for t in trends):
            for v in out.values():
                if isinstance(v, dict):
                    v["mtf_aligned"] = True
                    v["mtf_direction"] = "bear"
        else:
            for v in out.values():
                if isinstance(v, dict):
                    v["mtf_aligned"] = False
                    v["mtf_direction"] = "mixed"
