# StrategiesBtc/indicators.py  (BTCUSDm ONLY, 24/7, PRODUCTION-GRADE)
from __future__ import annotations

import logging
import math
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
except Exception:
    talib = None  # type: ignore

from config_btc import EngineConfig
from log_config import LOG_DIR as LOG_ROOT, get_log_path



# =============================================================================
# Logging (ERROR only)
# =============================================================================
LOG_DIR = LOG_ROOT

logger = logging.getLogger("indicators_btc")
logger.setLevel(logging.ERROR)
logger.propagate = False

if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler(
        str(get_log_path("feature_engine_btc.log")),
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    logger.addHandler(fh)


# =============================================================================
# Utils
# =============================================================================
def safe_last(arr: np.ndarray, default: float = 0.0) -> float:
    """Return last finite value from array; else default."""
    try:
        if arr is None:
            return float(default)
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return float(default)
        v = float(a[-1])
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _finite(x: float) -> bool:
    try:
        return bool(np.isfinite(x))
    except Exception:
        return False


def _require_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> bool:
    return all(c in df.columns for c in cols)


def _to_f64(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


# =============================================================================
# Indicator backend (TA-Lib + robust fast fallback)
# =============================================================================
class _Indicators:
    """
    TA-Lib backend with safe fallback to numpy (fast, deterministic).
    Fallback is used if talib is missing OR talib throws.
    """

    def __init__(self) -> None:
        self._has_talib = talib is not None

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
        cs0[0] = 0.0
        css0[0] = 0.0
        cs0[1:] = cs
        css0[1:] = css

        s = cs0[idx + 1] - cs0[start]
        ss = css0[idx + 1] - css0[start]
        cnt = (idx - start + 1).astype(np.float64)

        mean = s / cnt
        var = (ss / cnt) - (mean * mean)
        var = np.maximum(var, 0.0)
        return np.sqrt(var, dtype=np.float64)

    def EMA(self, x: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.EMA(_to_f64(x), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass
        return self._ema_rec(_to_f64(x), self._ema_alpha(int(period)))

    def SMA(self, x: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.SMA(_to_f64(x), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass
        return self._sma_fast(_to_f64(x), int(period))

    def STDDEV(self, x: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.STDDEV(_to_f64(x), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass
        return self._std_fast(_to_f64(x), int(period))

    def ATR(self, h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.ATR(_to_f64(h), _to_f64(l), _to_f64(c), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass

        h = _to_f64(h)
        l = _to_f64(l)
        c = _to_f64(c)

        prev_c = np.roll(c, 1)
        prev_c[0] = c[0]
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
        return self._wilder_rec(tr, int(period))

    def RSI(self, c: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.RSI(_to_f64(c), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass

        c = _to_f64(c)
        delta = np.diff(c, prepend=c[0])
        gain = np.maximum(delta, 0.0)
        loss = np.maximum(-delta, 0.0)

        avg_gain = self._wilder_rec(gain, int(period))
        avg_loss = self._wilder_rec(loss, int(period))

        rs = avg_gain / np.maximum(1e-12, avg_loss)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.astype(np.float64)

    def ADX(self, h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.ADX(_to_f64(h), _to_f64(l), _to_f64(c), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass

        h = _to_f64(h)
        l = _to_f64(l)
        c = _to_f64(c)

        prev_h = np.roll(h, 1)
        prev_l = np.roll(l, 1)
        prev_c = np.roll(c, 1)
        prev_h[0] = h[0]
        prev_l[0] = l[0]
        prev_c[0] = c[0]

        up_move = h - prev_h
        down_move = prev_l - l

        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
        atr = self._wilder_rec(tr, int(period))

        plus_di = 100.0 * (self._wilder_rec(plus_dm, int(period)) / np.maximum(1e-12, atr))
        minus_di = 100.0 * (self._wilder_rec(minus_dm, int(period)) / np.maximum(1e-12, atr))

        dx = 100.0 * (np.abs(plus_di - minus_di) / np.maximum(1e-12, (plus_di + minus_di)))
        adx = self._wilder_rec(dx, int(period))
        return adx.astype(np.float64)

    def MACD(
        self,
        c: np.ndarray,
        fastperiod: int,
        slowperiod: int,
        signalperiod: int,
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

        c = _to_f64(c)
        fast = self._ema_rec(c, self._ema_alpha(int(fastperiod)))
        slow = self._ema_rec(c, self._ema_alpha(int(slowperiod)))
        macd = fast - slow
        signal = self._ema_rec(macd, self._ema_alpha(int(signalperiod)))
        hist = macd - signal
        return macd.astype(np.float64), signal.astype(np.float64), hist.astype(np.float64)

    def BBANDS(
        self,
        c: np.ndarray,
        timeperiod: int,
        nbdevup: float,
        nbdevdn: float,
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

        x = _to_f64(c)
        mid = self._sma_fast(x, int(timeperiod))
        std = self._std_fast(x, int(timeperiod))
        upper = mid + float(nbdevup) * std
        lower = mid - float(nbdevdn) * std
        return upper.astype(np.float64), mid.astype(np.float64), lower.astype(np.float64)


# =============================================================================
# Outputs
# =============================================================================
@dataclass(frozen=True)
class AnomalyResult:
    score: float
    reasons: List[str]
    blocked: bool
    level: str  # "OK" | "WARN" | "BLOCK"


# =============================================================================
# Feature Engine (BTC scalping)
# =============================================================================
class Classic_FeatureEngine:
    """
    BTC scalping feature engine:
      - M1 entries + M5/M15 confirmations
      - Strong BTC-specific microstructure-friendly thresholds
      - Adds stable derived features used by signal/risk: atr_pct, bb_width, vwap_dist, slopes
      - Anomaly detector tuned for BTC (wick spikes, stop-runs, liquidity-thin jumps)
    """

    _REQUIRED_IND_KEYS = (
        "ema_short",
        "ema_mid",
        "ema_long",
        "ema_vlong",
        "atr_period",
        "rsi_period",
        "adx_period",
        "vol_lookback",
    )

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.timeframes = ["M1", "M5", "M15"]
        self._validate_cfg()

        self.ind = _Indicators()

        ind_cfg = self.cfg.indicator  # validated dict

        # pre-read periods (avoid dict lookup in hot path)
        self._ema_short_p = int(ind_cfg["ema_short"])
        self._ema_mid_p = int(ind_cfg["ema_mid"])
        self._ema_long_p = int(ind_cfg["ema_long"])
        self._ema_vlong_p = int(ind_cfg["ema_vlong"])
        self._atr_p = int(ind_cfg["atr_period"])
        self._rsi_p = int(ind_cfg["rsi_period"])
        self._adx_p = int(ind_cfg["adx_period"])
        self._vol_lb = int(ind_cfg["vol_lookback"])

        # ---------------- BTC tuned thresholds (config-driven; strong defaults) ----------------
        self.adx_trend_min = float(getattr(cfg, "adx_trend_lo", 22.0) or 22.0)
        self.adx_impulse_hi = float(getattr(cfg, "adx_impulse_hi", 42.0) or 42.0)
        self.z_vol_hot = float(getattr(cfg, "z_vol_hot", 2.2) or 2.2)
        self.bb_width_range_max = float(getattr(cfg, "bb_width_range_max", 0.0080) or 0.0080)
        self.rn_step = float(getattr(cfg, "rn_step", 100.0) or 100.0)
        self.fvg_min_atr_mult = float(getattr(cfg, "fvg_min_atr_mult", 0.35) or 0.35)
        self.sweep_min_atr_mult = float(getattr(cfg, "sweep_min_atr_mult", 0.15) or 0.15)
        self.vwap_window = int(getattr(cfg, "vwap_window", 30) or 30)

        # ---------------- anomaly config ----------------
        self._anom_wick_ratio = float(getattr(cfg, "anom_wick_ratio", 0.75) or 0.75)
        self._anom_range_atr = float(getattr(cfg, "anom_range_atr", 2.5) or 2.5)
        self._anom_gap_atr = float(getattr(cfg, "anom_gap_atr", 0.8) or 0.8)
        self._anom_stoprun_lookback = int(getattr(cfg, "anom_stoprun_lookback", 20) or 20)
        self._anom_stoprun_zvol = float(getattr(cfg, "anom_stoprun_zvol", 2.2) or 2.2)

        self._anom_warn_score = float(getattr(cfg, "anom_warn_score", 1.2) or 1.2)
        self._anom_block_score = float(getattr(cfg, "anom_block_score", 2.6) or 2.6)
        self._anom_persist_bars = int(getattr(cfg, "anom_persist_bars", 3) or 3)
        self._anom_persist_min_hits = int(getattr(cfg, "anom_persist_min_hits", 2) or 2)
        self._anom_confirm_zvol = float(getattr(cfg, "anom_confirm_zvol", 1.6) or 1.6)
        self._anom_tf_for_block = str(getattr(cfg, "anom_tf_for_block", "M1") or "M1")

        # indicator compute cache (single-entry)
        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._cache_out: Optional[Dict[str, Any]] = None
        self._cache_ts: float = 0.0
        self._cache_min_interval_ms: float = float(getattr(cfg, "indicator_cache_min_interval_ms", 0.0) or 0.0)

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------
    def _validate_cfg(self) -> None:
        ind = getattr(self.cfg, "indicator", None)
        if not isinstance(ind, dict):
            raise RuntimeError("EngineConfig.indicator dict is required for Classic_FeatureEngine")
        missing = [k for k in self._REQUIRED_IND_KEYS if k not in ind]
        if missing:
            raise RuntimeError(f"EngineConfig.indicator missing keys: {missing}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_indicators(self, df_dict: Dict[str, pd.DataFrame], shift: int = 1) -> Optional[Dict[str, Any]]:
        try:
            if not isinstance(df_dict, dict) or not df_dict:
                logger.error("compute_indicators: df_dict invalid/empty")
                return None
            if not isinstance(shift, int) or shift < 1:
                logger.error("compute_indicators: invalid shift=%s", shift)
                return None
            if "M1" not in df_dict or not isinstance(df_dict.get("M1"), pd.DataFrame) or df_dict["M1"].empty:
                logger.error("compute_indicators: M1 missing/empty (required)")
                return None

            frames = [tf for tf in self.timeframes if tf in df_dict]
            if not frames:
                logger.error("compute_indicators: no known timeframes in df_dict")
                return None

            # cache key
            computed_cache_key: Optional[Tuple[Any, ...]] = None
            try:
                key_parts: List[Tuple[str, int, int]] = []
                sh = int(shift)
                for tf in frames:
                    df = df_dict.get(tf)
                    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                        key_parts.append((tf, -1, 0))
                        continue
                    if len(df) < (sh + 2):
                        key_parts.append((tf, -1, int(len(df))))
                        continue

                    if "time" in df.columns:
                        t = df["time"].iloc[-1 - sh]
                        ts_ns = int(pd.Timestamp(t).value)
                    else:
                        ts_ns = int(len(df))

                    key_parts.append((tf, ts_ns, int(len(df))))

                computed_cache_key = (sh, tuple(key_parts))

                if self._cache_key == computed_cache_key and self._cache_out is not None:
                    if self._cache_min_interval_ms <= 0.0:
                        return self._cache_out
                    if (time.time() - float(self._cache_ts)) * 1000.0 < float(self._cache_min_interval_ms):
                        return self._cache_out
            except Exception:
                computed_cache_key = None

            out: Dict[str, Any] = {}
            confluence_score = 0.0
            block_tf_anom: Optional[AnomalyResult] = None

            for tf in frames:
                df = df_dict.get(tf)
                if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                if not _require_cols(df, ("open", "high", "low", "close")):
                    logger.error("compute_indicators: %s missing OHLC columns", tf)
                    continue

                required = self._min_bars(tf)
                if len(df) < (required + shift):
                    logger.error(
                        "Data insufficient for %s: len=%s < required+shift=%s",
                        tf,
                        len(df),
                        required + shift,
                    )
                    continue

                tf_out, inc, anom = self._compute_tf(tf=tf, df=df, shift=shift)
                if tf_out is None:
                    continue

                out[tf] = tf_out
                confluence_score += float(inc)

                if tf == self._anom_tf_for_block:
                    block_tf_anom = anom

            if "M1" not in out:
                logger.error("compute_indicators: M1 failed to produce output")
                return None

            mtf_aligned = self._check_mtf_alignment(out)
            if mtf_aligned:
                confluence_score += 2.0

            m1_data = out["M1"]
            signal_strength = self._determine_signal_strength(confluence_score, m1_data)

            if block_tf_anom is None:
                block_tf_anom = AnomalyResult(
                    score=float(m1_data.get("anomaly_score", 0.0)),
                    reasons=list(m1_data.get("anomaly_reasons", []))[:50],
                    blocked=bool(m1_data.get("trade_blocked", False)),
                    level=str(m1_data.get("anomaly_level", "OK")),
                )

            out["confluence_score"] = float(confluence_score)
            out["signal_strength"] = str(signal_strength)
            out["mtf_aligned"] = bool(mtf_aligned)

            out["anomaly_score"] = float(block_tf_anom.score)
            out["anomaly_reasons"] = [f"{self._anom_tf_for_block}:{r}" for r in block_tf_anom.reasons][:50]
            out["anomaly_level"] = str(block_tf_anom.level)
            out["trade_blocked"] = bool(block_tf_anom.blocked)

            # cache update
            try:
                if computed_cache_key is not None:
                    self._cache_key = computed_cache_key
                    self._cache_out = out
                    self._cache_ts = time.time()
            except Exception:
                pass

            return out

        except Exception as exc:
            logger.error("compute_indicators error: %s | tb=%s", exc, traceback.format_exc())
            return None

    # ------------------------------------------------------------------
    # Per-timeframe compute
    # ------------------------------------------------------------------
    def _compute_tf(self, *, tf: str, df: pd.DataFrame, shift: int) -> Tuple[Optional[Dict[str, Any]], float, AnomalyResult]:
        try:
            sh = int(shift)
            if sh <= 0:
                sh = 1

            dfp = df.iloc[:-sh]
            required = self._min_bars(tf)
            if len(dfp) < required:
                return None, 0.0, AnomalyResult(0.0, [], False, "OK")

            c = dfp["close"].to_numpy(dtype=np.float64, copy=False)
            h = dfp["high"].to_numpy(dtype=np.float64, copy=False)
            l = dfp["low"].to_numpy(dtype=np.float64, copy=False)
            o = dfp["open"].to_numpy(dtype=np.float64, copy=False)
            v = self._ensure_volume(dfp)  # already float64

            # --- core indicators (periods are pre-cached) ---
            ema9 = self.ind.EMA(c, self._ema_short_p)
            ema21 = self.ind.EMA(c, self._ema_mid_p)
            ema50 = self.ind.EMA(c, self._ema_long_p)
            ema200 = self.ind.EMA(c, self._ema_vlong_p)

            atr = self.ind.ATR(h, l, c, self._atr_p)
            rsi = self.ind.RSI(c, self._rsi_p)
            adx = self.ind.ADX(h, l, c, self._adx_p)

            macd, macd_signal, macd_hist = self.ind.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
            bb_u, bb_m, bb_l = self.ind.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)

            # --- derived features (BTC scalping-critical) ---
            close_last = float(c[-1])
            atr_last = float(safe_last(atr, 0.0))
            atr_pct = float(atr_last / close_last) if close_last > 0.0 and atr_last > 0.0 else 0.0

            bb_mid = float(safe_last(bb_m, close_last))
            bb_u_last = float(safe_last(bb_u, close_last))
            bb_l_last = float(safe_last(bb_l, close_last))
            # bb_width = (upper - lower) / mid  (stable, avoids extra ops)
            bb_width = float((bb_u_last - bb_l_last) / bb_mid) if bb_mid > 0.0 else 0.0

            ret1 = float((close_last / float(c[-2]) - 1.0)) if c.size >= 2 and float(c[-2]) > 0.0 else 0.0
            ret3 = float((close_last / float(c[-4]) - 1.0)) if c.size >= 4 and float(c[-4]) > 0.0 else 0.0

            if ema21.size >= 4 and atr_last > 0.0:
                ema21_slope = float((float(ema21[-1]) - float(ema21[-4])) / max(1e-9, atr_last))
            else:
                ema21_slope = 0.0

            rsi_slope = float(float(rsi[-1]) - float(rsi[-4])) if rsi.size >= 4 else 0.0
            macd_hist_slope = float(float(macd_hist[-1]) - float(macd_hist[-4])) if macd_hist.size >= 4 else 0.0

            # VWAP (tick_volume-based; best available in MT5 bars)
            vwap = self._vwap_from_bars(dfp, window=self.vwap_window)
            vwap_dist_atr = float((close_last - vwap) / max(1e-9, atr_last)) if _finite(vwap) and atr_last > 0.0 else 0.0

            # --- z-volume (FIXED: no invalid nan_to_num with array nan=...) ---
            vol_ma = self.ind.SMA(v, self._vol_lb)
            vol_std = self.ind.STDDEV(v, self._vol_lb)
            vol_std_safe = np.maximum(np.nan_to_num(vol_std, nan=0.0), 1e-9)

            # replace non-finite vol_ma elementwise with v (fast + correct)
            vol_ma_safe = np.where(np.isfinite(vol_ma), vol_ma, v)
            z_vol_series = (v - vol_ma_safe) / vol_std_safe
            zv = float(safe_last(z_vol_series, 0.0))

            # --- patterns / logic (BTC tuned) ---
            bull_fvg, bear_fvg = self._detect_fvg(dfp, atr)
            liq_sweep, bos_choch = self._liquidity_sweep(dfp, atr_last=atr_last, z_volume=zv, adx=adx)
            ob = self._order_block(dfp, atr_last=atr_last, v=v, z_vol_series=z_vol_series)

            near_rn = self._near_round_number(price=close_last, atr=atr_last)

            # Ислоҳ: Аз _detect_divergence ба _detect_divergence_swings гузаштам барои дақиқӣ беҳтар дар BTC
            rsi_div = self._detect_divergence_swings(ind=rsi, price=c)
            macd_div = self._detect_divergence_swings(ind=macd_hist, price=c)

            trend = self._determine_trend(c, ema21, ema50, ema200, adx)

            # --- anomaly detection (BTC) ---
            anom = self._detect_market_anomalies(
                dfp=dfp,
                atr_series=atr,
                z_vol_series=z_vol_series,
                adx=float(safe_last(adx)),
            )

            inc = self._calculate_confluence_increment(
                has_fvg=bool(bull_fvg or bear_fvg),
                has_sweep=bool(liq_sweep),
                has_ob=bool(ob),
                has_near_rn=bool(near_rn),
                has_div=bool((rsi_div != "none") or (macd_div != "none")),
                z_volume=float(zv),
                trend=str(trend),
                bb_width=float(bb_width),
                vwap_dist_atr=float(abs(vwap_dist_atr)),
            )

            tf_out = {
                # raw
                "close": close_last,
                "open": float(o[-1]),
                "high": float(h[-1]),
                "low": float(l[-1]),
                "body": float(abs(float(c[-1]) - float(o[-1]))),

                # core
                "atr": atr_last,
                "atr_pct": float(atr_pct),

                "ema9": float(safe_last(ema9)),
                "ema21": float(safe_last(ema21)),
                "ema50": float(safe_last(ema50)),
                "ema200": float(safe_last(ema200)),

                "rsi": float(safe_last(rsi)),
                "adx": float(safe_last(adx)),

                "macd": float(safe_last(macd)),
                "macd_signal": float(safe_last(macd_signal)),
                "macd_hist": float(safe_last(macd_hist)),

                "bb_upper": float(bb_u_last),
                "bb_mid": float(bb_mid),
                "bb_lower": float(bb_l_last),
                "bb_width": float(bb_width),

                # derived (scalping)
                "ret1": float(ret1),
                "ret3": float(ret3),
                "ema21_slope_atr": float(ema21_slope),
                "rsi_slope": float(rsi_slope),
                "macd_hist_slope": float(macd_hist_slope),

                "vwap": float(vwap) if _finite(vwap) else 0.0,
                "vwap_dist_atr": float(vwap_dist_atr),

                # volume
                "z_volume": float(zv),

                # patterns
                "fvg_bull": bool(bull_fvg),
                "fvg_bear": bool(bear_fvg),
                "liquidity_sweep": str(liq_sweep),
                "bos_choch": str(bos_choch),
                "order_block": str(ob),
                "near_round": bool(near_rn),
                "rsi_div": str(rsi_div),
                "macd_div": str(macd_div),

                # regime/trend
                "trend": str(trend),

                # anomalies
                "anomaly_score": float(anom.score),
                "anomaly_reasons": list(anom.reasons)[:50],
                "anomaly_level": str(anom.level),
                "trade_blocked": bool(anom.blocked),
            }

            return tf_out, float(inc), anom

        except Exception as exc:
            logger.error("_compute_tf error: %s | tb=%s", exc, traceback.format_exc())
            return None, 0.0, AnomalyResult(0.0, [], False, "OK")

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _min_bars(self, timeframe: str) -> int:
        if timeframe == "M1":
            return int(getattr(self.cfg, "min_bars_m1", 260) or 260)
        if timeframe in {"M5", "M15"}:
            return int(getattr(self.cfg, "min_bars_m5_m15", 220) or 220)
        return int(getattr(self.cfg, "min_bars_default", 200) or 200)

    def _determine_trend(self, c: np.ndarray, ema21: np.ndarray, ema50: np.ndarray, ema200: np.ndarray, adx: np.ndarray) -> str:
        adx_val = float(safe_last(adx))
        adx_min = float(self.adx_trend_min)

        if c[-1] > ema21[-1] > ema50[-1] and adx_val >= adx_min:
            return "strong_up"
        if c[-1] < ema21[-1] < ema50[-1] and adx_val >= adx_min:
            return "strong_down"

        if c[-1] >= ema200[-1]:
            return "weak_up"
        if c[-1] < ema200[-1]:
            return "weak_down"
        return "range"

    def _vwap_from_bars(self, df: pd.DataFrame, window: int) -> float:
        try:
            if df is None or df.empty:
                return float("nan")
            w = int(max(5, window))
            sub = df.iloc[-w:]
            if not _require_cols(sub, ("high", "low", "close")):
                return float("nan")
            high = sub["high"].to_numpy(dtype=np.float64, copy=False)
            low = sub["low"].to_numpy(dtype=np.float64, copy=False)
            close = sub["close"].to_numpy(dtype=np.float64, copy=False)
            tp = (high + low + close) / 3.0
            vol = self._ensure_volume(sub)
            vol_sum = float(np.sum(vol))
            if vol_sum <= 1e-9:
                return float(np.mean(tp))
            return float(np.sum(tp * vol) / vol_sum)
        except Exception:
            return float("nan")

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
        inc = 0.0

        if has_fvg:
            inc += 1.0
        if has_sweep:
            inc += 1.2
        if has_ob:
            inc += 1.0
        if has_div:
            inc += 0.9
        if has_near_rn:
            inc += 0.4

        if float(z_volume) >= float(self.z_vol_hot):
            inc += 0.8

        if bb_width > float(self.bb_width_range_max) * 1.8:
            inc -= 0.5

        if vwap_dist_atr >= 2.2:
            inc -= 0.6
        elif vwap_dist_atr <= 0.6:
            inc += 0.3

        if "strong_" in trend:
            inc += 0.4

        return float(max(-1.5, min(6.0, inc)))

    def _determine_signal_strength(self, score: float, m1_data: Dict[str, Any]) -> str:
        score = float(score)
        trend = str(m1_data.get("trend", ""))
        sweep = str(m1_data.get("liquidity_sweep", ""))

        if score >= 4.2:
            if m1_data.get("fvg_bull") or sweep == "bull" or ("up" in trend):
                return "strong_buy"
            if m1_data.get("fvg_bear") or sweep == "bear" or ("down" in trend):
                return "strong_sell"
            return "strong"

        if score >= 3.2:
            return "medium"

        return "neutral"

    def _check_mtf_alignment(self, out: Dict[str, Any]) -> bool:
        available = [k for k in ("M1", "M5", "M15") if k in out and isinstance(out[k], dict)]
        if len(available) < 2:
            return False

        trend_values = [str(out[tf].get("trend", "")) for tf in available]
        bullish = sum(1 for t in trend_values if "up" in t)
        bearish = sum(1 for t in trend_values if "down" in t)
        rng = sum(1 for t in trend_values if t == "range")
        total = len(trend_values)

        if bullish >= math.ceil(total * 0.67):
            return True
        if bearish >= math.ceil(total * 0.67):
            return True

        if (bullish + rng) >= math.ceil(total * 0.84):
            return True
        if (bearish + rng) >= math.ceil(total * 0.84):
            return True

        return False

    def _ensure_volume(self, df: pd.DataFrame) -> np.ndarray:
        try:
            if "tick_volume" in df.columns:
                return df["tick_volume"].fillna(0).to_numpy(dtype=np.float64, copy=False)
            if "real_volume" in df.columns:
                return df["real_volume"].fillna(0).to_numpy(dtype=np.float64, copy=False)

            hl_range = (df["high"] - df["low"]).fillna(0).to_numpy(dtype=np.float64, copy=False)
            close = df["close"].fillna(0).to_numpy(dtype=np.float64, copy=False)
            close_diff = np.abs(np.diff(close, prepend=close[0]))
            synth = (hl_range + close_diff) * 5000.0
            return synth.astype(np.float64, copy=False)
        except Exception as exc:
            logger.error("_ensure_volume error: %s | tb=%s", exc, traceback.format_exc())
            return np.zeros(len(df), dtype=np.float64)

    # ------------------------------------------------------------------
    # Anomaly detector (BTC tuned)
    # ------------------------------------------------------------------
    def _detect_market_anomalies(self, *, dfp: pd.DataFrame, atr_series: np.ndarray, z_vol_series: np.ndarray, adx: float) -> AnomalyResult:
        try:
            persist_bars = max(1, int(self._anom_persist_bars))
            min_hits = max(1, int(self._anom_persist_min_hits))
            lb = int(self._anom_stoprun_lookback)

            if len(dfp) < max(80, lb + persist_bars + 5):
                return AnomalyResult(0.0, [], False, "OK")

            o = dfp["open"].to_numpy(dtype=np.float64, copy=False)
            h = dfp["high"].to_numpy(dtype=np.float64, copy=False)
            l = dfp["low"].to_numpy(dtype=np.float64, copy=False)
            c = dfp["close"].to_numpy(dtype=np.float64, copy=False)

            atr_series = np.asarray(atr_series, dtype=np.float64)
            if atr_series.size != c.size:
                atr_series = np.full_like(c, float(safe_last(atr_series, default=0.0)))

            z_vol_series = np.asarray(z_vol_series, dtype=np.float64)
            if z_vol_series.size != c.size:
                z_vol_series = np.zeros_like(c)

            hit_reasons: Dict[str, int] = {}
            score_sum = 0.0
            z_confirm_hits = 0

            for k in range(1, persist_bars + 1):
                atr = float(atr_series[-k])
                if not _finite(atr) or atr <= 0.0:
                    continue

                o1 = float(o[-k])
                h1 = float(h[-k])
                l1 = float(l[-k])
                c1 = float(c[-k])
                prev_close = float(c[-k - 1])

                z_k = float(z_vol_series[-k])
                if z_k >= float(self._anom_confirm_zvol):
                    z_confirm_hits += 1

                rng = (h1 - l1)
                wick_up = h1 - max(o1, c1)
                wick_dn = min(o1, c1) - l1
                wick_dom = (max(wick_up, wick_dn) / max(1e-9, rng)) if rng > 0.0 else 0.0

                s = 0.0
                reasons_bar: List[str] = []

                if rng >= float(self._anom_range_atr) * atr:
                    reasons_bar.append("range_spike")
                    s += 1.1

                if wick_dom >= float(self._anom_wick_ratio) and rng >= 1.2 * atr:
                    reasons_bar.append("wick_spike")
                    s += 1.2

                if abs(o1 - prev_close) >= float(self._anom_gap_atr) * atr:
                    reasons_bar.append("gap_jump")
                    s += 1.0

                end = len(h) - k
                start = end - lb
                if start >= 0 and (end - start) >= max(6, lb // 2):
                    prev_high = float(np.max(h[start:end]))
                    prev_low = float(np.min(l[start:end]))

                    if (h1 > prev_high) and (c1 < prev_high):
                        if (z_k >= float(self._anom_stoprun_zvol)) or (wick_dom >= 0.6):
                            reasons_bar.append("stoprun_high_reject")
                            s += 1.6

                    if (l1 < prev_low) and (c1 > prev_low):
                        if (z_k >= float(self._anom_stoprun_zvol)) or (wick_dom >= 0.6):
                            reasons_bar.append("stoprun_low_reject")
                            s += 1.6

                if float(adx) >= float(self.adx_impulse_hi) and rng >= 1.7 * atr:
                    reasons_bar.append("impulse_adx_spike")
                    s += 0.7

                if rng >= 1.8 * atr and float(z_k) < 0.4:
                    reasons_bar.append("thin_liquidity_spike")
                    s += 0.7

                if reasons_bar:
                    score_sum += s
                    for r in reasons_bar:
                        hit_reasons[r] = hit_reasons.get(r, 0) + 1

            if not hit_reasons:
                return AnomalyResult(0.0, [], False, "OK")

            persistent = [r for r, cnt in hit_reasons.items() if cnt >= min_hits]
            reasons_out = sorted(persistent) if persistent else sorted(hit_reasons.keys())

            score = float(score_sum / float(max(1, persist_bars)))

            has_stoprun = ("stoprun_high_reject" in reasons_out) or ("stoprun_low_reject" in reasons_out)
            has_spike = ("wick_spike" in reasons_out) or ("gap_jump" in reasons_out) or ("range_spike" in reasons_out)

            blocked = False
            level = "OK"

            if score >= float(self._anom_block_score) and persistent:
                if has_stoprun:
                    blocked = True
                elif has_spike and z_confirm_hits >= 1:
                    blocked = True

            if blocked:
                level = "BLOCK"
            elif score >= float(self._anom_warn_score):
                level = "WARN"

            return AnomalyResult(score=float(score), reasons=reasons_out[:50], blocked=bool(blocked), level=str(level))

        except Exception as exc:
            logger.error("_detect_market_anomalies error: %s | tb=%s", exc, traceback.format_exc())
            return AnomalyResult(0.0, [], False, "OK")

    # ------------------------------------------------------------------
    # Patterns (BTC tuned)
    # ------------------------------------------------------------------
    def _detect_fvg(self, df: pd.DataFrame, atr: np.ndarray) -> Tuple[bool, bool]:
        try:
            if len(df) < 5:
                return False, False
            atr_last = float(safe_last(atr))
            if not _finite(atr_last) or atr_last <= 0.0:
                return False, False

            h = df["high"].to_numpy(dtype=np.float64, copy=False)
            l = df["low"].to_numpy(dtype=np.float64, copy=False)

            high_m3 = float(h[-3])
            low_m3 = float(l[-3])
            high_last = float(h[-1])
            low_last = float(l[-1])

            bull_gap = low_last - high_m3
            bear_gap = low_m3 - high_last

            thr = float(self.fvg_min_atr_mult) * atr_last
            return bool(bull_gap >= thr), bool(bear_gap >= thr)
        except Exception as exc:
            logger.error("_detect_fvg error: %s | tb=%s", exc, traceback.format_exc())
            return False, False

    def _liquidity_sweep(self, df: pd.DataFrame, *, atr_last: float, z_volume: float, adx: np.ndarray) -> Tuple[str, str]:
        try:
            if len(df) < 30:
                return "", ""
            if not _finite(atr_last) or atr_last <= 0.0:
                return "", ""

            h = df["high"].to_numpy(dtype=np.float64, copy=False)
            l = df["low"].to_numpy(dtype=np.float64, copy=False)
            c = df["close"].to_numpy(dtype=np.float64, copy=False)

            look = 24
            prev_high = float(np.max(h[-look:-1]))
            prev_low = float(np.min(l[-look:-1]))

            curr_high = float(h[-1])
            curr_low = float(l[-1])
            c1 = float(c[-1])
            c0 = float(c[-2])

            adx_val = float(safe_last(adx))
            adx_min = float(self.adx_trend_min)
            z_hot = float(self.z_vol_hot) * 0.85

            thr = float(self.sweep_min_atr_mult) * float(atr_last)

            sweep = ""
            bos_choch = ""

            if (curr_high >= (prev_high + thr)) and (c1 < prev_high) and (float(z_volume) >= z_hot) and (adx_val >= adx_min):
                sweep = "bear"
                bos_choch = "BOS_down" if c1 < c0 else "CHOCH_down"

            if (curr_low <= (prev_low - thr)) and (c1 > prev_low) and (float(z_volume) >= z_hot) and (adx_val >= adx_min):
                sweep = "bull"
                bos_choch = "BOS_up" if c1 > c0 else "CHOCH_up"

            return sweep, bos_choch
        except Exception as exc:
            logger.error("_liquidity_sweep error: %s | tb=%s", exc, traceback.format_exc())
            return "", ""

    def _order_block(self, df: pd.DataFrame, *, atr_last: float, v: np.ndarray, z_vol_series: np.ndarray) -> str:
        try:
            if len(df) < 40:
                return ""
            if not _finite(atr_last) or atr_last <= 0.0:
                return ""

            o = df["open"].to_numpy(dtype=np.float64, copy=False)
            c = df["close"].to_numpy(dtype=np.float64, copy=False)

            look = 18
            o_s = o[-look:]
            c_s = c[-look:]
            z_s = np.asarray(z_vol_series[-look:], dtype=np.float64)

            body = np.abs(c_s - o_s)
            body_thr = 0.9 * float(atr_last)
            z_thr = max(1.2, float(self.z_vol_hot) * 0.65)

            impulse = (body >= body_thr) & (z_s >= z_thr)
            if not bool(np.any(impulse)):
                return ""

            idx = int(np.argmax(np.where(impulse, body, -1e18)))
            imp_open = float(o_s[idx])
            imp_close = float(c_s[idx])

            last_close = float(c[-1])

            if imp_close < imp_open and last_close > imp_open:
                return "bull_ob"
            if imp_close > imp_open and last_close < imp_open:
                return "bear_ob"

            return ""
        except Exception as exc:
            logger.error("_order_block error: %s | tb=%s", exc, traceback.format_exc())
            return ""

    def _near_round_number(self, *, price: float, atr: float) -> bool:
        try:
            if not _finite(price) or price <= 0.0:
                return False

            step = float(self.rn_step) if float(self.rn_step) > 0.0 else 100.0
            rn_buffer_pct = float(getattr(self.cfg, "rn_buffer_pct", 0.0012) or 0.0012)

            buf = 0.35 * float(atr) if _finite(atr) and atr > 0.0 else float(price) * rn_buffer_pct
            buf = float(max(10.0, buf, float(price) * rn_buffer_pct))

            steps: List[float] = [step]
            if price >= 20_000.0:
                steps += [250.0, 500.0, 1000.0]
            if price >= 60_000.0:
                steps += [2000.0, 5000.0]

            for st in steps:
                lvl = round(price / st) * st
                if abs(price - lvl) <= buf:
                    return True

            return False
        except Exception as exc:
            logger.error("_near_round_number error: %s | tb=%s", exc, traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Divergence (swing-based, not naive diffs)
    # ------------------------------------------------------------------
    def _detect_divergence_swings(self, *, ind: np.ndarray, price: np.ndarray) -> str:
        try:
            n = 35
            min_sep = 4
            if len(price) < n or len(ind) < n:
                return "none"

            p = np.asarray(price[-n:], dtype=np.float64)
            i = np.asarray(ind[-n:], dtype=np.float64)
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(i)):
                return "none"

            lows = self._swing_points(p, mode="low", min_sep=min_sep)
            highs = self._swing_points(p, mode="high", min_sep=min_sep)

            if len(lows) >= 2:
                a, b = lows[-2], lows[-1]
                if p[b] < p[a] and i[b] > i[a]:
                    return "bullish"

            if len(highs) >= 2:
                a, b = highs[-2], highs[-1]
                if p[b] > p[a] and i[b] < i[a]:
                    return "bearish"

            return "none"
        except Exception as exc:
            logger.error("_detect_divergence_swings error: %s | tb=%s", exc, traceback.format_exc())
            return "none"

    def _swing_points(self, x: np.ndarray, *, mode: str, min_sep: int) -> List[int]:
        out: List[int] = []
        try:
            n = len(x)
            if n < 7:
                return out
            w = 2
            sep = int(max(1, min_sep))
            for k in range(w, n - w):
                seg = x[k - w : k + w + 1]
                if mode == "low":
                    if x[k] <= float(np.min(seg)):
                        if not out or (k - out[-1]) >= sep:
                            out.append(k)
                else:
                    if x[k] >= float(np.max(seg)):
                        if not out or (k - out[-1]) >= sep:
                            out.append(k)
            return out
        except Exception:
            return out


__all__ = ["Classic_FeatureEngine", "safe_last", "AnomalyResult"]
