# Strategies/feature_engine.py  (PRODUCTION-GRADE, OPTIMIZED / NO-STRUCTURE-CHANGE)
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
except Exception:  # TA-Lib may be missing in some environments
    talib = None  # type: ignore

from config_xau import EngineConfig
from log_config import LOG_DIR as LOG_ROOT, get_log_path

# =========================
# Logging (ERROR only)
# =========================
LOG_DIR = LOG_ROOT

logger = logging.getLogger("feature_engine")
logger.setLevel(logging.ERROR)
logger.propagate = False

if not logger.handlers:
    fh = logging.FileHandler(str(get_log_path("feature_engine.log")), encoding="utf-8", delay=True)
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    logger.addHandler(fh)


# =========================
# Utils
# =========================
def safe_last(arr: np.ndarray, default: float = 0.0) -> float:
    """Return last finite value from array; else default."""
    try:
        if arr is None or len(arr) == 0:
            return float(default)
        v = float(arr[-1])
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


# =========================
# Indicator backend (TA-Lib + fast numpy fallback)
# =========================
class _Indicators:
    """
    TA-Lib backend with fast NumPy fallback.
    Fallback is used if talib is missing OR talib throws.

    Goals:
      - no pandas overhead in fallback (scalping-speed)
      - stable for NaN/inf (failsafe clamps)
    """

    def __init__(self) -> None:
        self._has_talib = talib is not None

    @staticmethod
    def _ema_np(x: np.ndarray, period: int, *, wilder: bool = False) -> np.ndarray:
        x = _to_f64(x)
        n = int(max(1, period))
        out = np.empty_like(x, dtype=np.float64)
        if x.size == 0:
            return out

        alpha = (1.0 / n) if wilder else (2.0 / (n + 1.0))
        one_m = 1.0 - alpha

        v0 = float(x[0])
        if not np.isfinite(v0):
            v0 = 0.0
        out[0] = v0

        for i in range(1, x.size):
            xi = float(x[i])
            if not np.isfinite(xi):
                xi = float(out[i - 1])
            out[i] = alpha * xi + one_m * out[i - 1]
        return out

    @staticmethod
    def _sma_np(x: np.ndarray, period: int) -> np.ndarray:
        x = _to_f64(x)
        n = int(max(1, period))
        out = np.empty_like(x, dtype=np.float64)
        if x.size == 0:
            return out

        cs = np.cumsum(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
        for i in range(x.size):
            if i < n:
                out[i] = cs[i] / float(i + 1)
            else:
                out[i] = (cs[i] - cs[i - n]) / float(n)
        return out

    @staticmethod
    def _std_np(x: np.ndarray, period: int) -> np.ndarray:
        x = _to_f64(x)
        n = int(max(1, period))
        out = np.empty_like(x, dtype=np.float64)
        if x.size == 0:
            return out

        x0 = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cs = np.cumsum(x0, dtype=np.float64)
        cs2 = np.cumsum(x0 * x0, dtype=np.float64)

        for i in range(x.size):
            if i < n:
                m = cs[i] / float(i + 1)
                v = (cs2[i] / float(i + 1)) - (m * m)
            else:
                s = cs[i] - cs[i - n]
                s2 = cs2[i] - cs2[i - n]
                m = s / float(n)
                v = (s2 / float(n)) - (m * m)

            if v < 0.0:
                v = 0.0
            out[i] = math.sqrt(v)
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

        h = _to_f64(h)
        l = _to_f64(l)
        c = _to_f64(c)
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

        c = _to_f64(c)
        if c.size == 0:
            return np.asarray(c, dtype=np.float64)

        delta = np.empty_like(c)
        delta[0] = 0.0
        delta[1:] = c[1:] - c[:-1]

        gain = np.maximum(delta, 0.0)
        loss = np.maximum(-delta, 0.0)

        avg_gain = self._ema_np(gain, int(period), wilder=True)
        avg_loss = self._ema_np(loss, int(period), wilder=True)

        rs = avg_gain / np.maximum(1e-12, avg_loss)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.astype(np.float64, copy=False)

    def ADX(self, h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.ADX(_to_f64(h), _to_f64(l), _to_f64(c), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass

        h = _to_f64(h)
        l = _to_f64(l)
        c = _to_f64(c)
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
        plus_di = 100.0 * (self._ema_np(plus_dm, int(period), wilder=True) / np.maximum(1e-12, atr))
        minus_di = 100.0 * (self._ema_np(minus_dm, int(period), wilder=True) / np.maximum(1e-12, atr))

        dx = 100.0 * (np.abs(plus_di - minus_di) / np.maximum(1e-12, (plus_di + minus_di)))
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

        c = _to_f64(c)
        fast = self._ema_np(c, int(fastperiod), wilder=False)
        slow = self._ema_np(c, int(slowperiod), wilder=False)
        macd = fast - slow
        signal = self._ema_np(macd, int(signalperiod), wilder=False)
        hist = macd - signal
        return macd.astype(np.float64, copy=False), signal.astype(np.float64, copy=False), hist.astype(np.float64, copy=False)

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

        c = _to_f64(c)
        mid = self._sma_np(c, int(timeperiod))
        std = self._std_np(c, int(timeperiod))
        upper = mid + float(nbdevup) * std
        lower = mid - float(nbdevdn) * std
        return upper.astype(np.float64, copy=False), mid.astype(np.float64, copy=False), lower.astype(np.float64, copy=False)


@dataclass(frozen=True)
class AnomalyResult:
    score: float
    reasons: List[str]
    blocked: bool
    level: str  # "OK" | "WARN" | "BLOCK"


class Classic_FeatureEngine:
    """
    Feature engine (M1 entry with M5/M15 confirmations).

    compute_indicators(df_dict, shift=1) -> dict:
      - out["M1"/"M5"/"M15"] per-TF indicators + patterns + anomaly fields
      - out["confluence_score"], out["signal_strength"], out["mtf_aligned"]
      - out["anomaly_score"], out["anomaly_reasons"], out["anomaly_level"], out["trade_blocked"]
        (block based on cfg.anom_tf_for_block; default "M1")
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

        # ---------- thresholds (config-driven; backward compatible keys) ----------
        self.rn_step_xau = float(getattr(cfg, "rn_step_xau", 5.0) or 5.0)

        # prefer config_xau fields if present
        self.adx_trend_min = float(
            getattr(cfg, "adx_trend_min", getattr(cfg, "adx_trend_hi", 25.0)) or 25.0
        )
        self.adx_impulse_hi = float(
            getattr(cfg, "adx_impulse_hi", max(40.0, float(getattr(cfg, "adx_trend_hi", 25.0)) + 15.0)) or 45.0
        )

        self.z_vol_hot = float(getattr(cfg, "z_vol_hot", 2.5) or 2.5)
        self.bb_width_range_max = float(getattr(cfg, "bb_width_range_max", 0.005) or 0.005)

        # anomaly config
        self._anom_wick_ratio = float(getattr(cfg, "anom_wick_ratio", 0.70) or 0.70)
        self._anom_range_atr = float(getattr(cfg, "anom_range_atr", 2.2) or 2.2)
        self._anom_gap_atr = float(getattr(cfg, "anom_gap_atr", 0.55) or 0.55)
        self._anom_stoprun_lookback = int(getattr(cfg, "anom_stoprun_lookback", 12) or 12)
        self._anom_stoprun_zvol = float(getattr(cfg, "anom_stoprun_zvol", 2.5) or 2.5)

        self._anom_warn_score = float(getattr(cfg, "anom_warn_score", 1.2) or 1.2)
        self._anom_block_score = float(getattr(cfg, "anom_block_score", 2.6) or 2.6)
        self._anom_persist_bars = int(getattr(cfg, "anom_persist_bars", 3) or 3)
        self._anom_persist_min_hits = int(getattr(cfg, "anom_persist_min_hits", 2) or 2)
        self._anom_confirm_zvol = float(getattr(cfg, "anom_confirm_zvol", 1.8) or 1.8)

        self._anom_tf_for_block = str(getattr(cfg, "anom_tf_for_block", "M1") or "M1")

        # indicator compute cache (single-entry, safe)
        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._cache_out: Optional[Dict[str, Any]] = None
        self._cache_ts: float = 0.0
        self._cache_min_interval_ms: float = float(getattr(cfg, "indicator_cache_min_interval_ms", 0.0) or 0.0)

    # ------------------- config validation -------------------
    def _validate_cfg(self) -> None:
        ind = getattr(self.cfg, "indicator", None)
        if not isinstance(ind, dict):
            raise RuntimeError("EngineConfig.indicator dict is required for Classic_FeatureEngine")
        missing = [k for k in self._REQUIRED_IND_KEYS if k not in ind]
        if missing:
            raise RuntimeError(f"EngineConfig.indicator missing keys: {missing}")

    # ------------------- public API -------------------
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

            # ---- cache key (fast) ----
            computed_cache_key: Optional[Tuple[Any, ...]] = None
            try:
                key_parts: List[Tuple[str, int, int]] = []
                for tf in frames:
                    df = df_dict.get(tf)
                    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                        key_parts.append((tf, -1, 0))
                        continue
                    if len(df) < (int(shift) + 2):
                        key_parts.append((tf, -1, int(len(df))))
                        continue

                    if "time" in df.columns:
                        t = df["time"].iloc[-1 - int(shift)]
                        ts_ns = int(pd.Timestamp(t).value)
                    else:
                        ts_ns = int(len(df))

                    key_parts.append((tf, ts_ns, int(len(df))))

                computed_cache_key = (int(shift), tuple(key_parts))

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
                    logger.error("Data insufficient for %s: len=%s < required+shift=%s", tf, len(df), required + shift)
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

            # Update cache
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

    # ------------------- per-tf compute -------------------
    def _compute_tf(self, *, tf: str, df: pd.DataFrame, shift: int) -> Tuple[Optional[Dict[str, Any]], float, AnomalyResult]:
        try:
            # NOTE: slice without copy; pandas gives a view-like object often
            dfp = df.iloc[:-shift]
            required = self._min_bars(tf)
            if len(dfp) < required:
                return None, 0.0, AnomalyResult(0.0, [], False, "OK")

            # numpy views (fast)
            c = dfp["close"].to_numpy(dtype=np.float64, copy=False)
            h = dfp["high"].to_numpy(dtype=np.float64, copy=False)
            l = dfp["low"].to_numpy(dtype=np.float64, copy=False)
            o = dfp["open"].to_numpy(dtype=np.float64, copy=False)
            v = self._ensure_volume(dfp).astype(np.float64, copy=False)

            # core indicators
            ema9 = self.ind.EMA(c, int(self.cfg.indicator["ema_short"]))
            ema21 = self.ind.EMA(c, int(self.cfg.indicator["ema_mid"]))
            ema50 = self.ind.EMA(c, int(self.cfg.indicator["ema_long"]))
            ema200 = self.ind.EMA(c, int(self.cfg.indicator["ema_vlong"]))

            atr = self.ind.ATR(h, l, c, int(self.cfg.indicator["atr_period"]))
            rsi = self.ind.RSI(c, int(self.cfg.indicator["rsi_period"]))
            adx = self.ind.ADX(h, l, c, int(self.cfg.indicator["adx_period"]))

            macd, macd_signal, macd_hist = self.ind.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
            bb_u, bb_m, bb_l = self.ind.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)

            # z-volume (rolling mean/std) – fast numpy fallback
            vol_lookback = int(self.cfg.indicator["vol_lookback"])
            vol_ma = self.ind.SMA(v, vol_lookback)
            vol_std = self.ind.STDDEV(v, vol_lookback)
            vol_std_safe = np.maximum(np.nan_to_num(vol_std, nan=0.0), 1e-9)
            z_vol_series = (v - np.nan_to_num(vol_ma, nan=v)) / vol_std_safe
            zv = float(safe_last(z_vol_series, 0.0))

            # patterns / logic
            bull_fvg, bear_fvg = self._detect_fvg(dfp, atr)
            liq_sweep, bos_choch = self._liquidity_sweep(dfp, z_volume=zv, adx=adx)
            ob = self._order_block(dfp, atr, v)
            near_rn = self._near_round_number(float(c[-1]), float(safe_last(atr)))

            rsi_div = self._detect_divergence(rsi, c)
            macd_div = self._detect_divergence(macd, c)

            trend = self._determine_trend(c, ema21, ema50, ema200, adx)

            # anomaly detection
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
            )

            tf_out = {
                "close": float(c[-1]),
                "open": float(o[-1]),
                "high": float(h[-1]),
                "low": float(l[-1]),
                "body": float(abs(float(c[-1]) - float(o[-1]))),
                "atr": float(safe_last(atr)),
                "ema9": float(safe_last(ema9)),
                "ema21": float(safe_last(ema21)),
                "ema50": float(safe_last(ema50)),
                "ema200": float(safe_last(ema200)),
                "rsi": float(safe_last(rsi)),
                "adx": float(safe_last(adx)),
                "macd": float(safe_last(macd)),
                "macd_signal": float(safe_last(macd_signal)),
                "macd_hist": float(safe_last(macd_hist)),
                "bb_upper": float(safe_last(bb_u)),
                "bb_mid": float(safe_last(bb_m)),
                "bb_lower": float(safe_last(bb_l)),
                "z_volume": float(zv),
                "fvg_bull": bool(bull_fvg),
                "fvg_bear": bool(bear_fvg),
                "liquidity_sweep": str(liq_sweep),
                "bos_choch": str(bos_choch),
                "order_block": str(ob),
                "near_round": bool(near_rn),
                "rsi_div": str(rsi_div),
                "macd_div": str(macd_div),
                "trend": str(trend),
                "anomaly_score": float(anom.score),
                "anomaly_reasons": list(anom.reasons)[:50],
                "anomaly_level": str(anom.level),
                "trade_blocked": bool(anom.blocked),
            }

            return tf_out, float(inc), anom

        except Exception as exc:
            logger.error("_compute_tf error: %s | tb=%s", exc, traceback.format_exc())
            return None, 0.0, AnomalyResult(0.0, [], False, "OK")

    # ------------------- core helpers -------------------
    def _min_bars(self, timeframe: str) -> int:
        if timeframe == "M1":
            return int(getattr(self.cfg, "min_bars_m1", 220) or 220)
        if timeframe in {"M5", "M15"}:
            return int(getattr(self.cfg, "min_bars_m5_m15", 200) or 200)
        return int(getattr(self.cfg, "min_bars_default", 180) or 180)

    def _determine_trend(self, c: np.ndarray, ema21: np.ndarray, ema50: np.ndarray, ema200: np.ndarray, adx: np.ndarray) -> str:
        adx_val = float(safe_last(adx))
        adx_min = float(self.adx_trend_min)

        c1 = float(c[-1])
        e21 = float(safe_last(ema21))
        e50 = float(safe_last(ema50))
        e200 = float(safe_last(ema200))

        if c1 > e21 > e50 and adx_val > adx_min:
            return "strong_up"
        if c1 < e21 < e50 and adx_val > adx_min:
            return "strong_down"
        if c1 > e200:
            return "weak_up"
        if c1 < e200:
            return "weak_down"
        return "range"

    def _calculate_confluence_increment(self, has_fvg: bool, has_sweep: bool, has_ob: bool, has_near_rn: bool, has_div: bool, z_volume: float) -> float:
        inc = 0.0
        if has_fvg:
            inc += 1.0
        if has_sweep:
            inc += 1.0
        if has_ob:
            inc += 1.0
        if has_near_rn:
            inc += 0.5
        if has_div:
            inc += 1.0
        if float(z_volume) > float(self.z_vol_hot):
            inc += 1.0
        return inc

    def _determine_signal_strength(self, score: float, m1_data: Dict[str, Any]) -> str:
        score = float(score)
        trend = str(m1_data.get("trend", ""))
        sweep = str(m1_data.get("liquidity_sweep", ""))
        if score >= 4.0:
            if m1_data.get("fvg_bull") or sweep == "bull" or ("up" in trend):
                return "strong_buy"
            if m1_data.get("fvg_bear") or sweep == "bear" or ("down" in trend):
                return "strong_sell"
            return "strong"
        if score >= 3.0:
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

        if bullish >= total * 0.6:
            return True
        if bearish >= total * 0.6:
            return True
        if (bullish + rng) >= total * 0.8:
            return True
        if (bearish + rng) >= total * 0.8:
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

    # ------------------- anomaly detector -------------------
    def _detect_market_anomalies(self, *, dfp: pd.DataFrame, atr_series: np.ndarray, z_vol_series: np.ndarray, adx: float) -> AnomalyResult:
        try:
            persist_bars = max(1, int(self._anom_persist_bars))
            min_hits = max(1, int(self._anom_persist_min_hits))
            lb = int(self._anom_stoprun_lookback)

            if len(dfp) < max(60, lb + persist_bars + 5):
                return AnomalyResult(0.0, [], False, "OK")

            o = dfp["open"].to_numpy(dtype=np.float64, copy=False)
            h = dfp["high"].to_numpy(dtype=np.float64, copy=False)
            l = dfp["low"].to_numpy(dtype=np.float64, copy=False)
            c = dfp["close"].to_numpy(dtype=np.float64, copy=False)

            atr_series = np.asarray(atr_series, dtype=np.float64)
            if len(atr_series) != len(c):
                atr_series = np.full_like(c, float(safe_last(atr_series, default=0.0)), dtype=np.float64)

            z_vol_series = np.asarray(z_vol_series, dtype=np.float64)
            if len(z_vol_series) != len(c):
                z_vol_series = np.zeros_like(c, dtype=np.float64)

            hit_reasons: Dict[str, int] = {}
            score_sum = 0.0
            z_confirm_hits = 0

            for k in range(1, persist_bars + 1):
                atr = float(atr_series[-k])
                if not _finite(atr) or atr <= 0.0:
                    continue

                o1, h1, l1, c1 = float(o[-k]), float(h[-k]), float(l[-k]), float(c[-k])
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

                if rng >= self._anom_range_atr * atr:
                    reasons_bar.append("range_spike")
                    s += 1.0

                if wick_dom >= self._anom_wick_ratio and rng >= 1.2 * atr:
                    reasons_bar.append("wick_spike")
                    s += 1.1

                if abs(o1 - prev_close) >= self._anom_gap_atr * atr:
                    reasons_bar.append("gap_jump")
                    s += 1.0

                end = len(h) - k
                start = end - lb
                if start >= 0 and (end - start) >= max(3, lb // 2):
                    prev_high = float(np.max(h[start:end]))
                    prev_low = float(np.min(l[start:end]))

                    if (h1 > prev_high) and (c1 < prev_high):
                        if (z_k >= self._anom_stoprun_zvol) or (wick_dom >= 0.6):
                            reasons_bar.append("stoprun_high_reject")
                            s += 1.5

                    if (l1 < prev_low) and (c1 > prev_low):
                        if (z_k >= self._anom_stoprun_zvol) or (wick_dom >= 0.6):
                            reasons_bar.append("stoprun_low_reject")
                            s += 1.5

                if float(adx) >= float(self.adx_impulse_hi) and rng >= 1.8 * atr:
                    reasons_bar.append("impulse_adx_spike")
                    s += 0.7

                if rng >= 1.8 * atr and float(z_k) < 0.5:
                    reasons_bar.append("thin_liquidity_spike")
                    s += 0.6

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

    # ------------------- patterns -------------------
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

            bull_fvg = (bull_gap > 0.5 * atr_last)
            bear_fvg = (bear_gap > 0.5 * atr_last)
            return bool(bull_fvg), bool(bear_fvg)
        except Exception as exc:
            logger.error("_detect_fvg error: %s | tb=%s", exc, traceback.format_exc())
            return False, False

    def _liquidity_sweep(self, df: pd.DataFrame, z_volume: float, adx: np.ndarray) -> Tuple[str, str]:
        try:
            if len(df) < 25:
                return "", ""

            h = df["high"].to_numpy(dtype=np.float64, copy=False)
            l = df["low"].to_numpy(dtype=np.float64, copy=False)
            closes = df["close"].to_numpy(dtype=np.float64, copy=False)

            prev_high = float(np.max(h[-20:-1]))
            prev_low = float(np.min(l[-20:-1]))
            curr_high = float(h[-1])
            curr_low = float(l[-1])
            c1 = float(closes[-1])
            c0 = float(closes[-2])

            adx_val = float(safe_last(adx))
            sweep = ""
            bos_choch = ""

            z_hot = float(self.z_vol_hot)
            adx_min = float(self.adx_trend_min)

            if curr_high > prev_high and float(z_volume) > z_hot and adx_val > adx_min:
                sweep = "bear"
                bos_choch = "BOS_down" if c1 < c0 else "CHOCH_down"

            if curr_low < prev_low and float(z_volume) > z_hot and adx_val > adx_min:
                sweep = "bull"
                bos_choch = "BOS_up" if c1 > c0 else "CHOCH_up"

            return sweep, bos_choch
        except Exception as exc:
            logger.error("_liquidity_sweep error: %s | tb=%s", exc, traceback.format_exc())
            return "", ""

    def _order_block(self, df: pd.DataFrame, atr: np.ndarray, v: np.ndarray) -> str:
        try:
            if len(df) < 12 or len(v) < 12:
                return ""

            slice_size = 10
            o_vals = df["open"].to_numpy(dtype=np.float64, copy=False)[-slice_size:]
            c_vals = df["close"].to_numpy(dtype=np.float64, copy=False)[-slice_size:]
            v_slice = v[-slice_size:].astype(np.float64, copy=False)

            atr_last = float(safe_last(atr))
            if not _finite(atr_last) or atr_last <= 0.0:
                return ""

            # fast: compare with slice mean (stable + scalable)
            v_mean = float(np.mean(v_slice)) if v_slice.size else 0.0
            vol_ok = v_slice[:-1] > 1.5 * max(1e-9, v_mean)

            bear_body = (o_vals[:-1] - c_vals[:-1])
            bull_body = (c_vals[:-1] - o_vals[:-1])

            bear_mask = (bear_body > 1.5 * atr_last) & vol_ok
            if np.any(bear_mask):
                idx = int(np.argmax(np.where(bear_mask, bear_body, -1e12)))
                if float(c_vals[-1]) > float(o_vals[idx]):
                    return "bull_ob"

            bull_mask = (bull_body > 1.5 * atr_last) & vol_ok
            if np.any(bull_mask):
                idx = int(np.argmax(np.where(bull_mask, bull_body, -1e12)))
                if float(c_vals[-1]) < float(o_vals[idx]):
                    return "bear_ob"

            return ""
        except Exception as exc:
            logger.error("_order_block error: %s | tb=%s", exc, traceback.format_exc())
            return ""

    def _near_round_number(self, price: float, atr: float) -> bool:
        try:
            if not _finite(price):
                return False

            step = float(self.rn_step_xau)
            if step <= 0.0:
                return False

            # O(1) nearest-multiple check (faster than arange)
            # distance to nearest multiple of step
            rem = math.fmod(float(price), step)
            rem = abs(rem)
            dist = min(rem, abs(step - rem))

            buf = float(atr) * 0.5 if _finite(atr) and atr > 0.0 else float(price) * 0.001
            buf = float(max(0.05, buf))
            return bool(dist <= buf)
        except Exception as exc:
            logger.error("_near_round_number error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _detect_divergence(self, ind: np.ndarray, price: np.ndarray) -> str:
        try:
            if len(ind) < 6 or len(price) < 6:
                return "none"

            p = price[-5:].astype(np.float64, copy=False)
            i = ind[-5:].astype(np.float64, copy=False)
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(i)):
                return "none"

            p_diffs = np.diff(p)
            i_diffs = np.diff(i)

            bull = int(np.sum((p_diffs < 0.0) & (i_diffs > 0.0)))
            bear = int(np.sum((p_diffs > 0.0) & (i_diffs < 0.0)))

            if bull > bear and bull >= 2:
                return "bullish"
            if bear > bull and bear >= 2:
                return "bearish"
            return "none"
        except Exception as exc:
            logger.error("_detect_divergence error: %s | tb=%s", exc, traceback.format_exc())
            return "none"


__all__ = ["Classic_FeatureEngine", "safe_last", "AnomalyResult"]

