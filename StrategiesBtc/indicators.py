# StrategiesBtc/indicators.py  (BTCUSDm ONLY, 24/7, PRODUCTION-GRADE)
from __future__ import annotations

import logging
import math
import traceback
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
except Exception:
    talib = None  # type: ignore


# Prefer your BTC-only config module name.
from config_btc import EngineConfig
from log_config import LOG_DIR as LOG_ROOT, get_log_path
  # type: ignore

# =============================================================================
# Logging (ERROR only)
# =============================================================================
LOG_DIR = LOG_ROOT

logger = logging.getLogger("feature_engine_btc")
logger.setLevel(logging.ERROR)
logger.propagate = False

if not logger.handlers:
    fh = logging.FileHandler(str(get_log_path("feature_engine_btc.log")), encoding="utf-8")
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    logger.addHandler(fh)


# =============================================================================
# Utils
# =============================================================================
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


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


# =============================================================================
# Indicator backend (TA-Lib + robust fallback)
# =============================================================================
class _Indicators:
    """
    TA-Lib backend with safe fallback to pandas/numpy.
    Fallback is used if talib is missing OR talib throws.
    """

    def __init__(self) -> None:
        self._has_talib = talib is not None

    @staticmethod
    def _ema(x: np.ndarray, period: int) -> np.ndarray:
        s = pd.Series(_to_f64(x))
        return s.ewm(span=int(period), adjust=False).mean().to_numpy(dtype=np.float64)

    @staticmethod
    def _wilder_ema(x: np.ndarray, period: int) -> np.ndarray:
        s = pd.Series(_to_f64(x))
        alpha = 1.0 / max(1, int(period))
        return s.ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64)

    def EMA(self, x: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.EMA(_to_f64(x), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass
        return self._ema(x, period)

    def SMA(self, x: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.SMA(_to_f64(x), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass
        s = pd.Series(_to_f64(x))
        return s.rolling(int(period), min_periods=1).mean().to_numpy(dtype=np.float64)

    def STDDEV(self, x: np.ndarray, period: int) -> np.ndarray:
        try:
            if self._has_talib:
                return talib.STDDEV(_to_f64(x), int(period))  # type: ignore[attr-defined]
        except Exception:
            pass
        s = pd.Series(_to_f64(x))
        return s.rolling(int(period), min_periods=1).std(ddof=0).fillna(0.0).to_numpy(dtype=np.float64)

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
        return self._wilder_ema(tr, period)

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
        avg_gain = self._wilder_ema(gain, period)
        avg_loss = self._wilder_ema(loss, period)
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
        prev_h[0], prev_l[0], prev_c[0] = h[0], l[0], c[0]

        up_move = h - prev_h
        down_move = prev_l - l

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

        atr = self._wilder_ema(tr, period)
        plus_di = 100.0 * (self._wilder_ema(plus_dm, period) / np.maximum(1e-12, atr))
        minus_di = 100.0 * (self._wilder_ema(minus_dm, period) / np.maximum(1e-12, atr))

        dx = 100.0 * (np.abs(plus_di - minus_di) / np.maximum(1e-12, (plus_di + minus_di)))
        adx = self._wilder_ema(dx, period)
        return adx.astype(np.float64)

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
        fast = self._ema(c, fastperiod)
        slow = self._ema(c, slowperiod)
        macd = fast - slow
        signal = self._ema(macd, signalperiod)
        hist = macd - signal
        return macd.astype(np.float64), signal.astype(np.float64), hist.astype(np.float64)

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

        s = pd.Series(_to_f64(c))
        mid = s.rolling(int(timeperiod), min_periods=1).mean()
        std = s.rolling(int(timeperiod), min_periods=1).std(ddof=0).fillna(0.0)
        upper = mid + float(nbdevup) * std
        lower = mid - float(nbdevdn) * std
        return upper.to_numpy(dtype=np.float64), mid.to_numpy(dtype=np.float64), lower.to_numpy(dtype=np.float64)


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

        # ---------------- BTC tuned thresholds (config-driven; strong defaults) ----------------
        # ADX: for BTC M1 you need lower "trend min" vs FX, but still filter chop
        self.adx_trend_min = float(getattr(cfg, "adx_trend_lo", 22.0) or 22.0)
        # impulse means "too hot" (news spike / squeeze release)
        self.adx_impulse_hi = float(getattr(cfg, "adx_impulse_hi", 42.0) or 42.0)

        # Volume z-score hot threshold
        self.z_vol_hot = float(getattr(cfg, "z_vol_hot", 2.2) or 2.2)

        # BB width used for regime classification
        self.bb_width_range_max = float(getattr(cfg, "bb_width_range_max", 0.0080) or 0.0080)

        # Round-number step for BTC scalping (USD distance). Use cfg.rn_step if provided.
        self.rn_step = float(getattr(cfg, "rn_step", 100.0) or 100.0)

        # FVG sensitivity (BTC needs smaller threshold than XAU)
        self.fvg_min_atr_mult = float(getattr(cfg, "fvg_min_atr_mult", 0.35) or 0.35)

        # Liquidity sweep sensitivity
        self.sweep_min_atr_mult = float(getattr(cfg, "sweep_min_atr_mult", 0.15) or 0.15)

        # VWAP window (M1)
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
            dfp = df.iloc[:-shift]
            required = self._min_bars(tf)
            if len(dfp) < required:
                return None, 0.0, AnomalyResult(0.0, [], False, "OK")

            c = dfp["close"].to_numpy(dtype=np.float64)
            h = dfp["high"].to_numpy(dtype=np.float64)
            l = dfp["low"].to_numpy(dtype=np.float64)
            o = dfp["open"].to_numpy(dtype=np.float64)
            v = self._ensure_volume(dfp).astype(np.float64)

            # --- core indicators (config-driven periods) ---
            ema9 = self.ind.EMA(c, int(self.cfg.indicator["ema_short"]))
            ema21 = self.ind.EMA(c, int(self.cfg.indicator["ema_mid"]))
            ema50 = self.ind.EMA(c, int(self.cfg.indicator["ema_long"]))
            ema200 = self.ind.EMA(c, int(self.cfg.indicator["ema_vlong"]))

            atr = self.ind.ATR(h, l, c, int(self.cfg.indicator["atr_period"]))
            rsi = self.ind.RSI(c, int(self.cfg.indicator["rsi_period"]))
            adx = self.ind.ADX(h, l, c, int(self.cfg.indicator["adx_period"]))

            macd, macd_signal, macd_hist = self.ind.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
            bb_u, bb_m, bb_l = self.ind.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)

            # --- derived features (BTC scalping-critical) ---
            close_last = float(c[-1])
            atr_last = float(safe_last(atr, 0.0))
            atr_pct = float(atr_last / close_last) if close_last > 0 and atr_last > 0 else 0.0

            bb_mid = float(safe_last(bb_m, close_last))
            bb_std = float((float(safe_last(bb_u, close_last)) - float(safe_last(bb_l, close_last))) / 4.0) if bb_mid > 0 else 0.0
            bb_width = float((4.0 * bb_std / bb_mid) if bb_mid > 0 else 0.0)

            ret1 = float((close_last / float(c[-2]) - 1.0)) if len(c) >= 2 and float(c[-2]) > 0 else 0.0
            ret3 = float((close_last / float(c[-4]) - 1.0)) if len(c) >= 4 and float(c[-4]) > 0 else 0.0

            ema21_slope = float((float(ema21[-1]) - float(ema21[-4])) / max(1e-9, atr_last)) if len(ema21) >= 4 and atr_last > 0 else 0.0
            rsi_slope = float(float(rsi[-1]) - float(rsi[-4])) if len(rsi) >= 4 else 0.0
            macd_hist_slope = float(float(macd_hist[-1]) - float(macd_hist[-4])) if len(macd_hist) >= 4 else 0.0

            # VWAP (tick_volume-based; best available in MT5 bars)
            vwap = self._vwap_from_bars(dfp, window=self.vwap_window)
            vwap_dist_atr = float((close_last - vwap) / max(1e-9, atr_last)) if _finite(vwap) and atr_last > 0 else 0.0

            # --- z-volume ---
            vol_lookback = int(self.cfg.indicator["vol_lookback"])
            vol_ma = self.ind.SMA(v, vol_lookback)
            vol_std = self.ind.STDDEV(v, vol_lookback)
            vol_std_safe = np.maximum(np.nan_to_num(vol_std, nan=0.0), 1e-9)
            z_vol_series = (v - np.nan_to_num(vol_ma, nan=v)) / vol_std_safe
            zv = float(safe_last(z_vol_series, 0.0))

            # --- patterns / logic (BTC tuned) ---
            bull_fvg, bear_fvg = self._detect_fvg(dfp, atr)
            liq_sweep, bos_choch = self._liquidity_sweep(dfp, atr_last=atr_last, z_volume=zv, adx=adx)
            ob = self._order_block(dfp, atr_last=atr_last, v=v, z_vol_series=z_vol_series)

            near_rn = self._near_round_number(price=close_last, atr=atr_last)

            rsi_div = self._detect_divergence_swings(ind=rsi, price=c)
            macd_div = self._detect_divergence_swings(ind=macd_hist, price=c)  # hist divergence is cleaner for BTC scalps

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

                "bb_upper": float(safe_last(bb_u)),
                "bb_mid": float(safe_last(bb_m)),
                "bb_lower": float(safe_last(bb_l)),
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

        # BTC: trend means EMA stacking + ADX gate
        if c[-1] > ema21[-1] > ema50[-1] and adx_val >= adx_min:
            return "strong_up"
        if c[-1] < ema21[-1] < ema50[-1] and adx_val >= adx_min:
            return "strong_down"

        # softer bias relative to EMA200 (keeps you from fighting major drift)
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
            tp = (sub["high"].to_numpy(dtype=np.float64) + sub["low"].to_numpy(dtype=np.float64) + sub["close"].to_numpy(dtype=np.float64)) / 3.0
            vol = self._ensure_volume(sub).astype(np.float64)
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
        """
        BTC scalping weighting:
          - FVG + Sweep + OB are structure events
          - Divergence adds quality (avoid late entries)
          - z_volume hot confirms impulse (but we also penalize ultra-wide BB)
          - VWAP distance too large reduces quality (mean-reversion snap risk)
        """
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

        # regime-based shaping
        if bb_width > float(self.bb_width_range_max) * 1.8:
            inc -= 0.5  # too expanded: higher whip risk for scalps

        # VWAP distance shaping
        if vwap_dist_atr >= 2.2:
            inc -= 0.6  # stretched -> snapback risk
        elif vwap_dist_atr <= 0.6:
            inc += 0.3  # close to VWAP -> cleaner scalps

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

        # allow "bias + range" alignment
        if (bullish + rng) >= math.ceil(total * 0.84):
            return True
        if (bearish + rng) >= math.ceil(total * 0.84):
            return True

        return False

    def _ensure_volume(self, df: pd.DataFrame) -> np.ndarray:
        try:
            if "tick_volume" in df.columns:
                return df["tick_volume"].fillna(0).to_numpy(dtype=np.float64)
            if "real_volume" in df.columns:
                return df["real_volume"].fillna(0).to_numpy(dtype=np.float64)

            # fallback synthetic volume
            hl_range = (df["high"] - df["low"]).fillna(0).to_numpy(dtype=np.float64)
            close = df["close"].fillna(0).to_numpy(dtype=np.float64)
            close_diff = np.abs(np.diff(close, prepend=close[0]))
            synth = (hl_range + close_diff) * 5000.0
            return synth.astype(np.float64)
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

            o = dfp["open"].to_numpy(dtype=np.float64)
            h = dfp["high"].to_numpy(dtype=np.float64)
            l = dfp["low"].to_numpy(dtype=np.float64)
            c = dfp["close"].to_numpy(dtype=np.float64)

            atr_series = np.asarray(atr_series, dtype=np.float64)
            if len(atr_series) != len(c):
                atr_series = np.full_like(c, float(safe_last(atr_series, default=0.0)))

            z_vol_series = np.asarray(z_vol_series, dtype=np.float64)
            if len(z_vol_series) != len(c):
                z_vol_series = np.zeros_like(c)

            hit_reasons: Dict[str, int] = {}
            score_sum = 0.0
            z_confirm_hits = 0

            for k in range(1, persist_bars + 1):
                atr = float(atr_series[-k])
                if not _finite(atr) or atr <= 0:
                    continue

                o1, h1, l1, c1 = float(o[-k]), float(h[-k]), float(l[-k]), float(c[-k])
                prev_close = float(c[-k - 1])

                z_k = float(z_vol_series[-k])
                if z_k >= float(self._anom_confirm_zvol):
                    z_confirm_hits += 1

                rng = (h1 - l1)
                wick_up = h1 - max(o1, c1)
                wick_dn = min(o1, c1) - l1
                wick_dom = (max(wick_up, wick_dn) / max(1e-9, rng)) if rng > 0 else 0.0

                s = 0.0
                reasons_bar: List[str] = []

                # range spike: BTC news bars / liquidation cascades
                if rng >= float(self._anom_range_atr) * atr:
                    reasons_bar.append("range_spike")
                    s += 1.1

                # wick spike: stop hunts, liquidation wicks
                if wick_dom >= float(self._anom_wick_ratio) and rng >= 1.2 * atr:
                    reasons_bar.append("wick_spike")
                    s += 1.2

                # gap/jump between bars (crypto can jump even without session gaps)
                if abs(o1 - prev_close) >= float(self._anom_gap_atr) * atr:
                    reasons_bar.append("gap_jump")
                    s += 1.0

                # stoprun detection vs recent extremes
                end = len(h) - k
                start = end - lb
                if start >= 0 and (end - start) >= max(6, lb // 2):
                    prev_high = float(np.max(h[start:end]))
                    prev_low = float(np.min(l[start:end]))

                    # sweep high then reject
                    if (h1 > prev_high) and (c1 < prev_high):
                        if (z_k >= float(self._anom_stoprun_zvol)) or (wick_dom >= 0.6):
                            reasons_bar.append("stoprun_high_reject")
                            s += 1.6

                    # sweep low then reject
                    if (l1 < prev_low) and (c1 > prev_low):
                        if (z_k >= float(self._anom_stoprun_zvol)) or (wick_dom >= 0.6):
                            reasons_bar.append("stoprun_low_reject")
                            s += 1.6

                # impulse ADX spike (market too hot)
                if float(adx) >= float(self.adx_impulse_hi) and rng >= 1.7 * atr:
                    reasons_bar.append("impulse_adx_spike")
                    s += 0.7

                # thin liquidity spike (large range but low z-vol)
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

            # BTC: block only if persistent AND strong score
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
        """
        3-bar FVG:
          bull: low[-1] > high[-3] by >= fvg_min_atr_mult * ATR
          bear: low[-3] > high[-1] by >= fvg_min_atr_mult * ATR
        """
        try:
            if len(df) < 5:
                return False, False
            atr_last = float(safe_last(atr))
            if not _finite(atr_last) or atr_last <= 0:
                return False, False

            h = df["high"].to_numpy(dtype=np.float64)
            l = df["low"].to_numpy(dtype=np.float64)

            high_m3 = float(h[-3])
            low_m3 = float(l[-3])
            high_last = float(h[-1])
            low_last = float(l[-1])

            bull_gap = low_last - high_m3
            bear_gap = low_m3 - high_last

            thr = float(self.fvg_min_atr_mult) * atr_last
            bull_fvg = bull_gap >= thr
            bear_fvg = bear_gap >= thr
            return bool(bull_fvg), bool(bear_fvg)
        except Exception as exc:
            logger.error("_detect_fvg error: %s | tb=%s", exc, traceback.format_exc())
            return False, False

    def _liquidity_sweep(self, df: pd.DataFrame, *, atr_last: float, z_volume: float, adx: np.ndarray) -> Tuple[str, str]:
        """
        BTC sweep definition (cleaner):
          - Break recent high/low by >= sweep_min_atr_mult * ATR
          - Close returns inside prior range (rejection)
          - z_volume confirms activity
          - ADX must indicate at least some directional participation
        """
        try:
            if len(df) < 30:
                return "", ""
            if not _finite(atr_last) or atr_last <= 0:
                return "", ""

            h = df["high"].to_numpy(dtype=np.float64)
            l = df["low"].to_numpy(dtype=np.float64)
            c = df["close"].to_numpy(dtype=np.float64)

            look = 24
            prev_high = float(np.max(h[-look:-1]))
            prev_low = float(np.min(l[-look:-1]))

            curr_high = float(h[-1])
            curr_low = float(l[-1])
            c1 = float(c[-1])
            c0 = float(c[-2])

            adx_val = float(safe_last(adx))
            adx_min = float(self.adx_trend_min)
            z_hot = float(self.z_vol_hot) * 0.85  # BTC: allow slightly lower vol confirmations

            thr = float(self.sweep_min_atr_mult) * float(atr_last)

            sweep = ""
            bos_choch = ""

            # sweep high then reject -> bearish sweep
            if (curr_high >= (prev_high + thr)) and (c1 < prev_high) and (float(z_volume) >= z_hot) and (adx_val >= adx_min):
                sweep = "bear"
                bos_choch = "BOS_down" if c1 < c0 else "CHOCH_down"

            # sweep low then reject -> bullish sweep
            if (curr_low <= (prev_low - thr)) and (c1 > prev_low) and (float(z_volume) >= z_hot) and (adx_val >= adx_min):
                sweep = "bull"
                bos_choch = "BOS_up" if c1 > c0 else "CHOCH_up"

            return sweep, bos_choch
        except Exception as exc:
            logger.error("_liquidity_sweep error: %s | tb=%s", exc, traceback.format_exc())
            return "", ""

    def _order_block(self, df: pd.DataFrame, *, atr_last: float, v: np.ndarray, z_vol_series: np.ndarray) -> str:
        """
        BTC OB heuristic (scalping-safe):
          - Find a strong impulse candle (body >= 0.9*ATR) with elevated z-vol
          - If price re-enters that candle zone in the opposite direction -> label OB
        """
        try:
            if len(df) < 40 or len(v) < 40:
                return ""
            if not _finite(atr_last) or atr_last <= 0:
                return ""

            o = df["open"].to_numpy(dtype=np.float64)
            c = df["close"].to_numpy(dtype=np.float64)

            look = 18
            o_s = o[-look:]
            c_s = c[-look:]
            z_s = np.asarray(z_vol_series[-look:], dtype=np.float64)

            body = np.abs(c_s - o_s)
            body_thr = 0.9 * float(atr_last)
            z_thr = max(1.2, float(self.z_vol_hot) * 0.65)

            # candidate impulse candles
            impulse = (body >= body_thr) & (z_s >= z_thr)
            if not bool(np.any(impulse)):
                return ""

            # pick strongest body
            idx = int(np.argmax(np.where(impulse, body, -1e18)))
            imp_open = float(o_s[idx])
            imp_close = float(c_s[idx])

            last_close = float(c[-1])

            # If impulse was bearish (close < open) and now price reclaims above its open -> bullish OB
            if imp_close < imp_open and last_close > imp_open:
                return "bull_ob"

            # If impulse was bullish and now price loses below its open -> bearish OB
            if imp_close > imp_open and last_close < imp_open:
                return "bear_ob"

            return ""
        except Exception as exc:
            logger.error("_order_block error: %s | tb=%s", exc, traceback.format_exc())
            return ""

    def _near_round_number(self, *, price: float, atr: float) -> bool:
        """
        BTC round-number proximity:
          - Uses cfg.rn_step (default 100)
          - Also checks major psychological levels (1k, 5k) when price is high
          - Buffer uses max(0.35*ATR, price*rn_buffer_pct)
        """
        try:
            if not _finite(price) or price <= 0:
                return False

            step = float(self.rn_step)
            if step <= 0:
                step = 100.0

            rn_buffer_pct = float(getattr(self.cfg, "rn_buffer_pct", 0.0012) or 0.0012)  # 0.12% default
            buf = 0.35 * float(atr) if _finite(atr) and atr > 0 else float(price) * rn_buffer_pct
            buf = float(max(10.0, buf, float(price) * rn_buffer_pct))

            steps: List[float] = [step]
            if price >= 20_000:
                steps += [250.0, 500.0, 1000.0]
            if price >= 60_000:
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
        """
        Swing-based divergence:
          - find two recent swing lows/highs in last N bars
          - bullish: price makes lower low, indicator makes higher low
          - bearish: price makes higher high, indicator makes lower high
        """
        try:
            N = 35
            min_sep = 4
            if len(price) < N or len(ind) < N:
                return "none"

            p = price[-N:].astype(np.float64, copy=False)
            i = ind[-N:].astype(np.float64, copy=False)
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(i)):
                return "none"

            lows = self._swing_points(p, mode="low", min_sep=min_sep)
            highs = self._swing_points(p, mode="high", min_sep=min_sep)

            # bullish divergence
            if len(lows) >= 2:
                a, b = lows[-2], lows[-1]
                if p[b] < p[a] and i[b] > i[a]:
                    return "bullish"

            # bearish divergence
            if len(highs) >= 2:
                a, b = highs[-2], highs[-1]
                if p[b] > p[a] and i[b] < i[a]:
                    return "bearish"

            return "none"
        except Exception as exc:
            logger.error("_detect_divergence_swings error: %s | tb=%s", exc, traceback.format_exc())
            return "none"

    def _swing_points(self, x: np.ndarray, *, mode: str, min_sep: int) -> List[int]:
        """
        Lightweight swing detector:
          - swing low: x[k] is min in [k-2..k+2]
          - swing high: x[k] is max in [k-2..k+2]
        Enforces minimum separation.
        """
        out: List[int] = []
        try:
            n = len(x)
            if n < 7:
                return out
            w = 2
            for k in range(w, n - w):
                seg = x[k - w : k + w + 1]
                if mode == "low":
                    if x[k] <= float(np.min(seg)):
                        if not out or (k - out[-1]) >= int(min_sep):
                            out.append(k)
                else:
                    if x[k] >= float(np.max(seg)):
                        if not out or (k - out[-1]) >= int(min_sep):
                            out.append(k)
            return out
        except Exception:
            return out


__all__ = ["Classic_FeatureEngine", "safe_last", "AnomalyResult"]
