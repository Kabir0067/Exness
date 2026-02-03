# _btc_indicators/feature_engine.py — FIXED (production-grade)
from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_btc import EngineConfig

from .backend import _Indicators
from .logging_ import logger
from .utils import _clip01, _finite, _require_cols, _to_f64, safe_last


@dataclass(frozen=True)
class AnomalyResult:
    score: float
    reasons: List[str]
    blocked: bool
    level: str  # "OK" | "WARN" | "BLOCK"


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

        self._ema_short_p = int(ind_cfg["ema_short"])
        self._ema_mid_p = int(ind_cfg["ema_mid"])
        self._ema_long_p = int(ind_cfg["ema_long"])
        self._ema_vlong_p = int(ind_cfg["ema_vlong"])
        self._atr_p = int(ind_cfg["atr_period"])
        self._rsi_p = int(ind_cfg["rsi_period"])
        self._adx_p = int(ind_cfg["adx_period"])
        self._vol_lb = int(ind_cfg["vol_lookback"])

        self.adx_trend_min = float(getattr(cfg, "adx_trend_lo", 22.0) or 22.0)
        self.adx_impulse_hi = float(getattr(cfg, "adx_impulse_hi", 42.0) or 42.0)
        self.z_vol_hot = float(getattr(cfg, "z_vol_hot", 2.2) or 2.2)
        self.bb_width_range_max = float(getattr(cfg, "bb_width_range_max", 0.0080) or 0.0080)
        self.rn_step = float(getattr(cfg, "rn_step", 100.0) or 100.0)
        self.fvg_min_atr_mult = float(getattr(cfg, "fvg_min_atr_mult", 0.35) or 0.35)
        self.sweep_min_atr_mult = float(getattr(cfg, "sweep_min_atr_mult", 0.15) or 0.15)
        self.vwap_window = int(getattr(cfg, "vwap_window", 30) or 30)

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

        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._cache_out: Optional[Dict[str, Any]] = None
        self._cache_ts: float = 0.0
        self._cache_min_interval_ms: float = float(getattr(cfg, "indicator_cache_min_interval_ms", 0.0) or 0.0)

        # ============== INSTITUTIONAL ALPHA PARAMETERS ==============
        # Linear Regression Trend Analysis
        self._linreg_period = int(getattr(cfg, "linreg_period", 20) or 20)
        self._htf_slope_block_threshold = float(getattr(cfg, "htf_slope_block_threshold", 0.2) or 0.2)
        
        # Volatility Regime Detection
        self._vol_explosive_threshold = float(getattr(cfg, "volatility_explosive_threshold", 1.8) or 1.8)  # Higher for BTC
        self._vol_dead_threshold = float(getattr(cfg, "volatility_dead_threshold", 0.6) or 0.6)
        self._vol_atr_period = int(getattr(cfg, "vol_atr_period", 50) or 50)
        
        # Forecast Continuation
        self._forecast_vol_mult = float(getattr(cfg, "forecast_vol_mult", 1.5) or 1.5)

    # ------------------------------------------------------------------
    # small finite helper (kept internal; no API change)
    # ------------------------------------------------------------------
    @staticmethod
    def _ffill_finite(a: np.ndarray, default: float = 0.0) -> np.ndarray:
        x = np.asarray(a, dtype=np.float64)
        n = int(x.size)
        if n == 0:
            return x
        m = np.isfinite(x)
        if bool(np.all(m)):
            return x
        out = x.copy()
        if not bool(np.any(m)):
            out.fill(float(default))
            return out
        first = int(np.argmax(m))
        out[:first] = out[first]
        src = out.copy()
        idx = np.where(np.isfinite(src), np.arange(n, dtype=np.int64), first).astype(np.int64)
        np.maximum.accumulate(idx, out=idx)
        return src[idx]

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
            out["mtf_aligned"] = bool(mtf_aligned)

            out["anomaly_score"] = float(block_tf_anom.score)
            out["anomaly_reasons"] = [f"{self._anom_tf_for_block}:{r}" for r in block_tf_anom.reasons][:50]
            out["anomaly_level"] = str(block_tf_anom.level)
            out["trade_blocked"] = bool(block_tf_anom.blocked)

            # HARD safety: if anomaly blocks trading, signal_strength becomes neutral (prevents false trades)
            if bool(block_tf_anom.blocked):
                out["signal_strength"] = "neutral"
            else:
                out["signal_strength"] = str(signal_strength)

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

            c = self._ffill_finite(dfp["close"].to_numpy(dtype=np.float64, copy=False))
            h = self._ffill_finite(dfp["high"].to_numpy(dtype=np.float64, copy=False))
            l = self._ffill_finite(dfp["low"].to_numpy(dtype=np.float64, copy=False))
            o = self._ffill_finite(dfp["open"].to_numpy(dtype=np.float64, copy=False))
            v = self._ensure_volume(dfp)  # already float64

            close_last = float(c[-1])
            if not _finite(close_last) or close_last <= 0.0:
                return None, 0.0, AnomalyResult(0.0, ["bad_close"], False, "OK")

            # --- core indicators ---
            ema9 = self.ind.EMA(c, self._ema_short_p)
            ema21 = self.ind.EMA(c, self._ema_mid_p)
            ema50 = self.ind.EMA(c, self._ema_long_p)
            ema200 = self.ind.EMA(c, self._ema_vlong_p)

            atr = self.ind.ATR(h, l, c, self._atr_p)
            rsi = self.ind.RSI(c, self._rsi_p)
            adx = self.ind.ADX(h, l, c, self._adx_p)

            macd, macd_signal, macd_hist = self.ind.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
            bb_u, bb_m, bb_l = self.ind.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)

            atr_last = float(safe_last(atr, 0.0))
            atr_pct = float(atr_last / close_last) if close_last > 0.0 and atr_last > 0.0 else 0.0

            bb_mid = float(safe_last(bb_m, close_last))
            bb_u_last = float(safe_last(bb_u, close_last))
            bb_l_last = float(safe_last(bb_l, close_last))
            bb_width = float((bb_u_last - bb_l_last) / bb_mid) if bb_mid > 0.0 else 0.0

            ret1 = float((close_last / float(c[-2]) - 1.0)) if c.size >= 2 and float(c[-2]) > 0.0 else 0.0
            ret3 = float((close_last / float(c[-4]) - 1.0)) if c.size >= 4 and float(c[-4]) > 0.0 else 0.0

            if ema21.size >= 4 and atr_last > 0.0:
                ema21_slope = float((float(ema21[-1]) - float(ema21[-4])) / max(1e-9, atr_last))
            else:
                ema21_slope = 0.0

            rsi_slope = float(float(rsi[-1]) - float(rsi[-4])) if rsi.size >= 4 else 0.0
            macd_hist_slope = float(float(macd_hist[-1]) - float(macd_hist[-4])) if macd_hist.size >= 4 else 0.0

            vwap = self._vwap_from_bars(dfp, window=self.vwap_window)
            vwap_dist_atr = float((close_last - vwap) / max(1e-9, atr_last)) if _finite(vwap) and atr_last > 0.0 else 0.0

            # --- z-volume (compat: numpy older/newer) ---
            vol_ma = self.ind.SMA(v, self._vol_lb)
            vol_std = self.ind.STDDEV(v, self._vol_lb)

            try:
                vol_std_nn = np.nan_to_num(vol_std, nan=0.0, posinf=0.0, neginf=0.0)
            except TypeError:
                vol_std_nn = np.nan_to_num(vol_std)

            vol_std_safe = np.maximum(vol_std_nn, 1e-9)

            # replace non-finite vol_ma elementwise with v
            vol_ma_safe = np.where(np.isfinite(vol_ma), vol_ma, v)
            z_vol_series = (v - vol_ma_safe) / vol_std_safe
            z_vol_series = self._ffill_finite(z_vol_series, default=0.0)
            zv = float(safe_last(z_vol_series, 0.0))

            bull_fvg, bear_fvg = self._detect_fvg(dfp, atr)
            liq_sweep, bos_choch = self._liquidity_sweep(dfp, atr_last=atr_last, z_volume=zv, adx=adx)
            ob = self._order_block(dfp, atr_last=atr_last, v=v, z_vol_series=z_vol_series)

            near_rn = self._near_round_number(price=close_last, atr=atr_last)

            rsi_div = self._detect_divergence_swings(ind=rsi, price=c)
            macd_div = self._detect_divergence_swings(ind=macd_hist, price=c)

            trend = self._determine_trend(c, ema21, ema50, ema200, adx)

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
                "close": close_last,
                "open": float(o[-1]),
                "high": float(h[-1]),
                "low": float(l[-1]),
                "body": float(abs(float(c[-1]) - float(o[-1]))),

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

                "ret1": float(ret1),
                "ret3": float(ret3),
                "ema21_slope_atr": float(ema21_slope),
                "rsi_slope": float(rsi_slope),
                "macd_hist_slope": float(macd_hist_slope),

                "vwap": float(vwap) if _finite(vwap) else 0.0,
                "vwap_dist_atr": float(vwap_dist_atr),

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

        c_last = float(c[-1])
        e21 = float(safe_last(ema21, default=c_last))
        e50 = float(safe_last(ema50, default=c_last))
        e200 = float(safe_last(ema200, default=c_last))

        if c_last > e21 > e50 and adx_val >= adx_min:
            return "strong_up"
        if c_last < e21 < e50 and adx_val >= adx_min:
            return "strong_down"

        if c_last >= e200:
            return "weak_up"
        if c_last < e200:
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
            high = self._ffill_finite(sub["high"].to_numpy(dtype=np.float64, copy=False))
            low = self._ffill_finite(sub["low"].to_numpy(dtype=np.float64, copy=False))
            close = self._ffill_finite(sub["close"].to_numpy(dtype=np.float64, copy=False))
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

            # FIXED: make "strong" directional if data implies a side (reduces false neutral strong)
            mh = float(m1_data.get("macd_hist", 0.0) or 0.0)
            if "up" in trend or mh > 0.0:
                return "strong_buy"
            if "down" in trend or mh < 0.0:
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

            o = self._ffill_finite(dfp["open"].to_numpy(dtype=np.float64, copy=False))
            h = self._ffill_finite(dfp["high"].to_numpy(dtype=np.float64, copy=False))
            l = self._ffill_finite(dfp["low"].to_numpy(dtype=np.float64, copy=False))
            c = self._ffill_finite(dfp["close"].to_numpy(dtype=np.float64, copy=False))

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

            h = self._ffill_finite(df["high"].to_numpy(dtype=np.float64, copy=False))
            l = self._ffill_finite(df["low"].to_numpy(dtype=np.float64, copy=False))

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

            h = self._ffill_finite(df["high"].to_numpy(dtype=np.float64, copy=False))
            l = self._ffill_finite(df["low"].to_numpy(dtype=np.float64, copy=False))
            c = self._ffill_finite(df["close"].to_numpy(dtype=np.float64, copy=False))

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

            o = self._ffill_finite(df["open"].to_numpy(dtype=np.float64, copy=False))
            c = self._ffill_finite(df["close"].to_numpy(dtype=np.float64, copy=False))

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
    # Divergence (swing-based)
    # ------------------------------------------------------------------
    def _detect_divergence_swings(self, *, ind: np.ndarray, price: np.ndarray) -> str:
        try:
            n = 35
            min_sep = 4
            if len(price) < n or len(ind) < n:
                return "none"

            p = self._ffill_finite(np.asarray(price[-n:], dtype=np.float64), default=0.0)
            i = self._ffill_finite(np.asarray(ind[-n:], dtype=np.float64), default=0.0)

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

    # ==================== INSTITUTIONAL ALPHA FUNCTIONS ====================

    def linreg_trend_slope(self, closes: np.ndarray, period: Optional[int] = None) -> float:
        """
        Calculate Linear Regression Slope normalized to [-1, +1].
        Used for Fractal Market Analysis - detecting H1/M15 trend direction.
        
        Returns:
            float: Normalized slope (-1 = strong bearish, +1 = strong bullish, 0 = flat)
        """
        try:
            period = period or self._linreg_period
            if len(closes) < period:
                return 0.0
            
            y = closes[-period:].astype(np.float64)
            if not np.all(np.isfinite(y)):
                return 0.0
            
            x = np.arange(period, dtype=np.float64)
            
            # Linear regression: y = mx + b
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            if abs(denominator) < 1e-12:
                return 0.0
            
            slope = numerator / denominator
            
            # Normalize by price level (slope per bar as % of price)
            norm_factor = y_mean * 0.0005  # Adjusted for BTC price levels
            if abs(norm_factor) < 1e-12:
                return 0.0
            
            normalized_slope = slope / norm_factor
            return float(np.clip(normalized_slope, -1.0, 1.0))
            
        except Exception as exc:
            logger.error("linreg_trend_slope error: %s | tb=%s", exc, traceback.format_exc())
            return 0.0

    def volatility_regime(self, atr_series: np.ndarray, period: Optional[int] = None) -> Tuple[str, float]:
        """
        Detect volatility regime based on ATR ratio.
        
        Returns:
            Tuple[str, float]: (regime, atr_ratio)
            - regime: "explosive" (high vol), "normal", "dead" (low vol/consolidation)
            - atr_ratio: Current ATR / SMA(ATR, period)
        """
        try:
            period = period or self._vol_atr_period
            if len(atr_series) < period:
                return "normal", 1.0
            
            atr_current = float(atr_series[-1])
            atr_ma = float(np.mean(atr_series[-period:]))
            
            if not _finite(atr_current) or not _finite(atr_ma) or atr_ma <= 0:
                return "normal", 1.0
            
            ratio = atr_current / atr_ma
            
            if ratio >= self._vol_explosive_threshold:
                return "explosive", float(ratio)
            elif ratio <= self._vol_dead_threshold:
                return "dead", float(ratio)
            else:
                return "normal", float(ratio)
                
        except Exception as exc:
            logger.error("volatility_regime error: %s | tb=%s", exc, traceback.format_exc())
            return "normal", 1.0

    def forecast_continuation(self, df: pd.DataFrame, vol_series: np.ndarray) -> str:
        """
        Predict next candle continuation based on price breakout + volume confirmation.
        
        Logic:
        - If close > previous high AND volume > 1.5x average -> "bull_continuation"
        - If close < previous low AND volume > 1.5x average -> "bear_continuation"
        - Otherwise -> "none"
        
        Returns:
            str: "bull_continuation", "bear_continuation", or "none"
        """
        try:
            if len(df) < 3 or len(vol_series) < 20:
                return "none"
            
            close_now = float(df["close"].iloc[-1])
            high_prev = float(df["high"].iloc[-2])
            low_prev = float(df["low"].iloc[-2])
            vol_now = float(vol_series[-1])
            vol_avg = float(np.mean(vol_series[-20:]))
            
            if not all(_finite(x) for x in [close_now, high_prev, low_prev, vol_now, vol_avg]):
                return "none"
            
            vol_mult = self._forecast_vol_mult
            vol_confirmed = vol_now > (vol_avg * vol_mult)
            
            if close_now > high_prev and vol_confirmed:
                return "bull_continuation"
            elif close_now < low_prev and vol_confirmed:
                return "bear_continuation"
            else:
                return "none"
                
        except Exception as exc:
            logger.error("forecast_continuation error: %s | tb=%s", exc, traceback.format_exc())
            return "none"

    def htf_trend_gate(self, h1_closes: Optional[np.ndarray], m15_closes: Optional[np.ndarray]) -> Tuple[bool, bool, float, float]:
        """
        Higher Timeframe Trend Gate - Block signals against the macro trend.
        
        Returns:
            Tuple[bool, bool, float, float]: 
            - buy_allowed: True if H1 not strongly bearish
            - sell_allowed: True if H1 not strongly bullish
            - h1_slope: H1 linear regression slope
            - m15_slope: M15 linear regression slope
        """
        try:
            h1_slope = 0.0
            m15_slope = 0.0
            
            if h1_closes is not None and len(h1_closes) >= self._linreg_period:
                h1_slope = self.linreg_trend_slope(h1_closes)
            
            if m15_closes is not None and len(m15_closes) >= self._linreg_period:
                m15_slope = self.linreg_trend_slope(m15_closes)
            
            threshold = self._htf_slope_block_threshold
            
            # Block Buy if H1 is strongly bearish
            buy_allowed = h1_slope >= -threshold
            # Block Sell if H1 is strongly bullish  
            sell_allowed = h1_slope <= threshold
            
            return buy_allowed, sell_allowed, float(h1_slope), float(m15_slope)
            
        except Exception as exc:
            logger.error("htf_trend_gate error: %s | tb=%s", exc, traceback.format_exc())
            return True, True, 0.0, 0.0

    def god_tier_signal(
        self,
        order_block: str,
        rsi: float,
        divergence: str,
        h1_slope: float,
        fvg_bull: bool,
        fvg_bear: bool,
    ) -> Tuple[bool, str]:
        """
        Detect "God Tier" setups - rare high-probability entries.
        
        Conditions for God Tier BUY:
        - H1 Order Block touched (bull_ob)
        - RSI < 35 (oversold zone)
        - Bullish divergence OR FVG bull
        - H1 not strongly bearish
        
        Returns:
            Tuple[bool, str]: (is_god_tier, direction "buy"/"sell"/"none")
        """
        try:
            # God Tier BUY
            if (
                order_block == "bull_ob"
                and rsi < 35.0
                and (divergence == "bullish" or fvg_bull)
                and h1_slope >= -0.15
            ):
                return True, "buy"
            
            # God Tier SELL
            if (
                order_block == "bear_ob"
                and rsi > 65.0
                and (divergence == "bearish" or fvg_bear)
                and h1_slope <= 0.15
            ):
                return True, "sell"
            
            return False, "none"
            
        except Exception as exc:
            logger.error("god_tier_signal error: %s | tb=%s", exc, traceback.format_exc())
            return False, "none"
