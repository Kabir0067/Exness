from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_xau import EngineConfig

from .backend import _Indicators
from .logging_ import logger
from .utils import _finite, _require_cols, _ts_to_ns, safe_last


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

        # optional tuning knobs (no breaking changes)
        self._fvg_min_gap_atr = float(getattr(cfg, "fvg_min_gap_atr", 0.5) or 0.5)
        self._ob_body_atr = float(getattr(cfg, "ob_body_atr", 1.5) or 1.5)
        self._ob_vol_mult = float(getattr(cfg, "ob_vol_mult", 1.5) or 1.5)
        self._sweep_lookback = int(getattr(cfg, "sweep_lookback", 20) or 20)

        # ============== INSTITUTIONAL ALPHA PARAMETERS ==============
        # Linear Regression Trend Analysis
        self._linreg_period = int(getattr(cfg, "linreg_period", 20) or 20)
        self._htf_slope_block_threshold = float(getattr(cfg, "htf_slope_block_threshold", 0.2) or 0.2)
        
        # Volatility Regime Detection
        self._vol_explosive_threshold = float(getattr(cfg, "volatility_explosive_threshold", 1.5) or 1.5)
        self._vol_dead_threshold = float(getattr(cfg, "volatility_dead_threshold", 0.7) or 0.7)
        self._vol_atr_period = int(getattr(cfg, "vol_atr_period", 50) or 50)
        
        # Enhanced Order Block (H1 anchoring)
        self._ob_h1_lookback = int(getattr(cfg, "ob_h1_lookback", 50) or 50)
        
        # Forecast Continuation
        self._forecast_vol_mult = float(getattr(cfg, "forecast_vol_mult", 1.5) or 1.5)


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
            # shift=0 is valid for real-time (forming-candle) calculations.
            if not isinstance(shift, int) or shift < 0:
                logger.error("compute_indicators: invalid shift=%s", shift)
                return None
            if "M1" not in df_dict or not isinstance(df_dict.get("M1"), pd.DataFrame) or df_dict["M1"].empty:
                logger.error("compute_indicators: M1 missing/empty (required)")
                return None

            frames = [tf for tf in self.timeframes if tf in df_dict]
            if not frames:
                logger.error("compute_indicators: no known timeframes in df_dict")
                return None

            # ---- cache key (fast + robust time parsing) ----
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

                    ts_ns = 0
                    if "time" in df.columns:
                        t = df["time"].iloc[-1 - sh]
                        ts_ns = _ts_to_ns(t)
                    else:
                        ts_ns = int(len(df))

                    key_parts.append((tf, int(ts_ns), int(len(df))))

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

            if block_tf_anom is None:
                block_tf_anom = AnomalyResult(
                    score=float(m1_data.get("anomaly_score", 0.0)),
                    reasons=list(m1_data.get("anomaly_reasons", []))[:50],
                    blocked=bool(m1_data.get("trade_blocked", False)),
                    level=str(m1_data.get("anomaly_level", "OK")),
                )

            signal_strength = self._determine_signal_strength(confluence_score, m1_data)

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
            # IMPORTANT: df.iloc[:-0] returns empty, so handle shift=0 explicitly.
            dfp = df if int(shift) == 0 else df.iloc[:-shift]
            required = self._min_bars(tf)
            if len(dfp) < required:
                return None, 0.0, AnomalyResult(0.0, [], False, "OK")

            c = dfp["close"].to_numpy(dtype=np.float64, copy=False)
            h = dfp["high"].to_numpy(dtype=np.float64, copy=False)
            l = dfp["low"].to_numpy(dtype=np.float64, copy=False)
            o = dfp["open"].to_numpy(dtype=np.float64, copy=False)
            v = self._ensure_volume(dfp).astype(np.float64, copy=False)

            # hard gate: last bar must be finite OHLC (avoid NaN-propagation false signals)
            if not (np.isfinite(c[-1]) and np.isfinite(h[-1]) and np.isfinite(l[-1]) and np.isfinite(o[-1])):
                return None, 0.0, AnomalyResult(0.0, ["bad_ohlc_last"], True, "BLOCK")

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

            # z-volume (FIXED: no numpy nan_to_num with array nan=...)
            vol_lookback = int(self.cfg.indicator["vol_lookback"])
            vol_ma = self.ind.SMA(v, vol_lookback)
            vol_std = self.ind.STDDEV(v, vol_lookback)

            vol_ma_filled = np.where(np.isfinite(vol_ma), vol_ma, v)
            vol_std_filled = np.where(np.isfinite(vol_std), vol_std, 0.0)
            vol_std_safe = np.maximum(vol_std_filled, 1e-9)

            z_vol_series = (v - vol_ma_filled) / vol_std_safe
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
        # HARD gate: if blocked => neutral (prevent false orders)
        if bool(m1_data.get("trade_blocked", False)):
            return "neutral"

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
                atr_series = np.full_like(c, float(safe_last(atr_series, 0.0)), dtype=np.float64)

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

            thr = float(self._fvg_min_gap_atr) * atr_last
            bull_fvg = (bull_gap > thr)
            bear_fvg = (bear_gap > thr)
            return bool(bull_fvg), bool(bear_fvg)
        except Exception as exc:
            logger.error("_detect_fvg error: %s | tb=%s", exc, traceback.format_exc())
            return False, False

    def _liquidity_sweep(self, df: pd.DataFrame, z_volume: float, adx: np.ndarray) -> Tuple[str, str]:
        try:
            lb = int(max(10, self._sweep_lookback))
            if len(df) < (lb + 5):
                return "", ""

            h = df["high"].to_numpy(dtype=np.float64, copy=False)
            l = df["low"].to_numpy(dtype=np.float64, copy=False)
            closes = df["close"].to_numpy(dtype=np.float64, copy=False)

            prev_high = float(np.max(h[-(lb + 1) : -1]))
            prev_low = float(np.min(l[-(lb + 1) : -1]))
            curr_high = float(h[-1])
            curr_low = float(l[-1])
            c1 = float(closes[-1])
            c0 = float(closes[-2])

            adx_val = float(safe_last(adx))
            sweep = ""
            bos_choch = ""

            z_hot = float(self.z_vol_hot)
            adx_min = float(self.adx_trend_min)

            hit_high = bool(curr_high > prev_high and float(z_volume) > z_hot and adx_val > adx_min)
            hit_low = bool(curr_low < prev_low and float(z_volume) > z_hot and adx_val > adx_min)

            if hit_high and hit_low:
                return "", ""

            if hit_high:
                sweep = "bear"
                bos_choch = "BOS_down" if c1 < c0 else "CHOCH_down"
            elif hit_low:
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

            v_mean = float(np.mean(v_slice)) if v_slice.size else 0.0
            vol_ok = v_slice[:-1] > float(self._ob_vol_mult) * max(1e-9, v_mean)

            bear_body = (o_vals[:-1] - c_vals[:-1])
            bull_body = (c_vals[:-1] - o_vals[:-1])

            body_thr = float(self._ob_body_atr) * atr_last

            bear_mask = (bear_body > body_thr) & vol_ok
            if np.any(bear_mask):
                idx = int(np.argmax(np.where(bear_mask, bear_body, -1e12)))
                if float(c_vals[-1]) > float(o_vals[idx]):
                    return "bull_ob"

            bull_mask = (bull_body > body_thr) & vol_ok
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

            if bull > bear and bull >= 3:
                return "bullish"
            if bear > bull and bear >= 3:
                return "bearish"
            return "none"
        except Exception as exc:
            logger.error("_detect_divergence error: %s | tb=%s", exc, traceback.format_exc())
            return "none"

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
            norm_factor = y_mean * 0.001  # 0.1% of price per bar
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
