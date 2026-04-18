"""
Bot/Motor/inference.py - CatBoost inference decoupling.

Ин модул интерфейси InferenceEngine ва EngineModelMixin-ро
фароҳам меорад, ки мантиқи пешгӯӣ бо CatBoost-ро идора мекунанд.
"""

from __future__ import annotations

import pickle
import time
import traceback
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

# =============================================================================
# Imports
# =============================================================================
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import talib

from Backtest.engine import run_backtest
from core.config import MAX_GATE_DRAWDOWN, MIN_GATE_SHARPE, MIN_GATE_WIN_RATE
from core.ml_router import MLSignal, validate_payload_schema
from core.model_engine import gate_details, model_manager
from log_config import get_artifact_path
from mt5_client import mt5_async_call

from .models import (
    AssetCandidate,
    _env_truthy,
    _partial_gate_enabled,
    _required_gate_assets,
    log_err,
    log_health,
    parse_bar_key,
    tf_seconds,
)
from .pipeline import UTCScheduler

if TYPE_CHECKING:
    from .engine import MultiAssetTradingEngine

# =============================================================================
# Global Constants
# =============================================================================
MODEL_STATE_PATH = get_artifact_path("models", "model_state.pkl")

# =============================================================================
# Classes / Mixins
# =============================================================================


class InferenceEngine:
    """Decoupled CatBoost inference service for the motor."""

    def __init__(self, engine: "MultiAssetTradingEngine") -> None:
        self._e = engine

    # SAFE IMPROVEMENT + DEFENSIVE CODE: private helper to eliminate dozens of duplicated
    # "float( ... or 0.0)" patterns and make every float conversion explicitly safe
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return default

    # SAFE IMPROVEMENT + DEFENSIVE CODE: private helper to eliminate duplicated int/float parsing
    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(InferenceEngine._safe_float(value, default))
        except (TypeError, ValueError, OverflowError):
            return default

    # SAFE IMPROVEMENT + PERFORMANCE + REMOVE DUPLICATED LOGIC:
    # Centralized talib indicator computation (was duplicated in infer_catboost + atr block)
    # Now one single call, no repeated np.asarray / talib calls / import statements
    @staticmethod
    def _compute_live_indicators(
        df: Optional[pd.DataFrame], last_close: float = 0.0
    ) -> Dict[str, float]:
        indicators: Dict[str, float] = {
            "rsi_14": 50.0,
            "ema_20": last_close,
            "ema_50": last_close,
            "ema_200": last_close,
            "atr_14": max(last_close * 0.001, 1e-6) if last_close > 0.0 else 1e-6,
        }
        if (
            df is None
            or not isinstance(df, pd.DataFrame)
            or len(df) < 200
            or "Close" not in df.columns
        ):
            return indicators

        try:
            c = np.asarray(df["Close"].values, dtype=np.float64)
            if len(c) < 200:
                return indicators

            rsi_arr = talib.RSI(c, timeperiod=14)
            ema20_arr = talib.EMA(c, timeperiod=20)
            ema50_arr = talib.EMA(c, timeperiod=50)
            ema200_arr = talib.EMA(c, timeperiod=200)

            if np.isfinite(rsi_arr[-1]):
                indicators["rsi_14"] = float(rsi_arr[-1])
            if np.isfinite(ema20_arr[-1]):
                indicators["ema_20"] = float(ema20_arr[-1])
            if np.isfinite(ema50_arr[-1]):
                indicators["ema_50"] = float(ema50_arr[-1])
            if np.isfinite(ema200_arr[-1]):
                indicators["ema_200"] = float(ema200_arr[-1])

            if "High" in df.columns and "Low" in df.columns and len(c) >= 14:
                h = np.asarray(df["High"].values, dtype=np.float64)
                low_arr = np.asarray(df["Low"].values, dtype=np.float64)
                atr_arr = talib.ATR(h, low_arr, c, timeperiod=14)
                if np.isfinite(atr_arr[-1]) and atr_arr[-1] > 0.0:
                    indicators["atr_14"] = float(atr_arr[-1])

            return indicators
        except Exception:
            # DEFENSIVE CODE: silent fallback exactly as original (no crash, no log spam)
            return indicators

    @staticmethod
    def _payload_trend_bias(payload: Optional[Dict[str, Any]], tf: str) -> float:
        if not isinstance(payload, dict):
            return 0.0
        frame = payload.get(tf)
        if not isinstance(frame, dict):
            return 0.0

        try:
            close = InferenceEngine._safe_float(frame.get("last_close"))
            ema20 = InferenceEngine._safe_float(frame.get("ema_20"))
            ema50 = InferenceEngine._safe_float(frame.get("ema_50"))
            ema200 = InferenceEngine._safe_float(frame.get("ema_200"))
            rsi = InferenceEngine._safe_float(frame.get("rsi_14"), 50.0)
        except Exception:
            return 0.0

        bull = 0.0
        bear = 0.0
        if close > ema20 > ema50 > ema200 > 0.0:
            bull += 0.75
        elif close < ema20 < ema50 < ema200 and ema200 > 0.0:
            bear += 0.75

        if rsi >= 57.5:
            bull += min(0.25, (rsi - 57.5) / 25.0)
        elif rsi <= 42.5:
            bear += min(0.25, (42.5 - rsi) / 25.0)

        return float(max(-1.0, min(1.0, bull - bear)))

    @staticmethod
    def _live_regime_bias(
        df,
        *,
        adx_min: float,
        adx_trend_min: float,
    ) -> Tuple[float, Dict[str, float]]:
        metrics = {
            "adx": 0.0,
            "plus_di": 0.0,
            "minus_di": 0.0,
            "slope": 0.0,
        }
        if df is None or len(df) < 210:
            return 0.0, metrics

        try:
            h = np.asarray(df["High"].values, dtype=np.float64)
            low_arr = np.asarray(df["Low"].values, dtype=np.float64)
            c = np.asarray(df["Close"].values, dtype=np.float64)
            if len(c) < 210:
                return 0.0, metrics

            ema20 = talib.EMA(c, timeperiod=20)
            ema50 = talib.EMA(c, timeperiod=50)
            ema200 = talib.EMA(c, timeperiod=200)
            adx = talib.ADX(h, low_arr, c, timeperiod=14)
            plus_di = talib.PLUS_DI(h, low_arr, c, timeperiod=14)
            minus_di = talib.MINUS_DI(h, low_arr, c, timeperiod=14)
            atr = talib.ATR(h, low_arr, c, timeperiod=14)

            last_close = float(c[-1])
            last_ema20 = float(ema20[-1]) if np.isfinite(ema20[-1]) else 0.0
            last_ema50 = float(ema50[-1]) if np.isfinite(ema50[-1]) else 0.0
            last_ema200 = float(ema200[-1]) if np.isfinite(ema200[-1]) else 0.0
            last_adx = float(adx[-1]) if np.isfinite(adx[-1]) else 0.0
            last_plus_di = float(plus_di[-1]) if np.isfinite(plus_di[-1]) else 0.0
            last_minus_di = float(minus_di[-1]) if np.isfinite(minus_di[-1]) else 0.0
            last_atr = (
                float(atr[-1])
                if np.isfinite(atr[-1]) and atr[-1] > 0.0
                else max(last_close * 0.001, 1e-6)
            )

            metrics.update(
                {
                    "adx": last_adx,
                    "plus_di": last_plus_di,
                    "minus_di": last_minus_di,
                }
            )

            bull = 0.0
            bear = 0.0

            if last_close > last_ema20 > last_ema50 > last_ema200 > 0.0:
                bull += 0.55
            elif (
                last_close < last_ema20 < last_ema50 < last_ema200 and last_ema200 > 0.0
            ):
                bear += 0.55

            di_gap = abs(last_plus_di - last_minus_di)
            if last_plus_di > last_minus_di:
                bull += min(0.25, di_gap / 50.0)
            elif last_minus_di > last_plus_di:
                bear += min(0.25, di_gap / 50.0)

            slope_window = min(20, len(c))
            if slope_window >= 8:
                y = c[-slope_window:]
                x = np.arange(slope_window, dtype=np.float64)
                slope = float(np.polyfit(x, y, 1)[0])
                slope_norm = float(
                    np.clip((slope * slope_window) / max(last_atr, 1e-6), -3.0, 3.0)
                    / 3.0
                )
                metrics["slope"] = slope_norm
                if slope_norm > 0.0:
                    bull += 0.20 * abs(slope_norm)
                elif slope_norm < 0.0:
                    bear += 0.20 * abs(slope_norm)

            raw_bias = bull - bear
            if last_adx >= adx_trend_min:
                strength = 1.0
            elif last_adx >= adx_min:
                strength = 0.7
            else:
                strength = 0.35

            return float(max(-1.0, min(1.0, raw_bias * strength))), metrics
        except Exception:
            return 0.0, metrics

    def infer_catboost(
        self,
        asset: str,
        payloads: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    ) -> Optional[MLSignal]:
        """Run CatBoost model inference on live M1 data from MT5."""
        e = self._e
        if e._is_asset_blocked(asset):
            e._log_blocked_asset_skip(asset, "catboost_infer")
            return None
        payload = e._catboost_payloads.get(asset)
        if not payload:
            return None
        cache = getattr(e, "_catboost_signal_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(e, "_catboost_signal_cache", cache)
        payload_frame = self.extract_payload_frame(
            (payloads or {}).get("scalp") if isinstance(payloads, dict) else None
        )
        if not payload_frame:
            payload_frame = self.extract_payload_frame(
                (payloads or {}).get("intraday") if isinstance(payloads, dict) else None
            )
        # SAFE IMPROVEMENT: use private helper (was duplicated try/except)
        payload_ts_bar = self._safe_int(payload_frame.get("ts_bar"))

        cached = cache.get(asset)
        if (
            payload_ts_bar > 0
            and isinstance(cached, dict)
            and self._safe_int(cached.get("ts_bar")) == payload_ts_bar
            and isinstance(cached.get("signal"), MLSignal)
        ):
            return cached.get("signal")

        try:
            e._touch_runtime_progress()

            cb_model = payload["model"]
            pipeline = payload["pipeline"]

            pipe = e._xau if asset == "XAU" else e._btc
            if pipe is None:
                return None
            symbol = pipe.symbol
            if not symbol:
                return None

            n_bars = max(int(getattr(pipeline.cfg, "window_size", 60) or 60) + 200, 500)
            req_bars_fn = getattr(pipeline, "required_live_bars", None)
            if callable(req_bars_fn):
                try:
                    n_bars = max(n_bars, int(req_bars_fn()))
                except Exception:
                    pass
            rates = mt5_async_call(
                "copy_rates_from_pos",
                symbol,
                mt5.TIMEFRAME_M1,
                1,
                n_bars,
                timeout=2.0,
                default=None,
            )
            if rates is None or len(rates) < 100:
                log_health.warning(
                    "CATBOOST_SKIP | asset=%s reason=insufficient_bars bars=%s",
                    asset,
                    len(rates) if rates is not None else 0,
                )
                return None
            e._touch_runtime_progress()
            if len(rates) < n_bars:
                log_health.warning(
                    "CATBOOST_HISTORY_SHORT | asset=%s bars=%s requested=%s",
                    asset,
                    len(rates),
                    n_bars,
                )

            df = pd.DataFrame(rates)
            df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "tick_volume": "Volume",
                }
            )
            df = df.set_index("datetime")[["Open", "High", "Low", "Close", "Volume"]]

            if hasattr(pipeline, "transform_live"):
                X_series = pipeline.transform_live(df.copy())
            else:
                xy = pipeline.transform(df.copy())
                X_series = xy["X"]
            if X_series.empty:
                log_health.warning(
                    "CATBOOST_SKIP | asset=%s reason=empty_features", asset
                )
                return None
            e._touch_runtime_progress()

            X_last = np.stack([X_series.iloc[-1]])
            pred = cb_model.predict(X_last)
            pred_val = float(pred[0])
            e._touch_runtime_progress()

            static_threshold = max(
                self._safe_float(getattr(pipeline.cfg, "percent_increase", 0.0)), 1e-12
            )
            model_threshold = 0.0
            model_quantile = 0.0
            try:
                alpha_cal = (
                    payload.get("alpha_calibration", {})
                    if isinstance(payload, dict)
                    else {}
                )
                model_threshold = self._safe_float(alpha_cal.get("pred_abs_threshold"))
                model_quantile = self._safe_float(alpha_cal.get("pred_quantile"))
                if not np.isfinite(model_threshold) or model_threshold < 0.0:
                    model_threshold = 0.0
                if (
                    not np.isfinite(model_quantile)
                    or model_quantile < 0.0
                    or model_quantile > 99.9
                ):
                    model_quantile = 0.0
            except Exception:
                model_threshold = 0.0
                model_quantile = 0.0

            if asset not in e._catboost_pred_history:
                e._catboost_pred_history[asset] = deque(maxlen=200)
            e._catboost_pred_history[asset].append(abs(pred_val))

            now_ts = time.time()
            density_mult, density_min_zscore, density_min_flow, sig_24h = (
                e._signal_density_controls(asset, now_ts)
            )
            e._prune_signal_window(e._signal_emit_ts_global, now_ts)
            sig_24h_global = int(len(e._signal_emit_ts_global))

            hist = e._catboost_pred_history[asset]
            hist_arr = np.asarray(hist, dtype=np.float64)
            hist_q = (
                float(np.percentile(hist_arr, float(e._catboost_threshold_q)))
                if len(hist_arr) > 0
                else 0.0
            )
            model_q_thr = (
                float(np.percentile(hist_arr, model_quantile))
                if len(hist_arr) > 0 and model_quantile > 0.0
                else 0.0
            )
            if len(hist) >= int(e._catboost_hist_min):
                threshold = max(
                    static_threshold, model_threshold, model_q_thr, hist_q, 1e-12
                ) * float(density_mult)
                hist_mu = float(np.mean(hist_arr))
                hist_sigma = float(np.std(hist_arr))
                z_mag = (
                    (abs(pred_val) - hist_mu) / hist_sigma
                    if hist_sigma > 1e-12
                    else 0.0
                )
                p95_mag = max(float(np.percentile(hist_arr, 95)), threshold * 1.10)
                conf_ref = max(p95_mag - threshold, threshold * 0.15, 1e-12)
            else:
                threshold = max(static_threshold, model_threshold, 1e-12) * float(
                    density_mult
                )
                z_mag = 0.0
                p95_mag = threshold * 2.0
                conf_ref = max(threshold, 1e-12)

            mag = abs(pred_val)
            # ─── CALIBRATED PROBABILITY (Section 3) ─────────────────────
            # Prefer a calibrated P(direction_correct | |pred|) built on the
            # validation fold at training time. Falls back to the heuristic
            # mag/threshold confidence only if the calibrator is missing or
            # malformed — in that case `KELLY_DISABLED=1` (default) in the
            # risk layer ensures fixed-risk sizing.
            _calibrator = None
            try:
                _calibrator = alpha_cal.get("probability_calibrator") if isinstance(alpha_cal, dict) else None
            except Exception:
                _calibrator = None
            _calibrated_p: Optional[float] = None
            try:
                from core.utils import calibrated_probability as _calib_fn

                _calibrated_p = _calib_fn(pred_val, _calibrator)
            except Exception:
                _calibrated_p = None
            if _calibrated_p is not None:
                # Calibrated probability IS the confidence. We clamp to
                # [0.0, 0.99] to preserve downstream expectations (never
                # the perfect-confidence singular value).
                confidence = max(0.0, min(0.99, float(_calibrated_p)))
            else:
                # Legacy heuristic — only active when calibrator missing.
                if mag < threshold:
                    confidence = max(
                        0.05, min(0.74, (mag / max(threshold, 1e-12)) * 0.74)
                    )
                else:
                    rel = (mag - threshold) / conf_ref
                    rel = max(0.0, min(1.0, rel))
                    confidence = 0.75 + (0.24 * rel)
                confidence = max(0.0, min(0.99, confidence))

            last_close = InferenceEngine._safe_float(df["Close"].iloc[-1])

            # SAFE IMPROVEMENT + REMOVE DUPLICATED LOGIC + DEFENSIVE CODE:
            # Replaced two separate talib blocks (atr + live indicators) with single helper
            live_indicators = self._compute_live_indicators(df, last_close)
            atr_val = live_indicators["atr_14"]
            live_rsi = live_indicators["rsi_14"]
            live_ema20 = live_indicators["ema_20"]
            live_ema50 = live_indicators["ema_50"]
            live_ema200 = live_indicators["ema_200"]

            flow_confirm = 0.0
            try:
                feed = getattr(pipe, "feed", None)
                tick_stats_fn = getattr(feed, "tick_stats", None)
                if callable(tick_stats_fn):
                    df_tick = df.rename(
                        columns={
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "tick_volume",
                        }
                    )
                    ts = tick_stats_fn(df_tick.tail(300))
                    t_imb = self._safe_float(getattr(ts, "imbalance", 0.0))
                    t_delta = self._safe_float(getattr(ts, "tick_delta", 0.0))
                    flow_confirm = max(-1.0, min(1.0, 0.55 * t_imb + 0.45 * t_delta))
            except Exception:
                flow_confirm = 0.0
            e._touch_runtime_progress()

            if abs(pred_val) < threshold:
                out = MLSignal(
                    asset=asset,
                    signal="HOLD",
                    side="Neutral",
                    confidence=confidence,
                    reason=f"catboost_below_threshold:{pred_val:.6f}<{threshold:.6f}",
                    provider="catboost",
                    model="catboost_trained",
                    entry=None,
                    stop_loss=None,
                    take_profit=None,
                    scalp_payload={
                        "M1": {
                            "ts_bar": float(df.index[-1].timestamp()),
                            "last_close": last_close,
                            "atr_14": atr_val,
                            "rsi_14": live_rsi,
                            "ema_20": live_ema20,
                            "ema_50": live_ema50,
                            "ema_200": live_ema200,
                        }
                    },
                    intraday_payload=None,
                )
                if payload_ts_bar > 0:
                    cache[asset] = {"ts_bar": payload_ts_bar, "signal": out}
                return out
            if len(hist) >= int(e._catboost_hist_min) and z_mag < float(
                density_min_zscore
            ):
                out = MLSignal(
                    asset=asset,
                    signal="HOLD",
                    side="Neutral",
                    confidence=max(0.05, min(confidence, 0.70)),
                    reason=f"catboost_low_zscore:{z_mag:.2f}<{float(density_min_zscore):.2f}",
                    provider="catboost",
                    model="catboost_trained",
                    entry=None,
                    stop_loss=None,
                    take_profit=None,
                    scalp_payload={
                        "M1": {
                            "ts_bar": float(df.index[-1].timestamp()),
                            "last_close": last_close,
                            "atr_14": atr_val,
                            "rsi_14": live_rsi,
                            "ema_20": live_ema20,
                            "ema_50": live_ema50,
                            "ema_200": live_ema200,
                        }
                    },
                    intraday_payload=None,
                )
                if payload_ts_bar > 0:
                    cache[asset] = {"ts_bar": payload_ts_bar, "signal": out}
                return out
            if (pred_val * flow_confirm) < -float(density_min_flow):
                out = MLSignal(
                    asset=asset,
                    signal="HOLD",
                    side="Neutral",
                    confidence=max(0.05, min(confidence, 0.72)),
                    reason=f"catboost_flow_conflict:pred={pred_val:.6f}|flow={flow_confirm:+.3f}",
                    provider="catboost",
                    model="catboost_trained",
                    entry=None,
                    stop_loss=None,
                    take_profit=None,
                    scalp_payload={
                        "M1": {
                            "ts_bar": float(df.index[-1].timestamp()),
                            "last_close": last_close,
                            "atr_14": atr_val,
                            "rsi_14": live_rsi,
                            "ema_20": live_ema20,
                            "ema_50": live_ema50,
                            "ema_200": live_ema200,
                        }
                    },
                    intraday_payload=None,
                )
                if payload_ts_bar > 0:
                    cache[asset] = {"ts_bar": payload_ts_bar, "signal": out}
                return out

            scalp_payload = (
                (payloads or {}).get("scalp") if isinstance(payloads, dict) else None
            )
            intraday_payload = (
                (payloads or {}).get("intraday") if isinstance(payloads, dict) else None
            )
            m1_bias, regime_metrics = self._live_regime_bias(
                df,
                adx_min=self._safe_float(getattr(pipe.cfg, "adx_min", 20.0), 20.0),
                adx_trend_min=self._safe_float(
                    getattr(pipe.cfg, "adx_trend_min", 25.0), 25.0
                ),
            )
            h1_bias = self._payload_trend_bias(intraday_payload, "H1")
            pred_sign = 1.0 if pred_val > 0.0 else -1.0
            combined_bias = float(
                max(-1.0, min(1.0, (0.70 * m1_bias) + (0.30 * h1_bias)))
            )
            pred_margin = abs(pred_val) / max(threshold, 1e-12)
            strong_conflict = (
                (pred_sign * combined_bias) < 0.0
                and abs(combined_bias) >= 0.35
                and (
                    float(regime_metrics.get("adx", 0.0) or 0.0)
                    >= float(getattr(pipe.cfg, "adx_trend_min", 25.0) or 25.0)
                    or abs(h1_bias) >= 0.55
                    or pred_margin < 1.35
                )
            )
            if strong_conflict:
                out = MLSignal(
                    asset=asset,
                    signal="HOLD",
                    side="Neutral",
                    confidence=max(0.05, min(confidence, 0.74)),
                    reason=(
                        f"catboost_regime_conflict:pred={pred_val:.6f}|margin={pred_margin:.3f}|"
                        f"m1={m1_bias:+.3f}|h1={h1_bias:+.3f}|combo={combined_bias:+.3f}|"
                        f"adx={float(regime_metrics.get('adx', 0.0) or 0.0):.2f}|"
                        f"plus_di={float(regime_metrics.get('plus_di', 0.0) or 0.0):.2f}|"
                        f"minus_di={float(regime_metrics.get('minus_di', 0.0) or 0.0):.2f}"
                    ),
                    provider="catboost",
                    model="catboost_trained",
                    entry=None,
                    stop_loss=None,
                    take_profit=None,
                    scalp_payload=scalp_payload,
                    intraday_payload=intraday_payload,
                )
                if payload_ts_bar > 0:
                    cache[asset] = {"ts_bar": payload_ts_bar, "signal": out}
                return out

            signal = "STRONG BUY" if pred_val > 0 else "STRONG SELL"
            side = "Buy" if pred_val > 0 else "Sell"
            frame = {
                "ts_bar": float(df.index[-1].timestamp()),
                "last_close": last_close,
                "atr_14": atr_val,
                "rsi_14": live_rsi,
                "ema_20": live_ema20,
                "ema_50": live_ema50,
                "ema_200": live_ema200,
            }

            out = MLSignal(
                asset=asset,
                signal=signal,
                side=side,
                confidence=confidence,
                reason=(
                    f"catboost_pred:{pred_val:.6f}|thr:{threshold:.6f}|"
                    f"base_thr:{static_threshold:.6f}|model_thr:{model_threshold:.6f}|"
                    f"model_q:{model_quantile:.1f}|model_q_thr:{model_q_thr:.6f}|p95:{p95_mag:.6f}|"
                    f"sig24_asset:{sig_24h}|sig24_global:{sig_24h_global}|dens_mult:{density_mult:.3f}"
                ),
                provider="catboost",
                model="catboost_trained",
                entry=last_close,
                stop_loss=None,
                take_profit=None,
                scalp_payload=scalp_payload or {"M1": frame},
                intraday_payload=intraday_payload or {"H1": frame},
            )
            if payload_ts_bar > 0:
                cache[asset] = {"ts_bar": payload_ts_bar, "signal": out}
            return out
        except Exception as exc:
            log_err.error(
                "CATBOOST_INFER_ERROR | asset=%s err=%s | tb=%s",
                asset,
                exc,
                traceback.format_exc(),
            )
            return None

    def validate_payload_timestamp(
        self, asset: str, payload: Dict[str, Any], tf: str
    ) -> Tuple[bool, str]:
        if not isinstance(payload, dict):
            return False, "payload_not_dict"

        tf_obj = payload.get(tf)
        if not isinstance(tf_obj, dict):
            return False, f"tf_missing:{tf}"

        raw_ts = tf_obj.get("ts_bar")
        ts_value = 0.0
        if isinstance(raw_ts, (int, float)):
            ts_value = self._safe_float(raw_ts)
        elif isinstance(raw_ts, str):
            if "_" in raw_ts:
                return False, f"ts_format_mismatch:{raw_ts}"
            parsed = parse_bar_key(raw_ts)
            if parsed is None:
                return False, f"ts_parse_failed:{raw_ts}"
            ts_value = float(parsed.timestamp())
        else:
            return False, "ts_type_invalid"

        if ts_value < 1_000_000_000.0:
            return False, f"ts_epoch_invalid:{ts_value}"

        raw_t_close = tf_obj.get("t_close")
        if raw_t_close is not None:
            if isinstance(raw_t_close, (int, float)):
                t_close_value = self._safe_float(raw_t_close)
            elif isinstance(raw_t_close, str):
                if "_" in raw_t_close:
                    return False, f"t_close_format_mismatch:{raw_t_close}"
                parsed_close = parse_bar_key(raw_t_close)
                if parsed_close is None:
                    return False, f"t_close_parse_failed:{raw_t_close}"
                t_close_value = float(parsed_close.timestamp())
            else:
                return False, "t_close_type_invalid"

            bar_sec = self._safe_float(tf_seconds(tf), 60.0)
            tolerance = max(5.0, bar_sec * 2.0)
            if abs(ts_value - t_close_value) > tolerance:
                return (
                    False,
                    f"t_close_mismatch:{abs(ts_value - t_close_value):.1f}s>{tolerance:.1f}s",
                )

        age = time.time() - ts_value
        threshold = 180.0 if tf.startswith("M") else 21600.0
        if age < -15.0:
            return False, f"ts_in_future:{age:.1f}s"
        if age > threshold:
            return False, f"data_gap:{age:.1f}s>{threshold:.1f}s"
        return True, "ok"

    def validate_data_sync_payloads(
        self,
        asset: str,
        payloads: Dict[str, Optional[Dict[str, Any]]],
    ) -> Tuple[bool, str]:
        scalp = payloads.get("scalp")
        intraday = payloads.get("intraday")
        if not scalp and not intraday:
            return False, "no_payloads"

        if scalp:
            ok, reason = self.validate_payload_timestamp(asset, scalp, "M1")
            if not ok:
                return False, f"scalp_{reason}"

        if intraday:
            ok, reason = self.validate_payload_timestamp(asset, intraday, "H1")
            if not ok:
                return False, f"intraday_{reason}"

        schema_ok, schema_reason = validate_payload_schema(asset, payloads)
        if not schema_ok:
            return False, f"schema_{schema_reason}"

        return True, "ok"

    @staticmethod
    def extract_payload_frame(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        if isinstance(payload.get("M1"), dict):
            return dict(payload.get("M1") or {})
        if isinstance(payload.get("H1"), dict):
            return dict(payload.get("H1") or {})
        return {}

    def probabilistic_levels(
        self,
        *,
        side: str,
        entry: float,
        base_sl: float,
        base_tp: float,
        confidence: float,
        frame: Dict[str, Any],
    ) -> Tuple[float, float]:
        c = max(0.0, min(1.0, self._safe_float(confidence)))
        atr = self._safe_float(frame.get("atr_14"))
        if atr <= 0.0 and entry > 0.0:
            atr = max(0.00001, entry * 0.001)

        levels: list[float] = []
        for k in ("ema_20", "ema_50", "ema_200"):
            try:
                v = self._safe_float(frame.get(k))
                if v > 0.0:
                    levels.append(v)
            except Exception:
                pass

        if not levels:
            return float(base_sl), float(base_tp)

        support_cluster = min(levels)
        resistance_cluster = max(levels)
        variance = max(0.10, 1.0 - c)

        if side == "Buy":
            sl_cluster = support_cluster - atr * (0.35 + variance)
            tp_cluster = resistance_cluster + atr * (1.20 + c)
            sl = min(float(base_sl), sl_cluster) if sl_cluster > 0.0 else float(base_sl)
            tp = max(float(base_tp), tp_cluster)
            if not (sl < entry < tp):
                sl = min(float(base_sl), entry - atr * (1.20 + variance))
                tp = max(float(base_tp), entry + atr * (1.30 + c))
            return sl, tp

        sl_cluster = resistance_cluster + atr * (0.35 + variance)
        tp_cluster = support_cluster - atr * (1.20 + c)
        sl = max(float(base_sl), sl_cluster)
        tp = min(float(base_tp), tp_cluster)
        if not (tp < entry < sl):
            sl = max(float(base_sl), entry + atr * (1.20 + variance))
            tp = min(float(base_tp), entry - atr * (1.30 + c))
        return sl, tp

    @staticmethod
    def _frame_bar_key(frame: Dict[str, Any]) -> str:
        raw = frame.get("ts_bar")
        if raw is None:
            raw = frame.get("t_close")
        if isinstance(raw, (int, float)):
            try:
                return str(int(InferenceEngine._safe_float(raw)))
            except Exception:
                return str(raw)
        if isinstance(raw, str):
            s = raw.strip()
            if s:
                parsed = parse_bar_key(s)
                if parsed is not None:
                    return str(int(parsed.timestamp()))
                return s.replace(" ", "_")
        return str(int(time.time()))

    def _build_ml_bridge_candidate(
        self,
        asset: str,
        sig: MLSignal,
        pipe: Any,
        inst_candidate: AssetCandidate,
    ) -> Optional[AssetCandidate]:
        e = self._e
        cfg = e._xau_cfg if asset == "XAU" else e._btc_cfg
        if not bool(getattr(cfg, "ml_bridge_enabled", True)):
            return None

        pipeline_signal = str(getattr(inst_candidate, "signal", "") or "").strip()
        allow_neutral_pipeline = bool(
            getattr(cfg, "ml_bridge_allow_neutral_pipeline", False)
        )
        if pipeline_signal not in ("Buy", "Sell") and not allow_neutral_pipeline:
            log_health.info(
                "FSM_ML_BRIDGE_BLOCK | asset=%s ml_signal=%s pipeline_signal=%s reason=indicator_veto",
                asset,
                sig.signal,
                pipeline_signal or "Neutral",
            )
            return None

        sig_name = str(sig.signal or "").upper().strip()
        if sig_name not in ("STRONG BUY", "STRONG SELL"):
            return None

        conf = max(0.0, min(1.0, self._safe_float(sig.confidence)))
        min_conf = max(
            0.0,
            min(1.0, self._safe_float(getattr(cfg, "ml_bridge_min_confidence", 0.80))),
        )
        if conf < min_conf:
            return None

        frame = self.extract_payload_frame(sig.scalp_payload)
        if not frame:
            frame = self.extract_payload_frame(sig.intraday_payload)

        entry = self._safe_float(
            sig.entry
            or frame.get("last_close")
            or getattr(pipe, "last_market_close", 0.0)
        )
        if entry <= 0.0:
            log_health.info(
                "FSM_ML_BRIDGE_BLOCK | asset=%s ml_signal=%s reason=no_entry_price",
                asset,
                sig.signal,
            )
            return None

        try:
            open_positions = self._safe_int(e._asset_open_positions(asset))
        except Exception:
            open_positions = 0
        try:
            max_positions = self._safe_int(e._asset_max_positions(asset))
        except Exception:
            max_positions = 0

        adapt = {
            "regime": "ml_router",
            "confidence": conf,
            "atr": self._safe_float(frame.get("atr_14")),
            "atr_pct": self._safe_float(frame.get("atr_pct")),
            "ker": self._safe_float(frame.get("ker")),
            "rvi": self._safe_float(frame.get("rvi")),
            "stop_hunt_strength": self._safe_float(frame.get("stop_hunt_strength")),
            "ob_touch_proximity": self._safe_float(
                frame.get("ob_touch_proximity", frame.get("ob_pretouch_bias", 0.0))
            ),
        }

        try:
            plan = pipe.risk.plan_order(
                side=str(sig.side),
                confidence=conf,
                ind=dict(frame),
                adapt=adapt,
                entry=entry,
                open_positions=open_positions,
                max_positions=max_positions,
            )
        except Exception as exc:
            log_err.error("FSM_ML_BRIDGE_ERROR | asset=%s err=%s", asset, exc)
            return None

        if bool(plan.get("blocked", True)):
            log_health.info(
                "FSM_ML_BRIDGE_BLOCK | asset=%s ml_signal=%s reason=%s",
                asset,
                sig.signal,
                plan.get("reason", "plan_blocked"),
            )
            return None

        lot = e._apply_asset_lot_cap(
            asset, self._safe_float(plan.get("lot")), "ml_bridge"
        )
        if lot <= 0.0:
            return None

        base_reasons = tuple(str(r) for r in (inst_candidate.reasons or ()))
        bridge_reasons = (
            "ml_bridge:neutral_pipeline",
            f"ml_provider:{sig.provider}",
            f"ml_model:{sig.model}",
            f"ml_reason:{sig.reason}",
            "sniper_fsm",
        )
        signal_id = f"MLBRIDGE_{asset}_{self._frame_bar_key(frame)}_{sig.side}"
        latency_ms = max(
            self._safe_float(getattr(inst_candidate, "latency_ms", 0.0)), 0.0
        )

        log_health.info(
            "FSM_ML_BRIDGE | asset=%s signal=%s conf=%.3f lot=%.4f signal_id=%s",
            asset,
            sig.side,
            conf,
            lot,
            signal_id,
        )

        return AssetCandidate(
            asset=asset,
            symbol=str(getattr(inst_candidate, "symbol", "") or pipe.symbol),
            signal=str(sig.side),
            confidence=conf,
            lot=lot,
            sl=self._safe_float(plan.get("sl")),
            tp=self._safe_float(plan.get("tp")),
            latency_ms=latency_ms,
            blocked=False,
            reasons=tuple(list(base_reasons[:12]) + list(bridge_reasons)),
            signal_id=signal_id,
            raw_result={
                "institutional_candidate": inst_candidate.raw_result,
                "ml_signal": sig.signal,
                "provider": sig.provider,
                "model": sig.model,
                "reason": sig.reason,
                "confidence": sig.confidence,
                "bridge_plan": dict(plan),
            },
        )

    @staticmethod
    def _bridge_soft_block_reasons(reasons: Any) -> bool:
        reason_list = tuple(str(r or "").strip() for r in (reasons or ()))
        if not reason_list:
            return False
        soft_prefixes = (
            "baseline_warmup",
            "baseline_sync(",
        )
        return all(
            any(reason.startswith(prefix) for prefix in soft_prefixes)
            for reason in reason_list
        )

    def build_ml_candidate(self, asset: str, sig: MLSignal) -> Optional[AssetCandidate]:
        e = self._e
        if e._is_asset_blocked(asset):
            e._log_blocked_asset_skip(asset, "build_candidate")
            return None
        if sig.signal == "HOLD" or sig.side not in ("Buy", "Sell"):
            return None

        pipe = e._xau if asset == "XAU" else e._btc
        if pipe is None or pipe.risk is None:
            return None
        compute_candidate_fn = getattr(pipe, "compute_candidate", None)
        if not callable(compute_candidate_fn):
            return None

        try:
            inst_candidate = compute_candidate_fn()
        except Exception as exc:
            log_err.error("FSM_PIPELINE_CANDIDATE_ERROR | asset=%s err=%s", asset, exc)
            return None

        cfg = e._xau_cfg if asset == "XAU" else e._btc_cfg
        allow_neutral_pipeline = bool(
            getattr(cfg, "ml_bridge_allow_neutral_pipeline", False)
        )

        if inst_candidate is None:
            log_health.info(
                "FSM_PIPELINE_BLOCK | asset=%s ml_signal=%s reason=pipeline_candidate_none",
                asset,
                sig.signal,
            )
            return None

        if bool(inst_candidate.blocked):
            if allow_neutral_pipeline and self._bridge_soft_block_reasons(
                inst_candidate.reasons
            ):
                bridge_candidate = self._build_ml_bridge_candidate(
                    asset, sig, pipe, inst_candidate
                )
                if bridge_candidate is not None:
                    log_health.info(
                        "FSM_ML_BRIDGE_SOFT_BLOCK | asset=%s ml_signal=%s pipeline_signal=%s reasons=%s",
                        asset,
                        sig.signal,
                        inst_candidate.signal,
                        ",".join(
                            tuple(str(r) for r in (inst_candidate.reasons or ()))
                        )
                        or "-",
                    )
                    return bridge_candidate
            log_health.info(
                "FSM_PIPELINE_BLOCK | asset=%s ml_signal=%s pipeline_signal=%s reasons=%s",
                asset,
                sig.signal,
                inst_candidate.signal,
                ",".join(tuple(str(r) for r in (inst_candidate.reasons or ()))) or "-",
            )
            return None

        if str(inst_candidate.signal) not in ("Buy", "Sell"):
            if allow_neutral_pipeline:
                bridge_candidate = self._build_ml_bridge_candidate(
                    asset, sig, pipe, inst_candidate
                )
                if bridge_candidate is not None:
                    return bridge_candidate
            log_health.info(
                "FSM_PIPELINE_NEUTRAL | asset=%s ml_signal=%s pipeline_signal=%s reasons=%s",
                asset,
                sig.signal,
                inst_candidate.signal,
                ",".join(tuple(str(r) for r in (inst_candidate.reasons or ()))) or "-",
            )
            return None

        if str(inst_candidate.signal) != str(sig.side):
            log_health.info(
                "FSM_SIGNAL_CONFLICT | asset=%s ml=%s pipeline=%s ml_reason=%s pipeline_reasons=%s",
                asset,
                sig.side,
                inst_candidate.signal,
                sig.reason,
                ",".join(tuple(str(r) for r in (inst_candidate.reasons or ()))) or "-",
            )
            return None

        lot = e._apply_asset_lot_cap(
            asset, self._safe_float(inst_candidate.lot), "build"
        )
        if lot <= 0.0:
            return None

        base_reasons = tuple(str(r) for r in (inst_candidate.reasons or ()))
        ml_reasons = (
            f"ml_provider:{sig.provider}",
            f"ml_model:{sig.model}",
            f"ml_reason:{sig.reason}",
            "sniper_fsm",
        )
        reasons = tuple(list(base_reasons[:12]) + list(ml_reasons))
        confidence = max(
            0.0,
            min(
                1.0,
                min(
                    self._safe_float(inst_candidate.confidence),
                    self._safe_float(sig.confidence),
                ),
            ),
        )

        return AssetCandidate(
            asset=asset,
            symbol=str(inst_candidate.symbol or pipe.symbol),
            signal=str(inst_candidate.signal),
            confidence=confidence,
            lot=lot,
            sl=self._safe_float(inst_candidate.sl),
            tp=self._safe_float(inst_candidate.tp),
            latency_ms=self._safe_float(inst_candidate.latency_ms),
            blocked=False,
            reasons=reasons,
            signal_id=str(
                inst_candidate.signal_id or f"ML_{asset}_{int(time.time())}_{sig.side}"
            ),
            raw_result={
                "institutional_candidate": inst_candidate.raw_result,
                "ml_signal": sig.signal,
                "provider": sig.provider,
                "model": sig.model,
                "reason": sig.reason,
                "confidence": sig.confidence,
            },
        )


# --- Engine mixin extracted from engine.py --------------------------------


class EngineModelMixin:
    def _load_model_state(self) -> Optional[Dict[str, Any]]:
        """Load the exact Backtest artifact required by live engine."""
        try:
            if not MODEL_STATE_PATH.exists():
                return None
            with open(MODEL_STATE_PATH, "rb") as f:
                state = pickle.load(f)
            if not isinstance(state, dict):
                return None
            return state
        except Exception as exc:
            log_err.error(
                "MODEL_STATE_LOAD_ERROR | path=%s err=%s", MODEL_STATE_PATH, exc
            )
            return None

    @staticmethod
    def _default_backtest_asset() -> str:
        try:
            return "BTC" if UTCScheduler.is_weekend() else "XAU"
        except Exception:
            return "XAU"

    def _autobuild_model_state(self, reason: str) -> bool:
        """
        Self-heal bridge artifact when model_state is missing.
        Runs unified backtest once and re-checks the artifact.
        """
        asset = self._default_backtest_asset()
        try:
            log_health.warning(
                "MODEL_GATE_AUTOFIX_START | reason=%s asset=%s", reason, asset
            )
            run_backtest(asset)
            state = self._load_model_state()
            if state is None:
                log_err.error(
                    "MODEL_GATE_AUTOFIX_FAIL | reason=%s asset=%s detail=state_still_missing",
                    reason,
                    asset,
                )
                return False
            log_health.info(
                "MODEL_GATE_AUTOFIX_OK | asset=%s path=%s", asset, MODEL_STATE_PATH
            )
            return True
        except Exception as exc:
            log_err.error(
                "MODEL_GATE_AUTOFIX_FAIL | reason=%s asset=%s err=%s | tb=%s",
                reason,
                asset,
                exc,
                traceback.format_exc(),
            )
            return False

    def _check_model_health(self) -> None:
        """Ensure live trading starts only when gate-approved assets are available."""
        if self.dry_run:
            log_health.info("MODEL_GATE_BYPASSED | reason=dry_run")
            self._catboost_payloads = {}
            versions: list[str] = []
            try:
                details = gate_details(
                    required_assets=_required_gate_assets(),
                    allow_legacy_fallback=True,
                )
                assets = details.get("assets", {}) if isinstance(details, dict) else {}
                if isinstance(assets, dict):
                    for asset, st in sorted(assets.items(), key=lambda kv: str(kv[0])):
                        if not isinstance(st, dict) or not bool(st.get("ok", False)):
                            continue
                        version = str(st.get("model_version", "") or "").strip()
                        if not version:
                            continue
                        model = model_manager.load_model(version)
                        if (
                            isinstance(model, dict)
                            and "model" in model
                            and "pipeline" in model
                        ):
                            asset_key = str(asset).upper().strip()
                            self._catboost_payloads[asset_key] = model
                            versions.append(f"{asset_key}:{version}")
                            log_health.info(
                                "CATBOOST_LOADED | asset=%s version=%s mode=dry_run",
                                asset_key,
                                version,
                            )
            except Exception as exc:
                log_health.warning("DRY_RUN_MODEL_PREFLIGHT_SKIP | err=%s", exc)
            self._model_loaded = True
            self._backtest_passed = True
            self._model_version = "dry_run" if not versions else " | ".join(versions)
            self._gate_last_reason = "dry_run_bypass"
            self._blocked_assets = []
            return

        # Reset gate-scoped payloads each time to avoid stale models on partial reloads.
        self._catboost_payloads = {}
        self._blocked_assets = []

        required_assets = _required_gate_assets()
        details = gate_details(
            required_assets=required_assets, allow_legacy_fallback=True
        )
        gate_ok = bool(details.get("ok", False))
        gate_reason = str(details.get("reason", "unknown"))
        self._gate_last_reason = gate_reason
        assets = details.get("assets", {}) if isinstance(details, dict) else {}
        active_assets = list(required_assets)

        gate_items = []
        if isinstance(assets, dict):
            for asset, st in sorted(assets.items(), key=lambda kv: str(kv[0])):
                try:
                    gate_items.append(
                        (
                            str(asset),
                            bool(st.get("ok", False)),
                            str(st.get("reason", "unknown")),
                            str(st.get("model_version", "")),
                            InferenceEngine._safe_float(st.get("sharpe")),
                            InferenceEngine._safe_float(st.get("win_rate")),
                            InferenceEngine._safe_float(st.get("max_drawdown_pct")),
                            bool(st.get("legacy_fallback", False)),
                        )
                    )
                except Exception:
                    continue

        def _fallback_metrics_ok(st: Dict[str, Any]) -> bool:
            if not isinstance(st, dict):
                return False
            if not bool(st.get("real_backtest", False)):
                return False
            version = str(st.get("model_version", "") or "").strip()
            if not version:
                return False
            sharpe = InferenceEngine._safe_float(st.get("sharpe"))
            win_rate = InferenceEngine._safe_float(st.get("win_rate"))
            max_dd_raw = st.get("max_drawdown_pct", 1.0)
            max_dd = InferenceEngine._safe_float(max_dd_raw, 1.0)
            return bool(
                sharpe >= MIN_GATE_SHARPE
                and win_rate >= MIN_GATE_WIN_RATE
                and max_dd <= MAX_GATE_DRAWDOWN
            )

        gate_sig = "|".join(
            f"{a}:{int(ok)}:{r}:{v}:{s:.3f}:{w:.3f}:{d:.3f}:{int(lg)}"
            for a, ok, r, v, s, w, d, lg in gate_items
        )
        now = time.time()
        should_log_gate = (
            gate_sig != self._last_gate_status_sig
            or (now - self._last_gate_status_ts) >= self._gate_status_log_ttl
        )
        if should_log_gate:
            self._last_gate_status_sig = gate_sig
            self._last_gate_status_ts = now
            for (
                asset,
                ok,
                reason,
                version,
                sharpe,
                win_rate,
                max_dd,
                legacy,
            ) in gate_items:
                try:
                    log_health.info(
                        "ASSET_GATE_STATUS | asset=%s ok=%s reason=%s version=%s sharpe=%.3f win_rate=%.3f max_dd=%.3f legacy=%s",
                        asset,
                        ok,
                        reason,
                        version,
                        sharpe,
                        win_rate,
                        max_dd,
                        legacy,
                    )
                except Exception:
                    continue

        strict_gate_reason = gate_reason
        if (
            (not gate_ok)
            and _partial_gate_enabled(required_assets)
            and isinstance(assets, dict)
        ):
            partial_assets: list[str] = []
            for asset in required_assets:
                st = assets.get(asset, {})
                if isinstance(st, dict) and bool(st.get("ok", False)):
                    partial_assets.append(asset)
            partial_mode_reason = "partial_gate"

            if not partial_assets and _env_truthy(
                "PARTIAL_GATE_ALLOW_WFA_FALLBACK", "1"
            ):
                partial_assets = []
                for asset in required_assets:
                    st = assets.get(asset, {})
                    if not isinstance(st, dict):
                        continue
                    reason_s = str(st.get("reason", "") or "")
                    if reason_s.startswith(
                        ("state_wfa_failed", "meta_wfa_failed")
                    ) and _fallback_metrics_ok(st):
                        partial_assets.append(asset)
                if partial_assets:
                    partial_mode_reason = "partial_wfa_fallback"

            if not partial_assets and _env_truthy(
                "PARTIAL_GATE_ALLOW_SAMPLE_QUALITY_FALLBACK", "1"
            ):
                partial_assets = []
                for asset in required_assets:
                    st = assets.get(asset, {})
                    if not isinstance(st, dict):
                        continue
                    reason_s = str(st.get("reason", "") or "")
                    sample_only_unsafe = (
                        reason_s.startswith(
                            ("state_marked_unsafe:", "meta_marked_unsafe:")
                        )
                        and ("sample_quality_fail" in reason_s)
                        and ("wfa_fail" not in reason_s)
                        and ("stress_fail" not in reason_s)
                        and ("risk_of_ruin=" not in reason_s)
                    )
                    if sample_only_unsafe and _fallback_metrics_ok(st):
                        partial_assets.append(asset)
                if partial_assets:
                    partial_mode_reason = "partial_sample_quality_fallback"

            if partial_assets:
                active_assets = list(partial_assets)
                blocked_assets = [a for a in required_assets if a not in active_assets]
                gate_ok = True
                gate_reason = (
                    f"{partial_mode_reason}:{strict_gate_reason}:blocked={','.join(blocked_assets)}"
                    if blocked_assets
                    else f"{partial_mode_reason}:{strict_gate_reason}"
                )
                self._gate_last_reason = gate_reason
                if should_log_gate:
                    log_health.warning(
                        "MODEL_GATE_PARTIAL | active=%s blocked=%s strict_reason=%s mode=%s",
                        ",".join(active_assets),
                        ",".join(blocked_assets) if blocked_assets else "-",
                        strict_gate_reason,
                        partial_mode_reason,
                    )

        if not gate_ok:
            if should_log_gate:
                log_health.warning("MODEL_GATE_SKIP | reason=%s", gate_reason)
                log_err.error("GATE_BLOCK_REASON | reason=%s", gate_reason)
            self._model_loaded = False
            self._backtest_passed = False
            self._model_version = "N/A"
            self._model_sharpe = 0.0
            self._model_win_rate = 0.0
            self._blocked_assets = sorted(set(required_assets))
            if self._run.is_set():
                with self._lock:
                    self._manual_stop = True
                self._drain_queue(self._order_q)
                self._drain_queue(self._result_q)
                with self._order_state_lock:
                    self._order_rm_by_id.clear()
                    self._pending_order_meta.clear()
                log_health.warning(
                    "MODEL_GATE_MONITORING_ONLY | trading_disabled=True analytics_alive=True reason=%s",
                    gate_reason,
                )
            return

        versions: list[str] = []
        sharpes: list[float] = []
        wins: list[float] = []
        for asset in active_assets:
            st = assets.get(asset, {}) if isinstance(assets, dict) else {}
            version = str(st.get("model_version", "") or "").strip()
            if not version:
                log_health.warning(
                    "MODEL_GATE_SKIP | reason=missing_model_version asset=%s", asset
                )
                self._model_loaded = False
                self._backtest_passed = False
                return
            model = model_manager.load_model(version)
            if model is None:
                log_health.warning(
                    "MODEL_GATE_SKIP | reason=registry_load_failed asset=%s version=%s",
                    asset,
                    version,
                )
                self._model_loaded = False
                self._backtest_passed = False
                return
            versions.append(f"{asset}:{version}")
            sharpes.append(InferenceEngine._safe_float(st.get("sharpe")))
            wins.append(InferenceEngine._safe_float(st.get("win_rate")))
            if isinstance(model, dict) and "model" in model and "pipeline" in model:
                self._catboost_payloads[asset] = model
                log_health.info("CATBOOST_LOADED | asset=%s version=%s", asset, version)

        self._model_version = " | ".join(versions)
        self._model_sharpe = min(sharpes) if sharpes else 0.0
        self._model_win_rate = min(wins) if wins else 0.0
        self._model_loaded = True
        self._backtest_passed = True
        self._gate_last_reason = gate_reason
        self._blocked_assets = sorted(
            set(required_assets).difference(set(active_assets))
        )
        log_health.info(
            "MODEL_GATE_PASSED | versions=%s sharpe_min=%.3f win_rate_min=%.3f blocked=%s reason=%s",
            self._model_version,
            self._model_sharpe,
            self._model_win_rate,
            ",".join(self._blocked_assets) if self._blocked_assets else "-",
            gate_reason,
        )

    def _infer_catboost(
        self,
        asset: str,
        payloads: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    ) -> Optional[MLSignal]:
        return self._inference_engine.infer_catboost(asset, payloads=payloads)

    def reload_model(self) -> None:
        """Hot-reload model from disk after retraining."""
        with self._lock:
            self._check_model_health()
            if not (self._model_loaded and self._backtest_passed):
                raise RuntimeError("model_reload_failed:gate_blocked")
        log_health.info(
            "MODEL_RELOADED | version=%s sharpe=%.3f win_rate=%.3f",
            self._model_version,
            self._model_sharpe,
            self._model_win_rate,
        )


# =============================================================================
# Module Exports
# =============================================================================
__all__ = ["EngineModelMixin", "InferenceEngine", "MODEL_STATE_PATH"]
