from __future__ import annotations

import time
import traceback
from collections import deque
from typing import TYPE_CHECKING, Optional

import MetaTrader5 as mt5
from mt5_client import mt5_async_call

from core.ml_router import MLSignal

from .logging_setup import log_err, log_health

if TYPE_CHECKING:
    from .engine import MultiAssetTradingEngine


class InferenceEngine:
    """Decoupled CatBoost inference service for the motor."""

    def __init__(self, engine: "MultiAssetTradingEngine") -> None:
        self._e = engine

    def infer_catboost(self, asset: str) -> Optional[MLSignal]:
        """Run CatBoost model inference on live M1 data from MT5."""
        e = self._e
        if e._is_asset_blocked(asset):
            e._log_blocked_asset_skip(asset, "catboost_infer")
            return None
        payload = e._catboost_payloads.get(asset)
        if not payload:
            return None
        try:
            import numpy as np
            import pandas as pd

            cb_model = payload["model"]
            pipeline = payload["pipeline"]

            pipe = e._xau if asset == "XAU" else e._btc
            if pipe is None:
                return None
            symbol = pipe.symbol
            if not symbol:
                return None

            n_bars = max(int(getattr(pipeline.cfg, "window_size", 60) or 60) + 200, 500)
            rates = mt5_async_call(
                "copy_rates_from_pos",
                symbol,
                mt5.TIMEFRAME_M1,
                0,
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

            try:
                tick_now = mt5_async_call(
                    "symbol_info_tick",
                    symbol,
                    timeout=0.35,
                    default=None,
                )
                if tick_now is not None:
                    bid_now = float(getattr(tick_now, "bid", 0.0) or 0.0)
                    ask_now = float(getattr(tick_now, "ask", 0.0) or 0.0)
                    mid_now = 0.5 * (bid_now + ask_now) if bid_now > 0.0 and ask_now > 0.0 else 0.0
                    if mid_now > 0.0 and len(df) > 0:
                        last_idx = df.index[-1]
                        df.at[last_idx, "Close"] = mid_now
                        df.at[last_idx, "High"] = max(float(df.at[last_idx, "High"]), mid_now)
                        df.at[last_idx, "Low"] = min(float(df.at[last_idx, "Low"]), mid_now)
            except Exception:
                pass

            if hasattr(pipeline, "transform_live"):
                X_series = pipeline.transform_live(df.copy())
            else:
                xy = pipeline.transform(df.copy())
                X_series = xy["X"]
            if X_series.empty:
                log_health.warning("CATBOOST_SKIP | asset=%s reason=empty_features", asset)
                return None

            X_last = np.stack([X_series.iloc[-1]])
            pred = cb_model.predict(X_last)
            pred_val = float(pred[0])

            static_threshold = max(float(getattr(pipeline.cfg, "percent_increase", 0.0) or 0.0), 1e-12)
            model_threshold = 0.0
            model_quantile = 0.0
            try:
                alpha_cal = payload.get("alpha_calibration", {}) if isinstance(payload, dict) else {}
                model_threshold = float(alpha_cal.get("pred_abs_threshold", 0.0) or 0.0)
                model_quantile = float(alpha_cal.get("pred_quantile", 0.0) or 0.0)
                if not np.isfinite(model_threshold) or model_threshold < 0.0:
                    model_threshold = 0.0
                if not np.isfinite(model_quantile) or model_quantile < 0.0 or model_quantile > 99.9:
                    model_quantile = 0.0
            except Exception:
                model_threshold = 0.0
                model_quantile = 0.0

            if asset not in e._catboost_pred_history:
                e._catboost_pred_history[asset] = deque(maxlen=200)
            e._catboost_pred_history[asset].append(abs(pred_val))

            now_ts = time.time()
            density_mult, density_min_zscore, density_min_flow, sig_24h = e._signal_density_controls(asset, now_ts)
            e._prune_signal_window(e._signal_emit_ts_global, now_ts)
            sig_24h_global = int(len(e._signal_emit_ts_global))

            hist = e._catboost_pred_history[asset]
            hist_arr = np.asarray(hist, dtype=np.float64)
            hist_q = float(np.percentile(hist_arr, float(e._catboost_threshold_q))) if len(hist_arr) > 0 else 0.0
            model_q_thr = (
                float(np.percentile(hist_arr, model_quantile))
                if len(hist_arr) > 0 and model_quantile > 0.0
                else 0.0
            )
            if len(hist) >= int(e._catboost_hist_min):
                threshold = max(static_threshold, model_threshold, model_q_thr, hist_q, 1e-12) * float(density_mult)
                hist_mu = float(np.mean(hist_arr))
                hist_sigma = float(np.std(hist_arr))
                z_mag = (abs(pred_val) - hist_mu) / hist_sigma if hist_sigma > 1e-12 else 0.0
                p95_mag = max(float(np.percentile(hist_arr, 95)), threshold * 1.10)
                conf_ref = max(p95_mag - threshold, threshold * 0.15, 1e-12)
            else:
                threshold = max(static_threshold, model_threshold, 1e-12) * float(density_mult)
                z_mag = 0.0
                p95_mag = threshold * 2.0
                conf_ref = max(threshold, 1e-12)

            mag = abs(pred_val)
            if mag < threshold:
                confidence = max(0.05, min(0.74, (mag / max(threshold, 1e-12)) * 0.74))
            else:
                rel = (mag - threshold) / conf_ref
                rel = max(0.0, min(1.0, rel))
                confidence = 0.75 + (0.24 * rel)
            confidence = max(0.0, min(0.99, confidence))

            last_close = float(df["Close"].iloc[-1])
            atr_arr = None
            try:
                import talib

                h = np.asarray(df["High"].values, dtype=np.float64)
                l = np.asarray(df["Low"].values, dtype=np.float64)
                c = np.asarray(df["Close"].values, dtype=np.float64)
                atr_arr = talib.ATR(h, l, c, timeperiod=14)
            except Exception:
                pass
            atr_val = (
                float(atr_arr[-1])
                if atr_arr is not None and len(atr_arr) > 0 and np.isfinite(atr_arr[-1])
                else last_close * 0.001
            )

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
                    t_imb = float(getattr(ts, "imbalance", 0.0) or 0.0)
                    t_delta = float(getattr(ts, "tick_delta", 0.0) or 0.0)
                    flow_confirm = max(-1.0, min(1.0, 0.55 * t_imb + 0.45 * t_delta))
            except Exception:
                flow_confirm = 0.0

            if abs(pred_val) < threshold:
                return MLSignal(
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
                            "rsi_14": 50.0,
                            "ema_20": last_close,
                            "ema_50": last_close,
                            "ema_200": last_close,
                        }
                    },
                    intraday_payload=None,
                )
            if len(hist) >= int(e._catboost_hist_min) and z_mag < float(density_min_zscore):
                return MLSignal(
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
                            "rsi_14": 50.0,
                            "ema_20": last_close,
                            "ema_50": last_close,
                            "ema_200": last_close,
                        }
                    },
                    intraday_payload=None,
                )
            if (pred_val * flow_confirm) < -float(density_min_flow):
                return MLSignal(
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
                            "rsi_14": 50.0,
                            "ema_20": last_close,
                            "ema_50": last_close,
                            "ema_200": last_close,
                        }
                    },
                    intraday_payload=None,
                )

            signal = "STRONG BUY" if pred_val > 0 else "STRONG SELL"
            side = "Buy" if pred_val > 0 else "Sell"
            frame = {
                "ts_bar": float(df.index[-1].timestamp()),
                "last_close": last_close,
                "atr_14": atr_val,
                "rsi_14": 50.0,
                "ema_20": last_close,
                "ema_50": last_close,
                "ema_200": last_close,
            }

            return MLSignal(
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
                scalp_payload={"M1": frame},
                intraday_payload={"H1": frame},
            )
        except Exception as exc:
            log_err.error("CATBOOST_INFER_ERROR | asset=%s err=%s | tb=%s", asset, exc, traceback.format_exc())
            return None

