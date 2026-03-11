from __future__ import annotations

import time
import traceback
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import MetaTrader5 as mt5
from mt5_client import mt5_async_call

from core.ml_router import MLSignal, validate_payload_schema

from .logging_setup import log_err, log_health
from .models import AssetCandidate
from .utils import parse_bar_key, tf_seconds

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

    def validate_payload_timestamp(self, asset: str, payload: Dict[str, Any], tf: str) -> Tuple[bool, str]:
        if not isinstance(payload, dict):
            return False, "payload_not_dict"

        tf_obj = payload.get(tf)
        if not isinstance(tf_obj, dict):
            return False, f"tf_missing:{tf}"

        raw_ts = tf_obj.get("ts_bar")
        ts_value = 0.0
        if isinstance(raw_ts, (int, float)):
            ts_value = float(raw_ts)
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
                t_close_value = float(raw_t_close)
            elif isinstance(raw_t_close, str):
                if "_" in raw_t_close:
                    return False, f"t_close_format_mismatch:{raw_t_close}"
                parsed_close = parse_bar_key(raw_t_close)
                if parsed_close is None:
                    return False, f"t_close_parse_failed:{raw_t_close}"
                t_close_value = float(parsed_close.timestamp())
            else:
                return False, "t_close_type_invalid"

            bar_sec = float(tf_seconds(tf) or 60.0)
            tolerance = max(5.0, bar_sec * 2.0)
            if abs(ts_value - t_close_value) > tolerance:
                return False, f"t_close_mismatch:{abs(ts_value - t_close_value):.1f}s>{tolerance:.1f}s"

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
        c = max(0.0, min(1.0, float(confidence)))
        atr = float(frame.get("atr_14", 0.0) or 0.0)
        if atr <= 0.0 and entry > 0.0:
            atr = max(0.00001, entry * 0.001)

        levels: list[float] = []
        for k in ("ema_20", "ema_50", "ema_200"):
            try:
                v = float(frame.get(k, 0.0) or 0.0)
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

        payload = sig.scalp_payload if isinstance(sig.scalp_payload, dict) else sig.intraday_payload
        frame = self.extract_payload_frame(payload)
        entry = float(sig.entry or frame.get("last_close") or 0.0)
        atr = float(frame.get("atr_14", 0.0) or 0.0)
        atr_pct = (atr / entry) if atr > 0.0 and entry > 0.0 else 0.0

        ind = {
            "atr": atr,
            "atr_pct": atr_pct,
            "close": float(frame.get("last_close", entry) or entry),
        }
        adapt = {
            "atr": atr,
            "atr_pct": atr_pct,
            "confidence": max(0.0, min(1.0, float(sig.confidence))),
            "regime": "ml_router",
        }
        open_positions = e._asset_open_positions(asset)
        max_positions = e._asset_max_positions(asset)

        plan = pipe.risk.plan_order(
            side=sig.side,
            confidence=float(sig.confidence),
            ind=ind,
            adapt=adapt,
            entry=(entry if entry > 0.0 else None),
            open_positions=int(open_positions),
            max_positions=int(max_positions),
            df=None,
        )
        if bool(plan.get("blocked", True)):
            log_health.info(
                "FSM_RISK_BLOCK | asset=%s signal=%s reason=%s open=%d max=%d",
                asset,
                sig.signal,
                str(plan.get("reason", "unknown")),
                int(open_positions),
                int(max_positions),
            )
            return None

        plan_entry = float(plan.get("entry", 0.0) or 0.0)
        plan_sl = float(plan.get("sl", 0.0) or 0.0)
        plan_tp = float(plan.get("tp", 0.0) or 0.0)
        sl, tp = self.probabilistic_levels(
            side=sig.side,
            entry=plan_entry,
            base_sl=plan_sl,
            base_tp=plan_tp,
            confidence=float(sig.confidence),
            frame=frame,
        )
        lot = float(plan.get("lot", 0.0) or 0.0)
        lot = e._apply_asset_lot_cap(asset, lot, "build")
        if lot <= 0.0:
            return None

        ts_bar = 0
        try:
            ts_bar = int(frame.get("ts_bar", 0) or 0)
        except Exception:
            ts_bar = 0
        if ts_bar <= 0:
            ts_bar = int(time.time())

        reasons = (
            f"ml_provider:{sig.provider}",
            f"ml_model:{sig.model}",
            f"ml_reason:{sig.reason}",
            "sniper_fsm",
        )

        return AssetCandidate(
            asset=asset,
            symbol=str(pipe.symbol),
            signal=str(sig.side),
            confidence=float(sig.confidence),
            lot=lot,
            sl=float(sl),
            tp=float(tp),
            latency_ms=0.0,
            blocked=False,
            reasons=reasons,
            signal_id=f"ML_{asset}_{ts_bar}_{sig.side}",
            raw_result={
                "ml_signal": sig.signal,
                "provider": sig.provider,
                "model": sig.model,
                "reason": sig.reason,
                "confidence": sig.confidence,
            },
        )
