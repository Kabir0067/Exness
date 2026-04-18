# core/feature_engine.py - feature engineering and feature audits.

from __future__ import annotations

import logging

# ---- merged from core/feature_engine.py ----
# core/feature_engine.py — Unified FeatureEngine for all assets.
# Merges _btc_indicators/feature_engine.py (1106 lines)
#    and _xau_indicators/feature_engine.py (927 lines)
# into a single config-driven class.
import math
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
    tf_seconds,
)

log = logging.getLogger("core.data_engine")

FEATURES_VALID = "features valid"
NAN_DETECTED = "nan detected"
INSUFFICIENT_WARMUP = "insufficient warmup"
LOOKAHEAD_BIAS_DETECTED = "lookahead bias detected"
MTF_ALIGNMENT_ERROR = "mtf alignment error"
SCALING_ERROR = "scaling error"

_FEATURE_STATE_PRIORITY = {
    FEATURES_VALID: 0,
    INSUFFICIENT_WARMUP: 1,
    NAN_DETECTED: 2,
    SCALING_ERROR: 3,
    LOOKAHEAD_BIAS_DETECTED: 4,
    MTF_ALIGNMENT_ERROR: 5,
}


def safe_last(arr: np.ndarray, default: float = 0.0) -> float:
    """Extract last finite value from an array."""
    if arr is None or len(arr) == 0:
        return default
    v = float(arr[-1])
    return v if math.isfinite(v) else default


def _shifted_last(arr: np.ndarray, shift: int, default: float = 0.0) -> float:
    """
    Extract shift-aware last value.
    shift=0 → arr[-1]  (current bar)
    shift=1 → arr[-2]  (previous bar, prevents lookahead)
    """
    if arr is None or len(arr) == 0:
        return default
    pos = 1 + shift  # 1-based distance from end
    if pos > len(arr):
        return default
    v = float(arr[-pos])
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


@dataclass(frozen=True)
class FeatureAuditIssue:
    code: str
    message: str
    state: str


@dataclass
class FeatureAuditResult:
    state: str = FEATURES_VALID
    valid: bool = True
    issues: List[FeatureAuditIssue] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.issues is None:
            self.issues = []
        if self.metadata is None:
            self.metadata = {}

    def add_issue(
        self,
        code: str,
        message: str,
        state: str,
        **details: Any,
    ) -> None:
        issue = FeatureAuditIssue(code=code, message=message, state=state)
        if issue not in self.issues:
            self.issues.append(issue)
        if _FEATURE_STATE_PRIORITY[state] > _FEATURE_STATE_PRIORITY[self.state]:
            self.state = state
        self.valid = self.state == FEATURES_VALID
        if details:
            self.metadata[code] = details

    def merge(self, other: "FeatureAuditResult") -> "FeatureAuditResult":
        if other is None:
            return self
        self.metadata.update(other.metadata)
        for issue in other.issues:
            self.add_issue(issue.code, issue.message, issue.state)
        return self

    def reason_codes(self) -> List[str]:
        reasons = [f"feature_state:{self.state}"]
        for issue in self.issues:
            reasons.append(issue.code)
        return reasons


class FeatureIntegrityError(RuntimeError):
    def __init__(self, audit: FeatureAuditResult) -> None:
        self.audit = audit
        super().__init__(audit.state)


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

        self._last_feature_audit: FeatureAuditResult = FeatureAuditResult()

        self._validate_cfg()
        log.info("FeatureEngine initialized for %s", symbol)

    def _validate_cfg(self) -> None:
        for attr in [
            "_ema_short",
            "_ema_medium",
            "_rsi_period",
            "_atr_period",
            "_cci_period",
            "_mom_period",
            "_roc_period",
        ]:
            v = getattr(self, attr)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"Invalid indicator config: {attr}={v}")

    # ═════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═════════════════════════════════════════════════════════════════

    def compute_indicators(
        self,
        df_dict: Dict[str, pd.DataFrame],
        shift: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute indicators for all timeframes.

        Args:
            df_dict: {timeframe_str: DataFrame} with OHLCV data
            shift: 1 = use prev bar (prevent look-ahead), 0 = use current

        Returns:
            {timeframe_str: {indicator_name: value}} for each TF
        """
        shift_i = int(shift)
        audit = FeatureAuditResult(metadata={"shift": shift_i})
        self._last_feature_audit = audit
        if shift_i < 1:
            audit.add_issue(
                "shift_not_closed_bar",
                "Live feature computation must use closed bars only",
                LOOKAHEAD_BIAS_DETECTED,
                shift=shift_i,
            )
            raise FeatureIntegrityError(audit)

        result: Dict[str, Dict[str, Any]] = {}
        frame_meta: Dict[str, Dict[str, Any]] = {}

        for tf, df in df_dict.items():
            input_audit = self._audit_input_frame(tf=tf, df=df, shift=shift_i)
            audit.merge(input_audit)
            if not input_audit.valid:
                result[tf] = {}
                frame_meta[tf] = self._frame_meta(tf=tf, df=df, shift=shift_i)
                continue
            try:
                indicators = self._compute_tf(tf=tf, df=df, shift=shift_i)
            except Exception as exc:
                log.error(
                    "compute_indicators(tf=%s) failed: %s\n%s",
                    tf,
                    exc,
                    traceback.format_exc(),
                )
                raise RuntimeError(
                    f"feature computation failed for tf={tf}: {exc}"
                ) from exc
            meta = self._frame_meta(tf=tf, df=df, shift=shift_i)
            frame_meta[tf] = meta
            indicators["feature_lag_bars"] = shift_i
            indicators["feature_valid"] = True
            if meta.get("used_time") is not None:
                indicators["feature_source_time"] = meta["used_time"].isoformat()
            result[tf] = indicators
            audit.merge(
                self._audit_feature_output(
                    tf=tf,
                    df=df,
                    out=indicators,
                    shift=shift_i,
                    meta=meta,
                )
            )

        # MTF alignment
        if len(result) >= 2:
            self._check_mtf_alignment(result, frame_meta, audit)

        self._last_feature_audit = audit
        if not audit.valid:
            raise FeatureIntegrityError(audit)

        return result

    def last_feature_audit(self) -> FeatureAuditResult:
        return self._last_feature_audit

    def classify_market_regime(
        self,
        indicators: Dict[str, Dict[str, Any]],
        *,
        asset: str = "",
        now: Optional[Any] = None,
        primary_tf: Optional[str] = None,
    ) -> str:
        """Classify the live market state from closed-bar features only."""
        if not indicators:
            return "range"

        tf = str(
            primary_tf or getattr(self.cfg.symbol_params, "tf_primary", "M1") or "M1"
        )
        primary = (
            indicators.get(tf)
            or indicators.get("M1")
            or next(iter(indicators.values()), {})
        )
        if not isinstance(primary, dict) or not primary:
            return "range"

        asset_u = str(
            asset or getattr(self.cfg.symbol_params, "base", "") or ""
        ).upper()
        ts = self._coerce_timestamp(now)
        if ts is None:
            ts = pd.Timestamp.utcnow()
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        trend = str(primary.get("trend", "flat") or "flat").lower()
        forecast = str(primary.get("forecast", "none") or "none").lower()
        volatility = str(
            primary.get("volatility_regime", primary.get("regime", "normal"))
            or "normal"
        ).lower()
        adx = float(primary.get("adx", 0.0) or 0.0)
        atr_ratio = float(primary.get("atr_ratio", 1.0) or 1.0)
        linreg_slope = float(primary.get("linreg_slope", 0.0) or 0.0)
        confluence = float(primary.get("confluence", 0.0) or 0.0)
        stop_hunt_strength = abs(float(primary.get("stop_hunt_strength", 0.0) or 0.0))
        divergence = str(primary.get("divergence", "") or "").lower()
        anomaly = primary.get("anomaly")
        anomaly_level = str(getattr(anomaly, "level", "normal") or "normal").lower()

        tags: List[str] = []
        minute_utc = int(ts.hour * 60 + ts.minute)
        weekday = int(ts.weekday())

        if "BTC" in asset_u and weekday >= 5:
            tags.append("weekend BTC regime")

        if minute_utc in range(420, 481) or minute_utc in range(750, 811):
            tags.append("session open")
        elif minute_utc < 360 or minute_utc >= 1320:
            tags.append("session dead hours")

        if minute_utc in range(745, 851) or minute_utc in range(360, 391):
            tags.append("news hours")

        breakout = forecast in {"bull_continuation", "bear_continuation"}
        fake_breakout = bool(
            breakout
            and (
                stop_hunt_strength >= 0.60
                or divergence in {"bullish", "bearish"}
                or anomaly_level in {"warning", "critical"}
            )
        )

        if fake_breakout:
            tags.append("fake breakout")
        elif breakout:
            tags.append("breakout")

        if volatility == "explosive" or atr_ratio >= 1.50:
            tags.append("high volatility")
        elif volatility == "dead" or atr_ratio <= 0.70:
            tags.append("low volatility")

        strong_trend = bool(
            trend in {"bull", "bear"}
            and (
                adx >= 25.0
                or abs(linreg_slope)
                >= max(
                    1e-4,
                    abs(float(primary.get("close", 0.0) or 0.0)) * 1e-6,
                )
                or confluence >= 0.70
            )
        )

        if strong_trend:
            tags.append("trend strong")
        elif trend in {"bull", "bear"}:
            tags.append("trend weak")
        elif not breakout and volatility != "explosive":
            tags.append("range")

        if not tags:
            tags.append("range")

        unique_tags: List[str] = []
        for tag in tags:
            if tag not in unique_tags:
                unique_tags.append(tag)
        return "|".join(unique_tags[:4])

    def _required_warmup_bars(self, timeframe: str, shift: int) -> int:
        del timeframe
        return int(
            max(
                self._ema_anchor + shift + 5,
                self._macd_slow + self._macd_signal + shift + 5,
                self._bb_period + shift + 5,
                self._atr_period + shift + 5,
                self._stoch_k_period + self._stoch_slowd_period + shift + 5,
                50,
            )
        )

    def _audit_input_frame(
        self,
        *,
        tf: str,
        df: Optional[pd.DataFrame],
        shift: int,
    ) -> FeatureAuditResult:
        audit = FeatureAuditResult(metadata={"tf": tf})
        if df is None or len(df) == 0:
            audit.add_issue(
                f"{tf.lower()}_frame_missing",
                f"{tf} frame is missing",
                INSUFFICIENT_WARMUP,
            )
            return audit

        warmup_bars = self._required_warmup_bars(tf, shift)
        if len(df) < warmup_bars:
            audit.add_issue(
                f"{tf.lower()}_insufficient_warmup",
                f"{tf} requires {warmup_bars} bars, got {len(df)}",
                INSUFFICIENT_WARMUP,
                required=warmup_bars,
                actual=len(df),
            )

        cols = {c.lower(): c for c in df.columns}
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(set(cols)):
            audit.add_issue(
                f"{tf.lower()}_schema_error",
                f"{tf} frame is missing required OHLC columns",
                SCALING_ERROR,
            )
        return audit

    def _frame_meta(
        self,
        *,
        tf: str,
        df: Optional[pd.DataFrame],
        shift: int,
    ) -> Dict[str, Any]:
        tf_sec = int(tf_seconds(tf))
        meta: Dict[str, Any] = {
            "tf": tf,
            "tf_seconds": tf_sec,
            "shift": int(shift),
            "used_time": None,
            "latest_time": None,
            "used_close_time": None,
        }
        if df is None or len(df) <= shift:
            return meta

        time_col = None
        cols = {c.lower(): c for c in df.columns}
        if "time" in cols:
            time_col = cols["time"]

        used_idx = len(df) - 1 - shift
        latest_idx = len(df) - 1
        if time_col is not None:
            used_time = self._coerce_timestamp(df.iloc[used_idx][time_col])
            latest_time = self._coerce_timestamp(df.iloc[latest_idx][time_col])
        else:
            used_time = self._coerce_timestamp(df.index[used_idx])
            latest_time = self._coerce_timestamp(df.index[latest_idx])

        meta["used_time"] = used_time
        meta["latest_time"] = latest_time
        if used_time is not None:
            meta["used_close_time"] = used_time + pd.Timedelta(seconds=tf_sec)
        return meta

    def _audit_feature_output(
        self,
        *,
        tf: str,
        df: pd.DataFrame,
        out: Dict[str, Any],
        shift: int,
        meta: Dict[str, Any],
    ) -> FeatureAuditResult:
        audit = FeatureAuditResult(metadata={"tf": tf})
        if not isinstance(out, dict) or not out:
            audit.add_issue(
                f"{tf.lower()}_empty_features",
                f"{tf} features are empty after computation",
                INSUFFICIENT_WARMUP,
            )
            return audit

        for key, value in out.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                if not math.isfinite(float(value)):
                    audit.add_issue(
                        f"{tf.lower()}_{key}_nan",
                        f"{tf}:{key} is non-finite",
                        NAN_DETECTED,
                    )
            elif isinstance(value, AnomalyResult):
                if not math.isfinite(float(value.score)):
                    audit.add_issue(
                        f"{tf.lower()}_anomaly_score_nan",
                        f"{tf}: anomaly score is non-finite",
                        NAN_DETECTED,
                    )

        if out.get("feature_lag_bars") != shift:
            audit.add_issue(
                f"{tf.lower()}_feature_lag_mismatch",
                f"{tf} feature lag does not match requested shift",
                LOOKAHEAD_BIAS_DETECTED,
            )

        df = self._ensure_volume(df)
        cols = {c.lower(): c for c in df.columns}
        c = self._as_1d_float(df[cols.get("close", "Close")].to_numpy())
        h = self._as_1d_float(df[cols.get("high", "High")].to_numpy())
        low_arr = self._as_1d_float(df[cols.get("low", "Low")].to_numpy())
        used_len = len(c) - shift
        if used_len <= 0:
            audit.add_issue(
                f"{tf.lower()}_lookahead_underflow",
                f"{tf} used length is invalid for shift={shift}",
                LOOKAHEAD_BIAS_DETECTED,
            )
            return audit

        c_seen = c[:used_len]
        h_seen = h[:used_len]
        low_seen = low_arr[:used_len]
        price_min = float(np.min(c_seen))
        price_max = float(np.max(c_seen))

        self._check_bounded_feature(
            audit,
            tf=tf,
            key="rsi",
            value=out.get("rsi"),
            lower=0.0,
            upper=100.0,
        )
        self._check_bounded_feature(
            audit,
            tf=tf,
            key="adx",
            value=out.get("adx"),
            lower=0.0,
            upper=100.0,
        )
        self._check_bounded_feature(
            audit,
            tf=tf,
            key="stoch_k",
            value=out.get("stoch_k"),
            lower=0.0,
            upper=100.0,
        )
        self._check_bounded_feature(
            audit,
            tf=tf,
            key="stoch_d",
            value=out.get("stoch_d"),
            lower=0.0,
            upper=100.0,
        )
        self._check_bounded_feature(
            audit,
            tf=tf,
            key="ker",
            value=out.get("ker"),
            lower=0.0,
            upper=1.0,
        )
        self._check_bounded_feature(
            audit,
            tf=tf,
            key="rvi",
            value=out.get("rvi"),
            lower=0.0,
            upper=1.0,
        )
        self._check_bounded_feature(
            audit,
            tf=tf,
            key="confluence",
            value=out.get("confluence"),
            lower=0.0,
            upper=1.0,
        )
        self._check_bounded_feature(
            audit,
            tf=tf,
            key="ob_touch_proximity",
            value=out.get("ob_touch_proximity"),
            lower=0.0,
            upper=1.0,
        )
        self._check_bounded_feature(
            audit,
            tf=tf,
            key="linreg_slope",
            value=out.get("linreg_slope"),
            lower=-1.0,
            upper=1.0,
        )
        self._check_bounded_feature(
            audit,
            tf=tf,
            key="ob_pretouch_bias",
            value=out.get("ob_pretouch_bias"),
            lower=-1.0,
            upper=1.0,
        )
        self._check_bounded_feature(
            audit,
            tf=tf,
            key="stop_hunt_strength",
            value=out.get("stop_hunt_strength"),
            lower=-1.0,
            upper=1.0,
        )

        for ema_key in ("ema_short", "ema_medium", "ema_slow", "ema_anchor"):
            ema_val = float(out.get(ema_key, 0.0) or 0.0)
            tol = self._feature_tol(
                max(abs(price_max), abs(price_min), abs(ema_val), 1.0)
            )
            if ema_val < (price_min - tol) or ema_val > (price_max + tol):
                audit.add_issue(
                    f"{tf.lower()}_{ema_key}_range_error",
                    f"{tf}:{ema_key} fell outside the observed price envelope",
                    SCALING_ERROR,
                )

        atr_val = float(out.get("atr", 0.0) or 0.0)
        if atr_val < 0.0:
            audit.add_issue(
                f"{tf.lower()}_atr_negative",
                f"{tf}: ATR must be non-negative",
                SCALING_ERROR,
            )
        prev_close = c_seen[:-1]
        if len(prev_close) > 0:
            tr_seen = np.maximum(
                h_seen[1:] - low_seen[1:],
                np.maximum(
                    np.abs(h_seen[1:] - prev_close),
                    np.abs(low_seen[1:] - prev_close),
                ),
            )
            if tr_seen.size > 0:
                atr_tol = self._feature_tol(max(float(np.max(tr_seen)), atr_val, 1.0))
                if atr_val > float(np.max(tr_seen)) + atr_tol:
                    audit.add_issue(
                        f"{tf.lower()}_atr_reference_error",
                        f"{tf}: ATR exceeded the observed true-range envelope",
                        SCALING_ERROR,
                    )

        macd_val = float(out.get("macd", 0.0) or 0.0)
        macd_signal_val = float(out.get("macd_signal", 0.0) or 0.0)
        macd_hist_val = float(out.get("macd_hist", 0.0) or 0.0)
        macd_tol = self._feature_tol(
            max(abs(macd_val), abs(macd_signal_val), abs(macd_hist_val), 1.0)
        )
        if abs((macd_val - macd_signal_val) - macd_hist_val) > macd_tol:
            audit.add_issue(
                f"{tf.lower()}_macd_identity_error",
                f"{tf}: MACD histogram violated line-signal identity",
                SCALING_ERROR,
            )

        if float(out.get("bb_width", 0.0) or 0.0) < 0.0:
            audit.add_issue(
                f"{tf.lower()}_bb_width_negative",
                f"{tf}: Bollinger width must be non-negative",
                SCALING_ERROR,
            )
        if float(out.get("atr_pct", 0.0) or 0.0) < 0.0:
            audit.add_issue(
                f"{tf.lower()}_atr_pct_negative",
                f"{tf}: ATR percent must be non-negative",
                SCALING_ERROR,
            )
        if float(out.get("atr_ratio", 0.0) or 0.0) < 0.0:
            audit.add_issue(
                f"{tf.lower()}_atr_ratio_negative",
                f"{tf}: ATR ratio must be non-negative",
                SCALING_ERROR,
            )

        audit.merge(
            self._verify_reference_alignment(
                tf=tf,
                df=df,
                out=out,
                shift=shift,
                meta=meta,
            )
        )
        return audit

    def _verify_reference_alignment(
        self,
        *,
        tf: str,
        df: pd.DataFrame,
        out: Dict[str, Any],
        shift: int,
        meta: Dict[str, Any],
    ) -> FeatureAuditResult:
        audit = FeatureAuditResult(metadata={"tf": tf})
        del meta
        cols = {c.lower(): c for c in df.columns}
        c = self._as_1d_float(df[cols.get("close", "Close")].to_numpy())
        h = self._as_1d_float(df[cols.get("high", "High")].to_numpy())
        low_arr = self._as_1d_float(df[cols.get("low", "Low")].to_numpy())
        used_len = len(c) - shift
        if used_len <= 1:
            audit.add_issue(
                f"{tf.lower()}_reference_underflow",
                f"{tf}: reference validation length is insufficient",
                INSUFFICIENT_WARMUP,
            )
            return audit

        c_cut = c[:used_len]
        h_cut = h[:used_len]
        low_cut = low_arr[:used_len]
        c0 = float(c_cut[0]) if len(c_cut) > 0 else 0.0

        ema_ref = safe_last(
            self._nan_to_num(talib.EMA(c_cut, timeperiod=self._ema_medium), c0)
        )
        rsi_ref = safe_last(
            self._nan_to_num(talib.RSI(c_cut, timeperiod=self._rsi_period), 50.0)
        )
        macd_line_ref, macd_signal_ref, _ = talib.MACD(
            c_cut,
            fastperiod=self._macd_fast,
            slowperiod=self._macd_slow,
            signalperiod=self._macd_signal,
        )
        macd_line_ref = safe_last(self._nan_to_num(macd_line_ref, 0.0))
        macd_signal_ref = safe_last(self._nan_to_num(macd_signal_ref, 0.0))
        atr_ref = safe_last(
            self._nan_to_num(
                talib.ATR(h_cut, low_cut, c_cut, timeperiod=self._atr_period),
                0.0,
            )
        )

        checks = (
            ("ema_medium", float(out.get("ema_medium", 0.0) or 0.0), float(ema_ref)),
            ("rsi", float(out.get("rsi", 0.0) or 0.0), float(rsi_ref)),
            ("macd", float(out.get("macd", 0.0) or 0.0), float(macd_line_ref)),
            (
                "macd_signal",
                float(out.get("macd_signal", 0.0) or 0.0),
                float(macd_signal_ref),
            ),
            ("atr", float(out.get("atr", 0.0) or 0.0), float(atr_ref)),
        )
        for key, live_val, ref_val in checks:
            tol = self._feature_tol(max(abs(live_val), abs(ref_val), 1.0))
            if abs(live_val - ref_val) > tol:
                audit.add_issue(
                    f"{tf.lower()}_{key}_lookahead",
                    f"{tf}:{key} diverged from closed-bar reference",
                    LOOKAHEAD_BIAS_DETECTED,
                    live=live_val,
                    reference=ref_val,
                )
        return audit

    def _check_bounded_feature(
        self,
        audit: FeatureAuditResult,
        *,
        tf: str,
        key: str,
        value: Any,
        lower: float,
        upper: float,
    ) -> None:
        if value is None:
            return
        val = float(value)
        if not math.isfinite(val):
            audit.add_issue(
                f"{tf.lower()}_{key}_nan",
                f"{tf}:{key} is non-finite",
                NAN_DETECTED,
            )
            return
        tol = self._feature_tol(max(abs(lower), abs(upper), abs(val), 1.0))
        if val < (lower - tol) or val > (upper + tol):
            audit.add_issue(
                f"{tf.lower()}_{key}_scale_error",
                f"{tf}:{key} breached expected bounds [{lower}, {upper}]",
                SCALING_ERROR,
            )

    @staticmethod
    def _feature_tol(scale: float) -> float:
        return max(1e-8, float(scale) * 1e-8)

    @staticmethod
    def _coerce_timestamp(value: Any) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            if value.tzinfo is None:
                return value.tz_localize("UTC")
            return value.tz_convert("UTC")
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                return ts.tz_localize("UTC")
            return ts.tz_convert("UTC")
        except Exception:
            return None

    # ═════════════════════════════════════════════════════════════════
    # TIMEFRAME COMPUTATION
    # ═════════════════════════════════════════════════════════════════

    def _compute_tf(
        self,
        *,
        tf: str,
        df: pd.DataFrame,
        shift: int,
    ) -> Dict[str, Any]:
        """Compute all indicators for a single timeframe."""
        df = self._ensure_volume(df)
        min_bars = self._min_bars(tf)
        if len(df) < min_bars:
            return {}

        # Normalize columns
        cols = {c.lower(): c for c in df.columns}
        h = self._as_1d_float(df[cols.get("high", "High")].to_numpy())
        low_arr = self._as_1d_float(df[cols.get("low", "Low")].to_numpy())
        c = self._as_1d_float(df[cols.get("close", "Close")].to_numpy())
        o = self._as_1d_float(df[cols.get("open", "Open")].to_numpy())
        v = (
            self._as_1d_float(
                df[
                    cols.get("tick_volume", cols.get("volume", "tick_volume"))
                ].to_numpy()
            )
            if any(k in cols for k in ("tick_volume", "volume"))
            else np.ones(len(c), dtype=np.float64)
        )

        # Defensive alignment in case duplicated columns produced odd shapes.
        n = int(min(len(h), len(low_arr), len(c), len(o), len(v)))
        if n < min_bars:
            return {}
        if len(h) != n:
            h = h[-n:]
        if len(low_arr) != n:
            low_arr = low_arr[-n:]
        if len(c) != n:
            c = c[-n:]
        if len(o) != n:
            o = o[-n:]
        if len(v) != n:
            v = v[-n:]

        # Fill NaN/inf
        c = self._ffill_finite(c)
        h = self._ffill_finite(h, default=c[0] if len(c) > 0 else 0.0)
        low_arr = self._ffill_finite(low_arr, default=c[0] if len(c) > 0 else 0.0)
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
        atr = self._nan_to_num(
            talib.ATR(h, low_arr, c, timeperiod=self._atr_period), 0.0
        )
        atr_last = _shifted_last(atr, shift)
        _cl_pos = 1 + shift  # 1-based offset from end
        close_last = float(c[-_cl_pos]) if _cl_pos <= len(c) else float(c[-1])
        atr_pct = atr_last / close_last if close_last > 0 else 0.0

        # ── Fractal efficiency & volatility (shift-aware) ──
        _c_for_ker = c[: len(c) - shift] if shift > 0 and shift < len(c) else c
        ker = kaufman_efficiency_ratio(_c_for_ker, period=self._ker_period)
        rvi = relative_volatility_index(_c_for_ker, period=self._rvi_period)

        # ── ADX (TA-Lib) ──
        adx = self._nan_to_num(
            talib.ADX(h, low_arr, c, timeperiod=self._adx_period), 0.0
        )

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
        _bb_u = _shifted_last(bb_upper, shift)
        _bb_l = _shifted_last(bb_lower, shift)
        _bb_m = _shifted_last(bb_middle, shift)
        bb_width = (_bb_u - _bb_l) / _bb_m if _bb_m > 0 else 0.0
        bb_pctb = (close_last - _bb_l) / (_bb_u - _bb_l) if (_bb_u - _bb_l) > 0 else 0.5

        # ── Momentum & Volatility (TA-Lib) ──
        cci = self._nan_to_num(
            talib.CCI(h, low_arr, c, timeperiod=self._cci_period), 0.0
        )
        mom = self._nan_to_num(talib.MOM(c, timeperiod=self._mom_period), 0.0)
        roc = self._nan_to_num(talib.ROC(c, timeperiod=self._roc_period), 0.0)
        stoch_k, stoch_d = talib.STOCH(
            h,
            low_arr,
            c,
            fastk_period=self._stoch_k_period,
            slowk_period=self._stoch_slowk_period,
            slowk_matype=self._stoch_matype,
            slowd_period=self._stoch_slowd_period,
            slowd_matype=self._stoch_matype,
        )
        stoch_k = self._nan_to_num(stoch_k, 50.0)
        stoch_d = self._nan_to_num(stoch_d, 50.0)
        obv = self._nan_to_num(talib.OBV(c, v), 0.0)

        # ── Shift-aware arrays and DataFrame for pattern/helper functions ──
        # When shift>0, trim arrays/df so [-1] refers to the shifted bar,
        # preventing lookahead bias in backtest mode.  (forensic fix 2026-04-02)
        if shift > 0 and shift < len(c):
            _c_s = c[:-shift]
            _h_s = h[:-shift]
            _low_s = low_arr[:-shift]
            _o_s = o[:-shift]
            _v_s = v[:-shift]
            _atr_s = atr[:-shift]
            _rsi_s = rsi[:-shift]
            _df_s = df.iloc[:-shift]
        else:
            _c_s, _h_s, _low_s, _o_s, _v_s = c, h, low_arr, o, v
            _atr_s, _rsi_s = atr, rsi
            _df_s = df

        # Recompute atr_last and z_vol on shifted arrays for pattern functions
        _atr_last_s = safe_last(_atr_s)

        # ── VWAP (shift-aware) ──
        vwap = self._vwap(_df_s, window=20)

        # ── Volume Z-Score (shift-aware) ──
        z_vol = self._z_score(_v_s, period=50)
        z_vol_series = self._z_score_series(v, period=50)
        if shift > 0 and shift < len(z_vol_series):
            _z_vol_s = z_vol_series[:-shift]
        else:
            _z_vol_s = z_vol_series
        _z_vol_s_scalar = z_vol

        # ── Trend (shift-aware) ──
        trend = self._determine_trend(
            c,
            ema_m,
            ema_l,
            ema_a,
            adx,
            shift=shift,
        )

        # ── Linreg slope (shift-aware) ──
        linreg_slope = self.linreg_trend_slope(_c_s, period=20)

        # ── Volatility regime (shift-aware) ──
        vol_regime, atr_ratio = self.volatility_regime(_atr_s, period=50)

        # ── Divergence (shift-aware) ──
        div = self._detect_divergence_swings(ind=_rsi_s, price=_c_s)

        # ── Forecast (shift-aware) ──
        forecast = self.forecast_continuation(_df_s, _v_s)

        # ── Smart money patterns (shift-aware) ──
        fvg = self._detect_fvg(_df_s, _atr_s)
        sweep = self._liquidity_sweep(
            _df_s,
            atr_last=_atr_last_s,
            z_volume=_z_vol_s_scalar,
            adx=adx,
        )
        ob = self._order_block(
            _df_s,
            atr_last=_atr_last_s,
            v=_v_s,
            z_vol_series=_z_vol_s,
        )
        stop_hunt_side, stop_hunt_strength = self._stop_hunt_profile(
            df=_df_s,
            atr_last=_atr_last_s,
            z_volume=_z_vol_s_scalar,
            sweep=sweep,
        )
        ob_touch_prox, ob_pretouch_bias = self._order_block_pretouch(
            df=_df_s,
            atr_last=_atr_last_s,
        )
        near_rn = self._near_round_number(price=close_last, atr=atr_last)

        # ── Anomalies (shift-aware) ──
        anomaly = self._detect_market_anomalies(
            dfp=_df_s,
            atr_series=_atr_s,
            z_vol_series=_z_vol_s,
            adx=_shifted_last(adx, shift),
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

        # Build output (shift-aware): shift=0 → idx=-1 (current), shift=1 → idx=-2 (prev)
        idx = -(1 + shift) if (1 + shift) <= len(c) else -1
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
            "adx": _shifted_last(adx, shift),
            # Bollinger
            "bb_upper": _bb_u,
            "bb_lower": _bb_l,
            "bb_middle": _bb_m,
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
        """Vectorized forward-fill for non-finite values."""
        out = np.asarray(a, dtype=np.float64).copy()
        if out.size == 0:
            return out
        finite = np.isfinite(out)
        if finite.all():
            return out
        if not finite.any():
            out.fill(float(default))
            return out

        idx = np.where(finite, np.arange(out.size, dtype=np.int64), -1)
        np.maximum.accumulate(idx, out=idx)
        leading = idx < 0
        if np.any(leading):
            out[leading] = float(default)
        missing = ~finite & ~leading
        if np.any(missing):
            out[missing] = out[idx[missing]]
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
            low_arr = df[cols.get("low", "Low")].values

            vol_col = None
            for k in ("tick_volume", "volume"):
                if k in cols:
                    vol_col = cols[k]
                    break
            if vol_col is None:
                return float(c[-1]) if len(c) > 0 else 0.0
            v = df[vol_col].values.astype(np.float64)

            w = min(window, len(c))
            typical = (c[-w:] + h[-w:] + low_arr[-w:]) / 3.0
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
        p = max(2, int(period))
        if len(vals) == 0:
            return np.zeros(0, dtype=np.float64)
        ser = pd.Series(vals, dtype=np.float64)
        mean = ser.rolling(window=p, min_periods=p).mean().shift(1)
        std = ser.rolling(window=p, min_periods=p).std(ddof=0).shift(1)
        z = (ser - mean) / std.replace(0.0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return z.to_numpy(dtype=np.float64)

    # ═════════════════════════════════════════════════════════════════
    # PATTERN DETECTION
    # ═════════════════════════════════════════════════════════════════

    def _determine_trend(
        self,
        c: np.ndarray,
        ema21: np.ndarray,
        ema50: np.ndarray,
        ema200: np.ndarray,
        adx: np.ndarray,
        *,
        shift: int = 0,
    ) -> str:
        """Determine trend direction from EMA stack (shift-aware)."""
        s = _shifted_last(ema21, shift)
        m = _shifted_last(ema50, shift)
        l_val = _shifted_last(ema200, shift)
        if s > m > l_val > 0:
            return "bull"
        if s < m < l_val and l_val > 0:
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
            low_arr = df[cols.get("low", "Low")].values
            if len(h) < 3:
                return ""
            atr_last = safe_last(atr)
            if atr_last <= 0:
                return ""

            # Bullish FVG: gap between bar[-3] high and bar[-1] low
            gap = low_arr[-1] - h[-3]
            if gap > atr_last * 0.5:
                return "bull_fvg"

            # Bearish FVG
            gap = low_arr[-3] - h[-1]
            if gap > atr_last * 0.5:
                return "bear_fvg"
        except Exception:
            pass
        return ""

    def _liquidity_sweep(
        self,
        df: pd.DataFrame,
        *,
        atr_last: float,
        z_volume: float,
        adx: np.ndarray,
    ) -> str:
        """Detect liquidity sweeps (stop hunts)."""
        try:
            cols = {c.lower(): c for c in df.columns}
            h = df[cols.get("high", "High")].values
            low_arr = df[cols.get("low", "Low")].values
            c = df[cols.get("close", "Close")].values
            if len(h) < 10 or atr_last <= 0:
                return ""

            # Lookback for recent high/low
            look = min(20, len(h) - 1)
            recent_high = float(np.max(h[-look:-1]))
            recent_low = float(np.min(low_arr[-look:-1]))

            # Bullish sweep: price dips below recent low then closes above
            if low_arr[-1] < recent_low and c[-1] > recent_low:
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
        self,
        df: pd.DataFrame,
        *,
        atr_last: float,
        v: np.ndarray,
        z_vol_series: np.ndarray,
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
            low_series = pd.to_numeric(df[cols.get("low", "Low")], errors="coerce")
            o = pd.to_numeric(df[cols.get("open", "Open")], errors="coerce")
            c = pd.to_numeric(df[cols.get("close", "Close")], errors="coerce")
            if len(h) < 3:
                return side, float(signed * 0.30)
            rng = float((h.iloc[-1] - low_series.iloc[-1]) or 0.0)
            if rng <= 0.0:
                return side, float(signed * 0.30)
            lower_wick = float(
                max(0.0, min(o.iloc[-1], c.iloc[-1]) - low_series.iloc[-1])
            )
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
            b_anchor = (
                float(bull_ob.iloc[-1]) if np.isfinite(bull_ob.iloc[-1]) else np.nan
            )
            s_anchor = (
                float(bear_ob.iloc[-1]) if np.isfinite(bear_ob.iloc[-1]) else np.nan
            )
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
        self,
        *,
        ind: np.ndarray,
        price: np.ndarray,
    ) -> str:
        """Detect bullish/bearish RSI divergence."""
        try:
            if len(ind) < 20 or len(price) < 20:
                return ""

            p = price[-20:]
            r = ind[-20:]
            low_idx = [
                i
                for i in range(1, len(p) - 1)
                if np.isfinite(p[i])
                and p[i] <= p[i - 1]
                and p[i] <= p[i + 1]
                and np.isfinite(r[i])
            ]
            high_idx = [
                i
                for i in range(1, len(p) - 1)
                if np.isfinite(p[i])
                and p[i] >= p[i - 1]
                and p[i] >= p[i + 1]
                and np.isfinite(r[i])
            ]

            if len(low_idx) >= 2:
                i1, i2 = low_idx[-2], low_idx[-1]
                if float(p[i2]) < float(p[i1]) and float(r[i2]) > float(r[i1]):
                    return "bull_div"

            if len(high_idx) >= 2:
                i1, i2 = high_idx[-2], high_idx[-1]
                if float(p[i2]) > float(p[i1]) and float(r[i2]) < float(r[i1]):
                    return "bear_div"
        except Exception:
            pass
        return ""

    def _detect_market_anomalies(
        self,
        *,
        dfp: pd.DataFrame,
        atr_series: np.ndarray,
        z_vol_series: np.ndarray,
        adx: float,
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
        self,
        closes: np.ndarray,
        period: Optional[int] = None,
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
        self,
        atr_series: np.ndarray,
        period: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Detect volatility regime based on ATR ratio.
        Returns (regime, atr_ratio).
        Note: caller passes shift-trimmed atr_series so [-1] is the correct bar.
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
        self,
        df: pd.DataFrame,
        vol_series: np.ndarray,
    ) -> str:
        """Predict next candle continuation: breakout + volume confirmation."""
        try:
            cols = {c.lower(): c for c in df.columns}
            c = df[cols.get("close", "Close")].values
            h = df[cols.get("high", "High")].values
            low_arr = df[cols.get("low", "Low")].values

            if len(c) < 3:
                return "none"

            avg_vol = float(np.mean(vol_series[-20:])) if len(vol_series) >= 20 else 1.0
            last_vol = float(vol_series[-1]) if len(vol_series) > 0 else 0.0
            vol_mult = last_vol / avg_vol if avg_vol > 0 else 0.0

            if c[-1] > h[-2] and vol_mult > 1.5:
                return "bull_continuation"
            if c[-1] < low_arr[-2] and vol_mult > 1.5:
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
        self,
        score: float,
        m1_data: Dict[str, Any],
    ) -> str:
        if score >= 70:
            return "strong"
        elif score >= 50:
            return "moderate"
        elif score >= 30:
            return "weak"
        return "none"

    def _check_mtf_alignment(
        self,
        out: Dict[str, Any],
        frame_meta: Dict[str, Dict[str, Any]],
        audit: Optional[FeatureAuditResult] = None,
    ) -> None:
        """Check multi-timeframe trend alignment and closed-candle safety."""
        primary_tf = str(
            getattr(getattr(self.cfg, "symbol_params", None), "tf_primary", "") or ""
        )
        if primary_tf not in frame_meta and frame_meta:
            primary_tf = min(frame_meta, key=lambda x: int(tf_seconds(x)))
        primary_meta = frame_meta.get(primary_tf, {})
        primary_sec = int(primary_meta.get("tf_seconds", 0) or 0)
        primary_close_time = primary_meta.get("used_close_time")

        if primary_close_time is not None:
            for tf, meta in frame_meta.items():
                tf_sec = int(meta.get("tf_seconds", 0) or 0)
                used_close_time = meta.get("used_close_time")
                if tf_sec <= primary_sec or used_close_time is None:
                    continue
                if used_close_time > primary_close_time:
                    if audit is not None:
                        audit.add_issue(
                            f"{tf.lower()}_mtf_open_candle",
                            f"{tf} features were derived from a candle not yet closed",
                            MTF_ALIGNMENT_ERROR,
                            used_close_time=str(used_close_time),
                            primary_close_time=str(primary_close_time),
                        )
                    if isinstance(out.get(tf), dict):
                        out[tf]["mtf_closed"] = False
                elif isinstance(out.get(tf), dict):
                    out[tf]["mtf_closed"] = True

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


__all__ = (
    "FEATURES_VALID",
    "NAN_DETECTED",
    "INSUFFICIENT_WARMUP",
    "LOOKAHEAD_BIAS_DETECTED",
    "MTF_ALIGNMENT_ERROR",
    "SCALING_ERROR",
    "safe_last",
    "AnomalyResult",
    "FeatureAuditIssue",
    "FeatureAuditResult",
    "FeatureIntegrityError",
    "FeatureEngine",
)
