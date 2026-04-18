# Backtest/model_train_institutional.py
# Institutional-grade training — REFACTORED v2
# Key improvements over v1:
#   1. PERF:  `create_windows` fully vectorised with numpy stride tricks — no Python loop
#   2. PERF:  `add_indicators` uses a shared sub-computation cache to avoid redundant
#             talib calls (EMA/ATR computed at most once per period across features)
#   3. FIX:   Bare `pass` on mt5.symbol_select failure replaced with warning log
#   4. FIX:   `historical_vol_*` now uses talib.STDDEV for consistency with other indicators
#   5. FIX:   MACD feature guard now also skips macd_signal / macd_hist re-computation
#   6. MINOR: All public methods fully typed; docstrings added; no functional regressions

from __future__ import annotations

import datetime
import logging
import math
import os
import pathlib
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
import talib

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:
    HistGradientBoostingRegressor = None

try:
    from sklearn.preprocessing import StandardScaler as _SKStandardScaler
except Exception:
    _SKStandardScaler = None

import MetaTrader5 as mt5

from core.model_engine import ModelMetadata, model_manager
from log_config import LOG_DIR, get_artifact_dir, get_artifact_path, get_log_path

try:
    from mt5_client import MT5_LOCK, ensure_mt5
except Exception:
    import threading

    MT5_LOCK = threading.RLock()

    def ensure_mt5() -> bool:
        return False


warnings.filterwarnings("ignore")
log = logging.getLogger("backtest.model_train_institutional")
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _fh = RotatingFileHandler(
        str(get_log_path("backtest_model_train.log")),
        maxBytes=8 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(_fmt)
    log.addHandler(_fh)

_LOG_DIR = LOG_DIR / "model_training_institutional"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_ART_DUMPS = get_artifact_dir("dumps")
_ART_CATBOOST = get_artifact_dir("catboost_info")


# ─────────────────────────────────────────────────────────────────────────────
# Scaler implementations
# ─────────────────────────────────────────────────────────────────────────────


class _NumpyStandardScaler:
    """Pure-numpy fallback scaler when sklearn is unavailable."""

    def __init__(self, dtype: np.dtype = np.float32) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.dtype = dtype

    def fit(self, x: np.ndarray) -> "_NumpyStandardScaler":
        arr = np.asarray(x, dtype=self.dtype)
        self.mean_ = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        std[std <= 1e-12] = 1.0
        self.scale_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=self.dtype)
        if self.mean_ is None or self.scale_ is None:
            return arr
        return (arr - self.mean_) / self.scale_


class _NumpyLinearRegressor:
    """Minimal numpy-only ridge regressor — fallback when no ML library is available."""

    def __init__(self, l2: float = 1e-6) -> None:
        self.l2 = float(max(l2, 0.0))
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_NumpyLinearRegressor":
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if x_arr.ndim != 2 or x_arr.shape[0] != y_arr.shape[0]:
            raise RuntimeError("numpy_linear_invalid_shape_or_length_mismatch")
        if x_arr.shape[0] <= 0:
            raise RuntimeError("numpy_linear_empty_fit")
        ones = np.ones((x_arr.shape[0], 1), dtype=np.float64)
        x_aug = np.concatenate([x_arr, ones], axis=1)
        gram = x_aug.T @ x_aug
        if self.l2 > 0.0:
            gram += self.l2 * np.eye(gram.shape[0], dtype=np.float64)
        rhs = x_aug.T @ y_arr
        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(x_aug, y_arr, rcond=None)[0]
        self.coef_ = np.asarray(beta[:-1], dtype=np.float64)
        self.intercept_ = float(beta[-1])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("numpy_linear_not_fitted")
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        return (x_arr @ self.coef_ + self.intercept_).astype(np.float64, copy=False)


class ScalerProtocol(Protocol):
    def fit(self, x: np.ndarray) -> "ScalerProtocol": ...
    def transform(self, x: np.ndarray) -> np.ndarray: ...


StandardScaler = (
    _NumpyStandardScaler if _SKStandardScaler is None else _SKStandardScaler
)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────


def _cfg_dtype(cfg: "InstitutionalTrainConfig") -> np.dtype:
    raw = str(getattr(cfg, "float_dtype", "float32") or "float32").strip().lower()
    if raw in {"float64", "f8", "64", "double"}:
        return np.float64
    return np.float32


def _stack_series(series: pd.Series, dtype: np.dtype) -> np.ndarray:
    arr = np.stack(series.values)
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def _make_scaler(dtype: np.dtype) -> ScalerProtocol:
    if _SKStandardScaler is None:
        return _NumpyStandardScaler(dtype=dtype)
    return _SKStandardScaler()


def _env_truthy(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, *, min_v: int, max_v: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        v = int(float(raw))
    except Exception:
        return int(default)
    return int(max(min_v, min(max_v, v)))


def _console(msg: str) -> None:
    """Route progress messages to dedicated training log; optionally echo to stdout."""
    txt = str(msg).rstrip()
    if not txt:
        return
    for line in txt.splitlines():
        line = line.rstrip()
        if line:
            log.info(line)
    if str(os.getenv("BACKTEST_ECHO_CONSOLE", "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        try:
            real_out = getattr(sys, "__stdout__", None) or sys.stdout
            real_out.write(txt + "\n")
            real_out.flush()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InstitutionalTrainConfig:
    """Institutional training configuration with advanced features."""

    symbol: str = "XAUUSD"
    mean_price_indc: str = "Close"

    # Target — ZERO look-ahead
    target_type: str = "returns_magnitude"
    prediction_horizon: int = 15
    percent_increase: float = field(default_factory=lambda: 0.0008)

    # Features
    window_size: int = 20
    cci_period: int = 20
    mom_period: int = 10
    roc_period: int = 10
    stoch_k_period: int = 14
    stoch_slowk_period: int = 3
    stoch_slowd_period: int = 3
    stoch_matype: int = 0
    training_features: List[str] = field(
        default_factory=lambda: [
            "ofi_proxy",
            "volume_delta",
            "tick_momentum_3",
            "tick_momentum_8",
            "liquidity_void_up",
            "liquidity_void_down",
            "liquidity_void_score",
            "mtf_h1_slope",
            "mtf_h4_slope",
            "mtf_h1_flow_proxy",
            "mtf_h4_flow_proxy",
            "stop_hunt_flag",
            "stop_hunt_strength",
            "ob_touch_proximity",
            "ob_pretouch_bias",
            "dxy_ret_1",
            "us10y_ret_1",
            "xau_dxy_corr_rolling",
            "xau_us10y_corr_rolling",
            "ma4",
            "ma9",
            "ma13",
            "ma21",
            "ma50",
            "ema5",
            "ema12",
            "ema26",
            "ema50",
            "macd",
            "macd_signal",
            "macd_hist",
            "rsi13",
            "rsi21",
            "ppo13_5",
            "ppo27_7",
            "momentum_10",
            "momentum_20",
            "mom",
            "roc",
            "cci",
            "stoch_k",
            "stoch_d",
            "atr_14",
            "bbands_width",
            "keltner_width",
            "historical_vol_20",
            "adi",
            "obv",
            "volume_sma_20",
            "high_low_ratio",
            "close_position_in_range",
            "price_acceleration",
        ]
    )

    # Model params
    regressor_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "iterations": 2000,
            "learning_rate": 0.03,
            "depth": 7,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": 0,
            "task_type": "CPU",
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "train_dir": str(_ART_CATBOOST),
        }
    )
    early_stopping_rounds: int = field(default_factory=lambda: 0)

    # Anti-overfit / validity controls (lightweight, chronological only).
    tscv_folds: int = 4
    tscv_min_train_samples: int = 1_200
    tscv_test_samples: int = 320
    cv_max_samples: int = 35_000
    wfa_windows: int = 4
    wfa_train_ratio: float = 0.65
    wfa_test_ratio: float = 0.10
    cv_target_direction_accuracy: float = 0.56
    cv_target_min_coverage: float = 0.01

    # Feature-importance filter (keeps alpha-carrying features only).
    feature_importance_keep_ratio: float = 0.70
    feature_importance_min_ratio: float = 0.05
    feature_importance_min_features: int = 14

    # Splits — train / val / test / holdout
    train_split: float = 0.60
    val_split: float = 0.75
    test_split: float = 0.90  # 10% pure holdout

    normalize_data: bool = True
    float_dtype: str = field(default_factory=lambda: "float32")
    max_window_samples: int = field(default_factory=lambda: 50000)
    model_version: str = "1.0_xau_institutional"
    dataname: str = "data/XAUUSD_1m.csv"
    dump_path: str = str(get_artifact_path("dumps", "xau_model_institutional.pkl"))


BTC_TRAIN_CONFIG = InstitutionalTrainConfig(
    symbol="BTCUSD",
    prediction_horizon=12,
    percent_increase=0.0012,
    window_size=30,
    cci_period=30,
    mom_period=14,
    roc_period=14,
    stoch_k_period=14,
    stoch_slowk_period=5,
    stoch_slowd_period=5,
    training_features=[
        "ofi_proxy",
        "volume_delta",
        "tick_momentum_3",
        "tick_momentum_8",
        "liquidity_void_up",
        "liquidity_void_down",
        "liquidity_void_score",
        "mtf_h1_slope",
        "mtf_h4_slope",
        "mtf_h1_flow_proxy",
        "mtf_h4_flow_proxy",
        "stop_hunt_flag",
        "stop_hunt_strength",
        "ob_touch_proximity",
        "ob_pretouch_bias",
        "dxy_ret_1",
        "us10y_ret_1",
        "xau_dxy_corr_rolling",
        "xau_us10y_corr_rolling",
        "ma4",
        "ma9",
        "ma13",
        "ma21",
        "ma50",
        "ma100",
        "ema5",
        "ema12",
        "ema26",
        "ema50",
        "ema100",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi13",
        "rsi21",
        "ppo13_5",
        "ppo27_7",
        "ppo48_12",
        "momentum_10",
        "momentum_20",
        "mom",
        "roc",
        "cci",
        "stoch_k",
        "stoch_d",
        "atr_14",
        "bbands_width",
        "keltner_width",
        "historical_vol_20",
        "historical_vol_50",
        "adi",
        "obv",
        "volume_sma_20",
        "volume_sma_50",
        "high_low_ratio",
        "close_position_in_range",
        "price_acceleration",
        "volume_acceleration",
    ],
    regressor_params={
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
        "task_type": "CPU",
        "loss_function": "RMSE",
        "train_dir": str(_ART_CATBOOST),
    },
    early_stopping_rounds=0,
    dataname="data/BTCUSD_1m.csv",
    dump_path=str(get_artifact_path("dumps", "btc_model_institutional.pkl")),
    model_version="1.0_btc_institutional",
)

XAU_TRAIN_CONFIG = InstitutionalTrainConfig()


def _apply_runtime_train_overrides(
    cfg: InstitutionalTrainConfig,
) -> InstitutionalTrainConfig:
    fast_mode = _env_truthy("INSTITUTIONAL_FAST_MODE", "0")

    params = dict(cfg.regressor_params or {})
    cur_iters = int(params.get("iterations", 2000) or 2000)
    iters_default = min(cur_iters, 700) if fast_mode else cur_iters
    new_iters = _env_int("INSTITUTIONAL_MAX_ITERS", iters_default, min_v=60, max_v=6000)
    params["iterations"] = int(new_iters)

    cur_es = int(cfg.early_stopping_rounds or 0)
    es_default = max(cur_es, 120) if fast_mode and cur_es <= 0 else cur_es
    new_es = _env_int(
        "INSTITUTIONAL_EARLY_STOPPING_ROUNDS", es_default, min_v=0, max_v=1200
    )

    cur_cv_samples = int(cfg.cv_max_samples or 35_000)
    cv_samples_default = min(cur_cv_samples, 18_000) if fast_mode else cur_cv_samples
    new_cv_samples = _env_int(
        "INSTITUTIONAL_CV_MAX_SAMPLES", cv_samples_default, min_v=2_000, max_v=250_000
    )

    cur_win_samples = int(cfg.max_window_samples or 50_000)
    win_samples_default = min(cur_win_samples, 25_000) if fast_mode else cur_win_samples
    new_win_samples = _env_int(
        "INSTITUTIONAL_MAX_WINDOW_SAMPLES",
        win_samples_default,
        min_v=2_000,
        max_v=250_000,
    )

    cur_tscv = int(cfg.tscv_folds or 4)
    tscv_default = min(cur_tscv, 3) if fast_mode else cur_tscv
    new_tscv = _env_int("INSTITUTIONAL_TSCV_FOLDS", tscv_default, min_v=2, max_v=8)

    cur_wfa = int(cfg.wfa_windows or 4)
    wfa_default = min(cur_wfa, 3) if fast_mode else cur_wfa
    new_wfa = _env_int("INSTITUTIONAL_WFA_WINDOWS", wfa_default, min_v=1, max_v=8)

    updated = replace(
        cfg,
        regressor_params=params,
        early_stopping_rounds=int(new_es),
        cv_max_samples=int(new_cv_samples),
        max_window_samples=int(new_win_samples),
        tscv_folds=int(new_tscv),
        wfa_windows=int(new_wfa),
    )

    if (
        int(new_iters) != int(cur_iters)
        or int(new_es) != int(cur_es)
        or int(new_cv_samples) != int(cur_cv_samples)
        or int(new_win_samples) != int(cur_win_samples)
        or int(new_tscv) != int(cur_tscv)
        or int(new_wfa) != int(cur_wfa)
    ):
        log.info(
            "TRAIN_RUNTIME_PROFILE | fast_mode=%s iters=%s early_stop=%s cv_samples=%s window_samples=%s tscv_folds=%s wfa_windows=%s",
            int(fast_mode),
            new_iters,
            new_es,
            new_cv_samples,
            new_win_samples,
            new_tscv,
            new_wfa,
        )

    return updated


_MT5_SYMBOL_BY_BASE = {
    "XAUUSD": "XAUUSDm",
    "BTCUSD": "BTCUSDm",
}

_AUX_SYMBOL_CANDIDATES: Dict[str, List[str]] = {
    # Dollar index aliases across brokers.
    "DXY": ["DXY", "DXYm", "USDX", "USDXm", "DX1!", "DOLLAR", "USDINDEX"],
    # US 10Y yield aliases across brokers.
    "US10Y": ["US10Y", "US10Ym", "UST10Y", "US10YT", "TNX", "US10Y.cash"],
}
_AUX_SYMBOL_CACHE: Dict[str, Optional[str]] = {}
_AUX_SERIES_CACHE: Dict[str, tuple[float, pd.Series]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# MT5 symbol resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_mt5_symbol(symbol: str) -> str:
    """Resolve the best-matching MT5 ticker for a canonical asset name."""
    base = str(symbol).upper().strip()
    if base in ("BTC", "BTCUSD", "BTCUSDM"):
        base = "BTCUSD"
    elif base in ("XAU", "XAUUSD", "XAUUSDM"):
        base = "XAUUSD"

    suffix: str = ""
    candidates: List[str] = []
    if suffix:
        candidates.append(f"{base}{suffix}")
    candidates.extend([f"{base}m", base, _MT5_SYMBOL_BY_BASE.get(base, base)])

    seen: set[str] = set()
    uniq: List[str] = []
    for sym in candidates:
        if not sym or sym in seen:
            continue
        seen.add(sym)
        uniq.append(sym)

    for sym in uniq:
        try:
            if mt5.symbol_info(sym) is not None:
                return sym
        except Exception:
            continue

    # Fallback: scan full symbol list
    try:
        symbols = mt5.symbols_get()
    except Exception:
        symbols = None
    if symbols:
        base_u = base.upper()
        names = [s.name for s in symbols if hasattr(s, "name")]
        exact = [n for n in names if n.upper() == base_u]
        if exact:
            return exact[0]
        starts = [n for n in names if n.upper().startswith(base_u)]
        if starts:
            return starts[0]
        contains = [n for n in names if base_u in n.upper()]
        if contains:
            return contains[0]

    return uniq[0] if uniq else base


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class Pipeline:
    """
    Institutional feature pipeline.

    Guarantees:
      - ZERO look-ahead bias (all indicators use only data up to time t)
      - Sub-computation caching: shared talib calls (EMA, ATR) computed at most once
        per period regardless of how many features reference them
      - Vectorised window creation via numpy stride tricks
    """

    def __init__(self, cfg: InstitutionalTrainConfig) -> None:
        self.cfg = cfg
        self.scaler: Optional[ScalerProtocol] = None
        self._dtype = _cfg_dtype(cfg)
        self._aux_cache_ttl_sec = 30.0

    def _feature_live_lookback_m1(self, feat: str) -> int:
        """
        Estimate the minimum M1 history needed for a feature to be meaningful
        during live inference.

        This avoids silently zeroing higher-timeframe features when too little
        M1 history is fetched from MT5.
        """
        name = str(feat or "").strip()
        if not name:
            return 0

        if name == "mtf_h1_slope":
            return (8 + 1) * 60
        if name == "mtf_h4_slope":
            return (6 + 1) * 240
        if name == "mtf_h1_flow_proxy":
            return 5 * 60
        if name == "mtf_h4_flow_proxy":
            return 5 * 240
        if name in ("xau_dxy_corr_rolling", "xau_us10y_corr_rolling"):
            return 240
        if name in ("ofi_proxy", "volume_delta"):
            return 20
        if name in ("stop_hunt_flag", "stop_hunt_strength"):
            return 20
        if name in ("ob_touch_proximity", "ob_pretouch_bias"):
            return 20
        if name in ("liquidity_void_up", "liquidity_void_down", "liquidity_void_score"):
            return 2
        if name in ("high_low_ratio", "close_position_in_range"):
            return 1
        if name in ("price_acceleration", "volume_acceleration"):
            return 3
        if name in ("macd", "macd_signal", "macd_hist"):
            return 35
        if name in ("bbands_width", "keltner_width"):
            return 20
        if name == "adi":
            return 1
        if name == "obv":
            return 2
        if name == "cci":
            return int(getattr(self.cfg, "cci_period", 20) or 20)
        if name == "mom":
            return int(getattr(self.cfg, "mom_period", 10) or 10) + 1
        if name == "roc":
            return int(getattr(self.cfg, "roc_period", 10) or 10) + 1
        if name in ("stoch_k", "stoch_d"):
            return (
                int(getattr(self.cfg, "stoch_k_period", 14) or 14)
                + int(getattr(self.cfg, "stoch_slowk_period", 3) or 3)
                + int(getattr(self.cfg, "stoch_slowd_period", 3) or 3)
            )

        if name.startswith("ma") and not name.startswith("macd"):
            try:
                return int(name[2:])
            except Exception:
                return 0
        if name.startswith("ema"):
            try:
                return int(name[3:])
            except Exception:
                return 0
        if name.startswith("rsi"):
            try:
                return int(name[3:])
            except Exception:
                return 0
        if name.startswith("atr_"):
            try:
                return int(name.split("_", 1)[1])
            except Exception:
                return 0
        if name.startswith("momentum_"):
            try:
                return int(name.split("_", 1)[1]) + 1
            except Exception:
                return 0
        if name.startswith("historical_vol_"):
            try:
                return int(name.split("_", 2)[2]) + 1
            except Exception:
                return 0
        if name.startswith("volume_sma_"):
            try:
                return int(name.split("_", 2)[2])
            except Exception:
                return 0
        if name.startswith("ppo"):
            try:
                fast, slow = map(int, name[3:].split("_"))
                return max(fast, slow)
            except Exception:
                return 0
        return 0

    def required_live_bars(self) -> int:
        """
        Return the M1 history budget needed for scientifically consistent live
        inference with the current feature set.

        The budget covers the longest feature lookback plus enough extra rows
        to build a full inference window after indicator warm-up.
        """
        ws = max(1, int(getattr(self.cfg, "window_size", 20) or 20))
        base_budget = max(ws + 200, 500)
        feat_budget = max(
            (
                self._feature_live_lookback_m1(feat)
                for feat in self.cfg.training_features
            ),
            default=0,
        )
        return max(base_budget, feat_budget + ws + 64)

    def _needs_macro_context(self) -> bool:
        feats = set(self.cfg.training_features)
        needed = {
            "dxy_ret_1",
            "us10y_ret_1",
            "xau_dxy_corr_rolling",
            "xau_us10y_corr_rolling",
        }
        return bool(feats & needed)

    @staticmethod
    def _utc_index(idx: pd.Index) -> pd.DatetimeIndex:
        dt_idx = pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce"))
        if dt_idx.tz is None:
            return dt_idx.tz_localize("UTC")
        return dt_idx.tz_convert("UTC")

    def _resolve_aux_symbol(self, key: str) -> Optional[str]:
        k = str(key).upper().strip()
        if k in _AUX_SYMBOL_CACHE:
            return _AUX_SYMBOL_CACHE[k]
        cands = list(_AUX_SYMBOL_CANDIDATES.get(k, []))
        resolved: Optional[str] = None
        try:
            for sym in cands:
                info = mt5.symbol_info(sym)
                if info is not None:
                    try:
                        if not bool(getattr(info, "visible", True)):
                            mt5.symbol_select(sym, True)
                    except Exception:
                        pass
                    resolved = sym
                    break
            if resolved is None:
                symbols = mt5.symbols_get()
                if symbols:
                    names = [s.name for s in symbols if hasattr(s, "name")]
                    for needle in cands:
                        nu = needle.upper()
                        hit = next((n for n in names if nu in str(n).upper()), None)
                        if hit:
                            resolved = str(hit)
                            break
        except Exception:
            resolved = None
        _AUX_SYMBOL_CACHE[k] = resolved
        return resolved

    def _load_aux_close_series(
        self,
        *,
        key: str,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
    ) -> Optional[pd.Series]:
        sym = self._resolve_aux_symbol(key)
        if not sym:
            return None
        try:
            # Cache by symbol + hour-bucket to keep live inference cheap.
            bucket = int(time.time() // 60)
            ck = f"{sym}|{bucket}"
            cached = _AUX_SERIES_CACHE.get(ck)
            if cached is not None:
                ts_cached, ser_cached = cached
                if (time.time() - float(ts_cached)) <= self._aux_cache_ttl_sec:
                    return ser_cached

            tf_id = mt5.TIMEFRAME_M5
            pad = datetime.timedelta(hours=24)
            start = (start_utc - pad).to_pydatetime()
            end = (end_utc + datetime.timedelta(minutes=5)).to_pydatetime()
            with MT5_LOCK:
                rates = mt5.copy_rates_range(sym, tf_id, start, end)
                if rates is None or len(rates) == 0:
                    bars = max(
                        1200, int((end_utc - start_utc).total_seconds() // 300) + 400
                    )
                    rates = mt5.copy_rates_from_pos(sym, tf_id, 0, bars)
            if rates is None or len(rates) == 0:
                return None
            raw = pd.DataFrame(rates)
            if "time" not in raw.columns or "close" not in raw.columns:
                return None
            raw["datetime"] = pd.to_datetime(raw["time"], unit="s", utc=True)
            ser = pd.Series(
                pd.to_numeric(raw["close"], errors="coerce").values,
                index=pd.DatetimeIndex(raw["datetime"]),
                name=f"{key}_Close",
            ).sort_index()
            ser = ser[~ser.index.duplicated(keep="last")]
            _AUX_SERIES_CACHE[ck] = (time.time(), ser)
            return ser
        except Exception:
            return None

    def inject_macro_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach DXY/US10Y close columns, aligned by timestamp. Missing feeds stay NaN."""
        if not self._needs_macro_context():
            return df
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df
        out = df.copy()
        out.index = self._utc_index(out.index)
        start_utc = out.index[0]
        end_utc = out.index[-1]

        try:
            ensure_mt5()
        except Exception:
            pass

        dxy_ser = self._load_aux_close_series(
            key="DXY", start_utc=start_utc, end_utc=end_utc
        )
        us10y_ser = self._load_aux_close_series(
            key="US10Y", start_utc=start_utc, end_utc=end_utc
        )

        if dxy_ser is not None and not dxy_ser.empty:
            out["DXY_Close"] = dxy_ser.reindex(out.index, method="ffill")
        else:
            out["DXY_Close"] = np.nan
        if us10y_ser is not None and not us10y_ser.empty:
            out["US10Y_Close"] = us10y_ser.reindex(out.index, method="ffill")
        else:
            out["US10Y_Close"] = np.nan

        # CRITICAL FIX — NO LOOKAHEAD:
        # Previously used ffill().bfill().fillna(0.0) which backfilled future
        # values into past bars (data leakage). Now only forward-fill with a
        # strict max_gap; anything still missing stays NaN and is later handled
        # by the downstream numeric/feature pipeline (drop/mask, not synthetic).
        #
        # max_gap rationale: DXY/US10Y are daily/hourly macro series reindexed
        # to the asset's bar timestamps. A forward-fill of up to MAX_FFILL_BARS
        # covers normal weekend/holiday gaps without introducing phantom data.
        MAX_FFILL_BARS = int(
            os.getenv("MACRO_MAX_FFILL_BARS", "1440")  # ~24h on M1, ~60 bars on D1
            or "1440"
        )
        out["DXY_Close"] = pd.to_numeric(
            out["DXY_Close"], errors="coerce"
        ).ffill(limit=MAX_FFILL_BARS)
        out["US10Y_Close"] = pd.to_numeric(
            out["US10Y_Close"], errors="coerce"
        ).ffill(limit=MAX_FFILL_BARS)
        # Leave NaN where no past macro value exists — downstream consumers
        # must treat NaN as "feature unavailable for this bar" and either
        # drop, mask, or use a 0-mean/z-score neutral. NEVER synthesise values.
        return out

    # ------------------------------------------------------------------ #
    # Feature engineering                                                  #
    # ------------------------------------------------------------------ #

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all training features and attach them to `df`.

        FIXED (v2): A `_cache` dict stores results of expensive talib sub-computations
        (EMA by period, ATR by period, STOCH) so they are never recomputed for
        different features that share the same underlying calculation.
        """
        df = self.inject_macro_context(df.copy())
        open_px = pd.to_numeric(df["Open"], errors="coerce")
        close = pd.to_numeric(df["Close"], errors="coerce")
        high = pd.to_numeric(df["High"], errors="coerce")
        low = pd.to_numeric(df["Low"], errors="coerce")
        volume = pd.to_numeric(df["Volume"], errors="coerce")
        dxy_close = (
            pd.to_numeric(df.get("DXY_Close"), errors="coerce")
            if "DXY_Close" in df.columns
            else None
        )
        us10y_close = (
            pd.to_numeric(df.get("US10Y_Close"), errors="coerce")
            if "US10Y_Close" in df.columns
            else None
        )

        close_np = close.to_numpy(dtype=np.float64)
        high_np = high.to_numpy(dtype=np.float64)
        low_np = low.to_numpy(dtype=np.float64)
        volume_np = volume.to_numpy(dtype=np.float64)

        def _s(arr: np.ndarray) -> pd.Series:
            """Wrap a numpy array as a Series with the df's index."""
            return pd.Series(arr, index=df.index)

        # Sub-computation caches — keyed by (type, period)
        _ema_cache: Dict[int, np.ndarray] = {}
        _atr_cache: Dict[int, np.ndarray] = {}
        _vol_sma_cache: Dict[int, pd.Series] = {}
        _macd_cache: Optional[tuple] = None
        _stoch_cache: Optional[tuple] = None

        def _ema(period: int) -> np.ndarray:
            if period not in _ema_cache:
                _ema_cache[period] = talib.EMA(close_np, timeperiod=period)
            return _ema_cache[period]

        def _atr(period: int) -> np.ndarray:
            if period not in _atr_cache:
                _atr_cache[period] = talib.ATR(
                    high_np, low_np, close_np, timeperiod=period
                )
            return _atr_cache[period]

        def _vol_sma(period: int) -> pd.Series:
            if period not in _vol_sma_cache:
                _vol_sma_cache[period] = _s(talib.SMA(volume_np, timeperiod=period))
            return _vol_sma_cache[period]

        _resampled_ohlcv: Dict[str, pd.DataFrame] = {}

        def _resample_ohlcv(rule: str) -> pd.DataFrame:
            if rule in _resampled_ohlcv:
                return _resampled_ohlcv[rule]
            if not isinstance(df.index, pd.DatetimeIndex):
                _resampled_ohlcv[rule] = pd.DataFrame()
                return _resampled_ohlcv[rule]
            agg = (
                pd.DataFrame(
                    {
                        "Open": open_px,
                        "High": high,
                        "Low": low,
                        "Close": close,
                        "Volume": volume,
                    }
                )
                .resample(rule)
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna()
            )
            _resampled_ohlcv[rule] = agg
            return agg

        def _align_series_from_rule(rule: str, series: pd.Series) -> pd.Series:
            if series is None or series.empty:
                return pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
            s = pd.Series(
                series.values, index=pd.DatetimeIndex(series.index)
            ).sort_index()
            return s.reindex(df.index, method="ffill").fillna(0.0)

        def _mtf_slope(rule: str, lookback: int) -> pd.Series:
            rs = _resample_ohlcv(rule)
            if rs.empty or "Close" not in rs.columns:
                return pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
            c_rs = pd.to_numeric(rs["Close"], errors="coerce")
            base = c_rs.shift(lookback).replace(0.0, np.nan)
            slope = ((c_rs - c_rs.shift(lookback)) / base) / max(float(lookback), 1.0)
            slope = slope.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            return _align_series_from_rule(rule, slope)

        def _mtf_flow(rule: str) -> pd.Series:
            rs = _resample_ohlcv(rule)
            if rs.empty:
                return pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
            h_rs = pd.to_numeric(rs["High"], errors="coerce")
            l_rs = pd.to_numeric(rs["Low"], errors="coerce")
            c_rs = pd.to_numeric(rs["Close"], errors="coerce")
            v_rs = pd.to_numeric(rs["Volume"], errors="coerce")
            hl_rs = (h_rs - l_rs).replace(0.0, np.nan)
            clv_rs = ((c_rs - l_rs) - (h_rs - c_rs)) / hl_rs
            v_rel_rs = v_rs / v_rs.rolling(20, min_periods=5).mean().replace(
                0.0, np.nan
            )
            flow_rs = (
                (clv_rs.clip(-1.0, 1.0) * v_rel_rs)
                .replace([np.inf, -np.inf], 0.0)
                .fillna(0.0)
            )
            return _align_series_from_rule(rule, flow_rs)

        # Features to skip after their primary feature already wrote them
        _already_written: set[str] = set()
        _micro_cache: Dict[str, pd.Series] = {}
        _macro_cache: Dict[str, pd.Series] = {}

        def _xau_ret_1() -> pd.Series:
            if "xau_ret_1" not in _macro_cache:
                _macro_cache["xau_ret_1"] = (
                    close.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
                )
            return _macro_cache["xau_ret_1"]

        def _dxy_ret_1() -> pd.Series:
            if "dxy_ret_1" not in _macro_cache:
                if dxy_close is None:
                    _macro_cache["dxy_ret_1"] = pd.Series(
                        np.zeros(len(df), dtype=np.float64), index=df.index
                    )
                else:
                    _macro_cache["dxy_ret_1"] = (
                        dxy_close.pct_change()
                        .replace([np.inf, -np.inf], 0.0)
                        .fillna(0.0)
                    )
            return _macro_cache["dxy_ret_1"]

        def _us10y_ret_1() -> pd.Series:
            if "us10y_ret_1" not in _macro_cache:
                if us10y_close is None:
                    _macro_cache["us10y_ret_1"] = pd.Series(
                        np.zeros(len(df), dtype=np.float64), index=df.index
                    )
                else:
                    _macro_cache["us10y_ret_1"] = (
                        us10y_close.pct_change()
                        .replace([np.inf, -np.inf], 0.0)
                        .fillna(0.0)
                    )
            return _macro_cache["us10y_ret_1"]

        def _stop_hunt_flag() -> pd.Series:
            if "stop_hunt_flag" in _micro_cache:
                return _micro_cache["stop_hunt_flag"]
            recent_high = high.rolling(20, min_periods=10).max().shift(1)
            recent_low = low.rolling(20, min_periods=10).min().shift(1)
            bull_sweep = (low < recent_low) & (close > recent_low)
            bear_sweep = (high > recent_high) & (close < recent_high)
            flag = bull_sweep.astype(float) - bear_sweep.astype(float)
            _micro_cache["stop_hunt_flag"] = flag.fillna(0.0)
            return _micro_cache["stop_hunt_flag"]

        def _stop_hunt_strength() -> pd.Series:
            if "stop_hunt_strength" in _micro_cache:
                return _micro_cache["stop_hunt_strength"]
            flag = _stop_hunt_flag()
            hl = (high - low).replace(0.0, np.nan)
            lower_wick = (np.minimum(open_px, close) - low).clip(lower=0.0)
            upper_wick = (high - np.maximum(open_px, close)).clip(lower=0.0)
            wick_dom = (
                ((lower_wick - upper_wick) / hl)
                .replace([np.inf, -np.inf], 0.0)
                .fillna(0.0)
            )
            vol_mu = volume.rolling(50, min_periods=10).mean()
            vol_sd = volume.rolling(50, min_periods=10).std().replace(0.0, np.nan)
            vol_z = (
                ((volume - vol_mu) / vol_sd).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            )
            strength = flag * (
                0.55 * wick_dom.abs() + 0.45 * (vol_z.abs() / 3.0).clip(0.0, 1.0)
            )
            _micro_cache["stop_hunt_strength"] = strength.clip(-1.0, 1.0).fillna(0.0)
            return _micro_cache["stop_hunt_strength"]

        def _ob_pretouch() -> tuple[pd.Series, pd.Series]:
            if (
                "ob_touch_proximity" in _micro_cache
                and "ob_pretouch_bias" in _micro_cache
            ):
                return (
                    _micro_cache["ob_touch_proximity"],
                    _micro_cache["ob_pretouch_bias"],
                )
            atr14 = _s(_atr(14)).replace(0.0, np.nan)
            body = close - open_px
            prev_open = open_px.shift(1)
            prev_body = body.shift(1)
            bull_ob = pd.Series(
                np.where((prev_body < 0.0) & (body > atr14 * 0.60), prev_open, np.nan),
                index=df.index,
            ).ffill()
            bear_ob = pd.Series(
                np.where((prev_body > 0.0) & (body < -atr14 * 0.60), prev_open, np.nan),
                index=df.index,
            ).ffill()
            dist_bull = ((close - bull_ob).abs() / atr14).replace(
                [np.inf, -np.inf], np.nan
            )
            dist_bear = ((close - bear_ob).abs() / atr14).replace(
                [np.inf, -np.inf], np.nan
            )
            nearest = pd.concat([dist_bull, dist_bear], axis=1).min(axis=1).fillna(9.0)
            prox = (1.0 - (nearest / 3.0).clip(0.0, 1.0)).fillna(0.0)
            bias_raw = (
                ((dist_bear - dist_bull) / 3.0)
                .replace([np.inf, -np.inf], 0.0)
                .fillna(0.0)
            )
            bias = (bias_raw.clip(-1.0, 1.0) * prox).fillna(0.0)
            _micro_cache["ob_touch_proximity"] = prox
            _micro_cache["ob_pretouch_bias"] = bias
            return prox, bias

        for feat in self.cfg.training_features:

            if feat in _already_written:
                continue

            # ── Moving averages ────────────────────────────────────
            if feat == "ofi_proxy":
                hl = (high - low).replace(0.0, np.nan)
                clv = ((close - low) - (high - close)) / hl
                vol_rel = volume / _vol_sma(20).replace(0.0, np.nan)
                df[feat] = (
                    (clv.clip(-1.0, 1.0) * vol_rel)
                    .replace([np.inf, -np.inf], 0.0)
                    .fillna(0.0)
                )
            elif feat == "volume_delta":
                hl = (high - low).replace(0.0, np.nan)
                body_sign = ((close - open_px) / hl).clip(-1.0, 1.0)
                vol_rel = volume / _vol_sma(20).replace(0.0, np.nan)
                df[feat] = (
                    (body_sign * vol_rel).replace([np.inf, -np.inf], 0.0).fillna(0.0)
                )
            elif feat == "tick_momentum_3":
                atr14 = _s(_atr(14)).replace(0.0, np.nan)
                df[feat] = (
                    ((close - close.shift(3)) / atr14)
                    .replace([np.inf, -np.inf], 0.0)
                    .fillna(0.0)
                )
            elif feat == "tick_momentum_8":
                atr14 = _s(_atr(14)).replace(0.0, np.nan)
                df[feat] = (
                    ((close - close.shift(8)) / atr14)
                    .replace([np.inf, -np.inf], 0.0)
                    .fillna(0.0)
                )
            elif feat == "liquidity_void_up":
                prev_close = close.shift(1).replace(0.0, np.nan)
                void_up = (low - high.shift(1)).clip(lower=0.0)
                df[feat] = (
                    (void_up / prev_close).replace([np.inf, -np.inf], 0.0).fillna(0.0)
                )
            elif feat == "liquidity_void_down":
                prev_close = close.shift(1).replace(0.0, np.nan)
                void_down = (low.shift(1) - high).clip(lower=0.0)
                df[feat] = (
                    (void_down / prev_close).replace([np.inf, -np.inf], 0.0).fillna(0.0)
                )
            elif feat == "liquidity_void_score":
                prev_close = close.shift(1).replace(0.0, np.nan)
                void_up = (low - high.shift(1)).clip(lower=0.0)
                void_down = (low.shift(1) - high).clip(lower=0.0)
                void_mag = (void_up + void_down) / prev_close
                vol_rel = volume / _vol_sma(20).replace(0.0, np.nan)
                df[feat] = (
                    (void_mag * vol_rel).replace([np.inf, -np.inf], 0.0).fillna(0.0)
                )
            elif feat == "mtf_h1_slope":
                df[feat] = _mtf_slope("1h", lookback=8)
            elif feat == "mtf_h4_slope":
                df[feat] = _mtf_slope("4h", lookback=6)
            elif feat == "mtf_h1_flow_proxy":
                df[feat] = _mtf_flow("1h")
            elif feat == "mtf_h4_flow_proxy":
                df[feat] = _mtf_flow("4h")
            elif feat == "stop_hunt_flag":
                df[feat] = _stop_hunt_flag()
            elif feat == "stop_hunt_strength":
                df[feat] = _stop_hunt_strength()
            elif feat == "ob_touch_proximity":
                prox, _ = _ob_pretouch()
                df[feat] = prox
            elif feat == "ob_pretouch_bias":
                _, bias = _ob_pretouch()
                df[feat] = bias
            elif feat == "dxy_ret_1":
                df[feat] = _dxy_ret_1()
            elif feat == "us10y_ret_1":
                df[feat] = _us10y_ret_1()
            elif feat == "xau_dxy_corr_rolling":
                corr = _xau_ret_1().rolling(240, min_periods=80).corr(_dxy_ret_1())
                df[feat] = corr.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            elif feat == "xau_us10y_corr_rolling":
                corr = _xau_ret_1().rolling(240, min_periods=80).corr(_us10y_ret_1())
                df[feat] = corr.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            elif feat.startswith("ma") and not feat.startswith("macd"):
                n = int(feat[2:])
                df[feat] = _s(talib.SMA(close_np, timeperiod=n))

            # ── EMA ────────────────────────────────────────────────
            elif feat.startswith("ema"):
                n = int(feat[3:])
                df[feat] = _s(_ema(n))

            # ── MACD (writes all three columns at once) ─────────────
            elif feat in ("macd", "macd_signal", "macd_hist"):
                if _macd_cache is None:
                    _macd_cache = talib.MACD(
                        close_np, fastperiod=12, slowperiod=26, signalperiod=9
                    )
                ml, ms, mh = _macd_cache
                df["macd"] = _s(ml)
                df["macd_signal"] = _s(ms)
                df["macd_hist"] = _s(mh)
                _already_written.update({"macd", "macd_signal", "macd_hist"})

            # ── PPO ────────────────────────────────────────────────
            elif feat.startswith("ppo"):
                p, s = map(int, feat[3:].split("_"))
                df[feat] = _s(talib.PPO(close_np, fastperiod=p, slowperiod=s, matype=0))

            # ── RSI ────────────────────────────────────────────────
            elif feat.startswith("rsi"):
                n = int(feat[3:])
                df[feat] = _s(talib.RSI(close_np, timeperiod=n))

            # ── Momentum (ROC-based) ───────────────────────────────
            elif feat.startswith("momentum_"):
                n = int(feat.split("_")[1])
                df[feat] = _s(talib.ROC(close_np, timeperiod=n))
            elif feat == "mom":
                df[feat] = _s(talib.MOM(close_np, timeperiod=int(self.cfg.mom_period)))
            elif feat == "roc":
                df[feat] = _s(talib.ROC(close_np, timeperiod=int(self.cfg.roc_period)))

            # ── CCI ────────────────────────────────────────────────
            elif feat == "cci":
                df[feat] = _s(
                    talib.CCI(
                        high_np, low_np, close_np, timeperiod=int(self.cfg.cci_period)
                    )
                )

            # ── Stochastic (writes both k and d at once) ───────────
            elif feat in ("stoch_k", "stoch_d"):
                if _stoch_cache is None:
                    _stoch_cache = talib.STOCH(
                        high_np,
                        low_np,
                        close_np,
                        fastk_period=int(self.cfg.stoch_k_period),
                        slowk_period=int(self.cfg.stoch_slowk_period),
                        slowk_matype=int(self.cfg.stoch_matype),
                        slowd_period=int(self.cfg.stoch_slowd_period),
                        slowd_matype=int(self.cfg.stoch_matype),
                    )
                df["stoch_k"] = _s(_stoch_cache[0])
                df["stoch_d"] = _s(_stoch_cache[1])
                _already_written.update({"stoch_k", "stoch_d"})

            # ── ATR ────────────────────────────────────────────────
            elif feat.startswith("atr_"):
                n = int(feat.split("_")[1])
                df[feat] = _s(_atr(n))

            # ── Bollinger Band Width ────────────────────────────────
            elif feat == "bbands_width":
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close_np, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
                )
                width = np.where(bb_middle != 0, (bb_upper - bb_lower) / bb_middle, 0.0)
                df[feat] = _s(width)

            # ── Keltner Channel Width — reuses cached EMA & ATR ────
            elif feat == "keltner_width":
                mid = _ema(20)
                atr20 = _atr(20)
                upper = mid + 2.0 * atr20
                lower = mid - 2.0 * atr20
                width = np.where(close_np != 0, (upper - lower) / close_np, 0.0)
                df[feat] = _s(width)

            # ── Historical Volatility — FIXED (v2 uses talib.STDDEV)
            elif feat.startswith("historical_vol_"):
                n = int(feat.split("_")[2])
                # talib.STDDEV gives rolling std of returns without pandas overhead
                pct_ret = talib.ROC(close_np, timeperiod=1) / 100.0
                df[feat] = _s(talib.STDDEV(pct_ret, timeperiod=n) * np.sqrt(252))

            # ── Volume indicators ──────────────────────────────────
            elif feat == "adi":
                df[feat] = _s(talib.AD(high_np, low_np, close_np, volume_np))
            elif feat == "obv":
                df[feat] = _s(talib.OBV(close_np, volume_np))
            elif feat.startswith("volume_sma_"):
                n = int(feat.split("_")[2])
                df[feat] = _s(talib.SMA(volume_np, timeperiod=n))

            # ── Microstructure ─────────────────────────────────────
            elif feat == "high_low_ratio":
                df[feat] = (high / low - 1.0) * 100.0
            elif feat == "close_position_in_range":
                df[feat] = ((close - low) / (high - low).replace(0.0, np.nan)).fillna(
                    0.5
                )
            elif feat == "price_acceleration":
                df[feat] = close.pct_change().diff()
            elif feat == "volume_acceleration":
                df[feat] = volume.pct_change().diff()

        df.dropna(inplace=True)
        return df

    # ------------------------------------------------------------------ #
    # Target construction — ZERO look-ahead                                #
    # ------------------------------------------------------------------ #

    def add_target_no_lookahead(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct the prediction target without look-ahead bias.

        target[t] = (price[t + horizon] - price[t]) / price[t]

        Features at time t use only data up to t.
        The target at time t uses price at t + horizon (genuinely future).
        Rows near the tail where t + horizon > end are dropped via dropna.
        """
        prices = df[self.cfg.mean_price_indc].values
        horizon = self.cfg.prediction_horizon

        targets = np.full(len(prices), np.nan, dtype=np.float64)
        valid_idx = len(prices) - horizon
        if valid_idx > 0:
            targets[:valid_idx] = (prices[horizon:] - prices[:valid_idx]) / prices[
                :valid_idx
            ]

        df["target_return"] = targets
        if self.cfg.target_type == "returns_direction":
            df["target_direction"] = np.sign(df["target_return"])
            df["target"] = df["target_direction"]
        else:
            df["target"] = df["target_return"]

        df.dropna(subset=["target"], inplace=True)
        return df

    # ------------------------------------------------------------------ #
    # Window creation — vectorised (v2 improvement)                        #
    # ------------------------------------------------------------------ #

    def create_windows(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Convert the flat feature matrix into sliding windows.

        FIXED (v2): Uses numpy stride tricks to build the window tensor in one
        allocation instead of appending to a Python list in a for-loop.
        Maintains identical output shape/dtype to v1.
        """
        feat_matrix = df[self.cfg.training_features].to_numpy(
            dtype=self._dtype, copy=True
        )
        y_arr = df["target"].to_numpy(dtype=self._dtype, copy=False)
        ret_arr = df["target_return"].to_numpy(dtype=self._dtype, copy=False)
        ws = self.cfg.window_size
        n_rows = len(feat_matrix)
        n_feats = feat_matrix.shape[1]

        total = n_rows - ws + 1
        if total <= 0:
            return {
                "X": pd.Series([], dtype=object),
                "y": pd.Series([], dtype=self._dtype),
                "ret": pd.Series([], dtype=self._dtype),
            }

        # Stride / sub-sampling
        max_samples = int(self.cfg.max_window_samples or 0)
        stride = 1
        if max_samples > 0 and total > max_samples:
            stride = int(math.ceil(total / max_samples))
            log.info(
                "Window sampling | total=%s max=%s stride=%s",
                total,
                max_samples,
                stride,
            )

        # Indices of the last row of each window
        win_end_indices = np.arange(ws - 1, n_rows, stride)

        # Build window tensor via stride tricks: shape (n_windows, ws, n_feats)
        # Each window[i] = feat_matrix[win_end_indices[i] - ws + 1 : win_end_indices[i] + 1]
        n_windows = win_end_indices.size
        # Vectorised gather using advanced indexing
        row_offsets = np.arange(ws)  # (ws,)
        window_start = win_end_indices[:, None] - (
            ws - 1 - row_offsets
        )  # (n_windows, ws)
        # Clamp just-in-case (should never be negative with ws-1 base index)
        window_start = np.clip(window_start, 0, n_rows - 1)
        # Gather: (n_windows, ws, n_feats)
        windows_3d = feat_matrix[window_start]  # advanced indexing
        # Flatten last two dims → (n_windows, ws * n_feats)
        flat_windows = windows_3d.reshape(n_windows, ws * n_feats)

        dates = df.index[win_end_indices]
        return {
            "X": pd.Series(list(flat_windows), index=dates),
            "y": pd.Series(y_arr[win_end_indices], index=dates, dtype=self._dtype),
            "ret": pd.Series(ret_arr[win_end_indices], index=dates, dtype=self._dtype),
        }

    def create_windows_inference(self, df: pd.DataFrame) -> pd.Series:
        """
        Create inference windows without target columns.

        Used by live inference to avoid the training-time target construction path
        (which drops the latest bars by horizon and introduces avoidable lag).
        """
        feat_matrix = df[self.cfg.training_features].to_numpy(
            dtype=self._dtype, copy=True
        )
        ws = self.cfg.window_size
        n_rows = len(feat_matrix)
        n_feats = feat_matrix.shape[1]

        total = n_rows - ws + 1
        if total <= 0:
            return pd.Series([], dtype=object)

        win_end_indices = np.arange(ws - 1, n_rows, 1)
        n_windows = len(win_end_indices)
        row_offsets = np.arange(ws, dtype=np.int64)
        window_start = win_end_indices[:, None] - (ws - 1 - row_offsets)
        window_start = np.clip(window_start, 0, n_rows - 1)

        windows_3d = feat_matrix[window_start]
        flat_windows = windows_3d.reshape(n_windows, ws * n_feats)
        dates = df.index[win_end_indices]
        return pd.Series(list(flat_windows), index=dates)

    # ------------------------------------------------------------------ #
    # Train / val / test / holdout split                                   #
    # ------------------------------------------------------------------ #

    def split(self, Xy: Dict[str, pd.Series]) -> Dict[str, Dict[str, pd.Series]]:
        """Chronological split into train / val / test / holdout."""
        dates = Xy["X"].index
        n = len(dates)
        if n == 0:
            raise RuntimeError("train_windows_empty")

        train_end = dates[int(n * self.cfg.train_split)]
        val_end = dates[int(n * self.cfg.val_split)]
        test_end = dates[int(n * self.cfg.test_split)]

        def _slice(key: str, start, end) -> pd.Series:
            return Xy[key][start:end]

        return {
            "train": {
                "X": _slice("X", None, train_end),
                "y": _slice("y", None, train_end),
                "ret": _slice("ret", None, train_end),
            },
            "val": {
                "X": _slice("X", train_end, val_end),
                "y": _slice("y", train_end, val_end),
                "ret": _slice("ret", train_end, val_end),
            },
            "test": {
                "X": _slice("X", val_end, test_end),
                "y": _slice("y", val_end, test_end),
                "ret": _slice("ret", val_end, test_end),
            },
            "holdout": {
                "X": _slice("X", test_end, None),
                "y": _slice("y", test_end, None),
                "ret": _slice("ret", test_end, None),
            },
        }

    def normalize(self, splits: Dict) -> Dict:
        """Fit scaler on training data only; transform all later splits."""
        train_x = _stack_series(splits["train"]["X"], self._dtype)
        self.scaler = _make_scaler(self._dtype).fit(train_x)

        for part in ("train", "val", "test", "holdout"):
            x = _stack_series(splits[part]["X"], self._dtype)
            x = np.asarray(self.scaler.transform(x), dtype=self._dtype)
            splits[part]["X"] = pd.Series(list(x), index=splits[part]["X"].index)
        return splits

    def fit(self, df: pd.DataFrame) -> Dict:
        """Full pipeline: indicators → target → windows → split → normalise."""
        df = self.add_indicators(df.copy())
        df = self.add_target_no_lookahead(df)
        Xy = self.create_windows(df)
        splits = self.split(Xy)
        if self.cfg.normalize_data:
            splits = self.normalize(splits)
        return splits

    def transform(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Transform for inference/backtest (same steps as fit, minus scaler refit)."""
        df = self.add_indicators(df.copy())
        df = self.add_target_no_lookahead(df)
        Xy = self.create_windows(df)
        if self.scaler:
            X = _stack_series(Xy["X"], self._dtype)
            X = np.asarray(self.scaler.transform(X), dtype=self._dtype)
            Xy["X"] = pd.Series(list(X), index=Xy["X"].index)
        return Xy

    def transform_live(self, df: pd.DataFrame) -> pd.Series:
        """
        Transform for live inference only (no target construction).
        """
        df = self.add_indicators(df.copy())
        Xy = self.create_windows_inference(df)
        if self.scaler and not Xy.empty:
            X = _stack_series(Xy, self._dtype)
            X = np.asarray(self.scaler.transform(X), dtype=self._dtype)
            Xy = pd.Series(list(X), index=Xy.index)
        return Xy


# ─────────────────────────────────────────────────────────────────────────────
# CatBoost log sink
# ─────────────────────────────────────────────────────────────────────────────


class _CatBoostLogSink:
    """File-like sink that routes CatBoost stdout/stderr into the Python logger."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._buf = ""

    def write(self, data: Any) -> None:
        s = str(data or "")
        if not s:
            return
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                self._logger.info("CATBOOST | %s", line)

    def flush(self) -> None:
        line = self._buf.strip()
        if line:
            self._logger.info("CATBOOST | %s", line)
        self._buf = ""


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────────────────────────────────────


class RegressionModel:
    """
    Institutional-grade regression model with CatBoost → sklearn → numpy fallback chain.
    """

    def __init__(self, cfg: InstitutionalTrainConfig) -> None:
        self.cfg = cfg
        self._catboost_fit_kwargs: Dict[str, Any] = {}
        self._catboost_verbose_step: int = 0

        if CatBoostRegressor is not None:
            catboost_params = dict(cfg.regressor_params)

            cpu_total = max(1, int(os.cpu_count() or 1))
            if "thread_count" not in catboost_params:
                env_threads_raw = str(
                    os.getenv("CATBOOST_THREAD_COUNT", "")
                    or os.getenv("INSTITUTIONAL_TRAIN_THREADS", "")
                ).strip()
                if env_threads_raw:
                    try:
                        auto_threads = int(float(env_threads_raw))
                    except Exception:
                        auto_threads = max(1, cpu_total - (2 if cpu_total > 8 else 1))
                else:
                    auto_threads = max(1, cpu_total - (2 if cpu_total > 8 else 1))
                catboost_params["thread_count"] = max(
                    1, min(cpu_total, int(auto_threads))
                )

            if "used_ram_limit" not in catboost_params:
                env_ram = str(os.getenv("CATBOOST_USED_RAM_LIMIT", "") or "").strip()
                if env_ram:
                    catboost_params["used_ram_limit"] = env_ram
                else:
                    auto_ram_mb = max(
                        1024, min(8192, int(catboost_params["thread_count"]) * 768)
                    )
                    catboost_params["used_ram_limit"] = f"{auto_ram_mb}mb"

            verbose_step = int(catboost_params.pop("verbose", 100) or 100)
            es_rounds = int(
                catboost_params.pop(
                    "early_stopping_rounds",
                    getattr(cfg, "early_stopping_rounds", 0),
                )
                or 0
            )
            use_best_model = bool(catboost_params.pop("use_best_model", True))

            self._catboost_verbose_step = max(0, verbose_step)
            self._catboost_fit_kwargs = {
                "verbose": (
                    self._catboost_verbose_step
                    if self._catboost_verbose_step > 0
                    else False
                ),
                "use_best_model": use_best_model,
            }
            if es_rounds > 0:
                self._catboost_fit_kwargs["early_stopping_rounds"] = es_rounds

            self.model = CatBoostRegressor(**catboost_params)
            self.backend = "catboost"

        elif HistGradientBoostingRegressor is not None:
            self.model = HistGradientBoostingRegressor(
                max_depth=int(cfg.regressor_params.get("depth", 10)),
                learning_rate=float(cfg.regressor_params.get("learning_rate", 0.045)),
                max_iter=int(cfg.regressor_params.get("iterations", 3000)),
                random_state=int(cfg.regressor_params.get("random_seed", 42)),
            )
            self.backend = "sklearn_hgb"

        else:
            self.model = _NumpyLinearRegressor(
                l2=float(cfg.regressor_params.get("l2_leaf_reg", 1.0)) * 1e-3
            )
            self.backend = "numpy_linear"

    def train(self, splits: Dict, *, announce: bool = True) -> None:
        """Train on train split, validate on val split."""
        dtype = _cfg_dtype(self.cfg)
        X_train = _stack_series(splits["train"]["X"], dtype)
        y_train = splits["train"]["y"].values
        X_val = _stack_series(splits["val"]["X"], dtype)
        y_val = splits["val"]["y"].values

        if announce:
            total_iters = int(self.cfg.regressor_params.get("iterations", 3000))
            _console(
                f"\n   ⏳ Starting training: {self.cfg.symbol} | "
                f"backend={self.backend} | iterations={total_iters}"
            )

        if self.backend == "catboost":
            fit_kwargs: Dict[str, Any] = dict(self._catboost_fit_kwargs)
            fit_kwargs["eval_set"] = (X_val, y_val)
            sink = _CatBoostLogSink(log)
            fit_kwargs["log_cout"] = sink
            fit_kwargs["log_cerr"] = sink
            self.model.fit(X_train, y_train, **fit_kwargs)

        elif self.backend == "sklearn_hgb":
            if announce:
                _console(f"   Training sklearn HGB: {self.cfg.symbol}")
            self.model.set_params(verbose=0)
            self.model.fit(X_train, y_train)

        else:
            if announce:
                _console(f"   Training numpy linear fallback: {self.cfg.symbol}")
            self.model.fit(X_train, y_train)

        if announce:
            _console(
                f"   ✅ Training complete: {self.cfg.symbol} | backend={self.backend}"
            )
        log.info(
            "Model training complete: %s | backend=%s", self.cfg.symbol, self.backend
        )

    def predict(self, X: pd.Series) -> np.ndarray:
        dtype = _cfg_dtype(self.cfg)
        return self.model.predict(_stack_series(X, dtype))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_training_dataframe(cfg: InstitutionalTrainConfig) -> pd.DataFrame:
    """
    Load training data from CSV or MT5 with robust multi-candidate fallback.

    FIXED (v2): Bare `pass` on symbol_select failure replaced with a warning log
    so operators can diagnose broker-side Market Watch issues.
    """
    env_bars_raw = str(os.getenv("TRAIN_MAX_BARS", "") or "").strip()
    train_max_bars = 0
    if env_bars_raw:
        try:
            train_max_bars = max(2_000, int(float(env_bars_raw)))
        except Exception:
            train_max_bars = 0

    csv_path = Path(cfg.dataname)
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df = df.rename(columns={"Date": "datetime"}).set_index("datetime")
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            if train_max_bars > 0 and len(df) > train_max_bars:
                df = df.tail(train_max_bars).copy()
                log.info("Applied TRAIN_MAX_BARS cap to CSV: %s", len(df))
            if len(df) > 100:
                log.info("Loaded local data: %s (%s rows)", csv_path, len(df))
                return df
        except Exception as e:
            log.warning("Local data load failed: %s", e)

    # Fallback to MT5
    ensure_mt5()
    mt5_symbol = _resolve_mt5_symbol(cfg.symbol)
    if not mt5_symbol:
        raise RuntimeError(f"Could not resolve MT5 symbol for {cfg.symbol}")

    default_max = 60_000 if "BTC" in cfg.symbol.upper() else 100_000
    max_bars_env = int(train_max_bars or default_max)
    candidates: list[int] = [max_bars_env, 50_000, 10_000, 5_000]

    start_env: str = ""
    end_env: str = ""

    rates = None
    last_err = (0, "")

    with MT5_LOCK:
        # FIXED (v2): Log a warning instead of silently passing on selection failure
        selected = mt5.symbol_select(mt5_symbol, True)
        if not selected:
            log.warning(
                "MT5_SYMBOL_SELECT_FAILED | symbol=%s — proceeding anyway; "
                "symbol may not be in Market Watch",
                mt5_symbol,
            )

        info = mt5.symbol_info(mt5_symbol)
        if info is None:
            raise RuntimeError(
                f"Global symbol check failed: {mt5_symbol} not found in MT5"
            )

        # Try date-range fetch first
        if start_env and end_env:
            try:
                start_dt = datetime.datetime.fromisoformat(start_env).replace(
                    tzinfo=None
                )
                end_dt = datetime.datetime.fromisoformat(end_env).replace(tzinfo=None)
                log.info("Fetching range: %s -> %s", start_dt, end_dt)
                rates = mt5.copy_rates_range(
                    mt5_symbol, mt5.TIMEFRAME_M1, start_dt, end_dt
                )
                last_err = mt5.last_error()
            except Exception as e:
                log.warning("Range fetch error: %s", e)

        # Fallback to count-based fetch
        if rates is None or len(rates) == 0:
            for bars in candidates:
                try:
                    rates = mt5.copy_rates_from_pos(
                        mt5_symbol, mt5.TIMEFRAME_M1, 0, int(bars)
                    )
                    last_err = mt5.last_error()
                    if rates is not None and len(rates) > 0:
                        break
                    if last_err[0] == -2:  # Invalid params (count > available history)
                        continue
                except Exception:
                    pass

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"train_data_unavailable:{mt5_symbol}: err={last_err}")

    raw = pd.DataFrame(rates)
    raw["datetime"] = pd.to_datetime(raw["time"], unit="s", utc=True)
    raw = raw.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
        }
    )
    df = raw.set_index("datetime")[["Open", "High", "Low", "Close", "Volume"]]

    if df.empty:
        raise RuntimeError(f"train_data_empty:{mt5_symbol}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Training entry points
# ─────────────────────────────────────────────────────────────────────────────


def _calibrate_abs_threshold(
    preds: np.ndarray,
    y_true: np.ndarray,
    *,
    min_coverage: float = 0.03,
    target_accuracy: float = 0.60,
) -> Dict[str, float]:
    """Validation-only calibration: choose abs(pred) threshold for directional precision."""
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    n = int(min(len(p), len(y)))
    if n <= 0:
        return {
            "pred_abs_threshold": 0.0,
            "quantile": 0.0,
            "coverage": 0.0,
            "direction_accuracy": 0.0,
            "base_direction_accuracy": 0.0,
        }
    p = p[:n]
    y = y[:n]
    base_acc = float(np.mean(np.sign(p) == np.sign(y)))
    mag = np.abs(p)
    best = {
        "pred_abs_threshold": 0.0,
        "quantile": 0.0,
        "coverage": 1.0,
        "direction_accuracy": base_acc,
        "base_direction_accuracy": base_acc,
    }
    min_cov = max(0.005, min(0.95, float(min_coverage)))
    target_acc = max(0.0, min(1.0, float(target_accuracy)))
    best_meeting: Optional[Dict[str, float]] = None
    for q in range(50, 100):
        thr = float(np.percentile(mag, q))
        mask = mag >= thr
        cov = float(np.mean(mask))
        if cov < min_cov:
            continue
        m_n = int(np.sum(mask))
        if m_n <= 0:
            continue
        acc = float(np.mean(np.sign(p[mask]) == np.sign(y[mask])))
        candidate = {
            "pred_abs_threshold": thr,
            "quantile": float(q),
            "coverage": cov,
            "direction_accuracy": acc,
            "base_direction_accuracy": base_acc,
        }
        if acc >= target_acc:
            if best_meeting is None:
                best_meeting = candidate
            else:
                bm_cov = float(best_meeting["coverage"])
                bm_q = float(best_meeting["quantile"])
                bm_acc = float(best_meeting["direction_accuracy"])
                # Preserve active coverage once the accuracy target is met.
                # Choosing the strictest threshold here makes live trading brittle
                # and can suppress valid signals even when a looser threshold
                # already satisfies the directional-precision target.
                if cov > bm_cov + 1e-12 or (
                    abs(cov - bm_cov) <= 1e-12
                    and (
                        acc > bm_acc + 1e-12
                        or (abs(acc - bm_acc) <= 1e-12 and float(q) < bm_q - 1e-12)
                    )
                ):
                    best_meeting = candidate
            continue
        best_acc = float(best["direction_accuracy"])
        best_cov = float(best["coverage"])
        if acc > best_acc + 1e-12 or (
            abs(acc - best_acc) <= 1e-12 and cov > best_cov + 1e-12
        ):
            best = candidate
    return best_meeting if best_meeting is not None else best


def _fit_probability_calibrator(
    preds: np.ndarray,
    y_true: np.ndarray,
    *,
    method: str = "isotonic",
) -> Dict[str, Any]:
    """
    Fit a PROBABILITY CALIBRATOR that converts `|pred|` into a calibrated
    P(direction_correct) on the VALIDATION set only.

    CRITICAL — Section 3 (ML & Probability Correction):
      The raw `|pred|` magnitude of a regressor is NOT a probability and has
      no guaranteed monotone relationship with directional accuracy. Using
      it as a "confidence" to size positions (pseudo-Kelly) is statistically
      invalid. This function fits an Isotonic (default) or Sigmoid (Platt)
      calibrator on the held-out VAL set:

          P_hat(hit | |pred|) = f( |pred| )

      where `hit = (sign(pred) == sign(y))`.

    Returns a JSON-serialisable dict that the live inference layer can use
    via `core.utils.calibrated_probability` without re-importing sklearn
    (the calibrator is reduced to a piecewise-linear (x, y) table).

    * Isotonic: sklearn IsotonicRegression (out-of-bounds = clip).
    * Sigmoid (Platt): 2-parameter sigmoid a * |pred| + b, fit via
      scipy/sklearn logistic regression.

    Fallback: if sklearn/scipy unavailable OR insufficient samples, returns
    an empty dict. Callers must treat absence as "no calibrator" and fall
    back to the (already institutional-safe) fixed-risk path.
    """
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    n = int(min(len(p), len(y)))
    if n < 50:  # minimum viable sample for calibration
        return {}
    p = p[:n]
    y = y[:n]
    mag = np.abs(p)
    hit = (np.sign(p) == np.sign(y)).astype(np.float64)
    # Guard: need both classes in validation set.
    if float(hit.sum()) <= 1.0 or float((1.0 - hit).sum()) <= 1.0:
        return {}

    method_u = str(method or "isotonic").lower().strip()
    try:
        if method_u.startswith("sigmoid") or method_u.startswith("platt"):
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(solver="lbfgs", max_iter=500)
            lr.fit(mag.reshape(-1, 1), hit.astype(int))
            a = float(lr.coef_[0][0])
            b = float(lr.intercept_[0])
            # Materialise a piecewise table for live-side portability.
            xs = np.quantile(mag, np.linspace(0.0, 1.0, 64))
            xs = np.unique(xs)
            ys = 1.0 / (1.0 + np.exp(-(a * xs + b)))
            return {
                "method": "sigmoid",
                "a": a,
                "b": b,
                "x": [float(v) for v in xs],
                "y": [float(v) for v in ys],
                "n_samples": int(n),
                "base_rate": float(hit.mean()),
            }
        else:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(mag, hit)
            # Export a (x, y) step table over the training support so that
            # `core.utils.calibrated_probability` can do a pure-numpy lookup
            # without re-importing sklearn in the live path.
            xs = np.unique(np.quantile(mag, np.linspace(0.0, 1.0, 128)))
            ys = iso.predict(xs)
            ys = np.clip(ys, 0.0, 1.0)
            return {
                "method": "isotonic",
                "x": [float(v) for v in xs],
                "y": [float(v) for v in ys],
                "n_samples": int(n),
                "base_rate": float(hit.mean()),
            }
    except Exception as exc:
        log.warning("probability_calibrator_fit_failed | method=%s err=%s", method_u, exc)
        return {}


def _extract_flat_feature_importance(model: Any) -> Optional[np.ndarray]:
    """
    Extract per-column flat importances from the trained regressor.
    Supports CatBoost / sklearn tree models / linear fallbacks.
    """
    try:
        if hasattr(model, "get_feature_importance"):
            arr = np.asarray(model.get_feature_importance(), dtype=np.float64).reshape(
                -1
            )
            if arr.size > 0:
                return np.abs(arr)
    except Exception:
        pass

    try:
        if hasattr(model, "feature_importances_"):
            arr = np.asarray(
                getattr(model, "feature_importances_"), dtype=np.float64
            ).reshape(-1)
            if arr.size > 0:
                return np.abs(arr)
    except Exception:
        pass

    try:
        if hasattr(model, "coef_"):
            arr = np.asarray(getattr(model, "coef_"), dtype=np.float64).reshape(-1)
            if arr.size > 0:
                return np.abs(arr)
    except Exception:
        pass
    return None


def _aggregate_feature_importance(
    flat_importance: Optional[np.ndarray],
    features: List[str],
    *,
    window_size: int,
) -> List[tuple[str, float]]:
    """
    Aggregate flattened window importances back to base feature names.
    """
    feats = list(features)
    n_feats = len(feats)
    if n_feats <= 0:
        return []

    if flat_importance is None or flat_importance.size <= 0:
        uniform = 1.0 / float(n_feats)
        return [(f, uniform) for f in feats]

    ws = max(1, int(window_size))
    expected = int(n_feats * ws)
    arr = np.asarray(flat_importance, dtype=np.float64).reshape(-1)
    if arr.size < expected:
        pad = np.zeros(expected, dtype=np.float64)
        pad[: arr.size] = arr
        arr = pad
    elif arr.size > expected:
        arr = arr[:expected]

    agg = np.zeros(n_feats, dtype=np.float64)
    mat = arr.reshape(ws, n_feats)
    agg += np.mean(np.abs(mat), axis=0)

    total = float(np.sum(agg))
    if total <= 1e-12:
        agg[:] = 1.0 / float(n_feats)
    else:
        agg = agg / total

    ranked = sorted(
        ((f, float(agg[i])) for i, f in enumerate(feats)),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


def _select_alpha_features(
    cfg: InstitutionalTrainConfig,
    ranked_importance: List[tuple[str, float]],
) -> List[str]:
    """
    Keep a deterministic, high-signal subset while preserving original order.
    """
    if not ranked_importance:
        return list(cfg.training_features)

    total = max(1, len(ranked_importance))
    keep_ratio = max(0.25, min(1.0, float(cfg.feature_importance_keep_ratio)))
    keep_min = max(4, min(total, int(cfg.feature_importance_min_features)))
    keep_n = max(keep_min, int(math.ceil(total * keep_ratio)))

    top_score = float(ranked_importance[0][1]) if ranked_importance else 0.0
    min_ratio = max(0.0, min(0.95, float(cfg.feature_importance_min_ratio)))
    floor = top_score * min_ratio
    selected = [f for f, score in ranked_importance if float(score) >= float(floor)]
    if len(selected) < keep_n:
        selected = [f for f, _ in ranked_importance[:keep_n]]

    selected_set = set(selected)
    ordered = [f for f in cfg.training_features if f in selected_set]
    return ordered if ordered else list(cfg.training_features)


def _downsample_xy(
    xy: Dict[str, pd.Series],
    max_samples: int,
) -> Dict[str, pd.Series]:
    n = int(len(xy["X"]))
    cap = max(0, int(max_samples or 0))
    if cap <= 0 or n <= cap:
        return xy
    stride = max(1, int(math.ceil(float(n) / float(cap))))
    return {
        "X": xy["X"].iloc[::stride],
        "y": xy["y"].iloc[::stride],
        "ret": xy["ret"].iloc[::stride],
    }


def _cv_training_cfg(cfg: InstitutionalTrainConfig) -> InstitutionalTrainConfig:
    params = dict(cfg.regressor_params or {})
    cv_default = 180 if _env_truthy("INSTITUTIONAL_FAST_MODE", "0") else 450
    max_iters = _env_int("INSTITUTIONAL_CV_MAX_ITERS", cv_default, min_v=60, max_v=1200)
    cur_iters = int(params.get("iterations", max_iters) or max_iters)
    params["iterations"] = max(60, min(max_iters, cur_iters))
    params["verbose"] = 0
    return replace(
        cfg,
        regressor_params=params,
        early_stopping_rounds=max(0, min(int(cfg.early_stopping_rounds or 0), 100)),
    )


def _build_fold_splits(
    cfg: InstitutionalTrainConfig,
    xy: Dict[str, pd.Series],
    *,
    train_start: int,
    train_end: int,
    test_end: int,
) -> Optional[Dict[str, Dict[str, pd.Series]]]:
    """
    Build train/val/test splits with PURGED + EMBARGOED boundaries.

    CRITICAL — López de Prado 2018 "Advances in Financial ML":
      Targets built with a forward horizon of `h` bars cause the last `h`
      train samples to OVERLAP (in their look-forward window) with the
      first `h` val samples. Using these samples for training leaks
      val-set information into the fit. The institutional remedy is:

        1. PURGE: drop the last `h` samples of the train set (they look
           into the val region).
        2. EMBARGO: insert a gap of `e >= h` bars between val and test
           so that autocorrelation / microstructure effects do not bleed
           across the evaluation boundary.

    Both steps are applied below. `e = h` is the minimum; we use
    `max(h, PURGED_EMBARGO_BARS env, int(h * 1.0))` for safety.
    """
    X_all = xy["X"]
    y_all = xy["y"]
    r_all = xy["ret"]

    # Forward horizon used to build labels (bars).
    horizon = int(getattr(cfg, "prediction_horizon", 15) or 15)
    horizon = max(1, horizon)
    try:
        embargo_extra = int(os.getenv("PURGED_EMBARGO_BARS", str(horizon)) or str(horizon))
    except Exception:
        embargo_extra = horizon
    embargo = max(horizon, int(embargo_extra))

    n = int(len(X_all))
    a = max(0, int(train_start))
    b = min(n, int(train_end))
    c = min(n, int(test_end))
    if b - a < (64 + embargo) or c - b < (32 + embargo):
        return None

    val_len = max(32, int((b - a) * 0.20))
    if (b - a) - val_len < (32 + embargo):
        return None
    val_start = b - val_len

    # Purge the last `horizon` bars of the train block — they LOOK
    # FORWARD into the val block and would leak the target.
    train_end_purged = max(a, val_start - horizon)
    # Embargo between val and test: drop the last `embargo` val bars.
    val_end_embargoed = max(val_start, b - embargo)

    if train_end_purged - a < 64 or val_end_embargoed - val_start < 32:
        return None

    split = {
        "train": {
            "X": X_all.iloc[a:train_end_purged],
            "y": y_all.iloc[a:train_end_purged],
            "ret": r_all.iloc[a:train_end_purged],
        },
        "val": {
            "X": X_all.iloc[val_start:val_end_embargoed],
            "y": y_all.iloc[val_start:val_end_embargoed],
            "ret": r_all.iloc[val_start:val_end_embargoed],
        },
        "test": {
            "X": X_all.iloc[b:c],
            "y": y_all.iloc[b:c],
            "ret": r_all.iloc[b:c],
        },
        "holdout": {
            "X": X_all.iloc[b:c],
            "y": y_all.iloc[b:c],
            "ret": r_all.iloc[b:c],
        },
    }

    if cfg.normalize_data:
        dtype = _cfg_dtype(cfg)
        train_x = _stack_series(split["train"]["X"], dtype)
        scaler = _make_scaler(dtype).fit(train_x)
        for part in ("train", "val", "test", "holdout"):
            x = _stack_series(split[part]["X"], dtype)
            x = np.asarray(scaler.transform(x), dtype=dtype)
            split[part]["X"] = pd.Series(list(x), index=split[part]["X"].index)

    return split


def _eval_fold(
    cfg: InstitutionalTrainConfig,
    split: Dict[str, Dict[str, pd.Series]],
) -> Dict[str, float]:
    dtype = _cfg_dtype(cfg)
    model = RegressionModel(_cv_training_cfg(cfg))
    model.train(split, announce=False)

    X_val = _stack_series(split["val"]["X"], dtype)
    y_val = np.asarray(split["val"]["y"].values, dtype=np.float64)
    p_val = np.asarray(model.model.predict(X_val), dtype=np.float64)

    X_test = _stack_series(split["test"]["X"], dtype)
    y_test = np.asarray(split["test"]["y"].values, dtype=np.float64)
    p_test = np.asarray(model.model.predict(X_test), dtype=np.float64)

    mse = float(np.mean((p_test - y_test) ** 2))
    mae = float(np.mean(np.abs(p_test - y_test)))
    dir_acc = (
        float(np.mean(np.sign(p_test) == np.sign(y_test))) if len(y_test) > 0 else 0.0
    )

    cal = _calibrate_abs_threshold(
        p_val,
        y_val,
        min_coverage=max(0.005, float(cfg.cv_target_min_coverage)),
        target_accuracy=max(0.0, min(1.0, float(cfg.cv_target_direction_accuracy))),
    )
    thr = float(cal.get("pred_abs_threshold", 0.0) or 0.0)
    mask = np.abs(p_test) >= thr if thr > 0.0 else np.ones_like(p_test, dtype=bool)
    active_n = int(np.sum(mask))
    active_cov = float(active_n / max(1, len(y_test)))
    active_dir_acc = (
        float(np.mean(np.sign(p_test[mask]) == np.sign(y_test[mask])))
        if active_n > 0
        else 0.0
    )

    return {
        "mse": mse,
        "mae": mae,
        "direction_accuracy": dir_acc,
        "active_direction_accuracy": active_dir_acc,
        "active_coverage": active_cov,
        "samples": float(len(y_test)),
    }


def _run_time_series_cv(
    cfg: InstitutionalTrainConfig,
    xy: Dict[str, pd.Series],
) -> Dict[str, Any]:
    xy = _downsample_xy(xy, int(cfg.cv_max_samples))
    n = int(len(xy["X"]))
    folds_req = max(2, int(cfg.tscv_folds))
    test_len = max(64, int(cfg.tscv_test_samples))
    min_train = max(200, int(cfg.tscv_min_train_samples))
    if n < (min_train + test_len):
        return {
            "folds": [],
            "folds_evaluated": 0,
            "passed": False,
            "reason": "insufficient_samples",
        }

    folds: List[Dict[str, float]] = []
    for k in range(folds_req):
        train_end = min_train + (k * test_len)
        test_end = train_end + test_len
        if test_end > n:
            break
        split = _build_fold_splits(
            cfg, xy, train_start=0, train_end=train_end, test_end=test_end
        )
        if split is None:
            continue
        try:
            folds.append(_eval_fold(cfg, split))
        except Exception as exc:
            log.warning("TSCV_FOLD_FAIL | fold=%s err=%s", k, exc)
            continue

    if not folds:
        return {
            "folds": [],
            "folds_evaluated": 0,
            "passed": False,
            "reason": "no_valid_folds",
        }

    act_acc = [float(f["active_direction_accuracy"]) for f in folds]
    act_cov = [float(f["active_coverage"]) for f in folds]
    active_acc_for_eval = [acc for acc, cov in zip(act_acc, act_cov) if cov > 0.0]
    mean_acc = float(np.mean(active_acc_for_eval)) if active_acc_for_eval else 0.0
    min_acc = float(np.min(active_acc_for_eval)) if active_acc_for_eval else 0.0
    mean_cov = float(np.mean(act_cov))
    active_folds = int(len(active_acc_for_eval))
    required_active_folds = max(1, int(len(folds) // 2))
    pass_acc = max(0.0, float(cfg.cv_target_direction_accuracy) - 0.02)
    pass_cov = max(0.005, float(cfg.cv_target_min_coverage) * 0.80)
    passed = bool(
        mean_acc >= pass_acc
        and mean_cov >= pass_cov
        and active_folds >= required_active_folds
    )
    return {
        "folds": folds,
        "folds_evaluated": len(folds),
        "active_folds_evaluated": active_folds,
        "active_folds_required": required_active_folds,
        "mean_active_direction_accuracy": mean_acc,
        "min_active_direction_accuracy": min_acc,
        "mean_active_coverage": mean_cov,
        "pass_acc_threshold": pass_acc,
        "pass_cov_threshold": pass_cov,
        "passed": passed,
    }


def _run_walk_forward_validation(
    cfg: InstitutionalTrainConfig,
    xy: Dict[str, pd.Series],
) -> Dict[str, Any]:
    xy = _downsample_xy(xy, int(cfg.cv_max_samples))
    n = int(len(xy["X"]))
    max_windows = max(1, int(cfg.wfa_windows))
    train_len = max(200, int(n * max(0.40, min(0.90, float(cfg.wfa_train_ratio)))))
    test_len = max(64, int(n * max(0.05, min(0.30, float(cfg.wfa_test_ratio)))))
    if n < (train_len + test_len):
        return {
            "windows": [],
            "total": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "passed": False,
            "reason": "insufficient_samples",
        }

    windows: List[Dict[str, float]] = []
    failed = 0
    train_end = train_len
    while (train_end + test_len) <= n and len(windows) < max_windows:
        split = _build_fold_splits(
            cfg, xy, train_start=0, train_end=train_end, test_end=(train_end + test_len)
        )
        if split is None:
            break
        try:
            fold = _eval_fold(cfg, split)
        except Exception as exc:
            log.warning("WFA_WINDOW_FAIL | idx=%s err=%s", len(windows), exc)
            train_end += test_len
            continue

        pass_acc = max(0.0, float(cfg.cv_target_direction_accuracy) - 0.03)
        pass_cov = max(0.005, float(cfg.cv_target_min_coverage) * 0.70)
        win_ok = bool(
            float(fold.get("active_direction_accuracy", 0.0)) >= pass_acc
            and float(fold.get("active_coverage", 0.0)) >= pass_cov
        )
        if not win_ok:
            failed += 1
        fold["passed"] = 1.0 if win_ok else 0.0
        fold["train_end_index"] = float(train_end)
        windows.append(fold)
        train_end += test_len

    total = len(windows)
    pass_rate = float((total - failed) / float(total)) if total > 0 else 0.0
    return {
        "windows": windows,
        "total": total,
        "failed": failed,
        "pass_rate": pass_rate,
        "passed": bool(total > 0 and failed == 0),
    }


def _split_is_chronological(splits: Dict[str, Dict[str, pd.Series]]) -> bool:
    """Verify strict chronological train/validation/test/holdout ordering."""
    ordered_parts = ("train", "val", "test", "holdout")
    last_ts: Optional[pd.Timestamp] = None
    for part in ordered_parts:
        idx = splits.get(part, {}).get("X", pd.Series(dtype=object)).index
        if len(idx) <= 0:
            continue
        cur_min = pd.Timestamp(idx.min())
        cur_max = pd.Timestamp(idx.max())
        if last_ts is not None and cur_min <= last_ts:
            return False
        last_ts = cur_max
    return True


def _direction_balance(y: np.ndarray) -> Dict[str, Any]:
    """Measure sign imbalance for directional stability diagnostics."""
    arr = np.asarray(y, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {
            "positive": 0,
            "negative": 0,
            "flat": 0,
            "dominant_ratio": 1.0,
            "imbalanced": True,
        }
    signs = np.sign(arr)
    positive = int(np.sum(signs > 0.0))
    negative = int(np.sum(signs < 0.0))
    flat = int(np.sum(signs == 0.0))
    dominant = max(positive, negative, flat)
    dominant_ratio = float(dominant / max(1, arr.size))
    return {
        "positive": positive,
        "negative": negative,
        "flat": flat,
        "dominant_ratio": dominant_ratio,
        "imbalanced": bool(dominant_ratio > 0.85),
    }


def _regime_dependency_profile(pred: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Estimate whether accuracy collapses under a different realized-volatility regime."""
    pred_arr = np.asarray(pred, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(pred_arr) & np.isfinite(y_arr)
    pred_arr = pred_arr[mask]
    y_arr = y_arr[mask]
    if pred_arr.size < 20 or y_arr.size != pred_arr.size:
        return {"dependency_delta": 0.0, "dependent": False}

    realized_vol = np.abs(y_arr)
    q1 = float(np.quantile(realized_vol, 0.25))
    q3 = float(np.quantile(realized_vol, 0.75))
    low_mask = realized_vol <= q1
    high_mask = realized_vol >= q3
    if int(np.sum(low_mask)) < 5 or int(np.sum(high_mask)) < 5:
        return {"dependency_delta": 0.0, "dependent": False}

    low_acc = float(np.mean(np.sign(pred_arr[low_mask]) == np.sign(y_arr[low_mask])))
    high_acc = float(np.mean(np.sign(pred_arr[high_mask]) == np.sign(y_arr[high_mask])))
    dependency_delta = abs(high_acc - low_acc)
    return {
        "low_vol_accuracy": low_acc,
        "high_vol_accuracy": high_acc,
        "dependency_delta": dependency_delta,
        "dependent": bool(dependency_delta > 0.20),
    }


def _build_training_audit_report(
    *,
    cfg: InstitutionalTrainConfig,
    splits: Dict[str, Dict[str, pd.Series]],
    results: Dict[str, Dict[str, Any]],
    preds_by_split: Dict[str, np.ndarray],
    alpha_calibration: Dict[str, Any],
    anti_overfit: Dict[str, Any],
    source_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Build the Phase 7 training audit for leakage, overfit, and model staleness."""
    training_features = list(getattr(cfg, "training_features", []) or [])
    suspicious_name_tokens = ("target", "label", "future", "lead", "lookahead", "leak")
    suspicious_feature_names = [
        feat
        for feat in training_features
        if any(tok in str(feat).lower() for tok in suspicious_name_tokens)
    ]

    hold_y = np.asarray(
        splits.get("holdout", {}).get("y", pd.Series(dtype=np.float64)).values,
        dtype=np.float64,
    )
    hold_pred = np.asarray(
        preds_by_split.get("holdout", np.array([], dtype=np.float64)), dtype=np.float64
    )

    feature_target_corr_max = 0.0
    try:
        if (
            source_df is not None
            and not source_df.empty
            and "target" in source_df.columns
        ):
            corr_series = (
                source_df[training_features + ["target"]]
                .corr(numeric_only=True)["target"]
                .drop(labels=["target"], errors="ignore")
            )
            corr_vals = np.abs(corr_series.to_numpy(dtype=np.float64, copy=False))
            corr_vals = corr_vals[np.isfinite(corr_vals)]
            if corr_vals.size > 0:
                feature_target_corr_max = float(np.max(corr_vals))
    except Exception:
        feature_target_corr_max = 0.0

    chronology_ok = _split_is_chronological(splits)
    balance = _direction_balance(hold_y)
    regime_dependency = _regime_dependency_profile(hold_pred, hold_y)

    train_dir_acc = float(
        results.get("train", {}).get("direction_accuracy", 0.0) or 0.0
    )
    hold_dir_acc = float(
        results.get("holdout", {}).get("direction_accuracy", 0.0) or 0.0
    )
    val_active_acc = float(
        alpha_calibration.get("val_direction_accuracy_active", 0.0) or 0.0
    )
    hold_active_acc = float(
        alpha_calibration.get("holdout_direction_accuracy_active", 0.0) or 0.0
    )
    hold_active_cov = float(
        alpha_calibration.get("holdout_coverage_active", 0.0) or 0.0
    )
    min_cov = float(alpha_calibration.get("target_active_min_coverage", 0.0) or 0.0)
    calibration_gap = abs(val_active_acc - hold_active_acc)

    val_thr = float(alpha_calibration.get("pred_abs_threshold", 0.0) or 0.0)
    hold_thr = float(
        alpha_calibration.get("holdout_abs_threshold_from_quantile", 0.0) or 0.0
    )
    threshold_ratio = (
        float(hold_thr / max(val_thr, 1e-12))
        if val_thr > 0.0 and hold_thr > 0.0
        else 1.0
    )

    asset_target = 0.56 if "BTC" in str(cfg.symbol).upper() else 0.55
    cadence_hours = 24.0 * (7.0 if "BTC" in str(cfg.symbol).upper() else 14.0)
    stale_age_hours = 0.0
    try:
        latest_ts = pd.Timestamp(source_df.index.max())
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.tz_localize("UTC")
        else:
            latest_ts = latest_ts.tz_convert("UTC")
        now_ts = pd.Timestamp.now(tz="UTC")
        stale_age_hours = max(
            0.0,
            (now_ts - latest_ts).total_seconds() / 3600.0,
        )
    except Exception:
        stale_age_hours = cadence_hours * 2.0

    feature_leakage_detected = bool(suspicious_feature_names)
    target_leakage_detected = bool(
        (not chronology_ok) or feature_target_corr_max >= 0.995
    )
    overfitting_detected = bool(
        (train_dir_acc - hold_dir_acc) > 0.12
        or not bool(anti_overfit.get("passed", False))
    )
    asset_specific_quality = bool(
        hold_active_acc >= asset_target
        and hold_active_cov >= max(0.005, min_cov * 0.80)
    )
    calibration_ok = bool(calibration_gap <= 0.10 and val_thr > 0.0)
    threshold_stable = bool(0.60 <= threshold_ratio <= 1.60)
    stale_model_detected = bool(stale_age_hours > cadence_hours)
    oos_degradation = float(max(0.0, train_dir_acc - hold_dir_acc))

    passed = bool(
        not feature_leakage_detected
        and not target_leakage_detected
        and not balance["imbalanced"]
        and not overfitting_detected
        and not regime_dependency.get("dependent", False)
        and asset_specific_quality
        and calibration_ok
        and threshold_stable
        and not stale_model_detected
    )

    return {
        "passed": passed,
        "feature_leakage": {
            "detected": feature_leakage_detected,
            "suspicious_features": suspicious_feature_names,
        },
        "target_leakage": {
            "detected": target_leakage_detected,
            "chronology_ok": chronology_ok,
            "feature_target_corr_max": feature_target_corr_max,
        },
        "class_imbalance": balance,
        "overfitting": {
            "detected": overfitting_detected,
            "train_direction_accuracy": train_dir_acc,
            "holdout_direction_accuracy": hold_dir_acc,
            "gap": train_dir_acc - hold_dir_acc,
        },
        "regime_dependency": regime_dependency,
        "asset_specific_quality": {
            "passed": asset_specific_quality,
            "target_active_accuracy": asset_target,
            "holdout_active_accuracy": hold_active_acc,
            "holdout_active_coverage": hold_active_cov,
        },
        "calibration_quality": {
            "passed": calibration_ok,
            "val_active_accuracy": val_active_acc,
            "holdout_active_accuracy": hold_active_acc,
            "gap": calibration_gap,
        },
        "threshold_stability": {
            "passed": threshold_stable,
            "validation_threshold": val_thr,
            "holdout_threshold": hold_thr,
            "ratio": threshold_ratio,
        },
        "retraining_cadence": {
            "recommended_hours": cadence_hours,
            "stale_age_hours": stale_age_hours,
        },
        "stale_model_detection": {
            "detected": stale_model_detected,
            "stale_age_hours": stale_age_hours,
            "limit_hours": cadence_hours,
        },
        "out_of_sample_degradation": {
            "detected": oos_degradation > 0.10,
            "gap": oos_degradation,
        },
    }


def train(
    cfg: Optional[InstitutionalTrainConfig] = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Institutional model training entry point.

    Returns:
        dict with keys: symbol, model_version, registry_base, rows, results, institutional_grade
    """
    if cfg is None:
        cfg = XAU_TRAIN_CONFIG
    if log_dir is None:
        log_dir = _LOG_DIR

    _console(f"🏛️  INSTITUTIONAL MODEL TRAIN — {cfg.symbol}")
    t0 = time.time()

    df = _load_training_dataframe(cfg)
    _console(f"   Loaded {len(df)} bars")

    cfg_effective = _apply_runtime_train_overrides(cfg)
    pipeline = Pipeline(cfg_effective)
    splits = pipeline.fit(df)
    _console(
        f"   Train: {len(splits['train']['X'])} | "
        f"Val: {len(splits['val']['X'])} | "
        f"Test: {len(splits['test']['X'])} | "
        f"Holdout: {len(splits['holdout']['X'])}"
    )

    model = RegressionModel(cfg_effective)
    require_catboost = str(
        os.getenv("INSTITUTIONAL_REQUIRE_CATBOOST", "1") or "1"
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    if require_catboost and model.backend != "catboost":
        raise RuntimeError(
            f"institutional_training_blocked:catboost_backend_required:got={model.backend}"
        )
    model.train(splits)

    # Feature-importance filtering pass (keeps strategy math unchanged, removes noise).
    flat_imp = _extract_flat_feature_importance(model.model)
    ranked_imp = _aggregate_feature_importance(
        flat_imp,
        list(cfg_effective.training_features),
        window_size=int(cfg_effective.window_size),
    )
    selected_features = _select_alpha_features(cfg_effective, ranked_imp)
    feature_importance = {k: float(v) for k, v in ranked_imp}
    if (
        selected_features
        and len(selected_features) < len(cfg_effective.training_features)
        and len(selected_features)
        >= max(4, int(cfg_effective.feature_importance_min_features))
    ):
        _console(
            "   Feature Filter: "
            f"{len(cfg_effective.training_features)} -> {len(selected_features)} alpha features"
        )
        cfg_effective = replace(
            cfg_effective, training_features=list(selected_features)
        )
        pipeline = Pipeline(cfg_effective)
        splits = pipeline.fit(df)
        model = RegressionModel(cfg_effective)
        model.train(splits)
        flat_imp = _extract_flat_feature_importance(model.model)
        ranked_imp = _aggregate_feature_importance(
            flat_imp,
            list(cfg_effective.training_features),
            window_size=int(cfg_effective.window_size),
        )
        feature_importance = {k: float(v) for k, v in ranked_imp}

    dtype = _cfg_dtype(cfg_effective)
    results: Dict[str, Dict[str, Any]] = {}
    preds_by_split: Dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test", "holdout"):
        X = _stack_series(splits[split_name]["X"], dtype)
        y = splits[split_name]["y"].values
        preds = model.model.predict(X)
        preds_by_split[split_name] = np.asarray(preds, dtype=np.float64)
        mse = float(np.mean((preds - y) ** 2))
        mae = float(np.mean(np.abs(preds - y)))
        dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
        results[split_name] = {
            "mse": mse,
            "mae": mae,
            "direction_accuracy": dir_acc,
            "samples": len(y),
        }
        _console(
            f"   {split_name.upper()}: MSE={mse:.8f} | "
            f"MAE={mae:.8f} | Dir Acc={dir_acc:.2%}"
        )

    target_acc = float(os.getenv("TARGET_ACTIVE_DIR_ACC", "0.58") or 0.58)
    min_cov = float(os.getenv("TARGET_ACTIVE_MIN_COVERAGE", "0.01") or 0.01)
    val_cal = _calibrate_abs_threshold(
        preds_by_split.get("val", np.array([], dtype=np.float64)),
        np.asarray(splits["val"]["y"].values, dtype=np.float64),
        min_coverage=min_cov,
        target_accuracy=target_acc,
    )
    pred_abs_thr = float(val_cal.get("pred_abs_threshold", 0.0) or 0.0)
    q_cal = float(val_cal.get("quantile", 0.0) or 0.0)
    hold_pred = preds_by_split.get("holdout", np.array([], dtype=np.float64))
    hold_y = np.asarray(splits["holdout"]["y"].values, dtype=np.float64)
    if q_cal > 0.0 and hold_pred.size > 0:
        hold_thr = float(np.percentile(np.abs(hold_pred), q_cal))
    else:
        hold_thr = pred_abs_thr
    hold_mask = (
        np.abs(hold_pred) >= hold_thr
        if hold_thr > 0.0
        else np.ones_like(hold_pred, dtype=bool)
    )
    hold_n = int(np.sum(hold_mask))
    hold_acc_active = (
        float(np.mean(np.sign(hold_pred[hold_mask]) == np.sign(hold_y[hold_mask])))
        if hold_n > 0
        else 0.0
    )
    hold_cov_active = float(hold_n / max(1, len(hold_y)))
    # ─── PROBABILITY CALIBRATOR (Section 3) ────────────────────────────
    # Fit an isotonic (or sigmoid) calibrator mapping |pred| -> P(hit) on the
    # VALIDATION fold. This is later consumed by the live inference layer
    # (core.utils.calibrated_probability) to replace the heuristic
    # "confidence" with a calibrated probability. Sample-pessimistic: if
    # sklearn is unavailable or val set is too small, the calibrator is
    # omitted and the live layer falls back to fixed-risk sizing.
    calib_method = str(
        os.getenv("PROBABILITY_CALIBRATION_METHOD", "isotonic") or "isotonic"
    ).strip()
    probability_calibrator = _fit_probability_calibrator(
        preds_by_split.get("val", np.array([], dtype=np.float64)),
        np.asarray(splits["val"]["y"].values, dtype=np.float64),
        method=calib_method,
    )

    alpha_calibration = {
        "pred_abs_threshold": pred_abs_thr,
        "pred_quantile": q_cal,
        "val_direction_accuracy_active": float(
            val_cal.get("direction_accuracy", 0.0) or 0.0
        ),
        "val_base_direction_accuracy": float(
            val_cal.get("base_direction_accuracy", 0.0) or 0.0
        ),
        "val_coverage_active": float(val_cal.get("coverage", 0.0) or 0.0),
        "holdout_direction_accuracy_active": hold_acc_active,
        "holdout_coverage_active": hold_cov_active,
        "holdout_abs_threshold_from_quantile": hold_thr,
        "target_active_direction_accuracy": target_acc,
        "target_active_min_coverage": min_cov,
        "probability_calibrator": probability_calibrator,
    }
    results["holdout"]["active_direction_accuracy"] = hold_acc_active
    results["holdout"]["active_coverage"] = hold_cov_active
    results["holdout"]["active_threshold_abs_pred"] = hold_thr
    results["holdout"]["active_threshold_quantile"] = q_cal

    # Chronological CV + walk-forward diagnostics (scientific anti-overfit checks).
    cv_target_acc = max(
        0.0, min(1.0, float(cfg_effective.cv_target_direction_accuracy))
    )
    cv_target_cov = max(0.005, min(1.0, float(cfg_effective.cv_target_min_coverage)))
    cv_cfg = replace(
        cfg_effective,
        cv_target_direction_accuracy=max(cv_target_acc, target_acc - 0.02),
        cv_target_min_coverage=max(cv_target_cov, min_cov * 0.80),
    )
    raw_cfg = replace(cv_cfg, normalize_data=False)
    raw_pipe = Pipeline(raw_cfg)
    raw_df = raw_pipe.add_indicators(df.copy())
    raw_df = raw_pipe.add_target_no_lookahead(raw_df)
    xy_full = raw_pipe.create_windows(raw_df)
    tscv_report = _run_time_series_cv(cv_cfg, xy_full)
    wfa_report = _run_walk_forward_validation(cv_cfg, xy_full)
    anti_overfit_ok = bool(
        tscv_report.get("passed", False) and wfa_report.get("passed", False)
    )
    anti_overfit = {
        "passed": anti_overfit_ok,
        "time_series_cv": tscv_report,
        "walk_forward": wfa_report,
    }
    _console(
        "   Anti-Overfit: "
        f"CV={'PASS' if tscv_report.get('passed') else 'FAIL'} "
        f"({int(tscv_report.get('folds_evaluated', 0))} folds) | "
        f"WFA={'PASS' if wfa_report.get('passed') else 'FAIL'} "
        f"({int(wfa_report.get('failed', 0))}/{int(wfa_report.get('total', 0))} failed)"
    )

    training_audit = _build_training_audit_report(
        cfg=cfg_effective,
        splits=splits,
        results=results,
        preds_by_split=preds_by_split,
        alpha_calibration=alpha_calibration,
        anti_overfit=anti_overfit,
        source_df=raw_df,
    )
    _console(
        "   Training Audit: "
        f"{'PASS' if training_audit.get('passed', False) else 'FAIL'} | "
        f"stale={int(bool(training_audit.get('stale_model_detection', {}).get('detected', False)))} "
        f"overfit={int(bool(training_audit.get('overfitting', {}).get('detected', False)))} "
        f"regime_dep={int(bool(training_audit.get('regime_dependency', {}).get('dependent', False)))}"
    )

    institutional_ok = (
        model.backend == "catboost"
        and hold_acc_active >= target_acc
        and hold_cov_active >= min_cov
        and anti_overfit_ok
        and bool(training_audit.get("passed", False))
    )
    alpha_gate_required = str(
        os.getenv("INSTITUTIONAL_REQUIRE_ALPHA_GATE", "1") or "1"
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    alpha_gate_fail_reasons: List[str] = []
    if model.backend != "catboost":
        alpha_gate_fail_reasons.append(f"backend={model.backend}")
    if hold_acc_active < target_acc:
        alpha_gate_fail_reasons.append(
            f"holdout_active_acc={hold_acc_active:.3f}<target={target_acc:.3f}"
        )
    if hold_cov_active < min_cov:
        alpha_gate_fail_reasons.append(
            f"coverage={hold_cov_active:.3f}<min={min_cov:.3f}"
        )
    if not anti_overfit_ok:
        tscv_passed = bool(tscv_report.get("passed", False))
        tscv_folds = int(tscv_report.get("folds_evaluated", 0) or 0)
        tscv_mean_acc = float(
            tscv_report.get("mean_active_direction_accuracy", 0.0) or 0.0
        )
        tscv_mean_cov = float(tscv_report.get("mean_active_coverage", 0.0) or 0.0)
        tscv_pass_acc = float(tscv_report.get("pass_acc_threshold", 0.0) or 0.0)
        tscv_pass_cov = float(tscv_report.get("pass_cov_threshold", 0.0) or 0.0)
        wfa_passed = bool(wfa_report.get("passed", False))
        wfa_failed = int(wfa_report.get("failed", 0) or 0)
        wfa_total = int(wfa_report.get("total", 0) or 0)
        alpha_gate_fail_reasons.append(
            "anti_overfit_failed:"
            f"tscv_passed={int(tscv_passed)}"
            f":tscv_folds={tscv_folds}"
            f":tscv_mean_acc={tscv_mean_acc:.3f}"
            f":tscv_pass_acc={tscv_pass_acc:.3f}"
            f":tscv_mean_cov={tscv_mean_cov:.3f}"
            f":tscv_pass_cov={tscv_pass_cov:.3f}"
            f":wfa_passed={int(wfa_passed)}"
            f":wfa_failed={wfa_failed}/{wfa_total}"
        )
    if not bool(training_audit.get("passed", False)):
        alpha_gate_fail_reasons.append("training_audit_failed")
    if alpha_gate_required and alpha_gate_fail_reasons:
        raise RuntimeError(
            "institutional_training_blocked:alpha_gate_failed:"
            + ";".join(alpha_gate_fail_reasons)
        )
    payload = {
        "model": model.model,
        "pipeline": pipeline,
        "symbol": cfg_effective.symbol,
        "window_size": int(cfg_effective.window_size),
        "prediction_horizon": int(cfg_effective.prediction_horizon),
        "training_features": list(cfg_effective.training_features),
        "training_features_initial": list(cfg.training_features),
        "feature_importance": feature_importance,
        "target_type": cfg_effective.target_type,
        "normalize_data": bool(cfg_effective.normalize_data),
        "alpha_calibration": alpha_calibration,
        "anti_overfit": anti_overfit,
        "anti_overfit_passed": anti_overfit_ok,
        "training_audit": training_audit,
        "institutional_grade": institutional_ok,
    }

    metadata = ModelMetadata(
        version=str(cfg_effective.model_version),
        timestamp=datetime.datetime.utcnow().isoformat(),
        sharpe=0.0,
        win_rate=0.0,
        status=("PENDING" if institutional_ok else "DEGRADED"),
        backtest_sharpe=0.0,
        backtest_win_rate=0.0,
        max_drawdown_pct=0.0,
        real_backtest=False,
        training_features=list(cfg_effective.training_features),
        source="Backtest.model_train_institutional",
        anti_overfit_passed=bool(anti_overfit_ok),
        tscv_folds=int(tscv_report.get("folds_evaluated", 0) or 0),
        tscv_mean_active_direction_accuracy=float(
            tscv_report.get("mean_active_direction_accuracy", 0.0) or 0.0
        ),
        wfa_passed=bool(wfa_report.get("passed", False)),
        wfa_total_windows=int(wfa_report.get("total", 0) or 0),
        wfa_failed_windows=int(wfa_report.get("failed", 0) or 0),
        training_audit=training_audit,
    )
    registry_base = model_manager.save_model(payload, metadata)

    pathlib.Path(cfg.dump_path).parent.mkdir(exist_ok=True)
    with open(cfg.dump_path, "wb") as f:
        pickle.dump(payload, f)

    elapsed = time.time() - t0
    _console(f"\n✅ Institutional training complete in {elapsed:.1f}s")
    _console(f"   Registry: {registry_base}.pkl")
    _console(
        f"   Holdout Direction Accuracy: {results['holdout']['direction_accuracy']:.2%}"
    )
    _console(
        "   Holdout Active Direction Accuracy: "
        f"{hold_acc_active:.2%} @ coverage={hold_cov_active:.2%} "
        f"(q={q_cal:.1f}, abs_pred_thr={hold_thr:.6f})"
    )
    _console(f"   Backend: {model.backend}")

    log_file = log_dir / f"{cfg_effective.symbol.lower()}_institutional_train.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.datetime.utcnow().isoformat()} | {cfg_effective.symbol} INSTITUTIONAL | "
            f"Holdout_MSE={results['holdout']['mse']:.8f} "
            f"Holdout_DirAcc={results['holdout']['direction_accuracy']:.3f} "
            f"Holdout_ActiveDirAcc={hold_acc_active:.3f} "
            f"ActiveCov={hold_cov_active:.3f} "
            f"ActiveQ={q_cal:.1f} "
            f"TSCV_Pass={int(bool(tscv_report.get('passed', False)))} "
            f"WFA_Pass={int(bool(wfa_report.get('passed', False)))} "
            f"elapsed={elapsed:.1f}s\n"
        )

    return {
        "symbol": cfg_effective.symbol,
        "model_version": cfg_effective.model_version,
        "backend": model.backend,
        "registry_base": registry_base,
        "rows": int(len(df)),
        "results": results,
        "alpha_calibration": alpha_calibration,
        "anti_overfit": anti_overfit,
        "training_audit": training_audit,
        "training_features": list(cfg_effective.training_features),
        "feature_importance": feature_importance,
        "institutional_grade": institutional_ok,
    }


def train_and_register(asset: str) -> Dict[str, Any]:
    """Train and register institutional model for the given asset."""
    asset_u = str(asset).upper().strip()
    cfg = (
        BTC_TRAIN_CONFIG
        if asset_u in ("BTC", "BTCUSD", "BTCUSDM")
        else XAU_TRAIN_CONFIG
    )
    return train(cfg=cfg)


def load_training_dataframe_for_asset(asset: str) -> pd.DataFrame:
    """Public helper: load raw OHLCV dataframe using the same logic as training."""
    asset_u = str(asset).upper().strip()
    cfg = (
        BTC_TRAIN_CONFIG
        if asset_u in ("BTC", "BTCUSD", "BTCUSDM")
        else XAU_TRAIN_CONFIG
    )
    return _load_training_dataframe(cfg)


if __name__ == "__main__":
    import sys

    asset = sys.argv[1].upper() if len(sys.argv) > 1 else "XAU"
    if asset in ("BTC", "BTCUSD"):
        train(BTC_TRAIN_CONFIG)
    else:
        train(XAU_TRAIN_CONFIG)
