# Backtest/model_train_institutional.py
# Institutional-grade training with ZERO look-ahead bias and advanced features
from __future__ import annotations

import datetime
import math
import logging
import os
import pathlib
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field
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

from core.model_manager import ModelMetadata, model_manager
from log_config import LOG_DIR, get_artifact_dir, get_artifact_path, get_log_path
from mt5_client import MT5_LOCK, ensure_mt5

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


class _NumpyStandardScaler:
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
    """Minimal numpy-only linear regressor fallback when external ML libs are unavailable."""

    def __init__(self, l2: float = 1e-6) -> None:
        self.l2 = float(max(l2, 0.0))
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_NumpyLinearRegressor":
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if x_arr.ndim != 2 or y_arr.ndim != 1:
            raise RuntimeError("numpy_linear_invalid_shape")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise RuntimeError("numpy_linear_len_mismatch")
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
    def fit(self, x: np.ndarray) -> "ScalerProtocol":
        ...

    def transform(self, x: np.ndarray) -> np.ndarray:
        ...


if _SKStandardScaler is None:
    StandardScaler = _NumpyStandardScaler
else:
    StandardScaler = _SKStandardScaler


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


@dataclass
class InstitutionalTrainConfig:
    """Institutional training configuration with advanced features"""
    symbol: str = "XAUUSD"
    mean_price_indc: str = "Close"
    
    # Target prediction (ZERO look-ahead)
    target_type: str = "returns_magnitude"  # "returns_direction" is classifier-oriented
    prediction_horizon: int = 15  # Bars ahead to predict
    percent_increase: float = field(
        default_factory=lambda: float(os.getenv("TRAIN_SIGNAL_THRESHOLD_XAU", "0.0008") or "0.0008")
    )
    
    # Features - expanded for institutional use
    window_size: int = 20
    cci_period: int = 20
    mom_period: int = 10
    roc_period: int = 10
    stoch_k_period: int = 14
    stoch_slowk_period: int = 3
    stoch_slowd_period: int = 3
    stoch_matype: int = 0
    training_features: List[str] = field(default_factory=lambda: [
        # Trend indicators
        "ma4", "ma9", "ma13", "ma21", "ma50",
        "ema5", "ema12", "ema26", "ema50",
        "macd", "macd_signal", "macd_hist",
        
        # Momentum
        "rsi13", "rsi21",
        "ppo13_5", "ppo27_7",
        "momentum_10", "momentum_20",
        "mom", "roc", "cci", "stoch_k", "stoch_d",
        
        # Volatility
        "atr_14", "bbands_width", "keltner_width",
        "historical_vol_20",
        
        # Volume
        "adi", "obv", "volume_sma_20",
        
        # Microstructure
        "high_low_ratio", "close_position_in_range",
        "price_acceleration",
    ])
    
    # Model parameters - institutional grade
    regressor_params: Dict[str, Any] = field(default_factory=lambda: {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": int(os.getenv("TRAIN_VERBOSE_STEP", "0") or "0"),
        "task_type": "CPU",
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "train_dir": str(_ART_CATBOOST),
    })
    early_stopping_rounds: int = field(
        default_factory=lambda: int(os.getenv("TRAIN_EARLY_STOPPING_ROUNDS", "0") or "0")
    )
    
    # Data splits - institutional: train/val/test/holdout
    train_split: float = 0.60
    val_split: float = 0.75
    test_split: float = 0.90  # 10% pure holdout
    
    normalize_data: bool = True
    float_dtype: str = field(default_factory=lambda: str(os.getenv("TRAIN_DTYPE", "float32") or "float32"))
    max_window_samples: int = field(default_factory=lambda: int(os.getenv("TRAIN_MAX_SAMPLES", "50000") or "50000"))
    model_version: str = "1.0_xau_institutional"
    dataname: str = "data/XAUUSD_1m.csv"
    dump_path: str = str(get_artifact_path("dumps", "xau_model_institutional.pkl"))


BTC_TRAIN_CONFIG = InstitutionalTrainConfig(
    symbol="BTCUSD",
    prediction_horizon=12,
    percent_increase=float(os.getenv("TRAIN_SIGNAL_THRESHOLD_BTC", "0.0012") or "0.0012"),
    window_size=30,
    cci_period=30,
    mom_period=14,
    roc_period=14,
    stoch_k_period=14,
    stoch_slowk_period=5,
    stoch_slowd_period=5,
    training_features=[
        "ma4", "ma9", "ma13", "ma21", "ma50", "ma100",
        "ema5", "ema12", "ema26", "ema50", "ema100",
        "macd", "macd_signal", "macd_hist",
        "rsi13", "rsi21",
        "ppo13_5", "ppo27_7", "ppo48_12",
        "momentum_10", "momentum_20",
        "mom", "roc", "cci", "stoch_k", "stoch_d",
        "atr_14", "bbands_width", "keltner_width",
        "historical_vol_20", "historical_vol_50",
        "adi", "obv", "volume_sma_20", "volume_sma_50",
        "high_low_ratio", "close_position_in_range",
        "price_acceleration", "volume_acceleration",
    ],
    regressor_params={
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": int(os.getenv("TRAIN_VERBOSE_STEP", "0") or "0"),
        "task_type": "CPU",
        "loss_function": "RMSE",
        "train_dir": str(_ART_CATBOOST),
    },
    early_stopping_rounds=int(os.getenv("TRAIN_EARLY_STOPPING_ROUNDS", "0") or "0"),
    dataname="data/BTCUSD_1m.csv",
    dump_path=str(get_artifact_path("dumps", "btc_model_institutional.pkl")),
    model_version="1.0_btc_institutional",
)

XAU_TRAIN_CONFIG = InstitutionalTrainConfig()

_MT5_SYMBOL_BY_BASE = {
    "XAUUSD": "XAUUSDm",
    "BTCUSD": "BTCUSDm",
}


def _resolve_mt5_symbol(symbol: str) -> str:
    base = str(symbol).upper().strip()
    if base in ("BTC", "BTCUSD", "BTCUSDM"):
        base = "BTCUSD"
    elif base in ("XAU", "XAUUSD", "XAUUSDM"):
        base = "XAUUSD"

    suffix = str(os.getenv("MT5_SYMBOL_SUFFIX", "") or "").strip()
    candidates = []
    if suffix:
        candidates.append(f"{base}{suffix}")
    candidates.extend([f"{base}m", base, _MT5_SYMBOL_BY_BASE.get(base, base)])

    seen: set[str] = set()
    uniq: list[str] = []
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
    # Fallback: scan available symbols for base match
    try:
        symbols = mt5.symbols_get()
    except Exception:
        symbols = None
    if symbols:
        base_u = base.upper()
        names = [s.name for s in symbols if hasattr(s, "name")]
        # prefer exact/startswith matches
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


class Pipeline:
    """
    Institutional pipeline with:
    - ZERO look-ahead bias
    - Advanced technical indicators
    - Microstructure features
    - Multi-timeframe analysis
    """
    
    def __init__(self, cfg: InstitutionalTrainConfig) -> None:
        self.cfg = cfg
        self.scaler: Optional[ScalerProtocol] = None
        self._dtype = _cfg_dtype(cfg)
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add institutional-grade indicators with ZERO look-ahead"""
        close = pd.to_numeric(df["Close"], errors="coerce")
        high = pd.to_numeric(df["High"], errors="coerce")
        low = pd.to_numeric(df["Low"], errors="coerce")
        volume = pd.to_numeric(df["Volume"], errors="coerce")

        close_np = close.to_numpy(dtype=np.float64)
        high_np = high.to_numpy(dtype=np.float64)
        low_np = low.to_numpy(dtype=np.float64)
        volume_np = volume.to_numpy(dtype=np.float64)

        def _to_series(arr: np.ndarray) -> pd.Series:
            return pd.Series(arr, index=df.index)

        stoch_cache: Optional[tuple[np.ndarray, np.ndarray]] = None
        
        for feat in self.cfg.training_features:
            # Moving averages
            if feat.startswith("ma") and not feat.startswith("macd"):
                n = int(feat[2:])
                df[feat] = _to_series(talib.SMA(close_np, timeperiod=n))
            
            # Exponential moving averages
            elif feat.startswith("ema"):
                n = int(feat[3:])
                df[feat] = _to_series(talib.EMA(close_np, timeperiod=n))
            
            # MACD
            elif feat == "macd":
                macd_line, macd_signal, macd_hist = talib.MACD(
                    close_np, fastperiod=12, slowperiod=26, signalperiod=9
                )
                df["macd"] = _to_series(macd_line)
                df["macd_signal"] = _to_series(macd_signal)
                df["macd_hist"] = _to_series(macd_hist)
            
            # PPO
            elif feat.startswith("ppo"):
                p, s = map(int, feat[3:].split("_"))
                df[feat] = _to_series(talib.PPO(close_np, fastperiod=p, slowperiod=s, matype=0))
            
            # RSI
            elif feat.startswith("rsi"):
                n = int(feat[3:])
                df[feat] = _to_series(talib.RSI(close_np, timeperiod=n))
            
            # Momentum
            elif feat.startswith("momentum_"):
                n = int(feat.split("_")[1])
                df[feat] = _to_series(talib.ROC(close_np, timeperiod=n))

            elif feat == "mom":
                df[feat] = _to_series(talib.MOM(close_np, timeperiod=int(self.cfg.mom_period)))

            elif feat == "roc":
                df[feat] = _to_series(talib.ROC(close_np, timeperiod=int(self.cfg.roc_period)))

            elif feat == "cci":
                df[feat] = _to_series(talib.CCI(high_np, low_np, close_np, timeperiod=int(self.cfg.cci_period)))

            elif feat in ("stoch_k", "stoch_d"):
                if stoch_cache is None:
                    stoch_k, stoch_d = talib.STOCH(
                        high_np, low_np, close_np,
                        fastk_period=int(self.cfg.stoch_k_period),
                        slowk_period=int(self.cfg.stoch_slowk_period),
                        slowk_matype=int(self.cfg.stoch_matype),
                        slowd_period=int(self.cfg.stoch_slowd_period),
                        slowd_matype=int(self.cfg.stoch_matype),
                    )
                    stoch_cache = (stoch_k, stoch_d)
                if feat == "stoch_k":
                    df[feat] = _to_series(stoch_cache[0])
                else:
                    df[feat] = _to_series(stoch_cache[1])
            
            # ATR
            elif feat.startswith("atr_"):
                n = int(feat.split("_")[1])
                df[feat] = _to_series(talib.ATR(high_np, low_np, close_np, timeperiod=n))
            
            # Bollinger Bands Width
            elif feat == "bbands_width":
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close_np, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
                )
                width = np.where(bb_middle != 0, (bb_upper - bb_lower) / bb_middle, 0.0)
                df[feat] = _to_series(width)
            
            # Keltner Channel Width
            elif feat == "keltner_width":
                mid = talib.EMA(close_np, timeperiod=20)
                atr = talib.ATR(high_np, low_np, close_np, timeperiod=20)
                upper = mid + 2.0 * atr
                lower = mid - 2.0 * atr
                width = np.where(close_np != 0, (upper - lower) / close_np, 0.0)
                df[feat] = _to_series(width)
            
            # Historical Volatility
            elif feat.startswith("historical_vol_"):
                n = int(feat.split("_")[2])
                returns = close.pct_change()
                df[feat] = returns.rolling(window=n).std() * np.sqrt(252)
            
            # Volume indicators
            elif feat == "adi":
                df[feat] = _to_series(talib.AD(high_np, low_np, close_np, volume_np))
            
            elif feat == "obv":
                df[feat] = _to_series(talib.OBV(close_np, volume_np))
            
            elif feat.startswith("volume_sma_"):
                n = int(feat.split("_")[2])
                df[feat] = _to_series(talib.SMA(volume_np, timeperiod=n))
            
            # Microstructure features
            elif feat == "high_low_ratio":
                df[feat] = (high / low - 1.0) * 100
            
            elif feat == "close_position_in_range":
                df[feat] = ((close - low) / (high - low).replace(0.0, np.nan)).fillna(0.5)
            
            elif feat == "price_acceleration":
                returns = close.pct_change()
                df[feat] = returns.diff()
            
            elif feat == "volume_acceleration":
                vol_change = volume.pct_change()
                df[feat] = vol_change.diff()
        
        df.dropna(inplace=True)
        return df
    
    def add_target_no_lookahead(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target WITHOUT look-ahead bias.
        
        CRITICAL: We create a target at time t that predicts t+horizon.
        During training, we ensure that:
        - Features at time t use only data up to t
        - Target at time t uses data at t+horizon (which is future from t)
        
        During inference/live:
        - We have features up to time t
        - We predict t+horizon (but don't know actual yet)
        - We evaluate later when t+horizon arrives
        """
        prices = df[self.cfg.mean_price_indc].values
        horizon = self.cfg.prediction_horizon
        
        # Calculate forward returns
        # target[i] = (price[i+horizon] - price[i]) / price[i]
        # This is the TRUE future return we're trying to predict
        targets = np.full(len(prices), np.nan)
        
        valid_idx = len(prices) - horizon
        if valid_idx > 0:
            future_prices = prices[horizon:]
            current_prices = prices[:valid_idx]
            targets[:valid_idx] = (future_prices - current_prices) / current_prices
        
        df["target_return"] = targets
        
        # For classification variant
        if self.cfg.target_type == "returns_direction":
            df["target_direction"] = np.sign(df["target_return"])
            df["target"] = df["target_direction"]
        else:
            df["target"] = df["target_return"]
        
        # Drop rows where we don't have future data
        df.dropna(subset=["target"], inplace=True)
        
        return df
    
    def create_windows(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create time-series windows"""
        X = df[self.cfg.training_features].to_numpy(dtype=self._dtype, copy=False)
        y = df["target"].to_numpy(dtype=self._dtype, copy=False)
        target_ret = df["target_return"].to_numpy(dtype=self._dtype, copy=False)
        ws = self.cfg.window_size
        total = len(X) - ws + 1
        if total <= 0:
            return {
                "X": pd.Series([], dtype=object),
                "y": pd.Series([], dtype=self._dtype),
                "ret": pd.Series([], dtype=self._dtype),
            }

        max_samples = int(self.cfg.max_window_samples or 0)
        stride = 1
        if max_samples > 0 and total > max_samples:
            stride = int(math.ceil(total / max_samples))
            log.info(
                "Window sampling reduced | total=%s max=%s stride=%s",
                total,
                max_samples,
                stride,
            )
        
        Xw, yw, rw, idx = [], [], [], []
        for i in range(ws - 1, len(X), stride):
            Xw.append(X[i - ws + 1 : i + 1].ravel())
            yw.append(y[i])
            rw.append(target_ret[i])
            idx.append(i)
        
        dates = df.index[idx]
        return {
            "X": pd.Series(list(Xw), index=dates),
            "y": pd.Series(yw, index=dates),
            "ret": pd.Series(rw, index=dates),
        }
    
    def split(self, Xy: Dict[str, pd.Series]) -> Dict[str, Dict[str, pd.Series]]:
        """Split: train / val / test / holdout"""
        dates = Xy["X"].index
        n = len(dates)
        if n == 0:
            raise RuntimeError("train_windows_empty")
        
        train_end = dates[int(n * self.cfg.train_split)]
        val_end = dates[int(n * self.cfg.val_split)]
        test_end = dates[int(n * self.cfg.test_split)]
        
        return {
            "train": {
                "X": Xy["X"][:train_end],
                "y": Xy["y"][:train_end],
                "ret": Xy["ret"][:train_end],
            },
            "val": {
                "X": Xy["X"][train_end:val_end],
                "y": Xy["y"][train_end:val_end],
                "ret": Xy["ret"][train_end:val_end],
            },
            "test": {
                "X": Xy["X"][val_end:test_end],
                "y": Xy["y"][val_end:test_end],
                "ret": Xy["ret"][val_end:test_end],
            },
            "holdout": {
                "X": Xy["X"][test_end:],
                "y": Xy["y"][test_end:],
                "ret": Xy["ret"][test_end:],
            }
        }
    
    def normalize(self, splits: Dict) -> Dict:
        """Normalize using train+val only (test/holdout unseen)"""
        train_x = _stack_series(splits["train"]["X"], self._dtype)
        val_x = _stack_series(splits["val"]["X"], self._dtype)
        
        self.scaler = _make_scaler(self._dtype).fit(np.concatenate([train_x, val_x]))
        
        for part in ["train", "val", "test", "holdout"]:
            x = _stack_series(splits[part]["X"], self._dtype)
            x = np.asarray(self.scaler.transform(x), dtype=self._dtype)
            splits[part]["X"] = pd.Series(
                list(x),
                index=splits[part]["X"].index
            )
        
        return splits
    
    def fit(self, df: pd.DataFrame) -> Dict:
        """Full pipeline: indicators → target (no lookahead) → windows → split → normalize"""
        df = self.add_indicators(df.copy())
        df = self.add_target_no_lookahead(df)
        Xy = self.create_windows(df)
        splits = self.split(Xy)
        
        if self.cfg.normalize_data:
            splits = self.normalize(splits)
        
        return splits
    
    def transform(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Transform for inference (used in backtest)"""
        df = self.add_indicators(df.copy())
        df = self.add_target_no_lookahead(df)
        Xy = self.create_windows(df)
        
        if self.scaler:
            X = _stack_series(Xy["X"], self._dtype)
            X = np.asarray(self.scaler.transform(X), dtype=self._dtype)
            Xy["X"] = pd.Series(
                list(X),
                index=Xy["X"].index
            )
        
        return Xy


def _console(msg: str) -> None:
    """Route progress messages to dedicated backtest training log."""
    txt = str(msg).rstrip()
    if not txt:
        return
    for line in txt.splitlines():
        line = line.rstrip()
        if line:
            log.info(line)
    # Optional local echo for debugging sessions.
    if str(os.getenv("BACKTEST_ECHO_CONSOLE", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}:
        try:
            real_out = getattr(sys, "__stdout__", None) or sys.stdout
            real_out.write(txt + "\n")
            real_out.flush()
        except Exception:
            pass


class _CatBoostLogSink:
    """File-like sink to capture CatBoost stdout/stderr into logger."""

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


class RegressionModel:
    """Institutional-grade regression model with ensemble capability"""
    
    def __init__(self, cfg: InstitutionalTrainConfig) -> None:
        self.cfg = cfg
        self._catboost_fit_kwargs: Dict[str, Any] = {}
        self._catboost_verbose_step: int = 0

        if CatBoostRegressor is not None:
            # Keep fit-only knobs out of constructor kwargs.
            catboost_params = dict(cfg.regressor_params)
            verbose_step = int(catboost_params.pop("verbose", 100) or 100)
            es_rounds_raw = catboost_params.pop(
                "early_stopping_rounds",
                getattr(cfg, "early_stopping_rounds", 0),
            )
            es_rounds = int(es_rounds_raw or 0)
            use_best_model = bool(catboost_params.pop("use_best_model", True))

            self._catboost_verbose_step = max(0, verbose_step)
            self._catboost_fit_kwargs = {
                "verbose": (self._catboost_verbose_step if self._catboost_verbose_step > 0 else False),
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
        """Train with train set, validate on val set"""
        dtype = _cfg_dtype(self.cfg)
        X_train = _stack_series(splits["train"]["X"], dtype)
        y_train = splits["train"]["y"].values
        X_val = _stack_series(splits["val"]["X"], dtype)
        y_val = splits["val"]["y"].values
        
        total_iters = int(self.cfg.regressor_params.get("iterations", 3000))
        if announce:
            _console(f"\n   ⏳ Starting model training: {self.cfg.symbol} | backend={self.backend} | iterations={total_iters}")
            if self.backend == "catboost":
                if self._catboost_verbose_step > 0:
                    _console(
                        f"   CatBoost progress every {self._catboost_verbose_step} iterations"
                        " (logged to backtest_model_train.log)"
                    )
                else:
                    _console("   CatBoost verbose progress disabled (TRAIN_VERBOSE_STEP=0)")

        if self.backend == "catboost":
            fit_kwargs: Dict[str, Any] = dict(self._catboost_fit_kwargs)
            fit_kwargs["eval_set"] = (X_val, y_val)
            sink = _CatBoostLogSink(log)
            fit_kwargs["log_cout"] = sink
            fit_kwargs["log_cerr"] = sink
            self.model.fit(
                X_train, y_train,
                **fit_kwargs,
            )
        elif self.backend == "sklearn_hgb":
            # sklearn HistGradientBoosting
            if announce:
                _console(f"   Training {self.cfg.symbol}: sklearn HGB ({total_iters} max iterations)...")
            self.model.set_params(verbose=0)
            self.model.fit(X_train, y_train)
        else:
            if announce:
                _console(f"   Training {self.cfg.symbol}: numpy linear fallback...")
            self.model.fit(X_train, y_train)
        
        if announce:
            _console(f"   ✅ Model training finished: {self.cfg.symbol} | backend={self.backend}")
        log.info("Model training complete: %s | backend=%s", self.cfg.symbol, self.backend)
    
    def predict(self, X: pd.Series) -> np.ndarray:
        dtype = _cfg_dtype(self.cfg)
        return self.model.predict(_stack_series(X, dtype))


def _load_training_dataframe(cfg: InstitutionalTrainConfig) -> pd.DataFrame:
    """
    Load training data from CSV or MT5 with robust fallbacks and retries.
    """
    csv_path = Path(cfg.dataname)
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df = df.rename(columns={"Date": "datetime"}).set_index("datetime")
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            if len(df) > 100:
                log.info("Loaded local data: %s (%s rows)", csv_path, len(df))
                return df
        except Exception as e:
            log.warning("Local data load failed: %s", e)
    
    # Fallback to MT5
    ensure_mt5()
    mt5_symbol = _resolve_mt5_symbol(cfg.symbol)
    
    # Validation
    if not mt5_symbol:
        raise RuntimeError(f"Could not resolve MT5 symbol for {cfg.symbol}")

    # Configuration for fetch
    default_max = 60000 if "BTC" in cfg.symbol.upper() else 100000
    max_bars_env = int(os.getenv("TRAIN_MAX_BARS", str(default_max)) or str(default_max))
    # Candidates for bar counts: try max, then fallbacks if too heavy/fails
    candidates = [max_bars_env, 50000, 10000, 5000]
    
    start_env = str(os.getenv("TRAIN_START_DATE", "") or "").strip()
    end_env = str(os.getenv("TRAIN_END_DATE", "") or "").strip()
    
    df = pd.DataFrame()

    with MT5_LOCK:
        # 1. Ensure symbol is selected in Market Watch
        if not mt5.symbol_select(mt5_symbol, True):
            # Try to add it explicitly
            pass 
        
        # 2. PROBE: Check if symbol is valid generally
        info = mt5.symbol_info(mt5_symbol)
        if info is None:
            raise RuntimeError(f"Global symbol check failed: {mt5_symbol} not found in MT5")

        rates = None
        last_err = mt5.last_error()
        
        # 3. Try fetching by Date Range first if configured
        if start_env and end_env:
            try:
                start_dt = datetime.datetime.fromisoformat(start_env).replace(tzinfo=None)
                end_dt = datetime.datetime.fromisoformat(end_env).replace(tzinfo=None)
                log.info("Fetching range: %s -> %s", start_dt, end_dt)
                rates = mt5.copy_rates_range(mt5_symbol, mt5.TIMEFRAME_M1, start_dt, end_dt)
                last_err = mt5.last_error()
            except Exception as e:
                log.warning("Range fetch error: %s", e)

        # 4. Fallback to count-based fetch
        if rates is None or len(rates) == 0:
            for bars in candidates:
                # Ensure params are int/native types
                bars_req = int(bars)
                try:
                    rates = mt5.copy_rates_from_pos(mt5_symbol, mt5.TIMEFRAME_M1, 0, bars_req)
                    last_err = mt5.last_error()
                    if rates is not None and len(rates) > 0:
                        break
                    
                    # If invalid params which usually means "start_pos + count" > available history
                    # We continue to lower counts
                    if last_err[0] == -2: # Invalid params
                         continue
                except Exception:
                    pass

    # 5. Final validation
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"train_data_unavailable:{mt5_symbol}: err={last_err}")
    
    raw = pd.DataFrame(rates)
    # MT5 returns time in seconds, convert to UTC datetime
    raw["datetime"] = pd.to_datetime(raw["time"], unit="s", utc=True)
    
    # Rename columns to standard format
    raw = raw.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume"
    })
    
    df = raw.set_index("datetime")[["Open", "High", "Low", "Close", "Volume"]]
    
    if df.empty:
        raise RuntimeError(f"train_data_empty:{mt5_symbol}")
    
    return df

def train(
    cfg: Optional[InstitutionalTrainConfig] = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Institutional model training entry point.
    
    Returns:
        Training results with metrics on all splits.
    """
    if cfg is None:
        cfg = XAU_TRAIN_CONFIG
    if log_dir is None:
        log_dir = _LOG_DIR
    
    _console(f"🏛️  INSTITUTIONAL MODEL TRAIN — {cfg.symbol}")
    t0 = time.time()
    
    # Load data
    df = _load_training_dataframe(cfg)
    _console(f"   Loaded {len(df)} bars")
    
    # Pipeline with ZERO look-ahead
    pipeline = Pipeline(cfg)
    splits = pipeline.fit(df)
    
    _console(f"   Train: {len(splits['train']['X'])} | "
          f"Val: {len(splits['val']['X'])} | "
          f"Test: {len(splits['test']['X'])} | "
          f"Holdout: {len(splits['holdout']['X'])}")
    
    # Train model
    model = RegressionModel(cfg)
    model.train(splits)
    
    # Evaluate on all splits
    dtype = _cfg_dtype(cfg)
    results = {}
    for split_name in ["train", "val", "test", "holdout"]:
        X = _stack_series(splits[split_name]["X"], dtype)
        y = splits[split_name]["y"].values
        preds = model.model.predict(X)
        
        mse = float(np.mean((preds - y) ** 2))
        mae = float(np.mean(np.abs(preds - y)))
        
        # Direction accuracy (for returns)
        direction_acc = float(np.mean((np.sign(preds) == np.sign(y))))
        
        results[split_name] = {
            "mse": mse,
            "mae": mae,
            "direction_accuracy": direction_acc,
            "samples": len(y)
        }
        
        _console(f"   {split_name.upper()}: MSE={mse:.8f} | "
              f"MAE={mae:.8f} | Dir Acc={direction_acc:.2%}")
    
    # Save model payload
    payload = {
        "model": model.model,
        "pipeline": pipeline,
        "symbol": cfg.symbol,
        "window_size": int(cfg.window_size),
        "prediction_horizon": int(cfg.prediction_horizon),
        "training_features": list(cfg.training_features),
        "target_type": cfg.target_type,
        "normalize_data": bool(cfg.normalize_data),
        "institutional_grade": True,
    }
    
    # Save to registry
    metadata = ModelMetadata(
        version=str(cfg.model_version),
        timestamp=datetime.datetime.utcnow().isoformat(),
        sharpe=0.0,
        win_rate=0.0,
        status="PENDING",
        backtest_sharpe=0.0,
        backtest_win_rate=0.0,
        max_drawdown_pct=0.0,
        real_backtest=False,
        training_features=list(cfg.training_features),
        source="Backtest.model_train_institutional",
    )
    
    registry_base = model_manager.save_model(payload, metadata)
    
    # Save legacy dump
    pathlib.Path(cfg.dump_path).parent.mkdir(exist_ok=True)
    with open(cfg.dump_path, "wb") as f:
        pickle.dump(payload, f)
    
    elapsed = time.time() - t0
    _console(f"\n✅ Institutional training complete in {elapsed:.1f}s")
    _console(f"   Registry: {registry_base}.pkl")
    _console(f"   Holdout Direction Accuracy: {results['holdout']['direction_accuracy']:.2%}")
    
    # Write training log
    log_file = log_dir / f"{cfg.symbol.lower()}_institutional_train.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.datetime.utcnow().isoformat()} | {cfg.symbol} INSTITUTIONAL | "
            f"Holdout_MSE={results['holdout']['mse']:.8f} "
            f"Holdout_DirAcc={results['holdout']['direction_accuracy']:.3f} "
            f"elapsed={elapsed:.1f}s\n"
        )
    
    return {
        "symbol": cfg.symbol,
        "model_version": cfg.model_version,
        "registry_base": registry_base,
        "rows": int(len(df)),
        "results": results,
        "institutional_grade": True,
    }


def train_and_register(asset: str) -> Dict[str, Any]:
    """Train and register institutional model"""
    asset_u = str(asset).upper().strip()
    cfg = (
        BTC_TRAIN_CONFIG 
        if asset_u in ("BTC", "BTCUSD", "BTCUSDM") 
        else XAU_TRAIN_CONFIG
    )
    return train(cfg=cfg)


def load_training_dataframe_for_asset(asset: str) -> pd.DataFrame:
    """Public helper for backtest fallback: load dataframe with the same logic as training."""
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
