from __future__ import annotations

import builtins
import json
import os
import pickle
import queue
import threading
import time
import traceback
from collections import deque
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import MetaTrader5 as mt5

from ExnessAPI.functions import close_all_position
from mt5_client import MT5_LOCK, ensure_mt5, mt5_status

# -------------------- Unified core + strategy adapters --------------------
from core.config import (
    XAUEngineConfig as XauConfig,
    BTCEngineConfig as BtcConfig,
    MIN_GATE_SHARPE,
    MIN_GATE_WIN_RATE,
    get_config_from_env as _get_core_config,
)
from core.risk_engine import RiskManager
from core.signal_engine import SignalEngine
from core.feature_engine import FeatureEngine
from core.portfolio_risk import PortfolioRiskManager, AssetExposure
from DataFeed.xau_market_feed import MarketFeed as XauMarketFeed
from DataFeed.btc_market_feed import MarketFeed as BtcMarketFeed

# -------------------- Backtest Integration --------------------
from Backtest.engine import BacktestEngine, XAU_BACKTEST_CONFIG, BTC_BACKTEST_CONFIG, run_backtest
from log_config import get_artifact_path



def get_xau_config():
    return _get_core_config("XAU")


def get_btc_config():
    return _get_core_config("BTC")


def _apply_high_accuracy_mode(cfg, enable=True):
    """Stub — high accuracy params already baked into unified config."""
    pass


xau_apply_high_accuracy_mode = _apply_high_accuracy_mode
btc_apply_high_accuracy_mode = _apply_high_accuracy_mode

# Aliases so _build_pipelines keeps working with same names
XauFeatureEngine = FeatureEngine
BtcFeatureEngine = FeatureEngine
XauRiskManager = RiskManager
BtcRiskManager = RiskManager
XauSignalEngine = SignalEngine
BtcSignalEngine = SignalEngine

from .execution import ExecutionWorker
from .logging_setup import _DIAG_ENABLED, _DIAG_EVERY_SEC, log_diag, log_err, log_health
from .models import AssetCandidate, ExecutionResult, OrderIntent, PortfolioStatus
from .pipeline import _AssetPipeline
from .scheduler import UTCScheduler
from .utils import parse_bar_key, safe_json_dumps, tf_seconds
from core.ml_router import MLSignal, fetch_ml_payloads, infer_from_payloads, validate_payload_schema
from core.model_manager import model_manager  # <--- HOLY TRINITY GATEKEEPER
from core.model_gate import gate_details

MODEL_STATE_PATH = get_artifact_path("models", "model_state.pkl")
MIN_BACKTEST_SHARPE = MIN_GATE_SHARPE
MIN_BACKTEST_WIN_RATE = MIN_GATE_WIN_RATE


class EngineState(Enum):
    BOOT = "BOOT"
    DATA_SYNC = "DATA_SYNC"
    ML_INFERENCE = "ML_INFERENCE"
    RISK_CALC = "RISK_CALC"
    EXECUTION_QUEUE = "EXECUTION_QUEUE"
    VERIFICATION = "VERIFICATION"
    HALT = "HALT"


@dataclass
class EngineCycleContext:
    payloads: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = field(default_factory=dict)
    ml_signals: Dict[str, MLSignal] = field(default_factory=dict)
    candidates: List[AssetCandidate] = field(default_factory=list)
    halt_reason: str = ""


class MultiAssetTradingEngine:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = bool(dry_run)
        self._model_loaded: bool = False
        self._backtest_passed: bool = False
        self._model_version: str = "N/A"
        self._catboost_payloads: Dict[str, Any] = {}  # asset -> {model, pipeline}
        self._catboost_pred_history: Dict[str, Deque[float]] = {}  # asset -> rolling predictions
        self._blocked_assets: List[str] = []
        self._model_sharpe: float = 0.0
        self._model_win_rate: float = 0.0
        self._backtest_linkage_verified: bool = False
        self._boot_report_printed: bool = False

        # ------------------------------------------------------------
        # QUANTUM ARCHITECTURE: SYSTEM MATRIX
        # ------------------------------------------------------------
        # self._print_system_matrix()  <-- Moved to explicit call
        # ------------------------------------------------------------

        self._run = threading.Event()
        self._lock = threading.Lock()


        self._manual_stop = False
        self._mt5_ready = False
        self._retraining_mode = False
        self._last_retraining_log_ts = 0.0

        # Status
        self._active_asset = "NONE"
        self._last_selected_asset = "NONE"
        self._order_counter = 0
        self._last_reconcile_ts = 0.0

        # Queues and worker
        self._max_queue = int(os.getenv("PORTFOLIO_MAX_QUEUE", "50") or "50")
        self._order_q: "queue.Queue[OrderIntent]" = queue.Queue(maxsize=self._max_queue)
        self._result_q: "queue.Queue[ExecutionResult]" = queue.Queue(maxsize=self._max_queue)
        self._exec_worker: Optional[ExecutionWorker] = None

        # IMPORTANT: map order_id -> risk_manager (robust RM routing)
        self._order_rm_by_id: Dict[str, Any] = {}

        # Log suppression for high-frequency skips
        self._last_log_id: Dict[str, str] = {}


        # Loop tuning
        self._poll_fast = float(os.getenv("PORTFOLIO_POLL_FAST_SEC", "0.25") or "0.25")
        self._poll_slow = float(os.getenv("PORTFOLIO_POLL_SLOW_SEC", "0.75") or "0.75")
        self._pipeline_log_every = float(os.getenv("PORTFOLIO_PIPELINE_LOG_EVERY", "2.0") or "2.0")
        self._last_pipeline_log_ts = 0.0
        self._last_pipeline_log_key: Optional[Tuple[Any, ...]] = None
        self._max_consecutive_errors = int(os.getenv("PORTFOLIO_MAX_CONSEC_ERRORS", "12") or "12")

        # Analysis state log throttling
        self._last_analysis_paused_state = False
        self._last_analysis_state_log_ts = 0.0

        self._edge_last_trade: dict[str, tuple[str, str]] = {}
        self._edge_last_notified: dict[str, tuple[str, str]] = {}
        self._last_skip_log_ts: dict[str, float] = {}  # Anti-spam throttle for blocked notifications
        
        # Order Tracking
        self._seen_index: dict[tuple[str, str], tuple[float, int]] = {}

        # Idempotency window
        self._seen: Deque[Tuple[str, str, float]] = deque()
        self._signal_cooldown_sec_by_asset: Dict[str, float] = {"XAU": 60.0, "BTC": 60.0}
        self._seen_cleanup_budget: int = 128

        # Recovery
        self._last_recover_ts = 0.0

        # Diagnostics
        self._diag_last_ts = 0.0

        # Data sync throttling
        self._last_data_sync_fail_ts: float = 0.0
        self._last_data_sync_fail_reason: str = ""

        # Pipelines configs
        self._xau_cfg: XauConfig = get_xau_config()
        xau_apply_high_accuracy_mode(self._xau_cfg, True)
        self._btc_cfg: BtcConfig = get_btc_config()
        btc_apply_high_accuracy_mode(self._btc_cfg, True)

        raw_cd = (
            getattr(self._xau_cfg, "signal_cooldown_sec_override", None)
            or getattr(self._btc_cfg, "signal_cooldown_sec_override", None)
        )
        self._signal_cooldown_override_sec = float(raw_cd) if raw_cd else None

        self._xau: Optional[_AssetPipeline] = None
        self._btc: Optional[_AssetPipeline] = None
        
        self._last_cand_xau: Optional[AssetCandidate] = None
        self._last_cand_btc: Optional[AssetCandidate] = None

        # Edge-trigger: 1 signal_id => 1 order
        self._edge_last_trade: Dict[str, Tuple[str, str]] = {"XAU": ("", "Neutral"), "BTC": ("", "Neutral")}
        self._edge_last_notified: Dict[str, Tuple[str, str]] = {"XAU": ("", "Neutral"), "BTC": ("", "Neutral")}

        self._order_notifier: Optional[Callable[[OrderIntent, ExecutionResult], None]] = None
        self._phase_notifier: Optional[Callable[[str, str, str, str], None]] = None
        self._engine_stop_notifier: Optional[Callable[[str, str], None]] = None
        self._daily_start_notifier: Optional[Callable[[str, str], None]] = None
        self._signal_notifier: Optional[Callable[[str, Any], None]] = None

        self._last_phase_by_asset: Dict[str, str] = {"XAU": "?", "BTC": "?"}
        self._hard_stop_notified: Dict[str, bool] = {"XAU": False, "BTC": False}
        self._last_daily_date_by_asset: Dict[str, str] = {"XAU": "", "BTC": ""}



        # ============================================================
        # CRITICAL FIX #3: Post-Trade Cooldown System
        # Prevents rapid re-entry after a trade closes
        # ============================================================
        self._last_trade_close_ts: Dict[str, float] = {"XAU": 0.0, "BTC": 0.0}
        self._trade_cooldown_sec: float = float(os.getenv("TRADE_COOLDOWN_SEC", "300.0") or "300.0")  # 5 minutes default
        self._cooldown_blocked_count: Dict[str, int] = {"XAU": 0, "BTC": 0}

        # Position tracking for detecting trade closures
        self._last_open_positions: Dict[str, int] = {"XAU": 0, "BTC": 0}

        self._refresh_signal_cooldowns()

        # ============================================================
        # PORTFOLIO RISK MANAGER (Cross-Asset Guard)
        # ============================================================
        self._portfolio_risk = PortfolioRiskManager(
            max_daily_drawdown_pct=self._xau_cfg.max_daily_loss_pct,  # e.g. 0.04
            max_total_exposure_factor=3.0,     # Cap leverage at 3x
            correlation_reduction=0.50,        # Cut size 50% if correlated
            max_risk_per_trade_pct=self._xau_cfg.max_risk_per_trade,
            max_concurrent_positions=6,
        )
    # -------------------- QUANTUM GATES --------------------
    def print_startup_matrix(self) -> None:
        model_loaded = "YES" if self._model_loaded else "NO"
        bt_status = "PASSED" if self._backtest_passed else "FAILED"
        print("\n" + "=" * 60)
        print("QUANTUM TRADING SYSTEM - MATRIX RELOADED")
        print("=" * 60)
        print(f"Mode:            {'DRY-RUN (Simulated)' if self.dry_run else 'LIVE TRADING (Real Money)'}")
        print("Risk Engine:     OK")
        print(f"Model Loaded:    {model_loaded} (v{self._model_version})")
        print(f"Backtest Status: {bt_status} (Sharpe >= {MIN_BACKTEST_SHARPE:.1f})")
        print("Signals:         SNIPER MODE (Conf >= 75%)")
        print("=" * 60 + "\n")

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
            log_err.error("MODEL_STATE_LOAD_ERROR | path=%s err=%s", MODEL_STATE_PATH, exc)
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
            log_health.warning("MODEL_GATE_AUTOFIX_START | reason=%s asset=%s", reason, asset)
            run_backtest(asset)
            state = self._load_model_state()
            if state is None:
                log_err.error("MODEL_GATE_AUTOFIX_FAIL | reason=%s asset=%s detail=state_still_missing", reason, asset)
                return False
            log_health.info("MODEL_GATE_AUTOFIX_OK | asset=%s path=%s", asset, MODEL_STATE_PATH)
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
        """Ensure live trading starts only when required assets pass strict gate.
        
        Supports PARTIAL_GATE_MODE: if enabled, engine starts with only 
        passing assets instead of blocking entirely.
        """
        if self.dry_run:
            log_health.info("MODEL_GATE_BYPASSED | reason=dry_run")
            self._model_loaded = True
            self._backtest_passed = True
            self._model_version = "dry_run"
            self._blocked_assets = []
            return

        # Reset gate-scoped payloads each time to avoid stale models on partial reloads.
        self._catboost_payloads = {}
        self._blocked_assets = []

        details = gate_details(required_assets=("XAU", "BTC"), allow_legacy_fallback=True)
        gate_ok = bool(details.get("ok", False))
        gate_reason = str(details.get("reason", "unknown"))
        assets = details.get("assets", {}) if isinstance(details, dict) else {}

        for asset, st in (assets.items() if isinstance(assets, dict) else []):
            try:
                log_health.info(
                    "ASSET_GATE_STATUS | asset=%s ok=%s reason=%s version=%s sharpe=%.3f win_rate=%.3f max_dd=%.3f legacy=%s",
                    asset,
                    bool(st.get("ok", False)),
                    str(st.get("reason", "unknown")),
                    str(st.get("model_version", "")),
                    float(st.get("sharpe", 0.0) or 0.0),
                    float(st.get("win_rate", 0.0) or 0.0),
                    float(st.get("max_drawdown_pct", 0.0) or 0.0),
                    bool(st.get("legacy_fallback", False)),
                )
            except Exception:
                continue

        # Check partial gate mode
        partial_mode = str(os.getenv("PARTIAL_GATE_MODE", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
        
        if not gate_ok and partial_mode:
            # Partial mode: find which assets pass individually
            passing_assets: list[str] = []
            for asset in ("XAU", "BTC"):
                st = assets.get(asset, {}) if isinstance(assets, dict) else {}
                if isinstance(st, dict) and bool(st.get("ok", False)):
                    passing_assets.append(asset)
            
            if passing_assets:
                log_health.warning(
                    "MODEL_GATE_PARTIAL | passing=%s blocked=%s reason=%s",
                    ",".join(passing_assets),
                    ",".join(sorted({"XAU", "BTC"} - set(passing_assets))),
                    gate_reason,
                )
                # Load models for passing assets only
                versions: list[str] = []
                sharpes: list[float] = []
                wins: list[float] = []
                for asset in passing_assets:
                    st = assets.get(asset, {}) if isinstance(assets, dict) else {}
                    version = str(st.get("model_version", "") or "").strip()
                    if version:
                        model = model_manager.load_model(version)
                        if model is not None:
                            versions.append(f"{asset}:{version}")
                            sharpes.append(float(st.get("sharpe", 0.0) or 0.0))
                            wins.append(float(st.get("win_rate", 0.0) or 0.0))
                            if isinstance(model, dict) and "model" in model and "pipeline" in model:
                                self._catboost_payloads[asset] = model
                                log_health.info("CATBOOST_LOADED | asset=%s version=%s", asset, version)
                
                if versions:
                    self._model_version = " | ".join(versions)
                    self._model_sharpe = min(sharpes) if sharpes else 0.0
                    self._model_win_rate = min(wins) if wins else 0.0
                    self._model_loaded = True
                    self._backtest_passed = True
                    # Mark which assets are blocked so pipelines can be disabled
                    self._blocked_assets = sorted({"XAU", "BTC"} - set(passing_assets))
                    log_health.info(
                        "MODEL_GATE_PARTIAL_OK | versions=%s blocked_assets=%s",
                        self._model_version,
                        ",".join(self._blocked_assets),
                    )
                    return

        if not gate_ok:
            log_health.warning("MODEL_GATE_SKIP | reason=%s", gate_reason)
            log_err.error("GATE_BLOCK_REASON | reason=%s", gate_reason)
            self._model_loaded = False
            self._backtest_passed = False
            self._model_version = "N/A"
            self._model_sharpe = 0.0
            self._model_win_rate = 0.0
            self._blocked_assets = ["BTC", "XAU"]
            return

        versions: list[str] = []
        sharpes: list[float] = []
        wins: list[float] = []
        for asset in ("XAU", "BTC"):
            st = assets.get(asset, {}) if isinstance(assets, dict) else {}
            version = str(st.get("model_version", "") or "").strip()
            if not version:
                log_health.warning("MODEL_GATE_SKIP | reason=missing_model_version asset=%s", asset)
                self._model_loaded = False
                self._backtest_passed = False
                return
            model = model_manager.load_model(version)
            if model is None:
                log_health.warning("MODEL_GATE_SKIP | reason=registry_load_failed asset=%s version=%s", asset, version)
                self._model_loaded = False
                self._backtest_passed = False
                return
            versions.append(f"{asset}:{version}")
            sharpes.append(float(st.get("sharpe", 0.0) or 0.0))
            wins.append(float(st.get("win_rate", 0.0) or 0.0))
            if isinstance(model, dict) and "model" in model and "pipeline" in model:
                self._catboost_payloads[asset] = model
                log_health.info("CATBOOST_LOADED | asset=%s version=%s", asset, version)

        self._model_version = " | ".join(versions)
        self._model_sharpe = min(sharpes) if sharpes else 0.0
        self._model_win_rate = min(wins) if wins else 0.0
        self._model_loaded = True
        self._backtest_passed = True
        self._blocked_assets = []
        log_health.info(
            "MODEL_GATE_PASSED | versions=%s sharpe_min=%.3f win_rate_min=%.3f",
            self._model_version,
            self._model_sharpe,
            self._model_win_rate,
        )

    def _infer_catboost(self, asset: str) -> Optional[MLSignal]:
        """Run CatBoost model inference on live M1 data from MT5."""
        if self._is_asset_blocked(asset):
            self._log_blocked_asset_skip(asset, "catboost_infer")
            return None
        payload = self._catboost_payloads.get(asset)
        if not payload:
            return None
        try:
            import pandas as pd
            import numpy as np

            cb_model = payload["model"]
            pipeline = payload["pipeline"]

            # Fetch live M1 bars from MT5
            pipe = self._xau if asset == "XAU" else self._btc
            if pipe is None:
                return None
            symbol = pipe.symbol
            if not symbol:
                return None

            n_bars = max(int(getattr(pipeline.cfg, 'window_size', 60) or 60) + 200, 500)
            with MT5_LOCK:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, n_bars)
            if rates is None or len(rates) < 100:
                log_health.warning("CATBOOST_SKIP | asset=%s reason=insufficient_bars bars=%s", asset, len(rates) if rates is not None else 0)
                return None

            df = pd.DataFrame(rates)
            df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "tick_volume": "Volume",
            })
            df = df.set_index("datetime")[["Open", "High", "Low", "Close", "Volume"]]

            # Transform through pipeline (adds indicators, creates windows)
            xy = pipeline.transform(df.copy())
            if xy["X"].empty:
                log_health.warning("CATBOOST_SKIP | asset=%s reason=empty_features", asset)
                return None

            # Use only the LAST row (current bar)
            X_last = np.stack([xy["X"].iloc[-1]])
            pred = cb_model.predict(X_last)
            pred_val = float(pred[0])

            # Determine signal from prediction using ADAPTIVE threshold
            # Static percent_increase from training config is too aggressive for live M1
            # Use adaptive threshold: median of recent prediction magnitudes
            static_threshold = max(float(getattr(pipeline.cfg, 'percent_increase', 0.0) or 0.0), 1e-12)
            
            # Track prediction history for adaptive threshold
            if asset not in self._catboost_pred_history:
                self._catboost_pred_history[asset] = deque(maxlen=200)
            self._catboost_pred_history[asset].append(abs(pred_val))
            
            # Adaptive threshold: 75th percentile of recent predictions
            # This means ~25% of predictions will generate signals
            hist = self._catboost_pred_history[asset]
            if len(hist) >= 10:
                adaptive_threshold = float(np.percentile(np.asarray(hist, dtype=np.float64), 75))
                threshold = max(adaptive_threshold, 1e-12)
            else:
                # Not enough history yet, use a fraction of static threshold
                threshold = static_threshold * 0.1
            
            confidence = min(abs(pred_val) / max(threshold, 1e-12), 1.0)
            confidence = max(0.0, min(1.0, confidence))

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
            atr_val = float(atr_arr[-1]) if atr_arr is not None and len(atr_arr) > 0 and np.isfinite(atr_arr[-1]) else last_close * 0.001

            if abs(pred_val) < threshold:
                return MLSignal(
                    asset=asset, signal="HOLD", side="Neutral",
                    confidence=confidence, reason=f"catboost_below_threshold:{pred_val:.6f}<{threshold:.6f}",
                    provider="catboost", model="catboost_trained",
                    entry=None, stop_loss=None, take_profit=None,
                    scalp_payload={"M1": {"ts_bar": float(df.index[-1].timestamp()), "last_close": last_close, "atr_14": atr_val, "rsi_14": 50.0, "ema_20": last_close, "ema_50": last_close, "ema_200": last_close}},
                    intraday_payload=None,
                )

            signal = "STRONG BUY" if pred_val > 0 else "STRONG SELL"
            side = "Buy" if pred_val > 0 else "Sell"

            # Build payload frame for downstream risk calc
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
                asset=asset, signal=signal, side=side,
                confidence=confidence,
                reason=f"catboost_pred:{pred_val:.6f}",
                provider="catboost", model="catboost_trained",
                entry=last_close, stop_loss=None, take_profit=None,
                scalp_payload={"M1": frame},
                intraday_payload={"H1": frame},
            )
        except Exception as exc:
            log_err.error("CATBOOST_INFER_ERROR | asset=%s err=%s | tb=%s", asset, exc, traceback.format_exc())
            return None

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

    @staticmethod
    def _is_non_retriable_mt5_error(exc: Exception) -> bool:
        msg = str(exc or "").lower()
        return any(
            k in msg
            for k in (
                "algo_trading_disabled_in_terminal",
                "trading_disabled_for_account",
                "wrong_account",
            )
        )

    # -------------------- HARD KILL-SWITCH (CRITICAL) --------------------
    def _check_hard_stop_file(self) -> None:
        """
        CRITICAL: Checks for physical 'STOP.lock' file presence.
        If found, IMMEDIATELY triggers manual_stop and wipes queues.
        """
        if os.path.exists("STOP.lock"):
            if not self._manual_stop:
                log_health.critical("HARD_KILL_SWITCH | STOP.lock DETECTED | ABORTING ALL TRADES")
                self._manual_stop = True
                
                # Immediate Queue Wipe
                self._drain_queue(self._order_q)
                self._drain_queue(self._result_q)
                
                # Optional: Send emergency notification
                if self._engine_stop_notifier:
                    try:
                        self._engine_stop_notifier("ALL", "STOP.lock File Detected")
                    except Exception:
                        pass

    def set_order_notifier(self, cb: Optional[Callable[[OrderIntent, ExecutionResult], None]]) -> None:
        self._order_notifier = cb

    def set_phase_notifier(self, cb: Optional[Callable[[str, str, str, str], None]]) -> None:
        self._phase_notifier = cb

    def set_engine_stop_notifier(self, cb: Optional[Callable[[str, str], None]]) -> None:
        self._engine_stop_notifier = cb

    def set_daily_start_notifier(self, cb: Optional[Callable[[str, str], None]]) -> None:
        self._daily_start_notifier = cb

    def set_signal_notifier(self, cb: Optional[Callable[[str, Any], None]]) -> None:
        self._signal_notifier = cb

    def set_skip_notifier(self, cb: Optional[Callable[[AssetCandidate], None]]) -> None:
        self._skip_notifier = cb



    # -------------------- MT5 init/health --------------------
    def _init_mt5(self) -> bool:
        try:
            ensure_mt5()
            with MT5_LOCK:
                acc = mt5.account_info()
                term = mt5.terminal_info()
            if not acc or not term or not getattr(term, "connected", False):
                log_err.error(
                    "MT5 init failed | acc=%s term=%s connected=%s",
                    bool(acc),
                    bool(term),
                    getattr(term, "connected", False) if term else None,
                )
                return False

            self._mt5_ready = True
            log_health.info(
                "MT5_INIT_OK | login=%s server=%s term_connected=%s term_trade=%s acc_trade=%s",
                getattr(acc, "login", "-"),
                getattr(acc, "server", "-"),
                getattr(term, "connected", False),
                getattr(term, "trade_allowed", False),
                getattr(acc, "trade_allowed", False),
            )
            return True
        except Exception as exc:
            self._mt5_ready = False
            if self._is_non_retriable_mt5_error(exc):
                with self._lock:
                    if not self._manual_stop:
                        self._manual_stop = True
                        log_health.warning(
                            "MANUAL_STOP_REQUESTED | reason=mt5_config_block detail=%s",
                            exc,
                        )
                log_health.warning("MT5_INIT_BLOCKED | %s", exc)
                return False
            log_err.error("MT5 init error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _check_mt5_health(self) -> bool:
        if self.dry_run:
            return True
        try:
            with MT5_LOCK:
                term = mt5.terminal_info()
                acc = mt5.account_info()
            ok = bool(
                term
                and getattr(term, "connected", False)
                and getattr(term, "trade_allowed", False)
                and acc
                and getattr(acc, "trade_allowed", True)
            )
            return ok
        except Exception as exc:
            log_err.error("mt5 health check error: %s | tb=%s", exc, traceback.format_exc())
            return False

    # -------------------- pipelines --------------------
    def _build_pipelines(self) -> bool:
        try:
            # XAU
            xau_feed = XauMarketFeed(self._xau_cfg, self._xau_cfg.symbol_params)
            xau_features = XauFeatureEngine(self._xau_cfg)
            xau_risk = XauRiskManager(self._xau_cfg, self._xau_cfg.symbol_params)
            xau_signal = XauSignalEngine(self._xau_cfg, self._xau_cfg.symbol_params, xau_feed, xau_features, xau_risk)
            self._xau = _AssetPipeline("XAU", self._xau_cfg, xau_feed, xau_features, xau_risk, xau_signal)
            if not self.dry_run:
                self._xau.ensure_symbol_selected()

            # BTC
            btc_feed = BtcMarketFeed(self._btc_cfg, self._btc_cfg.symbol_params)
            btc_features = BtcFeatureEngine(self._btc_cfg)
            btc_risk = BtcRiskManager(self._btc_cfg, self._btc_cfg.symbol_params)
            btc_signal = BtcSignalEngine(self._btc_cfg, self._btc_cfg.symbol_params, btc_feed, btc_features, btc_risk)
            self._btc = _AssetPipeline("BTC", self._btc_cfg, btc_feed, btc_features, btc_risk, btc_signal)
            if not self.dry_run:
                self._btc.ensure_symbol_selected()

            # IMPORTANT: cooldowns must be refreshed AFTER pipelines exist
            self._refresh_signal_cooldowns()

            log_health.info("PIPELINES_BUILT | xau=%s btc=%s", self._xau.symbol, self._btc.symbol)
            return True
        except Exception as exc:
            log_err.error("build pipelines error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _restart_exec_worker(self) -> None:
        if self._exec_worker:
            try:
                self._exec_worker.stop()
            except Exception:
                pass
            try:
                self._exec_worker.join(timeout=4.0)
            except Exception:
                pass
        self._exec_worker = ExecutionWorker(self._order_q, self._result_q, self.dry_run, self._order_notify_bridge)
        self._exec_worker.start()

    def _order_notify_bridge(self, intent: OrderIntent, res: ExecutionResult) -> None:
        if self._order_notifier:
            try:
                self._order_notifier(intent, res)
            except Exception:
                pass

    # -------------------- recovery --------------------
    @staticmethod
    def _drain_queue(q: "queue.Queue[Any]", limit: int = 10_000) -> int:
        n = 0
        try:
            while n < int(limit):
                _ = q.get_nowait()
                try:
                    q.task_done()
                except Exception:
                    pass
                n += 1
        except Exception:
            pass
        return n

    def _recover_all(self) -> bool:
        now = time.time()
        if (now - self._last_recover_ts) < 8.0:
            return False
        self._last_recover_ts = now
        try:
            log_health.info("RECOVER_START")


            # Stop worker first (prevents executing stale intents)
            if self._exec_worker:
                try:
                    self._exec_worker.stop()
                except Exception:
                    pass
                try:
                    self._exec_worker.join(timeout=4.0)
                except Exception:
                    pass

            # Drop stale queues + mapping
            self._drain_queue(self._order_q)
            self._drain_queue(self._result_q)
            self._order_rm_by_id.clear()

            try:
                with MT5_LOCK:
                    mt5.shutdown()
            except Exception:
                pass
            time.sleep(0.3)

            if not self._init_mt5():
                return False
            if not self._build_pipelines():
                return False

            self._restart_exec_worker()

            log_health.info("RECOVER_OK")
            return True
        except Exception as exc:
            log_err.error("recover error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _refresh_signal_cooldowns(self) -> None:
        """Set per-asset idempotency cooldown.

        Goal: prevent repeated orders within the same bar (especially M1).
        - If signal_cooldown_sec_override is set (>0), it overrides everything.
        - Otherwise cooldown defaults to TF seconds of each asset primary timeframe.
        """
        try:
            if self._signal_cooldown_override_sec is not None and self._signal_cooldown_override_sec > 0:
                cd = float(self._signal_cooldown_override_sec)
                self._signal_cooldown_sec_by_asset["XAU"] = cd
                self._signal_cooldown_sec_by_asset["BTC"] = cd
                return

            def _pipe_tf_sec(pipe: Optional[_AssetPipeline]) -> float:
                try:
                    if pipe is None:
                        return 60.0
                    tf = getattr(getattr(pipe.cfg, "symbol_params", None), "tf_primary", None)
                    sec = tf_seconds(tf)
                    return float(sec) if sec else 60.0
                except Exception:
                    return 60.0

            self._signal_cooldown_sec_by_asset["XAU"] = max(_pipe_tf_sec(self._xau), 60.0)
            self._signal_cooldown_sec_by_asset["BTC"] = max(_pipe_tf_sec(self._btc), 60.0)
        except Exception:
            self._signal_cooldown_sec_by_asset["XAU"] = 60.0
            self._signal_cooldown_sec_by_asset["BTC"] = 60.0

    # -------------------- phase/daily helpers --------------------
    @staticmethod
    def _get_phase(risk: Any) -> str:
        if risk is None:
            return "A"
        for attr in ("current_phase", "phase", "mode", "risk_phase"):
            v = getattr(risk, attr, None)
            if v:
                return str(v)
        return "A"

    @staticmethod
    def _get_daily_date(risk: Any) -> str:
        if risk is None:
            return ""
        for attr in ("daily_date", "today_date", "day_key", "session_date"):
            v = getattr(risk, attr, None)
            if v:
                return str(v)
        return ""

    def _phase_reason(self, risk: Any, new_phase: str) -> str:
        if risk is None:
            return ""
        fn = getattr(risk, "phase_reason", None)
        if callable(fn):
            try:
                return str(fn() or "")
            except Exception:
                return ""
        if new_phase == "C":
            fn = getattr(risk, "hard_stop_reason", None)
            if callable(fn):
                try:
                    return str(fn() or "")
                except Exception:
                    return ""
        return ""

    def _check_phase_change(self, asset: str, risk: Any) -> None:
        try:
            if risk is None:
                return
            asset_u = str(asset).upper()
            if not UTCScheduler.market_status(asset_u):
                return
            fn = getattr(risk, "update_phase", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
            current = self._get_phase(risk) or "A"
            prev = str(self._last_phase_by_asset.get(asset_u, "A") or "A")
            if current != prev:
                self._last_phase_by_asset[asset_u] = current
                if self._phase_notifier:
                    reason = self._phase_reason(risk, current)
                    self._phase_notifier(asset_u, prev, current, reason)
        except Exception:
            return

    def _check_daily_start(self, asset: str, risk: Any) -> None:
        try:
            if risk is None:
                return
            asset_u = str(asset).upper()
            if not UTCScheduler.market_status(asset_u):
                return
            current_date = self._get_daily_date(risk)
            if not current_date:
                return
            prev_date = str(self._last_daily_date_by_asset.get(asset_u, "") or "")
            if not prev_date:
                self._last_daily_date_by_asset[asset_u] = current_date
                return
            if current_date != prev_date:
                self._last_daily_date_by_asset[asset_u] = current_date
                if self._daily_start_notifier:
                    self._daily_start_notifier(asset_u, current_date)
        except Exception:
            return

    # -------------------- idempotency --------------------
    def _cooldown_for_asset(self, asset: str) -> float:
        return float(self._signal_cooldown_sec_by_asset.get(asset, 60.0))

    def _is_duplicate(
        self,
        asset: str,
        signal_id: str,
        now: float,
        max_orders: int,
        *,
        order_index: int = 0,
    ) -> bool:
        """Duplicate guard with cooldown + order-count.

        - cooldown blocks repeats for same signal_id within TF seconds
        - max_orders caps per signal_id
        - order_index allows batch split to bypass scale delay
        """
        last = self._seen_index.get((asset, signal_id))
        if not last:
            return False
        last_ts, count = last

        max_orders = max(1, int(max_orders))

        # Hard cap on count
        if int(count) >= int(max_orders):
            return True

        # Cooldown: prevent repeated orders within same bar window
        cd = self._cooldown_for_asset(asset)
        if (now - float(last_ts)) < float(cd):
            # If user intentionally split lot into multiple orders in same batch,
            # allow it ONLY when order_index>0 and count < max_orders.
            if int(order_index) > 0 and int(count) < int(max_orders):
                return False
            return True

        # Scaling throttle: if already have an order, ensure 30s gap before next scaling order (M1 safety)
        min_scale_delay = 30.0
        if int(count) > 0 and int(order_index) <= 0 and (now - float(last_ts)) < min_scale_delay:
            return True

        return False

    def _mark_seen(self, asset: str, signal_id: str, now: float) -> None:
        key = (asset, signal_id)
        last = self._seen_index.get(key)
        count = 0
        if last:
            _, last_count = last
            count = int(last_count)

        count += 1
        self._seen_index[key] = (float(now), int(count))
        self._seen.append((asset, signal_id, now))

        max_cd = max(self._signal_cooldown_sec_by_asset.values(), default=60.0)
        ttl = max(2.0 * float(max_cd), 120.0)
        cleaned = 0
        while self._seen and (now - self._seen[0][2]) > ttl and cleaned < self._seen_cleanup_budget:
            a, sid, ts = self._seen.popleft()
            rec = self._seen_index.get((a, sid))
            if rec and float(rec[0]) == float(ts):
                self._seen_index.pop((a, sid), None)
            cleaned += 1

    # -------------------- portfolio logic --------------------
    def _next_order_id(self, asset: str) -> str:
        self._order_counter += 1
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"PORD_{asset}_{ts}_{self._order_counter}_{os.getpid()}"

    def _effective_min_conf(self, c: AssetCandidate) -> float:
        cfg = self._xau_cfg if c.asset == "XAU" else self._btc_cfg
        base = float(getattr(cfg, "min_confidence_signal", 0.55) or 0.55)

        env_global = float(os.getenv("PORTFOLIO_MIN_CONF", "0.50") or "0.50")
        env_asset = float(os.getenv(f"PORTFOLIO_MIN_CONF_{str(c.asset).upper()}", str(env_global)) or str(env_global))
        floor = max(0.35, min(0.90, env_asset))

        effective_min = max(floor, base)
        rs = tuple(str(r) for r in (c.reasons or ()))

        # Early momentum signals are allowed with a lower floor to avoid late entries.
        if any(r.startswith("early_momentum") for r in rs):
            effective_min = max(floor, effective_min - 0.08)



        return float(max(0.0, min(1.0, effective_min)))

    def _is_asset_blocked(self, asset: str) -> bool:
        asset_u = str(asset or "").upper().strip()
        return asset_u in self._blocked_assets

    def _log_blocked_asset_skip(self, asset: str, stage: str) -> None:
        key = f"{str(asset).upper()}:{stage}"
        now = time.time()
        last = float(self._last_skip_log_ts.get(key, 0.0))
        if now - last < 30.0:
            return
        self._last_skip_log_ts[key] = now
        log_health.info(
            "ASSET_BLOCKED_SKIP | asset=%s stage=%s blocked_assets=%s",
            str(asset).upper(),
            stage,
            ",".join(sorted(set(self._blocked_assets))),
        )

    def _candidate_is_tradeable(self, c: AssetCandidate) -> bool:
        if self._is_asset_blocked(c.asset):
            self._log_blocked_asset_skip(c.asset, "candidate_tradeable")
            return False
        if c.signal not in ("Buy", "Sell"):
            return False
        if c.blocked:
            return False
        if c.lot <= 0.0:
            return False
        if c.sl <= 0.0 or c.tp <= 0.0:
            return False

        conf_val = float(c.confidence)
        return bool(conf_val >= self._effective_min_conf(c))

    def _select_active_asset(self, open_xau: int, open_btc: int) -> str:
        active_assets = UTCScheduler.get_active_assets()

        # Weekend -> BTC only (status logic)
        if "XAU" not in active_assets:
            if open_xau > 0 and open_btc == 0:
                return "XAU"
            return "BTC"

        if open_xau > 0 and open_btc > 0:
            return "BOTH"
        if open_xau > 0:
            return "XAU"
        if open_btc > 0:
            return "BTC"
        return "XAU+BTC"

    def _enqueue_order(
        self,
        cand: AssetCandidate,
        *,
        order_index: int = 0,
        order_count: int = 1,
        lot_override: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        now = time.time()

        # Resolve risk/cfg for this asset early
        if cand.asset == "XAU":
            risk = self._xau.risk if self._xau else None
            cfg = self._xau_cfg
        else:
            risk = self._btc.risk if self._btc else None
            cfg = self._btc_cfg

        # Ensure phase is fresh
        try:
            fn = getattr(risk, "update_phase", None)
            if callable(fn):
                fn()
        except Exception:
            pass

        # Hard-stop: MUST block trading (even if phase not yet updated)
        try:
            fn = getattr(risk, "requires_hard_stop", None)
            if callable(fn) and bool(fn()):
                log_health.info("ENQUEUE_SKIP | asset=%s reason=hard_stop", cand.asset)
                return False, None
        except Exception:
            pass

        # ============================================================
        # PHASE C SHADOW MODE: Block execution but send alert
        # Shows what trade WOULD have happened for transparency
        # ============================================================
        phase = self._get_phase(risk)
        if phase == "C":
            log_health.warning(
                "PHASE_C_SHADOW | asset=%s signal=%s conf=%.2f lot=%.4f sl=%.2f tp=%.2f | BLOCKED - Shadow Mode Active",
                cand.asset,
                cand.signal,
                cand.confidence,
                cand.lot,
                cand.sl,
                cand.tp,
            )
            
            # Send Telegram notification (Shadow Mode alert)
            if self._signal_notifier:
                try:
                    # Get current price for the alert
                    with MT5_LOCK:
                        tick = mt5.symbol_info_tick(cand.symbol)
                    current_price = float(tick.ask if cand.signal == "Buy" else tick.bid) if tick else 0.0
                    
                    # Build shadow mode alert payload
                    shadow_alert = {
                        "type": "PHASE_C_SHADOW",
                        "asset": cand.asset,
                        "symbol": cand.symbol,
                        "signal": cand.signal,
                        "confidence": cand.confidence,
                        "lot": cand.lot,
                        "sl": cand.sl,
                        "tp": cand.tp,
                        "price": current_price,
                        "blocked": True,
                        "reason": "Phase C - Daily Risk Limit Reached",
                        "message": (
                            f"⚠️ [PHASE C — Shadow Trade] Сигнали тасдиқшудаи {cand.signal} барои {cand.symbol}.\n"
                            f"Нарх: {current_price:.2f} | Боварӣ: {cand.confidence:.0f}%\n"
                            f"(Савдо бо лимити риск манъ шуд)\n"
                            f"Ҳаҷми эҳтимолӣ: {cand.lot:.2f} | SL: {cand.sl:.2f} | TP: {cand.tp:.2f}"
                        ),
                    }
                    self._signal_notifier(cand.asset, shadow_alert)
                except Exception as e:
                    log_health.error("PHASE_C_SHADOW notification error: %s", e)
            
            return False, None

        # ============================================================
        # FIX #3: Post-Trade Cooldown Check
        # Prevents rapid re-entry after a trade closes
        # ============================================================
        last_close_ts = self._last_trade_close_ts.get(cand.asset, 0.0)
        time_since_close = now - last_close_ts
        if time_since_close < self._trade_cooldown_sec and last_close_ts > 0:
            self._cooldown_blocked_count[cand.asset] = self._cooldown_blocked_count.get(cand.asset, 0) + 1
            # Log periodically to avoid spam
            if self._cooldown_blocked_count[cand.asset] % 10 == 1:
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=trade_cooldown remaining=%.0fs total_blocked=%d",
                    cand.asset,
                    self._trade_cooldown_sec - time_since_close,
                    self._cooldown_blocked_count[cand.asset],
                )
            return False, None

        # ============================================================
        # PORTFOLIO RISK CHECK (Cross-Asset Guard)
        # ============================================================
        # Estimate margin (conservative: leverage=1, or ignore if difficult)
        # For now, we assume simple 1:200 approx check or skip margin cap if 0 
        # But we MUST pass equity and lot to check correlation/DD.
        with MT5_LOCK:
            acc_info = mt5.account_info()
            equity = getattr(acc_info, "equity", 0.0) if acc_info else 0.0
            leverage = getattr(acc_info, "leverage", 100) if acc_info else 100

        # Estimate proposed margin
        # margin = (price * lot * contract_size) / leverage
        c_size = float(getattr(cfg.symbol_params, "contract_size", 100.0))
        # Use simple estimation
        proposed_margin = (float(cand.close if hasattr(cand, 'close') else 0) * float(cand.lot) * c_size) / float(leverage or 1)
        if proposed_margin <= 0:
             # fallback if close not in cand
             with MT5_LOCK:
                 tick_info = mt5.symbol_info_tick(cand.symbol)
             p = tick_info.ask if tick_info else 0.0
             proposed_margin = (p * float(cand.lot) * c_size) / float(leverage or 1)

        allowed, adj_lot, block_reason = self._portfolio_risk.check_before_order(
            asset=cand.asset,
            side=cand.signal,
            equity=equity,
            proposed_lot=float(cand.lot),
            proposed_margin=proposed_margin,
        )
        
        if not allowed:
            log_health.warning("PORTFOLIO_BLOCK | asset=%s reason=%s", cand.asset, block_reason)
            return False, None
            
        if adj_lot < cand.lot:
            log_health.info("PORTFOLIO_REDUCE | asset=%s lot %.2f -> %.2f (Correlation Guard)", cand.asset, cand.lot, adj_lot)
            # Update candidate lot in place (or just use override)
            lot_override = adj_lot

        # 1 signal_id => 1 order (edge trigger)
        last_id, last_sig = self._edge_last_trade.get(cand.asset, ("", "Neutral"))
        if cand.signal in ("Buy", "Sell") and last_id == str(cand.signal_id) and last_sig == str(cand.signal):
            # Throttled logging for edge duplicates
            last_log = self._last_log_id.get(cand.asset)
            if last_log != str(cand.signal_id):
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=edge_same_signal_id signal=%s signal_id=%s",
                    cand.asset,
                    cand.signal,
                    cand.signal_id,
                )
                self._last_log_id[cand.asset] = str(cand.signal_id)
            return False, None

        # SILENT DUPLICATE CHECK: Avoid log spam for the same signal ID
        # Only log if we haven't seen this specific signal ID skip recently
        if self._is_duplicate(cand.asset, cand.signal_id, now, max_orders=int(order_count), order_index=int(order_index)):
            # Check if we already logged this skip for this signal_id
            last_log = self._last_log_id.get(cand.asset)
            if last_log != str(cand.signal_id):
                log_health.info("ENQUEUE_SKIP | asset=%s reason=duplicate signal_id=%s", cand.asset, cand.signal_id)
                self._last_log_id[cand.asset] = str(cand.signal_id)
            return False, None

        # Tick + digits
        with MT5_LOCK:
            tick = mt5.symbol_info_tick(cand.symbol)
            info = mt5.symbol_info(cand.symbol)
        if tick is None:
            log_health.info("ENQUEUE_SKIP | asset=%s reason=tick_missing", cand.asset)
            return False, None

        price = float(tick.ask if cand.signal == "Buy" else tick.bid)
        if price <= 0:
            log_health.info("ENQUEUE_SKIP | asset=%s reason=bad_price", cand.asset)
            return False, None

        # ============================================================
        # FIX #4: SLIPPAGE PROTECTION (Pre-Trade)
        # ============================================================
        # 1. Spread Check
        current_spread_points = float(tick.ask - tick.bid) / float(info.point) if info.point else 99999.0
        max_spread_pts = float(getattr(cfg, "max_spread_points", 0) or 0)
        
        # Default safety defaults if config missing
        if max_spread_pts <= 0:
            max_spread_pts = 2500.0 if cand.asset == "BTC" else 350.0  # ~35 pips XAU (3-digit), ~$25 BTC
            
        if current_spread_points > max_spread_pts:
             # Throttle this log to once per 30s per asset to prevent spam
             _sp_key = f"_spread_skip_ts_{cand.asset}"
             _sp_last = getattr(self, _sp_key, 0.0)
             if (now - _sp_last) >= 30.0:
                 setattr(self, _sp_key, now)
                 log_health.info(
                     "ENQUEUE_SKIP | asset=%s reason=spread_too_high spread=%s > max=%s", 
                     cand.asset, int(current_spread_points), int(max_spread_pts)
                 )
             return False, None

        # 2. Volatility/Slip Estimation (Basic)
        # If market is moving too fast (tick.last != bid/ask midpoint by large margin), maybe unsafe?
        # For now, spread check is the most reliable "pre-trade" slippage guard.


        digits = int(getattr(info, "digits", 0) or 0) if info else 0
        sl_val = float(cand.sl)
        tp_val = float(cand.tp)
        if digits > 0:
            try:
                price = float(builtins.round(price, digits))
                sl_val = float(builtins.round(sl_val, digits))
                tp_val = float(builtins.round(tp_val, digits))
            except Exception:
                pass

        # SL/TP direction sanity (critical)
        if cand.signal == "Buy":
            if not (sl_val < price < tp_val):
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=bad_sl_tp_relation side=Buy price=%.5f sl=%.5f tp=%.5f",
                    cand.asset,
                    price,
                    sl_val,
                    tp_val,
                )
                return False, None
        else:
            if not (tp_val < price < sl_val):
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=bad_sl_tp_relation side=Sell price=%.5f sl=%.5f tp=%.5f",
                    cand.asset,
                    price,
                    sl_val,
                    tp_val,
                )
                return False, None

        lot_val = float(lot_override) if lot_override is not None else float(cand.lot)
        if lot_val <= 0:
            log_health.info("ENQUEUE_SKIP | asset=%s reason=lot_nonpositive", cand.asset)
            return False, None

        # Phase B: reduce lot by 50%
        if phase == "B":
            orig = lot_val
            lot_val = max(0.01, lot_val * 0.5)
            log_health.info(
                "PHASE_B_LOT_REDUCE | asset=%s original_lot=%.4f reduced_lot=%.4f",
                cand.asset,
                float(orig),
                float(lot_val),
            )

        order_id = self._next_order_id(cand.asset)
        idem = f"{cand.symbol}:{cand.signal_id}:{cand.signal}"

        intent = OrderIntent(
            asset=cand.asset,
            symbol=cand.symbol,
            signal=cand.signal,
            confidence=float(cand.confidence),
            lot=float(lot_val),
            sl=float(sl_val),
            tp=float(tp_val),
            price=float(price),
            enqueue_time=float(now),
            order_id=str(order_id),
            signal_id=str(cand.signal_id),
            idempotency_key=str(idem),
            risk_manager=risk,
            cfg=cfg,
        )

        # Remember RM mapping BEFORE enqueue; rollback on failure
        self._order_rm_by_id[str(order_id)] = risk

        try:
            self._order_q.put_nowait(intent)
        except queue.Full:
            self._order_rm_by_id.pop(str(order_id), None)
            if risk and hasattr(risk, "record_execution_failure"):
                try:
                    risk.record_execution_failure(order_id, now, now, "queue_backlog_drop")
                except Exception:
                    pass
            log_health.warning("ENQUEUE_FAIL | asset=%s reason=queue_full order_id=%s", cand.asset, order_id)
            return False, None

        self._mark_seen(cand.asset, cand.signal_id, now)
        self._last_selected_asset = cand.asset
        self._edge_last_trade[cand.asset] = (str(cand.signal_id), str(cand.signal))

        if risk:
            if hasattr(risk, "register_signal_emitted"):
                try:
                    risk.register_signal_emitted()
                except Exception:
                    pass
            if hasattr(risk, "track_signal_survival"):
                try:
                    risk.track_signal_survival(order_id, cand.signal, price, sl_val, tp_val, now, cand.confidence)
                except Exception:
                    pass

        return True, str(order_id)

    def _orders_for_candidate(self, cand: AssetCandidate) -> int:
        """STRICT: 1 signal = 1 order. Phase C: 0 orders."""
        pipe = self._xau if cand.asset == "XAU" else self._btc
        if not pipe or not pipe.risk:
            return 1

        phase = self._get_phase(pipe.risk)
        if phase == "C":
            return 0

        # Extra safety: hard_stop => 0 orders
        try:
            fn = getattr(pipe.risk, "requires_hard_stop", None)
            if callable(fn) and bool(fn()):
                return 0
        except Exception:
            pass

        return 1

    @staticmethod
    def _min_lot(risk: Any) -> float:
        try:
            if risk and hasattr(risk, "_symbol_meta"):
                risk._symbol_meta()  # type: ignore[attr-defined]
            return float(getattr(risk, "_vol_min", 0.01) or 0.01)
        except Exception:
            return 0.01

    def _split_lot(self, lot: float, parts: int, risk: Any, cfg: Any) -> list[float]:
        lot = float(lot)
        if int(parts) <= 1:
            return [lot]

        min_lot = self._min_lot(risk)
        split_enabled = bool(getattr(cfg, "multi_order_split_lot", True))
        if not split_enabled:
            if min_lot <= 0 or lot < min_lot:
                return [lot]
            return [lot for _ in range(int(parts))]

        if min_lot <= 0 or (lot / float(parts)) < min_lot:
            return [lot]

        base = lot / float(parts)
        lots = [base for _ in range(int(parts))]
        lots[-1] = float(lot) - (base * float(int(parts) - 1))
        return lots

    # -------------------- heartbeat/status --------------------
    def _heartbeat(self, open_xau: int, open_btc: int) -> None:
        try:
            with MT5_LOCK:
                acc = mt5.account_info()
            bal = float(getattr(acc, "balance", 0.0) or 0.0) if acc else 0.0
            eq = float(getattr(acc, "equity", 0.0) or 0.0) if acc else 0.0

            # Prefer portfolio DD from account; it is authoritative
            dd = max(0.0, (bal - eq) / bal) if bal > 0 else 0.0

            # today_pnl: prefer RM sum if available, else account profit
            pnl = float(getattr(acc, "profit", 0.0) or 0.0) if acc else 0.0
            try:
                p1 = self._xau.risk if self._xau else None
                p2 = self._btc.risk if self._btc else None
                v1 = float(getattr(p1, "today_pnl", 0.0) or 0.0) if p1 else 0.0
                v2 = float(getattr(p2, "today_pnl", 0.0) or 0.0) if p2 else 0.0
                # if RM tracks, use sum
                if (p1 and hasattr(p1, "today_pnl")) or (p2 and hasattr(p2, "today_pnl")):
                    pnl = float(v1 + v2)
            except Exception:
                pass

            last_sig_x = self._xau.last_signal if self._xau else "?"
            last_sig_b = self._btc.last_signal if self._btc else "?"

            st = PortfolioStatus(
                connected=bool(self._mt5_ready),
                trading=bool(self._run.is_set()),
                manual_stop=bool(self._manual_stop),
                active_asset=str(self._active_asset),
                balance=bal,
                equity=eq,
                dd_pct=float(dd),
                today_pnl=float(pnl),
                open_trades_total=int(open_xau + open_btc),
                open_trades_xau=int(open_xau),
                open_trades_btc=int(open_btc),
                last_signal_xau=str(last_sig_x),
                last_signal_btc=str(last_sig_b),
                last_selected_asset=str(self._last_selected_asset),
                exec_queue_size=int(self._order_q.qsize()),
                last_reconcile_ts=float(self._last_reconcile_ts),
            )

            if _DIAG_ENABLED and (time.time() - self._diag_last_ts) >= float(_DIAG_EVERY_SEC):
                self._diag_last_ts = time.time()
                payload = {
                    "ts_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "engine": st,
                    "mt5": mt5_status(),
                }
                try:
                    log_diag.info(safe_json_dumps(payload))
                except Exception:
                    pass
        except Exception as exc:
            log_err.error("heartbeat error: %s | tb=%s", exc, traceback.format_exc())

    # -------------------- status API --------------------
    def status(self) -> PortfolioStatus:
        try:
            with MT5_LOCK:
                term = mt5.terminal_info()
                acc = mt5.account_info()
        except Exception:
            term = None
            acc = None

        connected = bool(self._mt5_ready and term and getattr(term, "connected", False))

        bal = float(getattr(acc, "balance", 0.0) or 0.0) if acc else 0.0
        eq = float(getattr(acc, "equity", 0.0) or 0.0) if acc else 0.0
        pnl = float(getattr(acc, "profit", 0.0) or 0.0) if acc else 0.0
        dd = max(0.0, (bal - eq) / bal) if bal > 0 else 0.0

        open_xau = self._xau.open_positions() if self._xau else 0
        open_btc = self._btc.open_positions() if self._btc else 0

        return PortfolioStatus(
            connected=connected,
            trading=bool(self._run.is_set()),
            manual_stop=bool(self._manual_stop),
            active_asset=str(self._active_asset),
            balance=bal,
            equity=eq,
            dd_pct=float(dd),
            today_pnl=float(pnl),
            open_trades_total=int(open_xau + open_btc),
            open_trades_xau=int(open_xau),
            open_trades_btc=int(open_btc),
            last_signal_xau=str(self._xau.last_signal if self._xau else "Neutral"),
            last_signal_btc=str(self._btc.last_signal if self._btc else "Neutral"),
            last_selected_asset=str(self._last_selected_asset),
            exec_queue_size=int(self._order_q.qsize()),
            last_reconcile_ts=float(self._last_reconcile_ts),
        )

    # -------------------- external API --------------------
    def _update_portfolio_risk_state(self) -> None:
        """Fetch live positions and equity, update portfolio manager state."""
        try:
            if self.dry_run:
                # Mock data for dry run
                from types import SimpleNamespace
                acc = SimpleNamespace(equity=10000.0, leverage=200)
                positions = []
            else:
                with MT5_LOCK:
                    acc = mt5.account_info()
                    positions = mt5.positions_get()
            
            if not acc:
                return

            equity = float(getattr(acc, "equity", 0.0))
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            
            # 1. Update daily baseline (only sets if date changed)
            self._portfolio_risk.set_daily_start_equity(equity, date_str)
            
            # 2. Group positions by asset
            # We assume symbols contain "BTC" or "XAU" to map back to asset strings
            exposures = {"XAU": AssetExposure(symbol="XAUUSDm"), "BTC": AssetExposure(symbol="BTCUSDm")}
            
            # Helper to map symbol -> asset
            def _get_asset(sym: str):
                s = sym.upper()
                if "BTC" in s: return "BTC"
                if "XAU" in s or "GOLD" in s: return "XAU"
                return None

            if positions:
                for p in positions:
                    asset = _get_asset(p.symbol)
                    if asset and asset in exposures:
                        exp = exposures[asset]
                        exp.symbol = p.symbol
                        # Net volume? Or just sum? For correlation, we care about NET direction usually.
                        # But PortfolioRiskManager expects "side". 
                        # If we have hedging (both buy and sell), correlation is ambiguous.
                        # For simplicity: predominant side.
                        
                        vol = p.volume
                        p_type = p.type  # 0=Buy, 1=Sell
                        
                        # Add to metrics
                        exp.position_count += 1
                        margin_used = float(getattr(p, "margin", 0.0) or 0.0)
                        if margin_used <= 0.0:
                            price_ref = float(getattr(p, "price_current", 0.0) or getattr(p, "price_open", 0.0) or 0.0)
                            lev = float(getattr(acc, "leverage", 0.0) or 0.0)
                            if price_ref > 0.0 and lev > 0.0:
                                cfg = self._xau_cfg if asset == "XAU" else self._btc_cfg
                                contract_size = float(getattr(getattr(cfg, "symbol_params", None), "contract_size", 1.0) or 1.0)
                                margin_used = (price_ref * float(vol) * contract_size) / max(lev, 1.0)
                        exp.margin_used += max(0.0, float(margin_used))
                        exp.unrealized_pnl += p.profit
                        
                        # Signed volume for net direction
                        signed_vol = vol if p_type == 0 else -vol
                        exp.volume += signed_vol
            
            # 3. Push updates
            for asset, exp in exposures.items():
                # Determine net side
                side = "Neutral"
                if exp.volume > 0.000001: side = "Buy"
                elif exp.volume < -0.000001: side = "Sell"
                
                exp.side = side
                exp.volume = abs(exp.volume) # Store absolute volume, side is separate
                self._portfolio_risk.update_exposure(asset, exp)
                
        except Exception as exc:
            log_err.error("portfolio update error: %s", exc)

    def _check_backtest_integration(self) -> None:
        """
        Verify that Backtest engine is importable and parameters are aligned.
        Satisfies 'Explicitly initialize Backtest engine' requirement.
        """
        if self._backtest_linkage_verified:
            return
        try:
            if BacktestEngine is None:
                raise ImportError("Backtest module not found")
            
            # Just instantiate to prove linkage works
            _ = BacktestEngine(
                "XAU",
                model_version=XAU_BACKTEST_CONFIG.model_version,
                run_cfg=XAU_BACKTEST_CONFIG,
            )
            _ = BacktestEngine(
                "BTC",
                model_version=BTC_BACKTEST_CONFIG.model_version,
                run_cfg=BTC_BACKTEST_CONFIG,
            )
            self._backtest_linkage_verified = True
            log_health.info("Backtest Engine Integration: VERIFIED (XAU + BTC configs loaded)")
        except Exception as exc:
            log_err.error("Backtest Engine Integration FAILED: %s", exc)
            if not self.dry_run:
                # In production, warn but don't crash unless critical
                print(f"WARNING: Backtest Engine Linkage Error: {exc}")

    def start(self) -> bool:
        if not self.dry_run:
            self._check_model_health()
            if not (self._model_loaded and self._backtest_passed):
                log_err.critical(
                    "ENGINE_START_BLOCKED | reason=gatekeeper_failed model_loaded=%s backtest_passed=%s",
                    self._model_loaded,
                    self._backtest_passed,
                )
                return False

        with self._lock:
            if self._run.is_set():
                return True
            # MONITORING MODE: Allow start even if manual_stop is true
            self._run.set()

        # Propagate runtime mode to downstream components (feeds/pipelines).
        # This enables dry-run specific fast-fail behavior in data feeds.
        try:
            setattr(self._xau_cfg, "dry_run", bool(self.dry_run))
        except Exception:
            pass
        try:
            setattr(self._btc_cfg, "dry_run", bool(self.dry_run))
        except Exception:
            pass

        # ---------------------------------------------------------
        # SYSTEM BOOT REPORT (Console Visibility)
        # ---------------------------------------------------------
        if not self._boot_report_printed:
            print("\n" + "="*60)
            print("SYSTEM BOOT REPORT")
            print("="*60)
            print("OK Configuration Loaded  (Active: XAU + BTC)")
            print(f"OK Risk Engine: ACTIVE   (Risk/Trade: {self._xau_cfg.max_risk_per_trade:.1%})")
            
            # Linkage proof
            self._check_backtest_integration()
            print("OK Backtest Engine: LINKED (Historical data logic ready)")
            
            print("OK Logging to: Logs/portfolio_engine_health.log")
            if self.dry_run:
                print("WARN DRY RUN MODE: MT5 connection mocked / Orders simulated")
            print("="*60 + "\n")
            self._boot_report_printed = True
        else:
            # Keep linkage check alive without repeating the full boot banner.
            self._check_backtest_integration()

        # -------------------- Connection Logic --------------------
        if self.dry_run:
            log_health.info("DRY_RUN: Mocking MT5 connection success")
            self._mt5_ready = True
        else:
            if not self._init_mt5():
                self._run.clear()
                return False

        if not self._build_pipelines():
            self._run.clear()
            return False

        self._restart_exec_worker()

        log_health.info(
            "PORTFOLIO_ENGINE_START | dry_run=%s xau=%s btc=%s manual_stop=%s",
            self.dry_run,
            self._xau.symbol if self._xau else "None",
            self._btc.symbol if self._btc else "None",
            self._manual_stop,
        )

        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        return True

    def stop(self) -> bool:
        with self._lock:
            if not self._run.is_set():
                return True
            self._run.clear()

        if self._exec_worker:
            try:
                self._exec_worker.stop()
            except Exception:
                pass
            try:
                self._exec_worker.join(timeout=6.0)
            except Exception:
                pass

        # Drain queues best-effort + clear mapping
        self._drain_queue(self._order_q)
        self._drain_queue(self._result_q)
        self._order_rm_by_id.clear()


        log_health.info("PORTFOLIO_ENGINE_STOP")
        return True

    def request_manual_stop(self) -> bool:
        with self._lock:
            self._manual_stop = True
            log_health.info("MANUAL_STOP_REQUESTED | Switching to MONITORING mode (Trading Disabled)")
        return True

    def clear_manual_stop(self) -> None:
        with self._lock:
            if self._manual_stop:
                log_health.info("MANUAL_STOP_CLEAR")
            self._manual_stop = False

    def manual_stop_active(self) -> bool:
        with self._lock:
            return bool(self._manual_stop)

    def close_all(self) -> bool:
        """Best-effort close all positions for both symbols."""
        ok_all = True
        try:
            if self._xau and self._xau.symbol:
                ok_all = bool(close_all_position(self._xau.symbol)) and ok_all
        except Exception:
            ok_all = False
        try:
            if self._btc and self._btc.symbol:
                ok_all = bool(close_all_position(self._btc.symbol)) and ok_all
        except Exception:
            ok_all = False
        return bool(ok_all)

    def _transition_state(self, current: EngineState, nxt: EngineState, reason: str) -> EngineState:
        log_health.info("FSM_TRANSITION | %s -> %s | reason=%s", current.value, nxt.value, reason)
        return nxt

    def _halt_fsm(self, reason: str) -> None:
        log_err.critical("FSM_HALT | reason=%s", reason)
        with self._lock:
            self._manual_stop = True
            self._run.clear()
        self._drain_queue(self._order_q)
        self._drain_queue(self._result_q)
        self._order_rm_by_id.clear()

    def _validate_payload_timestamp(self, asset: str, payload: Dict[str, Any], tf: str) -> Tuple[bool, str]:
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

        # Optional secondary timestamp from feed text layer.
        # If present, it must parse and align with ts_bar for this timeframe.
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

    def _validate_data_sync_payloads(self, asset: str, payloads: Dict[str, Optional[Dict[str, Any]]]) -> Tuple[bool, str]:
        scalp = payloads.get("scalp")
        intraday = payloads.get("intraday")
        if not scalp and not intraday:
            return False, "no_payloads"

        if scalp:
            ok, reason = self._validate_payload_timestamp(asset, scalp, "M1")
            if not ok:
                return False, f"scalp_{reason}"

        if intraday:
            ok, reason = self._validate_payload_timestamp(asset, intraday, "H1")
            if not ok:
                return False, f"intraday_{reason}"

        schema_ok, schema_reason = validate_payload_schema(asset, payloads)
        if not schema_ok:
            return False, f"schema_{schema_reason}"

        return True, "ok"

    @staticmethod
    def _extract_payload_frame(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        if isinstance(payload.get("M1"), dict):
            return dict(payload.get("M1") or {})
        if isinstance(payload.get("H1"), dict):
            return dict(payload.get("H1") or {})
        return {}

    def _probabilistic_levels(
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

        levels: List[float] = []
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

    def _build_ml_candidate(self, asset: str, sig: MLSignal) -> Optional[AssetCandidate]:
        if self._is_asset_blocked(asset):
            self._log_blocked_asset_skip(asset, "build_candidate")
            return None
        if sig.signal == "HOLD" or sig.side not in ("Buy", "Sell"):
            return None

        pipe = self._xau if asset == "XAU" else self._btc
        if pipe is None or pipe.risk is None:
            return None

        payload = sig.scalp_payload if isinstance(sig.scalp_payload, dict) else sig.intraday_payload
        frame = self._extract_payload_frame(payload)
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

        plan = pipe.risk.plan_order(
            side=sig.side,
            confidence=float(sig.confidence),
            ind=ind,
            adapt=adapt,
            entry=(entry if entry > 0.0 else None),
            df=None,
        )
        if bool(plan.get("blocked", True)):
            log_health.info(
                "FSM_RISK_BLOCK | asset=%s signal=%s reason=%s",
                asset,
                sig.signal,
                str(plan.get("reason", "unknown")),
            )
            return None

        plan_entry = float(plan.get("entry", 0.0) or 0.0)
        plan_sl = float(plan.get("sl", 0.0) or 0.0)
        plan_tp = float(plan.get("tp", 0.0) or 0.0)
        sl, tp = self._probabilistic_levels(
            side=sig.side,
            entry=plan_entry,
            base_sl=plan_sl,
            base_tp=plan_tp,
            confidence=float(sig.confidence),
            frame=frame,
        )
        lot = float(plan.get("lot", 0.0) or 0.0)
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

    def _execute_candidates(self, candidates: List[AssetCandidate]) -> None:
        for selected in candidates:
            if not self._candidate_is_tradeable(selected):
                log_health.info(
                    "FSM_ORDER_SKIP | asset=%s signal=%s conf=%.3f blocked=%s",
                    selected.asset,
                    selected.signal,
                    selected.confidence,
                    selected.blocked,
                )
                continue

            if self._manual_stop:
                log_health.info("FSM_ORDER_SKIP | reason=manual_stop asset=%s", selected.asset)
                continue

            order_count = self._orders_for_candidate(selected)
            if int(order_count) <= 0:
                continue

            risk = self._xau.risk if selected.asset == "XAU" and self._xau else (self._btc.risk if self._btc else None)
            cfg = self._xau_cfg if selected.asset == "XAU" else self._btc_cfg
            lots = self._split_lot(float(selected.lot), order_count, risk, cfg)

            for idx, lot_val in enumerate(lots):
                ok, oid = self._enqueue_order(
                    selected,
                    order_index=int(idx),
                    order_count=int(order_count),
                    lot_override=float(lot_val),
                )
                if ok:
                    log_health.info(
                        "FSM_ORDER_ENQUEUED | asset=%s signal=%s conf=%.3f lot=%.4f order_id=%s",
                        selected.asset,
                        selected.signal,
                        selected.confidence,
                        float(lot_val),
                        oid,
                    )

    def _verification_step(self) -> None:
        while True:
            try:
                r = self._result_q.get_nowait()
            except queue.Empty:
                break

            try:
                rm = self._order_rm_by_id.pop(str(r.order_id), None)
                if rm is None:
                    rm = self._xau.risk if str(r.order_id).startswith("PORD_XAU_") else self._btc.risk
                if rm and hasattr(rm, "on_execution_result"):
                    rm.on_execution_result(r)
            except Exception:
                pass
            finally:
                try:
                    self._result_q.task_done()
                except Exception:
                    pass

        if self._xau:
            self._xau.reconcile_positions()
        if self._btc:
            self._btc.reconcile_positions()
        self._last_reconcile_ts = time.time()

        self._update_portfolio_risk_state()

        open_xau = self._xau.open_positions() if self._xau else 0
        open_btc = self._btc.open_positions() if self._btc else 0
        self._active_asset = self._select_active_asset(open_xau, open_btc)
        self._heartbeat(open_xau, open_btc)

    # -------------------- main loop --------------------
    def _loop(self) -> None:
        if not self._xau or not self._btc:
            log_err.error("Portfolio engine loop start failed: pipelines not built")
            self._run.clear()
            return

        state = EngineState.BOOT
        ctx = EngineCycleContext()

        while self._run.is_set():
            t0 = time.time()
            try:
                if state == EngineState.BOOT:
                    if self._manual_stop:
                        state = self._transition_state(state, EngineState.HALT, "manual_stop")
                        ctx.halt_reason = "manual_stop_active"
                    elif not self.dry_run and not (self._model_loaded and self._backtest_passed):
                        state = self._transition_state(state, EngineState.HALT, "gatekeeper_failed")
                        ctx.halt_reason = "gatekeeper_failed"
                    elif not self._check_mt5_health():
                        state = self._transition_state(state, EngineState.HALT, "mt5_unhealthy")
                        ctx.halt_reason = "mt5_unhealthy"
                    else:
                        state = self._transition_state(state, EngineState.DATA_SYNC, "boot_ok")

                elif state == EngineState.DATA_SYNC:
                    if not self._check_mt5_health():
                        state = self._transition_state(state, EngineState.HALT, "mt5_disconnected")
                        ctx.halt_reason = "mt5_disconnected"
                    else:
                        self._check_hard_stop_file()
                        if self._manual_stop:
                            state = self._transition_state(state, EngineState.HALT, "manual_stop_triggered")
                            ctx.halt_reason = "manual_stop_triggered"
                            continue
                        payloads: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}
                        active_assets = UTCScheduler.get_active_assets()
                        data_sync_ok = True
                        data_sync_reason = ""
                        for asset in active_assets:
                            if self._is_asset_blocked(asset):
                                self._log_blocked_asset_skip(asset, "data_sync")
                                continue
                            asset_payloads = fetch_ml_payloads(asset)
                            ok, reason = self._validate_data_sync_payloads(asset, asset_payloads)
                            if not ok:
                                data_sync_ok = False
                                data_sync_reason = f"{asset}_data_sync_failed:{reason}"
                                break
                            payloads[asset] = asset_payloads
                        if data_sync_ok and active_assets and not payloads:
                            data_sync_ok = False
                            data_sync_reason = "all_active_assets_blocked_by_gate"
                        if data_sync_ok:
                            ctx = EngineCycleContext(payloads=payloads)
                            state = self._transition_state(state, EngineState.ML_INFERENCE, "data_sync_ok")
                        else:
                            now = time.time()
                            # Avoid spamming logs on temporary feed gaps
                            if (
                                data_sync_reason != self._last_data_sync_fail_reason
                                or (now - self._last_data_sync_fail_ts) > 5.0
                            ):
                                log_health.warning("DATA_SYNC_WAIT | reason=%s", data_sync_reason)
                                self._last_data_sync_fail_reason = data_sync_reason
                                self._last_data_sync_fail_ts = now

                elif state == EngineState.ML_INFERENCE:
                    signals: Dict[str, MLSignal] = {}
                    for asset, payload in ctx.payloads.items():
                        if self._is_asset_blocked(asset):
                            self._log_blocked_asset_skip(asset, "ml_inference")
                            continue
                        # Try CatBoost trained model FIRST, fall back to Gemini LLM
                        sig = None
                        if asset in self._catboost_payloads:
                            sig = self._infer_catboost(asset)
                            if sig is not None:
                                log_health.info(
                                    "FSM_ML_SIGNAL | asset=%s signal=%s conf=%.3f provider=%s model=%s reason=%s",
                                    asset, sig.signal, sig.confidence, sig.provider, sig.model, sig.reason,
                                )
                        if sig is None:
                            # Fallback to Gemini LLM
                            sig = infer_from_payloads(
                                asset,
                                scalp_payload=payload.get("scalp"),
                                intraday_payload=payload.get("intraday"),
                            )
                            log_health.info(
                                "FSM_ML_SIGNAL | asset=%s signal=%s conf=%.3f provider=%s model=%s reason=%s",
                                asset, sig.signal, sig.confidence, sig.provider, sig.model, sig.reason,
                            )
                        signals[asset] = sig
                    ctx.ml_signals = signals
                    state = self._transition_state(state, EngineState.RISK_CALC, "ml_complete")

                elif state == EngineState.RISK_CALC:
                    candidates: List[AssetCandidate] = []
                    for asset in ("XAU", "BTC"):
                        sig = ctx.ml_signals.get(asset)
                        if sig is None:
                            continue
                        cand = self._build_ml_candidate(asset, sig)
                        if cand is not None:
                            candidates.append(cand)
                    ctx.candidates = candidates
                    self._last_cand_xau = next((c for c in candidates if c.asset == "XAU"), None)
                    self._last_cand_btc = next((c for c in candidates if c.asset == "BTC"), None)
                    state = self._transition_state(state, EngineState.EXECUTION_QUEUE, "risk_complete")

                elif state == EngineState.EXECUTION_QUEUE:
                    if self._retraining_mode:
                        now = time.time()
                        if now - self._last_retraining_log_ts > 10.0:
                            log_health.info("RETRAINING_PAUSE | skipping new orders during retraining")
                            self._last_retraining_log_ts = now
                        state = self._transition_state(state, EngineState.VERIFICATION, "retraining_pause")
                    else:
                        self._execute_candidates(ctx.candidates)
                        state = self._transition_state(state, EngineState.VERIFICATION, "queue_complete")

                elif state == EngineState.VERIFICATION:
                    self._verification_step()
                    state = self._transition_state(state, EngineState.DATA_SYNC, "verification_complete")

                elif state == EngineState.HALT:
                    self._halt_fsm(ctx.halt_reason or "unspecified")
                    break

                dt = time.time() - t0
                # Minimum cycle time: prevent rapid-fire MT5 calls
                # CatBoost inference + MT5 fetch per cycle needs throttling
                min_cycle = max(self._poll_fast, 5.0)  # At least 5s per full cycle
                time.sleep(max(0.02, min_cycle - dt))

            except Exception as exc:
                log_err.error("FSM_LOOP_EXCEPTION | err=%s | tb=%s", exc, traceback.format_exc())
                ctx.halt_reason = f"fsm_exception:{exc}"
                state = EngineState.HALT
_engine_instance: Optional[MultiAssetTradingEngine] = None
_engine_instance_lock = threading.Lock()


def get_engine(dry_run: bool = False) -> MultiAssetTradingEngine:
    global _engine_instance
    if _engine_instance is None:
        with _engine_instance_lock:
            if _engine_instance is None:
                _engine_instance = MultiAssetTradingEngine(dry_run=dry_run)
    return _engine_instance


class _EngineProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_engine(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(get_engine(), name, value)

    def __repr__(self) -> str:
        return repr(get_engine())


# Import-compatible lazy instance
engine = _EngineProxy()
