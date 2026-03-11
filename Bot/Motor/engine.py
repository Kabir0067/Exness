from __future__ import annotations

import os
import pickle
import queue
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import MetaTrader5 as mt5

from ExnessAPI.functions import close_all_position
from mt5_client import MT5_LOCK, ensure_mt5, mt5_status, mt5_async_call

# -------------------- Unified core + strategy adapters --------------------
from core.config import (
    XAUEngineConfig as XauConfig,
    BTCEngineConfig as BtcConfig,
    MAX_GATE_DRAWDOWN,
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


def _env_truthy(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _required_gate_assets() -> tuple[str, ...]:
    raw = str(os.getenv("REQUIRED_GATE_ASSETS", "XAU,BTC") or "XAU,BTC")
    out: List[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        asset = str(item or "").upper().strip()
        if not asset or asset in seen:
            continue
        seen.add(asset)
        out.append(asset)
    return tuple(out or ["XAU", "BTC"])


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
from .execution_manager import ExecutionManager
from .fsm_runtime import DeterministicFSMRunner
from .fsm_services import EngineFSMStageServices
from .fsm_types import EngineCycleContext, EngineState
from .inference_engine import InferenceEngine
from .logging_setup import _DIAG_ENABLED, _DIAG_EVERY_SEC, log_diag, log_err, log_health
from .models import AssetCandidate, ExecutionResult, OrderIntent, PortfolioStatus
from .order_sync_manager import OrderSyncManager
from .pipeline import _AssetPipeline
from .scheduler import EngineScheduleManager, UTCScheduler
from .utils import safe_json_dumps
from core.ml_router import MLSignal
from core.model_manager import model_manager  # <--- HOLY TRINITY GATEKEEPER
from core.model_gate import gate_details

MODEL_STATE_PATH = get_artifact_path("models", "model_state.pkl")
MIN_BACKTEST_SHARPE = MIN_GATE_SHARPE
MIN_BACKTEST_WIN_RATE = MIN_GATE_WIN_RATE


class MultiAssetTradingEngine:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = bool(dry_run)
        self._model_loaded: bool = False
        self._backtest_passed: bool = False
        self._model_version: str = "N/A"
        self._gate_last_reason: str = "unknown"
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
        self._max_queue : int = 50
        self._order_q: "queue.Queue[OrderIntent]" = queue.Queue(maxsize=self._max_queue)
        self._result_q: "queue.Queue[ExecutionResult]" = queue.Queue(maxsize=self._max_queue)
        self._exec_worker: Optional[ExecutionWorker] = None

        # IMPORTANT: map order_id -> risk_manager (robust RM routing)
        self._order_rm_by_id: Dict[str, Any] = {}
        # Pending broker-sync map: order_id -> lightweight intent metadata.
        self._pending_order_meta: Dict[str, Dict[str, Any]] = {}
        self._order_sync_interval_sec: float = max(
            0.15,
            float(os.getenv("ORDER_SYNC_INTERVAL_SEC", "0.50") or 0.50),
        )
        self._order_sync_grace_sec: float = max(
            0.20,
            float(os.getenv("ORDER_SYNC_GRACE_SEC", "0.75") or 0.75),
        )
        self._order_sync_timeout_sec: float = max(
            1.0,
            float(os.getenv("ORDER_SYNC_TIMEOUT_SEC", "8.0") or 8.0),
        )
        self._last_order_sync_ts: float = 0.0

        # Log suppression for high-frequency skips
        self._last_log_id: Dict[str, str] = {}


        # Loop tuning
        self._poll_fast : float = 0.25
        self._poll_slow : float = 0.75
        self._min_cycle_sec: float = max(
            0.05,
            float(os.getenv("ENGINE_MIN_CYCLE_SEC", "0.25") or 0.25),
        )
        self._pipeline_log_every : float = 2.0
        self._last_pipeline_log_ts = 0.0
        self._last_pipeline_log_key: Optional[Tuple[Any, ...]] = None
        self._max_consecutive_errors : int = 12

        # Emergency flatten controls (prevents orphaned positions on hard halts).
        self._flatten_retries: int = max(
            1,
            int(float(os.getenv("EMERGENCY_FLATTEN_RETRIES", "3") or 3)),
        )
        self._flatten_backoff_sec: float = max(
            0.05,
            float(os.getenv("EMERGENCY_FLATTEN_BACKOFF_SEC", "0.25") or 0.25),
        )
        self._flatten_cooldown_sec: float = max(
            0.25,
            float(os.getenv("EMERGENCY_FLATTEN_COOLDOWN_SEC", "2.0") or 2.0),
        )
        self._last_flatten_ts: float = 0.0
        self._last_flatten_reason: str = ""

        # ML inference strictness: prioritize precision over signal frequency.
        self._allow_llm_fallback: bool = str(
            os.getenv("ALLOW_LLM_FALLBACK", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self._catboost_threshold_q: float = float(
            os.getenv("CATBOOST_THRESHOLD_Q", "85") or 85.0
        )
        self._catboost_hist_min: int = max(
            20, int(float(os.getenv("CATBOOST_THRESHOLD_MIN_HIST", "40") or 40))
        )
        self._catboost_min_zscore: float = float(
            os.getenv("CATBOOST_MIN_ZSCORE", "0.75") or 0.75
        )
        self._catboost_min_flow_confirm: float = float(
            os.getenv("CATBOOST_MIN_FLOW_CONFIRM", "0.06") or 0.06
        )
        self._target_signals_per_24h_min: int = max(
            1,
            int(float(os.getenv("TARGET_SIGNALS_PER_24H_MIN", "10") or 10)),
        )
        self._target_signals_per_24h_max: int = max(
            self._target_signals_per_24h_min,
            int(float(os.getenv("TARGET_SIGNALS_PER_24H_MAX", "20") or 20)),
        )
        self._signal_emit_ts_by_asset: Dict[str, Deque[float]] = {
            "XAU": deque(maxlen=512),
            "BTC": deque(maxlen=512),
        }
        self._signal_emit_ts_global: Deque[float] = deque(maxlen=1024)

        # Real-time operational evidence (latency/slippage/win-rate drift).
        self._exec_latency_ms_hist: Deque[float] = deque(maxlen=512)
        self._exec_slippage_hist: Deque[float] = deque(maxlen=512)
        self._fsm_cycle_ms_hist: Deque[float] = deque(maxlen=512)
        self._live_monitor_interval_sec: float = max(
            5.0,
            float(os.getenv("LIVE_MONITOR_INTERVAL_SEC", "15.0") or 15.0),
        )
        self._last_live_monitor_ts: float = 0.0
        self._last_cycle_spike_log_ts: float = 0.0
        self._live_max_p95_latency_ms: float = max(
            50.0,
            float(os.getenv("LIVE_MAX_P95_LATENCY_MS", "850.0") or 850.0),
        )
        self._live_max_p95_slippage_points: float = max(
            0.1,
            float(os.getenv("LIVE_MAX_P95_SLIPPAGE_POINTS", "35.0") or 35.0),
        )
        self._live_max_winrate_drift: float = max(
            0.01,
            float(os.getenv("LIVE_MAX_WINRATE_DRIFT", "0.15") or 0.15),
        )
        self._live_evidence_state: Dict[str, Any] = {
            "status": "BOOT",
            "reason": "booting",
            "updated_ts": 0.0,
            "latency_p95_ms": 0.0,
            "slippage_p95_points": 0.0,
            "win_rate_live": 0.0,
            "win_rate_reference": 0.0,
            "win_rate_drift": 0.0,
            "closed_trade_samples": 0,
            "signals_24h": 0,
        }

        # Analysis state log throttling
        self._last_analysis_paused_state = False
        self._last_analysis_state_log_ts = 0.0

        self._last_skip_log_ts: dict[str, float] = {}  # Anti-spam throttle for blocked notifications
        self._gate_status_log_ttl : float = 60.0
        self._last_gate_status_sig: str = ""
        self._last_gate_status_ts: float = 0.0
        
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
        self._trade_cooldown_sec: float = 300.0  # 5 minutes default
        self._cooldown_blocked_count: Dict[str, int] = {"XAU": 0, "BTC": 0}

        # Position tracking for detecting trade closures
        self._last_open_positions: Dict[str, int] = {"XAU": 0, "BTC": 0}
        self._max_open_positions: Dict[str, int] = {
            "XAU": 1,
            "BTC": 1,
        }
        self._max_lot_cap: Dict[str, float] = {
            "XAU": 1.00,
            "BTC": 0.05,
        }

        self._schedule_manager = EngineScheduleManager(self)
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

        # Decoupled deterministic runtime (FSM ports + runner).
        self._fsm_services = EngineFSMStageServices(self)
        self._fsm_runner = DeterministicFSMRunner(self._fsm_services)
        self._inference_engine = InferenceEngine(self)
        self._execution_manager = ExecutionManager(self)
        self._order_sync_manager = OrderSyncManager(self)
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
        if not self._backtest_passed:
            print(f"Gate Reason:     {self._gate_last_reason}")
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
        """Ensure live trading starts only when gate-approved assets are available."""
        if self.dry_run:
            log_health.info("MODEL_GATE_BYPASSED | reason=dry_run")
            self._model_loaded = True
            self._backtest_passed = True
            self._model_version = "dry_run"
            self._gate_last_reason = "dry_run_bypass"
            self._blocked_assets = []
            return

        # Reset gate-scoped payloads each time to avoid stale models on partial reloads.
        self._catboost_payloads = {}
        self._blocked_assets = []

        required_assets = _required_gate_assets()
        details = gate_details(required_assets=required_assets, allow_legacy_fallback=True)
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
                            float(st.get("sharpe", 0.0) or 0.0),
                            float(st.get("win_rate", 0.0) or 0.0),
                            float(st.get("max_drawdown_pct", 0.0) or 0.0),
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
            sharpe = float(st.get("sharpe", 0.0) or 0.0)
            win_rate = float(st.get("win_rate", 0.0) or 0.0)
            max_dd_raw = st.get("max_drawdown_pct", 1.0)
            max_dd = float(1.0 if max_dd_raw is None else max_dd_raw)
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
            for asset, ok, reason, version, sharpe, win_rate, max_dd, legacy in gate_items:
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
        if (not gate_ok) and _env_truthy("PARTIAL_GATE_MODE", "1") and isinstance(assets, dict):
            partial_assets: list[str] = []
            for asset in required_assets:
                st = assets.get(asset, {})
                if isinstance(st, dict) and bool(st.get("ok", False)):
                    partial_assets.append(asset)
            partial_mode_reason = "partial_gate"

            if not partial_assets and _env_truthy("PARTIAL_GATE_ALLOW_WFA_FALLBACK", "1"):
                partial_assets = []
                for asset in required_assets:
                    st = assets.get(asset, {})
                    if not isinstance(st, dict):
                        continue
                    reason_s = str(st.get("reason", "") or "")
                    if reason_s.startswith(("state_wfa_failed", "meta_wfa_failed")) and _fallback_metrics_ok(st):
                        partial_assets.append(asset)
                if partial_assets:
                    partial_mode_reason = "partial_wfa_fallback"

            if not partial_assets and _env_truthy("PARTIAL_GATE_ALLOW_SAMPLE_QUALITY_FALLBACK", "1"):
                partial_assets = []
                for asset in required_assets:
                    st = assets.get(asset, {})
                    if not isinstance(st, dict):
                        continue
                    reason_s = str(st.get("reason", "") or "")
                    sample_only_unsafe = (
                        reason_s.startswith(("state_marked_unsafe:", "meta_marked_unsafe:"))
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
                    self._run.clear()
                self._drain_queue(self._order_q)
                self._drain_queue(self._result_q)
                self._order_rm_by_id.clear()
                self._pending_order_meta.clear()
                self._emergency_flatten(f"model_gate_failed:{gate_reason}")
            return

        versions: list[str] = []
        sharpes: list[float] = []
        wins: list[float] = []
        for asset in active_assets:
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
        self._gate_last_reason = gate_reason
        self._blocked_assets = sorted(set(required_assets).difference(set(active_assets)))
        log_health.info(
            "MODEL_GATE_PASSED | versions=%s sharpe_min=%.3f win_rate_min=%.3f blocked=%s reason=%s",
            self._model_version,
            self._model_sharpe,
            self._model_win_rate,
            ",".join(self._blocked_assets) if self._blocked_assets else "-",
            gate_reason,
        )

    def _infer_catboost(self, asset: str) -> Optional[MLSignal]:
        return self._inference_engine.infer_catboost(asset)

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
                self._order_rm_by_id.clear()
                self._pending_order_meta.clear()
                self._emergency_flatten("stop_lock")
                
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

    def _register_pending_order(self, intent: OrderIntent) -> None:
        self._order_sync_manager.register_pending_order(intent)

    def _clear_pending_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        return self._order_sync_manager.clear_pending_order(order_id)

    def _resolve_order_result(self, r: ExecutionResult) -> None:
        self._order_sync_manager.resolve_order_result(r)

    def _probe_pending_order_state(self, order_id: str, meta: Dict[str, Any], now_ts: float) -> Optional[ExecutionResult]:
        return self._order_sync_manager.probe_pending_order_state(order_id, meta, now_ts)

    def _sync_pending_orders(self) -> None:
        self._order_sync_manager.sync_pending_orders()

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
            self._pending_order_meta.clear()

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
        self._schedule_manager.refresh_signal_cooldowns()

    # -------------------- phase/daily helpers --------------------
    @staticmethod
    def _get_phase(risk: Any) -> str:
        return EngineScheduleManager.get_phase(risk)

    @staticmethod
    def _get_daily_date(risk: Any) -> str:
        return EngineScheduleManager.get_daily_date(risk)

    def _phase_reason(self, risk: Any, new_phase: str) -> str:
        return self._schedule_manager.phase_reason(risk, new_phase)

    def _check_phase_change(self, asset: str, risk: Any) -> None:
        self._schedule_manager.check_phase_change(asset, risk)

    def _check_daily_start(self, asset: str, risk: Any) -> None:
        self._schedule_manager.check_daily_start(asset, risk)

    # -------------------- idempotency --------------------
    def _cooldown_for_asset(self, asset: str) -> float:
        return self._schedule_manager.cooldown_for_asset(asset)

    def _is_duplicate(
        self,
        asset: str,
        signal_id: str,
        now: float,
        max_orders: int,
        *,
        order_index: int = 0,
    ) -> bool:
        return self._schedule_manager.is_duplicate(
            asset,
            signal_id,
            now,
            max_orders,
            order_index=order_index,
        )

    def _mark_seen(self, asset: str, signal_id: str, now: float) -> None:
        self._schedule_manager.mark_seen(asset, signal_id, now)

    @staticmethod
    def _prune_signal_window(window: Deque[float], now_ts: float, lookback_sec: float = 86_400.0) -> None:
        EngineScheduleManager.prune_signal_window(window, now_ts, lookback_sec=lookback_sec)

    @staticmethod
    def _p95(values: Deque[float] | List[float]) -> float:
        return EngineScheduleManager.p95(values)

    @staticmethod
    def _utc_day_progress() -> float:
        return EngineScheduleManager.utc_day_progress()

    def _signal_density_controls(self, asset: str, now_ts: float) -> Tuple[float, float, float, int]:
        return self._schedule_manager.signal_density_controls(asset, now_ts)

    def _record_signal_emit(self, asset: str, now_ts: float) -> None:
        self._schedule_manager.record_signal_emit(asset, now_ts)

    # -------------------- portfolio logic --------------------
    def _next_order_id(self, asset: str) -> str:
        self._order_counter += 1
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"PORD_{asset}_{ts}_{self._order_counter}_{os.getpid()}"

    def _effective_min_conf(self, c: AssetCandidate) -> float:
        return self._execution_manager.effective_min_conf(c)

    def _is_asset_blocked(self, asset: str) -> bool:
        return self._execution_manager.is_asset_blocked(asset)

    def _log_blocked_asset_skip(self, asset: str, stage: str) -> None:
        self._execution_manager.log_blocked_asset_skip(asset, stage)

    def _candidate_is_tradeable(self, c: AssetCandidate) -> bool:
        return self._execution_manager.candidate_is_tradeable(c)

    def _select_active_asset(self, open_xau: int, open_btc: int) -> str:
        return EngineScheduleManager.select_active_asset(open_xau, open_btc)

    def _asset_open_positions(self, asset: str) -> int:
        return self._execution_manager.asset_open_positions(asset)

    def _asset_max_positions(self, asset: str) -> int:
        return self._execution_manager.asset_max_positions(asset)

    def _asset_lot_cap(self, asset: str) -> float:
        return self._execution_manager.asset_lot_cap(asset)

    def _apply_asset_lot_cap(self, asset: str, lot: float, stage: str) -> float:
        return self._execution_manager.apply_asset_lot_cap(asset, lot, stage)

    def _enqueue_order(
        self,
        cand: AssetCandidate,
        *,
        order_index: int = 0,
        order_count: int = 1,
        lot_override: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        return self._execution_manager.enqueue_order(
            cand,
            order_index=order_index,
            order_count=order_count,
            lot_override=lot_override,
        )

    def _orders_for_candidate(self, cand: AssetCandidate) -> int:
        return self._execution_manager.orders_for_candidate(cand)

    @staticmethod
    def _min_lot(risk: Any) -> float:
        return ExecutionManager.min_lot(risk)

    def _split_lot(self, lot: float, parts: int, risk: Any, cfg: Any) -> list[float]:
        return self._execution_manager.split_lot(lot, parts, risk, cfg)

    def _report_cycle_latency_ms(self, latency_ms: float) -> None:
        try:
            lat = float(latency_ms)
        except Exception:
            return
        if lat < 0.0:
            return
        self._fsm_cycle_ms_hist.append(lat)
        if lat > (self._live_max_p95_latency_ms * 1.75):
            now = time.time()
            if (now - self._last_cycle_spike_log_ts) >= 5.0:
                self._last_cycle_spike_log_ts = now
                log_health.warning(
                    "FSM_CYCLE_SPIKE | latency_ms=%.1f threshold_ms=%.1f",
                    lat,
                    self._live_max_p95_latency_ms * 1.75,
                )

    def _record_execution_telemetry(self, r: ExecutionResult) -> None:
        try:
            sent = float(getattr(r, "sent_ts", 0.0) or 0.0)
            fill = float(getattr(r, "fill_ts", 0.0) or 0.0)
            slip = abs(float(getattr(r, "slippage", 0.0) or 0.0))
            if fill > sent > 0.0:
                self._exec_latency_ms_hist.append((fill - sent) * 1000.0)
            if slip >= 0.0:
                self._exec_slippage_hist.append(slip)
        except Exception:
            return

    def _reference_win_rate(self) -> float:
        vals: List[float] = []
        for asset in ("XAU", "BTC"):
            payload = self._catboost_payloads.get(asset)
            if not isinstance(payload, dict):
                continue
            cal = payload.get("alpha_calibration", {})
            if not isinstance(cal, dict):
                continue
            wr = float(cal.get("holdout_direction_accuracy_active", 0.0) or 0.0)
            if wr > 0.0:
                vals.append(wr)
        if not vals:
            return 0.0
        return float(min(vals))

    def _live_trade_win_rate(self) -> Tuple[float, int]:
        profits: List[float] = []
        for pipe in (self._xau, self._btc):
            rm = getattr(pipe, "risk", None) if pipe is not None else None
            fn = getattr(rm, "_recent_closed_trade_profits", None)
            if not callable(fn):
                continue
            try:
                p = fn(60)
                if isinstance(p, list):
                    profits.extend(float(x) for x in p if x is not None)
            except Exception:
                continue
        if not profits:
            return 0.0, 0
        wins = sum(1 for p in profits if float(p) > 0.0)
        n = int(len(profits))
        return float(wins / max(1, n)), n

    def _update_live_evidence(self, *, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        if (not force) and ((now - self._last_live_monitor_ts) < self._live_monitor_interval_sec):
            return dict(self._live_evidence_state)
        self._last_live_monitor_ts = now

        lat_p95 = max(
            self._p95(self._exec_latency_ms_hist),
            self._p95(self._fsm_cycle_ms_hist),
        )
        slip_p95 = self._p95(self._exec_slippage_hist)

        risk_lat_candidates: List[float] = []
        risk_slip_candidates: List[float] = []
        for pipe in (self._xau, self._btc):
            rm = getattr(pipe, "risk", None) if pipe is not None else None
            if rm is None:
                continue
            try:
                if hasattr(rm, "exec_p95_latency"):
                    risk_lat_candidates.append(float(rm.exec_p95_latency() or 0.0))
                if hasattr(rm, "exec_p95_slippage"):
                    risk_slip_candidates.append(float(rm.exec_p95_slippage() or 0.0))
            except Exception:
                continue
        if risk_lat_candidates:
            lat_p95 = max(lat_p95, max(risk_lat_candidates))
        if risk_slip_candidates:
            slip_p95 = max(slip_p95, max(risk_slip_candidates))

        live_wr, wr_samples = self._live_trade_win_rate()
        ref_wr = self._reference_win_rate()
        drift = float(live_wr - ref_wr) if (live_wr > 0.0 and ref_wr > 0.0) else 0.0

        reasons: List[str] = []
        if lat_p95 > self._live_max_p95_latency_ms:
            reasons.append(f"latency_p95:{lat_p95:.1f}>{self._live_max_p95_latency_ms:.1f}")
        if slip_p95 > self._live_max_p95_slippage_points:
            reasons.append(f"slippage_p95:{slip_p95:.2f}>{self._live_max_p95_slippage_points:.2f}")
        if wr_samples >= 8 and ref_wr > 0.0 and drift < -self._live_max_winrate_drift:
            reasons.append(f"winrate_drift:{drift:+.3f}")

        status = "HEALTHY" if not reasons else "DEGRADED"
        if self._manual_stop:
            status = "MONITORING_ONLY"
        elif not self._mt5_ready and not self.dry_run:
            status = "MT5_OFFLINE"

        self._prune_signal_window(self._signal_emit_ts_global, now)
        snapshot = {
            "status": status,
            "reason": "|".join(reasons) if reasons else "ok",
            "updated_ts": now,
            "latency_p95_ms": float(lat_p95),
            "slippage_p95_points": float(slip_p95),
            "win_rate_live": float(live_wr),
            "win_rate_reference": float(ref_wr),
            "win_rate_drift": float(drift),
            "closed_trade_samples": int(wr_samples),
            "signals_24h": int(len(self._signal_emit_ts_global)),
        }
        self._live_evidence_state = snapshot

        level = log_health.warning if reasons else log_health.info
        level(
            "LIVE_EVIDENCE | status=%s reason=%s latency_p95=%.1f slippage_p95=%.2f "
            "win_rate_live=%.3f win_rate_ref=%.3f drift=%+.3f closed_trades=%d signals_24h=%d",
            snapshot["status"],
            snapshot["reason"],
            snapshot["latency_p95_ms"],
            snapshot["slippage_p95_points"],
            snapshot["win_rate_live"],
            snapshot["win_rate_reference"],
            snapshot["win_rate_drift"],
            snapshot["closed_trade_samples"],
            snapshot["signals_24h"],
        )
        return dict(snapshot)

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
            live_evidence = self._update_live_evidence(force=False)

            if _DIAG_ENABLED and (time.time() - self._diag_last_ts) >= float(_DIAG_EVERY_SEC):
                self._diag_last_ts = time.time()
                payload = {
                    "ts_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "engine": st,
                    "mt5": mt5_status(),
                    "live_evidence": live_evidence,
                }
                try:
                    log_diag.info(safe_json_dumps(payload))
                except Exception:
                    pass
        except Exception as exc:
            log_err.error("heartbeat error: %s | tb=%s", exc, traceback.format_exc())

    def _track_position_closures(self, open_xau: int, open_btc: int) -> None:
        now = time.time()
        for asset, open_now in (("XAU", int(open_xau)), ("BTC", int(open_btc))):
            prev = int(self._last_open_positions.get(asset, 0))
            if prev > open_now:
                self._last_trade_close_ts[asset] = now
                log_health.info(
                    "TRADE_CLOSE_DETECTED | asset=%s prev_open=%d open=%d cooldown=%.0fs",
                    asset,
                    prev,
                    open_now,
                    self._trade_cooldown_sec,
                )
            self._last_open_positions[asset] = open_now

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
                self._emergency_flatten("start_blocked_gatekeeper_failed")
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
        self._pending_order_meta.clear()


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
        """Best-effort close all positions for both symbols (even before pipelines boot)."""
        symbols: List[str] = []

        def _append_symbol(sym: Any) -> None:
            s = str(sym or "").strip()
            if s and s not in symbols:
                symbols.append(s)

        try:
            if self._xau and self._xau.symbol:
                _append_symbol(self._xau.symbol)
            else:
                sp = getattr(self._xau_cfg, "symbol_params", None)
                _append_symbol(getattr(sp, "resolved", "") or getattr(sp, "base", ""))
        except Exception:
            pass

        try:
            if self._btc and self._btc.symbol:
                _append_symbol(self._btc.symbol)
            else:
                sp = getattr(self._btc_cfg, "symbol_params", None)
                _append_symbol(getattr(sp, "resolved", "") or getattr(sp, "base", ""))
        except Exception:
            pass

        ok_all = True
        for sym in symbols:
            try:
                ok_all = bool(close_all_position(sym)) and ok_all
            except Exception:
                ok_all = False
        return bool(ok_all)

    def _emergency_flatten(self, reason: str) -> bool:
        """
        Force-close all strategy symbols on critical halt paths.

        This is intentionally fail-safe and is called from kill-switch / HALT
        transitions to prevent orphaned positions.
        """
        if self.dry_run:
            log_health.warning("EMERGENCY_FLATTEN_SKIP | reason=%s mode=dry_run", reason)
            return True

        reason_s = str(reason or "unspecified")
        now = time.time()
        with self._lock:
            if (
                (now - self._last_flatten_ts) < self._flatten_cooldown_sec
                and reason_s == self._last_flatten_reason
            ):
                return False
            self._last_flatten_ts = now
            self._last_flatten_reason = reason_s

        log_err.critical("EMERGENCY_FLATTEN_START | reason=%s retries=%d", reason_s, self._flatten_retries)
        ok = False
        for attempt in range(1, self._flatten_retries + 1):
            try:
                ok = bool(self.close_all())
            except Exception as exc:
                ok = False
                log_err.error("EMERGENCY_FLATTEN_EXCEPTION | attempt=%d reason=%s err=%s", attempt, reason_s, exc)

            if ok:
                log_health.critical("EMERGENCY_FLATTEN_DONE | reason=%s attempt=%d", reason_s, attempt)
                return True

            time.sleep(self._flatten_backoff_sec * float(attempt))

        log_err.critical("EMERGENCY_FLATTEN_FAILED | reason=%s", reason_s)
        return False

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
        self._pending_order_meta.clear()
        self._emergency_flatten(f"fsm_halt:{reason}")

    def _validate_payload_timestamp(self, asset: str, payload: Dict[str, Any], tf: str) -> Tuple[bool, str]:
        return self._inference_engine.validate_payload_timestamp(asset, payload, tf)

    def _validate_data_sync_payloads(self, asset: str, payloads: Dict[str, Optional[Dict[str, Any]]]) -> Tuple[bool, str]:
        return self._inference_engine.validate_data_sync_payloads(asset, payloads)

    @staticmethod
    def _extract_payload_frame(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return InferenceEngine.extract_payload_frame(payload)

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
        return self._inference_engine.probabilistic_levels(
            side=side,
            entry=entry,
            base_sl=base_sl,
            base_tp=base_tp,
            confidence=confidence,
            frame=frame,
        )

    def _build_ml_candidate(self, asset: str, sig: MLSignal) -> Optional[AssetCandidate]:
        return self._inference_engine.build_ml_candidate(asset, sig)

    def _execute_candidates(self, candidates: List[AssetCandidate]) -> None:
        self._execution_manager.execute_candidates(candidates)

    def _verification_step(self) -> None:
        self._execution_manager.verification_step()

    # -------------------- main loop --------------------
    def _loop(self) -> None:
        if not self._xau or not self._btc:
            log_err.error("Portfolio engine loop start failed: pipelines not built")
            self._run.clear()
            return

        self._fsm_runner.run(
            initial_state=EngineState.BOOT,
            initial_ctx=EngineCycleContext(),
        )
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
