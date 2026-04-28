"""
Core engine entry points and runtime coordination helpers.

Wires the multi-asset engine with inference, scheduling, execution,
and finite-state orchestration services.
"""

from __future__ import annotations

import os
import queue
import signal
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from core.config import (
    BTCEngineConfig as BtcConfig,
    MIN_GATE_SHARPE,
    MIN_GATE_WIN_RATE,
    XAUEngineConfig as XauConfig,
    get_config_from_env as _get_core_config,
)
from core.feature_engine import FeatureEngine
from core.portfolio_risk import PortfolioRiskManager
from core.risk_manager import RiskManager
from core.signal_engine import SignalEngine

from .execution import (
    EngineExecutionMixin,
    ExecutionManager,
    ExecutionWorker,
    OrderSyncManager,
)
from .fsm import DeterministicFSMRunner, EngineFSMStageServices, EngineRuntimeMixin
from .inference import EngineModelMixin, InferenceEngine
from .models import (
    AssetCandidate,
    ExecutionResult,
    MODEL_STATE_PATH,
    MONITORING_ONLY_HALT_REASONS,
    OrderIntent,
    RECOVERABLE_HALT_REASONS,
    _monitoring_only_mode,
    log_err,
    log_health,
)
from .pipeline import (
    EnginePipelineMixin,
    EngineScheduleManager,
    _AssetPipeline,
)

# Imports are grouped once at the module top to keep startup predictable.


def get_xau_config():
    return _get_core_config("XAU")


def get_btc_config():
    return _get_core_config("BTC")


def _apply_high_accuracy_mode(cfg, enable=True):
    """Mark configs so downstream code can detect high-accuracy profile use."""
    try:
        setattr(cfg, "high_accuracy_mode", bool(enable))
    except Exception:
        pass
    return cfg


xau_apply_high_accuracy_mode = _apply_high_accuracy_mode
btc_apply_high_accuracy_mode = _apply_high_accuracy_mode

# Aliases so _build_pipelines keeps working with same names
XauFeatureEngine = FeatureEngine
BtcFeatureEngine = FeatureEngine
XauRiskManager = RiskManager
BtcRiskManager = RiskManager
XauSignalEngine = SignalEngine
BtcSignalEngine = SignalEngine

MIN_BACKTEST_SHARPE = MIN_GATE_SHARPE
MIN_BACKTEST_WIN_RATE = MIN_GATE_WIN_RATE


class MultiAssetTradingEngine(
    EngineModelMixin,
    EnginePipelineMixin,
    EngineExecutionMixin,
    EngineRuntimeMixin,
):
    def __init__(self, dry_run: bool = False) -> None:
        # SAFE IMPROVEMENT: Fixed PEP 8 violations in all variable annotations
        # (removed extraneous spaces before colons). Purely stylistic.
        self.dry_run = bool(dry_run)
        self._model_loaded: bool = False
        self._backtest_passed: bool = False
        self._model_version: str = "N/A"
        self._gate_last_reason: str = "unknown"
        self._catboost_payloads: Dict[str, Any] = {}  # asset -> {model, pipeline}
        self._catboost_pred_history: Dict[str, Deque[float]] = (
            {}
        )  # asset -> rolling predictions
        self._catboost_signal_cache: Dict[str, Dict[str, Any]] = (
            {}
        )  # asset -> {"ts_bar": int, "signal": MLSignal}
        self._catboost_signal_cache_lock = threading.RLock()
        self._blocked_assets: List[str] = []
        self._model_sharpe: float = 0.0
        self._model_win_rate: float = 0.0
        self._backtest_linkage_verified: bool = False
        self._boot_report_printed: bool = False
        self._boot_ts: float = time.time()
        self._last_chaos_audit: Dict[str, Any] = {}
        self._last_housekeeping_ts: float = 0.0
        self._housekeeping_interval_sec: float = max(
            15.0,
            float(os.getenv("ENGINE_HOUSEKEEPING_SEC", "60.0") or 60.0),
        )
        self._log_growth_limit_bytes: int = max(
            8 * 1024 * 1024,
            int(
                float(os.getenv("ENGINE_LOG_GROWTH_LIMIT_BYTES", str(32 * 1024 * 1024)))
                or (32 * 1024 * 1024)
            ),
        )
        # ------------------------------------------------------------
        # QUANTUM ARCHITECTURE: SYSTEM MATRIX
        # ------------------------------------------------------------
        # self._print_system_matrix() <-- Moved to explicit call
        # ------------------------------------------------------------
        self._run = threading.Event()
        self._lock = threading.Lock()
        self._order_state_lock = threading.RLock()
        self._fsm_transition_log_lock = threading.RLock()
        self._fsm_transition_log_cache: Dict[str, float] = {}
        self._build_pipelines_lock = (
            threading.Lock()
        )  # serializes _build_pipelines calls
        self._starting = (
            False  # guarded by _lock; prevents duplicate loop thread creation
        )
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_started_ts: float = 0.0
        self._last_runtime_heartbeat_ts: float = 0.0
        self._runtime_start_grace_sec: float = max(
            3.0,
            float(os.getenv("ENGINE_RUNTIME_GRACE_SEC", "8.0") or 8.0),
        )
        self._runtime_stall_timeout_sec: float = max(
            6.0,
            float(os.getenv("ENGINE_RUNTIME_STALL_SEC", "20.0") or 20.0),
        )
        runtime_first_cycle_grace_default = max(
            35.0,
            self._runtime_stall_timeout_sec * 1.75,
        )
        self._runtime_first_cycle_grace_sec: float = max(
            self._runtime_stall_timeout_sec,
            float(
                os.getenv(
                    "ENGINE_FIRST_CYCLE_GRACE_SEC",
                    str(runtime_first_cycle_grace_default),
                )
                or runtime_first_cycle_grace_default
            ),
        )
        self._manual_stop = False
        self._user_manual_stop = False
        self._mt5_ready = False
        self._retraining_mode = False
        self._last_retraining_log_ts = 0.0
        # Status
        self._active_asset = "NONE"
        self._last_selected_asset = "NONE"
        self._order_counter = 0
        self._last_reconcile_ts = 0.0
        # Queues and worker
        self._max_queue: int = 50
        self._order_q: "queue.Queue[OrderIntent]" = queue.Queue(maxsize=self._max_queue)
        self._result_q: "queue.Queue[ExecutionResult]" = queue.Queue(
            maxsize=self._max_queue
        )
        self._exec_worker: Optional[ExecutionWorker] = None
        self._exec_aux_workers: List[ExecutionWorker] = []
        self._exec_parallelism: int = max(
            2,
            int(float(os.getenv("EXEC_WORKER_THREADS", "2") or 2)),
        )
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
        self._poll_fast: float = 0.25
        self._poll_slow: float = 0.75
        self._min_cycle_sec: float = max(
            0.05,
            float(os.getenv("ENGINE_MIN_CYCLE_SEC", "0.25") or 0.25),
        )
        self._pipeline_log_every: float = 2.0
        self._last_pipeline_log_ts = 0.0
        self._last_pipeline_log_key: Optional[Tuple[Any, ...]] = None
        self._max_consecutive_errors: int = 12
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
        self._portfolio_stop_triggered: bool = False
        self._portfolio_stop_reason: str = ""
        self._unsafe_account_reason: str = ""
        self._unsafe_position_grace_sec: float = max(
            0.0,
            float(os.getenv("UNSAFE_POSITION_GRACE_SEC", "20.0") or 20.0),
        )
        self._unsafe_account_snapshot: Dict[str, Any] = {
            "reason": "",
            "total_positions": 0,
            "symbol_counts": {},
            "foreign_symbols": [],
            "unmanaged_positions": 0,
            "zero_protection_positions": 0,
            "magic_counts": {},
            "position_issues": [],
            "grace_positions": [],
        }
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
        self._current_open_positions: Dict[str, int] = {"XAU": 0, "BTC": 0}
        self._reserved_open_positions: Dict[str, int] = {"XAU": 0, "BTC": 0}
        self._position_protection_interval_sec: float = max(
            0.75,
            float(os.getenv("POSITION_PROTECTION_INTERVAL_SEC", "2.0") or 2.0),
        )
        self._position_protection_last_ts_by_ticket: Dict[int, float] = {}
        self._position_protection_last_sl_by_ticket: Dict[int, float] = {}
        self._last_account_snapshot: Dict[str, float] = {
            "balance": 0.0,
            "equity": 0.0,
            "updated_ts": 0.0,
        }
        self._account_anomaly_factor: float = 50.0
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
        live_evidence_grace_default = max(30.0, self._runtime_start_grace_sec * 2.0)
        self._live_evidence_grace_sec: float = max(
            self._live_monitor_interval_sec,
            float(
                os.getenv(
                    "LIVE_EVIDENCE_GRACE_SEC",
                    str(live_evidence_grace_default),
                )
                or live_evidence_grace_default
            ),
        )
        self._live_evidence_min_latency_samples: int = max(
            5,
            int(float(os.getenv("LIVE_EVIDENCE_MIN_LATENCY_SAMPLES", "8") or 8)),
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
            "dd_pct": 0.0,
        }
        self._last_live_pause_reason: str = ""
        self._last_live_pause_log_ts: float = 0.0
        # Analysis state log throttling
        self._last_analysis_paused_state = False
        self._last_analysis_state_log_ts = 0.0
        self._last_skip_log_ts: dict[str, float] = (
            {}
        )  # Anti-spam throttle for blocked notifications
        self._spread_skip_ts: Dict[str, float] = {}
        self._gate_status_log_ttl: float = 60.0
        self._last_gate_status_sig: str = ""
        self._last_gate_status_ts: float = 0.0
        # Order Tracking
        self._seen_index: dict[tuple[str, str], tuple[float, int]] = {}
        # Idempotency window
        self._seen: Deque[Tuple[str, str, float]] = deque()
        self._signal_cooldown_sec_by_asset: Dict[str, float] = {
            "XAU": 60.0,
            "BTC": 60.0,
        }
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
        self._expected_mt5_login, self._expected_mt5_server = (
            self._expected_mt5_identity()
        )
        raw_cd = getattr(
            self._xau_cfg, "signal_cooldown_sec_override", None
        ) or getattr(self._btc_cfg, "signal_cooldown_sec_override", None)
        self._signal_cooldown_override_sec = float(raw_cd) if raw_cd else None
        self._xau: Optional[_AssetPipeline] = None
        self._btc: Optional[_AssetPipeline] = None
        self._last_cand_xau: Optional[AssetCandidate] = None
        self._last_cand_btc: Optional[AssetCandidate] = None
        # Edge-trigger: 1 signal_id => 1 order
        self._edge_last_trade: Dict[str, Tuple[str, str]] = {
            "XAU": ("", "Neutral"),
            "BTC": ("", "Neutral"),
        }
        self._edge_last_notified: Dict[str, Tuple[str, str]] = {
            "XAU": ("", "Neutral"),
            "BTC": ("", "Neutral"),
        }
        self._analysis_signal_last_notified: Dict[str, Tuple[str, str]] = {
            "XAU": ("", "Neutral"),
            "BTC": ("", "Neutral"),
        }
        self._order_notifier: Optional[
            Callable[[OrderIntent, ExecutionResult], None]
        ] = None
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
        xau_max_positions = max(
            1,
            int(getattr(self._xau_cfg, "max_open_positions_per_asset", 5) or 5),
        )
        btc_max_positions = max(
            1,
            int(getattr(self._btc_cfg, "max_open_positions_per_asset", 5) or 5),
        )
        portfolio_max_positions = max(
            1,
            min(
                int(getattr(self._xau_cfg, "max_concurrent_positions_total", 5) or 5),
                int(getattr(self._btc_cfg, "max_concurrent_positions_total", 5) or 5),
            ),
        )
        self._max_open_positions: Dict[str, int] = {
            "XAU": xau_max_positions,
            "BTC": btc_max_positions,
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
            max_total_exposure_factor=3.0,  # Cap leverage at 3x
            correlation_reduction=0.50,  # Cut size 50% if correlated
            max_risk_per_trade_pct=self._xau_cfg.max_risk_per_trade,
            max_concurrent_positions=portfolio_max_positions,
            max_total_drawdown_pct=float(
                getattr(self._xau_cfg, "max_drawdown", 0.12) or 0.12
            ),
            max_asset_exposure_factor=float(
                getattr(self._xau_cfg, "max_asset_exposure_factor", 1.5) or 1.5
            ),
            max_asset_risk_pct=float(
                getattr(
                    self._xau_cfg,
                    "max_asset_risk_per_asset_pct",
                    max(
                        0.0,
                        float(getattr(self._xau_cfg, "max_risk_per_trade", 0.0) or 0.0),
                    )
                    * 2.0,
                )
                or 0.03
            ),
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
        gate_ready = bool(self._model_loaded and self._backtest_passed)
        if self.dry_run:
            mode_text = "DRY-RUN (Simulated)"
        elif not gate_ready:
            mode_text = "MONITORING ONLY (Live MT5, trading disabled by model gate)"
        elif self.manual_stop_active():
            mode_text = "MONITORING ONLY (Live MT5, manual/system stop active)"
        else:
            mode_text = "LIVE TRADING (Real Money)"
        print("\n" + "=" * 60)
        print("QUANTUM TRADING SYSTEM - MATRIX RELOADED")
        print("=" * 60)
        print(f"Mode: {mode_text}")
        print("Risk Engine: OK")
        print(f"Model Loaded: {model_loaded} (v{self._model_version})")
        print(f"Backtest Status: {bt_status} (Sharpe >= {MIN_BACKTEST_SHARPE:.1f})")
        if not self._backtest_passed:
            print(f"Gate Reason: {self._gate_last_reason}")
        _display_floor = max(
            int(getattr(self._xau_cfg, "min_confidence", 75) or 75), 70
        )
        if gate_ready and not self.manual_stop_active():
            print(f"Signals: SNIPER MODE (Conf >= {_display_floor}%)")
        else:
            print("Signals: DISABLED until model gate and stop guards clear")
        print("=" * 60 + "\n")

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
                log_health.critical(
                    "HARD_KILL_SWITCH | STOP.lock DETECTED | ABORTING ALL TRADES"
                )
                self._manual_stop = True
                # Immediate Queue Wipe
                self._drain_queue(self._order_q)
                self._drain_queue(self._result_q)
                with self._order_state_lock:
                    self._order_rm_by_id.clear()
                    self._pending_order_meta.clear()
                    self._reserved_open_positions = {"XAU": 0, "BTC": 0}
                self._emergency_flatten("stop_lock")
                # Optional: Send emergency notification
                if self._engine_stop_notifier:
                    try:
                        self._engine_stop_notifier("ALL", "STOP.lock File Detected")
                    except Exception:
                        pass

    def set_order_notifier(
        self, cb: Optional[Callable[[OrderIntent, ExecutionResult], None]]
    ) -> None:
        self._order_notifier = cb

    def set_phase_notifier(
        self, cb: Optional[Callable[[str, str, str, str], None]]
    ) -> None:
        self._phase_notifier = cb

    def set_engine_stop_notifier(
        self, cb: Optional[Callable[[str, str], None]]
    ) -> None:
        self._engine_stop_notifier = cb

    def set_daily_start_notifier(
        self, cb: Optional[Callable[[str, str], None]]
    ) -> None:
        self._daily_start_notifier = cb

    def set_signal_notifier(self, cb: Optional[Callable[[str, Any], None]]) -> None:
        self._signal_notifier = cb

    def set_skip_notifier(self, cb: Optional[Callable[[AssetCandidate], None]]) -> None:
        self._skip_notifier = cb

    def start(self) -> bool:
        if not self.dry_run:
            self._check_model_health()
            if not (self._model_loaded and self._backtest_passed):
                with self._lock:
                    self._manual_stop = True
                log_err.error(
                    "ENGINE_START_MONITORING_GATEKEEPER_FAILED | trading_disabled=True analytics_alive=True model_loaded=%s backtest_passed=%s",
                    self._model_loaded,
                    self._backtest_passed,
                )
        runtime_snapshot = self.runtime_watchdog_snapshot()
        with self._lock:
            if bool(runtime_snapshot.get("trading_ok", False)) or bool(
                runtime_snapshot.get("starting", False)
            ):
                return True
            if self._run.is_set():
                log_health.warning(
                    "ENGINE_RUNTIME_RESTART | loop_alive=%s stale=%s heartbeat_age=%.1fs",
                    runtime_snapshot.get("loop_alive", False),
                    runtime_snapshot.get("stale", False),
                    float(runtime_snapshot.get("heartbeat_age_sec", 0.0) or 0.0),
                )
                self._run.clear()
            # MONITORING MODE: Allow start even if manual_stop is true
            self._run.set()
            self._loop_started_ts = time.time()
            self._last_runtime_heartbeat_ts = 0.0
            # Reset startup-cycle marker so watchdog grace applies until the
            # new runtime completes its first verification pass.
            self._last_reconcile_ts = 0.0
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
            print("\n" + "=" * 60)
            print("SYSTEM BOOT REPORT")
            print("=" * 60)
            print("OK Configuration Loaded (Active: XAU + BTC)")
            print(
                f"OK Risk Engine: ACTIVE (Risk/Trade: {self._xau_cfg.max_risk_per_trade:.1%})"
            )
            # Linkage proof
            self._check_backtest_integration()
            print("OK Backtest Engine: LINKED (Historical data logic ready)")
            print("OK Logging to: Logs/portfolio_engine_health.log")
            if self.dry_run:
                print("WARN DRY RUN MODE: MT5 connection mocked / Orders simulated")
            print("=" * 60 + "\n")
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
        unsafe_account_reason = self._preflight_live_account_state()
        if unsafe_account_reason:
            with self._lock:
                self._manual_stop = True
            log_health.warning(
                "ENGINE_START_MONITORING_UNSAFE_ACCOUNT_STATE | trading_disabled=True reason=%s",
                unsafe_account_reason,
            )
        # ─── Idempotency WAL reconciliation ───────────────────────────
        # Before accepting any new order intents, reconcile any PENDING/SENT
        # entries from a previous process against the broker's actual state.
        # This closes the "crash between order_send and ACK" window: if the
        # order actually executed, we mark it CONFIRMED; if the broker has
        # no record and the TTL expired, we mark it FAILED so retries are
        # permitted. Under no circumstances is a duplicate order sent.
        if not self.dry_run:
            try:
                import MetaTrader5 as _mt5  # local import; mt5 already ensured above

                from core.idempotency import (
                    get_default_journal,
                    reconcile_on_startup,
                )

                def _positions_getter():
                    try:
                        return _mt5.positions_get() or []
                    except Exception:
                        return []

                def _history_getter(frm: float, to: float):
                    try:
                        return (
                            _mt5.history_deals_get(
                                datetime.fromtimestamp(frm, tz=timezone.utc),
                                datetime.fromtimestamp(to, tz=timezone.utc),
                            )
                            or []
                        )
                    except Exception:
                        return []

                summary = reconcile_on_startup(
                    get_default_journal(),
                    positions_getter=_positions_getter,
                    history_deals_getter=_history_getter,
                )
                log_health.info(
                    "IDEMPOTENCY_RECONCILE | confirmed=%s failed=%s still_pending=%s",
                    int(summary.get("confirmed", 0)),
                    int(summary.get("failed", 0)),
                    int(summary.get("still_pending", 0)),
                )
            except Exception as _rec_exc:
                log_err.error(
                    "IDEMPOTENCY_RECONCILE_FAIL | err=%s", _rec_exc
                )

        self._restart_exec_worker()
        if _monitoring_only_mode():
            log_health.warning(
                "MONITORING_ONLY_PROFILE_ACTIVE | trading_disabled=True manual_stop=%s",
                self._manual_stop,
            )
        log_health.info(
            "PORTFOLIO_ENGINE_START | dry_run=%s xau=%s btc=%s manual_stop=%s",
            self.dry_run,
            self._xau.symbol if self._xau else "None",
            self._btc.symbol if self._btc else "None",
            self._manual_stop,
        )
        t = threading.Thread(
            target=self._loop, name="portfolio.engine.loop", daemon=True
        )
        with self._lock:
            if self._starting:
                log_health.warning("ENGINE_START_SKIPPED | reason=already_starting")
                return True
            self._starting = True
            self._loop_thread = t
        try:
            t.start()
        except Exception:
            with self._lock:
                self._starting = False
                if self._loop_thread is t:
                    self._loop_thread = None
                self._run.clear()
            log_err.error("ENGINE_START_THREAD_FAIL | err=%s", traceback.format_exc())
            raise
        finally:
            with self._lock:
                self._starting = False
        return True

    def stop(self) -> bool:
        with self._lock:
            self._run.clear()
        loop_thread = None
        with self._lock:
            loop_thread = self._loop_thread
        if loop_thread and loop_thread.is_alive():
            try:
                loop_thread.join(timeout=5.0)
            except Exception:
                pass
        with self._lock:
            self._loop_thread = None
            self._loop_started_ts = 0.0
            self._last_runtime_heartbeat_ts = 0.0
        self._stop_exec_workers(timeout=6.0)
        # Drain queues best-effort + clear mapping
        self._drain_queue(self._order_q)
        self._drain_queue(self._result_q)
        with self._order_state_lock:
            self._order_rm_by_id.clear()
            self._pending_order_meta.clear()
            self._reserved_open_positions = {"XAU": 0, "BTC": 0}
        log_health.info("PORTFOLIO_ENGINE_STOP")
        return True

    def request_manual_stop(self) -> bool:
        with self._lock:
            self._manual_stop = True
            self._user_manual_stop = True
            log_health.info(
                "MANUAL_STOP_REQUESTED | Switching to MONITORING mode (Trading Disabled)"
            )
        return True

    def request_system_monitoring_stop(self, reason: str = "") -> bool:
        """Disable trading without marking the pause as a user/manual stop."""
        with self._lock:
            self._manual_stop = True
            self._user_manual_stop = False
            log_health.warning(
                "SYSTEM_MONITORING_STOP_REQUESTED | trading_disabled=True reason=%s",
                str(reason or "system_guard"),
            )
        return True

    def clear_manual_stop(self) -> None:
        unsafe_reason = self._preflight_live_account_state()
        model_ready = True
        if not self.dry_run:
            self._check_model_health()
            model_ready = bool(self._model_loaded and self._backtest_passed)
        with self._lock:
            if _monitoring_only_mode():
                self._manual_stop = True
                self._user_manual_stop = False
                log_health.warning(
                    "MANUAL_STOP_CLEAR_BLOCKED_MONITORING_ONLY | trading_disabled=True"
                )
                return
            if os.path.exists("STOP.lock"):
                self._manual_stop = True
                self._user_manual_stop = True
                log_health.warning(
                    "MANUAL_STOP_CLEAR_BLOCKED_STOP_LOCK | trading_disabled=True"
                )
                return
            if unsafe_reason:
                self._manual_stop = True
                self._user_manual_stop = False
                log_health.warning(
                    "MANUAL_STOP_CLEAR_BLOCKED_UNSAFE_ACCOUNT_STATE | trading_disabled=True reason=%s",
                    unsafe_reason,
                )
                return
            if self._portfolio_stop_triggered or self._portfolio_stop_reason:
                self._manual_stop = True
                self._user_manual_stop = False
                log_health.warning(
                    "MANUAL_STOP_CLEAR_BLOCKED_PORTFOLIO_STOP | trading_disabled=True reason=%s",
                    self._portfolio_stop_reason,
                )
                return
            if not self.dry_run and not model_ready:
                self._manual_stop = True
                self._user_manual_stop = False
                log_health.warning(
                    "MANUAL_STOP_CLEAR_BLOCKED_GATEKEEPER | trading_disabled=True reason=%s",
                    self._gate_last_reason,
                )
                return
            if self._manual_stop:
                log_health.info("MANUAL_STOP_CLEAR")
            self._manual_stop = False
            self._user_manual_stop = False

    def manual_stop_active(self) -> bool:
        with self._lock:
            return bool(self._manual_stop)

    def _system_resume_trading_if_safe(self, trigger: str = "") -> bool:
        with self._lock:
            if not self._manual_stop or self._user_manual_stop:
                return False

        if os.path.exists("STOP.lock"):
            return False

        if not self.dry_run:
            self._check_model_health()
            if not (self._model_loaded and self._backtest_passed):
                return False

        unsafe_reason = self._preflight_live_account_state()
        with self._lock:
            if self._user_manual_stop:
                return False
            if unsafe_reason:
                self._manual_stop = True
                return False
            if self._portfolio_stop_triggered or self._portfolio_stop_reason:
                self._manual_stop = True
                return False
            if not self._manual_stop:
                return False
            self._manual_stop = False
            self._user_manual_stop = False

        log_health.info(
            "SYSTEM_AUTO_RESUME | trigger=%s",
            str(trigger or "conditions_cleared"),
        )
        return True


_engine_instance: Optional[MultiAssetTradingEngine] = None
_engine_instance_lock = threading.Lock()


def get_engine(dry_run: bool = False) -> MultiAssetTradingEngine:
    global _engine_instance
    requested_dry_run = bool(dry_run)
    with _engine_instance_lock:
        if _engine_instance is None:
            _engine_instance = MultiAssetTradingEngine(dry_run=requested_dry_run)
        elif bool(getattr(_engine_instance, "dry_run", False)) != requested_dry_run:
            raise ValueError(
                "get_engine(dry_run=...) mismatch: existing instance has "
                f"dry_run={bool(getattr(_engine_instance, 'dry_run', False))}, "
                f"requested dry_run={requested_dry_run}"
            )
    return _engine_instance


_signal_handlers_installed = False


def install_graceful_shutdown_handlers() -> bool:
    """Install process-level shutdown hooks for live daemon runtimes."""
    global _signal_handlers_installed
    if _signal_handlers_installed:
        return True
    if threading.current_thread() is not threading.main_thread():
        return False

    def _handle_shutdown(signum: int, _frame: Any) -> None:
        try:
            inst = _engine_instance
            if inst is not None:
                log_health.warning("ENGINE_SIGNAL_SHUTDOWN | signal=%s", signum)
                inst.stop()
        except Exception as exc:
            log_err.error("ENGINE_SIGNAL_SHUTDOWN_ERROR | signal=%s err=%s", signum, exc)

    try:
        signal.signal(signal.SIGTERM, _handle_shutdown)
        signal.signal(signal.SIGINT, _handle_shutdown)
        _signal_handlers_installed = True
        return True
    except Exception as exc:
        log_err.error("ENGINE_SIGNAL_HANDLER_INSTALL_ERROR | err=%s", exc)
        return False


class _EngineProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_engine(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(get_engine(), name, value)

    def __repr__(self) -> str:
        return repr(get_engine())


# Import-compatible lazy instance
install_graceful_shutdown_handlers()
engine = _EngineProxy()


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "BtcFeatureEngine",
    "BtcRiskManager",
    "BtcSignalEngine",
    "MIN_BACKTEST_SHARPE",
    "MIN_BACKTEST_WIN_RATE",
    "MODEL_STATE_PATH",
    "MONITORING_ONLY_HALT_REASONS",
    "MultiAssetTradingEngine",
    "RECOVERABLE_HALT_REASONS",
    "XauFeatureEngine",
    "XauRiskManager",
    "XauSignalEngine",
    "btc_apply_high_accuracy_mode",
    "engine",
    "get_engine",
    "install_graceful_shutdown_handlers",
    "xau_apply_high_accuracy_mode",
]
