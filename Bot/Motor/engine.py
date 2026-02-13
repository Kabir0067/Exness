from __future__ import annotations

import builtins
import os
import queue
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from dataclasses import replace
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import MetaTrader5 as mt5

from ExnessAPI.functions import close_all_position, market_is_open
from mt5_client import MT5_LOCK, ensure_mt5, mt5_status

# -------------------- XAU stack --------------------
from config_xau import EngineConfig as XauConfig
from config_xau import apply_high_accuracy_mode as xau_apply_high_accuracy_mode
from config_xau import get_config_from_env as get_xau_config
from DataFeed.xau_market_feed import MarketFeed as XauMarketFeed
from StrategiesXau.xau_indicators import Classic_FeatureEngine as XauFeatureEngine
from StrategiesXau.xau_risk_management import RiskManager as XauRiskManager
from StrategiesXau.xau_signal_engine import SignalEngine as XauSignalEngine

# -------------------- BTC stack --------------------
from config_btc import EngineConfig as BtcConfig
from config_btc import apply_high_accuracy_mode as btc_apply_high_accuracy_mode
from config_btc import get_config_from_env as get_btc_config
from DataFeed.btc_market_feed import MarketFeed as BtcMarketFeed
from StrategiesBtc.btc_indicators import Classic_FeatureEngine as BtcFeatureEngine
from StrategiesBtc.btc_risk_management import RiskManager as BtcRiskManager
from StrategiesBtc.btc_signal_engine import SignalEngine as BtcSignalEngine

from .execution import ExecutionWorker
from .logging_setup import _DIAG_ENABLED, _DIAG_EVERY_SEC, log_diag, log_err, log_health
from .models import AssetCandidate, ExecutionResult, OrderIntent, PortfolioStatus
from .pipeline import _AssetPipeline
from .scheduler import UTCScheduler
from .utils import safe_json_dumps, tf_seconds


class MultiAssetTradingEngine:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = bool(dry_run)

        self._run = threading.Event()
        self._lock = threading.Lock()

        self._manual_stop = False
        self._mt5_ready = False

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

        # Recovery
        self._last_recover_ts = 0.0

        # Diagnostics
        self._diag_last_ts = 0.0

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
            log_err.error("MT5 init error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _check_mt5_health(self) -> bool:
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
            self._xau.ensure_symbol_selected()

            # BTC
            btc_feed = BtcMarketFeed(self._btc_cfg, self._btc_cfg.symbol_params)
            btc_features = BtcFeatureEngine(self._btc_cfg)
            btc_risk = BtcRiskManager(self._btc_cfg, self._btc_cfg.symbol_params)
            btc_signal = BtcSignalEngine(self._btc_cfg, self._btc_cfg.symbol_params, btc_feed, btc_features, btc_risk)
            self._btc = _AssetPipeline("BTC", self._btc_cfg, btc_feed, btc_features, btc_risk, btc_signal)
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
            fn = getattr(risk, "update_phase", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
            current = self._get_phase(risk) or "A"
            prev = str(self._last_phase_by_asset.get(asset, "A") or "A")
            if current != prev:
                self._last_phase_by_asset[asset] = current
                if self._phase_notifier:
                    reason = self._phase_reason(risk, current)
                    self._phase_notifier(str(asset), prev, current, reason)
        except Exception:
            return

    def _check_daily_start(self, asset: str, risk: Any) -> None:
        try:
            if risk is None:
                return
            current_date = self._get_daily_date(risk)
            if not current_date:
                return
            prev_date = str(self._last_daily_date_by_asset.get(asset, "") or "")
            if not prev_date:
                self._last_daily_date_by_asset[asset] = current_date
                return
            if current_date != prev_date:
                self._last_daily_date_by_asset[asset] = current_date
                if self._daily_start_notifier:
                    self._daily_start_notifier(str(asset), current_date)
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
        while self._seen and (now - self._seen[0][2]) > ttl:
            a, sid, ts = self._seen.popleft()
            rec = self._seen_index.get((a, sid))
            if rec and float(rec[0]) == float(ts):
                self._seen_index.pop((a, sid), None)

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


    def _candidate_is_tradeable(self, c: AssetCandidate) -> bool:
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
                        "message": f"⚠️ [PHASE C - SHADOW TRADE] Verified {cand.signal} Signal on {cand.symbol}.\n"
                                   f"Price: {current_price:.2f} | Confidence: {cand.confidence:.0f}%\n"
                                   f"(Trade blocked by Risk Limit)\n"
                                   f"Would-be Lot: {cand.lot:.2f} | SL: {cand.sl:.2f} | TP: {cand.tp:.2f}",
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
    def start(self) -> bool:
        with self._lock:
            if self._run.is_set():
                return True
            # MONITORING MODE: Allow start even if manual_stop is true
            self._run.set()

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
            self._xau.symbol,
            self._btc.symbol,
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

    # -------------------- main loop --------------------
    def _loop(self) -> None:
        if not self._xau or not self._btc:
            log_err.error("Portfolio engine loop start failed: pipelines not built")
            self._run.clear()
            return

        consecutive_errors = 0

        while self._run.is_set():
            t0 = time.time()
            try:
                mt5_ok = self._check_mt5_health()
                if not mt5_ok:
                    consecutive_errors += 1
                    log_health.info("PIPELINE_STAGE | step=mt5_health ok=%s", mt5_ok)
                    if consecutive_errors >= self._max_consecutive_errors:
                        if self._recover_all():
                            consecutive_errors = 0
                        else:
                            time.sleep(0.8)
                    else:
                        time.sleep(0.6)
                    continue

                # CRITICAL: Hard Stop File Check (Every Iteration)
                self._check_hard_stop_file()

                # Don't trade if manual stop active
                if self._manual_stop:
                     # If we just entered manual stop, we already drained queues in _check_hard_stop_file
                     # Just ensure we loop quickly but don't execute
                     time.sleep(0.5)
                     continue

                # Hard stop notifications (do not trade when requires_hard_stop True)
                try:
                    for asset, pipe in (("XAU", self._xau), ("BTC", self._btc)):
                        risk = pipe.risk if pipe else None
                        hs = bool(risk and hasattr(risk, "requires_hard_stop") and risk.requires_hard_stop())
                        was = self._hard_stop_notified.get(asset, False)
                        if hs and not was:
                            log_health.info("HARD_STOP_TRIGGERED | asset=%s (trade_locked)", asset)
                            if self._engine_stop_notifier:
                                reason = self._phase_reason(risk, "C")
                                self._engine_stop_notifier(asset, reason)
                            self._hard_stop_notified[asset] = True
                        elif (not hs) and was:
                            log_health.info("HARD_STOP_CLEARED | asset=%s (trading_resumed)", asset)
                            self._hard_stop_notified[asset] = False
                except Exception:
                    pass

                # Validate market data
                x_ok = self._xau.validate_market_data()
                b_ok = self._btc.validate_market_data()
                rx = self._xau.last_market_reason
                rb = self._btc.last_market_reason
                ax = float(self._xau.last_bar_age_sec)
                ab = float(self._btc.last_bar_age_sec)

                now_ts = time.time()
                key = (x_ok, rx, b_ok, rb)
                if (now_ts - self._last_pipeline_log_ts) >= self._pipeline_log_every or key != self._last_pipeline_log_key:
                    log_health.info(
                        "PIPELINE_STAGE | step=market_data ok_xau=%s reason_xau=%s age_xau=%.1fs tf_xau=%s bars_xau=%s "
                        "close_xau=%.3f vol_xau=%s ts_xau=%s "
                        "ok_btc=%s reason_btc=%s age_btc=%.1fs tf_btc=%s bars_btc=%s close_btc=%.3f vol_btc=%s ts_btc=%s",
                        x_ok,
                        rx,
                        ax,
                        str(self._xau.last_market_tf),
                        int(self._xau.last_market_rows),
                        float(self._xau.last_market_close),
                        int(builtins.round(self._xau.last_market_volume)) if self._xau.last_market_volume else 0,
                        str(self._xau.last_market_ts),
                        b_ok,
                        rb,
                        ab,
                        str(self._btc.last_market_tf),
                        int(self._btc.last_market_rows),
                        float(self._btc.last_market_close),
                        int(builtins.round(self._btc.last_market_volume)) if self._btc.last_market_volume else 0,
                        str(self._btc.last_market_ts),
                    )
                    self._last_pipeline_log_ts = now_ts
                    self._last_pipeline_log_key = key

                # Reconcile positions
                self._xau.reconcile_positions()
                self._btc.reconcile_positions()
                self._last_reconcile_ts = time.time()

                # Phase change notifications (A/B/C)
                self._check_phase_change("XAU", self._xau.risk if self._xau else None)
                self._check_phase_change("BTC", self._btc.risk if self._btc else None)

                # New trading day notifications
                self._check_daily_start("XAU", self._xau.risk if self._xau else None)
                self._check_daily_start("BTC", self._btc.risk if self._btc else None)

                # Drain execution results
                while True:
                    try:
                        r = self._result_q.get_nowait()
                    except queue.Empty:
                        break

                    try:
                        rm = self._order_rm_by_id.pop(str(r.order_id), None)
                        if rm is None:
                            # fallback (legacy)
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

                # Positions and active asset (informational)
                open_xau = self._xau.open_positions()
                open_btc = self._btc.open_positions()
                self._active_asset = self._select_active_asset(open_xau, open_btc)

                # ============================================================
                # FIX #3: Detect Position Closures -> Trigger Cooldown
                # ============================================================
                now_ts = time.time()
                prev_xau = self._last_open_positions.get("XAU", 0)
                prev_btc = self._last_open_positions.get("BTC", 0)

                # XAU position closed
                if prev_xau > 0 and open_xau == 0:
                    self._last_trade_close_ts["XAU"] = now_ts
                    self._cooldown_blocked_count["XAU"] = 0  # Reset counter
                    log_health.info(
                        "TRADE_CLOSED | asset=XAU cooldown_started=%.0fs",
                        self._trade_cooldown_sec,
                    )

                # BTC position closed
                if prev_btc > 0 and open_btc == 0:
                    self._last_trade_close_ts["BTC"] = now_ts
                    self._cooldown_blocked_count["BTC"] = 0  # Reset counter
                    log_health.info(
                        "TRADE_CLOSED | asset=BTC cooldown_started=%.0fs",
                        self._trade_cooldown_sec,
                    )

                # Update position tracking
                self._last_open_positions["XAU"] = open_xau
                self._last_open_positions["BTC"] = open_btc

                self._heartbeat(open_xau, open_btc)

                # SAFETY: serial mode disabled (concurrent XAU + BTC trading)
                has_pos = False

                if has_pos != self._last_analysis_paused_state or (now_ts - self._last_analysis_state_log_ts) >= 60.0:
                    self._last_analysis_paused_state = has_pos
                    self._last_analysis_state_log_ts = now_ts

                if has_pos:
                    time.sleep(1.0)
                    continue

                # Compute candidates
                cand_x = self._xau.compute_candidate() if x_ok else None
                cand_b = self._btc.compute_candidate() if b_ok else None
                self._last_cand_xau = cand_x
                self._last_cand_btc = cand_b

                # Edge-trigger reset: when signal becomes Neutral, allow next Buy/Sell again
                if cand_x and str(cand_x.signal) == "Neutral":
                    self._edge_last_trade["XAU"] = ("", "Neutral")
                if cand_b and str(cand_b.signal) == "Neutral":
                    self._edge_last_trade["BTC"] = ("", "Neutral")

                candidates: list[AssetCandidate] = []
                if cand_x:
                    candidates.append(cand_x)
                if cand_b:
                    candidates.append(cand_b)

                # Execute all valid signals
                for selected in candidates:


                    # 1. NOTIFICATION LOGIC (Decoupled & Simulation Mode)
                    # Notify even if blocked (Phase C), as long as signal is high-confidence and valid type.
                    # This allows "Simulation Mode" where user sees signals but no trades occur.
                    notify_floor = max(0.55, min(0.90, self._effective_min_conf(selected) + 0.05))
                    is_valid_sig = (
                        selected.signal in ("Buy", "Sell")
                        and float(selected.confidence) >= float(notify_floor)
                    )
                    
                    if is_valid_sig:
                        last_nid, last_nsig = self._edge_last_notified.get(selected.asset, ("", "Neutral"))
                        # Notify on EDGE change (new signal ID or signal flip)
                        if str(selected.signal_id) != last_nid or str(selected.signal) != last_nsig:
                            if self._signal_notifier:
                                try:
                                    # Mark as "SIMULATION" in payload if blocked? 
                                    # Actually, let the handler decide, but we send raw result.
                                    # If blocked, it might be worth logging.
                                    self._signal_notifier(selected.asset, selected.raw_result)
                                except Exception:
                                    pass
                            self._edge_last_notified[selected.asset] = (str(selected.signal_id), str(selected.signal))

                    # 2. EXECUTION LOGIC (Strict)
                    # Hard-filter: Must be strictly tradeable (not blocked, good prices, etc)
                    # 2. EXECUTION LOGIC (Strict)
                    # Hard-filter: Must be strictly tradeable (not blocked, good prices, etc)
                    if not self._candidate_is_tradeable(selected):
                        # Detect if it was a high-confidence signal that was blocked (Hard Stop)
                        # We notify here because _candidate_is_tradeable returned False.
                        
                        # ENHANCEMENT: Explicitly add "low_confidence" reason if that was the cause
                        min_conf = self._effective_min_conf(selected)
                        
                        if float(selected.confidence) < min_conf:
                             # Create a modified copy with the new reason for notification
                             new_reasons = tuple(selected.reasons) + (f"low_confidence:{selected.confidence:.2f}<{min_conf:.2f}",)
                             selected = replace(selected, reasons=new_reasons)

                        if (getattr(selected, "blocked", False) or getattr(selected, "reasons", None)) and hasattr(self, "_skip_notifier") and self._skip_notifier:
                             # Check duplicate to prevent spamming the same blocked signal
                             # STABLE ID FIX: Now that signal_id is stable (bar-based), this check will correctly suppress
                             # repeated notifications for the SAME signal during the same bar.
                             if not self._is_duplicate(selected.asset, selected.signal_id, time.time(), max_orders=1, order_index=999):
                                 # ROBUST FIX: Mark seen FIRST to prevent infinite retry loops if notifier fails
                                 self._mark_seen(selected.asset, selected.signal_id, time.time())
                                 
                                 try:
                                     self._skip_notifier(selected)
                                     self._last_skip_log_ts[selected.asset] = time.time()
                                 except Exception:
                                     pass
                        continue

                    # Double-check Manual Stop for execution only
                    if self._manual_stop:
                        log_health.info(
                            "MONITORING_SIGNAL | asset=%s signal=%s conf=%.4f (Skipped execution: Manual Stop)",
                            selected.asset,
                            selected.signal,
                            selected.confidence,
                        )
                        continue

                    order_count = self._orders_for_candidate(selected)
                    if int(order_count) <= 0:
                        continue

                    risk = (
                        self._xau.risk
                        if selected.asset == "XAU" and self._xau
                        else (self._btc.risk if self._btc else None)
                    )
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
                                "ORDER_SELECTED | asset=%s symbol=%s signal=%s conf=%.4f lot=%.4f sl=%s tp=%s "
                                "order_id=%s batch=%s/%s reasons=%s",
                                selected.asset,
                                selected.symbol,
                                selected.signal,
                                selected.confidence,
                                float(lot_val),
                                selected.sl,
                                selected.tp,
                                oid,
                                int(idx) + 1,
                                int(order_count),
                                ",".join(selected.reasons) if selected.reasons else "-",
                            )
                        else:
                            # Suppress ORDER_SKIP log for duplicates to prevent spam
                            # We only care about skips that are NOT duplicates (e.g. risk blocks, bad price)
                            # Duplicates are already handled silently or with single log in _enqueue_order
                            if not self._is_duplicate(selected.asset, selected.signal_id, time.time(), max_orders=int(order_count), order_index=int(idx)):
                                log_health.info(
                                    "ORDER_SKIP | asset=%s symbol=%s signal=%s conf=%.4f blocked=%s batch=%s/%s reasons=%s",
                                    selected.asset,
                                    selected.symbol,
                                    selected.signal,
                                    selected.confidence,
                                    selected.blocked,
                                    int(idx) + 1,
                                    int(order_count),
                                    ",".join(selected.reasons) if selected.reasons else "-",
                                )
                                # Notify user about skipped signal (Hard Stop / Filtered) if configured
                                if hasattr(self, "_skip_notifier") and self._skip_notifier:
                                    try:
                                        self._skip_notifier(selected)
                                    except Exception:
                                        pass

                consecutive_errors = 0

                # ============================================================
                # CRITICAL FIX: STALE DATA AUTO-RECOVERY
                # ============================================================
                # NOTE: Use TICK age, not BAR age. M1 bars are always 0-60s old by design.
                tick_age_x = float(self._xau.last_tick_age_sec) if self._xau else 0.0
                tick_age_b = float(self._btc.last_tick_age_sec) if self._btc else 0.0

                # 1. Force Recovery if data is dead (>60s)
                # Check XAU (only if market open)
                if tick_age_x > 60.0 and market_is_open("XAU"):
                    log_health.warning("STALE_DATA_CRITICAL | asset=XAU tick_age=%.1fs > 60s -> FORCE RECOVERY", tick_age_x)
                    if self._recover_all():
                        consecutive_errors = 0
                    continue

                # Check BTC (always 24/7)
                if tick_age_b > 60.0:
                    log_health.warning("STALE_DATA_CRITICAL | asset=BTC tick_age=%.1fs > 60s -> FORCE RECOVERY", tick_age_b)
                    if self._recover_all():
                        consecutive_errors = 0
                    continue

                # 2. Adaptive sleep - SKIP SLEEP if TICK data is slightly stale (>5s) to catch up
                max_tick_age = max(tick_age_x, tick_age_b)
                if max_tick_age > 5.0:
                    # Tick data is stale - skip sleep to process faster
                    time.sleep(0.02)  # Tiny sleep to prevent CPU spin
                    continue
                    
                dt = time.time() - t0
                poll = self._poll_fast
                if self._order_q.qsize() >= max(3, self._max_queue // 2):
                    poll = self._poll_slow
                if any(c.latency_ms >= 250.0 for c in candidates):
                    poll = self._poll_slow
                time.sleep(max(0.02, poll - dt))

            except Exception as exc:
                consecutive_errors += 1
                log_err.error("portfolio loop error: %s | tb=%s", exc, traceback.format_exc())
                if consecutive_errors >= self._max_consecutive_errors:
                    if self._recover_all():
                        consecutive_errors = 0
                    else:
                        time.sleep(1.0)
                else:
                    time.sleep(0.6)


# Global instance (import-compatible)
engine = MultiAssetTradingEngine(dry_run=False)
