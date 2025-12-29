# Bot/engine.py  (PRODUCTION-GRADE)
from __future__ import annotations

import logging
import os
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import MetaTrader5 as mt5

from config import EngineConfig, apply_high_accuracy_mode, get_config_from_env
from DataFeed.market_feed import MarketFeed
from Strategies.indicators import Classic_FeatureEngine
from Strategies.risk_management import RiskManager
from Strategies.signal_engine import SignalEngine, SignalResult
from mt5_client import MT5_LOCK, ensure_mt5
from ExnessAPI.orders import close_all_position

# =============================================================================
# Logging
#   1) ERROR logger: only errors (Logs/engine.log)
#   2) HEALTH logger: heartbeat / liveness (Logs/engine_health.log)
# =============================================================================
os.makedirs("Logs", exist_ok=True)

log_err = logging.getLogger("engine")
log_err.setLevel(logging.ERROR)
log_err.propagate = False
if not log_err.handlers:
    fh = logging.FileHandler("Logs/engine.log", encoding="utf-8")
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    log_err.addHandler(fh)

log_health = logging.getLogger("engine.health")
log_health.setLevel(logging.INFO)
log_health.propagate = False
if not log_health.handlers:
    fh2 = logging.FileHandler("Logs/engine_health.log", encoding="utf-8")
    fh2.setLevel(logging.INFO)
    fh2.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    log_health.addHandler(fh2)


# =============================================================================
# Data models
# =============================================================================
@dataclass(frozen=True)
class EngineStatus:
    connected: bool
    trading: bool
    manual_stop: bool
    balance: float
    equity: float
    dd_pct: float
    today_pnl: float
    open_trades: int
    phase: str
    last_signal: str
    latency_ms: float
    queue_size: int
    last_reconcile_ts: float


@dataclass(frozen=True)
class OrderIntent:
    signal: str                 # "Buy" / "Sell"
    confidence: float           # 0..1
    lot: float
    sl: float
    tp: float
    price: float               # snapshot price at enqueue
    symbol: str
    enqueue_time: float
    order_id: str
    signal_id: str
    idempotency_key: str


@dataclass(frozen=True)
class ExecutionResult:
    order_id: str
    signal_id: str
    ok: bool
    reason: str
    sent_ts: float
    fill_ts: float
    req_price: float
    exec_price: float
    volume: float
    slippage: float
    retcode: int = 0


# =============================================================================
# Execution Worker (NO sleeping inside MT5_LOCK)
# =============================================================================
class ExecutionWorker(threading.Thread):
    def __init__(
        self,
        order_queue: "queue.Queue[OrderIntent]",
        result_queue: "queue.Queue[ExecutionResult]",
        risk_manager: Optional[RiskManager],
        cfg: EngineConfig,
        dry_run: bool = False,
    ):
        super().__init__(daemon=True)
        self.order_queue = order_queue
        self.result_queue = result_queue
        self.risk_manager = risk_manager
        self.cfg = cfg
        self.dry_run = bool(dry_run)
        self._run = threading.Event()
        self._run.set()

    def stop(self) -> None:
        self._run.clear()

    def run(self) -> None:
        while self._run.is_set():
            try:
                order = self.order_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                res = self._process(order)
                try:
                    self.result_queue.put_nowait(res)
                except Exception:
                    # result queue overflow shouldn't crash worker
                    pass
            except Exception as exc:
                log_err.error("ExecutionWorker crash | err=%s | tb=%s", exc, traceback.format_exc())
            finally:
                try:
                    self.order_queue.task_done()
                except Exception:
                    pass

    def _process(self, order: OrderIntent) -> ExecutionResult:
        sent_ts = time.time()

        # -------------------- DRY RUN --------------------
        if self.dry_run:
            fill_ts = time.time()

            # In dry-run we assume "expected" equals the snapshot enqueue price
            req_price = float(order.price)
            exec_price = float(order.price)

            # raw slippage (exec - expected) so sign logic remains consistent
            slippage = exec_price - req_price

            if self.risk_manager and hasattr(self.risk_manager, "record_execution_metrics"):
                # FIX: provide missing 'side' and defined req_price
                self.risk_manager.record_execution_metrics(
                    order.order_id,
                    order.signal,          # side
                    order.enqueue_time,
                    sent_ts,
                    fill_ts,
                    req_price,             # expected_price
                    exec_price,            # filled_price
                    slippage,              # raw
                )

            if self.risk_manager and hasattr(self.risk_manager, "record_trade"):
                self.risk_manager.record_trade(exec_price, order.sl, order.lot, order.tp, 1)

            return ExecutionResult(
                order_id=order.order_id,
                signal_id=order.signal_id,
                ok=True,
                reason="dry_run",
                sent_ts=sent_ts,
                fill_ts=fill_ts,
                req_price=req_price,
                exec_price=exec_price,
                volume=order.lot,
                slippage=slippage,
                retcode=0,
            )

        # -------------------- LIVE EXECUTION --------------------
        max_attempts = 3
        last_reason = "unknown"
        last_retcode = 0

        for attempt in range(1, max_attempts + 1):
            try:
                with MT5_LOCK:
                    info = mt5.symbol_info(order.symbol)
                    tick = mt5.symbol_info_tick(order.symbol)

                if info is None:
                    last_reason = "symbol_not_found"
                    time.sleep(0.35)
                    continue

                if tick is None:
                    last_reason = "tick_missing"
                    time.sleep(0.35)
                    continue

                req_price = float(tick.ask if order.signal == "Buy" else tick.bid)
                if req_price <= 0:
                    last_reason = "bad_price"
                    time.sleep(0.35)
                    continue

                order_type = mt5.ORDER_TYPE_BUY if order.signal == "Buy" else mt5.ORDER_TYPE_SELL

                filling = getattr(info, "filling_mode", None)
                if filling not in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
                    filling = mt5.ORDER_FILLING_IOC

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": order.symbol,
                    "volume": float(order.lot),
                    "type": order_type,
                    "price": req_price,
                    "sl": float(order.sl) if order.sl > 0 else 0.0,
                    "tp": float(order.tp) if order.tp > 0 else 0.0,
                    "deviation": int(getattr(self.cfg, "deviation", 50) or 50),
                    "magic": int(getattr(self.cfg, "magic", 987654) or 987654),
                    "comment": "xau_scalper",
                    "type_filling": int(filling),
                    "type_time": mt5.ORDER_TIME_GTC,
                }

                with MT5_LOCK:
                    result = mt5.order_send(request)

                if result is None:
                    last_reason = "order_send_none"
                    time.sleep(0.45)
                    continue

                last_retcode = int(getattr(result, "retcode", 0) or 0)

                if last_retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                    fill_ts = time.time()
                    exec_price = float(getattr(result, "price", req_price) or req_price)
                    exec_vol = float(getattr(result, "volume", order.lot) or order.lot)

                    # raw slippage (exec - expected)
                    slippage = exec_price - req_price

                    if self.risk_manager and hasattr(self.risk_manager, "record_trade"):
                        self.risk_manager.record_trade(exec_price, order.sl, exec_vol, order.tp, 1)

                    if self.risk_manager and hasattr(self.risk_manager, "record_execution_metrics"):
                        # FIX: pass side + correct expected_price (req_price), not enqueue snapshot
                        self.risk_manager.record_execution_metrics(
                            order.order_id,
                            order.signal,          # side
                            order.enqueue_time,
                            sent_ts,
                            fill_ts,
                            req_price,             # expected_price at send time
                            exec_price,            # filled_price
                            slippage,              # raw
                        )

                    return ExecutionResult(
                        order_id=order.order_id,
                        signal_id=order.signal_id,
                        ok=True,
                        reason="filled",
                        sent_ts=sent_ts,
                        fill_ts=fill_ts,
                        req_price=req_price,
                        exec_price=exec_price,
                        volume=exec_vol,
                        slippage=slippage,
                        retcode=last_retcode,
                    )

                # Retry only for requote / timeout
                if last_retcode in (mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_TIMEOUT):
                    last_reason = f"retry_retcode_{last_retcode}"
                    time.sleep(0.45)
                    continue

                last_reason = f"fail_retcode_{last_retcode}"
                break

            except Exception as exc:
                last_reason = f"exception_{type(exc).__name__}"
                time.sleep(0.45)

        fill_ts = time.time()
        if self.risk_manager and hasattr(self.risk_manager, "record_execution_failure"):
            self.risk_manager.record_execution_failure(order.order_id, order.enqueue_time, sent_ts, last_reason)

        return ExecutionResult(
            order_id=order.order_id,
            signal_id=order.signal_id,
            ok=False,
            reason=last_reason,
            sent_ts=sent_ts,
            fill_ts=fill_ts,
            req_price=float(order.price),
            exec_price=0.0,
            volume=float(order.lot),
            slippage=0.0,
            retcode=last_retcode,
        )


# =============================================================================
# Trading Engine Controller (PRODUCTION)
# =============================================================================
class TradingEngine:
    def __init__(self, dry_run: bool = False) -> None:
        self.cfg: EngineConfig = get_config_from_env()
        apply_high_accuracy_mode(self.cfg, True)

        # core
        self._dry_run = bool(dry_run)
        self._lock = threading.RLock()
        self._run = threading.Event()

        self._mt5_ready = False
        self._thread: Optional[threading.Thread] = None

        self._feed: Optional[MarketFeed] = None
        self._features: Optional[Classic_FeatureEngine] = None
        self._risk: Optional[RiskManager] = None
        self._signal: Optional[SignalEngine] = None

        # queues (bounded)
        self._max_queue = int(getattr(self.cfg, "max_exec_queue", 10) or 10)
        self._order_q: "queue.Queue[OrderIntent]" = queue.Queue(maxsize=self._max_queue)
        self._result_q: "queue.Queue[ExecutionResult]" = queue.Queue(maxsize=max(50, self._max_queue * 5))
        self._exec_worker: Optional[ExecutionWorker] = None

        # state
        self._order_counter = 0
        self._last_signal = "Neutral"
        self._last_latency_ms = 0.0
        self._last_reconcile_ts = 0.0

        # diagnostics
        self._loop_iteration = 0

        self._last_market_ok_ts = 0.0
        self._market_validate_every = float(getattr(self.cfg, "market_validate_interval_sec", 2.0) or 2.0)

        # idempotency (cooldown)
        self._seen_signals = deque(maxlen=4000)  # (signal_id, ts)
        self._seen_index: Dict[str, float] = {}
        self._signal_cooldown_sec = float(getattr(self.cfg, "signal_cooldown_sec", 2.0) or 2.0)

        # loop timings
        self._poll_fast = float(getattr(self.cfg, "poll_seconds_fast", 0.15) or 0.15)
        self._poll_slow = float(getattr(self.cfg, "poll_seconds_slow", 0.9) or 0.9)

        self._reconcile_interval = float(getattr(self.cfg, "reconcile_interval_sec", 15.0) or 15.0)

        # recovery control
        self._max_consecutive_errors = int(getattr(self.cfg, "max_consecutive_errors", 10) or 10)
        self._recover_backoff_s = float(getattr(self.cfg, "recover_backoff_sec", 10.0) or 10.0)
        self._last_recover_ts = 0.0

        # health/heartbeat
        self._heartbeat_every = float(getattr(self.cfg, "engine_heartbeat_sec", 5.0) or 5.0)
        self._last_heartbeat_ts = 0.0

        # manual pause flag (telegram stop button)
        self._manual_stop = False

    # -------------------- MT5 + stack --------------------
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

            log_health.info("MT5 initialization successful")
            self._mt5_ready = True
            return True
        except Exception as exc:
            log_err.error("MT5 init error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _ensure_symbol(self) -> None:
        symbol = self.cfg.symbol_params.resolved or self.cfg.symbol_params.base
        with MT5_LOCK:
            info = mt5.symbol_info(symbol)
            if info is None:
                ok = mt5.symbol_select(symbol, True)
                if not ok:
                    raise RuntimeError(f"symbol_select failed: {symbol}")
            else:
                if hasattr(info, "visible") and not info.visible:
                    ok = mt5.symbol_select(symbol, True)
                    if not ok:
                        raise RuntimeError(f"symbol_select failed (invisible): {symbol}")

    def _build_stack(self) -> None:
        sym = self.cfg.symbol_params
        self._feed = MarketFeed(self.cfg, sym)
        self._features = Classic_FeatureEngine(self.cfg)
        self._risk = RiskManager(self.cfg, sym)
        self._signal = SignalEngine(self.cfg, sym, self._feed, self._features, self._risk)

    def _cleanup_stack(self) -> None:
        self._signal = None
        self._risk = None
        self._features = None
        self._feed = None

    def _check_mt5_health(self) -> bool:
        try:
            with MT5_LOCK:
                term = mt5.terminal_info()
                acc = mt5.account_info()
            return bool(term and getattr(term, "connected", False) and getattr(term, "trade_allowed", False) and acc)
        except Exception as exc:
            log_err.error("MT5 health error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _recover_all(self) -> bool:
        now = time.time()
        if (now - self._last_recover_ts) < self._recover_backoff_s:
            return False  # prevent restart storm
        self._last_recover_ts = now

        self._mt5_ready = False
        self._cleanup_stack()

        if not self._init_mt5():
            return False

        try:
            self._ensure_symbol()
            self._build_stack()
            return True
        except Exception as exc:
            log_err.error("Recover rebuild error: %s | tb=%s", exc, traceback.format_exc())
            return False

    # -------------------- market data validation --------------------
    def _validate_market_data(self) -> bool:
        if not self._feed:
            return False

        now = time.time()
        if (now - self._last_market_ok_ts) < self._market_validate_every:
            return True

        try:
            df = self._feed.fetch_rates(self.cfg.symbol_params.tf_primary, 240)
            if df is None or df.empty or len(df) < 80:
                return False

            age = float(self._feed.last_bar_age(df))
            if age > 180.0:
                return False

            if df["close"].isna().any() or (df["close"] <= 0).any():
                return False

            self._last_market_ok_ts = now
            return True
        except Exception as exc:
            log_err.error("market_data_validate error: %s | tb=%s", exc, traceback.format_exc())
            return False

    # -------------------- idempotency --------------------
    def _is_duplicate_signal(self, signal_id: str, now: float) -> bool:
        last = self._seen_index.get(signal_id)
        return bool(last is not None and (now - last) < self._signal_cooldown_sec)

    def _mark_signal_seen(self, signal_id: str, now: float) -> None:
        self._seen_index[signal_id] = now
        self._seen_signals.append((signal_id, now))
        # cleanup old entries (60s window)
        while self._seen_signals and (now - self._seen_signals[0][1]) > 60.0:
            sid, ts = self._seen_signals.popleft()
            if self._seen_index.get(sid) == ts:
                self._seen_index.pop(sid, None)

    # -------------------- order enqueue --------------------
    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._order_counter}"

    def _enqueue_order(self, res: SignalResult) -> Tuple[bool, Optional[str]]:
        if not self._risk:
            return False, None

        symbol = self.cfg.symbol_params.resolved or self.cfg.symbol_params.base
        now = time.time()

        signal_id = getattr(res, "signal_id", None) or f"SIG_{int(now * 1000)}"

        if self._is_duplicate_signal(signal_id, now):
            log_health.info("ENQUEUE_SKIP | reason=duplicate signal_id=%s", signal_id)
            return False, None

        lot = float(getattr(res, "lot", 0.0) or 0.0)
        if lot <= 0.0:
            log_health.info(
                "ENQUEUE_SKIP | reason=no_lot signal=%s conf=%.3f lot=%.4f",
                res.signal,
                float(getattr(res, "confidence", 0.0) or 0.0),
                lot,
            )
            return False, None

        with MT5_LOCK:
            tick = mt5.symbol_info_tick(symbol)

        if tick is None:
            if hasattr(self._risk, "record_execution_failure"):
                oid = self._next_order_id()
                self._risk.record_execution_failure(oid, now, now, "tick_missing_enqueue")
            return False, None

        price = float(tick.ask if res.signal == "Buy" else tick.bid)
        if price <= 0:
            log_health.info("ENQUEUE_SKIP | reason=bad_price signal=%s price=%.5f", res.signal, price)
            return False, None

        order_id = self._next_order_id()
        idem_key = f"{symbol}:{signal_id}:{res.signal}"

        order = OrderIntent(
            signal=str(res.signal),
            confidence=float(res.confidence) / 100.0 if float(res.confidence) > 1 else float(res.confidence),
            lot=lot,
            sl=float(getattr(res, "sl", 0.0) or 0.0),
            tp=float(getattr(res, "tp", 0.0) or 0.0),
            price=price,
            symbol=symbol,
            enqueue_time=now,
            order_id=order_id,
            signal_id=signal_id,
            idempotency_key=idem_key,
        )

        try:
            self._order_q.put_nowait(order)
        except queue.Full:
            if hasattr(self._risk, "record_execution_failure"):
                self._risk.record_execution_failure(order_id, now, now, "queue_backlog_drop")
            log_health.warning("ENQUEUE_FAIL | reason=queue_full signal=%s order_id=%s", res.signal, order_id)
            return False, None

        self._mark_signal_seen(signal_id, now)

        if hasattr(self._risk, "track_signal_survival"):
            try:
                self._risk.track_signal_survival(
                    order_id, res.signal, price, order.sl, order.tp, now, float(res.confidence)
                )
            except Exception:
                pass

        return True, order_id

    # -------------------- reconcile --------------------
    def _reconcile_positions(self) -> None:
        now = time.time()
        if (now - self._last_reconcile_ts) < self._reconcile_interval:
            return
        self._last_reconcile_ts = now

        try:
            symbol = self.cfg.symbol_params.resolved or self.cfg.symbol_params.base
            with MT5_LOCK:
                pos = mt5.positions_get(symbol=symbol) or []
            open_cnt = len(pos)

            if self._risk and hasattr(self._risk, "on_reconcile_positions"):
                try:
                    self._risk.on_reconcile_positions(pos)
                except Exception:
                    pass

            max_expected = int(getattr(self.cfg, "max_positions", 3) or 3)
            if open_cnt > max_expected:
                log_err.error("RECONCILE: positions exceed max_expected | open=%s max=%s", open_cnt, max_expected)

        except Exception as exc:
            log_err.error("reconcile error: %s | tb=%s", exc, traceback.format_exc())

    # -------------------- heartbeat --------------------
    def _heartbeat(self) -> None:
        now = time.time()
        if (now - self._last_heartbeat_ts) < self._heartbeat_every:
            return
        self._last_heartbeat_ts = now

        symbol = self.cfg.symbol_params.resolved or self.cfg.symbol_params.base

        connected = False
        open_trades = 0
        bal = 0.0
        eq = 0.0
        dd_pct = 0.0
        pnl = 0.0

        try:
            with MT5_LOCK:
                term = mt5.terminal_info()
                acc = mt5.account_info()
                pos = mt5.positions_get(symbol=symbol) or []

            connected = bool(term and getattr(term, "connected", False))
            open_trades = len(pos)

            if acc:
                bal = float(getattr(acc, "balance", 0.0) or 0.0)
                eq = float(getattr(acc, "equity", 0.0) or 0.0)
                pnl = float(getattr(acc, "profit", 0.0) or 0.0)

            if bal > 0:
                dd_pct = max(0.0, (bal - eq) / bal)

            phase = ""
            if self._risk and hasattr(self._risk, "current_phase"):
                phase = str(getattr(self._risk, "current_phase") or "")

            log_health.info(
                "heartbeat | running=%s connected=%s symbol=%s open=%s bal=%.2f eq=%.2f dd=%.4f pnl=%.2f "
                "last_signal=%s lat=%.1fms q=%s phase=%s",
                self._run.is_set(),
                connected,
                symbol,
                open_trades,
                bal,
                eq,
                dd_pct,
                pnl,
                self._last_signal,
                self._last_latency_ms,
                self._order_q.qsize(),
                phase,
            )
        except Exception as exc:
            log_err.error("heartbeat error: %s | tb=%s", exc, traceback.format_exc())

    # -------------------- lifecycle --------------------
    def start(self) -> None:
        with self._lock:
            if self._run.is_set():
                return

            if self._manual_stop:
                log_health.info("ENGINE_START_SKIP | reason=manual_stop")
                return

            if not self._mt5_ready and not self._init_mt5():
                raise RuntimeError("MT5 init failed")

            self._ensure_symbol()
            self._build_stack()

            self._run.set()

            self._exec_worker = ExecutionWorker(
                self._order_q,
                self._result_q,
                self._risk,
                self.cfg,
                dry_run=self._dry_run,
            )
            self._exec_worker.start()

            self._thread = threading.Thread(target=self._loop, name="engine.loop", daemon=True)
            self._thread.start()

            log_health.info(
                "ENGINE_START | dry_run=%s symbol=%s",
                self._dry_run,
                self.cfg.symbol_params.resolved or self.cfg.symbol_params.base,
            )

    def stop(self) -> bool:
        with self._lock:
            if not self._run.is_set():
                return False
            was_running = True
            self._run.clear()

            if self._exec_worker:
                self._exec_worker.stop()

        if self._thread:
            self._thread.join(timeout=5.0)
        if self._exec_worker:
            self._exec_worker.join(timeout=5.0)

        try:
            while True:
                _ = self._order_q.get_nowait()
                self._order_q.task_done()
        except Exception:
            pass

        log_health.info("ENGINE_STOP")
        return was_running

    def request_manual_stop(self) -> bool:
        with self._lock:
            self._manual_stop = True
        return self.stop()

    def clear_manual_stop(self) -> None:
        with self._lock:
            if self._manual_stop:
                log_health.info("MANUAL_STOP_CLEAR")
            self._manual_stop = False

    def manual_stop_active(self) -> bool:
        with self._lock:
            return bool(self._manual_stop)

    def status(self) -> EngineStatus:
        symbol = self.cfg.symbol_params.resolved or self.cfg.symbol_params.base

        try:
            with MT5_LOCK:
                acc = mt5.account_info()
                positions = mt5.positions_get(symbol=symbol) or []

            bal = float(getattr(acc, "balance", 0.0) or 0.0) if acc else 0.0
            eq = float(getattr(acc, "equity", 0.0) or 0.0) if acc else 0.0
            pnl = float(getattr(acc, "profit", 0.0) or 0.0) if acc else 0.0

            dd_pct = max(0.0, (bal - eq) / bal) if bal > 0 else 0.0

            phase = ""
            if self._risk and hasattr(self._risk, "current_phase"):
                phase = str(getattr(self._risk, "current_phase") or "")

            return EngineStatus(
                connected=bool(self._mt5_ready),
                trading=bool(self._run.is_set()),
                manual_stop=bool(self._manual_stop),
                balance=bal,
                equity=eq,
                dd_pct=float(dd_pct),
                today_pnl=float(pnl),
                open_trades=len(positions),
                phase=phase,
                last_signal=str(self._last_signal),
                latency_ms=float(self._last_latency_ms),
                queue_size=int(self._order_q.qsize()),
                last_reconcile_ts=float(self._last_reconcile_ts),
            )
        except Exception as exc:
            log_err.error("status error: %s | tb=%s", exc, traceback.format_exc())
            return EngineStatus(
                connected=bool(self._mt5_ready),
                trading=bool(self._run.is_set()),
                manual_stop=bool(self._manual_stop),
                balance=0.0,
                equity=0.0,
                dd_pct=0.0,
                today_pnl=0.0,
                open_trades=0,
                phase="ERROR",
                last_signal=str(self._last_signal),
                latency_ms=float(self._last_latency_ms),
                queue_size=int(self._order_q.qsize()),
                last_reconcile_ts=float(self._last_reconcile_ts),
            )

    # -------------------- main loop --------------------
    def _loop(self) -> None:
        if not self._signal:
            log_err.error("Engine loop start failed: SignalEngine is None")
            self._run.clear()
            return

        consecutive_errors = 0

        while self._run.is_set():
            t0 = time.time()
            try:
                self._heartbeat()

                mt5_ok = self._check_mt5_health()
                log_health.info("PIPELINE_STAGE | step=mt5_health ok=%s", mt5_ok)
                if not mt5_ok:
                    consecutive_errors += 1
                    if consecutive_errors >= self._max_consecutive_errors:
                        ok = self._recover_all()
                        if ok:
                            consecutive_errors = 0
                        else:
                            time.sleep(0.8)
                    else:
                        time.sleep(0.6)
                    continue

                if self._risk and hasattr(self._risk, "requires_hard_stop") and self._risk.requires_hard_stop():
                    try:
                        close_all_position()
                    except Exception:
                        pass
                    log_health.info("HARD_STOP_TRIGGERED")
                    self._run.clear()
                    break

                data_ok = self._validate_market_data()
                log_health.info("PIPELINE_STAGE | step=market_data ok=%s", data_ok)
                if not data_ok:
                    consecutive_errors += 1
                    if consecutive_errors >= self._max_consecutive_errors:
                        ok = self._recover_all()
                        if ok:
                            consecutive_errors = 0
                        else:
                            time.sleep(0.8)
                    else:
                        time.sleep(0.5)
                    continue

                self._reconcile_positions()

                while True:
                    try:
                        r = self._result_q.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        if self._risk and hasattr(self._risk, "on_execution_result"):
                            self._risk.on_execution_result(r)
                    except Exception:
                        pass
                    finally:
                        try:
                            self._result_q.task_done()
                        except Exception:
                            pass

                prev_signal = self._last_signal
                self._loop_iteration += 1

                res: SignalResult = self._signal.compute(execute=False)

                current_signal = str(getattr(res, "signal", "Neutral"))
                self._last_signal = current_signal
                self._last_latency_ms = float(getattr(res, "latency_ms", 0.0) or 0.0)

                conf_raw = float(getattr(res, "confidence", 0.0) or 0.0)
                repeat = current_signal == prev_signal
                is_blocked = bool(getattr(res, "trade_blocked", False))
                reasons = list(getattr(res, "reasons", []) or [])
                reason_str = ",".join(str(r) for r in reasons[:5]) if reasons else "-"

                log_health.info(
                    "PIPELINE_STEP | iter=%s signal=%s prev=%s same=%s blocked=%s conf_raw=%.2f q=%s latency=%.1fms reasons=%s",
                    self._loop_iteration,
                    current_signal,
                    prev_signal,
                    repeat,
                    is_blocked,
                    conf_raw,
                    self._order_q.qsize(),
                    self._last_latency_ms,
                    reason_str,
                )

                consecutive_errors = 0

                if self._last_signal in ("Buy", "Sell"):
                    if not self._risk:
                        time.sleep(0.25)
                        continue

                    conf = conf_raw / 100.0 if conf_raw > 1 else conf_raw
                    decision = self._risk.can_trade(conf, self._last_signal)

                    lot_val = getattr(res, "lot", None)
                    try:
                        lot_float = float(lot_val) if lot_val is not None else 0.0
                    except Exception:
                        lot_float = 0.0

                    phase = str(getattr(self._risk, "current_phase", "") or "")
                    risk_reasons = list(getattr(decision, "reasons", []) or [])

                    log_health.info(
                        "RISK_DECISION | signal=%s conf=%.4f allowed=%s lot=%.4f phase=%s reasons=%s",
                        self._last_signal,
                        conf,
                        getattr(decision, "allowed", False),
                        lot_float,
                        phase,
                        ",".join(str(r) for r in risk_reasons[:5]) if risk_reasons else "-",
                    )

                    if getattr(decision, "allowed", False):
                        try:
                            self._risk.register_signal_emitted()
                        except Exception:
                            pass

                        ok, order_id = self._enqueue_order(res)
                        if ok:
                            log_health.info(
                                "ENQUEUE_OK | order_id=%s signal=%s conf=%.4f lot=%s",
                                order_id,
                                self._last_signal,
                                conf,
                                getattr(res, "lot", None),
                            )
                        else:
                            log_health.info(
                                "ENQUEUE_NOOP | signal=%s conf=%.4f reason=enqueue_rejected",
                                self._last_signal,
                                conf,
                            )
                    else:
                        log_health.info(
                            "RISK_BLOCK | signal=%s conf=%.4f phase=%s reasons=%s",
                            self._last_signal,
                            conf,
                            phase,
                            ",".join(str(r) for r in risk_reasons[:5]) if risk_reasons else "-",
                        )
                        time.sleep(0.25)
                        continue

                dt = time.time() - t0
                poll = self._poll_fast

                if self._order_q.qsize() >= max(3, self._max_queue // 2):
                    poll = self._poll_slow
                if self._last_latency_ms >= 250.0:
                    poll = self._poll_slow
                if consecutive_errors > 0:
                    poll = self._poll_slow

                time.sleep(max(0.08, poll - dt))

            except Exception as exc:
                consecutive_errors += 1
                log_err.error("loop error: %s | tb=%s", exc, traceback.format_exc())

                if consecutive_errors >= self._max_consecutive_errors:
                    ok = self._recover_all()
                    if ok:
                        consecutive_errors = 0
                    else:
                        time.sleep(1.0)
                else:
                    time.sleep(0.6)


# Global instance
engine = TradingEngine()
