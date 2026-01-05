"""Bot/portfolio_engine.py

Multi-asset (XAU + BTC) portfolio engine.

Goal:
- Run BOTH pipelines continuously (data validation, signals, risk).
- Execute trades for ONLY ONE asset at a time:
  - If there are open positions in XAU -> only XAU is allowed to open/manage new trades.
  - If there are open positions in BTC -> only BTC is allowed.
  - If there are no positions -> choose the better candidate (confidence-based).

This module is production-oriented:
- Rotating logs (error + health)
- Dedicated execution worker thread (non-blocking)
- MT5_LOCK respected
- Idempotency (signal cooldown) to prevent duplicate orders
"""

from __future__ import annotations

import logging
import re
import os
import builtins
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import MetaTrader5 as mt5

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ExnessAPI.order_execution import OrderExecutor, OrderRequest, OrderResult as ExecOrderResult
from ExnessAPI.orders import close_all_position
from mt5_client import MT5_LOCK, ensure_mt5

# -------------------- XAU stack --------------------
from config_xau import EngineConfig as XauConfig
from config_xau import apply_high_accuracy_mode as xau_apply_high_accuracy_mode
from config_xau import get_config_from_env as get_xau_config
from DataFeed.market_feed import MarketFeed as XauMarketFeed
from StrategiesXau.indicators import Classic_FeatureEngine as XauFeatureEngine
from StrategiesXau.risk_management import RiskManager as XauRiskManager
from StrategiesXau.signal_engine import SignalEngine as XauSignalEngine
from StrategiesXau.signal_engine import SignalResult as XauSignalResult

# -------------------- BTC stack --------------------
from config_btc import EngineConfig as BtcConfig
from config_btc import apply_high_accuracy_mode as btc_apply_high_accuracy_mode
from config_btc import get_config_from_env as get_btc_config
from DataFeed.btc_feed import MarketFeed as BtcMarketFeed
from StrategiesBtc.indicators import Classic_FeatureEngine as BtcFeatureEngine
from StrategiesBtc.risk_management import RiskManager as BtcRiskManager
from StrategiesBtc.signal_engine import SignalEngine as BtcSignalEngine
from StrategiesBtc.signal_engine import SignalResult as BtcSignalResult
from log_config import LOG_DIR as LOG_ROOT, get_log_path


# =============================================================================
# Logging
# =============================================================================
LOG_DIR = LOG_ROOT


def _rotating_file_logger(
    name: str,
    filename: str,
    level: int,
    max_bytes_env: str,
    backups_env: str,
    default_max_bytes: int = 5_242_880,  # 5MB
    default_backups: int = 5,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        fh = RotatingFileHandler(
            filename=str(get_log_path(filename)),
            maxBytes=int(os.getenv(max_bytes_env, str(default_max_bytes))),
            backupCount=int(os.getenv(backups_env, str(default_backups))),
            encoding="utf-8",
            delay=True,
        )
        fh.setLevel(level)
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)

    return logger


log_err = _rotating_file_logger(
    "portfolio.engine",
    "portfolio_engine.log",
    logging.ERROR,
    max_bytes_env="PORTFOLIO_ENGINE_LOG_MAX_BYTES",
    backups_env="PORTFOLIO_ENGINE_LOG_BACKUPS",
)

log_health = _rotating_file_logger(
    "portfolio.engine.health",
    "portfolio_engine_health.log",
    logging.INFO,
    max_bytes_env="PORTFOLIO_ENGINE_HEALTH_LOG_MAX_BYTES",
    backups_env="PORTFOLIO_ENGINE_HEALTH_LOG_BACKUPS",
)

# =============================================================================
# Timeframe helpers
# =============================================================================
_TF_SECONDS_MAP = {
    getattr(mt5, "TIMEFRAME_M1", -1): 60,
    getattr(mt5, "TIMEFRAME_M2", -1): 120,
    getattr(mt5, "TIMEFRAME_M3", -1): 180,
    getattr(mt5, "TIMEFRAME_M4", -1): 240,
    getattr(mt5, "TIMEFRAME_M5", -1): 300,
    getattr(mt5, "TIMEFRAME_M6", -1): 360,
    getattr(mt5, "TIMEFRAME_M10", -1): 600,
    getattr(mt5, "TIMEFRAME_M12", -1): 720,
    getattr(mt5, "TIMEFRAME_M15", -1): 900,
    getattr(mt5, "TIMEFRAME_M20", -1): 1200,
    getattr(mt5, "TIMEFRAME_M30", -1): 1800,
    getattr(mt5, "TIMEFRAME_H1", -1): 3600,
    getattr(mt5, "TIMEFRAME_H2", -1): 7200,
    getattr(mt5, "TIMEFRAME_H3", -1): 10800,
    getattr(mt5, "TIMEFRAME_H4", -1): 14400,
    getattr(mt5, "TIMEFRAME_H6", -1): 21600,
    getattr(mt5, "TIMEFRAME_H8", -1): 28800,
    getattr(mt5, "TIMEFRAME_H12", -1): 43200,
    getattr(mt5, "TIMEFRAME_D1", -1): 86400,
    getattr(mt5, "TIMEFRAME_W1", -1): 604800,
    getattr(mt5, "TIMEFRAME_MN1", -1): 2592000,
}


def _tf_seconds(tf: Any) -> Optional[float]:
    """Return timeframe seconds for MT5 enum int or common string like 'M1','M15','H1'."""
    if tf is None:
        return None
    if isinstance(tf, int):
        sec = _TF_SECONDS_MAP.get(tf)
        return float(sec) if sec else None
    if isinstance(tf, str):
        s = tf.strip().upper()
        if s.startswith("TIMEFRAME_"):
            s = s.replace("TIMEFRAME_", "")
        m = re.fullmatch(r"([MHDW])\s*(\d+)", s)
        if m:
            unit, n = m.group(1), int(m.group(2))
            if unit == "M":
                return float(n * 60)
            if unit == "H":
                return float(n * 3600)
            if unit == "D":
                return float(n * 86400)
            if unit == "W":
                return float(n * 604800)
        if s in ("MN1", "MO1", "MON1", "MONTH1"):
            return float(2592000)
    return None



# =============================================================================
# Data models
# =============================================================================
@dataclass(frozen=True)
class AssetCandidate:
    asset: str              # "XAU" / "BTC"
    symbol: str
    signal: str             # "Buy" / "Sell" / "Neutral"
    confidence: float       # 0..1
    lot: float
    sl: float
    tp: float
    latency_ms: float
    blocked: bool
    reasons: Tuple[str, ...]
    signal_id: str
    raw_result: Any


@dataclass(frozen=True)
class PortfolioStatus:
    connected: bool
    trading: bool
    manual_stop: bool
    active_asset: str
    balance: float
    equity: float
    dd_pct: float
    today_pnl: float
    open_trades_total: int
    open_trades_xau: int
    open_trades_btc: int
    last_signal_xau: str
    last_signal_btc: str
    last_selected_asset: str
    exec_queue_size: int
    last_reconcile_ts: float


@dataclass(frozen=True)
class OrderIntent:
    asset: str
    symbol: str
    signal: str
    lot: float
    sl: float
    tp: float
    price: float
    enqueue_time: float
    order_id: str
    signal_id: str
    idempotency_key: str
    risk_manager: Any
    cfg: Any


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
# Asset Pipeline
# =============================================================================
class _AssetPipeline:
    def __init__(
        self,
        asset: str,
        cfg: Any,
        feed: Any,
        features: Any,
        risk: Any,
        signal: Any,
    ) -> None:
        self.asset = str(asset)
        self.cfg = cfg
        self.feed = feed
        self.features = features
        self.risk = risk
        self.signal = signal

        self.last_signal = "Neutral"
        self.last_latency_ms = 0.0

        self.last_market_ok = True
        self.last_market_reason = "init"
        self.last_bar_age_sec = 0.0
        self.last_market_rows = 0
        self.last_market_close = 0.0
        self.last_market_volume = 0.0
        self.last_market_ts = "-"
        self.last_market_tf = str(getattr(self.cfg.symbol_params, "tf_primary", "-")) if getattr(self.cfg, "symbol_params", None) else "-"

        self._signal_log_every = float(os.getenv("PORTFOLIO_SIGNAL_LOG_SEC", "5.0"))
        self._last_signal_log_ts = 0.0

        # Market data freshness thresholds
        self._max_bar_age_mult = float(os.getenv("PORTFOLIO_MAX_BAR_AGE_MULT", "2.5"))
        self._min_bar_age_sec = float(os.getenv("PORTFOLIO_MIN_BAR_AGE_SEC", "180"))

        self._last_market_ok_ts = 0.0
        self._market_validate_every = float(getattr(cfg, "market_validate_interval_sec", 2.0) or 2.0)
        self._last_reconcile_ts = 0.0
        self._reconcile_interval = float(getattr(cfg, "reconcile_interval_sec", 15.0) or 15.0)

    @property
    def symbol(self) -> str:
        sp = getattr(self.cfg, "symbol_params", None)
        if not sp:
            return ""
        return str(getattr(sp, "resolved", "") or getattr(sp, "base", ""))

    def ensure_symbol_selected(self) -> None:
        symbol = self.symbol
        if not symbol:
            raise RuntimeError(f"{self.asset}: empty symbol")
        with MT5_LOCK:
            info = mt5.symbol_info(symbol)
            if info is None:
                ok = mt5.symbol_select(symbol, True)
                if not ok:
                    raise RuntimeError(f"{self.asset}: symbol_select failed: {symbol}")
            else:
                if hasattr(info, "visible") and not info.visible:
                    ok = mt5.symbol_select(symbol, True)
                    if not ok:
                        raise RuntimeError(f"{self.asset}: symbol_select failed (invisible): {symbol}")

    def _fetch_df_for_validation(self):
        tf = getattr(self.cfg.symbol_params, "tf_primary", None)
        if tf is None:
            self.last_market_reason = "tf_none"
            log_err.error("%s: tf_primary is None in config", self.asset)
            return None

        # Prefer feed.fetch_rates(tf, bars) if exists; else fallback.
        if hasattr(self.feed, "fetch_rates"):
            try:
                df = self.feed.fetch_rates(tf, 240)
                if df is not None:
                    return df
            except Exception as exc:
                # Warning must go to health logger (error logger is ERROR-only).
                log_health.warning("%s: fetch_rates failed: %s", self.asset, exc)

        # Try common get_rates(...) signatures
        if hasattr(self.feed, "get_rates"):
            # (tf, bars)
            try:
                df = self.feed.get_rates(tf, 240)
                if df is not None:
                    return df
            except TypeError:
                pass
            except Exception as exc:
                log_health.warning("%s: get_rates(tf,bars) failed: %s", self.asset, exc)

            # (symbol, tf, bars)
            try:
                df = self.feed.get_rates(self.symbol, tf, 240)
                if df is not None:
                    return df
            except TypeError:
                pass
            except Exception as exc:
                log_health.warning("%s: get_rates(symbol,tf,bars) failed: %s", self.asset, exc)

            # (symbol, tf)
            try:
                df = self.feed.get_rates(self.symbol, tf)
                if df is not None:
                    return df
            except Exception:
                pass

        # Final fallback: MT5 copy_rates_from_pos
        if pd is None:
            return None
        try:
            with MT5_LOCK:
                rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, 240)
            if rates is None or len(rates) == 0:
                return None
            return pd.DataFrame(rates)
        except Exception as exc:
            log_err.error("%s: mt5.copy_rates_from_pos fallback failed: %s | tb=%s", self.asset, exc, traceback.format_exc())
            return None


    def validate_market_data(self) -> bool:
        now = time.time()

        # Cache only SUCCESS for a short interval (prevents log spam and heavy fetch).
        if self.last_market_ok and (now - self._last_market_ok_ts) < self._market_validate_every:
            self.last_market_reason = "cached_ok"
            return True

        tf = getattr(self.cfg.symbol_params, "tf_primary", None)
        tf_sec = _tf_seconds(tf)
        max_age = float(self._min_bar_age_sec)
        if tf_sec:
            max_age = max(max_age, float(tf_sec) * float(self._max_bar_age_mult))

        try:
            df = self._fetch_df_for_validation()
            if df is None:
                self.last_market_ok = False
                self.last_market_reason = "df_none"
                return False

            if getattr(df, "empty", False) or len(df) < 80:
                n = 0 if getattr(df, "empty", False) else int(len(df))
                self.last_market_ok = False
                self.last_market_reason = f"df_short:{n}"
                return False

            # Age check: dynamic by timeframe (fixes M15/H1 always-failing under 180s).
            age: Optional[float] = None
            last_ts: Optional[float] = None
            if hasattr(self.feed, "last_bar_age"):
                try:
                    age = float(self.feed.last_bar_age(df))
                except Exception:
                    age = None

            if age is None:
                try:
                    last_t = None
                    if hasattr(df, "columns") and "time" in df.columns:
                        last_t = df["time"].iloc[-1]
                    elif hasattr(df, "columns") and "datetime" in df.columns:
                        last_t = df["datetime"].iloc[-1]
                    elif hasattr(df, "index") and len(df.index) > 0:
                        last_t = df.index[-1]

                    if last_t is not None:
                        if hasattr(last_t, "timestamp"):
                            last_ts = float(last_t.timestamp())
                        else:
                            last_ts = float(last_t)
                        # ms -> s
                        if last_ts > 1e12:
                            last_ts /= 1000.0
                        age = float(now - last_ts)
                except Exception:
                    age = None

            if last_ts is None:
                try:
                    if hasattr(df, "columns") and "time" in df.columns:
                        last_t = df["time"].iloc[-1]
                        if hasattr(last_t, "timestamp"):
                            last_ts = float(last_t.timestamp())
                        else:
                            last_ts = float(last_t)
                        if last_ts > 1e12:
                            last_ts /= 1000.0
                except Exception:
                    last_ts = None

            self.last_bar_age_sec = float(age or 0.0)
            if age is not None and age > max_age:
                self.last_market_ok = False
                self.last_market_reason = f"stale:{age:.1f}>{max_age:.1f}"
                return False

            # Basic sanity
            if hasattr(df, "columns") and "close" in df.columns:
                s = df["close"]
                try:
                    if s.isna().any():
                        self.last_market_ok = False
                        self.last_market_reason = "nan_close"
                        return False
                    if (s <= 0).any():
                        self.last_market_ok = False
                        self.last_market_reason = "bad_close"
                        return False
                    self.last_market_close = float(s.iloc[-1])
                except Exception:
                    pass

            self.last_market_rows = int(len(df))
            if hasattr(df, "columns") and "tick_volume" in df.columns:
                try:
                    self.last_market_volume = float(df["tick_volume"].iloc[-1])
                except Exception:
                    self.last_market_volume = 0.0
            else:
                self.last_market_volume = 0.0

            if last_ts is not None:
                try:
                    self.last_market_ts = datetime.utcfromtimestamp(last_ts).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    self.last_market_ts = "-"
            else:
                self.last_market_ts = "-"

            self.last_market_tf = str(tf or "-")

            self._last_market_ok_ts = now
            self.last_market_ok = True
            self.last_market_reason = "ok"
            return True

        except Exception as exc:
            self.last_market_ok = False
            self.last_market_reason = "exception"
            log_err.error("%s market_data_validate error: %s | tb=%s", self.asset, exc, traceback.format_exc())
            return False


    def reconcile_positions(self) -> None:
        now = time.time()
        if (now - self._last_reconcile_ts) < self._reconcile_interval:
            return
        self._last_reconcile_ts = now

        try:
            with MT5_LOCK:
                pos = mt5.positions_get(symbol=self.symbol) or []
            if self.risk and hasattr(self.risk, "on_reconcile_positions"):
                try:
                    self.risk.on_reconcile_positions(pos)
                except Exception:
                    pass
        except Exception as exc:
            log_err.error("%s reconcile error: %s | tb=%s", self.asset, exc, traceback.format_exc())

    def open_positions(self) -> int:
        try:
            with MT5_LOCK:
                pos = mt5.positions_get(symbol=self.symbol) or []
            return int(len(pos))
        except Exception:
            return 0

    def compute_candidate(self) -> Optional[AssetCandidate]:
        try:
            # IMPORTANT: execute=True to get LOT/SL/TP plan
            res = self.signal.compute(execute=True)

            signal = str(getattr(res, "signal", "Neutral") or "Neutral")
            self.last_signal = signal
            self.last_latency_ms = float(getattr(res, "latency_ms", 0.0) or 0.0)

            conf_raw = float(getattr(res, "confidence", 0.0) or 0.0)
            conf = conf_raw / 100.0 if conf_raw > 1 else conf_raw
            conf = max(0.0, min(1.0, float(conf)))

            lot = float(getattr(res, "lot", 0.0) or 0.0)
            sl = float(getattr(res, "sl", 0.0) or 0.0)
            tp = float(getattr(res, "tp", 0.0) or 0.0)

            blocked = bool(getattr(res, "trade_blocked", False))
            reasons = tuple(str(r) for r in (getattr(res, "reasons", []) or [])[:8])
            signal_id = str(getattr(res, "signal_id", "") or f"{self.asset}_SIG_{int(time.time() * 1000)}")

            now_ts = time.time()
            if (now_ts - self._last_signal_log_ts) >= self._signal_log_every:
                self._last_signal_log_ts = now_ts
                log_health.info(
                    "PIPELINE_STAGE | step=signals asset=%s signal=%s confidence=%.4f latency_ms=%.1f lot=%.4f sl=%.4f tp=%.4f blocked=%s reasons=%s",
                    self.asset,
                    signal,
                    conf,
                    self.last_latency_ms,
                    lot,
                    sl,
                    tp,
                    blocked,
                    ",".join(reasons) if reasons else "-",
                )

            return AssetCandidate(
                asset=self.asset,
                symbol=self.symbol,
                signal=signal,
                confidence=conf,
                lot=lot,
                sl=sl,
                tp=tp,
                latency_ms=self.last_latency_ms,
                blocked=blocked,
                reasons=reasons,
                signal_id=signal_id,
                raw_result=res,
            )
        except Exception as exc:
            log_err.error("%s compute_candidate error: %s | tb=%s", self.asset, exc, traceback.format_exc())
            return None


# =============================================================================
# Execution Worker (generic, supports per-order RiskManager)
# =============================================================================
class ExecutionWorker(threading.Thread):
    def __init__(
        self,
        order_queue: "queue.Queue[OrderIntent]",
        result_queue: "queue.Queue[ExecutionResult]",
        dry_run: bool,
    ) -> None:
        super().__init__(daemon=True)
        self.order_queue = order_queue
        self.result_queue = result_queue
        self.dry_run = bool(dry_run)
        self._run = threading.Event()
        self._run.set()

    def stop(self) -> None:
        self._run.clear()

    def run(self) -> None:
        while self._run.is_set():
            try:
                intent = self.order_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                res = self._process(intent)
                try:
                    self.result_queue.put_nowait(res)
                except Exception:
                    pass
            except Exception as exc:
                log_err.error("ExecutionWorker crash | err=%s | tb=%s", exc, traceback.format_exc())
            finally:
                try:
                    self.order_queue.task_done()
                except Exception:
                    pass

    @staticmethod
    def _get_hook(risk: Any, name: str) -> Optional[Callable[..., Any]]:
        if not risk:
            return None
        fn = getattr(risk, name, None)
        return fn if callable(fn) else None

    @staticmethod
    def _build_executor(risk: Any) -> OrderExecutor:
        normalize_volume_fn = None
        sanitize_stops_fn = None
        normalize_price_fn = None

        if risk and hasattr(risk, "_normalize_volume_floor"):
            normalize_volume_fn = lambda vol, info: float(risk._normalize_volume_floor(vol))  # type: ignore[attr-defined]
        if risk and hasattr(risk, "_apply_broker_constraints"):
            sanitize_stops_fn = lambda side, price, sl, tp: risk._apply_broker_constraints(side, price, sl, tp)  # type: ignore[attr-defined]
        if risk and hasattr(risk, "_normalize_price"):
            normalize_price_fn = lambda price: float(risk._normalize_price(price))  # type: ignore[attr-defined]

        return OrderExecutor(
            normalize_volume_fn=normalize_volume_fn,
            sanitize_stops_fn=sanitize_stops_fn,
            normalize_price_fn=normalize_price_fn,
        )

    def _process(self, intent: OrderIntent) -> ExecutionResult:
        sent_ts = time.time()
        risk = intent.risk_manager
        cfg = intent.cfg

        # ------------------- DRY RUN -------------------
        if self.dry_run:
            fill_ts = time.time()
            req_price = float(intent.price)
            exec_price = float(intent.price)
            slippage = 0.0

            log_health.info(
                "ORDER_EXEC_PREP | dry_run=%s asset=%s symbol=%s signal=%s order_id=%s price=%.5f lot=%.4f sl=%.5f tp=%.5f",
                True,
                intent.asset,
                intent.symbol,
                intent.signal,
                intent.order_id,
                req_price,
                float(intent.lot),
                float(intent.sl),
                float(intent.tp),
            )

            fn = self._get_hook(risk, "record_execution_metrics")
            if fn:
                try:
                    fn(intent.order_id, intent.signal, intent.enqueue_time, sent_ts, fill_ts, req_price, exec_price, slippage)
                except Exception:
                    pass

            fn = self._get_hook(risk, "record_trade")
            if fn:
                try:
                    fn(exec_price, float(intent.sl), float(intent.lot), float(intent.tp), 1)
                except Exception:
                    pass

            log_health.info(
                "ORDER_EXEC_RESULT | order_id=%s ok=%s reason=%s retcode=%s exec_price=%.5f slippage=%.3f latency_ms=%.1f",
                intent.order_id,
                True,
                "dry_run",
                0,
                exec_price,
                slippage,
                (fill_ts - sent_ts) * 1000.0,
            )
            return ExecutionResult(
                order_id=intent.order_id,
                signal_id=intent.signal_id,
                ok=True,
                reason="dry_run",
                sent_ts=sent_ts,
                fill_ts=fill_ts,
                req_price=req_price,
                exec_price=exec_price,
                volume=float(intent.lot),
                slippage=slippage,
                retcode=0,
            )

        # ------------------- LIVE -------------------
        comment = str(getattr(cfg, "order_comment", "portfolio_bot") or "portfolio_bot")
        deviation = int(getattr(cfg, "deviation", 80) or 80)
        magic = int(getattr(cfg, "magic", 777001) or 777001)

        req = OrderRequest(
            symbol=str(intent.symbol),
            signal=str(intent.signal),
            lot=float(intent.lot),
            sl=float(intent.sl),
            tp=float(intent.tp),
            price=float(intent.price),
            enqueue_time=float(intent.enqueue_time),
            order_id=str(intent.order_id),
            signal_id=str(intent.signal_id),
            deviation=deviation,
            magic=magic,
            comment=comment,
        )

        log_health.info(
            "ORDER_EXEC_PREP | dry_run=%s asset=%s symbol=%s signal=%s order_id=%s price=%.5f lot=%.4f sl=%.5f tp=%.5f",
            False,
            intent.asset,
            intent.symbol,
            intent.signal,
            intent.order_id,
            float(intent.price),
            float(intent.lot),
            float(intent.sl),
            float(intent.tp),
        )

        hooks = {
            "record_execution_metrics": self._get_hook(risk, "record_execution_metrics"),
            "record_trade": self._get_hook(risk, "record_trade"),
            "record_execution_failure": self._get_hook(risk, "record_execution_failure"),
        }
        hooks = {k: v for k, v in hooks.items() if v is not None}

        executor = self._build_executor(risk)
        r: ExecOrderResult = executor.send_market_order(req, max_attempts=3, telemetry_hooks=hooks)

        log_health.info(
            "ORDER_EXEC_RESULT | order_id=%s ok=%s reason=%s retcode=%s exec_price=%.5f slippage=%.3f latency_ms=%.1f",
            r.order_id,
            bool(r.ok),
            str(r.reason),
            int(r.retcode),
            float(r.exec_price),
            float(r.slippage),
            (float(r.fill_ts) - float(r.sent_ts)) * 1000.0,
        )

        return ExecutionResult(
            order_id=r.order_id,
            signal_id=r.signal_id,
            ok=bool(r.ok),
            reason=str(r.reason),
            sent_ts=float(r.sent_ts),
            fill_ts=float(r.fill_ts),
            req_price=float(r.req_price),
            exec_price=float(r.exec_price),
            volume=float(r.volume),
            slippage=float(r.slippage),
            retcode=int(r.retcode),
        )


# =============================================================================
# Multi-asset Trading Engine
# =============================================================================
class MultiAssetTradingEngine:
    def __init__(self, dry_run: bool = False) -> None:
        self._dry_run = bool(dry_run)
        self._lock = threading.RLock()
        self._run = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._mt5_ready = False

        # Manual stop is a hard pause from Telegram
        self._manual_stop = False

        # Portfolio state
        self._active_asset = "NONE"          # runtime lock based on open positions
        self._last_selected_asset = "NONE"
        self._order_counter = 0

        # Idempotency
        self._seen: Deque[Tuple[str, str, float]] = deque(maxlen=8000)  # (asset, signal_id, ts)
        self._seen_index: Dict[Tuple[str, str], float] = {}
        self._signal_cooldown_sec = float(os.getenv("PORTFOLIO_SIGNAL_COOLDOWN_SEC", "2.0"))

        # Queues
        self._max_queue = int(os.getenv("PORTFOLIO_MAX_EXEC_QUEUE", "12"))
        self._order_q: "queue.Queue[OrderIntent]" = queue.Queue(maxsize=self._max_queue)
        self._result_q: "queue.Queue[ExecutionResult]" = queue.Queue(maxsize=max(60, self._max_queue * 6))
        self._exec_worker: Optional[ExecutionWorker] = None

        # Loop timings
        self._poll_fast = float(os.getenv("PORTFOLIO_POLL_FAST", "0.20"))
        self._poll_slow = float(os.getenv("PORTFOLIO_POLL_SLOW", "1.00"))
        self._heartbeat_every = float(os.getenv("PORTFOLIO_HEARTBEAT_SEC", "5.0"))
        self._last_heartbeat_ts = 0.0

        # Throttle repetitive pipeline stage logs
        self._pipeline_log_every = float(os.getenv("PORTFOLIO_PIPELINE_LOG_SEC", "2.0"))
        self._last_pipeline_log_ts = 0.0
        self._last_pipeline_log_key: Tuple[Any, ...] = (None, None, None, None)
        self._last_reconcile_ts = 0.0
        self._reconcile_interval = float(os.getenv("PORTFOLIO_RECONCILE_INTERVAL_SEC", "15.0"))

        # Recovery
        self._max_consecutive_errors = int(os.getenv("PORTFOLIO_MAX_CONSECUTIVE_ERRORS", "12"))
        self._recover_backoff_s = float(os.getenv("PORTFOLIO_RECOVER_BACKOFF_SEC", "10.0"))
        self._last_recover_ts = 0.0

        # Pipelines
        self._xau_cfg: XauConfig = get_xau_config()
        xau_apply_high_accuracy_mode(self._xau_cfg, True)
        self._btc_cfg: BtcConfig = get_btc_config()
        btc_apply_high_accuracy_mode(self._btc_cfg, True)

        self._xau: Optional[_AssetPipeline] = None
        self._btc: Optional[_AssetPipeline] = None

        # cached last candidates
        self._last_cand_xau: Optional[AssetCandidate] = None
        self._last_cand_btc: Optional[AssetCandidate] = None

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
            log_health.info("MT5 initialization successful (portfolio)")
            return True
        except Exception as exc:
            log_err.error("MT5 init error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _check_mt5_health(self) -> bool:
        try:
            with MT5_LOCK:
                term = mt5.terminal_info()
                acc = mt5.account_info()
            return bool(term and getattr(term, "connected", False) and getattr(term, "trade_allowed", False) and acc)
        except Exception as exc:
            log_err.error("MT5 health error: %s | tb=%s", exc, traceback.format_exc())
            return False

    # -------------------- build pipelines --------------------
    def _build_pipelines(self) -> None:
        # XAU
        x_sym = self._xau_cfg.symbol_params
        x_feed = XauMarketFeed(self._xau_cfg, x_sym)
        x_feat = XauFeatureEngine(self._xau_cfg)
        x_risk = XauRiskManager(self._xau_cfg, x_sym)
        x_sig = XauSignalEngine(self._xau_cfg, x_sym, x_feed, x_feat, x_risk)
        self._xau = _AssetPipeline("XAU", self._xau_cfg, x_feed, x_feat, x_risk, x_sig)

        # BTC
        b_sym = self._btc_cfg.symbol_params
        b_feed = BtcMarketFeed(self._btc_cfg, b_sym)
        b_feat = BtcFeatureEngine(self._btc_cfg)
        b_risk = BtcRiskManager(self._btc_cfg, b_sym)
        b_sig = BtcSignalEngine(self._btc_cfg, b_sym, b_feed, b_feat, b_risk)
        self._btc = _AssetPipeline("BTC", self._btc_cfg, b_feed, b_feat, b_risk, b_sig)

        # Ensure symbols are available
        self._xau.ensure_symbol_selected()
        self._btc.ensure_symbol_selected()

    def _restart_exec_worker(self) -> None:
        try:
            if self._exec_worker:
                self._exec_worker.stop()
                self._exec_worker.join(timeout=3.0)
        except Exception:
            pass

        self._exec_worker = ExecutionWorker(self._order_q, self._result_q, dry_run=self._dry_run)
        self._exec_worker.start()

    def _recover_all(self) -> bool:
        now = time.time()
        if (now - self._last_recover_ts) < self._recover_backoff_s:
            return False
        self._last_recover_ts = now

        with self._lock:
            self._mt5_ready = False
            self._xau = None
            self._btc = None

            if not self._init_mt5():
                return False
            try:
                self._build_pipelines()
                if self._run.is_set():
                    self._restart_exec_worker()
                return True
            except Exception as exc:
                log_err.error("Recover rebuild error: %s | tb=%s", exc, traceback.format_exc())
                return False

    # -------------------- idempotency --------------------
    def _is_duplicate(self, asset: str, signal_id: str, now: float) -> bool:
        key = (asset, signal_id)
        last = self._seen_index.get(key)
        return bool(last is not None and (now - last) < self._signal_cooldown_sec)

    def _mark_seen(self, asset: str, signal_id: str, now: float) -> None:
        key = (asset, signal_id)
        self._seen_index[key] = now
        self._seen.append((asset, signal_id, now))
        while self._seen and (now - self._seen[0][2]) > 60.0:
            a, sid, ts = self._seen.popleft()
            if self._seen_index.get((a, sid)) == ts:
                self._seen_index.pop((a, sid), None)

    # -------------------- portfolio logic --------------------
    def _next_order_id(self, asset: str) -> str:
        self._order_counter += 1
        return f"PORD_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._order_counter}"

    @staticmethod
    def _candidate_is_tradeable(c: AssetCandidate) -> bool:
        if c.signal not in ("Buy", "Sell"):
            return False
        if c.blocked:
            return False
        if c.lot <= 0.0:
            return False
        if c.sl <= 0.0 or c.tp <= 0.0:
            return False
        if c.confidence <= 0.0:
            return False
        return True

    @staticmethod
    def _score(c: AssetCandidate) -> float:
        # Strict & stable scoring: confidence dominates.
        return float(c.confidence)

    def _select_active_asset(self, open_xau: int, open_btc: int) -> str:
        if open_xau > 0 and open_btc > 0:
            return "BOTH"  # unsafe state
        if open_xau > 0:
            return "XAU"
        if open_btc > 0:
            return "BTC"
        return "NONE"

    def _enqueue_order(self, cand: AssetCandidate) -> Tuple[bool, Optional[str]]:
        now = time.time()

        if self._is_duplicate(cand.asset, cand.signal_id, now):
            log_health.info("ENQUEUE_SKIP | asset=%s reason=duplicate signal_id=%s", cand.asset, cand.signal_id)
            return False, None

        # Price snapshot
        with MT5_LOCK:
            tick = mt5.symbol_info_tick(cand.symbol)
        if tick is None:
            log_health.info("ENQUEUE_SKIP | asset=%s reason=tick_missing", cand.asset)
            return False, None
        price = float(tick.ask if cand.signal == "Buy" else tick.bid)
        if price <= 0:
            log_health.info("ENQUEUE_SKIP | asset=%s reason=bad_price", cand.asset)
            return False, None

        order_id = self._next_order_id(cand.asset)
        idem = f"{cand.symbol}:{cand.signal_id}:{cand.signal}"

        # route risk/cfg by asset
        if cand.asset == "XAU":
            risk = self._xau.risk if self._xau else None
            cfg = self._xau_cfg
        else:
            risk = self._btc.risk if self._btc else None
            cfg = self._btc_cfg

        intent = OrderIntent(
            asset=cand.asset,
            symbol=cand.symbol,
            signal=cand.signal,
            lot=float(cand.lot),
            sl=float(cand.sl),
            tp=float(cand.tp),
            price=float(price),
            enqueue_time=float(now),
            order_id=str(order_id),
            signal_id=str(cand.signal_id),
            idempotency_key=str(idem),
            risk_manager=risk,
            cfg=cfg,
        )

        try:
            self._order_q.put_nowait(intent)
        except queue.Full:
            # record failure if available
            if risk and hasattr(risk, "record_execution_failure"):
                try:
                    risk.record_execution_failure(order_id, now, now, "queue_backlog_drop")
                except Exception:
                    pass
            log_health.warning("ENQUEUE_FAIL | asset=%s reason=queue_full order_id=%s", cand.asset, order_id)
            return False, None

        self._mark_seen(cand.asset, cand.signal_id, now)
        self._last_selected_asset = cand.asset

        # optional telemetry hook
        if risk and hasattr(risk, "track_signal_survival"):
            try:
                risk.track_signal_survival(order_id, cand.signal, price, cand.sl, cand.tp, now, cand.confidence)
            except Exception:
                pass

        return True, order_id

    # -------------------- heartbeat/status --------------------
    def _heartbeat(self, open_xau: int, open_btc: int) -> None:
        now = time.time()
        if (now - self._last_heartbeat_ts) < self._heartbeat_every:
            return
        self._last_heartbeat_ts = now

        bal = 0.0
        eq = 0.0
        pnl = 0.0
        dd = 0.0
        connected = False
        try:
            with MT5_LOCK:
                term = mt5.terminal_info()
                acc = mt5.account_info()
            connected = bool(term and getattr(term, "connected", False))
            if acc:
                bal = float(getattr(acc, "balance", 0.0) or 0.0)
                eq = float(getattr(acc, "equity", 0.0) or 0.0)
                pnl = float(getattr(acc, "profit", 0.0) or 0.0)
            if bal > 0:
                dd = max(0.0, (bal - eq) / bal)
        except Exception:
            pass

        log_health.info(
            "heartbeat | running=%s connected=%s active=%s last_sel=%s open_xau=%s open_btc=%s total_open=%s bal=%.2f eq=%.2f dd=%.4f pnl=%.2f "
            "sig_xau=%s sig_btc=%s q=%s manual_stop=%s",
            self._run.is_set(),
            connected,
            self._active_asset,
            self._last_selected_asset,
            open_xau,
            open_btc,
            (open_xau + open_btc),
            bal,
            eq,
            dd,
            pnl,
            self._xau.last_signal if self._xau else "-",
            self._btc.last_signal if self._btc else "-",
            self._order_q.qsize(),
            self._manual_stop,
        )

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

            self._build_pipelines()
            self._run.set()
            self._restart_exec_worker()

            self._thread = threading.Thread(target=self._loop, name="portfolio.engine.loop", daemon=True)
            self._thread.start()

            log_health.info(
                "PORTFOLIO_ENGINE_START | dry_run=%s xau=%s btc=%s",
                self._dry_run,
                self._xau.symbol if self._xau else "-",
                self._btc.symbol if self._btc else "-",
            )

    def stop(self) -> bool:
        with self._lock:
            if not self._run.is_set():
                return False
            self._run.clear()
            if self._exec_worker:
                self._exec_worker.stop()

        if self._thread:
            self._thread.join(timeout=6.0)
        if self._exec_worker:
            self._exec_worker.join(timeout=6.0)

        # drain order queue best-effort
        try:
            while True:
                _ = self._order_q.get_nowait()
                self._order_q.task_done()
        except Exception:
            pass

        log_health.info("PORTFOLIO_ENGINE_STOP")
        return True

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

                # Hard stop if any risk wants it
                try:
                    if self._xau.risk and hasattr(self._xau.risk, "requires_hard_stop") and self._xau.risk.requires_hard_stop():
                        close_all_position()
                        log_health.info("HARD_STOP_TRIGGERED | asset=XAU")
                        self._run.clear()
                        break
                    if self._btc.risk and hasattr(self._btc.risk, "requires_hard_stop") and self._btc.risk.requires_hard_stop():
                        close_all_position()
                        log_health.info("HARD_STOP_TRIGGERED | asset=BTC")
                        self._run.clear()
                        break
                except Exception:
                    pass

                # Validate market data
                x_ok = self._xau.validate_market_data()
                b_ok = self._btc.validate_market_data()
                rx = self._xau.last_market_reason if self._xau else "-"
                rb = self._btc.last_market_reason if self._btc else "-"
                ax = float(self._xau.last_bar_age_sec) if self._xau else 0.0
                ab = float(self._btc.last_bar_age_sec) if self._btc else 0.0

                now_ts = time.time()
                key = (x_ok, rx, b_ok, rb)
                if (now_ts - self._last_pipeline_log_ts) >= self._pipeline_log_every or key != self._last_pipeline_log_key:
                    x_bars = int(self._xau.last_market_rows) if self._xau else 0
                    b_bars = int(self._btc.last_market_rows) if self._btc else 0
                    x_close = float(self._xau.last_market_close) if self._xau else 0.0
                    b_close = float(self._btc.last_market_close) if self._btc else 0.0
                    x_vol = float(self._xau.last_market_volume) if self._xau else 0.0
                    b_vol = float(self._btc.last_market_volume) if self._btc else 0.0
                    x_ts = str(self._xau.last_market_ts) if self._xau else "-"
                    b_ts = str(self._btc.last_market_ts) if self._btc else "-"
                    x_tf = str(self._xau.last_market_tf) if self._xau else "-"
                    b_tf = str(self._btc.last_market_tf) if self._btc else "-"
                    log_health.info(
                        "PIPELINE_STAGE | step=market_data ok_xau=%s reason_xau=%s age_xau=%.1fs tf_xau=%s bars_xau=%s close_xau=%.3f vol_xau=%s ts_xau=%s "
                        "ok_btc=%s reason_btc=%s age_btc=%.1fs tf_btc=%s bars_btc=%s close_btc=%.3f vol_btc=%s ts_btc=%s",
                        x_ok,
                        rx,
                        ax,
                        x_tf,
                        x_bars,
                        x_close,
                        int(builtins.round(x_vol)) if x_vol else int(0),
                        x_ts,
                        b_ok,
                        rb,
                        ab,
                        b_tf,
                        b_bars,
                        b_close,
                        int(builtins.round(b_vol)) if b_vol else int(0),
                        b_ts,
                    )
                    self._last_pipeline_log_ts = now_ts
                    self._last_pipeline_log_key = key

                # Reconcile (throttled per pipeline)
                self._xau.reconcile_positions()
                self._btc.reconcile_positions()

                # Drain execution results
                while True:
                    try:
                        r = self._result_q.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        # route to proper risk manager
                        rm = self._xau.risk if (self._xau and r.order_id.startswith("PORD_XAU_")) else self._btc.risk
                        if rm and hasattr(rm, "on_execution_result"):
                            rm.on_execution_result(r)
                    except Exception:
                        pass
                    finally:
                        try:
                            self._result_q.task_done()
                        except Exception:
                            pass

                # Positions and active asset lock
                open_xau = self._xau.open_positions()
                open_btc = self._btc.open_positions()
                self._active_asset = self._select_active_asset(open_xau, open_btc)

                # Detect unsafe state (both open)
                if self._active_asset == "BOTH":
                    log_err.error("UNSAFE_STATE: both XAU and BTC positions are open -> trading frozen")
                    self._heartbeat(open_xau, open_btc)
                    time.sleep(self._poll_slow)
                    continue

                self._heartbeat(open_xau, open_btc)

                # Compute candidates (only if market looks ok, otherwise they will likely be neutral/blocked)
                cand_x = self._xau.compute_candidate() if x_ok else None
                cand_b = self._btc.compute_candidate() if b_ok else None
                self._last_cand_xau = cand_x
                self._last_cand_btc = cand_b

                # Decide which asset may trade now
                candidates: list[AssetCandidate] = []
                if self._active_asset == "XAU":
                    if cand_x:
                        candidates = [cand_x]
                elif self._active_asset == "BTC":
                    if cand_b:
                        candidates = [cand_b]
                else:
                    if cand_x:
                        candidates.append(cand_x)
                    if cand_b:
                        candidates.append(cand_b)

                # Filter tradeable
                candidates = [c for c in candidates if self._candidate_is_tradeable(c)]

                selected: Optional[AssetCandidate] = None
                if candidates:
                    selected = max(candidates, key=self._score)

                if selected:
                    ok, oid = self._enqueue_order(selected)
                    if ok:
                        log_health.info(
                            "ORDER_SELECTED | asset=%s symbol=%s signal=%s conf=%.4f lot=%.4f sl=%s tp=%s order_id=%s",
                            selected.asset,
                            selected.symbol,
                            selected.signal,
                            selected.confidence,
                            selected.lot,
                            selected.sl,
                            selected.tp,
                            oid,
                        )
                    else:
                        log_health.info(
                            "ORDER_SKIP | asset=%s symbol=%s signal=%s conf=%.4f blocked=%s reasons=%s",
                            selected.asset,
                            selected.symbol,
                            selected.signal,
                            selected.confidence,
                            selected.blocked,
                            ",".join(selected.reasons) if selected.reasons else "-",
                        )

                consecutive_errors = 0

                # Adaptive sleep
                dt = time.time() - t0
                poll = self._poll_fast
                if self._order_q.qsize() >= max(3, self._max_queue // 2):
                    poll = self._poll_slow
                if selected and selected.latency_ms >= 250.0:
                    poll = self._poll_slow
                time.sleep(max(0.10, poll - dt))

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


# Global instance (backward-compatible import style)
# Set PORTFOLIO_DRY_RUN=1 to disable real order_send (safe test mode).
engine = MultiAssetTradingEngine(dry_run=bool(int(os.getenv("PORTFOLIO_DRY_RUN", "0"))))
