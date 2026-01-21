from __future__ import annotations

import builtins
import json
import logging
import os
import re
import subprocess
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import MetaTrader5 as mt5

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ExnessAPI.order_execution import (
    OrderExecutor,
    OrderRequest,
    OrderResult as ExecOrderResult,
    log_health as exec_health_log,
)
from ExnessAPI.functions import close_all_position
from mt5_client import MT5_LOCK, ensure_mt5, mt5_status

# -------------------- XAU stack --------------------
from config_xau import EngineConfig as XauConfig
from config_xau import apply_high_accuracy_mode as xau_apply_high_accuracy_mode
from config_xau import get_config_from_env as get_xau_config
from DataFeed.xau_market_feed import MarketFeed as XauMarketFeed
from StrategiesXau.indicators import Classic_FeatureEngine as XauFeatureEngine
from StrategiesXau.risk_management import RiskManager as XauRiskManager
from StrategiesXau.signal_engine import SignalEngine as XauSignalEngine

# -------------------- BTC stack --------------------
from config_btc import EngineConfig as BtcConfig
from config_btc import apply_high_accuracy_mode as btc_apply_high_accuracy_mode
from config_btc import get_config_from_env as get_btc_config
from DataFeed.btc_market_feed import MarketFeed as BtcMarketFeed
from StrategiesBtc.indicators import Classic_FeatureEngine as BtcFeatureEngine
from StrategiesBtc.risk_management import RiskManager as BtcRiskManager
from StrategiesBtc.signal_engine import SignalEngine as BtcSignalEngine

from log_config import LOG_DIR as LOG_ROOT, get_log_path


# =============================================================================
# Logging
# =============================================================================

def _rotating_file_logger(
        name: str, 
        filename: str, 
        level: int, 
        max_bytes: int = 5_242_880, 
        backups: int = 5
    ) -> logging.Logger:
    
    logger = logging.getLogger(name)
    if logger.handlers:  # Кӯтоҳтарин роҳи санҷиши Idempotency
        return logger

    logger.setLevel(level)
    logger.propagate = False

    fh = RotatingFileHandler(
        filename=str(get_log_path(filename)),
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
        delay=True
    )
    
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    return logger


log_err = _rotating_file_logger(
    "portfolio.engine",
    "portfolio_engine.log",
    logging.ERROR,
)

log_health = _rotating_file_logger(
    "portfolio.engine.health",
    "portfolio_engine_health.log",
    logging.INFO,
)

# Structured snapshots (jsonl)
log_diag = _rotating_file_logger(
    "portfolio.engine.diag",
    "portfolio_engine_diag.jsonl",
    logging.INFO,
    max_bytes=10_485_760,  # 10MB
    backups=8,
)

_DIAG_ENABLED = True
_DIAG_EVERY_SEC = 30.0

# =============================================================================
# JSON encoding helper for diagnostics (dataclasses + common non-JSON types)
# =============================================================================
def _json_default(obj: object):
    try:
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass

    try:
        # Common containers
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="ignore")
    except Exception:
        pass

    try:
        from datetime import datetime, date
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
    except Exception:
        pass

    try:
        from pathlib import Path
        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass

    # numpy/pandas scalars (optional)
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except Exception:
        pass

    return str(obj)



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
    asset: str  # "XAU" / "BTC"
    symbol: str
    signal: str  # "Buy" / "Sell" / "Neutral"
    confidence: float  # 0..1
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
    confidence: float
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
    def __init__(self, asset: str, cfg: Any, feed: Any, features: Any, risk: Any, signal: Any) -> None:
        self.asset = str(asset)
        self.cfg = cfg
        self.feed = feed
        self.features = features
        self.risk = risk
        self.signal = signal

        # Signal
        self.last_signal = "Neutral"
        self.last_signal_ts = 0.0
        self.last_latency_ms = 0.0

        # Market snapshot
        self.last_market_ok = True
        self.last_market_reason = "init"
        self.last_bar_age_sec = 0.0
        self.last_market_rows = 0
        self.last_market_close = 0.0
        self.last_market_volume = 0.0
        self.last_market_ts = "-"
        self.last_market_tf = (
            str(getattr(self.cfg.symbol_params, "tf_primary", "-"))
            if getattr(self.cfg, "symbol_params", None)
            else "-"
        )

        self._signal_log_every = 5.0
        self._last_signal_log_ts = 0.0

        # Market freshness thresholds (configurable)
        self._max_bar_age_mult = float(getattr(cfg, "market_max_bar_age_mult", 2.0) or 2.0)
        self._min_bar_age_sec = float(getattr(cfg, "market_min_bar_age_sec", 120.0) or 120.0)

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
                if not mt5.symbol_select(symbol, True):
                    raise RuntimeError(f"{self.asset}: symbol_select failed: {symbol}")
            else:
                if hasattr(info, "visible") and (not bool(info.visible)):
                    if not mt5.symbol_select(symbol, True):
                        raise RuntimeError(f"{self.asset}: symbol_select failed (invisible): {symbol}")

    def _fetch_df_for_validation(self):
        tf = getattr(self.cfg.symbol_params, "tf_primary", None)
        if tf is None:
            self.last_market_reason = "tf_none"
            log_err.error("%s: tf_primary is None in config", self.asset)
            return None

        # Prefer feed.fetch_rates(tf, bars) if exists
        if hasattr(self.feed, "fetch_rates"):
            try:
                df = self.feed.fetch_rates(tf, 240)
                if df is not None:
                    return df
            except Exception as exc:
                log_health.warning("%s: fetch_rates failed: %s", self.asset, exc)

        # Try common get_rates signatures
        if hasattr(self.feed, "get_rates"):
            try:
                df = self.feed.get_rates(tf, 240)
                if df is not None:
                    return df
            except TypeError:
                pass
            except Exception as exc:
                log_health.warning("%s: get_rates(tf,bars) failed: %s", self.asset, exc)

            try:
                df = self.feed.get_rates(self.symbol, tf, 240)
                if df is not None:
                    return df
            except TypeError:
                pass
            except Exception as exc:
                log_health.warning("%s: get_rates(symbol,tf,bars) failed: %s", self.asset, exc)

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
            log_err.error(
                "%s: mt5.copy_rates_from_pos fallback failed: %s | tb=%s",
                self.asset,
                exc,
                traceback.format_exc(),
            )
            return None

    def validate_market_data(self) -> bool:
        now = time.time()

        # Cache only SUCCESS for a short interval (prevents heavy fetch)
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

            # Age check
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
                        if last_ts > 1e12:
                            last_ts /= 1000.0
                        age = float(now - last_ts)
                except Exception:
                    age = None

            self.last_bar_age_sec = float(age or 0.0)
            if age is not None and age > max_age:
                self.last_market_ok = False
                self.last_market_reason = f"stale:{age:.1f}>{max_age:.1f}"
                return False

            # Basic sanity on close
            if hasattr(df, "columns") and "close" in df.columns:
                try:
                    s = df["close"]
                    if getattr(s, "isna", lambda: False)().any():
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
                    self.last_market_ts = datetime.utcfromtimestamp(float(last_ts)).strftime("%Y-%m-%d %H:%M:%S")
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
            if bool(getattr(self.cfg, "ignore_external_positions", False)):
                try:
                    magic = int(getattr(self.cfg, "magic", 777001) or 777001)
                except Exception:
                    magic = 777001
                pos = [p for p in pos if int(getattr(p, "magic", 0) or 0) == magic]
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
            if bool(getattr(self.cfg, "ignore_external_positions", False)):
                try:
                    magic = int(getattr(self.cfg, "magic", 777001) or 777001)
                except Exception:
                    magic = 777001
                pos = [p for p in pos if int(getattr(p, "magic", 0) or 0) == magic]
            return int(len(pos))
        except Exception:
            return 0

    def compute_candidate(self) -> Optional[AssetCandidate]:
        try:
            # execute=True => get LOT/SL/TP plan from risk manager
            res = self.signal.compute(execute=True)

            signal = str(getattr(res, "signal", "Neutral") or "Neutral")
            self.last_signal = signal
            self.last_signal_ts = time.time()
            self.last_latency_ms = float(getattr(res, "latency_ms", 0.0) or 0.0)

            conf_raw = float(getattr(res, "confidence", 0.0) or 0.0)
            conf = conf_raw / 100.0 if conf_raw > 1 else conf_raw
            conf = max(0.0, min(1.0, float(conf)))

            lot = float(getattr(res, "lot", 0.0) or 0.0)
            sl = float(getattr(res, "sl", 0.0) or 0.0)
            tp = float(getattr(res, "tp", 0.0) or 0.0)

            blocked = bool(getattr(res, "trade_blocked", False))
            reasons = tuple(str(r) for r in (getattr(res, "reasons", []) or [])[:10])
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
# Execution Worker
# =============================================================================


class ExecutionWorker(threading.Thread):
    def __init__(
        self,
        order_queue: "queue.Queue[OrderIntent]",
        result_queue: "queue.Queue[ExecutionResult]",
        dry_run: bool,
        order_notify_cb: Optional[Callable[[OrderIntent, ExecutionResult], None]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.order_queue = order_queue
        self.result_queue = result_queue
        self.dry_run = bool(dry_run)
        self.order_notify_cb = order_notify_cb
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
        fn = getattr(risk, name, None) if risk else None
        return fn if callable(fn) else None

    @staticmethod
    def _build_executor(risk: Any) -> OrderExecutor:
        normalize_volume_fn = None
        sanitize_stops_fn = None
        normalize_price_fn = None

        # Best-effort adapters (won't raise if missing)
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

            for hook_name in ("record_execution_metrics", "record_trade"):
                fn = self._get_hook(risk, hook_name)
                if not fn:
                    continue
                try:
                    if hook_name == "record_execution_metrics":
                        fn(intent.order_id, intent.signal, intent.enqueue_time, sent_ts, fill_ts, req_price, exec_price, slippage)
                    else:
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

        if (not self.dry_run) and bool(r.ok) and self.order_notify_cb:
            try:
                self.order_notify_cb(
                    intent,
                    ExecutionResult(
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
                    ),
                )
            except Exception:
                pass

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
        self._manual_stop = False

        # Portfolio state
        self._active_asset = "NONE"
        self._last_selected_asset = "NONE"
        self._order_counter = 0

        # Idempotency
        self._seen: Deque[Tuple[str, str, float]] = deque(maxlen=8000)  # (asset, signal_id, ts)
        self._seen_index: Dict[Tuple[str, str], Tuple[float, int]] = {}
        self._signal_cooldown_override_sec: Optional[float] = None
        # Per-asset idempotency cooldown (defaults; refreshed after pipelines build)
        self._signal_cooldown_sec_by_asset: Dict[str, float] = {"XAU": 60.0, "BTC": 60.0}
        # Queues_ signal_cooldown_sec
        self._max_queue = 64
        self._order_q: "queue.Queue[OrderIntent]" = queue.Queue(maxsize=self._max_queue)
        self._result_q: "queue.Queue[ExecutionResult]" = queue.Queue(maxsize=max(60, self._max_queue * 6))
        self._exec_worker: Optional[ExecutionWorker] = None

        # Loop timings
        self._poll_fast = 0.05
        self._poll_slow = 0.20
        self._heartbeat_every = 5.0
        self._last_heartbeat_ts = 0.0
        self._exec_health_last_ts = 0.0

        self._pipeline_log_every = 2.0
        self._last_pipeline_log_ts = 0.0
        self._last_pipeline_log_key: Tuple[Any, ...] = (None, None, None, None)
        self._last_reconcile_ts = 0.0

        # Recovery
        self._max_consecutive_errors = 12   
        self._recover_backoff_s = 10.0
        self._last_recover_ts = 0.0

        # Diagnostics
        self._diag_last_ts = 0.0

        # Pipelines
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
        self._order_notifier: Optional[Callable[[OrderIntent, ExecutionResult], None]] = None
        self._phase_notifier: Optional[Callable[[str, str, str, str], None]] = None
        self._engine_stop_notifier: Optional[Callable[[str, str], None]] = None
        self._last_phase_by_asset: Dict[str, str] = {"XAU": "?", "BTC": "?"}

    def set_order_notifier(self, cb: Optional[Callable[[OrderIntent, ExecutionResult], None]]) -> None:
        self._order_notifier = cb

    def set_phase_notifier(self, cb: Optional[Callable[[str, str, str, str], None]]) -> None:
        self._phase_notifier = cb

    def set_engine_stop_notifier(self, cb: Optional[Callable[[str, str], None]]) -> None:
        self._engine_stop_notifier = cb

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

        self._last_phase_by_asset["XAU"] = str(getattr(self._xau.risk, "current_phase", "A") or "A")
        self._last_phase_by_asset["BTC"] = str(getattr(self._btc.risk, "current_phase", "A") or "A")

        self._refresh_signal_cooldowns()
        try:
            self._poll_fast = min(
                float(getattr(self._xau_cfg, "poll_seconds_fast", self._poll_fast) or self._poll_fast),
                float(getattr(self._btc_cfg, "poll_seconds_fast", self._poll_fast) or self._poll_fast),
            )
            self._poll_slow = max(self._poll_fast * 4.0, 0.20)
        except Exception:
            pass

        log_health.info("PIPELINES_BUILT | xau=%s btc=%s", self._xau.symbol, self._btc.symbol)

    def _restart_exec_worker(self) -> None:
        try:
            if self._exec_worker:
                self._exec_worker.stop()
                self._exec_worker.join(timeout=3.0)
        except Exception:
            pass

        self._exec_worker = ExecutionWorker(
            self._order_q,
            self._result_q,
            dry_run=self._dry_run,
            order_notify_cb=self._order_notifier,
        )
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

    def _refresh_signal_cooldowns(self) -> None:
        """Set per-asset idempotency cooldown.

        Goal: prevent repeated orders within the same bar (especially M1).
        - If PORTFOLIO_SIGNAL_COOLDOWN_SEC is set (>0), it overrides everything.
        - Otherwise cooldown defaults to TF seconds of each asset primary timeframe.
        """
        try:
            if self._signal_cooldown_override_sec is not None and self._signal_cooldown_override_sec > 0:
                cd = float(self._signal_cooldown_override_sec)
                self._signal_cooldown_sec_by_asset["XAU"] = cd
                self._signal_cooldown_sec_by_asset["BTC"] = cd
                return

            def _pipe_tf_sec(pipe) -> float:
                try:
                    if pipe is None:
                        return 60.0
                    tf = getattr(getattr(pipe.cfg, "symbol_params", None), "tf_primary", None)
                    sec = _tf_seconds(tf)
                    return float(sec) if sec else 60.0
                except Exception:
                    return 60.0

            # Block duplicates for at least the full bar duration
            self._signal_cooldown_sec_by_asset["XAU"] = max(_pipe_tf_sec(self._xau), 10.0)
            self._signal_cooldown_sec_by_asset["BTC"] = max(_pipe_tf_sec(self._btc), 10.0)
        except Exception:
            # Fail-safe
            self._signal_cooldown_sec_by_asset["XAU"] = 60.0
            self._signal_cooldown_sec_by_asset["BTC"] = 60.0

    def _phase_reason(self, risk: Any, new_phase: str) -> str:
        if new_phase == "C" and risk is not None:
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
            current = str(getattr(risk, "current_phase", "A") or "A")
            prev = str(self._last_phase_by_asset.get(asset, "A") or "A")
            if current != prev:
                self._last_phase_by_asset[asset] = current
                if self._phase_notifier:
                    reason = self._phase_reason(risk, current)
                    self._phase_notifier(str(asset), prev, current, reason)
        except Exception:
            return

    # -------------------- idempotency --------------------
    def _cooldown_for_asset(self, asset: str) -> float:
        return float(self._signal_cooldown_sec_by_asset.get(asset, 60.0))

    def _is_duplicate(self, asset: str, signal_id: str, now: float, max_orders: int) -> bool:
        last = self._seen_index.get((asset, signal_id))
        if not last:
            return False
        last_ts, count = last
        cooldown = self._cooldown_for_asset(asset)
        if (now - float(last_ts)) >= cooldown:
            return False
        return int(count) >= int(max_orders)

    def _mark_seen(self, asset: str, signal_id: str, now: float) -> None:
        key = (asset, signal_id)
        cooldown = self._cooldown_for_asset(asset)
        last = self._seen_index.get(key)
        count = 0
        if last:
            last_ts, last_count = last
            if (now - float(last_ts)) < cooldown:
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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"PORD_{asset}_{ts}_{self._order_counter}_{os.getpid()}"

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
        return float(c.confidence)

    def _select_active_asset(self, open_xau: int, open_btc: int) -> str:
        # Full Concurrent Mode: Always allow both unless hard stop
        if open_xau > 0 and open_btc > 0:
            return "BOTH"  # Now considered SAFE and allowed
        if open_xau > 0:
            return "XAU"
        if open_btc > 0:
            return "BTC"
        return "NONE"

    def _enqueue_order(
        self,
        cand: AssetCandidate,
        *,
        order_index: int = 0,
        order_count: int = 1,
        lot_override: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        now = time.time()

        if self._is_duplicate(cand.asset, cand.signal_id, now, max_orders=int(order_count)):
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

        lot_val = float(lot_override) if lot_override is not None else float(cand.lot)
        if lot_val <= 0:
            log_health.info("ENQUEUE_SKIP | asset=%s reason=lot_nonpositive", cand.asset)
            return False, None
        intent = OrderIntent(
            asset=cand.asset,
            symbol=cand.symbol,
            signal=cand.signal,
            confidence=float(cand.confidence),
            lot=float(lot_val),
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
            if risk and hasattr(risk, "record_execution_failure"):
                try:
                    risk.record_execution_failure(order_id, now, now, "queue_backlog_drop")
                except Exception:
                    pass
            log_health.warning("ENQUEUE_FAIL | asset=%s reason=queue_full order_id=%s", cand.asset, order_id)
            return False, None

        self._mark_seen(cand.asset, cand.signal_id, now)
        self._last_selected_asset = cand.asset

        if risk and hasattr(risk, "track_signal_survival"):
            try:
                risk.track_signal_survival(order_id, cand.signal, price, cand.sl, cand.tp, now, cand.confidence)
            except Exception:
                pass

        return True, order_id

    def _orders_for_candidate(self, cand: AssetCandidate) -> int:
        cfg = self._xau_cfg if cand.asset == "XAU" else self._btc_cfg

        max_orders = int(getattr(cfg, "multi_order_max_orders", 3) or 3)
        max_orders = max(1, min(6, max_orders))

        conf = float(cand.confidence)
        if conf > 1.5:
            conf = conf / 100.0

        # Read context from SignalResult if present
        regime = ""
        spread_bps = 0.0
        try:
            rr = cand.raw_result
            regime = str(getattr(rr, "regime", "") or "")
            spread_bps = float(getattr(rr, "spread_bps", 0.0) or 0.0)
        except Exception:
            pass

        # Hard rule: if spread high => never scale-in
        max_spread_bps_for_multi = float(getattr(cfg, "max_spread_bps_for_multi", 6.0) or 6.0)
        if spread_bps > max_spread_bps_for_multi:
            return 1

        tiers = list(getattr(cfg, "multi_order_confidence_tiers", (0.965, 0.985, 0.993)) or (0.965, 0.985, 0.993))
        tiers = sorted([float(x) for x in tiers if x is not None])

        # Range regime is noisier: demand higher confidence to scale
        if regime.lower().startswith("range"):
            tiers = [min(0.999, t + 0.004) for t in tiers]

        n = 1
        if conf >= tiers[0]:
            n = 2
        if conf >= tiers[1]:
            n = 3
        if len(tiers) >= 3 and conf >= tiers[2]:
            n = 3

        return int(min(max_orders, max(1, n)))

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
        now = time.time()
        if (now - self._last_heartbeat_ts) < self._heartbeat_every:
            return
        self._last_heartbeat_ts = now

        bal = eq = pnl = dd = 0.0
        connected = False
        term_trade = False
        acc_trade = False
        try:
            with MT5_LOCK:
                term = mt5.terminal_info()
                acc = mt5.account_info()
            connected = bool(term and getattr(term, "connected", False))
            term_trade = bool(term and getattr(term, "trade_allowed", False))
            if acc:
                acc_trade = bool(getattr(acc, "trade_allowed", True))
                bal = float(getattr(acc, "balance", 0.0) or 0.0)
                eq = float(getattr(acc, "equity", 0.0) or 0.0)
                pnl = float(getattr(acc, "profit", 0.0) or 0.0)
            if bal > 0:
                dd = max(0.0, (bal - eq) / bal)
        except Exception:
            pass

        log_health.info(
            "HEARTBEAT | running=%s mt5_ready=%s connected=%s term_trade=%s acc_trade=%s active=%s last_sel=%s open_xau=%s open_btc=%s total_open=%s bal=%.2f eq=%.2f dd=%.4f pnl=%.2f sig_xau=%s sig_btc=%s q=%s manual_stop=%s",
            self._run.is_set(),
            self._mt5_ready,
            connected,
            term_trade,
            acc_trade,
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

        self._maybe_log_diag()
        self._log_exec_health_snapshot(open_xau, open_btc)

    def _log_exec_health_snapshot(self, open_xau: int, open_btc: int) -> None:
        now = time.time()
        if (now - self._exec_health_last_ts) < self._heartbeat_every:
            return
        self._exec_health_last_ts = now

        def _symbol_snapshot(symbol: str) -> dict:
            if not symbol:
                return {"visible": None, "trade_mode": -1, "spread_points": 0.0}
            with MT5_LOCK:
                info = mt5.symbol_info(symbol)
                tick = mt5.symbol_info_tick(symbol)
            visible = bool(getattr(info, "visible", True)) if info else None
            trade_mode = int(getattr(info, "trade_mode", -1) or -1) if info else -1
            spread_points = 0.0
            if info and tick:
                point = float(getattr(info, "point", 0.0) or 0.0)
                bid = float(getattr(tick, "bid", 0.0) or 0.0)
                ask = float(getattr(tick, "ask", 0.0) or 0.0)
                if point > 0 and ask > 0 and bid > 0:
                    spread_points = float((ask - bid) / point)
            return {
                "visible": visible,
                "trade_mode": trade_mode,
                "spread_points": spread_points,
            }

        def _asset_status(pipe: Optional[_AssetPipeline], cand: Optional[AssetCandidate]) -> dict:
            if pipe is None:
                return {"present": False}
            risk = getattr(pipe, "risk", None)
            market_open = True
            try:
                if risk and hasattr(risk, "market_open_24_5"):
                    market_open = bool(risk.market_open_24_5())
            except Exception:
                market_open = True

            phase = "-"
            can_analyze = True
            try:
                if risk and hasattr(risk, "current_phase"):
                    phase = str(getattr(risk, "current_phase", "-"))
                if risk and hasattr(risk, "can_analyze"):
                    can_analyze = bool(risk.can_analyze())
            except Exception:
                pass

            conf = float(getattr(cand, "confidence", 0.0) or 0.0) if cand else 0.0
            blocked = bool(getattr(cand, "blocked", False)) if cand else False
            reasons = ",".join(getattr(cand, "reasons", []) or []) if cand else "-"

            return {
                "present": True,
                "market_ok": bool(getattr(pipe, "last_market_ok", True)),
                "market_reason": str(getattr(pipe, "last_market_reason", "-")),
                "market_open": bool(market_open),
                "signal": str(getattr(pipe, "last_signal", "Neutral")),
                "conf": float(conf),
                "blocked": bool(blocked),
                "reasons": reasons if reasons else "-",
                "phase": str(phase),
                "can_analyze": bool(can_analyze),
                "bar_age": float(getattr(pipe, "last_bar_age_sec", 0.0) or 0.0),
                "bar_tf": str(getattr(pipe, "last_market_tf", "-")),
                "bar_rows": int(getattr(pipe, "last_market_rows", 0) or 0),
                "bar_ts": str(getattr(pipe, "last_market_ts", "-")),
            }

        x = _asset_status(self._xau, self._last_cand_xau)
        b = _asset_status(self._btc, self._last_cand_btc)
        x_sym = _symbol_snapshot(self._xau.symbol if self._xau else "")
        b_sym = _symbol_snapshot(self._btc.symbol if self._btc else "")

        exec_health_log.info(
            "SYSTEM_STATUS | mt5_ready=%s active=%s manual_stop=%s last_sel=%s open_xau=%s open_btc=%s q=%s "
            "xau_ok=%s xau_reason=%s xau_open=%s xau_sig=%s xau_conf=%.3f xau_blocked=%s xau_phase=%s xau_analyze=%s xau_bar_age=%.1f "
            "btc_ok=%s btc_reason=%s btc_open=%s btc_sig=%s btc_conf=%.3f btc_blocked=%s btc_phase=%s btc_analyze=%s btc_bar_age=%.1f "
            "xau_visible=%s xau_trade_mode=%s xau_spread_pts=%.1f "
            "btc_visible=%s btc_trade_mode=%s btc_spread_pts=%.1f",
            self._mt5_ready,
            self._active_asset,
            self._manual_stop,
            self._last_selected_asset,
            open_xau,
            open_btc,
            self._order_q.qsize(),
            x.get("market_ok"),
            x.get("market_reason"),
            x.get("market_open"),
            x.get("signal"),
            float(x.get("conf", 0.0)),
            x.get("blocked"),
            x.get("phase"),
            x.get("can_analyze"),
            float(x.get("bar_age", 0.0)),
            b.get("market_ok"),
            b.get("market_reason"),
            b.get("market_open"),
            b.get("signal"),
            float(b.get("conf", 0.0)),
            b.get("blocked"),
            b.get("phase"),
            b.get("can_analyze"),
            float(b.get("bar_age", 0.0)),
            x_sym.get("visible"),
            x_sym.get("trade_mode"),
            float(x_sym.get("spread_points", 0.0)),
            b_sym.get("visible"),
            b_sym.get("trade_mode"),
            float(b_sym.get("spread_points", 0.0)),
        )

    # ---------------------------------------------------------------------
    # Diagnostics (NO TELEGRAM): deep snapshot to Logs/portfolio_engine_diag.jsonl
    # ---------------------------------------------------------------------
    def _maybe_log_diag(self) -> None:
        if not _DIAG_ENABLED:
            return
        now = time.time()
        if (now - self._diag_last_ts) < float(_DIAG_EVERY_SEC):
            return
        self._diag_last_ts = now

        try:
            snap = self._diag_snapshot()
            mt5s = snap.get("mt5", {}) if isinstance(snap, dict) else {}
            log_health.info(
                "DIAG_SUMMARY | mt5_ok=%s reason=%s connected=%s term_trade=%s acc_trade=%s last_error=%s",
                mt5s.get("ok"),
                mt5s.get("reason"),
                mt5s.get("terminal_connected"),
                mt5s.get("terminal_trade_allowed"),
                mt5s.get("account_trade_allowed"),
                mt5s.get("last_error"),
            )

            log_diag.info(json.dumps(snap, ensure_ascii=False, separators=(",", ":"), default=_json_default))
        except Exception:
            log_err.error("DIAG logging failed | tb=%s", traceback.format_exc())

    def _diag_snapshot(self) -> dict:
        try:
            st = self.status()
        except Exception:
            st = {"status_error": traceback.format_exc()}

        return {
            "ts_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "engine": st,
            "mt5": self._diag_mt5(),
            "assets": {
                "XAU": self._diag_asset(self._xau, self._last_cand_xau),
                "BTC": self._diag_asset(self._btc, self._last_cand_btc),
            },
        }

    def _diag_mt5(self) -> dict:
        ok, reason = False, "unknown"
        try:
            ok, reason = mt5_status()
        except Exception:
            reason = "mt5_status_exception"

        d: dict = {"ok": bool(ok), "reason": str(reason)}

        # terminal64.exe running?
        try:
            if os.name == "nt":
                out = subprocess.check_output(["tasklist"], text=True, errors="ignore").lower()
                d["terminal64_running"] = ("terminal64.exe" in out)
        except Exception:
            d["terminal64_running"] = None

        with MT5_LOCK:
            try:
                d["last_error"] = str(mt5.last_error())
            except Exception:
                d["last_error"] = "unknown"

            try:
                ti = mt5.terminal_info()
                if ti:
                    d["terminal_connected"] = bool(getattr(ti, "connected", False))
                    d["terminal_trade_allowed"] = bool(getattr(ti, "trade_allowed", False))
                    d["terminal_path"] = str(getattr(ti, "path", "") or "")
                    d["terminal_name"] = str(getattr(ti, "name", "") or "")
                    d["terminal_company"] = str(getattr(ti, "company", "") or "")
                    d["terminal_build"] = int(getattr(ti, "build", 0) or 0)
                else:
                    d["terminal_info_none"] = True
            except Exception:
                d["terminal_info_exc"] = traceback.format_exc()

            try:
                acc = mt5.account_info()
                if acc:
                    d["account_login"] = int(getattr(acc, "login", 0) or 0)
                    d["account_server"] = str(getattr(acc, "server", "") or "")
                    d["account_currency"] = str(getattr(acc, "currency", "") or "")
                    d["account_trade_allowed"] = bool(getattr(acc, "trade_allowed", False))
                    d["balance"] = float(getattr(acc, "balance", 0.0) or 0.0)
                    d["equity"] = float(getattr(acc, "equity", 0.0) or 0.0)
                    d["margin"] = float(getattr(acc, "margin", 0.0) or 0.0)
                    d["margin_free"] = float(getattr(acc, "margin_free", 0.0) or 0.0)
                    d["leverage"] = int(getattr(acc, "leverage", 0) or 0)
                else:
                    d["account_info_none"] = True
            except Exception:
                d["account_info_exc"] = traceback.format_exc()

        return d

    def _diag_asset(self, pipe: Optional[_AssetPipeline], cand: Optional[AssetCandidate]) -> dict:
        if pipe is None:
            return {"present": False}

        d: dict = {
            "present": True,
            "asset": pipe.asset,
            "symbol": pipe.symbol,
            "market": {
                "ok": bool(pipe.last_market_ok),
                "reason": str(pipe.last_market_reason),
                "bar_age_sec": float(pipe.last_bar_age_sec),
                "bars": int(pipe.last_market_rows),
                "close": float(pipe.last_market_close),
                "tick_volume": float(pipe.last_market_volume),
                "tf": str(pipe.last_market_tf),
                "ts": str(pipe.last_market_ts),
            },
            "signal": {
                "last": str(pipe.last_signal),
                "last_ts": float(pipe.last_signal_ts),
                "latency_ms": float(pipe.last_latency_ms),
            },
        }

        if cand is not None:
            d["candidate"] = {
                "signal": cand.signal,
                "confidence": float(cand.confidence),
                "lot": float(cand.lot),
                "sl": float(cand.sl),
                "tp": float(cand.tp),
                "blocked": bool(cand.blocked),
                "reasons": list(cand.reasons),
                "latency_ms": float(cand.latency_ms),
                "signal_id": str(cand.signal_id),
            }

        # Risk metrics snapshot if available
        try:
            rm = pipe.risk
            fn = getattr(rm, "metrics_snapshot", None)
            if callable(fn):
                d["risk"] = fn()  # type: ignore[misc]
        except Exception:
            d["risk_exc"] = traceback.format_exc()

        # Symbol/tick diagnostics
        if pipe.symbol:
            d["mt5_symbol"] = self._diag_symbol(pipe.symbol)

        return d

    def _diag_symbol(self, symbol: str) -> dict:
        d: dict = {"symbol": symbol}
        with MT5_LOCK:
            try:
                info = mt5.symbol_info(symbol)
                if info:
                    d["visible"] = bool(getattr(info, "visible", True))
                    d["trade_mode"] = int(getattr(info, "trade_mode", -1) or -1)
                    d["digits"] = int(getattr(info, "digits", 0) or 0)
                    d["point"] = float(getattr(info, "point", 0.0) or 0.0)
                    d["stops_level"] = int(getattr(info, "stops_level", 0) or 0)
                else:
                    d["symbol_info_none"] = True
            except Exception:
                d["symbol_info_exc"] = traceback.format_exc()

            try:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    bid = float(getattr(tick, "bid", 0.0) or 0.0)
                    ask = float(getattr(tick, "ask", 0.0) or 0.0)
                    d["bid"] = bid
                    d["ask"] = ask
                    d["time_msc"] = int(getattr(tick, "time_msc", 0) or 0)

                    point = float(d.get("point") or 0.0) or 0.0
                    if bid and ask and point:
                        d["spread_points"] = (ask - bid) / point

                    now_msc = int(time.time() * 1000)
                    if d.get("time_msc"):
                        d["tick_age_sec"] = max(0.0, (now_msc - int(d["time_msc"])) / 1000.0)
                else:
                    d["tick_none"] = True
            except Exception:
                d["tick_exc"] = traceback.format_exc()

        return d

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

        # Drain queue best-effort
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

                # Hard stop (stop engine for the day)
                try:
                    if self._xau.risk and hasattr(self._xau.risk, "requires_hard_stop") and self._xau.risk.requires_hard_stop():
                        log_health.info("HARD_STOP_TRIGGERED | asset=XAU (stop_for_day)")
                        if self._engine_stop_notifier:
                            reason = self._phase_reason(self._xau.risk, "C")
                            self._engine_stop_notifier("XAU", reason)
                        with self._lock:
                            self._manual_stop = True
                        self._run.clear()
                        break
                    if self._btc.risk and hasattr(self._btc.risk, "requires_hard_stop") and self._btc.risk.requires_hard_stop():
                        log_health.info("HARD_STOP_TRIGGERED | asset=BTC (stop_for_day)")
                        if self._engine_stop_notifier:
                            reason = self._phase_reason(self._btc.risk, "C")
                            self._engine_stop_notifier("BTC", reason)
                        with self._lock:
                            self._manual_stop = True
                        self._run.clear()
                        break
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
                        "PIPELINE_STAGE | step=market_data ok_xau=%s reason_xau=%s age_xau=%.1fs tf_xau=%s bars_xau=%s close_xau=%.3f vol_xau=%s ts_xau=%s "
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

                # Phase change notifications (A/B/C)
                self._check_phase_change("XAU", self._xau.risk if self._xau else None)
                self._check_phase_change("BTC", self._btc.risk if self._btc else None)

                # Drain execution results
                while True:
                    try:
                        r = self._result_q.get_nowait()
                    except queue.Empty:
                        break
                    try:
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

                # Positions and active asset (informational only now)
                open_xau = self._xau.open_positions()
                open_btc = self._btc.open_positions()
                self._active_asset = self._select_active_asset(open_xau, open_btc)

                # UNSAFE_STATE check removed to allow concurrent trading
                # self._active_asset is just for logging now

                self._heartbeat(open_xau, open_btc)

                # Compute candidates
                cand_x = self._xau.compute_candidate() if x_ok else None
                cand_b = self._btc.compute_candidate() if b_ok else None
                self._last_cand_xau = cand_x
                self._last_cand_btc = cand_b

                # Full Concurrent Selection: Check ALL candidates
                candidates: list[AssetCandidate] = []
                if cand_x:
                    candidates.append(cand_x)
                if cand_b:
                    candidates.append(cand_b)

                # Enqueue ALL valid candidates (don't pick just one)
                candidates = [c for c in candidates if self._candidate_is_tradeable(c)]

                # Execute all valid signals (Concurrent Execution - Zero Latency Batch)
                for selected in candidates:
                    order_count = self._orders_for_candidate(selected)
                    risk = self._xau.risk if selected.asset == "XAU" and self._xau else (
                        self._btc.risk if self._btc else None
                    )
                    cfg = self._xau_cfg if selected.asset == "XAU" else self._btc_cfg
                    lots = self._split_lot(float(selected.lot), order_count, risk, cfg)

                    for idx, lot_val in enumerate(lots):
                        # enqueue_order checks duplicate, then puts to Queue.
                        # The ExecutionWorker picks it up instantly.
                        ok, oid = self._enqueue_order(
                            selected,
                            order_index=int(idx),
                            order_count=int(order_count),
                            lot_override=float(lot_val),
                        )
                        if ok:
                            log_health.info(
                                "ORDER_SELECTED | asset=%s symbol=%s signal=%s conf=%.4f lot=%.4f sl=%s tp=%s order_id=%s batch=%s/%s reasons=%s",
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

                consecutive_errors = 0

                # Adaptive sleep
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
