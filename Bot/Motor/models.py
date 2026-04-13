from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from log_config import LOG_DIR as LOG_ROOT, get_log_path

LOG_DIR = LOG_ROOT

# Diagnostics toggles
_DIAG_ENABLED : bool = True
_DIAG_EVERY_SEC : float = 60.0

# Loggers
log_health = logging.getLogger("portfolio.engine.health")
log_err = logging.getLogger("portfolio.engine.err")
log_diag = logging.getLogger("portfolio.engine.diag")

log_health.setLevel(logging.INFO)
log_err.setLevel(logging.ERROR)
log_diag.setLevel(logging.INFO)

log_health.propagate = False
log_err.propagate = False
log_diag.propagate = False

# Route core risk logs to health file
log_core_risk = logging.getLogger("core.risk_engine")
log_core_risk.setLevel(logging.INFO)
log_core_risk.propagate = False
log_core_feature = logging.getLogger("core.data_engine")
log_core_feature.setLevel(logging.ERROR)
log_core_feature.propagate = False
log_core_signal = logging.getLogger("core.signal_engine")
log_core_signal.setLevel(logging.ERROR)
log_core_signal.propagate = False
log_core_risk_engine = logging.getLogger("core.risk_engine")
log_core_risk_engine.setLevel(logging.ERROR)
log_core_risk_engine.propagate = False

_FMT = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _add_handler(logger: logging.Logger, filename: str, level: int) -> None:
    if logger.handlers:
        return
    try:
        path = str(get_log_path(filename))
    except Exception:
        # fallback
        os.makedirs(str(LOG_DIR), exist_ok=True)
        path = os.path.join(str(LOG_DIR), filename)

    h = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=5, encoding="utf-8", delay=True)
    h.setLevel(level)
    h.setFormatter(_FMT)
    logger.addHandler(h)


_add_handler(log_health, "portfolio_engine_health.log", logging.INFO)
_add_handler(log_err, "portfolio_engine_error.log", logging.ERROR)
_add_handler(log_core_risk, "portfolio_engine_health.log", logging.INFO)
_add_handler(log_core_feature, "portfolio_engine_error.log", logging.ERROR)
_add_handler(log_core_signal, "portfolio_engine_error.log", logging.ERROR)
_add_handler(log_core_risk_engine, "portfolio_engine_error.log", logging.ERROR)
if _DIAG_ENABLED:
    _add_handler(log_diag, "portfolio_engine_diag.jsonl", logging.INFO)


# --- Utilities -------------------------------------------------------------
import json
import re
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import MetaTrader5 as mt5


def env_truthy(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def monitoring_only_mode() -> bool:
    return env_truthy("MONITORING_ONLY", "0")


def required_gate_assets() -> tuple[str, ...]:
    raw = str(os.getenv("REQUIRED_GATE_ASSETS", "XAU,BTC") or "XAU,BTC")
    out: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        asset = str(item or "").upper().strip()
        if not asset or asset in seen:
            continue
        seen.add(asset)
        out.append(asset)
    return tuple(out or ["XAU", "BTC"])


def partial_gate_enabled(required_assets_: Optional[tuple[str, ...]] = None) -> bool:
    assets = tuple(required_assets_ or required_gate_assets())
    if not env_truthy("PARTIAL_GATE_MODE", "0"):
        return False
    if len(assets) > 1 and env_truthy("STRICT_DUAL_ASSET_MODE", "0"):
        return False
    return True


# Backward-compatible private aliases used by extracted engine mixins.
_env_truthy = env_truthy
_monitoring_only_mode = monitoring_only_mode
_required_gate_assets = required_gate_assets
_partial_gate_enabled = partial_gate_enabled



def json_default(obj: object):
    """JSON encoder for diagnostics (dataclasses + common non-JSON types)."""
    try:
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass

    try:
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="ignore")
    except Exception:
        pass

    try:
        from datetime import datetime as _dt, date as _date

        if isinstance(obj, (_dt, _date)):
            return obj.isoformat()
    except Exception:
        pass

    try:
        from pathlib import Path

        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass

    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except Exception:
        pass

    return str(obj)


def safe_json_dumps(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=json_default)
    except Exception:
        try:
            return json.dumps({"error": "json.dumps_failed", "repr": repr(payload)}, ensure_ascii=False)
        except Exception:
            return "{\"error\":\"json.dumps_failed\"}"


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


def tf_seconds(tf: Any) -> Optional[float]:
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


def parse_bar_key(bar_key: str) -> Optional[datetime]:
    """Parse SignalResult.bar_key into datetime.
    Accepts ISO (with Z), epoch seconds, or epoch milliseconds (MT5 time).
    Returns None on failure. Used by baseline gating for gap/warmup protection.
    """
    if not bar_key:
        return None
    s = str(bar_key).strip()

    # ISO (including Z)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        pass

    # Epoch seconds / milliseconds
    if s.isdigit():
        n = int(s)
        if n >= 10_000_000_000:
            return datetime.fromtimestamp(n / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(n, tz=timezone.utc)

    return None


def best_effort_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log_err.error("best_effort_call error: %s | tb=%s", exc, traceback.format_exc())
        return None


# --- Runtime dataclasses ---------------------------------------------------
from dataclasses import dataclass, field
from typing import Any, Tuple


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
    controller_state: str = "stopped"
    risk_halt_reason: str = ""
    gate_reason: str = ""
    chaos_state: str = "unknown"
    blocked_assets: Tuple[str, ...] = field(default_factory=tuple)


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
    base_lot: float = 0.0
    phase_snapshot: str = ""


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
    order_ticket: int = 0
    deal_ticket: int = 0
    position_ticket: int = 0


# --- FSM types -------------------------------------------------------------
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from core.signal_engine import MLSignal



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
    candidates: list[AssetCandidate] = field(default_factory=list)
    halt_reason: str = ""


@dataclass(frozen=True)
class StepDecision:
    next_state: EngineState
    ctx: EngineCycleContext
    skip_sleep: bool = False

