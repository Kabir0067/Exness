"""
Bot/Motor/models.py - Core data structures and engine types.

Ин модул тамоми типизатсияи маълумотро (dataclass, enum) барои
мотори савдо дар бар мегирад. Ҳеҷ гуна логикаи мураккаб надорад,
танҳо таърифи структураҳо ва ёрирасонҳои хурд.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date as _date_type
from datetime import datetime, timezone
from datetime import datetime as _dt_type
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path as _Path_type
from typing import Any, Dict, Optional, Tuple

# DEFENSIVE CODE: Evaluate numpy existence safely.
try:
    import numpy as _np  # type: ignore

    _NUMPY_TYPES = (_np.integer, _np.floating, _np.bool_)
except ImportError:
    _NUMPY_TYPES = ()

# DEFENSIVE CODE: MT5 import wrapped.
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# External imports
from core.ml_router import MLSignal

try:
    from log_config import LOG_DIR as LOG_ROOT
    from log_config import get_log_path
except ImportError:
    LOG_ROOT = "/tmp/portfolio_logs"

    def get_log_path(filename: str) -> str:
        return os.path.join(LOG_ROOT, filename)


# =============================================================================
# Global Constants & Meta Definitions
# =============================================================================
LOG_DIR = LOG_ROOT

_DIAG_ENABLED: bool = True
_DIAG_EVERY_SEC: float = 60.0

_FMT = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

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

_TF_REGEX = re.compile(r"([MHDW])\s*(\d+)")

# =============================================================================
# Standard Setup / Global Loggers
# =============================================================================
log_health = logging.getLogger("portfolio.engine.health")
log_err = logging.getLogger("portfolio.engine.err")
log_diag = logging.getLogger("portfolio.engine.diag")
log_core_risk = logging.getLogger("core.risk_engine")
log_core_feature = logging.getLogger("core.data_engine")
log_core_signal = logging.getLogger("core.signal_engine")
log_core_risk_engine = logging.getLogger("core.risk_engine")

log_health.setLevel(logging.INFO)
log_err.setLevel(logging.ERROR)
log_diag.setLevel(logging.INFO)
log_core_risk.setLevel(logging.INFO)
log_core_feature.setLevel(logging.ERROR)
log_core_signal.setLevel(logging.ERROR)
log_core_risk_engine.setLevel(logging.ERROR)

log_health.propagate = False
log_err.propagate = False
log_diag.propagate = False
log_core_risk.propagate = False
log_core_feature.propagate = False
log_core_signal.propagate = False
log_core_risk_engine.propagate = False

if mt5 is None:
    log_err.error(
        "MetaTrader5 module not found. MT5-dependent functionalities will fail."
    )


def _add_handler(logger: logging.Logger, filename: str, level: int) -> None:
    if logger.handlers:
        return
    try:
        path = str(get_log_path(filename))
    except Exception:
        try:
            os.makedirs(str(LOG_DIR), exist_ok=True)
            path = os.path.join(str(LOG_DIR), filename)
        except OSError as e:
            print(
                f"CRITICAL: Log directory creation failed for {filename}: {e}",
                file=sys.stderr,
            )
            return

    try:
        h = RotatingFileHandler(
            path, maxBytes=2_000_000, backupCount=5, encoding="utf-8", delay=True
        )
        h.setLevel(level)
        h.setFormatter(_FMT)
        logger.addHandler(h)
    except OSError as e:
        print(
            f"CRITICAL: Failed to attach RotatingFileHandler to {path}: {e}",
            file=sys.stderr,
        )


_add_handler(log_health, "portfolio_engine_health.log", logging.INFO)
_add_handler(log_err, "portfolio_engine_error.log", logging.ERROR)
_add_handler(log_core_risk, "portfolio_engine_health.log", logging.INFO)
_add_handler(log_core_feature, "portfolio_engine_error.log", logging.ERROR)
_add_handler(log_core_signal, "portfolio_engine_error.log", logging.ERROR)
_add_handler(log_core_risk_engine, "portfolio_engine_error.log", logging.ERROR)

if _DIAG_ENABLED:
    _add_handler(log_diag, "portfolio_engine_diag.jsonl", logging.INFO)


# =============================================================================
# Custom Exceptions
# =============================================================================
# (Нигоҳ доштани сохтори стандартӣ, хатогиҳои махсус вуҷуд надоранд)


# =============================================================================
# Classes, Enums, Dataclasses
# =============================================================================
class EngineState(Enum):
    BOOT = "BOOT"
    DATA_SYNC = "DATA_SYNC"
    ML_INFERENCE = "ML_INFERENCE"
    RISK_CALC = "RISK_CALC"
    EXECUTION_QUEUE = "EXECUTION_QUEUE"
    VERIFICATION = "VERIFICATION"
    HALT = "HALT"


@dataclass(frozen=True)
class AssetCandidate:
    asset: str
    symbol: str
    signal: str
    confidence: float
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


@dataclass
class EngineCycleContext:
    payloads: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = field(
        default_factory=dict
    )
    ml_signals: Dict[str, MLSignal] = field(default_factory=dict)
    candidates: list[AssetCandidate] = field(default_factory=list)
    halt_reason: str = ""


@dataclass(frozen=True)
class StepDecision:
    next_state: EngineState
    ctx: EngineCycleContext
    skip_sleep: bool = False


# =============================================================================
# Helper Functions
# =============================================================================
def env_truthy(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def monitoring_only_mode() -> bool:
    return False


def required_gate_assets() -> tuple[str, ...]:
    raw = str(os.getenv("REQUIRED_GATE_ASSETS", "XAU,BTC") or "XAU,BTC")
    out: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        asset = item.strip().upper()
        if not asset or asset in seen:
            continue
        seen.add(asset)
        out.append(asset)
    return tuple(out) if out else ("XAU", "BTC")


def partial_gate_enabled(required_assets_: Optional[tuple[str, ...]] = None) -> bool:
    assets = tuple(
        required_assets_ if required_assets_ is not None else required_gate_assets()
    )
    if not env_truthy("PARTIAL_GATE_MODE", "0"):
        return False
    if len(assets) > 1 and env_truthy("STRICT_DUAL_ASSET_MODE", "0"):
        return False
    return True


_env_truthy = env_truthy
_monitoring_only_mode = monitoring_only_mode
_required_gate_assets = required_gate_assets
_partial_gate_enabled = partial_gate_enabled


def json_default(obj: object) -> Any:
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
        if isinstance(obj, (_dt_type, _date_type)):
            return obj.isoformat()
    except Exception:
        pass
    try:
        if isinstance(obj, _Path_type):
            return str(obj)
    except Exception:
        pass
    try:
        if _NUMPY_TYPES and isinstance(obj, _NUMPY_TYPES):
            return obj.item()
    except Exception:
        pass
    try:
        return str(obj)
    except Exception:
        return "<unserializable_object>"


def safe_json_dumps(payload: Any) -> str:
    try:
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), default=json_default
        )
    except Exception as primary_exc:
        try:
            return json.dumps(
                {
                    "error": "json.dumps_failed",
                    "reason": str(primary_exc),
                    "repr": repr(payload),
                },
                ensure_ascii=False,
            )
        except Exception:
            return (
                '{"error":"json.dumps_failed", "reason":"critical_serialization_fault"}'
            )


def tf_seconds(tf: Any) -> Optional[float]:
    if tf is None:
        return None
    if isinstance(tf, int):
        sec = _TF_SECONDS_MAP.get(tf)
        return float(sec) if sec else None
    if isinstance(tf, str):
        s = tf.strip().upper()
        if s.startswith("TIMEFRAME_"):
            s = s.replace("TIMEFRAME_", "")
        m = _TF_REGEX.fullmatch(s)
        if m:
            unit, n_str = m.group(1), m.group(2)
            try:
                n = int(n_str)
                if unit == "M":
                    return float(n * 60)
                if unit == "H":
                    return float(n * 3600)
                if unit == "D":
                    return float(n * 86400)
                if unit == "W":
                    return float(n * 604800)
            except ValueError:
                pass
        if s in {"MN1", "MO1", "MON1", "MONTH1"}:
            return 2592000.0
    return None


def parse_bar_key(bar_key: str) -> Optional[datetime]:
    if not bar_key:
        return None
    s = str(bar_key).strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        pass
    if s.isdigit():
        try:
            n = int(s)
            if n >= 10_000_000_000:
                return datetime.fromtimestamp(n / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(n, tz=timezone.utc)
        except Exception:
            pass
    return None


def best_effort_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        try:
            tb = traceback.format_exc()
            log_err.error("best_effort_call error: %s | tb=%s", exc, tb)
        except Exception:
            log_err.error("best_effort_call error: %s | (Traceback format failed)", exc)
        return None


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "AssetCandidate",
    "EngineCycleContext",
    "EngineState",
    "ExecutionResult",
    "LOG_DIR",
    "LOG_ROOT",
    "OrderIntent",
    "PortfolioStatus",
    "StepDecision",
    "_DIAG_ENABLED",
    "_DIAG_EVERY_SEC",
    "_FMT",
    "_TF_REGEX",
    "_TF_SECONDS_MAP",
    "best_effort_call",
    "env_truthy",
    "get_log_path",
    "json_default",
    "log_core_feature",
    "log_core_risk",
    "log_core_risk_engine",
    "log_core_signal",
    "log_diag",
    "log_err",
    "log_health",
    "monitoring_only_mode",
    "parse_bar_key",
    "partial_gate_enabled",
    "required_gate_assets",
    "safe_json_dumps",
    "tf_seconds",
]
