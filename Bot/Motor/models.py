"""
Core data structures and shared engine model types.

Defines dataclasses, enums, logger setup, and serialization helpers
used across the trading motor.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import re
import sys
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date as _date_type
from datetime import datetime as _dt_type
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path as _Path_type
from typing import Any, Dict, Optional, Tuple

# Resolve optional numpy types once at import time.
try:
    import numpy as _np  # type: ignore

    _NUMPY_TYPES = (_np.integer, _np.floating, _np.bool_)
except ImportError:
    _NUMPY_TYPES = ()

# Keep MT5 optional so shared types can import without terminal access.
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Local imports
from core.ml_router import MLSignal

try:
    from log_config import LOG_DIR as LOG_ROOT
    from log_config import get_artifact_path
    from log_config import get_log_path
except ImportError:
    LOG_ROOT = "/tmp/portfolio_logs"

    def get_log_path(filename: str) -> str:
        return os.path.join(LOG_ROOT, filename)

    def get_artifact_path(*parts: str) -> _Path_type:
        return _Path_type(LOG_ROOT).joinpath("Artifacts", *parts)


# =============================================================================
# Global Constants & Meta Definitions
# =============================================================================
LOG_DIR = LOG_ROOT

_DIAG_ENABLED: bool = True
_DIAG_EVERY_SEC: float = 60.0

_FMT = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

MODEL_STATE_PATH = get_artifact_path("models", "model_state.pkl")
RECOVERABLE_HALT_REASONS = frozenset({"mt5_unhealthy", "mt5_disconnected"})
MONITORING_ONLY_HALT_REASONS = frozenset(
    {
        "manual_stop",
        "manual_stop_active",
        "manual_stop_triggered",
    }
)

_MT5_TIMEFRAME_SECONDS = (
    ("TIMEFRAME_M1", 60),
    ("TIMEFRAME_M2", 120),
    ("TIMEFRAME_M3", 180),
    ("TIMEFRAME_M4", 240),
    ("TIMEFRAME_M5", 300),
    ("TIMEFRAME_M6", 360),
    ("TIMEFRAME_M10", 600),
    ("TIMEFRAME_M12", 720),
    ("TIMEFRAME_M15", 900),
    ("TIMEFRAME_M20", 1200),
    ("TIMEFRAME_M30", 1800),
    ("TIMEFRAME_H1", 3600),
    ("TIMEFRAME_H2", 7200),
    ("TIMEFRAME_H3", 10800),
    ("TIMEFRAME_H4", 14400),
    ("TIMEFRAME_H6", 21600),
    ("TIMEFRAME_H8", 28800),
    ("TIMEFRAME_H12", 43200),
    ("TIMEFRAME_D1", 86400),
    ("TIMEFRAME_W1", 604800),
    ("TIMEFRAME_MN1", 2592000),
)


def _build_tf_seconds_map() -> Dict[int, int]:
    out: Dict[int, int] = {}
    if mt5 is None:
        return out
    for attr_name, seconds in _MT5_TIMEFRAME_SECONDS:
        value = getattr(mt5, attr_name, None)
        if value is None:
            continue
        try:
            key = int(value)
        except Exception:
            continue
        if key < 0 or key in out:
            continue
        out[key] = int(seconds)
    return out


_TF_SECONDS_MAP = _build_tf_seconds_map()

_TF_REGEX = re.compile(r"([MHDW])\s*(\d+)")

# =============================================================================
# Standard Setup / Global Loggers
# =============================================================================
log_health = logging.getLogger("portfolio.engine.health")
log_err = logging.getLogger("portfolio.engine.err")
log_alert = logging.getLogger("portfolio.engine.alert")
log_diag = logging.getLogger("portfolio.engine.diag")
log_core_risk = logging.getLogger("core.risk_engine")
log_core_feature = logging.getLogger("core.data_engine")
log_core_signal = logging.getLogger("core.signal_engine")
log_core_risk_engine = logging.getLogger("core.risk_engine")

log_health.setLevel(logging.INFO)
log_err.setLevel(logging.ERROR)
log_alert.setLevel(logging.CRITICAL)
log_diag.setLevel(logging.INFO)
log_core_risk.setLevel(logging.INFO)
log_core_feature.setLevel(logging.ERROR)
log_core_signal.setLevel(logging.ERROR)
log_core_risk_engine.setLevel(logging.ERROR)

log_health.propagate = False
log_err.propagate = False
log_alert.propagate = False
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
_add_handler(log_alert, "portfolio_engine_alert.log", logging.CRITICAL)
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
    bar_key: str = ""
    timeframe: str = ""
    created_ts: float = 0.0
    source: str = "pipeline"


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
    bar_key: str = ""
    timeframe: str = ""
    created_ts: float = 0.0
    source: str = "pipeline"


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
    return env_truthy("MONITORING_ONLY_MODE", "0") or env_truthy(
        "ENGINE_MONITORING_ONLY", "0"
    ) or env_truthy("TRADING_DISABLED", "0")


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


_RESTRICTED_PICKLE_BUILTINS = frozenset(
    {
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "str",
        "bytes",
        "bytearray",
        "int",
        "float",
        "bool",
        "complex",
        "slice",
    }
)


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that rejects arbitrary globals/classes."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" and name in _RESTRICTED_PICKLE_BUILTINS:
            return getattr(__import__(module), name)
        raise pickle.UnpicklingError(f"unsafe pickle global: {module}.{name}")


def load_restricted_pickle(path: Any) -> Any:
    """Load a pickle artifact without permitting arbitrary code globals."""
    with open(path, "rb") as file_handle:
        payload = file_handle.read()
    return RestrictedUnpickler(io.BytesIO(payload)).load()


def _json_default_debug(stage: str, exc: Exception) -> None:
    try:
        if log_diag.isEnabledFor(logging.DEBUG):
            log_diag.debug("JSON_DEFAULT_CONVERSION_SKIP | stage=%s err=%s", stage, exc)
    except Exception:
        pass


def json_default(obj: object) -> Any:
    try:
        if is_dataclass(obj):
            return asdict(obj)
    except Exception as exc:
        _json_default_debug("dataclass", exc)
    try:
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="ignore")
    except Exception as exc:
        _json_default_debug("set_bytes", exc)
    try:
        if isinstance(obj, (_dt_type, _date_type)):
            return obj.isoformat()
    except Exception as exc:
        _json_default_debug("datetime", exc)
    try:
        if isinstance(obj, _Path_type):
            return str(obj)
    except Exception as exc:
        _json_default_debug("path", exc)
    try:
        if _NUMPY_TYPES and isinstance(obj, _NUMPY_TYPES):
            return obj.item()
    except Exception as exc:
        _json_default_debug("numpy", exc)
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


def signal_age_sec(
    bar_key: str,
    *,
    now_ts: Optional[float] = None,
    created_ts: float = 0.0,
) -> Optional[float]:
    parsed = parse_bar_key(bar_key)
    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        now_dt = (
            datetime.now(timezone.utc)
            if now_ts is None
            else datetime.fromtimestamp(float(now_ts), tz=timezone.utc)
        )
        return float((now_dt - parsed).total_seconds())
    if float(created_ts or 0.0) > 0.0 and now_ts is not None:
        return float(now_ts) - float(created_ts)
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
    "signal_age_sec",
    "tf_seconds",
]
