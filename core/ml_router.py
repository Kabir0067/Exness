"""
core/ml_router.py — ML payload validation and routing.

Orchestrates scalping and intraday AI inference across multiple providers,
validates payload schemas, resolves signal consensus, and enforces
confidence floors before emitting actionable MLSignal results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from AiAnalysis.intrd_ai_analys import analyse_intraday
from AiAnalysis.scalp_ai_analys import analyse
from DataFeed.ai_day_market_feed import (
    get_ai_payload_btc_intraday,
    get_ai_payload_xau_intraday,
)

try:
    from DataFeed.ai_scalp_market_feed import get_ai_payload_btc, get_ai_payload_xau
except ImportError:
    from DataFeed.ai_scalp_market_feed import get_ai_payload_btc, get_ai_payload_xau

log = logging.getLogger("core.signal_engine")


# =============================================================================
# Global Constants
# =============================================================================
ML_CONFIDENCE_FLOOR = 0.70
ML_PROVIDERS = {"gemini", "groq", "cerebras"}
ML_INFER_CACHE_TTL_SEC = 60.0
ML_REQUIRED_SCALP = {
    "M1": ("ts_bar", "last_close", "atr_14", "rsi_14", "ema_20", "ema_50", "ema_200"),
}
ML_REQUIRED_INTRADAY = {
    "H1": ("ts_bar", "last_close", "atr_14", "rsi_14", "ema_20", "ema_50", "ema_200"),
}
_INFER_CACHE: Dict[str, Tuple[float, MLSignal]] = {}
_INFER_CACHE_LOCK = threading.RLock()


# =============================================================================
# Data Structures
# =============================================================================
@dataclass(frozen=True)
class MLSignal:
    """Immutable result of ML inference across scalp and intraday pipelines."""

    asset: str
    signal: str
    side: str
    confidence: float
    reason: str
    provider: str
    model: str
    entry: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    scalp_payload: Optional[Dict[str, Any]]
    intraday_payload: Optional[Dict[str, Any]]


# =============================================================================
# Public API
# =============================================================================
def validate_payload_schema(
    asset: str, payloads: Dict[str, Optional[Dict[str, Any]]]
) -> Tuple[bool, str]:
    """Validate that both scalp and intraday payloads conform to required schema."""
    asset_u = str(asset).upper().strip()
    if asset_u not in ("XAU", "BTC"):
        return False, "unsupported_asset"

    ok_s, reason_s = _validate_payload_block(
        payloads.get("scalp"),
        tf_required=ML_REQUIRED_SCALP,
        block_name="scalp",
    )
    if not ok_s:
        return False, reason_s

    ok_i, reason_i = _validate_payload_block(
        payloads.get("intraday"),
        tf_required=ML_REQUIRED_INTRADAY,
        block_name="intraday",
    )
    if not ok_i:
        return False, reason_i

    return True, "ok"


def fetch_ml_payloads(asset: str) -> Dict[str, Optional[Dict[str, Any]]]:
    """Fetch scalp and intraday ML payloads for the given asset."""
    asset_u = str(asset).upper().strip()
    if asset_u == "XAU":
        return {
            "scalp": get_ai_payload_xau(),
            "intraday": get_ai_payload_xau_intraday(),
        }
    if asset_u == "BTC":
        return {
            "scalp": get_ai_payload_btc(),
            "intraday": get_ai_payload_btc_intraday(),
        }
    return {"scalp": None, "intraday": None}


def infer_from_payloads(
    asset: str,
    *,
    scalp_payload: Optional[Dict[str, Any]],
    intraday_payload: Optional[Dict[str, Any]],
) -> MLSignal:
    """Run ML inference on pre-fetched payloads and return a consensus signal."""
    asset_u = str(asset).upper().strip()
    payloads = {"scalp": scalp_payload, "intraday": intraday_payload}

    if asset_u not in ("XAU", "BTC"):
        return _hold(asset_u, "unsupported_asset", payloads=payloads)

    schema_ok, schema_reason = validate_payload_schema(asset_u, payloads)
    if not schema_ok:
        return _hold(
            asset_u, f"payload_schema_invalid:{schema_reason}", payloads=payloads
        )

    cache_key = _payload_cache_key(asset_u, payloads)
    now = time.time()
    with _INFER_CACHE_LOCK:
        cached = _INFER_CACHE.get(cache_key)
        if cached is not None:
            ts_cached, cached_signal = cached
            if (now - float(ts_cached or 0.0)) <= ML_INFER_CACHE_TTL_SEC:
                return cached_signal

    scalp = _infer_scalp(asset_u, scalp_payload)
    intraday = _infer_intraday(asset_u, intraday_payload)
    chosen = _pick_signal(scalp, intraday)

    provider = str(chosen.get("provider", "none")).lower()
    if provider not in ML_PROVIDERS:
        result = _hold(asset_u, f"provider_not_ml:{provider}", payloads=payloads)
        _cache_result(cache_key, result)
        return result

    conf = float(chosen.get("confidence", 0.0) or 0.0)
    if conf < ML_CONFIDENCE_FLOOR:
        result = _hold(
            asset_u,
            f"low_confidence:{conf:.3f}<{ML_CONFIDENCE_FLOOR:.2f}",
            payloads=payloads,
        )
        _cache_result(cache_key, result)
        return result

    sig = str(chosen.get("signal", "HOLD")).upper().strip()
    if sig == "BUY":
        result = MLSignal(
            asset=asset_u,
            signal="STRONG BUY",
            side="Buy",
            confidence=conf,
            reason=str(chosen.get("reason", "ml_buy")),
            provider=provider,
            model=str(chosen.get("model", "unknown")),
            entry=chosen.get("entry"),
            stop_loss=chosen.get("stop_loss"),
            take_profit=chosen.get("take_profit"),
            scalp_payload=scalp_payload,
            intraday_payload=intraday_payload,
        )
        _cache_result(cache_key, result)
        return result
    if sig == "SELL":
        result = MLSignal(
            asset=asset_u,
            signal="STRONG SELL",
            side="Sell",
            confidence=conf,
            reason=str(chosen.get("reason", "ml_sell")),
            provider=provider,
            model=str(chosen.get("model", "unknown")),
            entry=chosen.get("entry"),
            stop_loss=chosen.get("stop_loss"),
            take_profit=chosen.get("take_profit"),
            scalp_payload=scalp_payload,
            intraday_payload=intraday_payload,
        )
        _cache_result(cache_key, result)
        return result

    result = _hold(asset_u, "ml_hold_signal", payloads=payloads)
    _cache_result(cache_key, result)
    return result


def infer_asset(asset: str) -> MLSignal:
    """End-to-end inference: fetch payloads, validate, infer, and return signal."""
    payloads = fetch_ml_payloads(asset)
    return infer_from_payloads(
        asset,
        scalp_payload=payloads.get("scalp"),
        intraday_payload=payloads.get("intraday"),
    )


# =============================================================================
# Private Helpers
# =============================================================================
def _hold(
    asset: str, reason: str, *, payloads: Optional[Dict[str, Any]] = None
) -> MLSignal:
    """Construct a neutral HOLD signal with diagnostic reason."""
    p = payloads or {}
    return MLSignal(
        asset=str(asset).upper(),
        signal="HOLD",
        side="Neutral",
        confidence=0.0,
        reason=str(reason),
        provider="none",
        model="none",
        entry=None,
        stop_loss=None,
        take_profit=None,
        scalp_payload=p.get("scalp"),
        intraday_payload=p.get("intraday"),
    )


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Convert to float with fallback on failure."""
    try:
        return float(x)
    except Exception:
        return float(default)


def _payload_cache_key(
    asset: str, payloads: Dict[str, Optional[Dict[str, Any]]]
) -> str:
    try:
        raw = json.dumps(
            {"asset": str(asset).upper().strip(), "payloads": payloads},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        raw = repr((str(asset).upper().strip(), payloads))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_result(cache_key: str, result: MLSignal) -> None:
    now = time.time()
    with _INFER_CACHE_LOCK:
        _INFER_CACHE[cache_key] = (now, result)
        if len(_INFER_CACHE) <= 256:
            return
        expired = [
            key
            for key, (ts_cached, _) in _INFER_CACHE.items()
            if (now - float(ts_cached or 0.0)) > ML_INFER_CACHE_TTL_SEC
        ]
        for key in expired[:256]:
            _INFER_CACHE.pop(key, None)


def _validate_payload_block(
    payload: Optional[Dict[str, Any]],
    *,
    tf_required: Dict[str, Tuple[str, ...]],
    block_name: str,
) -> Tuple[bool, str]:
    """Validate a single payload block against required timeframe fields."""
    if not isinstance(payload, dict):
        return False, f"{block_name}:not_dict"
    for tf, required_fields in tf_required.items():
        tf_obj = payload.get(tf)
        if not isinstance(tf_obj, dict):
            return False, f"{block_name}:{tf}:missing"
        missing: List[str] = []
        for field_name in required_fields:
            if field_name not in tf_obj:
                missing.append(field_name)
                continue
            v = tf_obj.get(field_name)
            if field_name == "ts_bar":
                try:
                    if float(v) < 1_000_000_000.0:
                        missing.append(field_name)
                except Exception:
                    missing.append(field_name)
                continue
            fv = _safe_float(v, -1.0)
            if fv <= 0.0:
                missing.append(field_name)
        if missing:
            return False, f"{block_name}:{tf}:missing_fields={','.join(missing)}"
    return True, "ok"


def _normalize_ai_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize raw AI provider output into a consistent dictionary shape."""
    if not isinstance(result, dict):
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "provider": "none",
            "model": "none",
            "reason": "invalid_result",
        }

    signal = str(result.get("signal", "HOLD")).upper().strip()
    if signal not in ("BUY", "SELL", "HOLD"):
        signal = "HOLD"

    conf = _safe_float(result.get("confidence"), 0.0)
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0

    provider = str(result.get("provider", "none") or "none").strip().lower()
    model = str(result.get("model", "none") or "none").strip()
    reason = str(result.get("reason", "") or "").strip() or "no_reason"

    return {
        "signal": signal,
        "confidence": conf,
        "provider": provider,
        "model": model,
        "reason": reason,
        "entry": result.get("entry"),
        "stop_loss": result.get("stop_loss"),
        "take_profit": result.get("take_profit"),
    }


def _infer_scalp(asset: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Run scalp-timeframe inference with error isolation."""
    if not payload:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "provider": "none",
            "model": "none",
            "reason": "scalp_payload_missing",
        }
    try:
        return _normalize_ai_result(analyse(asset, payload))
    except Exception as exc:
        log.error("ML_ROUTER scalp failure | asset=%s err=%s", asset, exc)
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "provider": "none",
            "model": "none",
            "reason": f"scalp_error:{exc}",
        }


def _infer_intraday(asset: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Run intraday-timeframe inference with error isolation."""
    if not payload:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "provider": "none",
            "model": "none",
            "reason": "intraday_payload_missing",
        }
    try:
        return _normalize_ai_result(analyse_intraday(asset, payload))
    except Exception as exc:
        log.error("ML_ROUTER intraday failure | asset=%s err=%s", asset, exc)
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "provider": "none",
            "model": "none",
            "reason": f"intraday_error:{exc}",
        }


def _pick_signal(scalp: Dict[str, Any], intraday: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve consensus between scalp and intraday signals."""
    s_sig = str(scalp.get("signal", "HOLD"))
    i_sig = str(intraday.get("signal", "HOLD"))

    s_ok = s_sig in ("BUY", "SELL")
    i_ok = i_sig in ("BUY", "SELL")

    if s_ok and i_ok and s_sig != i_sig:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "provider": "router",
            "model": "consensus",
            "reason": "ml_conflict",
        }

    if s_ok and i_ok:
        return (
            scalp
            if float(scalp.get("confidence", 0.0))
            >= float(intraday.get("confidence", 0.0))
            else intraday
        )
    if s_ok:
        return scalp
    if i_ok:
        return intraday

    return (
        scalp
        if float(scalp.get("confidence", 0.0)) >= float(intraday.get("confidence", 0.0))
        else intraday
    )


# =============================================================================
# Module Exports
# =============================================================================
__all__ = (
    "ML_CONFIDENCE_FLOOR",
    "ML_PROVIDERS",
    "ML_REQUIRED_INTRADAY",
    "ML_REQUIRED_SCALP",
    "MLSignal",
    "fetch_ml_payloads",
    "infer_asset",
    "infer_from_payloads",
    "validate_payload_schema",
)
