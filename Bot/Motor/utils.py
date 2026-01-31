from __future__ import annotations

import json
import re
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import MetaTrader5 as mt5

from .logging_setup import log_err


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
