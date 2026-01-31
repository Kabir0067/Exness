from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd


def safe_last(arr: np.ndarray, default: float = 0.0) -> float:
    """Return last finite value from array (scan backwards); else default."""
    try:
        if arr is None:
            return float(default)
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return float(default)

        # scan from end to first finite
        for v in a[::-1]:
            fv = float(v)
            if np.isfinite(fv):
                return fv
        return float(default)
    except Exception:
        return float(default)


def _finite(x: float) -> bool:
    try:
        return bool(np.isfinite(x))
    except Exception:
        return False


def _require_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> bool:
    return all(c in df.columns for c in cols)


def _to_f64(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _ts_to_ns(t: Any) -> int:
    """
    Robust timestamp->ns converter:
      - datetime/Timestamp -> ns
      - int/float assumed: sec/ms/us/ns by magnitude
    """
    try:
        if isinstance(t, (pd.Timestamp,)):
            return int(t.value)

        if isinstance(t, (np.datetime64,)):
            return int(pd.Timestamp(t).value)

        if isinstance(t, (int, np.integer)):
            v = int(t)
            av = abs(v)
            if av < 10_000_000_000:
                return v * 1_000_000_000
            if av < 10_000_000_000_000:
                return v * 1_000_000
            if av < 10_000_000_000_000_000:
                return v * 1_000
            return v

        if isinstance(t, (float, np.floating)):
            v = float(t)
            av = abs(v)
            if av < 10_000_000_000.0:
                return int(v * 1_000_000_000.0)
            if av < 10_000_000_000_000.0:
                return int(v * 1_000_000.0)
            if av < 10_000_000_000_000_000.0:
                return int(v * 1_000.0)
            return int(v)

        return int(pd.Timestamp(t).value)
    except Exception:
        return 0
