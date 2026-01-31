# _btc_indicators/utils.py — FIXED (production-grade)
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd


def safe_last(arr: np.ndarray, default: float = 0.0) -> float:
    """Return last finite value from array; else default."""
    try:
        if arr is None:
            return float(default)
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return float(default)

        m = np.isfinite(a)
        if not bool(np.any(m)):
            return float(default)

        # true "last finite" (not just last element)
        idx = int(np.flatnonzero(m)[-1])
        return float(a[idx])
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


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))
