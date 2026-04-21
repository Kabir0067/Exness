"""
Drift, edge, and heartbeat monitoring for engine stability checks.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from log_config import get_log_path  # type: ignore

    _DEFAULT_HEARTBEAT = str(get_log_path("engine_heartbeat.json"))
except Exception:
    _DEFAULT_HEARTBEAT = os.path.join(os.getcwd(), "engine_heartbeat.json")


# =============================================================================
# Logging Setup
# =============================================================================
log_stability = logging.getLogger("core.stability_monitor")


# =============================================================================
# Private Helpers
# =============================================================================
def _env_truthy(name: str, default: str = "1") -> bool:
    """Return a truthy environment value."""
    try:
        return str(os.getenv(name, default) or default).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    except Exception:
        return default == "1"


def _env_float(name: str, default: float) -> float:
    """Return a float environment value."""
    try:
        raw_value = os.getenv(name, "")
        return float(raw_value) if raw_value else float(default)
    except Exception:
        return float(default)


# =============================================================================
# Drift Monitoring
# =============================================================================
def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute the population stability index."""
    reference_values = np.asarray(reference, dtype=np.float64).reshape(-1)
    current_values = np.asarray(current, dtype=np.float64).reshape(-1)

    reference_values = reference_values[np.isfinite(reference_values)]
    current_values = current_values[np.isfinite(current_values)]

    if reference_values.size < 20 or current_values.size < 20:
        return 0.0

    n_bins = max(2, int(n_bins))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(reference_values, quantiles)
    edges = np.unique(edges)

    if edges.size < 3:
        return 0.0

    reference_hist, _ = np.histogram(reference_values, bins=edges)
    current_hist, _ = np.histogram(current_values, bins=edges)
    epsilon = 1e-6
    reference_prob = (
        reference_hist.astype(np.float64) + epsilon
    ) / (reference_values.size + epsilon * edges.size)
    current_prob = (
        current_hist.astype(np.float64) + epsilon
    ) / (current_values.size + epsilon * edges.size)

    with np.errstate(divide="ignore", invalid="ignore"):
        psi = float(
            np.sum((current_prob - reference_prob) * np.log(current_prob / reference_prob))
        )

    if not math.isfinite(psi):
        return 0.0

    return float(abs(psi))


def ks_two_sample(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float]:
    """Compute the two-sample KS statistic and asymptotic p-value."""
    reference_values = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    current_values = np.sort(np.asarray(current, dtype=np.float64).reshape(-1))

    reference_values = reference_values[np.isfinite(reference_values)]
    current_values = current_values[np.isfinite(current_values)]

    n_reference = int(reference_values.size)
    n_current = int(current_values.size)
    if n_reference == 0 or n_current == 0:
        return 0.0, 1.0

    combined = np.concatenate([reference_values, current_values])
    combined.sort(kind="mergesort")
    cdf_reference = np.searchsorted(reference_values, combined, side="right") / n_reference
    cdf_current = np.searchsorted(current_values, combined, side="right") / n_current
    d_statistic = float(np.max(np.abs(cdf_reference - cdf_current)))

    effective_n = math.sqrt(n_reference * n_current / (n_reference + n_current))
    lam = (effective_n + 0.12 + 0.11 / effective_n) * d_statistic
    p_value = 0.0
    factor = 2.0
    sign = 1.0
    previous_term = 0.0

    for j in range(1, 101):
        term = sign * math.exp(-2.0 * (j * lam) ** 2)
        p_value += factor * term
        if abs(term) <= 1e-10 or abs(term) <= 1e-4 * previous_term:
            break
        previous_term = abs(term)
        sign *= -1.0

    p_value = max(0.0, min(1.0, p_value))
    return d_statistic, p_value


@dataclass
class DriftReport:
    feature: str
    psi: float
    ks_d: float
    ks_p: float
    severity: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable drift report."""
        return {
            "feature": self.feature,
            "psi": round(self.psi, 6),
            "ks_d": round(self.ks_d, 6),
            "ks_p": round(self.ks_p, 6),
            "severity": self.severity,
        }


def drift_report(
    reference: np.ndarray,
    current: np.ndarray,
    feature: str,
    *,
    psi_amber: float = 0.10,
    psi_red: float = 0.25,
    ks_p_red: float = 0.01,
) -> DriftReport:
    """Build a drift report from PSI and KS diagnostics."""
    psi = population_stability_index(reference, current)
    ks_d, ks_p = ks_two_sample(reference, current)

    if psi >= psi_red or ks_p < ks_p_red:
        severity = "red"
    elif psi >= psi_amber:
        severity = "amber"
    else:
        severity = "green"

    return DriftReport(
        feature=str(feature),
        psi=psi,
        ks_d=ks_d,
        ks_p=ks_p,
        severity=severity,
    )


# =============================================================================
# Edge Monitoring
# =============================================================================
@dataclass
class EdgeMonitorState:
    """Caller-owned edge monitoring state."""

    trades: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    last_update_ts: float = 0.0
    ref_mean: float = 0.0
    ref_sd: float = 0.0


def _rolling_win_rate(trades: Iterable[float], window: int) -> float:
    """Compute rolling win rate for recent trades."""
    trade_values = np.asarray(list(trades), dtype=np.float64)
    if trade_values.size == 0:
        return 0.0

    tail = trade_values[-max(1, int(window)) :]
    wins = float(np.sum(tail > 0.0))
    return float(wins / tail.size)


def _rolling_sharpe(trades: Iterable[float], window: int) -> float:
    """Compute rolling Sharpe for recent trades."""
    trade_values = np.asarray(list(trades), dtype=np.float64)
    if trade_values.size < 8:
        return 0.0

    tail = trade_values[-max(1, int(window)) :]
    mean_value = float(np.mean(tail))
    std_value = float(np.std(tail, ddof=1)) if tail.size > 1 else 0.0

    if std_value <= 1e-12:
        return 0.0

    return float(mean_value / std_value)


def update_cusum(
    state: EdgeMonitorState,
    trade_pnl: float,
    *,
    k_sigma: float = 0.5,
    reset_on_detect: bool = True,
) -> Tuple[float, float, bool]:
    """Update Page's CUSUM state for trade PnL."""
    if state.ref_sd <= 0.0:
        state.trades.append(float(trade_pnl))
        if len(state.trades) >= 30:
            trade_array = np.asarray(state.trades[-200:], dtype=np.float64)
            state.ref_mean = float(np.mean(trade_array))
            state.ref_sd = float(np.std(trade_array, ddof=1)) or 1.0
        return state.cusum_pos, state.cusum_neg, False

    state.trades.append(float(trade_pnl))
    z_score = (float(trade_pnl) - state.ref_mean) / max(state.ref_sd, 1e-9)
    state.cusum_pos = max(0.0, state.cusum_pos + z_score - float(k_sigma))
    state.cusum_neg = min(0.0, state.cusum_neg + z_score + float(k_sigma))
    neg_alarm = state.cusum_neg <= -abs(
        float(os.getenv("CUSUM_H_SIGMA", "5.0") or "5.0")
    )

    if neg_alarm and reset_on_detect:
        state.cusum_pos = 0.0
        state.cusum_neg = 0.0

    state.last_update_ts = time.time()
    return state.cusum_pos, state.cusum_neg, bool(neg_alarm)


@dataclass
class EdgeDiagnostic:
    rolling_winrate: float
    rolling_sharpe: float
    cusum_alarm: bool
    risk_multiplier: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable edge diagnostic."""
        return {
            "rolling_winrate": round(self.rolling_winrate, 4),
            "rolling_sharpe": round(self.rolling_sharpe, 4),
            "cusum_alarm": bool(self.cusum_alarm),
            "risk_multiplier": round(self.risk_multiplier, 4),
            "reason": self.reason,
        }


def edge_diagnostic(
    state: EdgeMonitorState,
    *,
    window: int = 50,
    min_winrate: float = 0.35,
    min_sharpe: float = 0.0,
    degrade_multiplier: float = 0.5,
    critical_multiplier: float = 0.25,
) -> EdgeDiagnostic:
    """Return the current edge diagnostic and risk multiplier."""
    rolling_winrate = _rolling_win_rate(state.trades, window)
    rolling_sharpe = _rolling_sharpe(state.trades, window)
    neg_alarm = state.cusum_neg <= -abs(
        float(os.getenv("CUSUM_H_SIGMA", "5.0") or "5.0")
    )
    candidates: List[Tuple[float, str]] = [(1.0, "ok")]

    if neg_alarm:
        candidates.append((float(critical_multiplier), "cusum_alarm"))

    if len(state.trades) >= max(20, int(window)) and rolling_winrate < float(
        min_winrate
    ):
        candidates.append(
            (
                float(degrade_multiplier),
                f"low_winrate:{rolling_winrate:.3f}<{min_winrate:.3f}",
            )
        )

    if len(state.trades) >= max(20, int(window)) and rolling_sharpe < float(
        min_sharpe
    ):
        candidates.append(
            (
                float(degrade_multiplier),
                f"low_sharpe:{rolling_sharpe:.3f}<{min_sharpe:.3f}",
            )
        )

    risk_multiplier, reason = min(candidates, key=lambda item: item[0])
    return EdgeDiagnostic(
        rolling_winrate=rolling_winrate,
        rolling_sharpe=rolling_sharpe,
        cusum_alarm=bool(neg_alarm),
        risk_multiplier=float(risk_multiplier),
        reason=str(reason),
    )


# =============================================================================
# Risk Governor
# =============================================================================
@dataclass
class AutoDegradeState:
    last_multiplier: float = 1.0
    last_reason: str = "init"
    last_update_ts: float = 0.0


class RiskGovernor:
    """Merge drift and edge diagnostics into one risk multiplier."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = AutoDegradeState()

    def update(
        self,
        *,
        drift_severity: str,
        edge: EdgeDiagnostic,
        drift_multipliers: Optional[Dict[str, float]] = None,
    ) -> AutoDegradeState:
        """Update the governor state and return a snapshot."""
        drift_map = drift_multipliers or {
            "green": 1.0,
            "amber": 0.5,
            "red": 0.25,
        }
        drift_multiplier = float(drift_map.get(str(drift_severity).lower(), 1.0))
        final_multiplier = min(float(drift_multiplier), float(edge.risk_multiplier))
        reason = (
            f"drift={drift_severity}:{drift_multiplier:.2f}|edge={edge.reason}:"
            f"{edge.risk_multiplier:.2f}"
        )

        with self._lock:
            self._state.last_multiplier = float(final_multiplier)
            self._state.last_reason = reason
            self._state.last_update_ts = time.time()
            return AutoDegradeState(**self._state.__dict__)

    def snapshot(self) -> AutoDegradeState:
        """Return the current governor state snapshot."""
        with self._lock:
            return AutoDegradeState(**self._state.__dict__)


# =============================================================================
# Heartbeat
# =============================================================================
class Heartbeat:
    """Write a throttled heartbeat file for an external watchdog."""

    _WIN_REPLACE_RETRIES: int = 3
    _WIN_REPLACE_BACKOFF_SEC: float = 0.015
    _ERR_LOG_THRESHOLD: int = 10

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = str(
            path or os.getenv("ENGINE_HEARTBEAT_PATH", "") or _DEFAULT_HEARTBEAT
        )
        self._lock = threading.Lock()
        self._enabled = _env_truthy("ENGINE_HEARTBEAT_ENABLED", "1")
        self._min_interval = max(
            0.0,
            _env_float("ENGINE_HEARTBEAT_MIN_INTERVAL_SEC", 5.0),
        )
        self._last_write_ts: float = 0.0
        self._consecutive_failures: int = 0

        try:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    @property
    def path(self) -> str:
        return self._path

    @property
    def enabled(self) -> bool:
        return self._enabled

    def beat(self, *, status: str = "alive", note: str = "") -> None:
        """Write a heartbeat if the throttle window allows it."""
        if not self._enabled:
            return

        now = time.time()
        if (now - self._last_write_ts) < self._min_interval:
            return

        payload = {
            "pid": int(os.getpid()),
            "ts": round(now, 3),
            "status": str(status),
            "note": str(note)[:200],
        }
        temp_path = f"{self._path}.tmp"

        with self._lock:
            if (now - self._last_write_ts) < self._min_interval:
                return

            try:
                with open(temp_path, "w", encoding="utf-8") as file_handle:
                    json.dump(payload, file_handle)
                    file_handle.flush()

                    try:
                        os.fsync(file_handle.fileno())
                    except Exception:
                        pass

                last_exc: Optional[BaseException] = None

                for attempt in range(self._WIN_REPLACE_RETRIES):
                    try:
                        os.replace(temp_path, self._path)
                        last_exc = None
                        break
                    except PermissionError as exc:
                        last_exc = exc
                        time.sleep(self._WIN_REPLACE_BACKOFF_SEC * (attempt + 1))
                    except OSError as exc:
                        if getattr(exc, "winerror", None) == 5:
                            last_exc = exc
                            time.sleep(
                                self._WIN_REPLACE_BACKOFF_SEC * (attempt + 1)
                            )
                        else:
                            raise

                if last_exc is not None:
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                    raise last_exc

                self._last_write_ts = now
                self._consecutive_failures = 0
            except Exception as exc:
                self._consecutive_failures += 1

                if self._consecutive_failures >= self._ERR_LOG_THRESHOLD:
                    log_stability.error(
                        "heartbeat_write_failed | path=%s consecutive=%d err=%s",
                        self._path,
                        self._consecutive_failures,
                        exc,
                    )
                else:
                    log_stability.debug(
                        "heartbeat_write_transient | path=%s attempt=%d err=%s",
                        self._path,
                        self._consecutive_failures,
                        exc,
                    )

    def read(self) -> Optional[Dict[str, Any]]:
        """Read the current heartbeat payload."""
        try:
            with open(self._path, "r", encoding="utf-8") as file_handle:
                return json.load(file_handle)
        except Exception:
            return None

    def age_seconds(self) -> float:
        """Return the age of the current heartbeat."""
        data = self.read()
        if not data:
            return float("inf")

        try:
            return max(0.0, time.time() - float(data.get("ts", 0.0) or 0.0))
        except Exception:
            return float("inf")


# =============================================================================
# Module State
# =============================================================================
_GOVERNOR: Optional[RiskGovernor] = None
_GOVERNOR_LOCK = threading.Lock()
_HEARTBEAT: Optional[Heartbeat] = None
_HEARTBEAT_LOCK = threading.Lock()


def get_default_governor() -> RiskGovernor:
    """Return the module-level default risk governor."""
    global _GOVERNOR

    with _GOVERNOR_LOCK:
        if _GOVERNOR is None:
            _GOVERNOR = RiskGovernor()

    return _GOVERNOR


def get_default_heartbeat() -> Heartbeat:
    """Return the module-level default heartbeat writer."""
    global _HEARTBEAT

    with _HEARTBEAT_LOCK:
        if _HEARTBEAT is None:
            _HEARTBEAT = Heartbeat()

    return _HEARTBEAT
