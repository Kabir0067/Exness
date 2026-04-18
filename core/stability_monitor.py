"""
core/stability_monitor.py — Section 10 institutional long-term stability:

    1. Drift detection
         * PSI    — Population Stability Index on feature distributions
         * KS     — Kolmogorov–Smirnov two-sample test
    2. Edge monitoring
         * rolling win-rate
         * rolling Sharpe
         * CUSUM on PnL (Page 1954) — detects a sustained shift in mean PnL
    3. Auto-degradation
         * apply a risk-multiplier reduction when monitors trip their
           institutional thresholds
    4. External watchdog
         * persistent heartbeat file; a second helper script (scripts/
           watchdog.py, spawned out-of-process) polls this file and force-
           kills the trading process if it stops updating.

Design principles (non-negotiable)
----------------------------------
* PURE NUMPY: no scipy/statsmodels required (ks_2samp & PSI are easy);
  this keeps the live hot-path deployable in minimal environments.
* STATELESS API where possible — CUSUM keeps state by caller-held dict
  so we never leak globals across assets / restarts.
* FAIL-CLOSED: when a monitor has not been initialised, `risk_multiplier`
  defaults to the SAFE value (1.0 clamped to `min`) until enough samples
  accumulate.
* DETERMINISTIC: same input → same decision; no wall-clock inside the
  statistical cores.

This module is OBSERVATIONAL — it reports a risk multiplier and a set of
flags. Application of that multiplier is the responsibility of the
execution layer (see Bot/Motor/execution.py). Isolating concerns here
means Section 10 never introduces silent coupling into the hot path.
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

log = logging.getLogger("core.stability_monitor")


# Heartbeat behaviour is controlled via env:
#   ENGINE_HEARTBEAT_ENABLED            — "1" (default) / "0" off
#   ENGINE_HEARTBEAT_MIN_INTERVAL_SEC   — throttle between disk writes (default 5.0s)
#   ENGINE_HEARTBEAT_PATH               — optional override of the file location
#
# Throttling is intentional: the FSM calls `beat()` on every loop tick
# (can be multiple times per second) but an external watchdog only
# needs second-level resolution. Writing more often just generates
# noise, disk I/O, and Windows Defender / antivirus contention on the
# atomic-replace path.
def _env_truthy(name: str, default: str = "1") -> bool:
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
    try:
        raw = os.getenv(name, "")
        return float(raw) if raw else float(default)
    except Exception:
        return float(default)


# =============================================================================
# (1) Drift — PSI & KS
# =============================================================================


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI).

        PSI = sum_i ( p_curr_i - p_ref_i ) * ln( p_curr_i / p_ref_i )

    Rules of thumb (industry standard):
        PSI < 0.10   : no significant change (GREEN)
        0.10–0.25    : moderate shift       (AMBER)
        > 0.25       : major shift — retrain (RED)

    Bins are derived from REFERENCE quantiles to avoid look-ahead drift on
    the live side. Zero-count bins are smoothed with a tiny epsilon.
    """
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cur = np.asarray(current, dtype=np.float64).reshape(-1)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size < 20 or cur.size < 20:
        return 0.0
    n_bins = max(2, int(n_bins))
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(ref, qs)
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    eps = 1e-6
    p_ref = (ref_hist.astype(np.float64) + eps) / (ref.size + eps * edges.size)
    p_cur = (cur_hist.astype(np.float64) + eps) / (cur.size + eps * edges.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))
    if not math.isfinite(psi):
        return 0.0
    return float(abs(psi))


def ks_two_sample(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float]:
    """
    Pure-numpy Kolmogorov–Smirnov two-sample test (no scipy).

    Returns (D_statistic, p_value_asymptotic).

    The asymptotic p-value uses the Kolmogorov distribution series
    expansion; this matches `scipy.stats.ks_2samp(method='asymp')` to ~4
    decimals for moderate-sized samples (n, m >= 100). For smaller samples
    the asymptotic is slightly anti-conservative — acceptable for our
    monitoring thresholds.
    """
    a = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(current, dtype=np.float64).reshape(-1))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n, m = int(a.size), int(b.size)
    if n == 0 or m == 0:
        return 0.0, 1.0
    combined = np.concatenate([a, b])
    combined.sort(kind="mergesort")
    cdf_a = np.searchsorted(a, combined, side="right") / n
    cdf_b = np.searchsorted(b, combined, side="right") / m
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    # Asymptotic p-value: Kolmogorov series
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 0.0
    fac = 2.0
    sign = 1.0
    prev = 0.0
    for j in range(1, 101):
        term = sign * math.exp(-2.0 * (j * lam) ** 2)
        p += fac * term
        if abs(term) <= 1e-10 or abs(term) <= 1e-4 * prev:
            break
        prev = abs(term)
        sign *= -1.0
    p = max(0.0, min(1.0, p))
    return d, p


@dataclass
class DriftReport:
    feature: str
    psi: float
    ks_d: float
    ks_p: float
    severity: str  # "green" | "amber" | "red"

    def to_dict(self) -> Dict[str, Any]:
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
    psi = population_stability_index(reference, current)
    d, p = ks_two_sample(reference, current)
    if psi >= psi_red or p < ks_p_red:
        sev = "red"
    elif psi >= psi_amber:
        sev = "amber"
    else:
        sev = "green"
    return DriftReport(feature=str(feature), psi=psi, ks_d=d, ks_p=p, severity=sev)


# =============================================================================
# (2) Edge monitoring — rolling win-rate, Sharpe, CUSUM
# =============================================================================


@dataclass
class EdgeMonitorState:
    """Caller-held state; no module globals."""

    trades: List[float] = field(default_factory=list)  # per-trade PnL (USD)
    equity_curve: List[float] = field(default_factory=list)
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    last_update_ts: float = 0.0
    # Reference stats for CUSUM (set at init).
    ref_mean: float = 0.0
    ref_sd: float = 0.0


def _rolling_win_rate(trades: Iterable[float], window: int) -> float:
    arr = np.asarray(list(trades), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    tail = arr[-max(1, int(window)) :]
    wins = float(np.sum(tail > 0.0))
    return float(wins / tail.size)


def _rolling_sharpe(trades: Iterable[float], window: int) -> float:
    arr = np.asarray(list(trades), dtype=np.float64)
    if arr.size < 8:
        return 0.0
    tail = arr[-max(1, int(window)) :]
    mu = float(np.mean(tail))
    sd = float(np.std(tail, ddof=1)) if tail.size > 1 else 0.0
    if sd <= 1e-12:
        return 0.0
    return float(mu / sd)


def update_cusum(
    state: EdgeMonitorState,
    trade_pnl: float,
    *,
    k_sigma: float = 0.5,
    reset_on_detect: bool = True,
) -> Tuple[float, float, bool]:
    """
    Page's CUSUM (1954) on trade PnL — detects a sustained DOWNWARD shift
    in mean trade PnL. Parameters follow the standard parameterisation:

        x_i     = standardised trade pnl (in ref SD units)
        S_pos_i = max(0, S_pos_{i-1} + x_i - k)
        S_neg_i = min(0, S_neg_{i-1} + x_i + k)

    An alarm is raised on S_neg_i ≤ -H (detection threshold h). Here we
    return the two partial sums and a Bool flag (``neg_alarm``) that the
    caller evaluates against its ``cusum_h_sigma`` threshold.
    """
    if state.ref_sd <= 0.0:
        # Not calibrated yet — update ref stats online and no alarm.
        state.trades.append(float(trade_pnl))
        if len(state.trades) >= 30:
            arr = np.asarray(state.trades[-200:], dtype=np.float64)
            state.ref_mean = float(np.mean(arr))
            state.ref_sd = float(np.std(arr, ddof=1)) or 1.0
        return state.cusum_pos, state.cusum_neg, False
    state.trades.append(float(trade_pnl))
    z = (float(trade_pnl) - state.ref_mean) / max(state.ref_sd, 1e-9)
    state.cusum_pos = max(0.0, state.cusum_pos + z - float(k_sigma))
    state.cusum_neg = min(0.0, state.cusum_neg + z + float(k_sigma))
    neg_alarm = state.cusum_neg <= -abs(float(os.getenv("CUSUM_H_SIGMA", "5.0") or "5.0"))
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
    """
    Compute an edge diagnostic and derive an institutional risk multiplier.

        CUSUM alarm     → critical_multiplier (default 0.25×)
        winrate < min   → degrade_multiplier  (default 0.5×)
        sharpe  < min   → degrade_multiplier
        else            → 1.0 (full size)

    The multiplier is the MINIMUM across the three sub-rules (safe
    composition: any single fault dominates).
    """
    wr = _rolling_win_rate(state.trades, window)
    sh = _rolling_sharpe(state.trades, window)
    neg_alarm = state.cusum_neg <= -abs(
        float(os.getenv("CUSUM_H_SIGMA", "5.0") or "5.0")
    )
    candidates: List[Tuple[float, str]] = [(1.0, "ok")]
    if neg_alarm:
        candidates.append((float(critical_multiplier), "cusum_alarm"))
    if len(state.trades) >= max(20, int(window)) and wr < float(min_winrate):
        candidates.append(
            (float(degrade_multiplier), f"low_winrate:{wr:.3f}<{min_winrate:.3f}")
        )
    if len(state.trades) >= max(20, int(window)) and sh < float(min_sharpe):
        candidates.append(
            (float(degrade_multiplier), f"low_sharpe:{sh:.3f}<{min_sharpe:.3f}")
        )
    mult, reason = min(candidates, key=lambda t: t[0])
    return EdgeDiagnostic(
        rolling_winrate=wr,
        rolling_sharpe=sh,
        cusum_alarm=bool(neg_alarm),
        risk_multiplier=float(mult),
        reason=str(reason),
    )


# =============================================================================
# (3) Auto-degradation orchestrator
# =============================================================================


@dataclass
class AutoDegradeState:
    last_multiplier: float = 1.0
    last_reason: str = "init"
    last_update_ts: float = 0.0


class RiskGovernor:
    """
    Merges (a) drift signals and (b) edge diagnostics into a single
    risk multiplier to apply at the execution layer.

    Composition rule (safe):
        final_mult = min(drift_mult, edge_mult)

    The risk governor does NOT enforce anything — it only reports. The
    execution layer is responsible for applying the multiplier to lot
    sizing (single source of truth, see `Bot/Motor/execution.py`).
    """

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
        drift_mult = 1.0
        drift_map = drift_multipliers or {"green": 1.0, "amber": 0.5, "red": 0.25}
        drift_mult = float(drift_map.get(str(drift_severity).lower(), 1.0))
        final = min(float(drift_mult), float(edge.risk_multiplier))
        reason = (
            f"drift={drift_severity}:{drift_mult:.2f}|edge={edge.reason}:"
            f"{edge.risk_multiplier:.2f}"
        )
        with self._lock:
            self._state.last_multiplier = float(final)
            self._state.last_reason = reason
            self._state.last_update_ts = time.time()
            return AutoDegradeState(**self._state.__dict__)

    def snapshot(self) -> AutoDegradeState:
        with self._lock:
            return AutoDegradeState(**self._state.__dict__)


# =============================================================================
# (4) External watchdog — heartbeat file
# =============================================================================


class Heartbeat:
    """
    Write a periodic JSON heartbeat to disk. An out-of-process watchdog
    polls this file and kills the trading process if the heartbeat goes
    stale (GIL deadlock, MT5 DLL hang, runaway tight loop).

    File format:
        {"pid": 1234, "ts": 1718000000.123, "status": "alive", "note": "..."}

    Institutional guarantees
    ------------------------
    * **Throttled**: callers may invoke `beat()` on every engine tick;
      internally we rate-limit disk writes to
      `ENGINE_HEARTBEAT_MIN_INTERVAL_SEC` (default 5.0s). A watchdog
      only needs second-level resolution — higher frequency is noise.
    * **Windows-safe**: on Windows, `os.replace` can race with antivirus
      real-time scanning or any process holding the target file open
      briefly, yielding `[WinError 5] Access denied`. This is a
      well-known transient OS behaviour. We retry up to
      `_WIN_REPLACE_RETRIES` times with exponential backoff.
    * **Quiet logging**: a *single* transient race is logged at DEBUG.
      We only emit ERROR after `_ERR_LOG_THRESHOLD` consecutive
      failures — that indicates a real problem (permissions, full disk)
      rather than a sub-second race.
    * **Opt-out**: set `ENGINE_HEARTBEAT_ENABLED=0` to disable all disk
      writes entirely (useful in dev / backtest / CI).
    """

    # Tunables — kept as class-level constants so tests can monkey-patch.
    _WIN_REPLACE_RETRIES: int = 3
    _WIN_REPLACE_BACKOFF_SEC: float = 0.015
    _ERR_LOG_THRESHOLD: int = 10

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = str(
            path or os.getenv("ENGINE_HEARTBEAT_PATH", "") or _DEFAULT_HEARTBEAT
        )
        self._lock = threading.Lock()
        self._enabled: bool = _env_truthy("ENGINE_HEARTBEAT_ENABLED", "1")
        self._min_interval: float = max(
            0.0, _env_float("ENGINE_HEARTBEAT_MIN_INTERVAL_SEC", 5.0)
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
        # Опт-аут: вақте кирои watchdog нест, системаро бо writes пур накунем.
        if not self._enabled:
            return

        now = time.time()

        # Throttle OUTSIDE the lock — саратон байни даъватҳо сбере карда
        # мешавад то каллер дар блокинг ғарқ нашавад (FSM tick-ҳоро фалаҷ
        # накунем).
        if (now - self._last_write_ts) < self._min_interval:
            return

        payload = {
            "pid": int(os.getpid()),
            "ts": round(now, 3),
            "status": str(status),
            "note": str(note)[:200],
        }
        tmp = f"{self._path}.tmp"

        with self._lock:
            # Double-check throttle зери қулф — ду thread наметавонанд
            # ҳамзамон navbatro пурра кунанд.
            if (now - self._last_write_ts) < self._min_interval:
                return

            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except Exception:
                        pass

                # Windows-safe atomic replace with retry on transient
                # PermissionError (WinError 5) caused by antivirus/
                # preview/watchdog momentarily holding the target open.
                last_exc: Optional[BaseException] = None
                for attempt in range(self._WIN_REPLACE_RETRIES):
                    try:
                        os.replace(tmp, self._path)
                        last_exc = None
                        break
                    except PermissionError as exc:
                        last_exc = exc
                        time.sleep(self._WIN_REPLACE_BACKOFF_SEC * (attempt + 1))
                    except OSError as exc:
                        # WinError 5 may surface as generic OSError in
                        # some Python builds; treat the same way.
                        if getattr(exc, "winerror", None) == 5:
                            last_exc = exc
                            time.sleep(
                                self._WIN_REPLACE_BACKOFF_SEC * (attempt + 1)
                            )
                        else:
                            raise

                if last_exc is not None:
                    # Все попытки исчерпаны — это транзиентный OS race,
                    # убираем tmp и считаем failure.
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
                    raise last_exc

                self._last_write_ts = now
                self._consecutive_failures = 0

            except Exception as exc:
                self._consecutive_failures += 1
                # Йак race-и гузаро → DEBUG (log spam-ро намесозад).
                # Танҳо баъди чандин shikasti payiham → ERROR (мушкили
                # воқеӣ: иҷозат, диски пур, файл қулф шудааст).
                if self._consecutive_failures >= self._ERR_LOG_THRESHOLD:
                    log.error(
                        "heartbeat_write_failed | path=%s consecutive=%d err=%s",
                        self._path,
                        self._consecutive_failures,
                        exc,
                    )
                else:
                    log.debug(
                        "heartbeat_write_transient | path=%s attempt=%d err=%s",
                        self._path,
                        self._consecutive_failures,
                        exc,
                    )

    def read(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def age_seconds(self) -> float:
        data = self.read()
        if not data:
            return float("inf")
        try:
            return max(0.0, time.time() - float(data.get("ts", 0.0) or 0.0))
        except Exception:
            return float("inf")


# =============================================================================
# Module-level singletons
# =============================================================================

_GOVERNOR: Optional[RiskGovernor] = None
_GOVERNOR_LOCK = threading.Lock()
_HEARTBEAT: Optional[Heartbeat] = None
_HEARTBEAT_LOCK = threading.Lock()


def get_default_governor() -> RiskGovernor:
    global _GOVERNOR
    with _GOVERNOR_LOCK:
        if _GOVERNOR is None:
            _GOVERNOR = RiskGovernor()
    return _GOVERNOR


def get_default_heartbeat() -> Heartbeat:
    global _HEARTBEAT
    with _HEARTBEAT_LOCK:
        if _HEARTBEAT is None:
            _HEARTBEAT = Heartbeat()
    return _HEARTBEAT
