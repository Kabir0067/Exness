# core/health.py — System Heartbeat & Health Monitor
# Writes structured health logs every 60 seconds to prove system is operational.
from __future__ import annotations

import json
import logging
import threading
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Optional



def _build_health_logger(log_path: Path) -> logging.Logger:
    """Build a dedicated rotating file logger for health monitoring."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("core.health.heartbeat")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fh = RotatingFileHandler(
            str(log_path),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
            delay=True,
        )
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    return logger


class HealthMonitor:
    """
    Production-grade system health monitor.

    Features:
      - 60-second heartbeat proving the engine is alive
      - MT5 connection status checks
      - Current risk exposure state
      - Critical error and latency spike detection
      - Structured JSON logging for machine parsing

    All output goes to: logs/engine/portfolio_engine_health.log
    """

    def __init__(
        self,
        log_dir: Path,
        *,
        heartbeat_interval_sec: float = 60.0,
        mt5_health_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        portfolio_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        latency_threshold_ms: float = 500.0,
    ) -> None:
        log_path = log_dir / "engine" / "portfolio_engine_health.log"
        self._logger = _build_health_logger(log_path)
        self._interval = heartbeat_interval_sec
        self._mt5_health_fn = mt5_health_fn
        self._portfolio_fn = portfolio_fn
        self._latency_threshold_ms = latency_threshold_ms

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._error_count: int = 0
        self._latency_spikes: int = 0
        self._last_loop_ms: float = 0.0

        # Module health tracking
        self._module_status: Dict[str, str] = {
            "signal_engine_btc": "UNKNOWN",
            "signal_engine_xau": "UNKNOWN",
            "risk_engine_btc": "UNKNOWN",
            "risk_engine_xau": "UNKNOWN",
            "execution_worker": "UNKNOWN",
            "portfolio_risk": "UNKNOWN",
        }

    # ─── Public API ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the heartbeat monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="HealthMonitor",
        )
        self._thread.start()
        self._logger.info("HEALTH_MONITOR_STARTED | interval=%ds", self._interval)

    def stop(self) -> None:
        """Gracefully stop the heartbeat thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._logger.info("HEALTH_MONITOR_STOPPED")

    def report_module_status(self, module: str, status: str) -> None:
        """
        Report the status of a specific module.
        modules: signal_engine_btc, risk_engine_btc, execution_worker, etc.
        status: OK, ERROR, STALE, INITIALIZING
        """
        self._module_status[module] = status

    def report_error(self, module: str, error: str) -> None:
        """Report a critical error for immediate logging."""
        self._error_count += 1
        self._logger.critical(
            "CRITICAL_ERROR | module=%s error=%s total_errors=%d",
            module, error, self._error_count,
        )

    def report_loop_latency(self, latency_ms: float) -> None:
        """Report main loop latency for spike detection."""
        self._last_loop_ms = latency_ms
        if latency_ms > self._latency_threshold_ms:
            self._latency_spikes += 1
            self._logger.warning(
                "LATENCY_SPIKE | latency_ms=%.1f threshold=%.1f spikes=%d",
                latency_ms, self._latency_threshold_ms, self._latency_spikes,
            )

    # ─── Heartbeat loop ─────────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        """Main heartbeat loop — runs in daemon thread."""
        while not self._stop_event.is_set():
            try:
                self._emit_heartbeat()
            except Exception as exc:
                self._error_count += 1
                self._logger.error(
                    "HEARTBEAT_ERROR | %s | tb=%s",
                    exc, traceback.format_exc(),
                )
            self._stop_event.wait(self._interval)

    def _emit_heartbeat(self) -> None:
        """Emit a single structured heartbeat entry."""
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        # MT5 connection status
        mt5_status = {"connected": False, "ping_ms": -1}
        if self._mt5_health_fn:
            try:
                mt5_status = self._mt5_health_fn()
            except Exception:
                mt5_status = {"connected": False, "error": "mt5_health_fn failed"}

        # Portfolio / risk exposure
        portfolio = {}
        if self._portfolio_fn:
            try:
                portfolio = self._portfolio_fn()
            except Exception:
                portfolio = {"error": "portfolio_fn failed"}

        # Determine overall health status
        all_ok = all(s == "OK" for s in self._module_status.values() if s != "UNKNOWN")
        mt5_connected = bool(mt5_status.get("connected", False))
        status = "ALIVE" if (all_ok and mt5_connected) else "DEGRADED"

        if not mt5_connected:
            status = "MT5_DISCONNECTED"
        if any(s == "ERROR" for s in self._module_status.values()):
            status = "MODULE_FAILURE"

        heartbeat = {
            "type": "HEARTBEAT",
            "timestamp": ts,
            "status": status,
            "mt5": mt5_status,
            "risk_exposure": portfolio,
            "modules": dict(self._module_status),
            "diagnostics": {
                "loop_latency_ms": round(self._last_loop_ms, 1),
                "total_errors": self._error_count,
                "latency_spikes": self._latency_spikes,
            },
        }

        self._logger.info("HEARTBEAT | %s", json.dumps(heartbeat, default=str))
