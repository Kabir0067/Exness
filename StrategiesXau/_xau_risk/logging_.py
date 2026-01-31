from __future__ import annotations

import csv
import json
import logging
import threading
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Any

import MetaTrader5 as mt5

from datetime import datetime, timezone

from log_config import LOG_DIR as LOG_ROOT, get_log_path

def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

# ============================================================
# ERROR-only logging (isolated) + rotation
# ============================================================
LOG_DIR = LOG_ROOT

log_risk = logging.getLogger("risk_xau")
log_risk.setLevel(logging.ERROR)
log_risk.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in log_risk.handlers):
    fh = RotatingFileHandler(
        filename=str(get_log_path("risk_manager_xau.log")),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    log_risk.addHandler(fh)

_FILE_LOCK = threading.Lock()

class RiskLogger:
    """
    Handles CSV stats recording and execution metrics to keep main RiskManager clean.
    """
    def __init__(self):
        self._hist_slip_counts: Dict[str, int] = {}
        self._reject_counts: Dict[str, int] = None
        
    @staticmethod
    def _slip_bins() -> List[float]:
        return [-float("inf"), -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, float("inf")]

    @staticmethod
    def _bin_label(a: float, b: float) -> str:
        return f"{a} to {b}"

    def _load_histogram_counts(self, path: Path, bins: List[float]) -> Dict[str, int]:
        counts: Dict[str, int] = {self._bin_label(bins[i], bins[i + 1]): 0 for i in range(len(bins) - 1)}
        if not path.exists():
            return counts
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    k = str(row.get("bin", "")).strip()
                    if k in counts:
                        try:
                            counts[k] = int(float(row.get("count", 0) or 0))
                        except Exception:
                            counts[k] = 0
        except Exception:
            return counts
        return counts

    def _write_histogram_counts_atomic(self, path: Path, counts: Dict[str, int]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["bin", "count"])
                w.writeheader()
                for k, v in counts.items():
                    w.writerow({"bin": k, "count": int(v)})
            if path.exists():
                path.unlink()
            tmp.rename(path)
        except Exception:
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def _delay_bins() -> List[float]:
        return [0, 50, 100, 150, 200, 250, 300, 400, 500, 1000, float("inf")]

    def update_execution_analysis(self, metrics: dict) -> None:
        try:
            s = float(metrics.get("slippage_points", 0.0) or 0.0)
            d = float(metrics.get("latency_send_to_fill_ms", 0.0) or 0.0)
            
            with _FILE_LOCK:
                # slippage histogram
                slippage_file = LOG_DIR / "slippage_histogram_xau.csv"
                bins_s = self._slip_bins()
                counts_s = self._load_histogram_counts(slippage_file, bins_s)

                for i in range(len(bins_s) - 1):
                    if bins_s[i] <= s < bins_s[i + 1]:
                        label = self._bin_label(bins_s[i], bins_s[i + 1])
                        counts_s[label] = int(counts_s.get(label, 0)) + 1
                        break
                
                self._write_histogram_counts_atomic(slippage_file, counts_s)

                # delay histogram
                delay_file = LOG_DIR / "fill_delay_histogram_xau.csv"
                bins_d = self._delay_bins()
                counts_d = self._load_histogram_counts(delay_file, bins_d)

                for i in range(len(bins_d) - 1):
                    if bins_d[i] <= d < bins_d[i + 1]:
                        label = self._bin_label(bins_d[i], bins_d[i + 1])
                        counts_d[label] = int(counts_d.get(label, 0)) + 1
                        break
                
                self._write_histogram_counts_atomic(delay_file, counts_d)

        except Exception as exc:
            log_risk.error("update_execution_analysis error: %s", exc)

    def record_metrics(self, metrics: dict) -> None:
        """Append metrics to CSV and JSONL."""
        try:
            with _FILE_LOCK:
                csv_file = LOG_DIR / "execution_quality_xau.csv"
                write_header = not csv_file.exists()
                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(metrics)

                jsonl_file = LOG_DIR / "execution_metrics_xau.jsonl"
                with open(jsonl_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            
            self.update_execution_analysis(metrics)
        except Exception as exc:
            log_risk.error("record_metrics error: %s", exc)

    def record_failure(self, order_id: str, enqueue_time: float, send_time: float, reason: str) -> None:
        try:
            metrics = {
                "order_id": str(order_id),
                "timestamp": _utcnow().isoformat(),
                "enqueue_time": float(enqueue_time),
                "send_time": float(send_time),
                "reason": str(reason),
                "latency_enqueue_to_send_ms": (float(send_time) - float(enqueue_time)) * 1000.0,
            }

            with _FILE_LOCK:
                csv_file = LOG_DIR / "execution_failures_xau.csv"
                write_header = not csv_file.exists()
                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(metrics)
            
            self.update_rejection_analysis(str(reason))
        except Exception as exc:
            log_risk.error("record_failure error: %s", exc)

    def update_rejection_analysis(self, reason: str) -> None:
        try:
            reject_file = LOG_DIR / "reject_rate_analysis_xau.csv"
            with _FILE_LOCK:
                if self._reject_counts is None:
                    self._reject_counts = {}
                    if reject_file.exists():
                        try:
                            with open(reject_file, "r", encoding="utf-8", newline="") as f:
                                r = csv.DictReader(f)
                                for row in r:
                                    rs = str(row.get("reason", "")).strip()
                                    if not rs:
                                        continue
                                    try:
                                        self._reject_counts[rs] = int(float(row.get("count", 0) or 0))
                                    except Exception:
                                        self._reject_counts[rs] = 0
                        except Exception:
                            self._reject_counts = {}

                key = str(reason or "unknown").strip() or "unknown"
                self._reject_counts[key] = int(self._reject_counts.get(key, 0)) + 1

                total = int(sum(self._reject_counts.values()))
                tmp = reject_file.with_suffix(reject_file.suffix + ".tmp")
                with open(tmp, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["reason", "count", "total_attempts", "reject_rate"])
                    w.writeheader()
                    for rs, cnt in sorted(self._reject_counts.items(), key=lambda x: (-x[1], x[0])):
                        rate = float(cnt) / max(1, total)
                        w.writerow(
                            {
                                "reason": rs,
                                "count": int(cnt),
                                "total_attempts": int(total),
                                "reject_rate": float(rate),
                            }
                        )
                
                if reject_file.exists():
                    reject_file.unlink()
                tmp.rename(reject_file)

        except Exception as exc:
            log_risk.error("update_rejection_analysis error: %s", exc)
