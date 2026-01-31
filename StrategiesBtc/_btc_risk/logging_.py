# _btc_risk/logging_.py
from __future__ import annotations

import csv
import json
import logging
import threading
import time
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from log_config import LOG_DIR as LOG_ROOT, get_log_path


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


LOG_DIR = LOG_ROOT

log_risk = logging.getLogger("risk_btc")
log_risk.setLevel(logging.ERROR)
log_risk.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in log_risk.handlers):
    fh = RotatingFileHandler(
        filename=str(get_log_path("risk_manager_btc.log")),
        maxBytes=5242880,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    log_risk.addHandler(fh)

_FILE_LOCK = threading.Lock()


def _atomic_replace(path: Path, content_writer) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content_writer(tmp)
        tmp.replace(path)  # atomic on same filesystem
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


class RiskLogger:
    """
    CSV stats and execution metrics for BTC risk.
    Optimized:
      - Histograms + reject analysis in RAM
      - Periodic flush to CSV (atomic)
      - Append-only metrics CSV/JSONL
    """

    _HIST_FLUSH_SEC = 5.0
    _REJECT_FLUSH_SEC = 7.0

    def __init__(self):
        self._slip_bins = [-float("inf"), -50, -30, -20, -10, -5, 0, 5, 10, 20, 30, 50, float("inf")]
        self._delay_bins = [0, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, float("inf")]

        self._slip_counts: Dict[str, int] = self._init_bin_counts(self._slip_bins)
        self._delay_counts: Dict[str, int] = self._init_bin_counts(self._delay_bins)

        self._reject_counts: Dict[str, int] = {}
        self._hist_last_flush_ts: float = 0.0
        self._reject_last_flush_ts: float = 0.0

        # try load existing hist/reject (best-effort)
        try:
            self._load_existing_hist()
        except Exception:
            pass
        try:
            self._load_existing_rejects()
        except Exception:
            pass

    @staticmethod
    def _bin_label(a: float, b: float) -> str:
        return f"{a} to {b}"

    def _init_bin_counts(self, bins: List[float]) -> Dict[str, int]:
        return {self._bin_label(bins[i], bins[i + 1]): 0 for i in range(len(bins) - 1)}

    def _find_bin_label(self, bins: List[float], x: float) -> str:
        for i in range(len(bins) - 1):
            if bins[i] <= x < bins[i + 1]:
                return self._bin_label(bins[i], bins[i + 1])
        return self._bin_label(bins[-2], bins[-1])

    def _load_counts_csv(self, path: Path) -> Dict[str, int]:
        out: Dict[str, int] = {}
        if not path.exists():
            return out
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    k = str(row.get("bin", "")).strip()
                    if not k:
                        continue
                    try:
                        out[k] = int(float(row.get("count", 0) or 0))
                    except Exception:
                        out[k] = 0
        except Exception:
            return {}
        return out

    def _load_existing_hist(self) -> None:
        with _FILE_LOCK:
            slip_file = LOG_DIR / "slippage_histogram_btc.csv"
            delay_file = LOG_DIR / "fill_delay_histogram_btc.csv"
            s = self._load_counts_csv(slip_file)
            d = self._load_counts_csv(delay_file)
            for k in self._slip_counts.keys():
                if k in s:
                    self._slip_counts[k] = int(s[k])
            for k in self._delay_counts.keys():
                if k in d:
                    self._delay_counts[k] = int(d[k])

    def _load_existing_rejects(self) -> None:
        reject_file = LOG_DIR / "reject_rate_analysis_btc.csv"
        if not reject_file.exists():
            return
        with _FILE_LOCK:
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

    def _flush_histograms(self, *, force: bool = False) -> None:
        now = time.time()
        if (not force) and (now - self._hist_last_flush_ts) < self._HIST_FLUSH_SEC:
            return
        self._hist_last_flush_ts = now

        def write_slip(tmp: Path) -> None:
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["bin", "count"])
                w.writeheader()
                for k, v in self._slip_counts.items():
                    w.writerow({"bin": k, "count": int(v)})

        def write_delay(tmp: Path) -> None:
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["bin", "count"])
                w.writeheader()
                for k, v in self._delay_counts.items():
                    w.writerow({"bin": k, "count": int(v)})

        with _FILE_LOCK:
            _atomic_replace(LOG_DIR / "slippage_histogram_btc.csv", write_slip)
            _atomic_replace(LOG_DIR / "fill_delay_histogram_btc.csv", write_delay)

    def _flush_rejects(self, *, force: bool = False) -> None:
        now = time.time()
        if (not force) and (now - self._reject_last_flush_ts) < self._REJECT_FLUSH_SEC:
            return
        self._reject_last_flush_ts = now

        total = int(sum(self._reject_counts.values()))
        rows = sorted(self._reject_counts.items(), key=lambda kv: (-kv[1], kv[0]))

        def write_rej(tmp: Path) -> None:
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["reason", "count", "total_attempts", "reject_rate"])
                w.writeheader()
                for rs, cnt in rows:
                    rate = float(cnt) / max(1, total)
                    w.writerow(
                        {"reason": rs, "count": int(cnt), "total_attempts": int(total), "reject_rate": float(rate)}
                    )

        with _FILE_LOCK:
            _atomic_replace(LOG_DIR / "reject_rate_analysis_btc.csv", write_rej)

    @staticmethod
    def _ordered_fieldnames(keys: List[str]) -> List[str]:
        preferred = [
            "order_id",
            "timestamp",
            "side",
            "enqueue_time",
            "send_time",
            "fill_time",
            "expected_price",
            "filled_price",
            "slippage",
            "slippage_points",
            "latency_enqueue_to_send_ms",
            "latency_send_to_fill_ms",
            "total_latency_ms",
            "reason",
        ]
        out: List[str] = []
        s = set(keys)
        for k in preferred:
            if k in s:
                out.append(k)
        for k in keys:
            if k not in set(out):
                out.append(k)
        return out

    def update_execution_analysis(self, metrics: dict) -> None:
        """
        RAM histogram update; periodic flush.
        Accepts both slippage_points and latency_send_to_fill_ms.
        """
        try:
            s = float(metrics.get("slippage_points", 0.0) or 0.0)
            d = float(metrics.get("latency_send_to_fill_ms", 0.0) or 0.0)

            # slippage is ABS in your system; still normalize to abs to be safe
            s = abs(s)

            slip_label = self._find_bin_label(self._slip_bins, s)
            delay_label = self._find_bin_label(self._delay_bins, d)

            with _FILE_LOCK:
                self._slip_counts[slip_label] = int(self._slip_counts.get(slip_label, 0) + 1)
                self._delay_counts[delay_label] = int(self._delay_counts.get(delay_label, 0) + 1)

            self._flush_histograms(force=False)
        except Exception as exc:
            log_risk.error("update_execution_analysis error: %s | tb=%s", exc, traceback.format_exc())

    def record_metrics(self, metrics: dict) -> None:
        try:
            with _FILE_LOCK:
                csv_file = LOG_DIR / "execution_quality_btc.csv"
                jsonl_file = LOG_DIR / "execution_metrics_btc.jsonl"

                fieldnames = self._ordered_fieldnames(list(metrics.keys()))
                write_header = not csv_file.exists()

                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    row = {k: metrics.get(k, "") for k in fieldnames}
                    writer.writerow(row)

                with open(jsonl_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            self.update_execution_analysis(metrics)
        except Exception as exc:
            log_risk.error("record_metrics error: %s | tb=%s", exc, traceback.format_exc())

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
                csv_file = LOG_DIR / "execution_failures_btc.csv"
                write_header = not csv_file.exists()
                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(metrics)

            self.update_rejection_analysis(str(reason))
        except Exception as exc:
            log_risk.error("record_failure error: %s | tb=%s", exc, traceback.format_exc())

    def update_rejection_analysis(self, reason: str) -> None:
        try:
            key = str(reason or "unknown").strip() or "unknown"
            with _FILE_LOCK:
                self._reject_counts[key] = int(self._reject_counts.get(key, 0) + 1)
            self._flush_rejects(force=False)
        except Exception as exc:
            log_risk.error("update_rejection_analysis error: %s | tb=%s", exc, traceback.format_exc())
