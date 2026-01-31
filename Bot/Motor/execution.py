from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Callable, Optional

from ExnessAPI.order_execution import (
    OrderExecutor,
    OrderRequest,
    OrderResult as ExecOrderResult,
    log_health as exec_health_log,
)

from .logging_setup import log_err, log_health
from .models import ExecutionResult, OrderIntent


class ExecutionWorker(threading.Thread):
    def __init__(
        self,
        order_queue: "queue.Queue[OrderIntent]",
        result_queue: "queue.Queue[ExecutionResult]",
        dry_run: bool,
        order_notify_cb: Optional[Callable[[OrderIntent, ExecutionResult], None]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.order_queue = order_queue
        self.result_queue = result_queue
        self.dry_run = bool(dry_run)
        self.order_notify_cb = order_notify_cb
        # IMPORTANT: do not shadow threading.Thread._stop
        self._stop_evt = threading.Event()
        self._executor: Optional[OrderExecutor] = None

    def stop(self) -> None:
        self._stop_evt.set()

    def _get_hook(self, intent: OrderIntent) -> Optional[Callable[[ExecOrderResult], None]]:
        rm = intent.risk_manager
        if rm and hasattr(rm, "execution_hook"):
            fn = getattr(rm, "execution_hook", None)
            if callable(fn):
                return fn  # type: ignore[return-value]
        return None

    def _get_telemetry_hooks(self, intent: OrderIntent) -> Optional[dict[str, Callable[..., None]]]:
        rm = intent.risk_manager
        if not rm:
            return None

        hooks: dict[str, Callable[..., None]] = {}
        rec_metrics = getattr(rm, "record_execution_metrics", None)
        if callable(rec_metrics):
            hooks["record_execution_metrics"] = rec_metrics

        rec_failure = getattr(rm, "record_execution_failure", None)
        if callable(rec_failure):
            hooks["record_execution_failure"] = rec_failure

        return hooks or None

    def _build_executor(self) -> OrderExecutor:
        if self._executor is None:
            self._executor = OrderExecutor()
        return self._executor

    def _process(self, intent: OrderIntent) -> ExecutionResult:
        sent = time.time()
        try:
            req = OrderRequest(
                symbol=str(intent.symbol),
                signal=str(intent.signal),
                lot=float(intent.lot),
                sl=float(intent.sl),
                tp=float(intent.tp),
                price=float(intent.price),
                order_id=str(intent.order_id),
                signal_id=str(intent.signal_id),
                enqueue_time=float(intent.enqueue_time),
                magic=int(getattr(intent.cfg, "magic", 777001) or 777001),
                comment=str(getattr(intent.cfg, "comment", "portfolio") or "portfolio"),
            )
            ex = self._build_executor()
            hooks = self._get_telemetry_hooks(intent)
            r: ExecOrderResult = ex.send_market_order(req, telemetry_hooks=hooks)

            filled = float(getattr(r, "fill_ts", 0.0) or time.time())
            ok = bool(getattr(r, "ok", False))
            reason = str(getattr(r, "reason", "") or "")
            retcode = int(getattr(r, "retcode", 0) or 0)

            return ExecutionResult(
                order_id=str(intent.order_id),
                signal_id=str(intent.signal_id),
                ok=ok,
                reason=reason,
                sent_ts=float(getattr(r, "sent_ts", sent) or sent),
                fill_ts=filled,
                req_price=float(getattr(r, "req_price", intent.price) or intent.price),
                exec_price=float(getattr(r, "exec_price", intent.price) or intent.price),
                volume=float(getattr(r, "volume", intent.lot) or intent.lot),
                slippage=float(getattr(r, "slippage", 0.0) or 0.0),
                retcode=retcode,
            )
        except Exception as exc:
            log_err.error("execution worker error: %s | tb=%s", exc, traceback.format_exc())
            return ExecutionResult(
                order_id=str(intent.order_id),
                signal_id=str(intent.signal_id),
                ok=False,
                reason=f"exception:{exc}",
                sent_ts=sent,
                fill_ts=sent,
                req_price=float(intent.price),
                exec_price=float(intent.price),
                volume=float(intent.lot),
                slippage=0.0,
                retcode=-1,
            )

    def _fallback_risk_update(self, intent: OrderIntent, res: ExecutionResult) -> None:
        """If result_queue is unavailable/full, ensure RM still sees the result."""
        rm = intent.risk_manager
        if rm and hasattr(rm, "on_execution_result"):
            try:
                rm.on_execution_result(res)
            except Exception:
                pass

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                intent = self.order_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                res = self._process(intent)

                # Try to publish result to engine
                published = False
                try:
                    self.result_queue.put_nowait(res)
                    published = True
                except Exception:
                    published = False

                # If not published, update RM directly
                if not published:
                    self._fallback_risk_update(intent, res)

                # External notify bridge (telegram, etc.)
                if self.order_notify_cb:
                    try:
                        self.order_notify_cb(intent, res)
                    except Exception:
                        pass

                # Optional executor health log
                try:
                    exec_health_log.info(
                        "EXEC_HEALTH | order_id=%s ok=%s reason=%s",
                        res.order_id,
                        res.ok,
                        res.reason,
                    )
                except Exception:
                    pass

            finally:
                try:
                    self.order_queue.task_done()
                except Exception:
                    pass
