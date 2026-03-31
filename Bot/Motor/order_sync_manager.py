from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import MetaTrader5 as mt5
from mt5_client import MT5_LOCK

from .logging_setup import log_err, log_health
from .models import ExecutionResult, OrderIntent

if TYPE_CHECKING:
    from .engine import MultiAssetTradingEngine


class OrderSyncManager:
    """Owns pending-order state tracking and broker reconciliation."""

    def __init__(self, engine: "MultiAssetTradingEngine") -> None:
        self._e = engine

    def register_pending_order(self, intent: OrderIntent) -> None:
        e = self._e
        with e._order_state_lock:
            e._pending_order_meta[str(intent.order_id)] = {
                "asset": str(intent.asset),
                "symbol": str(intent.symbol),
                "signal": str(intent.signal),
                "signal_id": str(intent.signal_id),
                "confidence": float(intent.confidence),
                "lot": float(intent.lot),
                "sl": float(intent.sl),
                "tp": float(intent.tp),
                "req_price": float(intent.price),
                "enqueue_ts": float(intent.enqueue_time),
                "magic": int(getattr(intent.cfg, "magic", 777001) or 777001),
                "risk": intent.risk_manager,
            }

    def clear_pending_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        with self._e._order_state_lock:
            return self._e._pending_order_meta.pop(str(order_id), None)

    @staticmethod
    def _should_defer_result(r: ExecutionResult, meta: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(meta, dict) or bool(r.ok):
            return False
        reason = str(r.reason or "").lower()
        return (
            reason.startswith("order_send_none")
            or reason.startswith("retry_retcode_")
            or reason.startswith("exception_")
        )

    @staticmethod
    def _build_intent_snapshot(order_id: str, meta: Dict[str, Any]) -> Any:
        return SimpleNamespace(
            asset=str(meta.get("asset", "") or ""),
            symbol=str(meta.get("symbol", "") or ""),
            signal=str(meta.get("signal", "") or ""),
            confidence=float(meta.get("confidence", 0.0) or 0.0),
            lot=float(meta.get("lot", 0.0) or 0.0),
            sl=float(meta.get("sl", 0.0) or 0.0),
            tp=float(meta.get("tp", 0.0) or 0.0),
            price=float(meta.get("req_price", 0.0) or 0.0),
            enqueue_time=float(meta.get("enqueue_ts", 0.0) or 0.0),
            order_id=str(order_id),
            signal_id=str(meta.get("signal_id", "") or ""),
        )

    @staticmethod
    def _call_mt5(method_name: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(mt5, str(method_name), None)
        if not callable(fn):
            raise AttributeError(f"mt5.{method_name} not callable")
        with MT5_LOCK:
            return fn(*args, **kwargs)

    def resolve_order_result(self, r: ExecutionResult) -> None:
        e = self._e
        order_id = str(r.order_id)
        with e._order_state_lock:
            meta = e._pending_order_meta.get(order_id)
        if self._should_defer_result(r, meta):
            log_health.info(
                "ORDER_SYNC_WAIT | order_id=%s reason=%s retcode=%s",
                order_id,
                r.reason,
                r.retcode,
            )
            return

        e._record_execution_telemetry(r)
        with e._order_state_lock:
            rm = e._order_rm_by_id.pop(order_id, None)
            meta = e._pending_order_meta.pop(order_id, None)
        if rm is None and isinstance(meta, dict):
            rm = meta.get("risk")
        if rm is None:
            asset_hint = str((meta or {}).get("asset", "") or "").upper().strip()
            if asset_hint == "XAU" and e._xau is not None:
                rm = e._xau.risk
            elif asset_hint == "BTC" and e._btc is not None:
                rm = e._btc.risk
            else:
                log_health.warning(
                    "ORDER_SYNC_ORPHAN_RESULT | order_id=%s ok=%s reason=%s",
                    order_id,
                    r.ok,
                    r.reason,
                )
                return
        if rm and hasattr(rm, "on_execution_result"):
            try:
                rm.on_execution_result(r)
            except Exception as exc:
                log_err.error("ORDER_SYNC_RISK_UPDATE_ERROR | order_id=%s err=%s", order_id, exc)
        if (
            isinstance(meta, dict)
            and str(r.reason or "").startswith("sync_")
            and getattr(e, "_order_notifier", None)
        ):
            try:
                e._order_notifier(self._build_intent_snapshot(order_id, meta), r)
            except Exception:
                pass

    def probe_pending_order_state(
        self,
        order_id: str,
        meta: Dict[str, Any],
        now_ts: float,
    ) -> Optional[ExecutionResult]:
        e = self._e
        try:
            symbol = str(meta.get("symbol", "") or "")
            if not symbol:
                return None
            signal = str(meta.get("signal", "") or "")
            if signal not in ("Buy", "Sell"):
                return None
            want_pos_type = mt5.POSITION_TYPE_BUY if signal == "Buy" else mt5.POSITION_TYPE_SELL
            want_deal_type = mt5.ORDER_TYPE_BUY if signal == "Buy" else mt5.ORDER_TYPE_SELL
            enqueue_ts = float(meta.get("enqueue_ts", 0.0) or 0.0)
            req_price = float(meta.get("req_price", 0.0) or 0.0)
            req_lot = float(meta.get("lot", 0.0) or 0.0)
            magic = int(meta.get("magic", 777001) or 777001)
            signal_id = str(meta.get("signal_id", "") or "")

            positions = self._call_mt5("positions_get", symbol=symbol) or []

            for p in positions:
                try:
                    if int(getattr(p, "type", -1) or -1) != int(want_pos_type):
                        continue
                    if int(getattr(p, "magic", 0) or 0) != int(magic):
                        continue
                    p_vol = float(getattr(p, "volume", 0.0) or 0.0)
                    p_price = float(getattr(p, "price_open", 0.0) or req_price)
                    p_ticket = int(getattr(p, "ticket", 0) or 0)
                    p_time = float(getattr(p, "time", 0.0) or now_ts)
                    if p_time + 2.0 < enqueue_ts:
                        continue
                    if req_lot > 0.0 and p_vol <= 0.0:
                        continue
                    return ExecutionResult(
                        order_id=str(order_id),
                        signal_id=signal_id,
                        ok=True,
                        reason="sync_recovered_position",
                        sent_ts=float(enqueue_ts or now_ts),
                        fill_ts=float(p_time if p_time > 0 else now_ts),
                        req_price=float(req_price),
                        exec_price=float(p_price if p_price > 0 else req_price),
                        volume=float(p_vol if p_vol > 0 else req_lot),
                        slippage=float(abs((p_price if p_price > 0 else req_price) - req_price)),
                        retcode=0,
                        order_ticket=0,
                        deal_ticket=0,
                        position_ticket=int(p_ticket),
                    )
                except Exception:
                    continue

            if (now_ts - enqueue_ts) < e._order_sync_timeout_sec:
                return None

            start_dt = datetime.fromtimestamp(max(0.0, enqueue_ts - 5.0))
            end_dt = datetime.fromtimestamp(now_ts + 1.0)
            deals = self._call_mt5("history_deals_get", start_dt, end_dt) or []
            entry_in = int(getattr(mt5, "DEAL_ENTRY_IN", 0) or 0)
            best: Optional[Tuple[float, float, float, int]] = None
            best_score: Optional[Tuple[float, float]] = None
            for d in deals:
                try:
                    if str(getattr(d, "symbol", "")) != symbol:
                        continue
                    if int(getattr(d, "magic", 0) or 0) != int(magic):
                        continue
                    if int(getattr(d, "entry", 0) or 0) != entry_in:
                        continue
                    if int(getattr(d, "type", -1) or -1) != int(want_deal_type):
                        continue
                    d_time = float(getattr(d, "time", 0.0) or 0.0)
                    if d_time + 2.0 < enqueue_ts:
                        continue
                    d_vol = float(getattr(d, "volume", 0.0) or 0.0)
                    d_price = float(getattr(d, "price", 0.0) or req_price)
                    d_ticket = int(getattr(d, "ticket", 0) or 0)
                    vol_delta = abs(d_vol - req_lot)
                    score = (float(vol_delta), -float(d_time))
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (d_time, d_price, d_vol, d_ticket)
                except Exception:
                    continue

            if best is not None:
                b_time, b_price, b_vol, b_ticket = best
                return ExecutionResult(
                    order_id=str(order_id),
                    signal_id=signal_id,
                    ok=True,
                    reason="sync_recovered_deal",
                    sent_ts=float(enqueue_ts or now_ts),
                    fill_ts=float(b_time if b_time > 0 else now_ts),
                    req_price=float(req_price),
                    exec_price=float(b_price if b_price > 0 else req_price),
                    volume=float(b_vol if b_vol > 0 else req_lot),
                    slippage=float(abs((b_price if b_price > 0 else req_price) - req_price)),
                    retcode=0,
                    order_ticket=0,
                    deal_ticket=int(b_ticket),
                    position_ticket=0,
                )

            return ExecutionResult(
                order_id=str(order_id),
                signal_id=signal_id,
                ok=False,
                reason="sync_timeout_no_broker_state",
                sent_ts=float(enqueue_ts or now_ts),
                fill_ts=float(now_ts),
                req_price=float(req_price),
                exec_price=float(req_price),
                volume=float(req_lot),
                slippage=0.0,
                retcode=-3,
                order_ticket=0,
                deal_ticket=0,
                position_ticket=0,
            )
        except Exception as exc:
            log_err.error("ORDER_SYNC_PROBE_ERROR | order_id=%s err=%s", order_id, exc)
            return None

    def sync_pending_orders(self) -> None:
        e = self._e
        if e.dry_run:
            return
        now_ts = time.time()
        with e._order_state_lock:
            if not e._pending_order_meta:
                return
            if (now_ts - e._last_order_sync_ts) < e._order_sync_interval_sec:
                return
            e._last_order_sync_ts = now_ts
            pending_items = list(e._pending_order_meta.items())[:64]

        for order_id, meta in pending_items:
            enqueue_ts = float(meta.get("enqueue_ts", 0.0) or 0.0)
            if enqueue_ts > 0.0 and (now_ts - enqueue_ts) < e._order_sync_grace_sec:
                continue
            sync_res = self.probe_pending_order_state(order_id, meta, now_ts)
            if sync_res is None:
                continue
            log_health.info(
                "ORDER_SYNC_RESOLVED | order_id=%s ok=%s reason=%s",
                order_id,
                sync_res.ok,
                sync_res.reason,
            )
            try:
                self.resolve_order_result(sync_res)
            except Exception as exc:
                log_err.error("ORDER_SYNC_RESOLVE_ERROR | order_id=%s err=%s", order_id, exc)
