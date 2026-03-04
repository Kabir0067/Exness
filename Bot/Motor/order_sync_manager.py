from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import MetaTrader5 as mt5
from mt5_client import mt5_async_call

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
        e._pending_order_meta[str(intent.order_id)] = {
            "asset": str(intent.asset),
            "symbol": str(intent.symbol),
            "signal": str(intent.signal),
            "signal_id": str(intent.signal_id),
            "lot": float(intent.lot),
            "req_price": float(intent.price),
            "enqueue_ts": float(intent.enqueue_time),
            "magic": int(getattr(intent.cfg, "magic", 777001) or 777001),
            "risk": intent.risk_manager,
        }

    def clear_pending_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        return self._e._pending_order_meta.pop(str(order_id), None)

    def resolve_order_result(self, r: ExecutionResult) -> None:
        e = self._e
        order_id = str(r.order_id)
        e._record_execution_telemetry(r)
        rm = e._order_rm_by_id.pop(order_id, None)
        meta = self.clear_pending_order(order_id)
        if rm is None and isinstance(meta, dict):
            rm = meta.get("risk")
        if rm is None:
            rm = e._xau.risk if order_id.startswith("PORD_XAU_") else e._btc.risk
        if rm and hasattr(rm, "on_execution_result"):
            rm.on_execution_result(r)

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

            positions = mt5_async_call(
                "positions_get",
                symbol=symbol,
                timeout=0.7,
                default=[],
                retries=1,
                repair_on_transport_error=True,
                direct_fallback=True,
            ) or []

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
            deals = mt5_async_call(
                "history_deals_get",
                start_dt,
                end_dt,
                timeout=0.9,
                default=[],
                retries=1,
                repair_on_transport_error=True,
                direct_fallback=True,
            ) or []
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
        if not e._pending_order_meta:
            return
        now_ts = time.time()
        if (now_ts - e._last_order_sync_ts) < e._order_sync_interval_sec:
            return
        e._last_order_sync_ts = now_ts

        for order_id, meta in list(e._pending_order_meta.items())[:64]:
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

