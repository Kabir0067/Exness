"""ExnessAPI/order_execution.py (PRODUCTION-GRADE)

Market order execution with:
- Broker-safe volume/price/stops normalization
- Retry logic for transient retcodes (requote/timeout/off quotes)
- Post-fill SL/TP attachment (TRADE_ACTION_SLTP) when initial stops are rejected/ignored
- Telemetry hooks (optional)
- Rotating logs under Logs/

This module is deterministic and side-effect safe:
- Never sleeps under MT5_LOCK
- Never raises from telemetry hooks

NOTE: No trading system can be guaranteed "100% error-free" because broker/market/network
conditions can change. This module is built to fail safe and explain why.
"""

from __future__ import annotations

import logging
import math
import os
import time
import traceback
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Optional

import MetaTrader5 as mt5

from mt5_client import MT5_LOCK, ensure_mt5
from log_config import LOG_DIR as LOG_ROOT, get_log_path


# =============================================================================
# Logging (Logs/order_execution.log + Logs/order_execution_health.log)
# =============================================================================
LOG_DIR = LOG_ROOT


def _rotating_file_logger(
    name: str,
    filename: str,
    level: int,
    max_bytes_env: str,
    backups_env: str,
    default_max_bytes: int = 5_242_880,  # 5MB
    default_backups: int = 5,
) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = False

    if not any(isinstance(h, RotatingFileHandler) for h in log.handlers):
        fh = RotatingFileHandler(
            filename=str(get_log_path(filename)),
            maxBytes=int(os.getenv(max_bytes_env, str(default_max_bytes))),
            backupCount=int(os.getenv(backups_env, str(default_backups))),
            encoding="utf-8",
            delay=True,
        )
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s")
        )
        log.addHandler(fh)

    return log


log_err = _rotating_file_logger(
    "order_execution",
    "order_execution.log",
    logging.ERROR,
    max_bytes_env="ORDER_EXEC_LOG_MAX_BYTES",
    backups_env="ORDER_EXEC_LOG_BACKUPS",
)

log_health = _rotating_file_logger(
    "order_execution.health",
    "order_execution_health.log",
    logging.INFO,
    max_bytes_env="ORDER_EXEC_HEALTH_LOG_MAX_BYTES",
    backups_env="ORDER_EXEC_HEALTH_LOG_BACKUPS",
)


# =============================================================================
# Models
# =============================================================================
@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    signal: str  # "Buy" or "Sell"
    lot: float
    sl: float
    tp: float
    price: float  # snapshot price at enqueue
    order_id: str
    signal_id: str
    enqueue_time: float
    deviation: int = 50
    magic: int = 987654
    comment: str = "xau_scalper"


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    signal_id: str
    ok: bool
    reason: str
    sent_ts: float
    fill_ts: float
    req_price: float
    exec_price: float
    volume: float
    slippage: float
    retcode: int = 0


# =============================================================================
# Executor
# =============================================================================
class OrderExecutor:
    """Production-grade MT5 market order executor."""

    def __init__(
        self,
        *,
        normalize_volume_fn: Optional[Callable[[float, Any], float]] = None,
        sanitize_stops_fn: Optional[Callable[[str, float, float, float], tuple[float, float]]] = None,
        normalize_price_fn: Optional[Callable[[float], float]] = None,
        close_on_sltp_fail: Optional[bool] = None,
    ) -> None:
        self.normalize_volume_fn = normalize_volume_fn
        self.sanitize_stops_fn = sanitize_stops_fn
        self.normalize_price_fn = normalize_price_fn

        # Fail-safe: if SL/TP cannot be attached, optionally close the position quickly.
        # Default: enabled ("1") to avoid naked exposure.
        env_default = os.getenv("CLOSE_ON_SLTP_FAIL", "1").strip().lower() in ("1", "true", "yes")
        self.close_on_sltp_fail = bool(env_default if close_on_sltp_fail is None else close_on_sltp_fail)

    def send_market_order(
        self,
        request: OrderRequest,
        *,
        max_attempts: int = 3,
        telemetry_hooks: Optional[dict[str, Callable[..., Any]]] = None,
    ) -> OrderResult:
        sent_ts = time.time()
        hooks = telemetry_hooks or {}

        side = str(request.signal)
        if side not in ("Buy", "Sell"):
            return self._fail(request, sent_ts, "bad_side", 0)

        planned_sl = float(request.sl) if float(request.sl) > 0 else 0.0
        planned_tp = float(request.tp) if float(request.tp) > 0 else 0.0

        # MT5 retcodes
        RC_DONE = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
        RC_DONE_PARTIAL = int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010))
        RC_REQUOTE = int(getattr(mt5, "TRADE_RETCODE_REQUOTE", 10004))
        RC_TIMEOUT = int(getattr(mt5, "TRADE_RETCODE_TIMEOUT", 10012))
        RC_PRICE_CHANGED = int(getattr(mt5, "TRADE_RETCODE_PRICE_CHANGED", 10020))
        RC_OFF_QUOTES = int(getattr(mt5, "TRADE_RETCODE_OFF_QUOTES", 10021))
        RC_INVALID_STOPS = int(getattr(mt5, "TRADE_RETCODE_INVALID_STOPS", 10016))
        RC_INVALID_PRICE = int(getattr(mt5, "TRADE_RETCODE_INVALID_PRICE", 10015))

        last_reason = "unknown"
        last_retcode = 0

        for attempt in range(1, int(max_attempts) + 1):
            try:
                ensure_mt5()

                # Fast terminal health check (avoid useless retries)
                if not self._trade_allowed():
                    last_reason = "trade_not_allowed"
                    break

                with MT5_LOCK:
                    info = mt5.symbol_info(request.symbol)
                if info is None:
                    last_reason = "symbol_info_none"
                    time.sleep(0.12)
                    continue

                # Ensure symbol is selected/visible
                if hasattr(info, "visible") and not bool(getattr(info, "visible", True)):
                    with MT5_LOCK:
                        _ = mt5.symbol_select(request.symbol, True)

                with MT5_LOCK:
                    tick = mt5.symbol_info_tick(request.symbol)
                if tick is None:
                    last_reason = "tick_none"
                    time.sleep(0.12)
                    continue

                req_price = float(tick.ask) if side == "Buy" else float(tick.bid)
                if req_price <= 0.0:
                    last_reason = "bad_req_price"
                    time.sleep(0.12)
                    continue

                if self.normalize_price_fn:
                    try:
                        req_price = float(self.normalize_price_fn(req_price))
                    except Exception:
                        pass

                order_type = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL

                filling = getattr(info, "filling_mode", None)
                if filling not in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
                    filling = mt5.ORDER_FILLING_IOC

                vol = self._normalize_volume(float(request.lot), info)
                sl_s, tp_s = self._sanitize_stops(side, req_price, planned_sl, planned_tp)

                send_ts = time.time()
                log_health.info(
                    "ORDER_SEND | attempt=%s side=%s symbol=%s lot=%.4f req=%.5f sl=%.5f tp=%.5f dev=%s",
                    attempt,
                    side,
                    request.symbol,
                    vol,
                    req_price,
                    sl_s,
                    tp_s,
                    int(request.deviation),
                )

                result = self._send_deal(
                    request.symbol,
                    int(order_type),
                    req_price,
                    vol,
                    sl_s,
                    tp_s,
                    int(filling),
                    request,
                )
                sent_ts = send_ts

                if result is None:
                    last_reason = "order_send_none"
                    time.sleep(0.25)
                    continue

                last_retcode = int(getattr(result, "retcode", 0) or 0)

                # If broker rejected stops/price, fill without stops then attach.
                used_stops = True
                if last_retcode in (RC_INVALID_STOPS, RC_INVALID_PRICE) and (planned_sl > 0.0 or planned_tp > 0.0):
                    used_stops = False
                    log_health.info("ORDER_RETRY_NO_STOPS | retcode=%s", last_retcode)
                    result2 = self._send_deal(
                        request.symbol,
                        int(order_type),
                        req_price,
                        vol,
                        0.0,
                        0.0,
                        int(filling),
                        request,
                    )
                    if result2 is not None:
                        result = result2
                        last_retcode = int(getattr(result2, "retcode", 0) or 0)

                if last_retcode in (RC_DONE, RC_DONE_PARTIAL):
                    fill_ts = time.time()
                    exec_price = float(getattr(result, "price", req_price) or req_price)
                    exec_vol = float(getattr(result, "volume", vol) or vol)
                    slippage = abs(exec_price - req_price)

                    # Attach SL/TP if:
                    # - we had to send without stops, OR
                    # - broker accepted but position still has missing stops
                    if planned_sl > 0.0 or planned_tp > 0.0:
                        attached = self._ensure_sltp(
                            symbol=request.symbol,
                            side=side,
                            exec_price=exec_price,
                            volume=exec_vol,
                            planned_sl=planned_sl,
                            planned_tp=planned_tp,
                            magic=int(request.magic),
                            rc_done=RC_DONE,
                            rc_done_partial=RC_DONE_PARTIAL,
                        )
                        if (not attached) and self.close_on_sltp_fail:
                            self._fail_safe_close(request.symbol, side, exec_vol, int(request.magic))

                    positions_open = 1
                    try:
                        with MT5_LOCK:
                            positions_open = len(mt5.positions_get(symbol=request.symbol) or [])
                    except Exception:
                        positions_open = 1

                    self._safe_hook(hooks, "record_trade", exec_price, float(request.sl), exec_vol, float(request.tp), int(positions_open))
                    self._safe_hook(
                        hooks,
                        "record_execution_metrics",
                        request.order_id,
                        side,
                        float(request.enqueue_time),
                        float(sent_ts),
                        float(fill_ts),
                        float(req_price),
                        float(exec_price),
                        float(slippage),
                    )

                    log_health.info(
                        "ORDER_FILLED | order_id=%s side=%s symbol=%s ret=%s req=%.5f exec=%.5f vol=%.4f slip=%.5f",
                        request.order_id,
                        side,
                        request.symbol,
                        last_retcode,
                        req_price,
                        exec_price,
                        exec_vol,
                        slippage,
                    )

                    return OrderResult(
                        order_id=request.order_id,
                        signal_id=request.signal_id,
                        ok=True,
                        reason="filled",
                        sent_ts=sent_ts,
                        fill_ts=fill_ts,
                        req_price=req_price,
                        exec_price=exec_price,
                        volume=exec_vol,
                        slippage=slippage,
                        retcode=last_retcode,
                    )

                if last_retcode in (RC_REQUOTE, RC_TIMEOUT, RC_PRICE_CHANGED, RC_OFF_QUOTES):
                    last_reason = f"retry_retcode_{last_retcode}"
                    time.sleep(0.45)
                    continue

                last_reason = f"fail_retcode_{last_retcode}"
                break

            except Exception as exc:
                last_reason = f"exception_{type(exc).__name__}"
                log_err.error(
                    "Order process error | attempt=%s err=%s tb=%s",
                    attempt,
                    exc,
                    traceback.format_exc(),
                )
                time.sleep(0.25)

        self._safe_hook(hooks, "record_execution_failure", request.order_id, float(request.enqueue_time), float(sent_ts), last_reason)
        log_err.error(
            "ORDER_FAILED | order_id=%s side=%s symbol=%s reason=%s retcode=%s",
            request.order_id,
            side,
            request.symbol,
            last_reason,
            last_retcode,
        )
        return self._fail(request, sent_ts, last_reason, last_retcode)

    # -------------------- helpers --------------------
    @staticmethod
    def _safe_hook(hooks: dict[str, Callable[..., Any]], name: str, *args: Any) -> None:
        fn = hooks.get(name)
        if not fn:
            return
        try:
            fn(*args)
        except Exception:
            return

    def _fail(self, request: OrderRequest, sent_ts: float, reason: str, retcode: int) -> OrderResult:
        return OrderResult(
            order_id=request.order_id,
            signal_id=request.signal_id,
            ok=False,
            reason=reason,
            sent_ts=sent_ts,
            fill_ts=sent_ts,
            req_price=float(request.price),
            exec_price=0.0,
            volume=float(request.lot),
            slippage=0.0,
            retcode=int(retcode),
        )

    def _normalize_volume(self, vol: float, info: Any) -> float:
        if self.normalize_volume_fn:
            try:
                return float(self.normalize_volume_fn(vol, info))
            except Exception:
                pass

        step = float(getattr(info, "volume_step", 0.01) or 0.01)
        vmin = float(getattr(info, "volume_min", step) or step)
        vmax = float(getattr(info, "volume_max", vol) or vol)

        if step <= 0:
            step = 0.01
        vol = max(vmin, min(vmax, float(vol)))

        k = math.floor(vol / step)
        vol2 = k * step
        decimals = max(0, min(8, len(str(step).split(".")[-1])))
        vol2 = round(vol2, decimals)

        return float(vol2 if vol2 >= vmin else vmin)

    def _sanitize_stops(self, side: str, req_price: float, sl: float, tp: float) -> tuple[float, float]:
        if self.sanitize_stops_fn:
            try:
                sl2, tp2 = self.sanitize_stops_fn(side, req_price, sl, tp)
                return float(sl2 or 0.0), float(tp2 or 0.0)
            except Exception:
                return float(sl or 0.0), float(tp or 0.0)
        return float(sl or 0.0), float(tp or 0.0)

    @staticmethod
    def _trade_allowed() -> bool:
        try:
            with MT5_LOCK:
                term = mt5.terminal_info()
            if not term:
                return False
            return bool(getattr(term, "connected", False) and getattr(term, "trade_allowed", False))
        except Exception:
            return False

    @staticmethod
    def _send_deal(
        symbol: str,
        order_type: int,
        price: float,
        volume: float,
        sl: float,
        tp: float,
        filling: int,
        request: OrderRequest,
    ) -> Any:
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": int(order_type),
            "price": float(price),
            "sl": float(sl) if float(sl) > 0 else 0.0,
            "tp": float(tp) if float(tp) > 0 else 0.0,
            "deviation": int(request.deviation),
            "magic": int(request.magic),
            "comment": str(request.comment),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": int(filling),
        }
        with MT5_LOCK:
            return mt5.order_send(req)

    def _ensure_sltp(
        self,
        *,
        symbol: str,
        side: str,
        exec_price: float,
        volume: float,
        planned_sl: float,
        planned_tp: float,
        magic: int,
        rc_done: int,
        rc_done_partial: int,
    ) -> bool:
        """Attach SL/TP if missing or invalid. Returns True if SL/TP confirmed set."""
        if planned_sl <= 0.0 and planned_tp <= 0.0:
            return True

        sl2, tp2 = self._sanitize_stops(side, exec_price, planned_sl, planned_tp)
        if sl2 <= 0.0 and tp2 <= 0.0:
            return False

        # Find freshest matching position by magic + side + volume proximity + time proximity.
        ticket = self._find_position_ticket(symbol=symbol, side=side, volume=volume, magic=magic)
        if not ticket:
            log_err.error("SLTP: position_not_found | symbol=%s side=%s vol=%.4f", symbol, side, volume)
            return False

        # If already set -> done
        try:
            with MT5_LOCK:
                pos_list = mt5.positions_get(ticket=int(ticket)) or []
            if pos_list:
                p = pos_list[0]
                cur_sl = float(getattr(p, "sl", 0.0) or 0.0)
                cur_tp = float(getattr(p, "tp", 0.0) or 0.0)
                if (planned_sl > 0.0 and cur_sl > 0.0) or (planned_tp > 0.0 and cur_tp > 0.0):
                    return True
        except Exception:
            pass

        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": symbol,
            "sl": float(sl2) if float(sl2) > 0 else 0.0,
            "tp": float(tp2) if float(tp2) > 0 else 0.0,
            "magic": int(magic),
            "comment": "xau_scalper_sltp",
        }

        for i in range(1, 4):
            with MT5_LOCK:
                r = mt5.order_send(req)
            rc = int(getattr(r, "retcode", 0) or 0) if r is not None else 0
            log_health.info("SLTP_SET | try=%s ticket=%s rc=%s sl=%.5f tp=%.5f", i, ticket, rc, sl2, tp2)
            if r is not None and rc in (rc_done, rc_done_partial):
                return True
            time.sleep(0.25)

        log_err.error("SLTP_SET_FAILED | ticket=%s symbol=%s", ticket, symbol)
        return False

    @staticmethod
    def _find_position_ticket(*, symbol: str, side: str, volume: float, magic: int) -> Optional[int]:
        try:
            deadline = time.time() + 2.0
            want_type = mt5.POSITION_TYPE_BUY if side == "Buy" else mt5.POSITION_TYPE_SELL

            while time.time() < deadline:
                with MT5_LOCK:
                    positions = mt5.positions_get(symbol=symbol) or []

                candidates: list[tuple[float, float, int]] = []
                for p in positions:
                    try:
                        if int(getattr(p, "magic", 0) or 0) != int(magic):
                            continue
                        if int(getattr(p, "type", -1) or -1) != int(want_type):
                            continue
                        pv = float(getattr(p, "volume", 0.0) or 0.0)
                        pt = float(getattr(p, "time", 0.0) or 0.0)
                        tk = int(getattr(p, "ticket", 0) or 0)
                        if tk <= 0 or pv <= 0.0:
                            continue

                        # Prefer closest volume + newest
                        dv = abs(pv - float(volume))
                        candidates.append((pt, dv, tk))
                    except Exception:
                        continue

                if candidates:
                    candidates.sort(key=lambda x: (-x[0], x[1]))
                    return int(candidates[0][2])

                time.sleep(0.12)

        except Exception:
            return None

        return None

    @staticmethod
    def _fail_safe_close(symbol: str, side: str, volume: float, magic: int) -> None:
        """Emergency close all matching positions for this symbol+magic+side.

        Only triggers when SL/TP attachment fails and CLOSE_ON_SLTP_FAIL=1.
        """
        try:
            want_type = mt5.POSITION_TYPE_BUY if side == "Buy" else mt5.POSITION_TYPE_SELL
            with MT5_LOCK:
                positions = mt5.positions_get(symbol=symbol) or []

            for p in positions:
                try:
                    if int(getattr(p, "magic", 0) or 0) != int(magic):
                        continue
                    if int(getattr(p, "type", -1) or -1) != int(want_type):
                        continue
                    ticket = int(getattr(p, "ticket", 0) or 0)
                    pv = float(getattr(p, "volume", 0.0) or 0.0)
                    if ticket <= 0 or pv <= 0.0:
                        continue

                    # Close position by opposite deal
                    with MT5_LOCK:
                        tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        continue

                    price = float(tick.bid) if side == "Buy" else float(tick.ask)
                    close_type = mt5.ORDER_TYPE_SELL if side == "Buy" else mt5.ORDER_TYPE_BUY
                    req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "position": int(ticket),
                        "volume": float(pv),
                        "type": int(close_type),
                        "price": float(price),
                        "deviation": 50,
                        "magic": int(magic),
                        "comment": "failsafe_close_sltp",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    with MT5_LOCK:
                        r = mt5.order_send(req)
                    log_err.error("FAILSAFE_CLOSE | ticket=%s rc=%s", ticket, int(getattr(r, "retcode", 0) or 0))

                except Exception:
                    continue
        except Exception:
            return


__all__ = ["OrderRequest", "OrderResult", "OrderExecutor"]
