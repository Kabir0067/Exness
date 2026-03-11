from __future__ import annotations

import builtins
import queue
import time
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from mt5_client import mt5_async_call

from .logging_setup import log_err, log_health
from .models import AssetCandidate, OrderIntent

if TYPE_CHECKING:
    from .engine import MultiAssetTradingEngine


class ExecutionManager:
    """Owns enqueue-time order gating and queue dispatch semantics."""

    def __init__(self, engine: "MultiAssetTradingEngine") -> None:
        self._e = engine

    def enqueue_order(
        self,
        cand: AssetCandidate,
        *,
        order_index: int = 0,
        order_count: int = 1,
        lot_override: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        e = self._e
        now = time.time()

        if cand.asset == "XAU":
            risk = e._xau.risk if e._xau else None
            cfg = e._xau_cfg
        else:
            risk = e._btc.risk if e._btc else None
            cfg = e._btc_cfg

        try:
            fn = getattr(risk, "update_phase", None)
            if callable(fn):
                fn()
        except Exception:
            pass

        try:
            fn = getattr(risk, "requires_hard_stop", None)
            if callable(fn) and bool(fn()):
                log_health.info("ENQUEUE_SKIP | asset=%s reason=hard_stop", cand.asset)
                return False, None
        except Exception:
            pass

        phase = e._get_phase(risk)
        if phase == "C":
            log_health.warning(
                "PHASE_C_SHADOW | asset=%s signal=%s conf=%.2f lot=%.4f sl=%.2f tp=%.2f | BLOCKED - Shadow Mode Active",
                cand.asset,
                cand.signal,
                cand.confidence,
                cand.lot,
                cand.sl,
                cand.tp,
            )
            if e._signal_notifier:
                try:
                    tick = mt5_async_call(
                        "symbol_info_tick",
                        cand.symbol,
                        timeout=0.35,
                        default=None,
                    )
                    current_price = float(tick.ask if cand.signal == "Buy" else tick.bid) if tick else 0.0
                    shadow_alert = {
                        "type": "PHASE_C_SHADOW",
                        "asset": cand.asset,
                        "symbol": cand.symbol,
                        "signal": cand.signal,
                        "confidence": cand.confidence,
                        "lot": cand.lot,
                        "sl": cand.sl,
                        "tp": cand.tp,
                        "price": current_price,
                        "blocked": True,
                        "reason": "Phase C - Daily Risk Limit Reached",
                        "message": (
                            f"вљ пёЏ [PHASE C вЂ” Shadow Trade] РЎРёРіРЅР°Р»Рё С‚Р°СЃРґРёТ›С€СѓРґР°Рё {cand.signal} Р±Р°СЂРѕРё {cand.symbol}.\n"
                            f"РќР°СЂС…: {current_price:.2f} | Р‘РѕРІР°СЂУЈ: {cand.confidence:.0f}%\n"
                            f"(РЎР°РІРґРѕ Р±Рѕ Р»РёРјРёС‚Рё СЂРёСЃРє РјР°РЅСЉ С€СѓРґ)\n"
                            f"ТІР°Т·РјРё СЌТіС‚РёРјРѕР»УЈ: {cand.lot:.2f} | SL: {cand.sl:.2f} | TP: {cand.tp:.2f}"
                        ),
                    }
                    e._signal_notifier(cand.asset, shadow_alert)
                except Exception as exc:
                    log_health.error("PHASE_C_SHADOW notification error: %s", exc)
            return False, None

        last_close_ts = e._last_trade_close_ts.get(cand.asset, 0.0)
        time_since_close = now - last_close_ts
        if time_since_close < e._trade_cooldown_sec and last_close_ts > 0:
            e._cooldown_blocked_count[cand.asset] = e._cooldown_blocked_count.get(cand.asset, 0) + 1
            if e._cooldown_blocked_count[cand.asset] % 10 == 1:
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=trade_cooldown remaining=%.0fs total_blocked=%d",
                    cand.asset,
                    e._trade_cooldown_sec - time_since_close,
                    e._cooldown_blocked_count[cand.asset],
                )
            return False, None

        open_positions = e._asset_open_positions(cand.asset)
        max_positions = e._asset_max_positions(cand.asset)
        if max_positions > 0 and open_positions >= max_positions:
            log_health.info(
                "ENQUEUE_SKIP | asset=%s reason=max_positions open=%d max=%d",
                cand.asset,
                open_positions,
                max_positions,
            )
            return False, None

        acc_info = mt5_async_call("account_info", timeout=0.4, default=None)
        equity = getattr(acc_info, "equity", 0.0) if acc_info else 0.0
        leverage = getattr(acc_info, "leverage", 100) if acc_info else 100

        c_size = float(getattr(cfg.symbol_params, "contract_size", 100.0))
        proposed_margin = (float(cand.close if hasattr(cand, "close") else 0) * float(cand.lot) * c_size) / float(
            leverage or 1
        )
        if proposed_margin <= 0:
            tick_info = mt5_async_call(
                "symbol_info_tick",
                cand.symbol,
                timeout=0.35,
                default=None,
            )
            p = tick_info.ask if tick_info else 0.0
            proposed_margin = (p * float(cand.lot) * c_size) / float(leverage or 1)

        allowed, adj_lot, block_reason = e._portfolio_risk.check_before_order(
            asset=cand.asset,
            side=cand.signal,
            equity=equity,
            proposed_lot=float(cand.lot),
            proposed_margin=proposed_margin,
        )
        if not allowed:
            log_health.warning("PORTFOLIO_BLOCK | asset=%s reason=%s", cand.asset, block_reason)
            return False, None

        if adj_lot < cand.lot:
            log_health.info(
                "PORTFOLIO_REDUCE | asset=%s lot %.2f -> %.2f (Correlation Guard)",
                cand.asset,
                cand.lot,
                adj_lot,
            )
            lot_override = adj_lot

        last_id, last_sig = e._edge_last_trade.get(cand.asset, ("", "Neutral"))
        if cand.signal in ("Buy", "Sell") and last_id == str(cand.signal_id) and last_sig == str(cand.signal):
            last_log = e._last_log_id.get(cand.asset)
            if last_log != str(cand.signal_id):
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=edge_same_signal_id signal=%s signal_id=%s",
                    cand.asset,
                    cand.signal,
                    cand.signal_id,
                )
                e._last_log_id[cand.asset] = str(cand.signal_id)
            return False, None

        if e._is_duplicate(cand.asset, cand.signal_id, now, max_orders=int(order_count), order_index=int(order_index)):
            last_log = e._last_log_id.get(cand.asset)
            if last_log != str(cand.signal_id):
                log_health.info("ENQUEUE_SKIP | asset=%s reason=duplicate signal_id=%s", cand.asset, cand.signal_id)
                e._last_log_id[cand.asset] = str(cand.signal_id)
            return False, None

        tick = mt5_async_call(
            "symbol_info_tick",
            cand.symbol,
            timeout=0.35,
            default=None,
        )
        info = mt5_async_call(
            "symbol_info",
            cand.symbol,
            timeout=0.5,
            default=None,
        )
        if tick is None:
            log_health.info("ENQUEUE_SKIP | asset=%s reason=tick_missing", cand.asset)
            return False, None
        if info is None:
            log_health.info("ENQUEUE_SKIP | asset=%s reason=symbol_info_missing", cand.asset)
            return False, None

        price = float(tick.ask if cand.signal == "Buy" else tick.bid)
        if price <= 0:
            log_health.info("ENQUEUE_SKIP | asset=%s reason=bad_price", cand.asset)
            return False, None

        current_spread_points = float(tick.ask - tick.bid) / float(info.point) if info.point else 99999.0
        max_spread_pts = float(getattr(cfg, "max_spread_points", 0) or 0)
        if max_spread_pts <= 0:
            max_spread_pts = 2500.0 if cand.asset == "BTC" else 350.0

        if current_spread_points > max_spread_pts:
            sp_key = f"_spread_skip_ts_{cand.asset}"
            sp_last = getattr(e, sp_key, 0.0)
            if (now - sp_last) >= 30.0:
                setattr(e, sp_key, now)
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=spread_too_high spread=%s > max=%s",
                    cand.asset,
                    int(current_spread_points),
                    int(max_spread_pts),
                )
            return False, None

        digits = int(getattr(info, "digits", 0) or 0) if info else 0
        sl_val = float(cand.sl)
        tp_val = float(cand.tp)
        if digits > 0:
            try:
                price = float(builtins.round(price, digits))
                sl_val = float(builtins.round(sl_val, digits))
                tp_val = float(builtins.round(tp_val, digits))
            except Exception:
                pass

        if cand.signal == "Buy":
            if not (sl_val < price < tp_val):
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=bad_sl_tp_relation side=Buy price=%.5f sl=%.5f tp=%.5f",
                    cand.asset,
                    price,
                    sl_val,
                    tp_val,
                )
                return False, None
        else:
            if not (tp_val < price < sl_val):
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=bad_sl_tp_relation side=Sell price=%.5f sl=%.5f tp=%.5f",
                    cand.asset,
                    price,
                    sl_val,
                    tp_val,
                )
                return False, None

        lot_val = float(lot_override) if lot_override is not None else float(cand.lot)
        lot_val = e._apply_asset_lot_cap(cand.asset, lot_val, "enqueue")
        if lot_val <= 0:
            log_health.info("ENQUEUE_SKIP | asset=%s reason=lot_nonpositive", cand.asset)
            return False, None

        base_lot_val = float(lot_val)
        if phase == "B":
            orig = lot_val
            lot_val = max(0.01, lot_val * 0.5)
            log_health.info(
                "PHASE_B_LOT_REDUCE | asset=%s original_lot=%.4f reduced_lot=%.4f",
                cand.asset,
                float(orig),
                float(lot_val),
            )

        gate_reason = "ok"
        if risk and hasattr(risk, "pre_order_gate"):
            gate_fn = getattr(risk, "pre_order_gate", None)
            if callable(gate_fn):
                try:
                    allowed, gated_lot, gate_reason = gate_fn(
                        side=cand.signal,
                        confidence=float(cand.confidence),
                        lot=float(lot_val),
                        entry_price=float(price),
                        sl=float(sl_val),
                        tp=float(tp_val),
                        signal_id=str(cand.signal_id),
                        stage="enqueue",
                        base_lot=float(base_lot_val),
                        phase_snapshot=str(phase),
                    )
                except TypeError:
                    allowed, gated_lot, gate_reason = gate_fn(  # type: ignore[misc]
                        side=cand.signal,
                        confidence=float(cand.confidence),
                        lot=float(lot_val),
                        entry_price=float(price),
                        sl=float(sl_val),
                        tp=float(tp_val),
                    )
                except Exception as exc:
                    log_err.error("ENQUEUE_GATE_EXCEPTION | asset=%s err=%s", cand.asset, exc)
                    return False, None

                if not allowed:
                    log_health.info(
                        "ENQUEUE_SKIP | asset=%s reason=gate_blocked gate_reason=%s",
                        cand.asset,
                        gate_reason,
                    )
                    return False, None
                lot_val = float(gated_lot or lot_val)
                lot_val = e._apply_asset_lot_cap(cand.asset, lot_val, "enqueue_gate")
                if lot_val <= 0.0:
                    log_health.info("ENQUEUE_SKIP | asset=%s reason=gate_lot_nonpositive", cand.asset)
                    return False, None

        order_id = e._next_order_id(cand.asset)
        idem = f"{cand.symbol}:{cand.signal_id}:{cand.signal}"

        intent = OrderIntent(
            asset=cand.asset,
            symbol=cand.symbol,
            signal=cand.signal,
            confidence=float(cand.confidence),
            lot=float(lot_val),
            sl=float(sl_val),
            tp=float(tp_val),
            price=float(price),
            enqueue_time=float(now),
            order_id=str(order_id),
            signal_id=str(cand.signal_id),
            idempotency_key=str(idem),
            risk_manager=risk,
            cfg=cfg,
            base_lot=float(base_lot_val),
            phase_snapshot=str(phase),
        )

        e._order_rm_by_id[str(order_id)] = risk
        e._register_pending_order(intent)

        try:
            e._order_q.put(intent, timeout=0.15)
        except queue.Full:
            e._order_rm_by_id.pop(str(order_id), None)
            e._clear_pending_order(str(order_id))
            if risk and hasattr(risk, "record_execution_failure"):
                try:
                    risk.record_execution_failure(order_id, now, now, "queue_backlog_drop")
                except Exception:
                    pass
            log_health.warning("ENQUEUE_FAIL | asset=%s reason=queue_full order_id=%s", cand.asset, order_id)
            return False, None

        e._mark_seen(cand.asset, cand.signal_id, now)
        e._last_selected_asset = cand.asset
        e._edge_last_trade[cand.asset] = (str(cand.signal_id), str(cand.signal))

        if risk:
            if hasattr(risk, "register_signal_emitted"):
                try:
                    risk.register_signal_emitted()
                except Exception:
                    pass
            if hasattr(risk, "track_signal_survival"):
                try:
                    risk.track_signal_survival(order_id, cand.signal, price, sl_val, tp_val, now, cand.confidence)
                except Exception:
                    pass

        return True, str(order_id)

    def effective_min_conf(self, c: AssetCandidate) -> float:
        e = self._e
        cfg = e._xau_cfg if c.asset == "XAU" else e._btc_cfg
        base = float(getattr(cfg, "min_confidence_signal", 0.55) or 0.55)

        env_global: float = 0.50
        env_asset: float = float(env_global)
        floor = max(0.35, min(0.90, env_asset))

        effective_min = max(floor, base)
        rs = tuple(str(r) for r in (c.reasons or ()))

        if any(r.startswith("early_momentum") for r in rs):
            effective_min = max(floor, effective_min - 0.08)

        return float(max(0.0, min(1.0, effective_min)))

    def is_asset_blocked(self, asset: str) -> bool:
        e = self._e
        asset_u = str(asset or "").upper().strip()
        return asset_u in e._blocked_assets

    def log_blocked_asset_skip(self, asset: str, stage: str) -> None:
        e = self._e
        key = f"{str(asset).upper()}:{stage}"
        now = time.time()
        last = float(e._last_skip_log_ts.get(key, 0.0))
        if now - last < 30.0:
            return
        e._last_skip_log_ts[key] = now
        log_health.info(
            "ASSET_BLOCKED_SKIP | asset=%s stage=%s blocked_assets=%s",
            str(asset).upper(),
            stage,
            ",".join(sorted(set(e._blocked_assets))),
        )

    def candidate_is_tradeable(self, c: AssetCandidate) -> bool:
        if self.is_asset_blocked(c.asset):
            self.log_blocked_asset_skip(c.asset, "candidate_tradeable")
            return False
        if c.signal not in ("Buy", "Sell"):
            return False
        if c.blocked:
            return False
        if c.lot <= 0.0:
            return False
        if c.sl <= 0.0 or c.tp <= 0.0:
            return False

        conf_val = float(c.confidence)
        return bool(conf_val >= self.effective_min_conf(c))

    def asset_open_positions(self, asset: str) -> int:
        e = self._e
        pipe = e._xau if str(asset).upper() == "XAU" else e._btc
        if pipe is None:
            return 0
        try:
            return int(pipe.open_positions())
        except Exception:
            return 0

    def asset_max_positions(self, asset: str) -> int:
        e = self._e
        asset_u = str(asset or "").upper().strip()
        return max(1, int(e._max_open_positions.get(asset_u, 1) or 1))

    def asset_lot_cap(self, asset: str) -> float:
        e = self._e
        asset_u = str(asset or "").upper().strip()
        cfg = e._xau_cfg if asset_u == "XAU" else e._btc_cfg
        broker_max = float(getattr(getattr(cfg, "symbol_params", None), "lot_max", 0.0) or 0.0)
        cap = float(e._max_lot_cap.get(asset_u, broker_max if broker_max > 0.0 else 0.0) or 0.0)
        if broker_max > 0.0:
            if cap <= 0.0:
                cap = broker_max
            else:
                cap = min(cap, broker_max)
        return max(0.0, cap)

    def apply_asset_lot_cap(self, asset: str, lot: float, stage: str) -> float:
        lot_val = float(lot)
        cap = self.asset_lot_cap(asset)
        if cap > 0.0 and lot_val > cap:
            log_health.warning(
                "LOT_CAP_APPLIED | asset=%s stage=%s lot=%.4f cap=%.4f",
                str(asset).upper(),
                stage,
                lot_val,
                cap,
            )
            return float(cap)
        return lot_val

    def orders_for_candidate(self, cand: AssetCandidate) -> int:
        pipe = self._e._xau if cand.asset == "XAU" else self._e._btc
        if not pipe or not pipe.risk:
            return 1

        phase = self._e._get_phase(pipe.risk)
        if phase == "C":
            return 0

        try:
            fn = getattr(pipe.risk, "requires_hard_stop", None)
            if callable(fn) and bool(fn()):
                return 0
        except Exception:
            pass

        return 1

    @staticmethod
    def min_lot(risk: Any) -> float:
        try:
            if risk and hasattr(risk, "_symbol_meta"):
                risk._symbol_meta()  # type: ignore[attr-defined]
            return float(getattr(risk, "_vol_min", 0.01) or 0.01)
        except Exception:
            return 0.01

    def split_lot(self, lot: float, parts: int, risk: Any, cfg: Any) -> List[float]:
        lot = float(lot)
        if int(parts) <= 1:
            return [lot]

        min_lot = self.min_lot(risk)
        split_enabled = bool(getattr(cfg, "multi_order_split_lot", True))
        if not split_enabled:
            if min_lot <= 0 or lot < min_lot:
                return [lot]
            return [lot for _ in range(int(parts))]

        if min_lot <= 0 or (lot / float(parts)) < min_lot:
            return [lot]

        base = lot / float(parts)
        lots = [base for _ in range(int(parts))]
        lots[-1] = float(lot) - (base * float(int(parts) - 1))
        return lots

    def execute_candidates(self, candidates: List[AssetCandidate]) -> None:
        e = self._e
        for selected in candidates:
            if not self.candidate_is_tradeable(selected):
                log_health.info(
                    "FSM_ORDER_SKIP | asset=%s signal=%s conf=%.3f blocked=%s",
                    selected.asset,
                    selected.signal,
                    selected.confidence,
                    selected.blocked,
                )
                continue

            if e._manual_stop:
                log_health.info("FSM_ORDER_SKIP | reason=manual_stop asset=%s", selected.asset)
                continue

            open_positions = self.asset_open_positions(selected.asset)
            max_positions = self.asset_max_positions(selected.asset)
            if max_positions > 0 and open_positions >= max_positions:
                log_health.info(
                    "FSM_ORDER_SKIP | asset=%s reason=max_positions open=%d max=%d",
                    selected.asset,
                    open_positions,
                    max_positions,
                )
                continue

            order_count = self.orders_for_candidate(selected)
            if int(order_count) <= 0:
                continue

            risk = e._xau.risk if selected.asset == "XAU" and e._xau else (e._btc.risk if e._btc else None)
            cfg = e._xau_cfg if selected.asset == "XAU" else e._btc_cfg
            lots = self.split_lot(float(selected.lot), order_count, risk, cfg)
            signal_enqueued = False

            for idx, lot_val in enumerate(lots):
                ok, oid = self.enqueue_order(
                    selected,
                    order_index=int(idx),
                    order_count=int(order_count),
                    lot_override=float(lot_val),
                )
                if ok:
                    if not signal_enqueued:
                        emit_ts = time.time()
                        e._record_signal_emit(selected.asset, emit_ts)

                        last_notified_id, last_notified_sig = e._edge_last_notified.get(
                            selected.asset,
                            ("", "Neutral"),
                        )
                        current_sig_id = str(selected.signal_id)
                        current_sig = str(selected.signal)
                        if (
                            e._signal_notifier
                            and not (
                                last_notified_id == current_sig_id
                                and last_notified_sig == current_sig
                            )
                        ):
                            try:
                                e._signal_notifier(
                                    selected.asset,
                                    {
                                        "type": "LIVE_ENQUEUED",
                                        "asset": selected.asset,
                                        "signal": selected.signal,
                                        "confidence": float(selected.confidence),
                                        "reasons": list(selected.reasons or ()),
                                        "signal_id": current_sig_id,
                                        "order_id": str(oid or ""),
                                        "lot": float(lot_val),
                                        "phase": (e._get_phase(risk) if risk is not None else ""),
                                        "blocked": False,
                                    },
                                )
                            except Exception as exc:
                                log_err.error(
                                    "SIGNAL_NOTIFY_ERROR | asset=%s signal_id=%s err=%s",
                                    selected.asset,
                                    current_sig_id,
                                    exc,
                                )
                        e._edge_last_notified[selected.asset] = (current_sig_id, current_sig)
                        signal_enqueued = True

                    log_health.info(
                        "FSM_ORDER_ENQUEUED | asset=%s signal=%s conf=%.3f lot=%.4f order_id=%s",
                        selected.asset,
                        selected.signal,
                        selected.confidence,
                        float(lot_val),
                        oid,
                    )

    def verification_step(self) -> None:
        e = self._e
        while True:
            try:
                r = e._result_q.get_nowait()
            except queue.Empty:
                break

            try:
                e._resolve_order_result(r)
            except Exception:
                pass
            finally:
                try:
                    e._result_q.task_done()
                except Exception:
                    pass

        e._sync_pending_orders()

        if e._xau:
            e._xau.reconcile_positions()
        if e._btc:
            e._btc.reconcile_positions()
        e._last_reconcile_ts = time.time()

        e._update_portfolio_risk_state()

        open_xau = e._xau.open_positions() if e._xau else 0
        open_btc = e._btc.open_positions() if e._btc else 0
        e._track_position_closures(open_xau, open_btc)
        e._active_asset = e._select_active_asset(open_xau, open_btc)
        e._heartbeat(open_xau, open_btc)
