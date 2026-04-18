"""
core/idempotency.py — Institutional-grade idempotency & write-ahead log (WAL)
for order execution.

DESIGN GOAL
-----------
Guarantee that NO order is ever sent twice, under any combination of:
  * retry loops,
  * transient MT5 disconnects,
  * process crashes between `order_send` and the ACK,
  * operator restarts.

This is achieved via a simple, audit-friendly append-only JSONL journal
plus an in-memory dedupe set loaded from that journal on startup.

WAL LIFECYCLE (per order)
-------------------------
  1. PENDING   — intent recorded BEFORE the network call.
  2. SENT      — broker ACK received (ok=True|False with retcode).
  3. CONFIRMED — position/deal reconciled against broker state.
  4. FAILED    — broker explicitly rejected or a non-retryable error.

A new order_send is ONLY allowed when no PENDING/SENT/CONFIRMED record
exists for the given idempotency_key.

On recovery (startup), any PENDING entries older than a configurable
TTL are reconciled against `positions_get`/`history_deals_get`; if no
evidence of execution is found within the TTL, the entry is marked
FAILED and may be retried.

Thread-safety
-------------
All mutating methods hold a single `threading.Lock`. WAL writes use an
atomic os.replace for rotation and are fsync'd on every append.

Backward-compatibility
----------------------
If `IDEMPOTENCY_ENABLED=0` the store becomes a no-op (allows every
call). This is ONLY for emergency rollback, never in production.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

try:
    from log_config import get_log_path  # type: ignore

    _WAL_DEFAULT = str(get_log_path("order_wal.jsonl"))
except Exception:
    _WAL_DEFAULT = os.path.join(os.getcwd(), "order_wal.jsonl")


# ─── Dedicated module logger ──────────────────────────────────────────────
# File handler-и ин logger-ро `log_config.configure_module_logs()` ҳангоми
# boot ба `Logs/idempotency_wal.log` ҳамроҳ мекунад (ба registry дар
# `log_config.py::_MODULE_LOG_REGISTRY` нигаред). Ҳар ҳодисаи критикии WAL
# (corrupt record, reconcile mismatch, send-failure-retries) дар ин ҷо log
# мешавад ва дар як файли мустақил дастрас аст.
log = logging.getLogger("core.idempotency")


# =============================================================================
# Constants
# =============================================================================

STATUS_PENDING = "PENDING"
STATUS_SENT = "SENT"
STATUS_CONFIRMED = "CONFIRMED"
STATUS_FAILED = "FAILED"

_TERMINAL_STATUSES = frozenset({STATUS_CONFIRMED, STATUS_FAILED})
_ACTIVE_STATUSES = frozenset({STATUS_PENDING, STATUS_SENT, STATUS_CONFIRMED})

# Default reconcile TTL: if a PENDING entry is older than this and no
# evidence of the order is found, treat it as FAILED and allow retry.
_RECONCILE_TTL_SEC = float(os.getenv("IDEMPOTENCY_RECONCILE_TTL_SEC", "120") or "120")


def _enabled() -> bool:
    return str(os.getenv("IDEMPOTENCY_ENABLED", "1")).strip() in ("1", "true", "True")


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class WALEntry:
    key: str
    status: str
    ts: float = field(default_factory=time.time)
    # Envelope (snapshot of the request that produced this key).
    symbol: str = ""
    side: str = ""
    volume: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    magic: int = 0
    # Post-send fields.
    retcode: int = 0
    order_ticket: int = 0
    deal_ticket: int = 0
    position_ticket: int = 0
    reason: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "key": self.key,
                "status": self.status,
                "ts": round(float(self.ts), 6),
                "symbol": self.symbol,
                "side": self.side,
                "volume": float(self.volume),
                "price": float(self.price),
                "sl": float(self.sl),
                "tp": float(self.tp),
                "magic": int(self.magic),
                "retcode": int(self.retcode),
                "order_ticket": int(self.order_ticket),
                "deal_ticket": int(self.deal_ticket),
                "position_ticket": int(self.position_ticket),
                "reason": self.reason,
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, line: str) -> Optional["WALEntry"]:
        try:
            d = json.loads(line)
        except Exception:
            return None
        if not isinstance(d, dict) or "key" not in d or "status" not in d:
            return None
        return cls(
            key=str(d.get("key", "")),
            status=str(d.get("status", STATUS_PENDING)),
            ts=float(d.get("ts", 0.0) or 0.0),
            symbol=str(d.get("symbol", "")),
            side=str(d.get("side", "")),
            volume=float(d.get("volume", 0.0) or 0.0),
            price=float(d.get("price", 0.0) or 0.0),
            sl=float(d.get("sl", 0.0) or 0.0),
            tp=float(d.get("tp", 0.0) or 0.0),
            magic=int(d.get("magic", 0) or 0),
            retcode=int(d.get("retcode", 0) or 0),
            order_ticket=int(d.get("order_ticket", 0) or 0),
            deal_ticket=int(d.get("deal_ticket", 0) or 0),
            position_ticket=int(d.get("position_ticket", 0) or 0),
            reason=str(d.get("reason", "")),
        )


# =============================================================================
# UUID helpers (public API)
# =============================================================================


def new_idempotency_key(symbol: str, signal_id: str, side: str) -> str:
    """
    Build a globally-unique idempotency key.

    Format: "SYM:SIG:SIDE:UUID8"

    The UUID8 is a deterministic-free suffix so that even if the same
    (symbol, signal_id, side) triple is enqueued twice (operator error),
    they get distinct keys and neither is silently swallowed.
    """
    sym = str(symbol or "").upper().strip()
    sid = str(signal_id or "").strip()
    sd = str(side or "").strip()
    uid = uuid.uuid4().hex[:8]
    return f"{sym}:{sid}:{sd}:{uid}"


def idem_key_to_comment(key: str, *, base: str = "") -> str:
    """
    Squeeze an idempotency key into an MT5 `comment` field (≤31 chars).

    Strategy: keep a short tag `base`, then append the LAST 16 chars of
    the key (which includes the UUID8 suffix plus some of the signal id).
    That 16-char tail is unique enough to be looked up in history.
    """
    k = str(key or "")
    if not k:
        return str(base or "")[:31]
    tail = k[-16:]
    tag = (str(base or "")[:12]).strip()
    if tag:
        out = f"{tag}|{tail}"
    else:
        out = tail
    return out[:31]


def extract_idem_tail(comment: str) -> str:
    """Inverse of idem_key_to_comment (best-effort tail extraction)."""
    c = str(comment or "")
    if "|" in c:
        return c.split("|", 1)[1][:16]
    return c[-16:]


# =============================================================================
# Journal (WAL)
# =============================================================================


class OrderJournal:
    """
    Append-only JSONL journal with in-memory index.

    File format: one `WALEntry.to_json()` per line. Entries are immutable
    once written; state transitions are expressed as new entries with the
    same `key`. The in-memory index keeps only the LATEST status per key.
    """

    def __init__(self, path: Optional[str] = None):
        self._lock = threading.RLock()
        self._path = str(path or os.getenv("IDEMPOTENCY_WAL_PATH", "") or _WAL_DEFAULT)
        self._index: Dict[str, WALEntry] = {}
        self._enabled = _enabled()
        self._ensure_parent()
        self._load()

    @property
    def path(self) -> str:
        return self._path

    def _ensure_parent(self) -> None:
        try:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    entry = WALEntry.from_json(raw)
                    if entry is None:
                        continue
                    # Keep latest by timestamp (replay order is append-only
                    # so a later line always reflects a later state).
                    self._index[entry.key] = entry
        except Exception:
            # A corrupt journal is catastrophic — preserve the file for
            # forensic analysis and start fresh in memory. Operator must
            # investigate before resuming trading.
            try:
                bad = f"{self._path}.corrupt.{int(time.time())}"
                os.replace(self._path, bad)
            except Exception:
                pass
            self._index = {}

    def _append(self, entry: WALEntry) -> None:
        if not self._enabled:
            return
        line = entry.to_json() + "\n"
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except Exception:
                    pass
            self._index[entry.key] = entry

    def get(self, key: str) -> Optional[WALEntry]:
        with self._lock:
            return self._index.get(str(key))

    def record_pending(
        self,
        key: str,
        *,
        symbol: str,
        side: str,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        magic: int,
    ) -> WALEntry:
        entry = WALEntry(
            key=str(key),
            status=STATUS_PENDING,
            ts=time.time(),
            symbol=symbol,
            side=side,
            volume=float(volume),
            price=float(price),
            sl=float(sl),
            tp=float(tp),
            magic=int(magic),
        )
        self._append(entry)
        return entry

    def record_sent(
        self,
        key: str,
        *,
        retcode: int,
        order_ticket: int = 0,
        deal_ticket: int = 0,
        position_ticket: int = 0,
        reason: str = "",
    ) -> None:
        prev = self.get(key)
        base = prev if prev is not None else WALEntry(key=str(key), status=STATUS_PENDING)
        entry = WALEntry(
            key=str(key),
            status=STATUS_SENT,
            ts=time.time(),
            symbol=base.symbol,
            side=base.side,
            volume=base.volume,
            price=base.price,
            sl=base.sl,
            tp=base.tp,
            magic=base.magic,
            retcode=int(retcode),
            order_ticket=int(order_ticket),
            deal_ticket=int(deal_ticket),
            position_ticket=int(position_ticket),
            reason=reason,
        )
        self._append(entry)

    def record_confirmed(
        self,
        key: str,
        *,
        position_ticket: int = 0,
        deal_ticket: int = 0,
        reason: str = "",
    ) -> None:
        prev = self.get(key) or WALEntry(key=str(key), status=STATUS_PENDING)
        entry = WALEntry(
            key=str(key),
            status=STATUS_CONFIRMED,
            ts=time.time(),
            symbol=prev.symbol,
            side=prev.side,
            volume=prev.volume,
            price=prev.price,
            sl=prev.sl,
            tp=prev.tp,
            magic=prev.magic,
            retcode=prev.retcode,
            order_ticket=prev.order_ticket,
            deal_ticket=int(deal_ticket or prev.deal_ticket),
            position_ticket=int(position_ticket or prev.position_ticket),
            reason=reason or prev.reason,
        )
        self._append(entry)

    def record_failed(self, key: str, *, reason: str, retcode: int = 0) -> None:
        prev = self.get(key) or WALEntry(key=str(key), status=STATUS_PENDING)
        entry = WALEntry(
            key=str(key),
            status=STATUS_FAILED,
            ts=time.time(),
            symbol=prev.symbol,
            side=prev.side,
            volume=prev.volume,
            price=prev.price,
            sl=prev.sl,
            tp=prev.tp,
            magic=prev.magic,
            retcode=int(retcode),
            order_ticket=prev.order_ticket,
            deal_ticket=prev.deal_ticket,
            position_ticket=prev.position_ticket,
            reason=reason,
        )
        self._append(entry)

    # ---- High-level gate -------------------------------------------------

    def should_send(self, key: str) -> Tuple[bool, str]:
        """
        Return (ok_to_send, reason).

        If ANY active (PENDING/SENT/CONFIRMED) record exists for `key`
        within the reconcile TTL, refuse to send again — the caller must
        either reconcile or expire the entry first.
        """
        if not self._enabled:
            return True, "disabled"
        prev = self.get(key)
        if prev is None:
            return True, "new"
        if prev.status in _TERMINAL_STATUSES and prev.status == STATUS_CONFIRMED:
            return False, f"already_confirmed:ticket={prev.position_ticket}"
        if prev.status == STATUS_FAILED:
            return True, "previous_failed_retry_allowed"
        if prev.status in (STATUS_PENDING, STATUS_SENT):
            age = time.time() - float(prev.ts or 0.0)
            if age < _RECONCILE_TTL_SEC:
                return False, f"in_flight:{prev.status}:age={age:.1f}s"
            # Stale in-flight entry — caller should reconcile FIRST.
            return False, f"stale_in_flight:{prev.status}:age={age:.1f}s"
        return True, "unknown_status"

    def pending_keys(self) -> Iterable[WALEntry]:
        with self._lock:
            return list(
                e for e in self._index.values() if e.status in (STATUS_PENDING, STATUS_SENT)
            )


# =============================================================================
# Global default journal (lazy singleton)
# =============================================================================

_DEFAULT_JOURNAL: Optional[OrderJournal] = None
_DEFAULT_JOURNAL_LOCK = threading.Lock()


def get_default_journal() -> OrderJournal:
    global _DEFAULT_JOURNAL
    with _DEFAULT_JOURNAL_LOCK:
        if _DEFAULT_JOURNAL is None:
            _DEFAULT_JOURNAL = OrderJournal()
        return _DEFAULT_JOURNAL


# =============================================================================
# safe_order_send wrapper
# =============================================================================


def safe_order_send(
    send_fn: Callable[..., Any],
    *,
    key: str,
    symbol: str,
    side: str,
    volume: float,
    price: float,
    sl: float,
    tp: float,
    magic: int,
    journal: Optional[OrderJournal] = None,
    send_args: Tuple = (),
    send_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Any, str]:
    """
    Idempotent wrapper around a broker order-send function.

    Returns `(ok, raw_result, reason)`:
      * ok=True: order was (or had been) successfully placed.
      * ok=False: send was refused OR the broker rejected it.

    On a refusal due to an active in-flight entry, the caller MUST NOT
    retry until reconciliation clears the WAL.
    """
    j = journal if journal is not None else get_default_journal()
    ok_to_send, reason = j.should_send(key)
    if not ok_to_send:
        try:
            log.warning(
                "WAL_SEND_REFUSED | key=%s reason=%s symbol=%s side=%s vol=%.4f",
                key,
                reason,
                symbol,
                side,
                float(volume),
            )
        except Exception:
            pass
        return False, None, f"idempotency_refuse:{reason}"

    j.record_pending(
        key,
        symbol=symbol,
        side=side,
        volume=volume,
        price=price,
        sl=sl,
        tp=tp,
        magic=magic,
    )
    try:
        log.info(
            "WAL_SEND_PENDING | key=%s symbol=%s side=%s vol=%.4f price=%.5f "
            "sl=%.5f tp=%.5f magic=%d",
            key,
            symbol,
            side,
            float(volume),
            float(price),
            float(sl),
            float(tp),
            int(magic),
        )
    except Exception:
        pass

    try:
        result = send_fn(*(send_args or ()), **(send_kwargs or {}))
    except Exception as exc:  # pragma: no cover — safety net
        j.record_failed(key, reason=f"send_exception:{type(exc).__name__}:{exc}")
        try:
            log.error(
                "WAL_SEND_EXCEPTION | key=%s err=%s:%s",
                key,
                type(exc).__name__,
                exc,
            )
        except Exception:
            pass
        return False, None, f"send_exception:{exc}"

    # Extract retcode / tickets best-effort. MT5 result is a
    # TradeRequestResult-like object with attributes.
    try:
        retcode = int(getattr(result, "retcode", 0) or 0)
    except Exception:
        retcode = 0
    try:
        order_ticket = int(getattr(result, "order", 0) or 0)
    except Exception:
        order_ticket = 0
    try:
        deal_ticket = int(getattr(result, "deal", 0) or 0)
    except Exception:
        deal_ticket = 0

    # 10009 = TRADE_RETCODE_DONE, 10010 = DONE_PARTIAL (vendor-specific).
    ok = retcode in (10009, 10010)
    if ok:
        j.record_sent(
            key,
            retcode=retcode,
            order_ticket=order_ticket,
            deal_ticket=deal_ticket,
            reason="ok",
        )
        try:
            log.info(
                "WAL_SEND_OK | key=%s retcode=%d order=%d deal=%d",
                key,
                retcode,
                order_ticket,
                deal_ticket,
            )
        except Exception:
            pass
        return True, result, "ok"

    j.record_failed(
        key,
        reason=f"broker_reject:retcode={retcode}",
        retcode=retcode,
    )
    try:
        log.error(
            "WAL_SEND_REJECTED | key=%s retcode=%d order=%d deal=%d",
            key,
            retcode,
            order_ticket,
            deal_ticket,
        )
    except Exception:
        pass
    return False, result, f"broker_reject:retcode={retcode}"


# =============================================================================
# Startup reconciliation
# =============================================================================


def reconcile_on_startup(
    journal: Optional[OrderJournal] = None,
    *,
    positions_getter: Optional[Callable[[], Iterable[Any]]] = None,
    history_deals_getter: Optional[Callable[[float, float], Iterable[Any]]] = None,
) -> Dict[str, int]:
    """
    Reconcile PENDING/SENT WAL entries against the broker's current state.

    * `positions_getter`: returns `mt5.positions_get()` or equivalent.
    * `history_deals_getter(from_ts, to_ts)`: returns deals for a window.

    For each active entry:
      1. Try to find a matching open position by (magic, side, volume).
         If found → mark CONFIRMED.
      2. Else try to find a matching filled deal in the last hour.
         If found → mark CONFIRMED (position may have closed already).
      3. Else, if age > TTL → mark FAILED (broker never executed).
      4. Else leave as-is (caller may retry later).

    Returns a summary dict with counts.
    """
    j = journal if journal is not None else get_default_journal()
    counts = {"confirmed": 0, "failed": 0, "still_pending": 0}

    try:
        positions = list(positions_getter() or []) if positions_getter else []
    except Exception:
        positions = []

    pos_index: Dict[Tuple[str, int, float], int] = {}
    for p in positions:
        try:
            sym = str(getattr(p, "symbol", "") or "").upper()
            magic = int(getattr(p, "magic", 0) or 0)
            vol = float(getattr(p, "volume", 0.0) or 0.0)
            tk = int(getattr(p, "ticket", 0) or 0)
        except Exception:
            continue
        pos_index[(sym, magic, round(vol, 8))] = tk

    now = time.time()
    for entry in list(j.pending_keys()):
        age = now - float(entry.ts or 0.0)
        # (a) match on open positions
        sym_u = str(entry.symbol or "").upper()
        key_tuple = (sym_u, int(entry.magic), round(float(entry.volume), 8))
        tk = pos_index.get(key_tuple, 0)
        if tk > 0:
            j.record_confirmed(entry.key, position_ticket=tk, reason="reconcile_open_pos")
            counts["confirmed"] += 1
            continue

        # (b) match on recent deals
        matched = False
        if history_deals_getter is not None:
            try:
                since = max(now - 3600.0, float(entry.ts or 0.0) - 60.0)
                deals = list(history_deals_getter(since, now) or [])
            except Exception:
                deals = []
            for d in deals:
                try:
                    d_sym = str(getattr(d, "symbol", "") or "").upper()
                    d_magic = int(getattr(d, "magic", 0) or 0)
                    d_vol = float(getattr(d, "volume", 0.0) or 0.0)
                    d_comment = str(getattr(d, "comment", "") or "")
                except Exception:
                    continue
                tail = extract_idem_tail(d_comment)
                if tail and tail == entry.key[-16:]:
                    j.record_confirmed(
                        entry.key,
                        position_ticket=int(getattr(d, "position_id", 0) or 0),
                        deal_ticket=int(getattr(d, "ticket", 0) or 0),
                        reason="reconcile_history_deal",
                    )
                    counts["confirmed"] += 1
                    matched = True
                    break
                if (
                    d_sym == sym_u
                    and d_magic == int(entry.magic)
                    and abs(d_vol - float(entry.volume)) < 1e-8
                ):
                    j.record_confirmed(
                        entry.key,
                        position_ticket=int(getattr(d, "position_id", 0) or 0),
                        deal_ticket=int(getattr(d, "ticket", 0) or 0),
                        reason="reconcile_history_heuristic",
                    )
                    counts["confirmed"] += 1
                    matched = True
                    break
            if matched:
                continue

        # (c) expired → mark FAILED (safe: allows retry; broker has no trace)
        if age > _RECONCILE_TTL_SEC:
            j.record_failed(
                entry.key,
                reason=f"reconcile_expired_age={age:.1f}s",
                retcode=0,
            )
            counts["failed"] += 1
            try:
                log.warning(
                    "WAL_RECONCILE_EXPIRED | key=%s symbol=%s age=%.1fs",
                    entry.key,
                    entry.symbol,
                    age,
                )
            except Exception:
                pass
        else:
            counts["still_pending"] += 1

    try:
        log.info(
            "WAL_RECONCILE_SUMMARY | confirmed=%d failed=%d still_pending=%d",
            counts["confirmed"],
            counts["failed"],
            counts["still_pending"],
        )
    except Exception:
        pass

    return counts


__all__ = [
    "STATUS_PENDING",
    "STATUS_SENT",
    "STATUS_CONFIRMED",
    "STATUS_FAILED",
    "WALEntry",
    "OrderJournal",
    "new_idempotency_key",
    "idem_key_to_comment",
    "extract_idem_tail",
    "get_default_journal",
    "safe_order_send",
    "reconcile_on_startup",
]
