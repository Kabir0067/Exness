"""
Idempotency and write-ahead logging helpers for order execution.
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


# =============================================================================
# Global Constants
# =============================================================================
STATUS_PENDING = "PENDING"
STATUS_SENT = "SENT"
STATUS_CONFIRMED = "CONFIRMED"
STATUS_FAILED = "FAILED"

_TERMINAL_STATUSES = frozenset({STATUS_CONFIRMED, STATUS_FAILED})
_ACTIVE_STATUSES = frozenset({STATUS_PENDING, STATUS_SENT, STATUS_CONFIRMED})
_RECONCILE_TTL_SEC = float(
    os.getenv("IDEMPOTENCY_RECONCILE_TTL_SEC", "120") or "120"
)
_WAL_COMPACT_BYTES = int(
    os.getenv("IDEMPOTENCY_WAL_COMPACT_BYTES", str(2 * 1024 * 1024))
    or str(2 * 1024 * 1024)
)
_WAL_COMPACT_MIN_KEYS = int(
    os.getenv("IDEMPOTENCY_WAL_COMPACT_MIN_KEYS", "512") or "512"
)


# =============================================================================
# Logging Setup
# =============================================================================
log_idempotency = logging.getLogger("core.idempotency")


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class WALEntry:
    key: str
    status: str
    ts: float = field(default_factory=time.time)
    symbol: str = ""
    side: str = ""
    volume: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    magic: int = 0
    retcode: int = 0
    order_ticket: int = 0
    deal_ticket: int = 0
    position_ticket: int = 0
    reason: str = ""

    def to_json(self) -> str:
        """Serialize the WAL entry to compact JSON."""
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
        """Parse a WAL entry from a JSON line."""
        try:
            payload = json.loads(line)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        if "key" not in payload or "status" not in payload:
            return None

        return cls(
            key=str(payload.get("key", "")),
            status=str(payload.get("status", STATUS_PENDING)),
            ts=float(payload.get("ts", 0.0) or 0.0),
            symbol=str(payload.get("symbol", "")),
            side=str(payload.get("side", "")),
            volume=float(payload.get("volume", 0.0) or 0.0),
            price=float(payload.get("price", 0.0) or 0.0),
            sl=float(payload.get("sl", 0.0) or 0.0),
            tp=float(payload.get("tp", 0.0) or 0.0),
            magic=int(payload.get("magic", 0) or 0),
            retcode=int(payload.get("retcode", 0) or 0),
            order_ticket=int(payload.get("order_ticket", 0) or 0),
            deal_ticket=int(payload.get("deal_ticket", 0) or 0),
            position_ticket=int(payload.get("position_ticket", 0) or 0),
            reason=str(payload.get("reason", "")),
        )


# =============================================================================
# Public API
# =============================================================================
def new_idempotency_key(symbol: str, signal_id: str, side: str) -> str:
    """Build a unique idempotency key for an order request."""
    normalized_symbol = str(symbol or "").upper().strip()
    normalized_signal_id = str(signal_id or "").strip()
    normalized_side = str(side or "").strip()
    unique_suffix = uuid.uuid4().hex[:8]
    return (
        f"{normalized_symbol}:{normalized_signal_id}:{normalized_side}:"
        f"{unique_suffix}"
    )


def idem_key_to_comment(key: str, *, base: str = "") -> str:
    """Fit an idempotency key into the MT5 comment field."""
    normalized_key = str(key or "")
    if not normalized_key:
        return str(base or "")[:31]

    tail = normalized_key[-16:]
    base_tag = (str(base or "")[:12]).strip()

    if base_tag:
        comment = f"{base_tag}|{tail}"
    else:
        comment = tail

    return comment[:31]


def extract_idem_tail(comment: str) -> str:
    """Return the lookup tail from an MT5 comment."""
    normalized_comment = str(comment or "")
    if "|" in normalized_comment:
        return normalized_comment.split("|", 1)[1][:16]
    return normalized_comment[-16:]


class OrderJournal:
    """Append-only JSONL journal with an in-memory latest-state index."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._path = str(
            path or os.getenv("IDEMPOTENCY_WAL_PATH", "") or _WAL_DEFAULT
        )
        self._index: Dict[str, WALEntry] = {}
        self._enabled = _enabled()
        self._ensure_parent()
        self._load()

    @property
    def path(self) -> str:
        return self._path

    def get(self, key: str) -> Optional[WALEntry]:
        """Return the latest WAL entry for the key."""
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
        """Record a pending order intent."""
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
        """Record a broker acknowledgement."""
        previous_entry = self.get(key)
        base_entry = (
            previous_entry
            if previous_entry is not None
            else WALEntry(key=str(key), status=STATUS_PENDING)
        )

        entry = WALEntry(
            key=str(key),
            status=STATUS_SENT,
            ts=time.time(),
            symbol=base_entry.symbol,
            side=base_entry.side,
            volume=base_entry.volume,
            price=base_entry.price,
            sl=base_entry.sl,
            tp=base_entry.tp,
            magic=base_entry.magic,
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
        """Record a confirmed broker execution."""
        previous_entry = self.get(key) or WALEntry(
            key=str(key),
            status=STATUS_PENDING,
        )

        entry = WALEntry(
            key=str(key),
            status=STATUS_CONFIRMED,
            ts=time.time(),
            symbol=previous_entry.symbol,
            side=previous_entry.side,
            volume=previous_entry.volume,
            price=previous_entry.price,
            sl=previous_entry.sl,
            tp=previous_entry.tp,
            magic=previous_entry.magic,
            retcode=previous_entry.retcode,
            order_ticket=previous_entry.order_ticket,
            deal_ticket=int(deal_ticket or previous_entry.deal_ticket),
            position_ticket=int(position_ticket or previous_entry.position_ticket),
            reason=reason or previous_entry.reason,
        )
        self._append(entry)

    def record_failed(self, key: str, *, reason: str, retcode: int = 0) -> None:
        """Record a failed order attempt."""
        previous_entry = self.get(key) or WALEntry(
            key=str(key),
            status=STATUS_PENDING,
        )

        entry = WALEntry(
            key=str(key),
            status=STATUS_FAILED,
            ts=time.time(),
            symbol=previous_entry.symbol,
            side=previous_entry.side,
            volume=previous_entry.volume,
            price=previous_entry.price,
            sl=previous_entry.sl,
            tp=previous_entry.tp,
            magic=previous_entry.magic,
            retcode=int(retcode),
            order_ticket=previous_entry.order_ticket,
            deal_ticket=previous_entry.deal_ticket,
            position_ticket=previous_entry.position_ticket,
            reason=reason,
        )
        self._append(entry)

    def should_send(self, key: str) -> Tuple[bool, str]:
        """Return whether the order key is eligible to be sent."""
        if not self._enabled:
            return True, "disabled"

        previous_entry = self.get(key)
        if previous_entry is None:
            return True, "new"

        if (
            previous_entry.status in _TERMINAL_STATUSES
            and previous_entry.status == STATUS_CONFIRMED
        ):
            return False, (
                f"already_confirmed:ticket={previous_entry.position_ticket}"
            )

        if previous_entry.status == STATUS_FAILED:
            return True, "previous_failed_retry_allowed"

        if previous_entry.status in (STATUS_PENDING, STATUS_SENT):
            age_seconds = time.time() - float(previous_entry.ts or 0.0)
            if age_seconds < _RECONCILE_TTL_SEC:
                return False, (
                    f"in_flight:{previous_entry.status}:age={age_seconds:.1f}s"
                )

            return False, (
                f"stale_in_flight:{previous_entry.status}:age={age_seconds:.1f}s"
            )

        return True, "unknown_status"

    def pending_keys(self) -> Iterable[WALEntry]:
        """Return active WAL entries."""
        with self._lock:
            return list(
                entry
                for entry in self._index.values()
                if entry.status in (STATUS_PENDING, STATUS_SENT)
            )

    def _ensure_parent(self) -> None:
        try:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _fsync_parent_dir(self) -> None:
        try:
            dir_fd = os.open(str(Path(self._path).parent), os.O_RDONLY)
        except Exception:
            return

        try:
            os.fsync(dir_fd)
        except Exception:
            pass
        finally:
            try:
                os.close(dir_fd)
            except Exception:
                pass

    def _should_compact_locked(self) -> bool:
        if len(self._index) < max(1, _WAL_COMPACT_MIN_KEYS):
            return False
        try:
            return os.path.getsize(self._path) >= max(1024, _WAL_COMPACT_BYTES)
        except Exception:
            return False

    def _compact_locked(self) -> None:
        tmp_path = f"{self._path}.tmp"
        entries = sorted(
            self._index.values(),
            key=lambda entry: (float(entry.ts or 0.0), str(entry.key)),
        )

        try:
            with open(tmp_path, "w", encoding="utf-8") as file_handle:
                for entry in entries:
                    file_handle.write(entry.to_json() + "\n")
                file_handle.flush()
                try:
                    os.fsync(file_handle.fileno())
                except Exception:
                    pass
            os.replace(tmp_path, self._path)
            self._fsync_parent_dir()
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return

        try:
            corrupt_lines = 0
            with open(self._path, "r", encoding="utf-8") as file_handle:
                for raw_line in file_handle:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue

                    entry = WALEntry.from_json(raw_line)
                    if entry is None:
                        corrupt_lines += 1
                        continue

                    self._index[entry.key] = entry
            if corrupt_lines > 0:
                try:
                    log_idempotency.warning(
                        "WAL_LOAD_SKIPPED_LINES | path=%s corrupt_lines=%s",
                        self._path,
                        corrupt_lines,
                    )
                except Exception:
                    pass
        except Exception:
            try:
                corrupt_path = f"{self._path}.corrupt.{int(time.time())}"
                os.replace(self._path, corrupt_path)
            except Exception:
                pass

            self._index = {}

    def _append(self, entry: WALEntry) -> None:
        if not self._enabled:
            return

        line = entry.to_json() + "\n"

        with self._lock:
            with open(self._path, "a", encoding="utf-8") as file_handle:
                file_handle.write(line)
                file_handle.flush()

                try:
                    os.fsync(file_handle.fileno())
                except Exception:
                    pass

            self._fsync_parent_dir()
            self._index[entry.key] = entry
            if self._should_compact_locked():
                self._compact_locked()


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
    """Wrap an order-send call with WAL-backed idempotency."""
    active_journal = journal if journal is not None else get_default_journal()
    ok_to_send, reason = active_journal.should_send(key)

    if not ok_to_send:
        try:
            log_idempotency.warning(
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

    active_journal.record_pending(
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
        log_idempotency.info(
            "WAL_SEND_PENDING | key=%s symbol=%s side=%s vol=%.4f "
            "price=%.5f sl=%.5f tp=%.5f magic=%d",
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
    except Exception as exc:  # pragma: no cover
        active_journal.record_failed(
            key,
            reason=f"send_exception:{type(exc).__name__}:{exc}",
        )

        try:
            log_idempotency.error(
                "WAL_SEND_EXCEPTION | key=%s err=%s:%s",
                key,
                type(exc).__name__,
                exc,
            )
        except Exception:
            pass

        return False, None, f"send_exception:{exc}"

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

    ok = retcode in (10009, 10010)

    if ok:
        active_journal.record_sent(
            key,
            retcode=retcode,
            order_ticket=order_ticket,
            deal_ticket=deal_ticket,
            reason="ok",
        )

        try:
            log_idempotency.info(
                "WAL_SEND_OK | key=%s retcode=%d order=%d deal=%d",
                key,
                retcode,
                order_ticket,
                deal_ticket,
            )
        except Exception:
            pass

        return True, result, "ok"

    active_journal.record_failed(
        key,
        reason=f"broker_reject:retcode={retcode}",
        retcode=retcode,
    )

    try:
        log_idempotency.error(
            "WAL_SEND_REJECTED | key=%s retcode=%d order=%d deal=%d",
            key,
            retcode,
            order_ticket,
            deal_ticket,
        )
    except Exception:
        pass

    return False, result, f"broker_reject:retcode={retcode}"


def reconcile_on_startup(
    journal: Optional[OrderJournal] = None,
    *,
    positions_getter: Optional[Callable[[], Iterable[Any]]] = None,
    history_deals_getter: Optional[Callable[[float, float], Iterable[Any]]] = None,
) -> Dict[str, int]:
    """Reconcile active WAL entries against broker state."""
    active_journal = journal if journal is not None else get_default_journal()
    counts = {"confirmed": 0, "failed": 0, "still_pending": 0}

    try:
        positions = list(positions_getter() or []) if positions_getter else []
    except Exception:
        positions = []

    position_index: Dict[Tuple[str, int, float], int] = {}

    for position in positions:
        try:
            symbol = str(getattr(position, "symbol", "") or "").upper()
            magic = int(getattr(position, "magic", 0) or 0)
            volume = float(getattr(position, "volume", 0.0) or 0.0)
            ticket = int(getattr(position, "ticket", 0) or 0)
        except Exception:
            continue

        position_index[(symbol, magic, round(volume, 8))] = ticket

    now = time.time()

    for entry in list(active_journal.pending_keys()):
        age_seconds = now - float(entry.ts or 0.0)
        normalized_symbol = str(entry.symbol or "").upper()
        position_key = (
            normalized_symbol,
            int(entry.magic),
            round(float(entry.volume), 8),
        )
        ticket = position_index.get(position_key, 0)

        if ticket > 0:
            active_journal.record_confirmed(
                entry.key,
                position_ticket=ticket,
                reason="reconcile_open_pos",
            )
            counts["confirmed"] += 1
            continue

        matched = False

        if history_deals_getter is not None:
            try:
                since = max(now - 3600.0, float(entry.ts or 0.0) - 60.0)
                deals = list(history_deals_getter(since, now) or [])
            except Exception:
                deals = []

            for deal in deals:
                try:
                    deal_symbol = str(getattr(deal, "symbol", "") or "").upper()
                    deal_magic = int(getattr(deal, "magic", 0) or 0)
                    deal_volume = float(getattr(deal, "volume", 0.0) or 0.0)
                    deal_comment = str(getattr(deal, "comment", "") or "")
                except Exception:
                    continue

                tail = extract_idem_tail(deal_comment)

                if tail and tail == entry.key[-16:]:
                    active_journal.record_confirmed(
                        entry.key,
                        position_ticket=int(
                            getattr(deal, "position_id", 0) or 0
                        ),
                        deal_ticket=int(getattr(deal, "ticket", 0) or 0),
                        reason="reconcile_history_deal",
                    )
                    counts["confirmed"] += 1
                    matched = True
                    break

                if (
                    deal_symbol == normalized_symbol
                    and deal_magic == int(entry.magic)
                    and abs(deal_volume - float(entry.volume)) < 1e-8
                ):
                    active_journal.record_confirmed(
                        entry.key,
                        position_ticket=int(
                            getattr(deal, "position_id", 0) or 0
                        ),
                        deal_ticket=int(getattr(deal, "ticket", 0) or 0),
                        reason="reconcile_history_heuristic",
                    )
                    counts["confirmed"] += 1
                    matched = True
                    break

            if matched:
                continue

        if age_seconds > _RECONCILE_TTL_SEC:
            active_journal.record_failed(
                entry.key,
                reason=f"reconcile_expired_age={age_seconds:.1f}s",
                retcode=0,
            )
            counts["failed"] += 1

            try:
                log_idempotency.warning(
                    "WAL_RECONCILE_EXPIRED | key=%s symbol=%s age=%.1fs",
                    entry.key,
                    entry.symbol,
                    age_seconds,
                )
            except Exception:
                pass
        else:
            counts["still_pending"] += 1

    try:
        log_idempotency.info(
            "WAL_RECONCILE_SUMMARY | confirmed=%d failed=%d still_pending=%d",
            counts["confirmed"],
            counts["failed"],
            counts["still_pending"],
        )
    except Exception:
        pass

    return counts


# =============================================================================
# Private Helpers
# =============================================================================
def _enabled() -> bool:
    """Return whether idempotency is enabled."""
    return str(os.getenv("IDEMPOTENCY_ENABLED", "1")).strip() in (
        "1",
        "true",
        "True",
    )


# =============================================================================
# Module State
# =============================================================================
_DEFAULT_JOURNAL: Optional[OrderJournal] = None
_DEFAULT_JOURNAL_LOCK = threading.Lock()


def get_default_journal() -> OrderJournal:
    """Return the module-level default journal."""
    global _DEFAULT_JOURNAL

    with _DEFAULT_JOURNAL_LOCK:
        if _DEFAULT_JOURNAL is None:
            _DEFAULT_JOURNAL = OrderJournal()

    return _DEFAULT_JOURNAL


# =============================================================================
# Module Exports
# =============================================================================
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
