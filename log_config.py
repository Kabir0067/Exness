"""
Centralized logging configuration for the Exness trading stack.

All modules should import LOG_DIR (or helpers) to ensure
consistent log paths under the project root (./Logs by default).

This module is ALSO the single source of truth for per-module dedicated
log files — see `_MODULE_LOG_REGISTRY` and `configure_module_logs()`.
Every institutional-grade module (idempotency WAL, news blackout,
stability monitor, MT5 client, order execution, data feeds …) has its
own rotating file handler attached from that registry so that failures
can always be traced to a dedicated, unambiguous log file.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final, Iterable, Optional, Tuple, Union

# =============================================================================
# Paths
# =============================================================================
_BASE_DIR: Final[Path] = Path(__file__).resolve().parent

LOG_DIR: Final[Path] = (_BASE_DIR / "Logs").resolve()
ARTIFACTS_DIR: Final[Path] = (_BASE_DIR / "Artifacts").resolve()

# Ensure required directories exist; never fail during import
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass


# =============================================================================
# Global Constants
# =============================================================================
_CFG_LOCK = threading.Lock()
_MODULE_LOGS_CONFIGURED_ONCE = False


# =============================================================================
# Public API
# =============================================================================
def configure_logging(
    *,
    level: Union[str, int] = "INFO",
    system_log_name: Optional[str] = None,
    console: bool = True,
    max_bytes: int = 15 * 1024 * 1024,
    backup_count: int = 7,
) -> Optional[RotatingFileHandler]:
    """
    Initialize global logging configuration.

    Idempotent, thread-safe, and production-hardened.
    """
    lvl = _as_level(level)

    with _CFG_LOCK:
        root = logging.getLogger()

        try:
            root.setLevel(lvl)
        except Exception:
            root.setLevel(logging.INFO)

        file_handler: Optional[RotatingFileHandler] = None

        if system_log_name:
            try:
                log_path = get_log_path(system_log_name)
                base_fn = str(log_path)

                existing = _find_file_handler(root, base_fn)

                if existing is not None:
                    existing.setLevel(lvl)
                    existing.setFormatter(_fmt())
                    file_handler = existing
                else:
                    fh = RotatingFileHandler(
                        base_fn,
                        maxBytes=int(max_bytes),
                        backupCount=int(backup_count),
                        encoding="utf-8",
                        delay=True,
                    )
                    fh.setLevel(lvl)
                    fh.setFormatter(_fmt())
                    root.addHandler(fh)
                    file_handler = fh

            except Exception as exc:
                logging.getLogger(__name__).error(
                    "Failed to configure file handler: %s", exc
                )

        if console:
            try:
                ch = _find_console_handler(root)
                if ch is None:
                    ch = logging.StreamHandler(stream=_stdout_stream())
                    root.addHandler(ch)

                ch.setLevel(lvl)
                ch.setFormatter(_fmt())

            except Exception as exc:
                logging.getLogger(__name__).error(
                    "Failed to configure console handler: %s", exc
                )

        try:
            logging.captureWarnings(True)
        except Exception:
            pass

        try:
            attach_global_handler_to_loggers(file_handler)
        except Exception as exc:
            logging.getLogger(__name__).error(
                "attach_global_handler_to_loggers failed: %s", exc
            )

        return file_handler


def attach_global_handler_to_loggers(
    handler: Optional[logging.Handler],
    names: Optional[Iterable[str]] = None,
) -> None:
    """Attach a handler to existing loggers with propagate=False."""
    if handler is None:
        return

    prefixes = tuple(names or ())

    for name, obj in logging.root.manager.loggerDict.items():
        try:
            if not isinstance(obj, logging.Logger):
                continue

            if prefixes:
                matched = False
                for p in prefixes:
                    if name == p or name.startswith(f"{p}."):
                        matched = True
                        break
                if not matched:
                    continue

            if obj.propagate:
                continue

            if handler in obj.handlers:
                continue

            obj.addHandler(handler)

        except Exception:
            continue


def get_log_path(*parts: str) -> Path:
    """Return a file path within LOG_DIR."""
    try:
        path = LOG_DIR.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return LOG_DIR / "fallback.log"


def build_logger(
    name: str,
    filename: str,
    level: int = logging.INFO,
    *,
    max_bytes: int = 5 * 1024 * 1024,
    backups: int = 5,
    propagate: bool = False,
) -> logging.Logger:
    """
    Create or reuse a rotating file logger instance.

    Idempotent: a handler for the exact same `baseFilename` is never added
    twice — its level/formatter are merely refreshed. `delay=True` means
    the log file is created on the first actual write (not at handler
    construction time), so unused loggers leave no ghost files behind.

    Parameters
    ----------
    name : str
        Fully-qualified logger name (e.g. ``core.idempotency``).
    filename : str
        Relative filename under ``LOG_DIR``.
    level : int
        Effective level for both the logger and the attached handler.
    max_bytes : int
        Rotation threshold (bytes). Default: 5 MiB.
    backups : int
        Number of rotated backups retained. Default: 5.
    propagate : bool
        If False (recommended for dedicated file loggers) the records do
        not bubble up to the root logger, avoiding duplicate lines in
        ``stdout.log``.
    """
    logger = logging.getLogger(name)

    try:
        logger.setLevel(int(level))
    except Exception:
        logger.setLevel(logging.INFO)

    logger.propagate = bool(propagate)

    try:
        log_path = str(get_log_path(filename))
    except Exception:
        try:
            os.makedirs(str(LOG_DIR), exist_ok=True)
            log_path = os.path.join(str(LOG_DIR), filename)
        except Exception:
            return logger

    existing = None
    for handler in logger.handlers:
        try:
            if (
                isinstance(handler, RotatingFileHandler)
                and getattr(handler, "baseFilename", "") == log_path
            ):
                existing = handler
                break
        except Exception:
            continue

    if existing is None:
        try:
            handler = RotatingFileHandler(
                log_path,
                maxBytes=int(max_bytes),
                backupCount=int(backups),
                encoding="utf-8",
                delay=True,
            )
            handler.setLevel(int(level))
            handler.setFormatter(_fmt())
            logger.addHandler(handler)
        except OSError as exc:
            logging.getLogger(__name__).error(
                "Failed to create logger handler for %s: %s", name, exc
            )
        except Exception as exc:
            logging.getLogger(__name__).error(
                "Unexpected error creating handler for %s: %s", name, exc
            )
    else:
        try:
            existing.setLevel(int(level))
            existing.setFormatter(_fmt())
        except Exception:
            pass

    return logger


def get_artifact_dir(*parts: str) -> Path:
    """Return a directory within ARTIFACTS_DIR."""
    try:
        path = ARTIFACTS_DIR.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return ARTIFACTS_DIR


def get_artifact_path(*parts: str) -> Path:
    """Return a file path within ARTIFACTS_DIR."""
    try:
        path = ARTIFACTS_DIR.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return ARTIFACTS_DIR / "fallback.dat"


def log_dir_stats() -> tuple[int, int]:
    """Compute total size (bytes) and file count within LOG_DIR."""
    total = 0
    count = 0

    try:
        for p in LOG_DIR.rglob("*"):
            try:
                if p.is_file():
                    st = p.stat()
                    count += 1
                    total += int(st.st_size)
            except Exception:
                continue
    except Exception:
        return 0, 0

    return total, count


# =============================================================================
# Private Helpers
# =============================================================================
def _stdout_stream():
    """Guarantee a valid stdout stream fallback."""
    try:
        return getattr(sys, "__stdout__", sys.stdout)
    except Exception:
        return sys.stdout


def _fmt() -> logging.Formatter:
    """Return the standard log formatter."""
    return logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _as_level(level: Union[str, int]) -> Union[str, int]:
    """Normalize and validate log level defensively."""
    try:
        if isinstance(level, int):
            return int(level)

        s = str(level).upper().strip()
        if s.isdigit():
            return int(s)

        if s in logging._nameToLevel:  # type: ignore[attr-defined]
            return s

        raise ValueError(f"Invalid log level: {level!r}")
    except Exception as exc:
        logging.getLogger(__name__).error("Invalid log level: %s", exc)
        return "INFO"


def _find_file_handler(
    root: logging.Logger, base_filename: str
) -> Optional[RotatingFileHandler]:
    """Locate existing rotating file handler by filename."""
    for handler in root.handlers:
        try:
            if (
                isinstance(handler, RotatingFileHandler)
                and getattr(handler, "baseFilename", "") == base_filename
            ):
                return handler
        except Exception:
            continue
    return None


def _find_console_handler(
    root: logging.Logger,
) -> Optional[logging.StreamHandler]:
    """Identify console handler while excluding file handlers."""
    for handler in root.handlers:
        try:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                return handler
        except Exception:
            continue
    return None


# =============================================================================
# Per-Module Dedicated Log Files
# =============================================================================
_MODULE_LOG_REGISTRY: Final[Tuple[Tuple[str, str, int, int, int], ...]] = (
    # ─── CORE (институтсионалӣ) ────────────────────────────────────────────
    ("core.idempotency", "idempotency_wal.log", logging.INFO, 5_000_000, 10),
    ("core.news_blackout", "news_blackout.log", logging.INFO, 2_000_000, 5),
    ("core.stability_monitor", "stability_monitor.log", logging.INFO, 5_000_000, 10),
    ("core.model_engine", "model_engine.log", logging.INFO, 5_000_000, 7),
    ("core.signal_engine", "signal_engine.log", logging.INFO, 5_000_000, 7),
    ("core.risk_engine", "risk_engine.log", logging.INFO, 5_000_000, 7),
    ("core.utils", "core_utils.log", logging.INFO, 2_000_000, 5),
    ("core.clock_sync", "clock_sync.log", logging.INFO, 2_000_000, 5),
    ("core.data_engine", "data_engine.log", logging.INFO, 5_000_000, 7),
    ("core.portfolio_risk", "portfolio_risk.log", logging.INFO, 2_000_000, 5),
    # ─── Engine.ERR (пештар декларатсия буд, файл нест) ────────────────────
    ("portfolio.engine.err", "portfolio_engine_error.log", logging.ERROR, 5_000_000, 10),
    # ─── ExnessAPI / MT5 клиент ────────────────────────────────────────────
    ("functions", "exness_api_functions.log", logging.INFO, 5_000_000, 7),
    ("history", "exness_api_history.log", logging.INFO, 2_000_000, 5),
    ("order_execution.health", "order_execution.log", logging.INFO, 5_000_000, 10),
    ("order_execution", "order_execution.log", logging.INFO, 5_000_000, 10),
    ("mt5", "mt5_client.log", logging.INFO, 5_000_000, 7),
    # `ExnessAPI/daily_balance.py` getLogger(__name__)-ро истифода мебарад,
    # ки номашро "ExnessAPI.daily_balance" мекунад. Бе entry-и поёна навиштаҳо
    # ба stdout.log мерафтанд — ҳоло файли шахсии худро доранд.
    ("ExnessAPI.daily_balance", "exness_api_daily_balance.log", logging.INFO, 2_000_000, 5),
    # ─── Data feeds (чаро файлҳо size=0 буданд) ────────────────────────────
    ("ai.market_feed", "ai_market_feed.log", logging.INFO, 5_000_000, 7),
    ("ai.intraday_market_feed", "ai_intraday_market_feed.log", logging.INFO, 5_000_000, 7),
    ("feed_xau", "xau_market_feed.log", logging.INFO, 2_000_000, 5),
    ("feed_btc", "btc_market_feed.log", logging.INFO, 2_000_000, 5),
    # ─── Стратегия ва bot ───────────────────────────────────────────────────
    ("strategies.xau", "strategies_xau.log", logging.INFO, 2_000_000, 5),
    ("strategies.btc", "strategies_btc.log", logging.INFO, 2_000_000, 5),
    ("telegram.bot", "telegram_bot.log", logging.INFO, 5_000_000, 7),
    ("TeleBot", "telegram_lib.log", logging.WARNING, 2_000_000, 5),
    # ─── Backtest (ҳангоми retraining-и онлайн фаъол) ──────────────────────
    ("backtest.engine_institutional", "backtest_engine.log", logging.INFO, 5_000_000, 5),
    ("backtest.model_train_institutional", "backtest_model_train.log", logging.INFO, 5_000_000, 5),
    ("backtest.metrics_institutional", "backtest_metrics.log", logging.INFO, 2_000_000, 3),
)


def configure_module_logs(*, force: bool = False) -> int:
    """
        Шумораи entry-ҳои registry, ки муваффақ коркард шуданд.
    """
    global _MODULE_LOGS_CONFIGURED_ONCE
    with _CFG_LOCK:
        if _MODULE_LOGS_CONFIGURED_ONCE and not force:
            return 0

        ok_count = 0
        for entry in _MODULE_LOG_REGISTRY:
            try:
                name, file_name, level, max_bytes, backups = entry
                build_logger(
                    name,
                    file_name,
                    level=int(level),
                    max_bytes=int(max_bytes),
                    backups=int(backups),
                    propagate=False,
                )
                ok_count += 1
            except Exception as exc:
                try:
                    logging.getLogger(__name__).error(
                        "configure_module_logs: failed for %r — %s",
                        entry,
                        exc,
                    )
                except Exception:
                    pass

        _MODULE_LOGS_CONFIGURED_ONCE = True
        return ok_count


def list_configured_modules() -> Tuple[str, ...]:
    """Номҳои logger-ҳои сабтшудаи registry-ро бармегардонад."""
    return tuple(entry[0] for entry in _MODULE_LOG_REGISTRY)


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "ARTIFACTS_DIR",
    "LOG_DIR",
    "attach_global_handler_to_loggers",
    "build_logger",
    "configure_logging",
    "configure_module_logs",
    "get_artifact_dir",
    "get_artifact_path",
    "get_log_path",
    "list_configured_modules",
    "log_dir_stats",
]
