"""
Central logging helpers for the Exness trading stack.
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
_NOISY_LOGGER_LEVELS: Final[Tuple[Tuple[str, int], ...]] = (
    ("urllib3", logging.WARNING),
    ("requests", logging.WARNING),
    ("telebot", logging.INFO),
    ("matplotlib", logging.WARNING),
    ("PIL", logging.WARNING),
    ("catboost", logging.WARNING),
    ("asyncio", logging.WARNING),
)


# =============================================================================
# Logging Setup
# =============================================================================
log_config = logging.getLogger(__name__)


def _quiet_library_loggers() -> None:
    """Clamp noisy third-party loggers to production-safe levels."""
    for logger_name, level in _NOISY_LOGGER_LEVELS:
        try:
            logging.getLogger(logger_name).setLevel(level)
        except Exception:
            continue


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
    """Configure root logging in an idempotent, thread-safe way."""
    resolved_level = _as_level(level)

    with _CFG_LOCK:
        root = logging.getLogger()

        try:
            root.setLevel(resolved_level)
        except Exception:
            root.setLevel(logging.INFO)

        file_handler: Optional[RotatingFileHandler] = None

        if system_log_name:
            try:
                log_path = get_log_path(system_log_name)
                base_filename = str(log_path)
                existing = _find_file_handler(root, base_filename)

                if existing is not None:
                    existing.setLevel(resolved_level)
                    existing.setFormatter(_fmt())
                    file_handler = existing
                else:
                    file_handler = RotatingFileHandler(
                        base_filename,
                        maxBytes=int(max_bytes),
                        backupCount=int(backup_count),
                        encoding="utf-8",
                        delay=True,
                    )
                    file_handler.setLevel(resolved_level)
                    file_handler.setFormatter(_fmt())
                    root.addHandler(file_handler)
            except Exception as exc:
                log_config.error("Failed to configure file handler: %s", exc)

        if console:
            try:
                console_handler = _find_console_handler(root)
                if console_handler is None:
                    console_handler = logging.StreamHandler(stream=_stdout_stream())
                    root.addHandler(console_handler)

                console_handler.setLevel(resolved_level)
                console_handler.setFormatter(_fmt())
            except Exception as exc:
                log_config.error("Failed to configure console handler: %s", exc)

        try:
            logging.captureWarnings(True)
        except Exception:
            pass

        try:
            attach_global_handler_to_loggers(file_handler)
        except Exception as exc:
            log_config.error("attach_global_handler_to_loggers failed: %s", exc)

        try:
            _quiet_library_loggers()
        except Exception as exc:
            log_config.error("quiet_library_loggers failed: %s", exc)

        return file_handler


def attach_global_handler_to_loggers(
    handler: Optional[logging.Handler],
    names: Optional[Iterable[str]] = None,
) -> None:
    """Attach a handler to existing non-propagating loggers."""
    if handler is None:
        return

    prefixes = tuple(names or ())

    for logger_name, logger_obj in logging.root.manager.loggerDict.items():
        try:
            if not isinstance(logger_obj, logging.Logger):
                continue

            if prefixes:
                matched = False
                for prefix in prefixes:
                    if logger_name == prefix or logger_name.startswith(f"{prefix}."):
                        matched = True
                        break
                if not matched:
                    continue

            if logger_obj.propagate or handler in logger_obj.handlers:
                continue

            logger_obj.addHandler(handler)
        except Exception:
            continue


def get_log_path(*parts: str) -> Path:
    """Return a file path under the project log directory."""
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
    """Create or reuse a rotating file logger."""
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

    existing_handler: Optional[RotatingFileHandler] = None

    for handler in logger.handlers:
        try:
            if (
                isinstance(handler, RotatingFileHandler)
                and getattr(handler, "baseFilename", "") == log_path
            ):
                existing_handler = handler
                break
        except Exception:
            continue

    if existing_handler is None:
        try:
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=int(max_bytes),
                backupCount=int(backups),
                encoding="utf-8",
                delay=True,
            )
            file_handler.setLevel(int(level))
            file_handler.setFormatter(_fmt())
            logger.addHandler(file_handler)
        except OSError as exc:
            log_config.error(
                "Failed to create logger handler for %s: %s",
                name,
                exc,
            )
        except Exception as exc:
            log_config.error(
                "Unexpected error creating handler for %s: %s",
                name,
                exc,
            )
    else:
        try:
            existing_handler.setLevel(int(level))
            existing_handler.setFormatter(_fmt())
        except Exception:
            pass

    return logger


def get_artifact_dir(*parts: str) -> Path:
    """Return a directory under the project artifacts directory."""
    try:
        path = ARTIFACTS_DIR.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return ARTIFACTS_DIR


def get_artifact_path(*parts: str) -> Path:
    """Return a file path under the project artifacts directory."""
    try:
        path = ARTIFACTS_DIR.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return ARTIFACTS_DIR / "fallback.dat"


def log_dir_stats() -> tuple[int, int]:
    """Return total log size in bytes and file count."""
    total_size = 0
    file_count = 0

    try:
        for path in LOG_DIR.rglob("*"):
            try:
                if path.is_file():
                    stats = path.stat()
                    file_count += 1
                    total_size += int(stats.st_size)
            except Exception:
                continue
    except Exception:
        return 0, 0

    return total_size, file_count


def configure_module_logs(*, force: bool = False) -> int:
    """Configure dedicated module loggers from the registry."""
    global _MODULE_LOGS_CONFIGURED_ONCE

    with _CFG_LOCK:
        if _MODULE_LOGS_CONFIGURED_ONCE and not force:
            return 0

        configured_count = 0

        for registry_entry in _MODULE_LOG_REGISTRY:
            try:
                name, file_name, level, max_bytes, backups = registry_entry
                build_logger(
                    name,
                    file_name,
                    level=int(level),
                    max_bytes=int(max_bytes),
                    backups=int(backups),
                    propagate=False,
                )
                configured_count += 1
            except Exception as exc:
                try:
                    log_config.error(
                        "configure_module_logs failed for %r: %s",
                        registry_entry,
                        exc,
                    )
                except Exception:
                    pass

        _MODULE_LOGS_CONFIGURED_ONCE = True
        return configured_count


def list_configured_modules() -> Tuple[str, ...]:
    """Return registered logger names from the module registry."""
    return tuple(entry[0] for entry in _MODULE_LOG_REGISTRY)


# =============================================================================
# Private Helpers
# =============================================================================
def _stdout_stream():
    """Return a usable stdout stream."""
    try:
        return getattr(sys, "__stdout__", sys.stdout)
    except Exception:
        return sys.stdout


def _fmt() -> logging.Formatter:
    """Return the standard log formatter."""
    return logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _as_level(level: Union[str, int]) -> Union[str, int]:
    """Normalize a log level value."""
    try:
        if isinstance(level, int):
            return int(level)

        level_name = str(level).upper().strip()
        if level_name.isdigit():
            return int(level_name)

        if level_name in logging._nameToLevel:  # type: ignore[attr-defined]
            return level_name

        raise ValueError(f"Invalid log level: {level!r}")
    except Exception as exc:
        log_config.error("Invalid log level: %s", exc)
        return "INFO"


def _find_file_handler(
    root: logging.Logger,
    base_filename: str,
) -> Optional[RotatingFileHandler]:
    """Return the rotating file handler for the target path, if present."""
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


def _find_console_handler(root: logging.Logger) -> Optional[logging.StreamHandler]:
    """Return the console stream handler, excluding file handlers."""
    for handler in root.handlers:
        try:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler,
                logging.FileHandler,
            ):
                return handler
        except Exception:
            continue

    return None


# =============================================================================
# Module Registry
# =============================================================================
_MODULE_LOG_REGISTRY: Final[Tuple[Tuple[str, str, int, int, int], ...]] = (
    ("core.idempotency", "idempotency_wal.log", logging.INFO, 5_000_000, 10),
    ("core.news_blackout", "news_blackout.log", logging.INFO, 2_000_000, 5),
    (
        "core.stability_monitor",
        "stability_monitor.log",
        logging.INFO,
        5_000_000,
        10,
    ),
    ("core.model_engine", "model_engine.log", logging.INFO, 5_000_000, 7),
    ("core.signal_engine", "signal_engine.log", logging.INFO, 5_000_000, 7),
    ("core.risk_engine", "risk_engine.log", logging.INFO, 5_000_000, 7),
    ("core.utils", "core_utils.log", logging.INFO, 2_000_000, 5),
    ("core.clock_sync", "clock_sync.log", logging.INFO, 2_000_000, 5),
    ("core.data_engine", "data_engine.log", logging.INFO, 5_000_000, 7),
    ("core.portfolio_risk", "portfolio_risk.log", logging.INFO, 2_000_000, 5),
    (
        "portfolio.engine.err",
        "portfolio_engine_error.log",
        logging.ERROR,
        5_000_000,
        10,
    ),
    ("functions", "exness_api_functions.log", logging.INFO, 5_000_000, 7),
    ("history", "exness_api_history.log", logging.INFO, 2_000_000, 5),
    (
        "order_execution.health",
        "order_execution.log",
        logging.INFO,
        5_000_000,
        10,
    ),
    ("order_execution", "order_execution.log", logging.INFO, 5_000_000, 10),
    ("mt5", "mt5_client.log", logging.INFO, 5_000_000, 7),
    (
        "ExnessAPI.daily_balance",
        "exness_api_daily_balance.log",
        logging.INFO,
        2_000_000,
        5,
    ),
    ("ai.market_feed", "ai_market_feed.log", logging.INFO, 5_000_000, 7),
    (
        "ai.intraday_market_feed",
        "ai_intraday_market_feed.log",
        logging.INFO,
        5_000_000,
        7,
    ),
    ("feed_xau", "xau_market_feed.log", logging.INFO, 2_000_000, 5),
    ("feed_btc", "btc_market_feed.log", logging.INFO, 2_000_000, 5),
    ("strategies.xau", "strategies_xau.log", logging.INFO, 2_000_000, 5),
    ("strategies.btc", "strategies_btc.log", logging.INFO, 2_000_000, 5),
    ("telegram.bot", "telegram_bot.log", logging.INFO, 5_000_000, 7),
    ("TeleBot", "telegram_lib.log", logging.WARNING, 2_000_000, 5),
    (
        "backtest.engine_institutional",
        "backtest_engine.log",
        logging.INFO,
        5_000_000,
        5,
    ),
    (
        "backtest.model_train_institutional",
        "backtest_model_train.log",
        logging.INFO,
        5_000_000,
        5,
    ),
    (
        "backtest.metrics_institutional",
        "backtest_metrics.log",
        logging.INFO,
        2_000_000,
        3,
    ),
)


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
