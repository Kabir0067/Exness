"""
Central logging helpers for the Exness trading stack.

Provides idempotent, thread-safe configuration of the Python logging subsystem
with support for rotating file handlers, console output, per-module loggers,
and automatic quieting of noisy third-party libraries. Designed for zero-downtime
production trading environments.
"""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final, Iterable, Optional, Tuple, Union

# =============================================================================
# Paths (created at import time with operator-visible warnings on failure)
# =============================================================================
_BASE_DIR: Final[Path] = Path(__file__).resolve().parent

LOG_DIR: Final[Path] = (_BASE_DIR / "Logs").resolve()
ARTIFACTS_DIR: Final[Path] = (_BASE_DIR / "Artifacts").resolve()

try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError, RuntimeError) as exc:
    # Cannot use logging yet; emit to stderr so operators see startup issues
    print(f"[log_config] WARNING: Failed to create required directories: {exc}", file=sys.stderr)


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

# Cached formatter (avoids repeated format-string parsing on every handler creation)
_FORMATTER: Final[logging.Formatter] = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


# =============================================================================
# Logging Setup
# =============================================================================
log_config = logging.getLogger(__name__)


def _quiet_library_loggers() -> None:
    """Clamp noisy third-party loggers to production-safe levels (idempotent)."""
    for logger_name, level in _NOISY_LOGGER_LEVELS:
        try:
            logging.getLogger(logger_name).setLevel(level)
        except Exception:
            # Never let a single library break the entire quieting pass
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
    """
    Configure root logging in an idempotent, thread-safe way.

    Safe to call multiple times; subsequent invocations update levels/formatters
    on existing handlers but do not duplicate them. All mutations are protected
    by an internal lock. Third-party loggers are automatically quieted and
    warnings are captured.

    Args:
        level: Desired root level (accepts str name e.g. "DEBUG" or int).
        system_log_name: Optional filename (under LOG_DIR) for the primary
            rotating system log. If None, no file handler is attached.
        console: Attach (or update) a StreamHandler to stdout.
        max_bytes: Max size per log file before rotation.
        backup_count: Number of rotated backups to retain.

    Returns:
        The RotatingFileHandler instance for the system log (if created/updated),
        otherwise None.
    """
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
    """
    Attach a handler to all existing non-propagating loggers (optionally filtered).

    Used internally to ensure module loggers that have propagate=False still
    receive the primary system log output. Only loggers that do not already
    contain the handler are modified.

    Args:
        handler: The handler instance to attach (typically a RotatingFileHandler).
        names: Optional iterable of logger name prefixes to filter; if provided,
            only loggers whose name matches exactly or starts with "prefix." are
            considered.
    """
    if handler is None:
        return

    prefixes: Tuple[str, ...] = tuple(names or ())

    # Snapshot to avoid "dictionary changed size during iteration" in rare races
    for logger_name, logger_obj in list(logging.root.manager.loggerDict.items()):
        try:
            if not isinstance(logger_obj, logging.Logger):
                continue

            if prefixes and not any(
                logger_name == prefix or logger_name.startswith(f"{prefix}.")
                for prefix in prefixes
            ):
                continue

            if logger_obj.propagate or handler in logger_obj.handlers:
                continue

            logger_obj.addHandler(handler)
        except Exception:
            # One misbehaving logger must never abort the attach pass
            continue


def get_log_path(*parts: str) -> Path:
    """
    Return a file path under the project log directory (creating parents).

    Always succeeds: on any error a fallback path under LOG_DIR is returned
    so callers never receive an invalid path.

    Args:
        *parts: Path components to append under LOG_DIR.

    Returns:
        Absolute Path to the requested (or fallback) log file.
    """
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
    Create or reuse a dedicated rotating-file logger for a module.

    Idempotent: if a RotatingFileHandler pointing at the same baseFilename
    already exists on the logger, it is reused and updated. The logger's
    propagate flag is set as requested (default False for dedicated logs).

    Args:
        name: Logger name (e.g. "core.risk_engine").
        filename: Log file name (relative to LOG_DIR).
        level: Minimum level for this logger and its handler.
        max_bytes: Rotation size threshold.
        backups: Number of rotated files to keep.
        propagate: Whether events should bubble to the root logger.

    Returns:
        The configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    try:
        logger.setLevel(int(level))
    except Exception:
        logger.setLevel(logging.INFO)

    logger.propagate = bool(propagate)

    # get_log_path guarantees a valid path (creates parents or returns fallback)
    log_path = str(get_log_path(filename))

    existing_handler: Optional[RotatingFileHandler] = next(
        (
            h
            for h in logger.handlers
            if isinstance(h, RotatingFileHandler)
            and getattr(h, "baseFilename", "") == log_path
        ),
        None,
    )

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
            # Common FS errors (disk full, permission, path too long)
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
            # Handler may be closed or in transitional state; ignore
            pass

    return logger


def get_artifact_dir(*parts: str) -> Path:
    """
    Return a directory under the project artifacts directory (creating it).

    Args:
        *parts: Subdirectory components.

    Returns:
        Absolute Path to the requested artifact directory.
    """
    try:
        path = ARTIFACTS_DIR.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return ARTIFACTS_DIR


def get_artifact_path(*parts: str) -> Path:
    """
    Return a file path under the project artifacts directory (creating parents).

    Args:
        *parts: Path components.

    Returns:
        Absolute Path to the requested artifact file.
    """
    try:
        path = ARTIFACTS_DIR.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return ARTIFACTS_DIR / "fallback.dat"


def log_dir_stats() -> tuple[int, int]:
    """
    Return total size (bytes) and file count of everything under LOG_DIR.

    Used for operational monitoring / disk-space alerts. Errors are swallowed
    so the trading process is never impacted by transient FS issues.

    Returns:
        (total_size_bytes, file_count)
    """
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
    """
    Configure all dedicated module loggers declared in the internal registry.

    Idempotent unless force=True. Must be called after configure_logging if
    you want the module-specific files to exist. Thread-safe.

    Args:
        force: Re-configure even if already done in this process.

    Returns:
        Number of module loggers successfully configured (or updated).
    """
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
                    # If even error logging fails we still continue
                    pass

        _MODULE_LOGS_CONFIGURED_ONCE = True
        return configured_count


def list_configured_modules() -> Tuple[str, ...]:
    """Return the ordered tuple of all registered module logger names."""
    return tuple(entry[0] for entry in _MODULE_LOG_REGISTRY)


# =============================================================================
# Private Helpers (optimized for clarity and early exit)
# =============================================================================
def _stdout_stream():
    """Return the original stdout stream (resilient to sys.stdout redirection)."""
    try:
        return getattr(sys, "__stdout__", sys.stdout)
    except Exception:
        return sys.stdout


def _fmt() -> logging.Formatter:
    """Return the process-wide cached log formatter (no re-parsing of format string)."""
    return _FORMATTER


def _as_level(level: Union[str, int]) -> int:
    """
    Normalize any user-supplied level (str or int) to a Python logging int level.

    Accepts numeric strings, level names ("DEBUG", "INFO", ...), or ints.
    Unknown values fall back to INFO with an error log. Always returns int
    for consistent downstream use with setLevel().
    """
    try:
        if isinstance(level, int):
            return int(level)

        level_name = str(level).upper().strip()
        if level_name.isdigit():
            return int(level_name)

        # Works on all Python 3.x (private map is stable); 3.11+ also has getLevelNamesMapping
        name_to_level = getattr(logging, "_nameToLevel", {})
        if level_name in name_to_level:
            return int(name_to_level[level_name])

        log_config.error("Invalid log level %r – falling back to INFO", level)
        return logging.INFO
    except Exception as exc:
        log_config.error("Invalid log level %r: %s – falling back to INFO", level, exc)
        return logging.INFO


def _find_file_handler(
    root: logging.Logger, base_filename: str
) -> Optional[RotatingFileHandler]:
    """Return the first RotatingFileHandler whose baseFilename matches, or None."""
    return next(
        (
            h
            for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and getattr(h, "baseFilename", "") == base_filename
        ),
        None,
    )


def _find_console_handler(root: logging.Logger) -> Optional[logging.StreamHandler]:
    """Return the first StreamHandler that is not a FileHandler, or None."""
    return next(
        (
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ),
        None,
    )


# =============================================================================
# Module Registry (immutable – do not modify at runtime)
# =============================================================================
_MODULE_LOG_REGISTRY: Final[Tuple[Tuple[str, str, int, int, int], ...]] = (
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
    ("portfolio.engine.err", "portfolio_engine_error.log", logging.ERROR, 5_000_000, 10),
    ("functions", "exness_api_functions.log", logging.INFO, 5_000_000, 7),
    ("history", "exness_api_history.log", logging.INFO, 2_000_000, 5),
    ("order_execution.health", "order_execution.log", logging.INFO, 5_000_000, 10),
    ("order_execution", "order_execution.log", logging.INFO, 5_000_000, 10),
    ("mt5", "mt5_client.log", logging.INFO, 5_000_000, 7),
    ("ExnessAPI.daily_balance", "exness_api_daily_balance.log", logging.INFO, 2_000_000, 5),
    ("ai.market_feed", "ai_market_feed.log", logging.INFO, 5_000_000, 7),
    ("ai.intraday_market_feed", "ai_intraday_market_feed.log", logging.INFO, 5_000_000, 7),
    ("feed_xau", "xau_market_feed.log", logging.INFO, 2_000_000, 5),
    ("feed_btc", "btc_market_feed.log", logging.INFO, 2_000_000, 5),
    ("strategies.xau", "strategies_xau.log", logging.INFO, 2_000_000, 5),
    ("strategies.btc", "strategies_btc.log", logging.INFO, 2_000_000, 5),
    ("telegram.bot", "telegram_bot.log", logging.INFO, 5_000_000, 7),
    ("TeleBot", "telegram_lib.log", logging.WARNING, 2_000_000, 5),
    ("backtest.engine_institutional", "backtest_engine.log", logging.INFO, 5_000_000, 5),
    ("backtest.model_train_institutional", "backtest_model_train.log", logging.INFO, 5_000_000, 5),
    ("backtest.metrics_institutional", "backtest_metrics.log", logging.INFO, 2_000_000, 3),
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