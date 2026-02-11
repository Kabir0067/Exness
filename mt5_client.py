# mt5_client.py (SENIOR / PRODUCTION / ENV-CREDS ONLY / IPC-HARDENED / SINGLE-INSTANCE SAFE)
from __future__ import annotations

import atexit
import glob
import logging
import os
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Tuple

import MetaTrader5 as mt5

from log_config import LOG_DIR as LOG_ROOT, get_log_path

# =============================================================================
# Public lock: protect EVERY mt5.* call
# =============================================================================
MT5_LOCK = threading.RLock()

# Protect init/shutdown sequence
_INIT_LOCK = threading.Lock()

# Internal state
_initialized: bool = False
_last_health_reason: str = "not_initialized"
_last_health_log_ts_mono: float = 0.0  # monotonic

# Single-instance file lock state
_lock_guard = threading.Lock()
_lock_fp = None
_lock_acquired = False

# Auth failure throttling (prevents restart storms)
_auth_block_until_mono: float = 0.0
_auth_block_reason: str = ""


# =============================================================================
# Configuration (ENV for credentials only)
# =============================================================================
@dataclass(frozen=True)
class MT5Credentials:
    login: int
    password: str
    server: str


@dataclass(frozen=True)
class MT5ClientConfig:
    creds: MT5Credentials

    mt5_path: Optional[str] = None
    portable: bool = False
    autostart: bool = True
    require_path_on_windows: bool = True

    single_instance_lock: bool = True
    taskkill_on_ipc_timeout: bool = True
    timeout_ms: int = 300_000
    start_wait_sec: float = 5.0
    ready_timeout_sec: float = 120.0
    max_retries: int = 6
    retry_backoff_base_sec: float = 1.2
    backoff_cap_sec: float = 12.0

    # NEW: if auth failed, block re-tries for N seconds (engine will stop storming)
    auth_fail_cooldown_sec: float = 600.0

    log_level: int = logging.ERROR
    log_max_bytes: int = 5_242_880
    log_backups: int = 5
    health_log_throttle_sec: float = 15.0

    ipc_markers: Tuple[str, ...] = ("ipc timeout", "-10005", "(-10005")
    require_correct_account: bool = True


class MT5AuthError(RuntimeError):
    """Hard error: wrong login/password/server (do not retry in a tight loop)."""


# =============================================================================
# Logging
# =============================================================================
logger = logging.getLogger("mt5")
logger.setLevel(logging.ERROR)
logger.propagate = False


def _setup_logger(cfg: MT5ClientConfig) -> None:
    logger.setLevel(int(cfg.log_level))
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return

    fh = RotatingFileHandler(
        filename=str(get_log_path("mt5.log")),
        maxBytes=int(cfg.log_max_bytes),
        backupCount=int(cfg.log_backups),
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(int(cfg.log_level))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    logger.addHandler(fh)


# =============================================================================
# Helpers
# =============================================================================
def _is_windows() -> bool:
    return os.name == "nt"


def _mono() -> float:
    return time.monotonic()


def _last_error() -> Tuple[int, str]:
    with MT5_LOCK:
        try:
            code, msg = mt5.last_error()
            return int(code), str(msg)
        except Exception:
            return -99999, "unknown_mt5_error"


def _normalize_env_str(s: str) -> str:
    # strips quotes + whitespace + CR/LF (common .env/cmd issues)
    x = (s or "").strip()
    if len(x) >= 2 and ((x[0] == x[-1] == '"') or (x[0] == x[-1] == "'")):
        x = x[1:-1].strip()
    return x.replace("\r", "").replace("\n", "")


def _sleep_backoff(base: float, attempt: int, cap: float) -> None:
    a = int(attempt)
    delay = float(base) * (2.0 ** a)
    delay = min(float(cap), delay) + min(0.25, 0.03 * a)
    time.sleep(delay)


def _throttled_health_log(cfg: MT5ClientConfig, reason: str) -> None:
    global _last_health_log_ts_mono
    now = _mono()
    if now - _last_health_log_ts_mono >= float(cfg.health_log_throttle_sec):
        # SENIOR FIX: Downgrade to WARNING. The system auto-heals immediately after this.
        # ERROR should be reserved for unrecoverable failures or final exhausted retries.
        logger.warning("Fast path health failed (attempting auto-heal): %s", reason)
        _last_health_log_ts_mono = now


def _validate_cfg(cfg: MT5ClientConfig) -> None:
    if cfg is None:
        raise RuntimeError("MT5ClientConfig is None")

    login = int(getattr(cfg.creds, "login", 0) or 0)
    password = str(getattr(cfg.creds, "password", "") or "")
    server = str(getattr(cfg.creds, "server", "") or "")

    if login <= 0:
        raise RuntimeError("MT5 credentials invalid: login must be > 0")
    if not password:
        raise RuntimeError("MT5 credentials invalid: password is empty")
    if not server:
        raise RuntimeError("MT5 credentials invalid: server is empty")

    if int(cfg.timeout_ms) <= 0:
        raise RuntimeError("timeout_ms must be > 0")
    if float(cfg.ready_timeout_sec) <= 0:
        raise RuntimeError("ready_timeout_sec must be > 0")
    if int(cfg.max_retries) <= 0:
        raise RuntimeError("max_retries must be > 0")


def _is_auth_failed(err: Tuple[int, str]) -> bool:
    code, msg = int(err[0]), str(err[1]).lower()
    if code == -6:
        return True
    if "authorization failed" in msg:
        return True
    if "invalid account" in msg:
        return True
    return False


# =============================================================================
# Single-instance lock
# =============================================================================
def _lock_file_path() -> Path:
    base = Path(str(LOG_ROOT)) if str(LOG_ROOT) else (Path.cwd() / "Logs")
    base.mkdir(parents=True, exist_ok=True)
    return base / "mt5_single_instance.lock"


def _read_lock_file(p: Path) -> str:
    try:
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        pass
    return ""


def _write_lock_info(p: Path) -> None:
    try:
        info = f"pid={os.getpid()} ts={int(time.time())}\n"
        _lock_fp.seek(0)
        _lock_fp.truncate(0)
        _lock_fp.write(info)
        _lock_fp.flush()
        try:
            os.fsync(_lock_fp.fileno())
        except Exception:
            pass
    except Exception:
        pass


def _acquire_single_instance_lock(cfg: MT5ClientConfig) -> None:
    global _lock_fp, _lock_acquired
    if not cfg.single_instance_lock:
        return

    with _lock_guard:
        if _lock_acquired and _lock_fp:
            return

        p = _lock_file_path()
        try:
            _lock_fp = open(p, "a+", encoding="utf-8")
            _lock_fp.seek(0)

            if _is_windows():
                import msvcrt  # type: ignore

                try:
                    _lock_fp.seek(0)
                    msvcrt.locking(_lock_fp.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    other = _read_lock_file(p)
                    raise RuntimeError(f"Another engine process is running (lock busy). Stop it first. lock_info={other}")
            else:
                import fcntl  # type: ignore

                try:
                    fcntl.flock(_lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    other = _read_lock_file(p)
                    raise RuntimeError(f"Another engine process is running (lock busy). Stop it first. lock_info={other}")

            _lock_acquired = True
            _write_lock_info(p)

        except Exception as exc:
            try:
                if _lock_fp:
                    _lock_fp.close()
            except Exception:
                pass
            _lock_fp = None
            _lock_acquired = False
            raise RuntimeError(f"Failed to acquire single-instance lock: {exc}") from exc


def _release_single_instance_lock() -> None:
    global _lock_fp, _lock_acquired
    with _lock_guard:
        if not _lock_acquired or not _lock_fp:
            return
        try:
            if _is_windows():
                import msvcrt  # type: ignore

                try:
                    _lock_fp.seek(0)
                    msvcrt.locking(_lock_fp.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            else:
                import fcntl  # type: ignore

                try:
                    fcntl.flock(_lock_fp.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        finally:
            try:
                _lock_fp.close()
            except Exception:
                pass
            _lock_fp = None
            _lock_acquired = False


# =============================================================================
# Windows path resolution (NO ENV)
# =============================================================================
def _resolve_mt5_path_windows(mt5_path: Optional[str]) -> str:
    if mt5_path and os.path.exists(mt5_path):
        return mt5_path

    candidates: list[str] = []
    for base in (r"C:\Program Files", r"C:\Program Files (x86)"):
        candidates += glob.glob(base + r"\MetaTrader 5*\terminal64.exe")
        candidates += glob.glob(base + r"\MetaQuotes*\terminal64.exe")
        candidates += glob.glob(base + r"\Exness*\terminal64.exe")

    user_home = Path.home()
    for root in (user_home / "AppData" / "Roaming", user_home / "AppData" / "Local"):
        if root.exists():
            candidates += glob.glob(str(root / "MetaQuotes" / "Terminal" / "*" / "terminal64.exe"))

    for x in candidates:
        if x and os.path.exists(x):
            return x
    return ""


def _is_terminal_running_windows() -> bool:
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq terminal64.exe"],
            text=True,
            errors="ignore",
        ).lower()
        return "terminal64.exe" in out
    except Exception:
        return False


def _taskkill_terminal_windows() -> None:
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", "terminal64.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def _start_terminal_windows(path: str, portable: bool) -> None:
    if not path or not os.path.exists(path):
        raise RuntimeError(f"MT5 path invalid: {path}")

    cmd = [path]
    if portable:
        cmd.append("/portable")

    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW

    cwd = str(Path(path).resolve().parent)
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        cwd=cwd,
    )


# =============================================================================
# Health check
# =============================================================================
@dataclass(frozen=True)
class Health:
    ok: bool
    reason: str


def _health(cfg: MT5ClientConfig) -> Health:
    try:
        ti = mt5.terminal_info()
        if ti is None:
            return Health(False, "terminal_info_none")
        if not bool(getattr(ti, "connected", False)):
            return Health(False, "terminal_not_connected")
        if not bool(getattr(ti, "trade_allowed", False)):
            return Health(False, "algo_trading_disabled_in_terminal")

        ai = mt5.account_info()
        if ai is None:
            return Health(False, "account_info_none")

        if cfg.require_correct_account:
            if int(getattr(ai, "login", -1)) != int(cfg.creds.login):
                return Health(False, "wrong_account")

        if not bool(getattr(ai, "trade_allowed", False)):
            return Health(False, "trading_disabled_for_account")

        return Health(True, "ok")
    except Exception:
        logger.error("health exception | tb=%s", traceback.format_exc())
        return Health(False, "health_check_exception")


def _wait_ready(cfg: MT5ClientConfig, timeout_sec: float) -> None:
    deadline = _mono() + float(timeout_sec)
    last = "waiting"
    while _mono() < deadline:
        with MT5_LOCK:
            h = _health(cfg)
        last = h.reason
        if h.ok:
            return
        time.sleep(0.20)
    hint = ""
    if last == "algo_trading_disabled_in_terminal":
        hint = " Enable Algo Trading in MT5: Tools → Options → Expert Advisors → Allow Algo Trading (or press AutoTrading button on toolbar)."
    raise RuntimeError(f"mt5_health_failed:{last}{hint}")


# =============================================================================
# Initialize + login (multi-fallback)
# =============================================================================
def _init_and_login(cfg: MT5ClientConfig, mt5_path: str) -> None:
    with MT5_LOCK:
        try:
            mt5.shutdown()
        except Exception:
            pass
    time.sleep(0.10)

    init_kwargs = {
        "path": str(mt5_path) if mt5_path else None,
        "portable": bool(cfg.portable),
        "timeout": int(cfg.timeout_ms),
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    # Attempt #1: initialize with creds
    with MT5_LOCK:
        ok1 = mt5.initialize(
            login=int(cfg.creds.login),
            password=str(cfg.creds.password),
            server=str(cfg.creds.server),
            **init_kwargs,
        )
    if ok1:
        return

    e1 = _last_error()

    # If auth failed here, still try the second path once (some terminals need init first),
    # but if it fails again with auth -> hard fail.
    with MT5_LOCK:
        try:
            mt5.shutdown()
        except Exception:
            pass
    time.sleep(0.10)

    with MT5_LOCK:
        ok2 = mt5.initialize(**init_kwargs)
    if not ok2:
        e2 = _last_error()
        if _is_auth_failed(e1) or _is_auth_failed(e2):
            raise MT5AuthError(f"authorization_failed: login={cfg.creds.login} server={cfg.creds.server} path={mt5_path} | init_with_creds={e1} init_no_creds={e2}")
        raise RuntimeError(f"mt5.initialize failed: {e2} | first_try={e1}")

    with MT5_LOCK:
        logged = mt5.login(
            login=int(cfg.creds.login),
            password=str(cfg.creds.password),
            server=str(cfg.creds.server),
            timeout=int(cfg.timeout_ms),
        )
    if not logged:
        e3 = _last_error()
        with MT5_LOCK:
            try:
                mt5.shutdown()
            except Exception:
                pass
        if _is_auth_failed(e1) or _is_auth_failed(e3):
            raise MT5AuthError(f"authorization_failed: login={cfg.creds.login} server={cfg.creds.server} path={mt5_path} | init_with_creds={e1} login={e3}")
        raise RuntimeError(f"mt5.login failed: {e3} | first_try={e1}")


def _is_ipc_timeout(cfg: MT5ClientConfig, exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(m in msg for m in cfg.ipc_markers):
        return True
    code, last = _last_error()
    if code == -10005:
        return True
    if any(m in str(last).lower() for m in cfg.ipc_markers):
        return True
    return False


# =============================================================================
# Default config loader (ENV for credentials only)
# =============================================================================
def _default_config_from_env() -> MT5ClientConfig:
    try:
        from config_xau import get_config_from_env as _get_cfg  # lazy import
        cfg = _get_cfg()
    except Exception:
        try:
            from config_btc import get_config_from_env as _get_cfg  # lazy import
            cfg = _get_cfg()
        except Exception as exc:
            raise RuntimeError("Unable to build MT5 config from env-backed config files") from exc

    creds = MT5Credentials(
        login=int(getattr(cfg, "login", 0) or 0),
        password=_normalize_env_str(str(getattr(cfg, "password", "") or "")),
        server=_normalize_env_str(str(getattr(cfg, "server", "") or "")),
    )

    return MT5ClientConfig(
        creds=creds,
        mt5_path=getattr(cfg, "mt5_path", None),
        portable=bool(getattr(cfg, "mt5_portable", False)),
        autostart=bool(getattr(cfg, "mt5_autostart", True)),
        timeout_ms=int(getattr(cfg, "mt5_timeout_ms", 300_000) or 300_000),
        require_path_on_windows=False,
        auth_fail_cooldown_sec=float(getattr(cfg, "mt5_auth_cooldown_sec", 600.0) or 600.0),
        ready_timeout_sec=float(getattr(cfg, "mt5_ready_timeout_sec", 120.0) or 120.0),
    )


# =============================================================================
# Public API
# =============================================================================
def ensure_mt5(cfg: Optional[MT5ClientConfig] = None) -> mt5:
    global _initialized, _last_health_reason, _auth_block_until_mono, _auth_block_reason

    if cfg is None:
        cfg = _default_config_from_env()

    _validate_cfg(cfg)
    _setup_logger(cfg)

    # AUTH cooldown gate (prevents infinite storms)
    now = _mono()
    if now < _auth_block_until_mono:
        raise MT5AuthError(f"auth_blocked:{_auth_block_reason}")

    _acquire_single_instance_lock(cfg)

    mt5_path = ""
    if _is_windows():
        mt5_path = _resolve_mt5_path_windows(cfg.mt5_path)
        if cfg.require_path_on_windows and (not mt5_path or not os.path.exists(mt5_path)):
            raise RuntimeError("MT5 terminal64.exe path required on Windows. Provide MT5ClientConfig.mt5_path.")
    else:
        mt5_path = cfg.mt5_path or ""

    with MT5_LOCK:
        if _initialized:
            h = _health(cfg)
            _last_health_reason = h.reason
            if h.ok:
                return mt5
            _throttled_health_log(cfg, h.reason)
            
            # Robustness: If terminal_info is None, the IPC link is likely dead or the terminal is hung.
            # Instead of polite shutdown, force a kill to ensure a clean slate.
            if h.reason == "terminal_info_none" and _is_windows() and cfg.taskkill_on_ipc_timeout:
                logger.warning("Detecting zombie terminal (terminal_info_none). Forcing taskkill before restart.")
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                _taskkill_terminal_windows()
            else:
                try:
                    mt5.shutdown()
                except Exception:
                    pass
            
            _initialized = False

    last_exc: Optional[Exception] = None

    with _INIT_LOCK:
        for attempt in range(int(cfg.max_retries)):
            try:
                if _is_windows() and cfg.autostart:
                    if not _is_terminal_running_windows():
                        _start_terminal_windows(mt5_path, portable=cfg.portable)
                        if float(cfg.start_wait_sec) > 0:
                            time.sleep(float(cfg.start_wait_sec))
                    else:
                        if attempt == 0 and float(cfg.start_wait_sec) > 0:
                            time.sleep(min(0.75, float(cfg.start_wait_sec)))

                _init_and_login(cfg, mt5_path)
                _wait_ready(cfg, timeout_sec=float(cfg.ready_timeout_sec))

                with MT5_LOCK:
                    _initialized = True
                    _last_health_reason = "ok"

                return mt5

            except MT5AuthError as exc:
                # Hard stop + cooldown (this is ALWAYS creds/server)
                last_exc = exc
                _auth_block_reason = str(exc)
                _auth_block_until_mono = _mono() + float(cfg.auth_fail_cooldown_sec)

                with MT5_LOCK:
                    try:
                        mt5.shutdown()
                    except Exception:
                        pass
                    _initialized = False
                    _last_health_reason = "auth_failed"

                logger.error(
                    "ensure_mt5 AUTH FAILED (cooldown %.0fs) | err=%s",
                    float(cfg.auth_fail_cooldown_sec),
                    exc,
                )
                raise

            except Exception as exc:
                last_exc = exc

                is_ipc = _is_windows() and bool(cfg.taskkill_on_ipc_timeout) and _is_ipc_timeout(cfg, exc)
                if is_ipc:
                    logger.error("IPC timeout -> taskkill terminal64.exe + retry | err=%s | last_error=%s", exc, _last_error())
                    with MT5_LOCK:
                        try:
                            mt5.shutdown()
                        except Exception:
                            pass
                    _taskkill_terminal_windows()
                    if cfg.autostart:
                        try:
                            _start_terminal_windows(mt5_path, portable=cfg.portable)
                            if float(cfg.start_wait_sec) > 0:
                                time.sleep(float(cfg.start_wait_sec))
                        except Exception as exc2:
                            logger.error("MT5 restart failed: %s | tb=%s", exc2, traceback.format_exc())

                with MT5_LOCK:
                    try:
                        mt5.shutdown()
                    except Exception:
                        pass
                    _initialized = False
                    _last_health_reason = f"init_failed:{type(exc).__name__}"

                logger.error(
                    "ensure_mt5 attempt %d/%d failed: %s | tb=%s",
                    attempt + 1,
                    int(cfg.max_retries),
                    exc,
                    traceback.format_exc(),
                )

                if attempt < int(cfg.max_retries) - 1:
                    _sleep_backoff(float(cfg.retry_backoff_base_sec), attempt, cap=float(cfg.backoff_cap_sec))

        raise RuntimeError(f"MT5 initialization failed after {cfg.max_retries} attempts: {last_exc}")


def shutdown_mt5() -> None:
    global _initialized, _last_health_reason
    with _INIT_LOCK:
        with MT5_LOCK:
            try:
                mt5.shutdown()
            except Exception as exc:
                logger.error("MT5 shutdown error: %s | tb=%s", exc, traceback.format_exc())
            _initialized = False
            _last_health_reason = "shutdown"
    _release_single_instance_lock()


def mt5_status() -> Tuple[bool, str]:
    with MT5_LOCK:
        return bool(_initialized), str(_last_health_reason)


atexit.register(shutdown_mt5)

__all__ = ["ensure_mt5", "shutdown_mt5", "mt5_status", "MT5_LOCK", "MT5ClientConfig", "MT5Credentials", "MT5AuthError", "mt5"]
