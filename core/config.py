"""
core/config.py — Unified configuration system and shared contracts.

Provides asset-specific tuning (BTC/XAU) via environment overrides,
centralized definitions for backtest quality gates, cross-module data structures,
and strict pre-flight environment checks to fail fast on malformed deployments.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Optional, Tuple

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    from cryptography.fernet import Fernet, InvalidToken
except Exception:
    Fernet = None  # type: ignore[assignment]
    InvalidToken = Exception  # type: ignore[assignment]

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("⚠️ WARNING: 'python-dotenv' not installed. .env file will NOT be loaded.")


# =============================================================================
# Global Constants & Meta Definitions
# =============================================================================
ALLOWED_SYMBOLS: Tuple[str, ...] = ("XAUUSDm", "XAUUSDm.", "BTCUSDm", "BTCUSDm.")

MIN_GATE_SHARPE: float = 0.5
MIN_GATE_WIN_RATE: float = 0.52
MAX_GATE_DRAWDOWN: float = 0.25

WFA_MIN_WINDOWS: int = 3
WFA_MIN_PASS_RATE: float = 0.60

_BOOL_TRUE = frozenset({"1", "true", "yes", "y", "on"})
_BOOL_FALSE = frozenset({"0", "false", "no", "n", "off"})
_SECRET_KEY_ENV_NAMES: Tuple[str, ...] = (
    "CONFIG_SECRET_KEY",
    "CREDENTIALS_MASTER_KEY",
    "EXNESS_SECRET_KEY",
)

_REQUIRED_ENV_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("EXNESS_LOGIN",),
    ("EXNESS_PASSWORD",),
    ("EXNESS_SERVER",),
    ("TG_TOKEN", "BOT_TOKEN"),
    ("TG_ADMIN_ID", "ADMIN_ID"),
)

_ENV_TEMPLATE = """# ==========================================
# EXNESS TRADING BOT CREDENTIALS
# ==========================================
EXNESS_LOGIN=
EXNESS_PASSWORD=
EXNESS_SERVER=MT5Real
# Optional encrypted secret support:
# CONFIG_SECRET_KEY=
# EXNESS_PASSWORD_ENC=

# ==========================================
# TELEGRAM NOTIFICATIONS
# ==========================================
TG_TOKEN=
TG_ADMIN_ID=
# Optional encrypted secret support:
# TG_TOKEN_ENC=
"""

_ENV_HINT_SHOWN = False
_ENV_HINT_LOCK = threading.Lock()

# Mapping of string timeframes to MT5 enum values (if available)
_TF_MAP_DRAFT: Dict[str, int] = {}
if mt5 is not None:
    _TF_MAP_DRAFT = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,
        "M5": mt5.TIMEFRAME_M5,
        "M10": mt5.TIMEFRAME_M10,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
TF_MAP = MappingProxyType(_TF_MAP_DRAFT)


# =============================================================================
# Core Environment Verification Logic
# =============================================================================
def _env_path() -> Path:
    return Path(".env")


def _env_first(*names: str) -> str:
    for name in names:
        for candidate in (
            name,
            f"{name}_FILE",
            f"{name}_ENC",
            f"{name}_ENC_FILE",
        ):
            value = os.environ.get(candidate, "").strip()
            if value:
                return value
    return ""


def _allow_missing_tg() -> bool:
    return False


def _auto_dry_run_on_missing_env() -> bool:
    return False


def _is_dry_run() -> bool:
    return False


def _missing_required_env_vars() -> List[str]:
    missing: List[str] = []
    for aliases in _REQUIRED_ENV_GROUPS:
        if _allow_missing_tg() and aliases[0] in ("TG_TOKEN", "TG_ADMIN_ID"):
            continue
        if not _env_first(*aliases):
            missing.append(aliases[0])
    return missing


def _existing_env_keys(path: Path) -> set:
    keys: set = set()
    if not path.exists():
        return keys
    try:
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip().lstrip("\ufeff")
            if key:
                keys.add(key)
    except Exception:
        return set()
    return keys


def _ensure_env_template(path: Path) -> bool:
    """Ensure the .env file exists; append missing blocks if necessary."""
    created = False
    if not path.exists():
        path.write_text(_ENV_TEMPLATE, encoding="utf-8")
        return True

    existing = _existing_env_keys(path)
    missing_keys: List[str] = []
    for aliases in _REQUIRED_ENV_GROUPS:
        if any(
            a in existing
            or f"{a}_FILE" in existing
            or f"{a}_ENC" in existing
            or f"{a}_ENC_FILE" in existing
            for a in aliases
        ):
            continue
        missing_keys.append(aliases[0])

    if missing_keys:
        with path.open("a", encoding="utf-8") as f:
            f.write("\n# Added by self-healing config check\n")
            for key in missing_keys:
                f.write(f"{key}=\n")

    return created


def preflight_env() -> Tuple[bool, List[str], str]:
    """Validate startup credentials and auto-heal .env configuration."""
    path = _env_path()
    created = False
    try:
        created = _ensure_env_template(path)
    except Exception as exc:
        return (
            False,
            [],
            f"❌ Failed to create .env template at {path.resolve()}: {exc}",
        )

    try:
        if "load_dotenv" in globals():
            load_dotenv(override=False)
    except Exception:
        pass

    missing = _missing_required_env_vars()
    if not missing:
        return True, [], ""

    msg = (
        "⚠️ Generated .env file. Please fill in your credentials to proceed."
        if created
        else "⚠️ .env is missing required values. Please fill in your credentials to proceed."
    )
    return False, missing, msg


def _fail_missing_env(name: str) -> None:
    """Abort configuration due to missing credentials, ensuring clear feedback."""
    global _ENV_HINT_SHOWN
    ok, missing, msg = preflight_env()
    should_print = False

    with _ENV_HINT_LOCK:
        if not ok and not _ENV_HINT_SHOWN:
            _ENV_HINT_SHOWN = True
            should_print = True

    if should_print:
        if msg:
            print(msg)
        if missing:
            print(f"Missing env vars: {', '.join(missing)}")
        print(f"Please update: {_env_path().resolve()}")

    raise OSError(f"Missing required env var: {name}")


def _read_secret_file(path_raw: str) -> str:
    path_s = str(path_raw or "").strip()
    if not path_s:
        return ""
    path = Path(path_s)
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        raise OSError(f"Failed to read secret file: {path}") from exc


def _secret_key_value() -> str:
    for name in _SECRET_KEY_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value

    key_file = os.environ.get("CONFIG_SECRET_KEY_FILE", "").strip()
    if key_file:
        return _read_secret_file(key_file)

    return ""


def _decrypt_secret_value(token: str, source_name: str) -> str:
    token_s = str(token or "").strip()
    if not token_s:
        return ""

    secret_key = _secret_key_value()
    if not secret_key:
        raise OSError(
            f"Encrypted credential set for {source_name}, but CONFIG_SECRET_KEY is missing"
        )

    if Fernet is None:
        raise OSError(
            "Encrypted credentials require the 'cryptography' package to be installed"
        )

    try:
        return (
            Fernet(secret_key.encode("utf-8"))
            .decrypt(token_s.encode("utf-8"))
            .decode("utf-8")
            .strip()
        )
    except InvalidToken as exc:
        raise OSError(
            f"Encrypted credential for {source_name} could not be decrypted"
        ) from exc


def _env_secret_value(name: str, alias: Optional[str] = None) -> str:
    candidates = [str(name or "").strip()]
    if alias:
        candidates.append(str(alias).strip())

    for candidate in candidates:
        value = os.environ.get(candidate, "").strip()
        if value:
            return value

    for candidate in candidates:
        file_path = os.environ.get(f"{candidate}_FILE", "").strip()
        if file_path:
            value = _read_secret_file(file_path)
            if value:
                return value

    for candidate in candidates:
        token = os.environ.get(f"{candidate}_ENC", "").strip()
        if token:
            return _decrypt_secret_value(token, f"{candidate}_ENC")

    for candidate in candidates:
        token_file = os.environ.get(f"{candidate}_ENC_FILE", "").strip()
        if token_file:
            token = _read_secret_file(token_file)
            if token:
                return _decrypt_secret_value(token, f"{candidate}_ENC_FILE")

    return ""


def _env_required(name: str, alias: Optional[str] = None) -> str:
    """Require an env var to be set for live startup."""
    v = _env_secret_value(name, alias=alias)

    if not v:
        if _allow_missing_tg() and name in (
            "TG_TOKEN",
            "TG_ADMIN_ID",
            "BOT_TOKEN",
            "ADMIN_ID",
        ):
            return ""
        if _is_dry_run():
            return ""
        _fail_missing_env(name)

    return v


def _env_int(
    name: str, default: Optional[int] = None, alias: Optional[str] = None
) -> int:
    """Fetch an environment integer with fallback behavior."""
    raw = os.environ.get(name, "").strip()
    if not raw and alias:
        raw = os.environ.get(alias, "").strip()

    if not raw:
        if default is not None:
            return default
        if _is_dry_run():
            return 0
        _fail_missing_env(name)

    try:
        return int(raw)
    except ValueError:
        raise OSError(f"CONFIG ERROR: '{name}' must be an integer, got '{raw}'")


def _env_float(name: str, default: Optional[float] = None) -> float:
    """Fetch an environment float with fallback behavior."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        if default is not None:
            return default
        _fail_missing_env(name)
    try:
        return float(raw)
    except ValueError:
        raise OSError(f"CONFIG ERROR: '{name}' must be a float, got '{raw}'")


def _env_bool(name: str, default: bool = False) -> bool:
    """Safely interpret standard boolean expressions from environment variables."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _BOOL_TRUE:
        return True
    if raw in _BOOL_FALSE:
        return False
    return default


# =============================================================================
# Asset & Parameter Validators
# =============================================================================
def _validate_tf(tf: str, field_name: str) -> None:
    if tf not in TF_MAP and TF_MAP:
        raise ValueError(f"{field_name} '{tf}' not in TF_MAP: {list(TF_MAP.keys())}")


def _validate_pct01(x: float, name: str) -> None:
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {x}")


def _validate_pos(x: float, name: str) -> None:
    if x <= 0:
        raise ValueError(f"{name} must be > 0, got {x}")


def _parse_sessions(raw: str) -> List[Tuple[int, int]]:
    """Parse session limits map, handling formats like '0-24'."""
    sessions = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        s, e = part.split("-")
        sessions.append((int(s.strip()), int(e.strip())))
    return sessions


# =============================================================================
# Config Dataclasses
# =============================================================================
@dataclass
class BaseSymbolParams:
    """Base asset tuning structure for shared exchange variables."""

    base: str = ""
    resolved: Optional[str] = None
    tf_primary: str = "M1"
    tf_confirm: str = "M5"
    tf_long: str = "M15"
    entry_mode: str = "market"
    micro_window_sec: int = 3
    micro_min_tps: float = 1.0
    micro_max_tps: float = 220.0
    micro_imb_thresh: float = 0.24
    micro_spread_med_x: float = 1.55
    quote_flips_max: int = 22
    micro_tstat_thresh: float = 0.50
    pullback_atr_mult: float = 0.28
    spread_limit_pct: float = 0.00025

    lot_step: float = 0.01
    lot_min: float = 0.01
    lot_max: float = 100.0
    hard_lot_cap: float = 1.0
    contract_size: float = 100.0
    digits: int = 2
    point_value: float = 1.0

    is_24_7: bool = False
    tf_daily: str = "D1"
    market_start_minutes: int = 0
    market_end_minutes: int = 1440
    rollover_blackout_start: int = 0
    rollover_blackout_end: int = 0

    @property
    def symbol(self) -> str:
        return self.resolved or self.base

    def validate(self) -> None:
        if TF_MAP:
            _validate_tf(self.tf_primary, "tf_primary")
            _validate_tf(self.tf_confirm, "tf_confirm")
            _validate_tf(self.tf_long, "tf_long")
            _validate_tf(self.tf_daily, "tf_daily")


@dataclass
class BTCSymbolParams(BaseSymbolParams):
    """Overrides for Bitcoin execution dynamics."""

    base: str = "BTCUSDm"
    is_24_7: bool = True
    micro_max_tps: float = 320.0
    quote_flips_max: int = 26
    pullback_atr_mult: float = 0.30
    atr_rel_hi: float = 0.0100
    bb_width_range_max: float = 0.0080
    spread_limit_pct: float = 0.0005

    contract_size: float = 1.0
    hard_lot_cap: float = 0.50
    digits: int = 2
    point_value: float = 1.0


@dataclass
class XAUSymbolParams(BaseSymbolParams):
    """Overrides for Gold execution dynamics."""

    base: str = "XAUUSDm"
    is_24_7: bool = False
    market_start_minutes: int = 1
    market_end_minutes: int = 1439
    rollover_blackout_start: int = 1435
    rollover_blackout_end: int = 65
    spread_limit_pct: float = 0.00025

    contract_size: float = 100.0
    digits: int = 3
    point_value: float = 0.1


@dataclass
class BaseEngineConfig:
    """Unified engine risk logic matrix across all tradeable assets."""

    login: int = 0
    password: str = ""
    server: str = ""
    telegram_token: str = ""
    admin_id: int = 0

    symbol_params: BaseSymbolParams = field(default_factory=BaseSymbolParams)

    http_pool_conn: int = 20
    http_pool_max: int = 20

    daily_target_pct: float = 0.02
    max_daily_loss_pct: float = 0.04
    protect_drawdown_from_peak_pct: float = 0.03
    max_drawdown: float = 0.06
    max_risk_per_trade: float = 0.015
    analysis_cooldown_sec: float = 12.0
    max_orders_per_signal: int = 1
    max_open_positions_per_asset: int = 5
    max_concurrent_positions_total: int = 5
    signal_cooldown_sec_override: float = 0.0

    atr_sl_multiplier: float = 2.5
    atr_tp_min_multiplier: float = 3.0
    atr_tp_max_multiplier: float = 5.0
    breakeven_trigger_pct: float = 0.40
    trailing_stop_atr_mult: float = 1.5
    ker_period: int = 10
    rvi_period: int = 14
    rvi_weight: float = 1.0
    ker_chaos_mult: float = 1.0

    phase_a_target: float = 0.01
    phase_b_target: float = 0.015

    kill_min_trades: int = 6
    kill_expectancy: float = -0.5
    kill_winrate: float = 0.30
    cooling_expectancy: float = -0.2
    cooling_sec: float = 600.0

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "trend": 24.0,
            "momentum": 15.0,
            "volatility": 10.0,
            "structure": 12.0,
            "flow": 15.0,
            "tick_momentum": 12.0,
            "volume_delta": 8.0,
            "mean_reversion": 4.0,
        }
    )

    min_confidence: int = 75
    signal_min_score: float = 70.0
    ml_bridge_enabled: bool = True
    ml_bridge_min_confidence: float = 0.80
    ml_bridge_allow_neutral_pipeline: bool = True
    analysis_notify_live_ml: bool = True
    same_bar_retry_sec: float = 2.5
    adx_min: float = 20.0
    adx_trend_min: float = 25.0
    high_accuracy_mode: bool = True
    low_volume_sniper: float = 0.15

    mtf_penalty_enabled: bool = True
    mtf_slope_thresh: float = 0.10
    mtf_m5_penalty: float = 0.20
    mtf_m15_penalty: float = 0.50

    mtf_hard_gate_enabled: bool = True
    mtf_h1_flow_gate: float = 0.18
    mtf_h4_flow_gate: float = 0.22
    mtf_h4_veto_enabled: bool = True

    stop_hunt_min_strength: float = 0.30
    stop_hunt_align_bonus: int = 8
    stop_hunt_conflict_penalty: int = 16
    stop_hunt_conflict_veto_strength: float = 0.55

    macro_gate_enabled: bool = True
    macro_gate_ttl_sec: float = 15.0
    macro_dxy_weight: float = 0.60
    macro_us10y_weight: float = 0.40
    macro_bias_block: float = 0.28
    macro_bias_penalty: float = 0.12

    exec_max_p95_latency_ms: float = 550.0
    exec_max_p95_slippage_points: float = 20.0
    exec_max_spread_points: float = 500.0
    exec_max_ewma_slippage_points: float = 15.0
    exec_breaker_sec: float = 120.0

    active_sessions: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 24)])
    ignore_sessions: bool = True

    magic: int = 777001
    comment: str = "portfolio"
    pause_analysis_on_position: bool = True
    hedge_flip_enabled: bool = False
    pyramid_enabled: bool = False
    tz_local: str = "Asia/Dushanbe"

    poll_seconds_fast: float = 1.0
    market_min_bar_age_sec: float = 30.0
    market_max_bar_age_mult: float = 2.0
    ignore_microstructure: bool = False
    spread_cb_pct: float = 0.0010
    max_spread_points: float = 350.0
    flash_crash_sigma: float = 3.5
    flash_crash_lookback: int = 30
    flash_crash_cooldown_sec: float = 300.0
    circuit_breaker_atr_ratio: float = 3.0
    circuit_breaker_gap_atr_mult: float = 2.0
    circuit_breaker_cooldown_sec: float = 1800.0
    spread_spike_sigma: float = 3.0
    spread_spike_sigma_threshold: float = 3.0
    spread_gate_multiplier: float = 1.5

    d1_confluence_weight: float = 5.0
    d1_bars_required: int = 60

    xau_overlap_start_hour_utc: int = 12
    xau_overlap_end_hour_utc: int = 16
    xau_overlap_trail_mult: float = 0.80

    vol_lookback: int = 80
    extreme_lookback: int = 120
    conformal_window: int = 300
    brier_window: int = 800
    meta_h_bars: int = 6

    telegram_read_timeout: int = 60
    telegram_connect_timeout: int = 60

    ema_short: int = 8
    ema_medium: int = 21
    ema_slow: int = 50
    ema_anchor: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    cci_period: int = 20
    mom_period: int = 10
    roc_period: int = 10
    stoch_k_period: int = 14
    stoch_slowk_period: int = 3
    stoch_slowd_period: int = 3
    stoch_matype: int = 0

    def validate(self) -> None:
        _validate_pct01(self.daily_target_pct, "daily_target_pct")
        _validate_pct01(self.max_daily_loss_pct, "max_daily_loss_pct")
        _validate_pct01(self.max_risk_per_trade, "max_risk_per_trade")
        _validate_pos(self.atr_sl_multiplier, "atr_sl_multiplier")
        _validate_pos(self.atr_tp_min_multiplier, "atr_tp_min_multiplier")
        self.symbol_params.validate()

    @property
    def indicator(self) -> Dict[str, int]:
        return {
            "ema_short": self.ema_short,
            "ema_medium": self.ema_medium,
            "ema_slow": self.ema_slow,
            "ema_anchor": self.ema_anchor,
            "rsi_period": self.rsi_period,
            "atr_period": self.atr_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "adx_period": self.adx_period,
            "bb_period": self.bb_period,
            "cci_period": self.cci_period,
            "mom_period": self.mom_period,
            "roc_period": self.roc_period,
            "stoch_k_period": self.stoch_k_period,
            "stoch_slowk_period": self.stoch_slowk_period,
            "stoch_slowd_period": self.stoch_slowd_period,
            "stoch_matype": self.stoch_matype,
        }


@dataclass
class BTCEngineConfig(BaseEngineConfig):
    """Bitcoin specialized tuning configurations."""

    symbol_params: BaseSymbolParams = field(default_factory=BTCSymbolParams)
    magic: int = 777002
    comment: str = "btc_scalp"

    atr_sl_multiplier: float = 3.0
    atr_tp_min_multiplier: float = 4.0
    atr_tp_max_multiplier: float = 6.0

    active_sessions: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 24)])
    ignore_sessions: bool = True

    poll_seconds_fast: float = 0.12
    market_min_bar_age_sec: float = 120.0
    spread_cb_pct: float = 0.0075
    max_spread_points: float = 2500.0

    cci_period: int = 30
    mom_period: int = 14
    roc_period: int = 14
    stoch_k_period: int = 14
    stoch_slowk_period: int = 5
    stoch_slowd_period: int = 5


@dataclass
class XAUEngineConfig(BaseEngineConfig):
    """Gold specialized tuning configurations."""

    symbol_params: BaseSymbolParams = field(default_factory=XAUSymbolParams)
    magic: int = 777001
    comment: str = "xau_scalp"

    cci_period: int = 20
    mom_period: int = 10
    roc_period: int = 10
    stoch_k_period: int = 14
    stoch_slowk_period: int = 3
    stoch_slowd_period: int = 3

    atr_sl_multiplier: float = 2.5
    atr_tp_min_multiplier: float = 3.0
    atr_tp_max_multiplier: float = 3.0


# =============================================================================
# Helper Datatable Dataclasses
# =============================================================================
@dataclass
class KillSwitchState:
    """Manages consecutive losses vs expectancy drops to enforce trading pauses."""

    status: str = "ACTIVE"
    cooling_until_ts: float = 0.0
    last_expectancy: float = 0.0
    last_winrate: float = 0.0
    last_trades: int = 0

    def update(
        self,
        profits: List[float],
        now: float,
        min_trades: int,
        kill_expectancy: float,
        kill_winrate: float,
        cooling_expectancy: float,
        cooling_sec: float,
    ) -> None:
        n = len(profits)
        self.last_trades = n
        if n < min_trades:
            self.status = "ACTIVE"
            return

        wins = sum(1 for p in profits if p > 0)
        wr = wins / n if n > 0 else 0.0
        exp = sum(profits) / n if n > 0 else 0.0

        self.last_expectancy = exp
        self.last_winrate = wr

        if exp < kill_expectancy and wr < kill_winrate:
            self.cooling_until_ts = now + cooling_sec
            self.status = "KILLED"
            return

        if self.status == "KILLED" and now < self.cooling_until_ts:
            return

        if now < self.cooling_until_ts:
            self.status = "COOLING"
            return

        if exp < cooling_expectancy:
            self.cooling_until_ts = now + cooling_sec
            self.status = "COOLING"
            return

        self.status = "ACTIVE"


@dataclass
class AccountCache:
    ts: float = 0.0
    balance: float = 0.0
    equity: float = 0.0
    margin_free: float = 0.0


@dataclass
class SignalThrottle:
    daily_count: int = 0
    hour_window_start_ts: float = field(default_factory=time.time)
    hour_window_count: int = 0

    def register(self, max_per_hour: int = 20) -> bool:
        now = time.time()
        hour_start = datetime.fromtimestamp(now, tz=timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        hour_start_ts = float(hour_start.timestamp())
        if hour_start_ts != float(self.hour_window_start_ts or 0.0):
            self.hour_window_start_ts = hour_start_ts
            self.hour_window_count = 0
        if self.hour_window_count >= max_per_hour:
            return False
        self.hour_window_count += 1
        self.daily_count += 1
        return True


@dataclass
class SignalResult:
    signal: str = "Neutral"
    symbol: str = ""
    confidence: int = 0
    spread_pct: float = 0.0
    regime: str = "normal"
    signal_id: str = ""
    bar_key: str = "no_bar"
    reasons: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    trade_blocked: bool = False
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lot: float = 0.0
    timeframe: str = ""
    data_state: str = "data valid"
    feature_state: str = "features valid"


@dataclass
class AssetCandidate:
    asset: str = ""
    signal: str = "Neutral"
    confidence: int = 0
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lot: float = 0.0
    signal_id: str = ""
    bar_key: str = "no_bar"
    reasons: List[str] = field(default_factory=list)
    spread_pct: float = 0.0
    regime: str = ""
    timeframe: str = ""
    trade_blocked: bool = False


@dataclass
class OrderIntent:
    symbol: str = ""
    signal: str = ""
    lot: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    price: float = 0.0
    order_id: str = ""
    signal_id: str = ""
    enqueue_time: float = 0.0
    confidence: float = 0.0
    cfg: object = None
    risk_manager: object = None


@dataclass
class ExecutionResult:
    order_id: str = ""
    signal_id: str = ""
    ok: bool = False
    reason: str = ""
    sent_ts: float = 0.0
    fill_ts: float = 0.0
    req_price: float = 0.0
    exec_price: float = 0.0
    volume: float = 0.0
    slippage: float = 0.0
    retcode: int = 0
    order_ticket: int = 0
    deal_ticket: int = 0
    position_ticket: int = 0


@dataclass
class PortfolioStatus:
    timestamp: float = 0.0
    total_equity: float = 0.0
    total_margin_used: float = 0.0
    open_positions_btc: int = 0
    open_positions_xau: int = 0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    phase_btc: str = "A"
    phase_xau: str = "A"
    mt5_connected: bool = False
    heartbeat_status: str = "ALIVE"


# =============================================================================
# Factory Generators & Decorators
# =============================================================================
def get_btc_config_from_env() -> BTCEngineConfig:
    cfg = BTCEngineConfig(
        login=_env_int("EXNESS_LOGIN"),
        password=_env_required("EXNESS_PASSWORD"),
        server=_env_required("EXNESS_SERVER"),
        telegram_token=_env_required("TG_TOKEN", alias="BOT_TOKEN"),
        admin_id=_env_int("TG_ADMIN_ID", alias="ADMIN_ID"),
    )
    cfg.validate()
    return cfg


def get_xau_config_from_env() -> XAUEngineConfig:
    cfg = XAUEngineConfig(
        login=_env_int("EXNESS_LOGIN"),
        password=_env_required("EXNESS_PASSWORD"),
        server=_env_required("EXNESS_SERVER"),
        telegram_token=_env_required("TG_TOKEN", alias="BOT_TOKEN"),
        admin_id=_env_int("TG_ADMIN_ID", alias="ADMIN_ID"),
    )
    cfg.validate()
    return cfg


def get_config_from_env(asset: str = "XAU") -> BaseEngineConfig:
    asset_norm = str(asset).upper().strip()
    if asset_norm in ("BTC", "BTCUSD", "BTCUSDm"):
        return get_btc_config_from_env()
    return get_xau_config_from_env()


def apply_high_accuracy_mode(cfg: BaseEngineConfig) -> BaseEngineConfig:
    cfg.high_accuracy_mode = True
    cfg.min_confidence = max(int(cfg.min_confidence), 85)
    cfg.signal_min_score = max(float(cfg.signal_min_score), 75.0)
    cfg.mtf_penalty_enabled = True
    cfg.spread_gate_multiplier = min(
        float(getattr(cfg, "spread_gate_multiplier", 1.5) or 1.5),
        1.5,
    )
    return cfg


EngineConfig = BaseEngineConfig
SymbolParams = BaseSymbolParams

# =============================================================================
# Module Exports
# =============================================================================
__all__ = (
    "ALLOWED_SYMBOLS",
    "AccountCache",
    "AssetCandidate",
    "BTCEngineConfig",
    "BTCSymbolParams",
    "BaseEngineConfig",
    "BaseSymbolParams",
    "EngineConfig",
    "ExecutionResult",
    "KillSwitchState",
    "MAX_GATE_DRAWDOWN",
    "MIN_GATE_SHARPE",
    "MIN_GATE_WIN_RATE",
    "OrderIntent",
    "PortfolioStatus",
    "SignalResult",
    "SignalThrottle",
    "SymbolParams",
    "TF_MAP",
    "WFA_MIN_PASS_RATE",
    "WFA_MIN_WINDOWS",
    "XAUEngineConfig",
    "XAUSymbolParams",
    "apply_high_accuracy_mode",
    "get_btc_config_from_env",
    "get_config_from_env",
    "get_xau_config_from_env",
    "preflight_env",
)
