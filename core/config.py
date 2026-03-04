# core/config.py — Unified configuration system with asset-specific overrides.
# Eliminates ~90% duplication between config_btc.py and config_xau.py.
from __future__ import annotations

import os
import threading
from types import MappingProxyType
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # Allow import without MT5 for testing

from .utils import clamp01


# =============================================================================
# Constants
# =============================================================================

TF_MAP: Dict[str, int] = {}
if mt5 is not None:
    TF_MAP = {
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
TF_MAP = MappingProxyType(dict(TF_MAP))

# =============================================================================
# Model/Backtest quality gates (single source of truth)
# =============================================================================
MIN_GATE_SHARPE: float = 0.5
MIN_GATE_WIN_RATE: float = 0.52
MAX_GATE_DRAWDOWN: float = 0.25

# Walk-forward acceptance (pass-rate based, not all-windows-must-pass)
WFA_MIN_WINDOWS: int = 3
WFA_MIN_PASS_RATE: float = 0.60


# =============================================================================
# ENV helpers (strict + fast) — shared by all configs
# =============================================================================

from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env from current or parent dirs
except ImportError:
    print("⚠️  WARNING: 'python-dotenv' not installed. .env file will NOT be loaded.")

_BOOL_TRUE = frozenset({"1", "true", "yes", "y", "on"})
_BOOL_FALSE = frozenset({"0", "false", "no", "n", "off"})

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

# ==========================================
# TELEGRAM NOTIFICATIONS
# ==========================================
TG_TOKEN=
TG_ADMIN_ID=
"""

_ENV_HINT_SHOWN = False
_ENV_HINT_LOCK = threading.Lock()


def _env_path() -> Path:
    return Path(".env")


def _env_first(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return ""


def _allow_missing_tg() -> bool:
    return _env_bool("ALLOW_MISSING_TG", default=True)


def _auto_dry_run_on_missing_env() -> bool:
    return _env_bool("AUTO_DRY_RUN_ON_MISSING_ENV", default=True)


def _is_dry_run() -> bool:
    return _env_bool("DRY_RUN", default=False) or _auto_dry_run_on_missing_env()


def _missing_required_env_vars() -> List[str]:
    missing: List[str] = []
    for aliases in _REQUIRED_ENV_GROUPS:
        # Optionally skip TG requirements
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
    """
    Ensure .env exists with the required keys.
    Returns True when the template file was created from scratch.
    """
    created = False
    if not path.exists():
        path.write_text(_ENV_TEMPLATE, encoding="utf-8")
        return True

    existing = _existing_env_keys(path)
    missing_keys: List[str] = []
    for aliases in _REQUIRED_ENV_GROUPS:
        if any(a in existing for a in aliases):
            continue
        missing_keys.append(aliases[0])
    if missing_keys:
        with path.open("a", encoding="utf-8") as f:
            f.write("\n# Added by self-healing config check\n")
            for key in missing_keys:
                f.write(f"{key}=\n")
    return created


def preflight_env() -> Tuple[bool, List[str], str]:
    """
    Validate startup credentials and auto-heal .env shape.
    Returns (ok, missing_vars, message).
    """
    path = _env_path()
    created = False
    try:
        created = _ensure_env_template(path)
    except Exception as exc:
        msg = f"❌ Failed to create .env template at {path.resolve()}: {exc}"
        return False, [], msg

    try:
        if "load_dotenv" in globals():
            load_dotenv(override=False)
    except Exception:
        pass

    missing = _missing_required_env_vars()
    if not missing:
        return True, [], ""

    if created:
        msg = "⚠️ Generated .env file. Please fill in your credentials to proceed."
    else:
        msg = "⚠️ .env is missing required values. Please fill in your credentials to proceed."
    return False, missing, msg



def _fail_missing_env(name: str):
    """
    Auto-healing: create/fix .env template and fail fast with clear guidance.
    """
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


def _env_required(name: str, alias: Optional[str] = None) -> str:
    """Return env value if present and non-empty, else exit gracefully."""
    v = os.environ.get(name, "").strip()
    if not v and alias:
        v = os.environ.get(alias, "").strip()

    if not v:
        # Allow missing TG if configured
        if _allow_missing_tg() and name in ("TG_TOKEN", "TG_ADMIN_ID", "BOT_TOKEN", "ADMIN_ID"):
            return ""
        # Allow missing env in dry-run mode
        if _is_dry_run():
            return ""
        _fail_missing_env(name)
    return v


def _env_int(name: str, default: Optional[int] = None, alias: Optional[str] = None) -> int:
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
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _BOOL_TRUE:
        return True
    if raw in _BOOL_FALSE:
        return False
    return default


# =============================================================================
# Validation helpers
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
    """Parse session string like '0-24' or '0-12,12-24' into list of (start, end) tuples."""
    sessions = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        s, e = part.split("-")
        sessions.append((int(s.strip()), int(e.strip())))
    return sessions


# =============================================================================
# Base Symbol Params (shared between BTC and XAU)
# =============================================================================

@dataclass
class BaseSymbolParams:
    """Asset-specific symbol parameters. Subclass for BTC/XAU."""
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
    # Contract / lot parameters (defaults for forex-style)
    lot_step: float = 0.01
    lot_min: float = 0.01
    lot_max: float = 100.0
    contract_size: float = 100.0      # 1 lot = 100 oz (XAU), 1 lot = 1 BTC, etc.
    digits: int = 2                    # Price precision (XAU=2, BTC=2)
    point_value: float = 1.0           # $ value per point per lot
    # Market hours
    is_24_7: bool = False
    # Daily timeframe for D1 confluence analysis
    tf_daily: str = "D1"
    market_start_minutes: int = 0      # Minutes from midnight UTC
    market_end_minutes: int = 1440     # Minutes from midnight UTC
    rollover_blackout_start: int = 0   # Minutes from midnight for rollover
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
    """BTC-specific symbol params — 24/7 crypto market."""
    base: str = "BTCUSDm"
    is_24_7: bool = True
    micro_max_tps: float = 320.0
    quote_flips_max: int = 26
    pullback_atr_mult: float = 0.30
    atr_rel_hi: float = 0.0100
    bb_width_range_max: float = 0.0080
    spread_limit_pct: float = 0.0005
    # BTC contract/lot specifics
    contract_size: float = 1.0         # 1 lot = 1 BTC
    digits: int = 2                    # BTC price to 2 decimals
    point_value: float = 1.0           # $1 per point per lot


@dataclass
class XAUSymbolParams(BaseSymbolParams):
    """XAU-specific symbol params — 24/5 forex market with rollover blackout."""
    base: str = "XAUUSDm"
    is_24_7: bool = False
    market_start_minutes: int = 1      # 00:01 UTC
    market_end_minutes: int = 1439     # 23:59 UTC
    rollover_blackout_start: int = 1435  # ~23:55 UTC (server time)
    rollover_blackout_end: int = 65      # ~01:05 UTC (server time) — 70min window
    spread_limit_pct: float = 0.00025
    # XAU contract/lot specifics
    contract_size: float = 100.0       # 1 lot = 100 oz gold
    digits: int = 2                    # Gold price to 2 decimals (e.g. 2045.30)
    point_value: float = 1.0           # $1 per point per lot


# =============================================================================
# Base Engine Config (shared fields)
# =============================================================================

@dataclass
class BaseEngineConfig:
    """
    Shared engine configuration. ~90% of fields are identical between BTC and XAU.
    Subclass for asset-specific overrides.
    """

    # ─── Credentials ─────────────────────────────────────────────
    login: int = 0
    password: str = ""
    server: str = ""
    telegram_token: str = ""
    admin_id: int = 0

    # ─── Symbol params ───────────────────────────────────────────
    symbol_params: BaseSymbolParams = field(default_factory=BaseSymbolParams)

    # ─── Network ─────────────────────────────────────────────────
    http_pool_conn: int = 20
    http_pool_max: int = 20

    # ─── Risk management ─────────────────────────────────────────
    daily_target_pct: float = 0.02
    max_daily_loss_pct: float = 0.04
    protect_drawdown_from_peak_pct: float = 0.03
    max_drawdown: float = 0.06
    max_risk_per_trade: float = 0.015             # 1.5% — conservative institutional default
    analysis_cooldown_sec: float = 12.0
    max_orders_per_signal: int = 1
    signal_cooldown_sec_override: float = 0.0

    # ─── ATR-based SL/TP ─────────────────────────────────────────
    atr_sl_multiplier: float = 2.5
    atr_tp_min_multiplier: float = 3.0   # Must be > atr_sl_multiplier for R:R >= 1.0
    atr_tp_max_multiplier: float = 5.0
    breakeven_trigger_pct: float = 0.40
    trailing_stop_atr_mult: float = 1.5
    # Fractal stop tuning
    ker_period: int = 10
    rvi_period: int = 14
    rvi_weight: float = 1.0
    ker_chaos_mult: float = 1.0

    # ─── Phase escalation ────────────────────────────────────────
    phase_a_target: float = 0.01     # +1% → Phase B
    phase_b_target: float = 0.015    # +1.5% → Phase C

    # ─── Kill switch ─────────────────────────────────────────────
    kill_min_trades: int = 6
    kill_expectancy: float = -0.5
    kill_winrate: float = 0.30
    cooling_expectancy: float = -0.2
    cooling_sec: float = 600.0

    # ─── Scoring weights ─────────────────────────────────────────
    weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 24.0,
        "momentum": 15.0,
        "volatility": 10.0,
        "structure": 12.0,
        "flow": 15.0,
        "tick_momentum": 12.0,
        "volume_delta": 8.0,
        "mean_reversion": 4.0,
    })

    # ─── Signal ──────────────────────────────────────────────────
    min_confidence: int = 80         # QUANTUM SNIPER MODE: Only high-prob trades
    signal_min_score: float = 70.0   # Minimum internal score to even consider
    adx_min: float = 20.0
    adx_trend_min: float = 25.0
    high_accuracy_mode: bool = True
    low_volume_sniper: float = 0.15
    # MTF confluence penalties
    mtf_penalty_enabled: bool = True
    mtf_slope_thresh: float = 0.10
    mtf_m5_penalty: float = 0.20
    mtf_m15_penalty: float = 0.50
    # Hard confluence veto: M15 entries must agree with H1/H4 institutional flow
    mtf_hard_gate_enabled: bool = True
    mtf_h1_flow_gate: float = 0.18
    mtf_h4_flow_gate: float = 0.22
    mtf_h4_veto_enabled: bool = True
    # Stop-hunt / liquidity sweep handling
    stop_hunt_min_strength: float = 0.30
    stop_hunt_align_bonus: int = 8
    stop_hunt_conflict_penalty: int = 16
    stop_hunt_conflict_veto_strength: float = 0.55
    # XAU macro pressure gate (DXY + US10Y)
    macro_gate_enabled: bool = True
    macro_gate_ttl_sec: float = 15.0
    macro_dxy_weight: float = 0.60
    macro_us10y_weight: float = 0.40
    macro_bias_block: float = 0.28
    macro_bias_penalty: float = 0.12

    # ─── Execution quality ───────────────────────────────────────
    exec_max_p95_latency_ms: float = 550.0
    exec_max_p95_slippage_points: float = 20.0
    exec_max_spread_points: float = 500.0
    exec_max_ewma_slippage_points: float = 15.0
    exec_breaker_sec: float = 120.0

    # ─── Session ─────────────────────────────────────────────────
    active_sessions: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 24)])
    ignore_sessions: bool = True

    # ─── Operational ─────────────────────────────────────────────
    magic: int = 777001
    comment: str = "portfolio"
    pause_analysis_on_position: bool = True
    hedge_flip_enabled: bool = False
    pyramid_enabled: bool = False
    tz_local: str = "Asia/Dushanbe"    # Primary display timezone

    # ─── Market data / feed tuning ───────────────────────────────
    poll_seconds_fast: float = 1.0     # Market feed polling interval (seconds)
    market_min_bar_age_sec: float = 30.0
    market_max_bar_age_mult: float = 2.0
    ignore_microstructure: bool = False
    spread_cb_pct: float = 0.0010      # Spread circuit-breaker (% of price)
    max_spread_points: float = 350.0   # Maximum allowed spread in points
    # Flash-crash guard (BTC)
    flash_crash_sigma: float = 3.5
    flash_crash_lookback: int = 30
    flash_crash_cooldown_sec: float = 300.0
    # ─── Volatility circuit breaker (Black Swan protection) ────────
    circuit_breaker_atr_ratio: float = 3.0    # ATR/SMA(ATR,50) > 3 = extreme
    circuit_breaker_gap_atr_mult: float = 2.0 # Price gap > 2×ATR between bars
    circuit_breaker_cooldown_sec: float = 1800.0  # 30min cooldown after trigger
    # ─── Spread spike detection ───────────────────────────────────
    spread_spike_sigma: float = 3.0           # Block when spread > 3σ of history
    spread_spike_sigma_threshold: float = 3.0 # Alias for compatibility with new guards
    spread_gate_multiplier: float = 1.5       # Tightened from 2.0 (institutional)
    # ─── D1 confluence scoring ────────────────────────────────────
    d1_confluence_weight: float = 5.0         # ±5 pts in 100-point scoring
    d1_bars_required: int = 60                # Minimum D1 bars for analysis
    # XAU session overlap tightening (UTC hours)
    xau_overlap_start_hour_utc: int = 12
    xau_overlap_end_hour_utc: int = 16
    xau_overlap_trail_mult: float = 0.80

    # ─── Lookback / window params for feeds ──────────────────────
    vol_lookback: int = 80
    extreme_lookback: int = 120
    conformal_window: int = 300
    brier_window: int = 800
    meta_h_bars: int = 6

    # ─── Telegram ─────────────────────────────────────────────────
    telegram_read_timeout: int = 60
    telegram_connect_timeout: int = 60

    # ─── Indicator config ────────────────────────────────────────
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
        """Validate all config fields."""
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
    """BTC-specific engine config — 24/7 crypto market."""
    symbol_params: BaseSymbolParams = field(default_factory=BTCSymbolParams)
    magic: int = 777002
    comment: str = "btc_scalp"
    # BTC has higher volatility → wider ATR multipliers
    atr_sl_multiplier: float = 3.0     # BTC needs wider SL due to higher vol
    atr_tp_min_multiplier: float = 4.0   # Must be > SL mult for R:R >= 1.0
    atr_tp_max_multiplier: float = 6.0  # Wider max TP range for crypto moves
    # Crypto sessions are 24/7
    active_sessions: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 24)])
    ignore_sessions: bool = True
    # BTC-specific market data tuning
    poll_seconds_fast: float = 0.12    # Crypto needs fast polling
    market_min_bar_age_sec: float = 120.0
    spread_cb_pct: float = 0.0075      # BTC spreads are wider (~0.75%)
    max_spread_points: float = 2500.0   # BTC spread ceiling in points
    # BTC indicator tuning (more smoothing)
    cci_period: int = 30
    mom_period: int = 14
    roc_period: int = 14
    stoch_k_period: int = 14
    stoch_slowk_period: int = 5
    stoch_slowd_period: int = 5


@dataclass
class XAUEngineConfig(BaseEngineConfig):
    """XAU-specific engine config — 24/5 forex with rollover blackout."""
    symbol_params: BaseSymbolParams = field(default_factory=XAUSymbolParams)
    magic: int = 777001
    comment: str = "xau_scalp"
    # XAU indicator tuning (faster response)
    cci_period: int = 20
    mom_period: int = 10
    roc_period: int = 10
    stoch_k_period: int = 14
    stoch_slowk_period: int = 3
    stoch_slowd_period: int = 3
    # Gold has tighter spreads → tighter ATR multipliers
    atr_sl_multiplier: float = 2.5     # Tighter SL for gold's lower relative vol
    atr_tp_min_multiplier: float = 3.0   # Must be > SL mult for R:R >= 1.0
    atr_tp_max_multiplier: float = 3.0


# =============================================================================
# Factory functions
# =============================================================================

def get_btc_config_from_env() -> BTCEngineConfig:
    """Build BTC config from environment variables."""
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
    """Build XAU config from environment variables."""
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
    """Unified dispatcher — returns the right config for the given asset."""
    asset = str(asset).upper().strip()
    if asset in ("BTC", "BTCUSD", "BTCUSDm"):
        return get_btc_config_from_env()
    return get_xau_config_from_env()


ALLOWED_SYMBOLS = ("XAUUSDm", "XAUUSDm.", "BTCUSDm", "BTCUSDm.")


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


# Back-compat aliases used by legacy callers (mt5_client, bot_utils, etc.)
EngineConfig = BaseEngineConfig
SymbolParams = BaseSymbolParams
