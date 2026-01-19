# config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5
from dotenv import load_dotenv
from log_config import get_log_path


load_dotenv()

ALLOWED_SYMBOLS: Tuple[str, ...] = ("XAUUSDm",)
XAU_SESSIONS_DEFAULT: List[Tuple[int, int]] = [(0, 24)]

GOLD_MARKET_START_MINUTES: int = 0 * 60 + 5
GOLD_MARKET_END_MINUTES: int = 23 * 60 + 55

TF_MAP: Dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
}


def _env_required(name: str) -> str:
    """Return env value if present and non-empty, else raise."""
    val = os.getenv(name)
    if val is None or not str(val).strip():
        raise RuntimeError(f".env error: missing/empty variable → {name}")
    return str(val).strip()


def _env_int(name: str, default: Optional[int] = None) -> int:
    val = os.getenv(name)
    if val is None or not str(val).strip():
        if default is None:
            raise RuntimeError(f".env error: missing/empty variable → {name}")
        return int(default)
    try:
        return int(str(val).strip())
    except ValueError as exc:
        raise RuntimeError(f".env error: invalid int for {name} = {val!r}") from exc


def _env_float(name: str, default: Optional[float] = None) -> float:
    val = os.getenv(name)
    if val is None or not str(val).strip():
        if default is None:
            raise RuntimeError(f".env error: missing/empty variable → {name}")
        return float(default)
    try:
        return float(str(val).strip())
    except ValueError as exc:
        raise RuntimeError(f".env error: invalid float for {name} = {val!r}") from exc


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None or not str(val).strip():
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _validate_tf(tf: str, field_name: str) -> None:
    if tf not in TF_MAP:
        raise RuntimeError(f"config error: invalid {field_name}={tf!r}. Allowed={list(TF_MAP.keys())}")


def _validate_sessions(sessions: List[Tuple[int, int]]) -> None:
    if not sessions:
        raise RuntimeError("config error: active_sessions is empty")
    for start_h, end_h in sessions:
        if not (0 <= int(start_h) <= 24 and 0 <= int(end_h) <= 24):
            raise RuntimeError(f"config error: invalid session hours {(start_h, end_h)}; must be in [0..24]")
        if int(start_h) == int(end_h):
            raise RuntimeError(f"config error: session {(start_h, end_h)} is zero-length")
        # Режими шабонарӯзӣ: (0,24) иҷозат, дигар ҳолатҳо бояд start < end
        if (int(start_h), int(end_h)) != (0, 24) and int(start_h) > int(end_h):
            raise RuntimeError(f"config error: session {(start_h, end_h)} crosses midnight; split into two sessions")


@dataclass
class SymbolParams:
    base: str = "XAUUSDm"
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

    def validate(self) -> None:
        sym = self.resolved or self.base
        if sym not in ALLOWED_SYMBOLS:
            raise RuntimeError(f"config error: symbol {sym!r} not allowed. Allowed={ALLOWED_SYMBOLS}")
        _validate_tf(self.tf_primary, "tf_primary")
        _validate_tf(self.tf_confirm, "tf_confirm")
        _validate_tf(self.tf_long, "tf_long")

        if self.entry_mode not in {"market"}:
            raise RuntimeError(f"config error: entry_mode {self.entry_mode!r} not supported (only 'market')")

        if self.micro_window_sec <= 0:
            raise RuntimeError("config error: micro_window_sec must be > 0")
        if self.micro_min_tps <= 0 or self.micro_max_tps <= 0:
            raise RuntimeError("config error: micro_min_tps/micro_max_tps must be > 0")
        if self.micro_min_tps >= self.micro_max_tps:
            raise RuntimeError("config error: micro_min_tps must be < micro_max_tps")
        if not (0.0 < self.spread_limit_pct < 0.01):
            raise RuntimeError("config error: spread_limit_pct out of range")


@dataclass
class EngineConfig:
    login: int
    password: str
    server: str
    telegram_token: str
    admin_id: int

    symbol_params: SymbolParams = field(default_factory=SymbolParams)

    # Telegram / HTTP client tuning (no ENV)
    http_pool_conn: int = 20
    http_pool_max: int = 20
    telegram_read_timeout: int = 60
    telegram_connect_timeout: int = 60

    tz_local: str = "Asia/Dushanbe"
    active_sessions: List[Tuple[int, int]] = field(default_factory=lambda: list(XAU_SESSIONS_DEFAULT))

    # MT5 control (used by mt5_client.py)
    mt5_path: Optional[str] = None
    mt5_portable: bool = False
    mt5_autostart: bool = True
    mt5_timeout_ms: int = 10_000

    daily_target_pct: float = 0.10
    ultra_confidence_min: float = 0.95
    protect_drawdown_from_peak_pct: float = 0.30
    max_daily_loss_pct: float = 0.10
    daily_loss_b_pct: float = 0.02
    daily_loss_c_pct: float = 0.03
    enforce_daily_limits: bool = True
    ignore_daily_stop_for_trading: bool = False
    enforce_drawdown_limits: bool = False
    ignore_external_positions: bool = True
    magic: int = 777001

    min_confidence_signal: float = 0.84
    conf_min: int = 88
    conf_min_low: int = 80
    conf_min_high: int = 93

    adx_trend_lo: float = 18.0
    adx_trend_hi: float = 27.0
    atr_rel_lo: float = 0.00055
    atr_rel_hi: float = 0.0028
    min_body_pct_of_atr: float = 0.12
    min_bar_age_sec: int = 1

    ema_short: int = 9
    ema_mid: int = 21
    ema_long: int = 50
    ema_vlong: int = 200
    atr_period: int = 14
    rsi_period: int = 14
    adx_period: int = 14
    vol_lookback: int = 80
    rn_step: float = 20.0

    poll_seconds_fast: float = 0.05
    decision_debounce_ms: float = 50.0
    analysis_cooldown_sec: float = 0.0
    cooldown_seconds: float = 0.0
    overnight_block_hours: float = 3.0
    signal_cooldown_sec_override: Optional[float] = None

    health_window_minutes: int = 180
    enable_telemetry: bool = True
    enable_debug_logging: bool = False

    # Telemetry correlation (keep consistent with bot)
    correlation_symbol: str = "pax-gold"
    correlation_vs_currency: str = "usd"
    correlation_refresh_sec: int = 300

    fixed_volume: Optional[float] = None
    max_risk_per_trade: float = 0.02
    max_positions: int = 3
    multi_order_tp_bonus_pct: float = 0.18
    multi_order_sl_tighten_pct: float = 0.25
    multi_order_confidence_tiers: Tuple[float, float] = (0.94, 0.97)
    multi_order_max_orders: int = 3

    islamic_min_leverage: int = 1
    require_swap_free: bool = False

    max_drawdown: float = 0.09
    max_trades_per_hour: int = 20
    max_signals_per_day: int = 0

    sl_atr_mult_trend: float = 1.25
    tp_atr_mult_trend: float = 2.5
    sl_atr_mult_range: float = 1.45
    tp_atr_mult_range: float = 1.8
    tp_rr_cap: float = 1.8
    min_rr: float = 1.05
    sltp_cost_spread_mult: float = 1.8
    sltp_cost_slip_mult: float = 1.0
    sltp_cost_move_mult: float = 0.6
    sltp_sl_floor_mult: float = 1.15
    sltp_tp_floor_mult: float = 1.6
    be_trigger_R: float = 0.8
    be_lock_spread_mult: float = 1.25
    trail_atr_mult: float = 1.05

    signal_amplification: float = 1.25
    weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 0.55,
        "momentum": 0.27,
        "meanrev": 0.10,
        "structure": 0.05,
        "volume": 0.03,
    })

    adaptive_enabled: bool = True
    trail_on_entry: bool = True
    use_squeeze_filter: bool = True
    hedge_flip_enabled: bool = False
    pyramid_enabled: bool = False

    log_csv_path: str = field(default_factory=lambda: str(get_log_path("signals_error_only.csv")))

    extreme_lookback: int = 120
    extreme_atr_mult: float = 1.7
    extreme_near_pct: float = 0.12
    rn_step_xau: float = 5.0
    rn_buffer_pct: float = 0.0012

    atr_percentile_lookback: int = 1000
    tod_boost_minutes: int = 120
    slippage_window: int = 40
    slippage_limit_ticks: float = 15.0
    latency_rtt_ms_limit: int = 300
    cooldown_after_latency_s: int = 300

    ultimate_mode: bool = True
    ensemble_w: Tuple[float, float, float] = (0.55, 0.33, 0.12)
    meta_barrier_R: float = 0.65
    meta_h_bars: int = 6
    conformal_window: int = 300
    conformal_q: float = 0.88
    brier_window: int = 800
    tc_bps: float = 1.5
    rtt_cb_ms: int = 450
    spread_cb_pct: float = 0.0010
    slippage_backoff: float = 0.5

    # Policy toggles
    ignore_sessions: bool = True
    pause_analysis_on_position_open: bool = False
    ignore_microstructure: bool = True

    def validate(self) -> None:
        if self.login <= 0:
            raise RuntimeError("config error: login must be > 0")
        if not self.password.strip():
            raise RuntimeError("config error: password empty")
        if not self.server.strip():
            raise RuntimeError("config error: server empty")
        if not self.telegram_token.strip():
            raise RuntimeError("config error: telegram_token empty")
        if self.admin_id <= 0:
            raise RuntimeError("config error: admin_id must be > 0")

        self.symbol_params.validate()
        _validate_sessions(self.active_sessions)

        if not (0.0 < self.max_risk_per_trade <= 0.10):
            raise RuntimeError("config error: max_risk_per_trade out of range (0..0.10]")
        if self.max_positions <= 0:
            raise RuntimeError("config error: max_positions must be > 0")
        if self.poll_seconds_fast <= 0:
            raise RuntimeError("config error: poll_seconds_fast must be > 0")

    @property
    def indicator(self) -> Dict[str, int]:
        return {
            "ema_short": int(self.ema_short),
            "ema_mid": int(self.ema_mid),
            "ema_long": int(self.ema_long),
            "ema_vlong": int(self.ema_vlong),
            "atr_period": int(self.atr_period),
            "rsi_period": int(self.rsi_period),
            "adx_period": int(self.adx_period),
            "vol_lookback": int(self.vol_lookback),
        }


def get_config_from_env() -> EngineConfig:
    cfg = EngineConfig(
        login=_env_int("EXNESS_LOGIN"),
        password=_env_required("EXNESS_PASSWORD"),
        server=_env_required("EXNESS_SERVER"),
        telegram_token=_env_required("BOT_TOKEN"),
        admin_id=_env_int("ADMIN_ID"),
    )

    cfg.validate()
    return cfg


def apply_high_accuracy_mode(cfg: EngineConfig, enable: bool = True) -> None:
    if not enable:
        return
    cfg.min_confidence_signal = 0.84
    cfg.ultra_confidence_min = 0.95
    cfg.max_risk_per_trade = 0.02
    cfg.tp_atr_mult_trend = 2.5
    cfg.use_squeeze_filter = True
    cfg.hedge_flip_enabled = False
    cfg.pyramid_enabled = False
    cfg.poll_seconds_fast = 0.05


__all__ = [
    "ALLOWED_SYMBOLS",
    "XAU_SESSIONS_DEFAULT",
    "GOLD_MARKET_START_MINUTES",
    "GOLD_MARKET_END_MINUTES",
    "TF_MAP",
    "SymbolParams",
    "EngineConfig",
    "get_config_from_env",
    "apply_high_accuracy_mode",
]
