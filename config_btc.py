# config.py  (BTCUSDm ONLY, 24/7, PRODUCTION-GRADE)
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover (env mismatch fallback)
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:  # type: ignore[misc]
        return False

from log_config import get_log_path

# Load .env once at import (secrets-only). Does not override already-set env vars.
load_dotenv(override=False)

# =============================================================================
# BTC ONLY / 24-7 ONLY
# =============================================================================

ALLOWED_SYMBOLS: Tuple[str, ...] = ("BTCUSDm",)
CRYPTO_SESSIONS_24_7: List[Tuple[int, int]] = [(0, 24)]

TF_MAP: Dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
}


# =============================================================================
# ENV helpers
# =============================================================================

def _env_required(name: str) -> str:
    val = os.getenv(name)
    if val is None or not str(val).strip():
        raise RuntimeError(f".env error: missing/empty variable → {name}")
    return str(val).strip()


def _env_int(name: str, default: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        if default is None:
            raise RuntimeError(f".env error: missing/empty variable → {name}")
        return int(default)

    s = str(raw).strip()
    try:
        return int(s)
    except ValueError as exc:
        raise RuntimeError(f".env error: invalid int for {name} = {raw!r}") from exc


def _env_float(name: str, default: Optional[float] = None) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        if default is None:
            raise RuntimeError(f".env error: missing/empty variable → {name}")
        return float(default)

    s = str(raw).strip()
    try:
        return float(s)
    except ValueError as exc:
        raise RuntimeError(f".env error: invalid float for {name} = {raw!r}") from exc


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _validate_tf(tf: str, field_name: str) -> None:
    if tf not in TF_MAP:
        allowed = ", ".join(TF_MAP.keys())
        raise RuntimeError(
            f"config error: invalid {field_name}={tf!r}. Allowed=[{allowed}]"
        )


def _parse_sessions(raw: str) -> List[Tuple[int, int]]:
    """
    ACTIVE_SESSIONS format:
      "0-24" or "0-12,12-24"
    For BTC we enforce 0-24, but parser is kept for strict validation.
    """
    out: List[Tuple[int, int]] = []
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    for p in parts:
        if "-" not in p:
            raise RuntimeError(
                f"config error: bad session item {p!r}, expected 'start-end'"
            )
        a_str, b_str = [x.strip() for x in p.split("-", 1)]
        try:
            a = int(a_str)
            b = int(b_str)
        except ValueError as exc:
            raise RuntimeError(
                f"config error: bad session item {p!r}, hours must be integers"
            ) from exc

        if not (0 <= a < b <= 24):
            raise RuntimeError(
                f"config error: bad session range {p!r}, expected 0<=start<end<=24"
            )
        out.append((a, b))
    return out


def _validate_sessions_24_7(sessions: List[Tuple[int, int]]) -> None:
    norm = [(int(a), int(b)) for a, b in (sessions or [])]
    if len(norm) != 1 or norm[0] != (0, 24):
        raise RuntimeError("config error: BTC must run 24/7. Set ACTIVE_SESSIONS=0-24")


# =============================================================================
# Symbol params (BTC scalping tuned)
# =============================================================================

@dataclass(slots=True)
class SymbolParams:
    base: str = "BTCUSDm"
    resolved: Optional[str] = None

    # Timeframes (BTC scalping)
    tf_primary: str = "M1"
    tf_confirm: str = "M5"
    tf_long: str = "M15"

    entry_mode: str = "market"

    # Microstructure gate (ticks/second + imbalance + quote flips)
    micro_window_sec: int = 3
    micro_min_tps: float = 1.0
    micro_max_tps: float = 360.0  # Ислоҳ: Аз 220 ба 360 барои BTC волатилӣ (сигналҳо дар ҳаракатҳои тез кам намешаванд)
    micro_imb_thresh: float = 0.26
    micro_spread_med_x: float = 1.75
    quote_flips_max: int = 26
    micro_tstat_thresh: float = 0.50

    pullback_atr_mult: float = 0.30

    # Regime detection (BTC volatility thresholds)
    atr_rel_hi: float = 0.0100
    bb_width_range_max: float = 0.0080

    # Spread caps (CRITICAL FOR BTC)
    spread_limit_pct: float = 0.0040  # Ислоҳ: Аз 0.0050 ба 0.0040 кам карда шуд барои филтри спреди тангтар (зиёнҳо кам)

    def symbol(self) -> str:
        return (self.resolved or self.base).strip()

    def validate(self) -> None:
        sym = self.symbol()
        if sym not in ALLOWED_SYMBOLS:
            raise RuntimeError(
                f"config error: symbol must be BTCUSDm only. Got={sym!r}"
            )

        _validate_tf(self.tf_primary, "tf_primary")
        _validate_tf(self.tf_confirm, "tf_confirm")
        _validate_tf(self.tf_long, "tf_long")

        if self.entry_mode != "market":
            raise RuntimeError("config error: entry_mode only supports 'market'")

        if int(self.micro_window_sec) <= 0:
            raise RuntimeError("config error: micro_window_sec must be > 0")

        mn = float(self.micro_min_tps)
        mx = float(self.micro_max_tps)
        if mn <= 0.0 or mx <= 0.0:
            raise RuntimeError("config error: micro_min_tps/micro_max_tps must be > 0")
        if mn >= mx:
            raise RuntimeError("config error: micro_min_tps must be < micro_max_tps")

        imb = float(self.micro_imb_thresh)
        if not (0.0 < imb < 1.0):
            raise RuntimeError("config error: micro_imb_thresh must be in (0..1)")

        if float(self.micro_spread_med_x) <= 0.0:
            raise RuntimeError("config error: micro_spread_med_x must be > 0")

        if int(self.quote_flips_max) < 0:
            raise RuntimeError("config error: quote_flips_max must be >= 0")

        tstat = float(self.micro_tstat_thresh)
        if tstat <= 0.0:
            raise RuntimeError("config error: micro_tstat_thresh must be > 0")

        sp = float(self.spread_limit_pct)
        if not (0.0 < sp < 0.05):
            raise RuntimeError("config error: spread_limit_pct out of range (0..0.05)")


# =============================================================================
# Engine config (BTC only, 24/7)
# =============================================================================

@dataclass(slots=True)
class EngineConfig:
    # Credentials
    login: int = field(default_factory=lambda: _env_int("EXNESS_LOGIN"))
    password: str = field(default_factory=lambda: _env_required("EXNESS_PASSWORD"))
    server: str = field(default_factory=lambda: _env_required("EXNESS_SERVER"))
    telegram_token: str = field(default_factory=lambda: _env_required("BOT_TOKEN"))
    admin_id: int = field(default_factory=lambda: _env_int("ADMIN_ID"))

    symbol_params: SymbolParams = field(default_factory=SymbolParams)

    # Runtime
    tz_local: str = "Asia/Dushanbe"
    active_sessions: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0, 24)]
    )

    # MT5 control (used by mt5_client.py)
    mt5_path: Optional[str] = None
    mt5_portable: bool = False
    mt5_autostart: bool = True
    mt5_timeout_ms: int = 30_000

    # BTC is 24/7: no overnight block
    overnight_block_hours: float = 0.0

    # Risk (BTC scalping)
    # ---------------------------------------------------------------------------
    daily_target_pct: float = 0.10  # Phase B starts at 10% profit (was 10%)
    # ---------------------------------------------------------------------------
    
    max_daily_loss_pct: float = 0.03  # Ислоҳ: Аз 0.05 ба 0.03 кам карда шуд барои лимити зиёнҳои рӯзона (фоидаоварӣ беҳтар)
    protect_drawdown_from_peak_pct: float = 0.20  # Ислоҳ: Аз 0.30 ба 0.20 барои ҳимояи зудтар
    max_drawdown: float = 0.12
    daily_loss_b_pct: float = 0.03  # Ислоҳ: Аз 0.02 ба 0.03 - барои фаъолияти беҳтар
    daily_loss_c_pct: float = 0.06  # Ислоҳ: Аз 0.05 ба 0.06 - барои фаъолияти беҳтар
    enforce_daily_limits: bool = True
    ignore_daily_stop_for_trading: bool = False
    enforce_drawdown_limits: bool = False
    ignore_external_positions: bool = True
    magic: int = 777001

    # Signal quality
    min_confidence_signal: float = 0.80  # Ислоҳ: Аз 0.85 ба 0.80 барои сигналҳои зиёдтар
    conf_min: int = 80  # Ислоҳ: Аз 85 ба 80
    conf_min_low: int = 80  # Ислоҳ: Аз 85 ба 80
    conf_min_high: int = 90
    ultra_confidence_min: float = 0.90
    confidence_bias: float = 50.0
    confidence_gain: float = 70.0  # Ислоҳ: Аз 120 ба 70 барои баланси беҳтар
    net_norm_signal_threshold: float = 0.15  # Ислоҳ: Аз 0.10 ба 0.15 барои сигналҳои қавитар
    strong_conf_min: int = 90
    require_ema_stack: bool = True

    # Indicators
    ema_short: int = 9
    ema_mid: int = 21
    ema_long: int = 50
    ema_vlong: int = 200
    atr_period: int = 14
    rsi_period: int = 14
    adx_period: int = 14
    vol_lookback: int = 80

    adx_trend_lo: float = 16.0
    adx_trend_hi: float = 32.0
    adx_impulse_hi: float = 42.0
    atr_rel_lo: float = 0.00080
    atr_rel_hi: float = 0.01000
    min_body_pct_of_atr: float = 0.09
    min_bar_age_sec: int = 1

    # Volume thresholds (BTC scalping)
    z_vol_hot: float = 2.2

    # Pattern detection (BTC)
    rn_step: float = 100.0
    rn_buffer_pct: float = 0.0012
    fvg_min_atr_mult: float = 0.35
    sweep_min_atr_mult: float = 0.15
    vwap_window: int = 30

    # Anomaly detection (BTC)
    anom_wick_ratio: float = 0.75
    anom_range_atr: float = 2.5
    anom_gap_atr: float = 0.8
    anom_stoprun_lookback: int = 20
    anom_stoprun_zvol: float = 2.2
    anom_warn_score: float = 1.2
    anom_block_score: float = 2.6
    anom_persist_bars: int = 3
    anom_persist_min_hits: int = 2
    anom_confirm_zvol: float = 1.6
    anom_tf_for_block: str = "M1"

    # Indicator cache
    indicator_cache_min_interval_ms: float = 0.0

    # Minimum bars for analysis
    min_bars_m1: int = 200
    min_bars_m5_m15: int = 180
    min_bars_default: int = 160

    # Engine loop (BTC fast)
    poll_seconds_fast: float = 0.05
    decision_debounce_ms: float = 20.0
    analysis_cooldown_sec: float = 0.0
    cooldown_seconds: float = 0.0
    signal_cooldown_sec_override: Optional[float] = None

    # Position sizing (force fixed volume for scalping stability)
    fixed_volume: float = 0.01
    max_risk_per_trade: float = 0.015

    # ---------------------------------------------------------------------------   
    max_positions: int = 3
    # ---------------------------------------------------------------------------


    # Multi-order shaping
    multi_order_tp_bonus_pct: float = 0.15  # Ислоҳ: Аз 0.12 ба 0.15 барои фоида зиёдтар
    multi_order_sl_tighten_pct: float = 0.00
    multi_order_confidence_tiers: Tuple[float, float, float] = (0.85, 0.90, 0.90)
    multi_order_max_orders: int = 3
    max_spread_bps_for_multi: float = 6.0

    # Limits (24/7 BTC)
    max_trades_per_hour: int = 100
    max_signals_per_day: int = 0

    # SL/TP (ATR)
    sl_atr_mult_trend: float = 1.15  # Ислоҳ: Аз 1.25 ба 1.15 барои SL тангтар
    tp_atr_mult_trend: float = 2.7  # Ислоҳ: Аз 2.5 ба 2.7 барои TP калонтар
    sl_atr_mult_range: float = 1.35
    tp_atr_mult_range: float = 2.0
    tp_rr_cap: float = 2.0  # Ислоҳ: Аз 1.8 ба 2.0

    min_rr: float = 1.0
    sltp_cost_spread_mult: float = 1.8
    sltp_cost_slip_mult: float = 1.0
    sltp_cost_move_mult: float = 0.6
    sltp_sl_floor_mult: float = 1.15
    sltp_tp_floor_mult: float = 1.75
    signal_amplification: float = 1.10
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "trend": 0.50,  # Ислоҳ: Аз 0.55 ба 0.50 (баланси беҳтар барои BTC)
            "momentum": 0.25,
            "meanrev": 0.10,
            "structure": 0.10,  # Аз 0.05 ба 0.10 (BTC ба структура вобаста аст)
            "volume": 0.05,  # Аз 0.03 ба 0.05
        }
    )

    # Trailing / BE
    be_trigger_R: float = 0.70
    be_lock_spread_mult: float = 1.20
    trail_atr_mult: float = 0.95
    trail_on_entry: bool = True

    # Execution breaker limits
    slippage_limit_ticks: float = 12.0  # Ислоҳ: Аз 15 ба 12 барои slippage тангтар
    latency_rtt_ms_limit: int = 250  # Ислоҳ: Аз 300 ба 250
    cooldown_after_latency_s: int = 300

    # Hard spread circuit breaker (percent-of-price)
    spread_cb_pct: float = 0.0008  # Ислоҳ: Аз 0.0010 ба 0.0008
    rtt_cb_ms: int = 400  # Ислоҳ: Аз 450 ба 400
    slippage_backoff: float = 0.4  # Ислоҳ: Аз 0.5 ба 0.4

    # Execution quality monitoring (BTC scalping)
    exec_window: int = 300
    exec_max_p95_latency_ms: float = 550.0  # Ислоҳ: Аз 650 ба 550 барои latency беҳтар
    exec_max_p95_slippage_points: float = 20.0  # Ислоҳ: Аз 30 ба 20
    exec_max_spread_points: float = 5000.0  # Ислоҳ: Аз 100.0 ба 5000.0 (барои BTC spread)
    exec_max_ewma_slippage_points: float = 15.0  # Ислоҳ: Аз 18 ба 15
    exec_breaker_sec: float = 120.0

    # Account snapshot cache
    account_snapshot_cache_sec: float = 0.5

    # ATR period for percentile calculation
    atr_period_for_percentile: int = 14

    # Optional spread cap (points) - None = disabled
    max_spread_points: Optional[float] = None

    # Crypto-specific blackouts (BTC 24/7 with optional maintenance windows)
    crypto_blackout_windows_utc: Optional[List[Dict[str, int]]] = None
    rollover_blackout_sec: float = 0.0

    # Telemetry (optional)
    enable_telemetry: bool = True
    enable_debug_logging: bool = False
    correlation_symbol: str = "bitcoin"
    correlation_vs_currency: str = "usd"
    correlation_refresh_sec: int = 300

    # Strategy flags (keep compatible with your system)
    adaptive_enabled: bool = True
    use_squeeze_filter: bool = False
    hedge_flip_enabled: bool = False
    pyramid_enabled: bool = False

    log_csv_path: str = field(
        default_factory=lambda: str(get_log_path("signals_error_only_btc.csv"))
    )

    # Policy toggles
    ignore_sessions: bool = True
    pause_analysis_on_position_open: bool = False
    ignore_microstructure: bool = True
    micro_rr: float = 1.0
    micro_buffer_pct: float = 0.0007
    micro_vol_mult: float = 1.6

    # Signal stability / debounce (faster scalping)
    signal_stability_eps: float = 0.012

    # Market freshness (portfolio pipeline)
    market_min_bar_age_sec: float = 120.0
    market_max_bar_age_mult: float = 2.0
    market_validate_interval_sec: float = 1.0

    # Meta/conformal gates (looser for scalping)
    conformal_window: int = 240
    conformal_q: float = 0.90  # Ислоҳ: Аз 0.95 ба 0.90 барои сигналҳои зиёдтар (аммо дақиқ)
    meta_barrier_R: float = 0.30  # Ислоҳ: Аз 0.40 ба 0.30 - барои сигналҳои зиёдтар
    meta_h_bars: int = 4
    tc_bps: float = 0.8  # Ислоҳ: Аз 1.0 ба 0.8 - барои сигналҳои зиёдтар

    # Multi-order behavior
    multi_order_split_lot: bool = True

    def validate(self) -> None:
        if int(self.login) <= 0:
            raise RuntimeError("config error: login must be > 0")
        if not str(self.password).strip():
            raise RuntimeError("config error: password empty")
        if not str(self.server).strip():
            raise RuntimeError("config error: server empty")
        if not str(self.telegram_token).strip():
            raise RuntimeError("config error: telegram_token empty")
        if int(self.admin_id) <= 0:
            raise RuntimeError("config error: admin_id must be > 0")

        self.symbol_params.validate()
        _validate_sessions_24_7(self.active_sessions)

        m = float(self.max_risk_per_trade)
        if not (0.0 < m <= 0.10):
            raise RuntimeError("config error: max_risk_per_trade out of range (0..0.10]")

        if int(self.max_positions) <= 0:
            raise RuntimeError("config error: max_positions must be > 0")

        if float(self.poll_seconds_fast) <= 0.0:
            raise RuntimeError("config error: poll_seconds_fast must be > 0")

        if float(self.fixed_volume) <= 0.0:
            raise RuntimeError("config error: fixed_volume must be > 0")

        if not (0.0 < float(self.min_confidence_signal) <= 1.0):
            raise RuntimeError("config error: min_confidence_signal must be in (0..1]")

        if not (0.0 < float(self.ultra_confidence_min) <= 1.0):
            raise RuntimeError("config error: ultra_confidence_min must be in (0..1]")

        if not (0.0 < float(self.max_daily_loss_pct) < float(self.max_drawdown)):
            raise RuntimeError("config error: max_daily_loss_pct must be < max_drawdown")

        _validate_tf(self.anom_tf_for_block, "anom_tf_for_block")

        if float(self.spread_cb_pct) <= 0.0 or float(self.spread_cb_pct) >= 0.05:
            raise RuntimeError("config error: spread_cb_pct out of range (0..0.05)")

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
    # Env for secrets ONLY
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

    cfg.min_confidence_signal = 0.85  # Ислоҳ: Аз 0.82 ба 0.85
    cfg.ultra_confidence_min = 0.90
    cfg.max_risk_per_trade = 0.012
    cfg.poll_seconds_fast = 0.05

    # Ensure independent list instance (avoid accidental shared mutations)
    cfg.active_sessions = [(0, 24)]
    cfg.overnight_block_hours = 0.0

    cfg.trail_on_entry = True
    cfg.ignore_microstructure = True
    cfg.use_squeeze_filter = False


__all__ = [
    "ALLOWED_SYMBOLS",
    "CRYPTO_SESSIONS_24_7",
    "TF_MAP",
    "SymbolParams",
    "EngineConfig",
    "get_config_from_env",
    "apply_high_accuracy_mode",
]
