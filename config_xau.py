# config.py  (PRODUCTION / FAST / CLEAN / NO-STRUCTURE-CHANGE)
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5
from dotenv import load_dotenv

from log_config import get_log_path

# Keep import-time .env loading (system-wide behavior). No structure change.
load_dotenv(override=False)

# =============================================================================
# Constants
# =============================================================================
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

_BOOL_TRUE = {"1", "true", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "no", "n", "off"}


# =============================================================================
# ENV helpers (strict + fast)
# =============================================================================
def _env_required(name: str) -> str:
    """Return env value if present and non-empty, else raise."""
    v = os.environ.get(name)
    if v is None:
        raise RuntimeError(f".env error: missing variable → {name}")
    s = str(v).strip()
    if not s:
        raise RuntimeError(f".env error: empty variable → {name}")
    return s


def _env_int(name: str, default: Optional[int] = None) -> int:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        if default is None:
            raise RuntimeError(f".env error: missing/empty variable → {name}")
        return int(default)
    s = str(v).strip()
    try:
        x = int(s)
    except Exception as exc:
        raise RuntimeError(f".env error: invalid int for {name} = {s!r}") from exc
    return x


def _env_float(name: str, default: Optional[float] = None) -> float:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        if default is None:
            raise RuntimeError(f".env error: missing/empty variable → {name}")
        return float(default)
    s = str(v).strip()
    try:
        x = float(s)
    except Exception as exc:
        raise RuntimeError(f".env error: invalid float for {name} = {s!r}") from exc
    if x != x or x in (float("inf"), float("-inf")):
        raise RuntimeError(f".env error: non-finite float for {name} = {s!r}")
    return float(x)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return bool(default)
    s = str(v).strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    raise RuntimeError(
        f".env error: invalid bool for {name} = {s!r} (use one of {sorted(_BOOL_TRUE | _BOOL_FALSE)})"
    )


# =============================================================================
# Validation helpers
# =============================================================================
def _validate_tf(tf: str, field_name: str) -> None:
    if tf not in TF_MAP:
        raise RuntimeError(f"config error: invalid {field_name}={tf!r}. Allowed={list(TF_MAP.keys())}")


def _validate_sessions(sessions: List[Tuple[int, int]]) -> None:
    if not sessions:
        raise RuntimeError("config error: active_sessions is empty")

    norm: List[Tuple[int, int]] = []
    for item in sessions:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise RuntimeError(f"config error: invalid session {item!r}; must be (start_hour, end_hour)")
        start_h, end_h = item
        sh = int(start_h)
        eh = int(end_h)

        if not (0 <= sh <= 24 and 0 <= eh <= 24):
            raise RuntimeError(f"config error: invalid session hours {(start_h, end_h)}; must be in [0..24]")

        if sh == eh:
            raise RuntimeError(f"config error: session {(start_h, end_h)} is zero-length")

        # 24/7 ok
        if (sh, eh) == (0, 24):
            norm.append((0, 24))
            continue

        # Disallow crossing midnight here (explicit split is required)
        if sh > eh:
            raise RuntimeError(f"config error: session {(start_h, end_h)} crosses midnight; split into two sessions")

        norm.append((sh, eh))

    # If 24/7 session exists, it must be the only one (prevents ambiguity).
    if (0, 24) in norm and len(norm) != 1:
        raise RuntimeError("config error: session (0, 24) cannot be combined with other sessions")

    # No overlaps (touching boundaries is allowed).
    norm_sorted = sorted(norm, key=lambda x: (x[0], x[1]))
    for i in range(1, len(norm_sorted)):
        prev_s, prev_e = norm_sorted[i - 1]
        cur_s, cur_e = norm_sorted[i]
        if cur_s < prev_e:
            raise RuntimeError(f"config error: overlapping sessions {norm_sorted[i - 1]} and {norm_sorted[i]}")


def _validate_pct01(x: float, name: str) -> None:
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        raise RuntimeError(f"config error: {name} is non-finite")
    if not (0.0 <= v <= 1.0):
        raise RuntimeError(f"config error: {name} out of range [0..1]")


def _validate_pos(x: float, name: str) -> None:
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        raise RuntimeError(f"config error: {name} is non-finite")
    if v <= 0:
        raise RuntimeError(f"config error: {name} must be > 0")


# =============================================================================
# Config models
# =============================================================================
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
    spread_limit_pct: float = 0.00025  # Филтри спреди тангтар дар тилло

    def validate(self) -> None:
        sym = (self.resolved or self.base or "").strip()
        if sym not in ALLOWED_SYMBOLS:
            raise RuntimeError(f"config error: symbol {sym!r} not allowed. Allowed={ALLOWED_SYMBOLS}")

        _validate_tf(self.tf_primary, "tf_primary")
        _validate_tf(self.tf_confirm, "tf_confirm")
        _validate_tf(self.tf_long, "tf_long")

        if self.entry_mode != "market":
            raise RuntimeError(f"config error: entry_mode {self.entry_mode!r} not supported (only 'market')")

        if int(self.micro_window_sec) <= 0 or int(self.micro_window_sec) > 60:
            raise RuntimeError("config error: micro_window_sec must be in [1..60]")

        _validate_pos(self.micro_min_tps, "micro_min_tps")
        _validate_pos(self.micro_max_tps, "micro_max_tps")
        if float(self.micro_min_tps) >= float(self.micro_max_tps):
            raise RuntimeError("config error: micro_min_tps must be < micro_max_tps")

        _validate_pct01(self.micro_imb_thresh, "micro_imb_thresh")
        if float(self.micro_imb_thresh) == 0.0:
            raise RuntimeError("config error: micro_imb_thresh must be > 0")

        _validate_pos(self.micro_spread_med_x, "micro_spread_med_x")

        qfm = int(self.quote_flips_max)
        if qfm < 0 or qfm > 5000:
            raise RuntimeError("config error: quote_flips_max out of range [0..5000]")

        tstat = float(self.micro_tstat_thresh)
        if tstat != tstat or tstat in (float("inf"), float("-inf")) or not (0.0 < tstat <= 10.0):
            raise RuntimeError("config error: micro_tstat_thresh out of range (0..10]")

        pab = float(self.pullback_atr_mult)
        if pab != pab or pab in (float("inf"), float("-inf")) or not (0.0 < pab <= 10.0):
            raise RuntimeError("config error: pullback_atr_mult out of range (0..10]")

        if not (0.0 < float(self.spread_limit_pct) < 0.01):
            raise RuntimeError("config error: spread_limit_pct out of range (0..0.01)")


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
    mt5_timeout_ms: int = 30_000

    # ---------------------------------------------------------------------------
    daily_target_pct: float = 0.15  # Phase B starts at 15% profit
    # ---------------------------------------------------------------------------

    ultra_confidence_min: float = 0.90
    protect_drawdown_from_peak_pct: float = 0.20

    max_daily_loss_pct: float = 0.05
    daily_loss_b_pct: float = 0.05
    daily_loss_c_pct: float = 0.05

    enforce_daily_limits: bool = True
    ignore_daily_stop_for_trading: bool = False
    enforce_drawdown_limits: bool = False
    ignore_external_positions: bool = True
    magic: int = 777001
    dry_run: bool = False  # SIMULATION MODE: If True, logic runs but NO orders sent.

    min_confidence_signal: float = 0.85
    conf_min: int = 85
    conf_min_low: int = 85
    conf_min_high: int = 90
    confidence_bias: float = 50.0
    confidence_gain: float = 70.0
    net_norm_signal_threshold: float = 0.15
    strong_conf_min: int = 90
    require_ema_stack: bool = True

    adx_trend_lo: float = 16.0
    adx_trend_hi: float = 27.0
    atr_rel_lo: float = 0.00055
    atr_rel_hi: float = 0.0028
    min_body_pct_of_atr: float = 0.09
    min_bar_age_sec: int = 1

    ema_short: int = 9
    ema_mid: int = 21
    ema_long: int = 50
    ema_vlong: int = 200
    atr_period: int = 14
    rsi_period: int = 14
    adx_period: int = 14
    vol_lookback: int = 80

    # === MEDIUM SCALPING 1-15 дақиқа (УСТУВОР) ===
    poll_seconds_fast: float = 0.50
    poll_seconds_slow: float = 1.50
    decision_debounce_ms: float = 500.0
    analysis_cooldown_sec: float = 1.0
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

    # ---------------------------------------------------------------------------
    max_positions: int = 1
    # ---------------------------------------------------------------------------

    # --- Multi-order tuning (SAFE scalping) ---
    multi_order_tp_bonus_pct: float = 0.15
    multi_order_sl_tighten_pct: float = 0.00
    multi_order_confidence_tiers: Tuple[float, float, float] = (0.85, 0.90, 0.90)
    multi_order_max_orders: int = 3
    max_spread_bps_for_multi: float = 6.0

    islamic_min_leverage: int = 1
    require_swap_free: bool = False

    max_drawdown: float = 0.09
    max_trades_per_hour: int = 20
    max_signals_per_day: int = 0

    sl_atr_mult_trend: float = 1.15
    tp_atr_mult_trend: float = 2.7
    sl_atr_mult_range: float = 1.35
    tp_atr_mult_range: float = 2.0
    tp_rr_cap: float = 2.0
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
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "trend": 0.50,
            "momentum": 0.25,
            "meanrev": 0.10,
            "structure": 0.10,
            "volume": 0.05,
        }
    )

    adaptive_enabled: bool = True
    trail_on_entry: bool = True
    use_squeeze_filter: bool = False
    hedge_flip_enabled: bool = False
    pyramid_enabled: bool = False

    log_csv_path: str = field(default_factory=lambda: str(get_log_path("signals_error_only.csv")))

    extreme_lookback: int = 120
    extreme_atr_mult: float = 1.7
    extreme_near_pct: float = 0.12
    rn_step_xau: float = 5.0
    rn_buffer_pct: float = 0.0012

    min_bars_m1: int = 200
    min_bars_m5_m15: int = 180
    min_bars_default: int = 160

    signal_stability_eps: float = 0.012

    market_min_bar_age_sec: float = 120.0
    market_max_bar_age_mult: float = 2.0
    market_validate_interval_sec: float = 1.0

    # Data freshness guards (stale data rejection)
    tick_stale_threshold_sec: float = 15.0  # Reject signals if tick data older than this
    volume_filter_mult: float = 0.4  # Current vol must be > vol_filter_mult * 20-period MA

    # NOTE: duplicated fields below are kept (NO-STRUCTURE-CHANGE); defaults aligned to the final values.
    meta_barrier_R: float = 0.50
    meta_h_bars: int = 6
    conformal_q: float = 0.90
    tc_bps: float = 1.0

    multi_order_split_lot: bool = True

    atr_percentile_lookback: int = 1000
    tod_boost_minutes: int = 120
    slippage_window: int = 40
    slippage_limit_ticks: float = 12.0
    latency_rtt_ms_limit: int = 250
    cooldown_after_latency_s: int = 300

    ultimate_mode: bool = True
    ensemble_w: Tuple[float, float, float] = (0.55, 0.33, 0.12)
    meta_barrier_R: float = 0.50
    meta_h_bars: int = 6
    conformal_window: int = 300
    conformal_q: float = 0.90
    brier_window: int = 800
    tc_bps: float = 1.0
    rtt_cb_ms: int = 400
    spread_cb_pct: float = 0.0008
    slippage_backoff: float = 0.4

    # Execution quality
    exec_window: int = 300
    exec_max_p95_latency_ms: float = 550.0
    exec_max_p95_slippage_points: float = 20.0
    exec_max_spread_points: float = 500.0  # limit 500 points (0.50 USD)
    exec_max_ewma_slippage_points: float = 15.0
    exec_breaker_sec: float = 120.0

    # Policy toggles
    ignore_sessions: bool = True
    pause_analysis_on_position_open: bool = True
    ignore_microstructure: bool = True

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
        _validate_sessions(self.active_sessions)

        if not str(self.tz_local).strip():
            raise RuntimeError("config error: tz_local empty")

        if int(self.http_pool_conn) <= 0 or int(self.http_pool_max) <= 0:
            raise RuntimeError("config error: http_pool_conn/http_pool_max must be > 0")
        if int(self.telegram_read_timeout) <= 0 or int(self.telegram_connect_timeout) <= 0:
            raise RuntimeError("config error: telegram timeouts must be > 0")

        if int(self.mt5_timeout_ms) <= 0:
            raise RuntimeError("config error: mt5_timeout_ms must be > 0")

        _validate_pct01(self.daily_target_pct, "daily_target_pct")
        if float(self.daily_target_pct) == 0.0:
            raise RuntimeError("config error: daily_target_pct must be > 0")

        _validate_pct01(self.ultra_confidence_min, "ultra_confidence_min")
        _validate_pct01(self.min_confidence_signal, "min_confidence_signal")
        if not (0.0 < float(self.min_confidence_signal) <= 1.0):
            raise RuntimeError("config error: min_confidence_signal out of range (0..1]")

        _validate_pct01(self.protect_drawdown_from_peak_pct, "protect_drawdown_from_peak_pct")
        if float(self.protect_drawdown_from_peak_pct) == 0.0:
            raise RuntimeError("config error: protect_drawdown_from_peak_pct must be > 0")

        _validate_pct01(self.max_daily_loss_pct, "max_daily_loss_pct")
        _validate_pct01(self.daily_loss_b_pct, "daily_loss_b_pct")
        _validate_pct01(self.daily_loss_c_pct, "daily_loss_c_pct")
        if float(self.daily_loss_b_pct) > float(self.max_daily_loss_pct):
            raise RuntimeError("config error: daily_loss_b_pct must be <= max_daily_loss_pct")
        if float(self.daily_loss_c_pct) > float(self.max_daily_loss_pct):
            raise RuntimeError("config error: daily_loss_c_pct must be <= max_daily_loss_pct")

        if int(self.magic) <= 0:
            raise RuntimeError("config error: magic must be > 0")

        if not (0.0 < float(self.max_risk_per_trade) <= 0.10):
            raise RuntimeError("config error: max_risk_per_trade out of range (0..0.10]")

        if int(self.max_positions) <= 0:
            raise RuntimeError("config error: max_positions must be > 0")

        if float(self.poll_seconds_fast) <= 0:
            raise RuntimeError("config error: poll_seconds_fast must be > 0")
        if float(self.poll_seconds_slow) <= 0:
            raise RuntimeError("config error: poll_seconds_slow must be > 0")
        if float(self.decision_debounce_ms) < 0:
            raise RuntimeError("config error: decision_debounce_ms must be >= 0")
        if float(self.analysis_cooldown_sec) < 0:
            raise RuntimeError("config error: analysis_cooldown_sec must be >= 0")
        if float(self.cooldown_seconds) < 0:
            raise RuntimeError("config error: cooldown_seconds must be >= 0")
        if float(self.overnight_block_hours) < 0:
            raise RuntimeError("config error: overnight_block_hours must be >= 0")

        if int(self.health_window_minutes) <= 0:
            raise RuntimeError("config error: health_window_minutes must be > 0")
        if int(self.correlation_refresh_sec) <= 0:
            raise RuntimeError("config error: correlation_refresh_sec must be > 0")

        if self.fixed_volume is not None:
            _validate_pos(self.fixed_volume, "fixed_volume")

        # Confidence integer gates
        for n in ("conf_min", "conf_min_low", "conf_min_high", "strong_conf_min"):
            v = int(getattr(self, n))
            if not (0 <= v <= 100):
                raise RuntimeError(f"config error: {n} out of range [0..100]")
        if int(self.conf_min_high) < int(self.conf_min):
            raise RuntimeError("config error: conf_min_high must be >= conf_min")

        # Score shaping
        _validate_pos(self.confidence_gain, "confidence_gain")
        if float(self.confidence_bias) != float(self.confidence_bias) or float(self.confidence_bias) in (
            float("inf"),
            float("-inf"),
        ):
            raise RuntimeError("config error: confidence_bias is non-finite")

        _validate_pct01(self.net_norm_signal_threshold, "net_norm_signal_threshold")

        # Trend / vol sanity
        if float(self.adx_trend_lo) <= 0 or float(self.adx_trend_hi) <= 0:
            raise RuntimeError("config error: adx_trend_* must be > 0")
        if float(self.adx_trend_lo) >= float(self.adx_trend_hi):
            raise RuntimeError("config error: adx_trend_lo must be < adx_trend_hi")
        _validate_pos(self.atr_rel_lo, "atr_rel_lo")
        _validate_pos(self.atr_rel_hi, "atr_rel_hi")
        if float(self.atr_rel_lo) >= float(self.atr_rel_hi):
            raise RuntimeError("config error: atr_rel_lo must be < atr_rel_hi")
        _validate_pct01(self.min_body_pct_of_atr, "min_body_pct_of_atr")

        # Periods
        for n in ("ema_short", "ema_mid", "ema_long", "ema_vlong", "atr_period", "rsi_period", "adx_period", "vol_lookback"):
            if int(getattr(self, n)) <= 0:
                raise RuntimeError(f"config error: {n} must be > 0")

        # Multi-order rules
        _validate_pct01(self.multi_order_tp_bonus_pct, "multi_order_tp_bonus_pct")
        _validate_pct01(self.multi_order_sl_tighten_pct, "multi_order_sl_tighten_pct")
        t1, t2, t3 = (float(self.multi_order_confidence_tiers[0]), float(self.multi_order_confidence_tiers[1]), float(self.multi_order_confidence_tiers[2]))
        for i, tv in enumerate((t1, t2, t3), start=1):
            _validate_pct01(tv, f"multi_order_confidence_tiers[{i}]")
        if not (t1 <= t2 <= t3):
            raise RuntimeError("config error: multi_order_confidence_tiers must be non-decreasing")
        if t1 < float(self.min_confidence_signal):
            raise RuntimeError("config error: multi_order_confidence_tiers[0] must be >= min_confidence_signal")
        if int(self.multi_order_max_orders) <= 0 or int(self.multi_order_max_orders) > 20:
            raise RuntimeError("config error: multi_order_max_orders out of range [1..20]")
        _validate_pos(self.max_spread_bps_for_multi, "max_spread_bps_for_multi")

        # Risk / limits
        _validate_pct01(self.max_drawdown, "max_drawdown")
        if not (0.0 < float(self.max_drawdown) <= 0.50):
            raise RuntimeError("config error: max_drawdown out of range (0..0.50]")
        if int(self.max_trades_per_hour) <= 0:
            raise RuntimeError("config error: max_trades_per_hour must be > 0")
        if int(self.max_signals_per_day) < 0:
            raise RuntimeError("config error: max_signals_per_day must be >= 0")

        # SL/TP sanity
        for n in (
            "sl_atr_mult_trend",
            "tp_atr_mult_trend",
            "sl_atr_mult_range",
            "tp_atr_mult_range",
            "tp_rr_cap",
            "min_rr",
            "sltp_cost_spread_mult",
            "sltp_cost_slip_mult",
            "sltp_cost_move_mult",
            "sltp_sl_floor_mult",
            "sltp_tp_floor_mult",
            "be_trigger_R",
            "be_lock_spread_mult",
            "trail_atr_mult",
        ):
            _validate_pos(getattr(self, n), n)
        if float(self.tp_rr_cap) < 1.0:
            raise RuntimeError("config error: tp_rr_cap must be >= 1.0")
        if float(self.min_rr) < 1.0:
            raise RuntimeError("config error: min_rr must be >= 1.0")

        _validate_pos(self.signal_amplification, "signal_amplification")

        # Weights: strict keys + sum ~= 1.0
        expected_keys = ("trend", "momentum", "meanrev", "structure", "volume")
        for k in expected_keys:
            if k not in self.weights:
                raise RuntimeError(f"config error: weights missing key {k!r}")
            _validate_pos(self.weights[k], f"weights[{k}]")
        for k in self.weights.keys():
            if k not in expected_keys:
                raise RuntimeError(f"config error: weights has unexpected key {k!r}")
        wsum = float(sum(float(self.weights[k]) for k in expected_keys))
        if wsum != wsum or wsum in (float("inf"), float("-inf")):
            raise RuntimeError("config error: weights sum is non-finite")
        if abs(wsum - 1.0) > 1e-6:
            raise RuntimeError(f"config error: weights must sum to 1.0 (got {wsum:.6f})")

        if not str(self.log_csv_path).strip():
            raise RuntimeError("config error: log_csv_path empty")

        # Market validators
        _validate_pos(self.market_min_bar_age_sec, "market_min_bar_age_sec")
        _validate_pos(self.market_max_bar_age_mult, "market_max_bar_age_mult")
        _validate_pos(self.market_validate_interval_sec, "market_validate_interval_sec")

        # Meta / conformal / costs
        _validate_pos(self.meta_barrier_R, "meta_barrier_R")
        if not (0.0 < float(self.meta_barrier_R) <= 5.0):
            raise RuntimeError("config error: meta_barrier_R out of range (0..5]")
        if int(self.meta_h_bars) <= 0 or int(self.meta_h_bars) > 50:
            raise RuntimeError("config error: meta_h_bars out of range [1..50]")
        _validate_pct01(self.conformal_q, "conformal_q")
        if float(self.conformal_q) == 0.0:
            raise RuntimeError("config error: conformal_q must be > 0")
        _validate_pos(self.tc_bps, "tc_bps")

        # Execution breakers
        if int(self.exec_window) <= 0:
            raise RuntimeError("config error: exec_window must be > 0")
        _validate_pos(self.exec_max_p95_latency_ms, "exec_max_p95_latency_ms")
        _validate_pos(self.exec_max_p95_slippage_points, "exec_max_p95_slippage_points")
        _validate_pos(self.exec_max_spread_points, "exec_max_spread_points")
        _validate_pos(self.exec_max_ewma_slippage_points, "exec_max_ewma_slippage_points")
        if int(self.exec_breaker_sec) <= 0:
            raise RuntimeError("config error: exec_breaker_sec must be > 0")

        # Latency / slippage guards
        _validate_pos(self.slippage_limit_ticks, "slippage_limit_ticks")
        if int(self.latency_rtt_ms_limit) <= 0:
            raise RuntimeError("config error: latency_rtt_ms_limit must be > 0")
        if int(self.cooldown_after_latency_s) < 0:
            raise RuntimeError("config error: cooldown_after_latency_s must be >= 0")

        # Ensemble weights sum
        ew0, ew1, ew2 = float(self.ensemble_w[0]), float(self.ensemble_w[1]), float(self.ensemble_w[2])
        for i, ev in enumerate((ew0, ew1, ew2), start=1):
            _validate_pos(ev, f"ensemble_w[{i}]")
        ewsum = ew0 + ew1 + ew2
        if abs(ewsum - 1.0) > 1e-6:
            raise RuntimeError(f"config error: ensemble_w must sum to 1.0 (got {ewsum:.6f})")

        # Circuit-breaker params
        if int(self.rtt_cb_ms) <= 0:
            raise RuntimeError("config error: rtt_cb_ms must be > 0")
        _validate_pos(self.spread_cb_pct, "spread_cb_pct")
        if not (0.0 < float(self.spread_cb_pct) < 0.01):
            raise RuntimeError("config error: spread_cb_pct out of range (0..0.01)")
        _validate_pct01(self.slippage_backoff, "slippage_backoff")

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


# =============================================================================
# Public API
# =============================================================================
def get_config_from_env() -> EngineConfig:
    cfg = EngineConfig(
        login=_env_int("EXNESS_LOGIN"),
        password=_env_required("EXNESS_PASSWORD"),
        server=_env_required("EXNESS_SERVER"),
        telegram_token=_env_required("BOT_TOKEN"),
        admin_id=_env_int("ADMIN_ID"),
        dry_run=_env_bool("DRY_RUN", False),
    )
    cfg.validate()
    return cfg


def apply_high_accuracy_mode(cfg: EngineConfig, enable: bool = True) -> None:
    if not enable:
        return
    # Keep your original behavior; only clean assignments.
    cfg.min_confidence_signal = 0.85
    cfg.ultra_confidence_min = 0.90
    cfg.max_risk_per_trade = 0.02
    cfg.tp_atr_mult_trend = 2.7
    cfg.use_squeeze_filter = False
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
