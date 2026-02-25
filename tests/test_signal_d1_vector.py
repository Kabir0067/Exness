from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import core.risk_engine as risk_mod
from core.config import BTCSymbolParams, BaseEngineConfig
from core.risk_engine import RiskManager
from core.signal_engine import SignalEngine


class _DummyFeed:
    pass


class _DummyFeatureEngine:
    pass


def _mk_df(*, slope: float, n: int = 80, start: float = 100.0, vol: float = 100.0) -> pd.DataFrame:
    x = np.arange(n, dtype=np.float64)
    close = start + slope * x
    high = close + 0.1
    low = close - 0.1
    return pd.DataFrame(
        {
            "time": 1_700_000_000 + x * 60.0,
            "open": close - 0.02,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": np.full(n, vol, dtype=np.float64),
        }
    )


@pytest.fixture()
def se(monkeypatch: pytest.MonkeyPatch) -> SignalEngine:
    monkeypatch.setattr(risk_mod, "mt5", None)
    sp = BTCSymbolParams()
    cfg = BaseEngineConfig(symbol_params=sp)
    rm = RiskManager(cfg, sp)
    return SignalEngine(cfg, sp, _DummyFeed(), _DummyFeatureEngine(), rm)


def test_vector_alignment_perfect_alignment(se: SignalEngine) -> None:
    up = _mk_df(slope=1.0)
    score = se._vector_alignment(
        indp={"linreg_slope": 1.0},
        indc={"linreg_slope": 1.0},
        indl={"linreg_slope": 1.0},
        dfp=up,
        dfl=up,
        dfh=up,
        dfd=up,
    )
    assert score > 0.95


def test_vector_alignment_total_conflict(se: SignalEngine) -> None:
    up = _mk_df(slope=1.0)
    flat = _mk_df(slope=0.0)
    down = _mk_df(slope=-1.0)
    score = se._vector_alignment(
        indp={"linreg_slope": 1.0},
        indc={"linreg_slope": 0.0},
        indl={"linreg_slope": 0.0},
        dfp=up,      # M1 up
        dfl=flat,    # M15 flat
        dfh=flat,    # H1 flat
        dfd=down,    # D1 down (opposite)
    )
    # Opposite dominant vectors should collapse toward neutral/zero alignment.
    assert score <= 0.10


def test_vector_alignment_flat_market_no_nan(se: SignalEngine) -> None:
    flat = _mk_df(slope=0.0)
    score = se._vector_alignment(
        indp={"linreg_slope": 0.0},
        indc={"linreg_slope": 0.0},
        indl={"linreg_slope": 0.0},
        dfp=flat,
        dfl=flat,
        dfh=flat,
        dfd=flat,
    )
    assert math.isfinite(score)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.5, abs=1e-9)


def test_momentum_ignition_triggers_on_volume_spike_and_breakout(se: SignalEngine) -> None:
    df = _mk_df(slope=0.02, n=90, start=100.0, vol=100.0)
    # Force ignition conditions on the latest bar.
    df.loc[df.index[-1], "tick_volume"] = 700.0
    prev_high = float(df["high"].iloc[-21:-1].max())
    df.loc[df.index[-1], "close"] = prev_high + 1.5
    df.loc[df.index[-1], "high"] = prev_high + 1.7
    df.loc[df.index[-1], "low"] = prev_high + 1.2

    score = se._momentum_ignition_score(df)
    assert 0.0 <= score <= 1.0
    assert score >= 0.75
