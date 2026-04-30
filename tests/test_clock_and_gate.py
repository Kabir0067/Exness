import threading
import time


class _FakeTick:
    def __init__(self, epoch: float) -> None:
        self.time = float(epoch)


class _FakeMT5:
    def __init__(self, epochs: list[float]) -> None:
        self._epochs = list(epochs)

    def symbol_info_tick(self, _symbol: str):
        if not self._epochs:
            return None
        return _FakeTick(self._epochs.pop(0))


def test_clock_sync_allows_stable_broker_offset() -> None:
    from core.data_integrity import ServerClockSync

    base = time.time()
    sync = ServerClockSync(probe_symbol="XAUUSDm", ttl_sec=60.0)
    sync._mt5 = _FakeMT5([base + 54.0, base + 54.02])
    sync._mt5_lock = threading.RLock()

    sync._probe(base)
    sync._last_probe_ts = 0.0
    exceeded, jitter = sync.is_drift_exceeded(3.0)

    assert exceeded is False
    assert jitter < 3.0
    assert abs(sync.drift_seconds) > 40.0


def test_clock_sync_blocks_clock_jumps() -> None:
    from core.data_integrity import ServerClockSync

    base = time.time()
    sync = ServerClockSync(probe_symbol="XAUUSDm", ttl_sec=60.0)
    sync._mt5 = _FakeMT5([base + 10.0, base + 25.0])
    sync._mt5_lock = threading.RLock()

    sync._probe(base)
    sync._last_probe_ts = 0.0
    exceeded, jitter = sync.is_drift_exceeded(3.0)

    assert exceeded is True
    assert jitter > 3.0


def test_partial_gate_is_enabled_by_default(monkeypatch) -> None:
    from runmain.gate import _partial_gate_enabled

    monkeypatch.delenv("PARTIAL_GATE_MODE", raising=False)
    monkeypatch.delenv("STRICT_DUAL_ASSET_MODE", raising=False)

    assert _partial_gate_enabled(("XAU", "BTC")) is True


def test_strict_dual_asset_mode_blocks_partial_gate(monkeypatch) -> None:
    from runmain.gate import _partial_gate_enabled

    monkeypatch.setenv("PARTIAL_GATE_MODE", "1")
    monkeypatch.setenv("STRICT_DUAL_ASSET_MODE", "1")

    assert _partial_gate_enabled(("XAU", "BTC")) is False
