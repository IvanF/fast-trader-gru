"""Tests for market telemetry and pattern detection."""

from .market_telemetry import compute_telemetry, detect_pattern_events, SymbolEventState
from .features import SymbolBuffer


def _buf_with_mid(mid: float, support: float, resistance: float):
    buf = SymbolBuffer(symbol="TEST")
    bids = [{"price": str(support), "size": "5000"}]
    asks = [{"price": str(resistance), "size": "5000"}]
    buf.add_orderbook(1_700_000_000_000, bids, asks)
    buf.add_trade(1_700_000_000_100, mid, 10.0, "Buy")
    return buf


def test_breakout_down_detected():
    buf = _buf_with_mid(100.0, 99.0, 101.0)
    state = SymbolEventState(prev_mid=100.0, support_wall=99.0)
    snap = compute_telemetry(buf)
    snap.mid = 98.85
    snap.bid_wall_price = 99.0
    events = detect_pattern_events(buf, snap, state)
    types = [e.event_type for e in events]
    assert "breakout_down" in types


def test_telemetry_fields_populated():
    buf = _buf_with_mid(100.0, 99.5, 100.5)
    snap = compute_telemetry(buf)
    fields = snap.as_fields()
    assert fields["mid"] > 0
    assert "obi" in fields
    assert "vwap_deviation_pct" in fields
    assert "atr_pct" in fields
