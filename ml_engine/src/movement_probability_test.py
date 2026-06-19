"""Tests for softmax movement probability."""

import numpy as np

from .features import SymbolBuffer
from .movement_probability import (
    MovementState,
    build_state_vector,
    softmax_movement_probability,
)


def _book_with_imbalance(bid_size: float, ask_size: float):
    buf = SymbolBuffer(symbol="TEST")
    bids = [{"price": "100.0", "size": str(bid_size)}]
    asks = [{"price": "100.1", "size": str(ask_size)}]
    buf.add_orderbook(1_700_000_000_000, bids, asks)
    for i in range(5):
        buf.add_trade(1_700_000_000_100 + i, 100.05, 10.0, "Buy")
    return buf


def test_softmax_sums_to_one():
    buf = _book_with_imbalance(5000, 1000)
    mp = softmax_movement_probability(buf, MovementState())
    total = mp.p_short + mp.p_neutral + mp.p_long
    assert abs(total - 1.0) < 1e-6


def test_bullish_book_skews_long():
    buf = _book_with_imbalance(8000, 500)
    mp = softmax_movement_probability(buf, MovementState())
    assert mp.p_long > mp.p_short


def test_bearish_book_skews_short():
    buf = _book_with_imbalance(500, 8000)
    mp = softmax_movement_probability(buf, MovementState())
    assert mp.p_short > mp.p_long


def test_state_vector_bounds():
    buf = _book_with_imbalance(3000, 3000)
    x = build_state_vector(buf, MovementState())
    assert x[0] >= -1.0 and x[0] <= 1.0
    assert x[1] >= -1.0 and x[1] <= 1.0
