"""Tests for liquidity-aware TP calculation."""

from liquidity_tp import compute_liquidity_tp_prices


def _book_short():
    bids = [
        {"price": "0.5260", "size": "10"},
        {"price": "0.5255", "size": "12"},
        {"price": "0.5250", "size": "500"},  # wall
        {"price": "0.5245", "size": "8"},
        {"price": "0.5240", "size": "600"},  # wall
        {"price": "0.5235", "size": "9"},
    ]
    asks = [{"price": "0.5270", "size": "20"}]
    return bids, asks


def test_short_tp_filters_far_walls():
    bids, asks = _book_short()
    tps = compute_liquidity_tp_prices(
        "SHORT",
        bids,
        asks,
        entry_price=0.5270,
        tick_size=0.0001,
        tick_offset=2,
        max_distance_pct=0.003,
    )
    assert tps == []
    tps = compute_liquidity_tp_prices(
        "SHORT",
        bids,
        asks,
        entry_price=0.5270,
        tick_size=0.0001,
        tick_offset=2,
        max_distance_pct=0.004,
    )
    assert len(tps) >= 1
    assert all((0.5270 - tp) / 0.5270 <= 0.004 + 1e-9 for tp in tps)


def test_short_tp_above_bid_walls_sorted_desc():
    bids, asks = _book_short()
    tps = compute_liquidity_tp_prices(
        "SHORT",
        bids,
        asks,
        entry_price=0.5270,
        tick_size=0.0001,
        tick_offset=2,
    )
    assert len(tps) >= 2
    assert tps == sorted(tps, reverse=True)
    assert all(tp < 0.5270 for tp in tps)
    assert all(tp > 0.5230 for tp in tps)


def test_long_tp_below_ask_walls_sorted_asc():
    bids = [{"price": "0.5260", "size": "20"}]
    asks = [
        {"price": "0.5270", "size": "10"},
        {"price": "0.5275", "size": "400"},
        {"price": "0.5280", "size": "12"},
        {"price": "0.5290", "size": "550"},
    ]
    tps = compute_liquidity_tp_prices(
        "LONG",
        bids,
        asks,
        entry_price=0.5265,
        tick_size=0.0001,
        tick_offset=2,
    )
    assert len(tps) >= 1
    assert tps == sorted(tps)
    assert all(tp > 0.5265 for tp in tps)
