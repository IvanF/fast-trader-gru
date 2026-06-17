"""Exit level planner — mirrors OMS grid logic using NN outputs + orderbook."""

from __future__ import annotations

import math
from typing import Any

from .liquidity_tp import compute_liquidity_tp_prices


BASE_GRID_SPACING = 0.0025


def _cap_vol_multiplier(vol_mult: float, cap: float) -> float:
    vm = vol_mult if vol_mult > 0 else 1.0
    if cap > 0:
        vm = min(vm, cap)
    return max(vm, 0.5)


def _aggressive_maker_entry(
    direction: str,
    bids: list,
    asks: list,
    tick_size: float,
    maker_ticks: int,
) -> float:
    if maker_ticks <= 0:
        return 0.0
    if tick_size <= 0:
        tick_size = 0.0001
    bid = _level_price(bids[0]) if bids else 0.0
    ask = _level_price(asks[0]) if asks else 0.0
    offset = maker_ticks * tick_size
    if direction == "LONG":
        if bid <= 0:
            return 0.0
        price = bid + offset
        if ask > 0 and price >= ask:
            price = ask - tick_size
        return _round_tick(max(price, bid), tick_size)
    if direction == "SHORT":
        if ask <= 0:
            return 0.0
        price = ask - offset
        if bid > 0 and price <= bid:
            price = bid + tick_size
        return _round_tick(min(price, ask) if price > 0 else ask, tick_size)
    return 0.0


def _level_price(level: Any) -> float:
    if isinstance(level, dict):
        raw = level.get("price") or level.get("Price")
        return float(raw) if raw else 0.0
    if isinstance(level, (list, tuple)) and level:
        return float(level[0])
    return 0.0


def _level_size(level: Any) -> float:
    if isinstance(level, dict):
        raw = level.get("size") or level.get("Size")
        return float(raw) if raw else 0.0
    if isinstance(level, (list, tuple)) and len(level) > 1:
        return float(level[1])
    return 0.0


def _mid(bids: list, asks: list) -> float:
    if not bids or not asks:
        return 0.0
    return (_level_price(bids[0]) + _level_price(asks[0])) / 2.0


def _round_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(price / tick) * tick


def _find_wall(levels: list, low: float, high: float) -> tuple[float, float]:
    prices, sizes = [], []
    for lv in levels:
        p, s = _level_price(lv), _level_size(lv)
        if low <= p <= high:
            prices.append(p)
            sizes.append(s)
    if not sizes:
        return 0.0, 0.0
    idx = max(range(len(sizes)), key=lambda i: sizes[i])
    return prices[idx], sizes[idx]


def _fee_aware_breakeven(fill: float, direction: str, fee_pct: float, tick: float) -> float:
    if fill <= 0 or fee_pct <= 0:
        return fill
    if direction == "LONG":
        return _round_tick(fill * (1 + fee_pct), tick)
    return _round_tick(fill * (1 - fee_pct), tick)


def _enforce_min_tp(fill: float, tp: float, direction: str, min_pct: float, tick: float) -> float:
    if fill <= 0 or min_pct <= 0:
        return tp
    if direction == "LONG":
        floor = fill * (1 + min_pct)
        if tp <= 0 or tp < floor:
            return _round_tick(floor, tick)
    else:
        ceiling = fill * (1 - min_pct)
        if tp <= 0 or tp > ceiling:
            return _round_tick(ceiling, tick)
    return _round_tick(tp, tick)


def build_exit_plan(
    direction: str,
    bids: list,
    asks: list,
    vol_mult: float,
    regime: str,
    confidence: float,
    tick_size: float = 0.0001,
    fallback_mid: float = 0.0,
    vol_multiplier_cap: float = 2.0,
    entry_maker_ticks: int = 2,
    min_tp_pct: float = 0.002,
    fee_breakeven_pct: float = 0.0015,
    max_tp_pct: float = 0.008,
) -> dict[str, Any]:
    """Compute entry anchor, SL behind liquidity wall, TP ladder from NN regime/vol."""
    mid = _mid(bids, asks)
    if mid <= 0 and fallback_mid > 0:
        mid = fallback_mid
    if mid <= 0:
        return {}

    vm = _cap_vol_multiplier(vol_mult, vol_multiplier_cap)
    spacing = BASE_GRID_SPACING * vm
    range_pct = spacing * 4
    risk = spacing * mid

    low = mid * (1 - range_pct)
    high = mid * (1 + range_pct)

    if direction == "LONG":
        wall_p, _ = _find_wall(bids, low, high)
        if wall_p <= 0:
            wall_p = mid * 0.99
        sl = wall_p - tick_size * 2
        maker_entry = _aggressive_maker_entry(direction, bids, asks, tick_size, entry_maker_ticks)
        entry = maker_entry if maker_entry > 0 else mid - spacing * mid
        if sl >= entry:
            sl = entry - risk
        tps = [entry + risk, entry + risk * 2, entry + risk * 3]
    else:
        wall_p, _ = _find_wall(asks, low, high)
        if wall_p <= 0:
            wall_p = mid * 1.01
        sl = wall_p + tick_size * 2
        maker_entry = _aggressive_maker_entry(direction, bids, asks, tick_size, entry_maker_ticks)
        entry = maker_entry if maker_entry > 0 else mid + spacing * mid
        if sl <= entry:
            sl = entry + risk
        tps = [entry - risk, entry - risk * 2, entry - risk * 3]

    r_mult = {"Trending": 2.5, "Breakout": 3.0, "Choppy": 1.5}.get(regime, 2.0) * vm
    conf_scale = 0.85 + 0.15 * min(confidence, 1.0)
    if direction == "LONG":
        tps.append(entry + risk * r_mult * conf_scale)
    else:
        tps.append(entry - risk * r_mult * conf_scale)

    entry = _round_tick(entry, tick_size)
    sl = _round_tick(sl, tick_size)
    anchor = entry if entry > 0 else mid
    tps = [
        _enforce_min_tp(anchor, tp, direction, min_tp_pct, tick_size)
        for tp in tps
        if tp > 0
    ]
    if tps:
        be = _fee_aware_breakeven(anchor, direction, fee_breakeven_pct, tick_size)
        if direction == "LONG":
            tps[0] = max(tps[0], be)
        else:
            tps[0] = min(tps[0], be)
    tps = [_round_tick(tp, tick_size) for tp in tps if tp > 0]

    tp_prices = compute_liquidity_tp_prices(
        direction,
        bids,
        asks,
        entry_price=entry,
        tick_size=tick_size,
        max_distance_pct=max_tp_pct,
    )

    result = {
        "entry_price": entry,
        "stop_loss": sl,
        "take_profits": tps,
        "wall_price": wall_p,
    }
    if tp_prices:
        result["tp_prices"] = tp_prices
    return result
