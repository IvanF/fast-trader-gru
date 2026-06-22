"""Exit level planner — smart adaptive SL with S/R awareness, knife detection, and macro trend."""

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


def _find_strongest_level(levels: list, ref: float, direction: str) -> tuple[float, float]:
    """Find the strongest liquidity level relative to ref price.

    For SHORT SL: scan asks ABOVE ref (resistance).
    For LONG SL: scan bids BELOW ref (support).
    Returns (price, size) of the strongest level.
    """
    candidates = []
    for lv in levels:
        p, s = _level_price(lv), _level_size(lv)
        if s <= 0:
            continue
        if direction == "SHORT" and p > ref:
            candidates.append((p, s))
        elif direction == "LONG" and p < ref:
            candidates.append((p, s))
    if not candidates:
        return 0.0, 0.0
    idx = max(range(len(candidates)), key=lambda i: candidates[i][1])
    return candidates[idx]


def _find_nearest_level(levels: list, ref: float, direction: str) -> tuple[float, float]:
    """Find the nearest liquidity level relative to ref price.

    For SHORT SL: scan asks ABOVE ref, return nearest.
    For LONG SL: scan bids BELOW ref, return nearest.
    Returns (price, size) of the nearest level.
    """
    candidates = []
    for lv in levels:
        p, s = _level_price(lv), _level_size(lv)
        if s <= 0:
            continue
        if direction == "SHORT" and p > ref:
            candidates.append((p, s))
        elif direction == "LONG" and p < ref:
            candidates.append((p, s))
    if not candidates:
        return 0.0, 0.0
    idx = min(range(len(candidates)), key=lambda i: abs(candidates[i][0] - ref))
    return candidates[idx]


def _detect_knife_conditions(
    macro_trend_5m: float,
    macro_trend_15m: float,
    obi: float,
    regime: str,
    direction: str,
) -> tuple[bool, float]:
    """Detect if market is in a knife condition (rapid adverse movement).

    Returns (is_knife, tighten_factor).
    tighten_factor: 1.0 = normal, <1.0 = tighten SL (e.g., 0.5 = half the normal SL distance).
    """
    knife_score = 0.0

    if direction == "LONG":
        if macro_trend_5m < -0.003:
            knife_score += 0.4
        if macro_trend_15m < -0.008:
            knife_score += 0.3
        if obi < -0.2:
            knife_score += 0.2
        if regime == "Breakout" and macro_trend_5m < -0.002:
            knife_score += 0.1
    else:
        if macro_trend_5m > 0.003:
            knife_score += 0.4
        if macro_trend_15m > 0.008:
            knife_score += 0.3
        if obi > 0.2:
            knife_score += 0.2
        if regime == "Breakout" and macro_trend_5m > 0.002:
            knife_score += 0.1

    if knife_score >= 0.5:
        tighten = max(0.3, 1.0 - knife_score * 0.8)
        return True, tighten
    return False, 1.0


def _compute_smart_sl(
    direction: str,
    entry: float,
    bids: list,
    asks: list,
    mid: float,
    tick_size: float,
    risk: float,
    max_sl_pct: float,
    knife_tighten: float,
    macro_trend_15m: float,
) -> float:
    """Compute smart SL using S/R levels, capped by max distance, tightened for knives.

    For LONG: SL below nearest support (bid wall) + 2 ticks.
    For SHORT: SL above nearest resistance (ask wall) + 2 ticks.
    """
    if direction == "LONG":
        support_price, _ = _find_nearest_level(bids, entry, "LONG")
        if support_price > 0:
            sl = support_price - tick_size * 2
        else:
            sl = entry - risk

        if macro_trend_15m < -0.005:
            trend_tighten = max(0.6, 1.0 + macro_trend_15m * 10)
            sl = entry - (entry - sl) * trend_tighten

        sl = entry - (entry - sl) * knife_tighten

        max_sl_dist = entry * max_sl_pct
        min_sl = entry - max_sl_dist
        if sl < min_sl:
            sl = min_sl

        if sl >= entry:
            sl = entry - risk

    else:
        resistance_price, _ = _find_nearest_level(asks, entry, "SHORT")
        if resistance_price > 0:
            sl = resistance_price + tick_size * 2
        else:
            sl = entry + risk

        if macro_trend_15m > 0.005:
            trend_tighten = max(0.6, 1.0 - macro_trend_15m * 10)
            sl = entry + (sl - entry) * trend_tighten

        sl = entry + (sl - entry) * knife_tighten

        max_sl_dist = entry * max_sl_pct
        max_sl = entry + max_sl_dist
        if sl > max_sl:
            sl = max_sl

        if sl <= entry:
            sl = entry + risk

    return _round_tick(sl, tick_size)


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
    max_sl_pct: float = 0.012,
    macro_trend_5m: float = 0.0,
    macro_trend_15m: float = 0.0,
) -> dict[str, Any]:
    """Compute entry anchor, smart SL (S/R-aware + knife detection), TP ladder from NN regime/vol."""
    mid = _mid(bids, asks)
    if mid <= 0 and fallback_mid > 0:
        mid = fallback_mid
    if mid <= 0:
        return {}

    vm = _cap_vol_multiplier(vol_mult, vol_multiplier_cap)
    spacing = BASE_GRID_SPACING * vm
    risk = spacing * mid

    obi = 0.0
    if bids and asks:
        bid_vol = sum(_level_size(b) for b in bids[:10])
        ask_vol = sum(_level_size(a) for a in asks[:10])
        total = bid_vol + ask_vol
        if total > 0:
            obi = (bid_vol - ask_vol) / total

    is_knife, knife_tighten = _detect_knife_conditions(
        macro_trend_5m, macro_trend_15m, obi, regime, direction,
    )

    maker_entry = _aggressive_maker_entry(direction, bids, asks, tick_size, entry_maker_ticks)
    if direction == "LONG":
        entry = maker_entry if maker_entry > 0 else mid - spacing * mid
    else:
        entry = maker_entry if maker_entry > 0 else mid + spacing * mid

    sl = _compute_smart_sl(
        direction, entry, bids, asks, mid, tick_size, risk,
        max_sl_pct, knife_tighten, macro_trend_15m,
    )

    if is_knife:
        risk = abs(entry - sl)

    r_mult = {"Trending": 2.5, "Breakout": 3.0, "Choppy": 1.5}.get(regime, 2.0) * vm
    conf_scale = 0.85 + 0.15 * min(confidence, 1.0)
    if direction == "LONG":
        tps = [entry + risk, entry + risk * 2, entry + risk * 3]
        tps.append(entry + risk * r_mult * conf_scale)
    else:
        tps = [entry - risk, entry - risk * 2, entry - risk * 3]
        tps.append(entry - risk * r_mult * conf_scale)

    entry = _round_tick(entry, tick_size)
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

    wall_price = 0.0
    if direction == "LONG":
        support_p, _ = _find_nearest_level(bids, entry, "LONG")
        wall_price = support_p
    else:
        resistance_p, _ = _find_nearest_level(asks, entry, "SHORT")
        wall_price = resistance_p

    result = {
        "entry_price": entry,
        "stop_loss": sl,
        "take_profits": tps,
        "wall_price": wall_price,
    }
    if tp_prices:
        result["tp_prices"] = tp_prices
    return result
