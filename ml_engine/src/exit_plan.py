"""Exit level planner — liquidity-aware TP/SL with fee compensation.

Core idea:
- TP: Place near liquidity walls where price will slow down
- TP must compensate: entry fee + exit fee + spread + profit margin
- SL: Place beyond support/resistance walls
- SL must give room for noise but limit max loss

Fee structure (Bybit):
- Entry: 0.055% (taker)
- Exit: 0.020% (taker)
- Total round-trip: 0.075%
- Min profitable move: ~0.10%
"""

from __future__ import annotations

import math
from typing import Any

from .liquidity_tp import (
    compute_liquidity_tp_prices,
    compute_liquidity_sl,
    MIN_PROFITABLE_MOVE_PCT,
    ENTRY_FEE_PCT,
    EXIT_FEE_PCT,
)


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


def _infer_tick_size(price: float) -> float:
    if price <= 0:
        return 0.0001
    if price >= 1000:
        return 0.1
    if price >= 100:
        return 0.01
    if price >= 1:
        return 0.0001
    return 0.00001


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


def _detect_knife_conditions(
    macro_trend_5m: float,
    macro_trend_15m: float,
    obi: float,
    regime: str,
    direction: str,
) -> tuple[bool, float, float]:
    """Detect if market is in a knife condition (adverse trend).
    
    Returns: (is_knife, tighten_factor, trend_penalty)
    - tighten_factor: multiply SL and TP distances by this (lower = tighter)
    - trend_penalty: subtract from trade_score (0.0 = no penalty)
    """
    knife_score = 0.0

    if direction == "LONG":
        if macro_trend_5m < -0.001:
            knife_score += 0.3
        if macro_trend_5m < -0.003:
            knife_score += 0.2
        if macro_trend_15m < -0.005:
            knife_score += 0.3
        if macro_trend_15m < -0.010:
            knife_score += 0.1
        if obi < -0.2:
            knife_score += 0.1
        if regime == "Breakout" and macro_trend_5m < -0.001:
            knife_score += 0.1
    else:
        if macro_trend_5m > 0.001:
            knife_score += 0.3
        if macro_trend_5m > 0.003:
            knife_score += 0.2
        if macro_trend_15m > 0.005:
            knife_score += 0.3
        if macro_trend_15m > 0.010:
            knife_score += 0.1
        if obi > 0.2:
            knife_score += 0.1
        if regime == "Breakout" and macro_trend_5m > 0.001:
            knife_score += 0.1

    if knife_score >= 0.4:
        tighten = max(0.25, 1.0 - knife_score * 0.8)
        trend_penalty = min(0.3, knife_score * 0.3)
        return True, tighten, trend_penalty
    return False, 1.0, 0.0


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
    min_tp_pct: float = 0.001,
    fee_breakeven_pct: float = 0.00075,
    max_tp_pct: float = 0.020,
    max_sl_pct: float = 0.015,
    macro_trend_5m: float = 0.0,
    macro_trend_15m: float = 0.0,
) -> dict[str, Any]:
    """Compute entry, SL, and TP using liquidity-aware algorithm.

    TP Logic:
    1. Find liquidity walls in order book (large orders where price slows)
    2. Place TP just before walls (1-2 ticks before)
    3. Only include TPs that cover fees + spread + min profit
    4. Sort by distance (closest first) and take top N

    SL Logic:
    1. Find support/resistance walls (opposite side of trade)
    2. Place SL beyond the wall (2 ticks past)
    3. Cap at max_sl_pct to limit loss
    """
    mid = _mid(bids, asks)
    if mid <= 0 and fallback_mid > 0:
        mid = fallback_mid
    if mid <= 0:
        return {}

    tick = tick_size if tick_size > 0 else _infer_tick_size(mid)

    obi = 0.0
    if bids and asks:
        bid_vol = sum(_level_size(b) for b in bids[:10])
        ask_vol = sum(_level_size(a) for a in asks[:10])
        total = bid_vol + ask_vol
        if total > 0:
            obi = (bid_vol - ask_vol) / total

    is_knife, knife_tighten, trend_penalty = _detect_knife_conditions(
        macro_trend_5m, macro_trend_15m, obi, regime, direction,
    )

    # Entry price
    maker_entry = _aggressive_maker_entry(direction, bids, asks, tick, entry_maker_ticks)
    if direction == "LONG":
        entry = maker_entry if maker_entry > 0 else mid - _round_tick(mid * 0.0005, tick)
    else:
        entry = maker_entry if maker_entry > 0 else mid + _round_tick(mid * 0.0005, tick)

    entry = _round_tick(entry, tick)

    # SL: liquidity-aware, beyond support/resistance walls, scaled by vol_mult
    sl_min_pct = 0.004 * vol_mult
    sl_max_pct = max_sl_pct * vol_mult
    sl = compute_liquidity_sl(
        direction, bids, asks, entry, tick,
        min_sl_distance_pct=sl_min_pct,
        max_sl_distance_pct=sl_max_pct,
        hard_max_sl_pct=0.008,
    )
    if sl <= 0:
        # Fallback: use percentage-based SL
        if direction == "LONG":
            sl = _round_tick(entry * (1 - max_sl_pct), tick)
        else:
            sl = _round_tick(entry * (1 + max_sl_pct), tick)

    # Knife: do NOT tighten SL — entering against trend needs wider stop to avoid noise.
    # Only tighten TP to take quicker profits.

    # TP: liquidity-aware, near walls, fee-compensated
    tp_prices = compute_liquidity_tp_prices(
        direction, bids, asks, entry, tick,
        max_levels=4,
        max_distance_pct=max_tp_pct,
    )

    # Fallback: if no liquidity TPs found, use fee-compensated minimum
    if not tp_prices:
        min_profit_pct = max(min_tp_pct, MIN_PROFITABLE_MOVE_PCT * 3.0)  # 3x fees minimum
        if direction == "LONG":
            tp_prices = [_round_tick(entry * (1 + min_profit_pct), tick)]
        else:
            tp_prices = [_round_tick(entry * (1 - min_profit_pct), tick)]

    # Knife: tighten TP distances (closer to entry when against trend)
    if is_knife and knife_tighten < 1.0:
        tightened_tps = []
        for tp in tp_prices:
            tp_dist = abs(tp - entry)
            new_dist = tp_dist * knife_tighten
            if direction == "LONG":
                tightened_tps.append(_round_tick(entry + new_dist, tick))
            else:
                tightened_tps.append(_round_tick(entry - new_dist, tick))
        tp_prices = tightened_tps

    # Build result
    result = {
        "entry_price": entry,
        "stop_loss": sl,
        "take_profits": tp_prices,
        "wall_price": 0.0,
        "trend_penalty": trend_penalty,
        "fee_info": {
            "entry_fee_pct": ENTRY_FEE_PCT,
            "exit_fee_pct": EXIT_FEE_PCT,
            "total_fee_pct": ENTRY_FEE_PCT + EXIT_FEE_PCT,
            "min_profitable_pct": MIN_PROFITABLE_MOVE_PCT,
        },
    }

    # Add wall price for reference
    if direction == "LONG" and asks:
        # Nearest resistance wall
        for lv in asks[:20]:
            p, s = _level_price(lv), _level_size(lv)
            if p > entry and s > 0:
                result["wall_price"] = p
                break
    elif direction == "SHORT" and bids:
        # Nearest support wall
        for lv in bids[:20]:
            p, s = _level_price(lv), _level_size(lv)
            if p < entry and s > 0:
                result["wall_price"] = p
                break

    return result
