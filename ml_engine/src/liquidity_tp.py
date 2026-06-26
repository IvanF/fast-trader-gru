"""Liquidity-aware take-profit levels from order book clusters.

Core idea: Price slows down near liquidity walls (large orders).
TP should be placed just before these walls, ensuring the move
compensates for exchange fees (entry + exit) and adds profit.

Fee structure (Bybit):
- Entry fee (taker): 0.055%
- Exit fee (taker): 0.020%
- Total round-trip: 0.075%
- Spread buffer: ~0.010%
- Min profitable move: ~0.10% (fees + spread + margin)
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple


# Fee constants (Bybit taker fees)
ENTRY_FEE_PCT = 0.00055  # 0.055%
EXIT_FEE_PCT = 0.00020   # 0.020%
SPREAD_BUFFER_PCT = 0.00010  # 0.010% typical spread
MIN_PROFIT_MARGIN_PCT = 0.00030  # 0.030% minimum profit after fees

# Total minimum move to be profitable
MIN_PROFITABLE_MOVE_PCT = ENTRY_FEE_PCT + EXIT_FEE_PCT + SPREAD_BUFFER_PCT + MIN_PROFIT_MARGIN_PCT


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


def _round_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(price / tick) * tick


def _find_liquidity_walls(
    levels: Sequence[Any],
    top_n: int = 30,
    window: int = 3,
    spike_ratio: float = 1.5,
) -> List[Tuple[float, float]]:
    """Find price levels with unusually large order sizes (walls).

    A wall is a price level where size exceeds local average by spike_ratio.
    These are areas where price is likely to slow down or reverse.
    """
    parsed: List[Tuple[float, float]] = []
    for lv in levels[:top_n]:
        p, s = _level_price(lv), _level_size(lv)
        if p > 0 and s > 0:
            parsed.append((p, s))
    if not parsed:
        return []

    walls: List[Tuple[float, float]] = []
    for i, (price, size) in enumerate(parsed):
        start = max(0, i - window)
        end = min(len(parsed), i + window + 1)
        neighbors = [parsed[j][1] for j in range(start, end) if j != i]
        local_avg = sum(neighbors) / len(neighbors) if neighbors else size
        if local_avg <= 0:
            continue
        if size >= local_avg * spike_ratio:
            walls.append((price, size))

    tick = _infer_tick_size(parsed[0][0])
    deduped: List[Tuple[float, float]] = []
    for price, size in walls:
        if any(abs(price - p) <= tick * 3 for p, _ in deduped):
            continue
        deduped.append((price, size))
    return deduped


# Alias for backward compatibility
_liquidity_nodes = _find_liquidity_walls


def _cluster_walls(
    walls: List[Tuple[float, float]],
    tick: float,
    cluster_pct: float = 0.002,
) -> List[Tuple[float, float, float]]:
    """Cluster nearby walls and sum their sizes.

    Returns (price, total_size, wall_count) for each cluster.
    """
    if not walls:
        return []

    sorted_walls = sorted(walls, key=lambda x: x[0])
    clusters: List[List[Tuple[float, float]]] = [[sorted_walls[0]]]

    for price, size in sorted_walls[1:]:
        last_cluster = clusters[-1]
        last_price = last_cluster[-1][0]
        if abs(price - last_price) / max(last_price, 1e-8) <= cluster_pct:
            last_cluster.append((price, size))
        else:
            clusters.append([(price, size)])

    result = []
    for cluster in clusters:
        avg_price = sum(p for p, _ in cluster) / len(cluster)
        total_size = sum(s for _, s in cluster)
        result.append((avg_price, total_size, len(cluster)))
    return result


def compute_liquidity_tp_prices(
    direction: str,
    bids: Sequence[Any],
    asks: Sequence[Any],
    entry_price: float = 0.0,
    tick_size: float = 0.0,
    top_n: int = 30,
    tick_offset: int = 2,
    max_levels: int = 4,
    max_distance_pct: float = 0.020,
) -> List[float]:
    """Find TP levels near liquidity walls.

    For LONG: scan asks (resistance walls above entry).
              Place TP 1-2 ticks below each wall.
    For SHORT: scan bids (support walls below entry).
               Place TP 1-2 ticks above each wall.

    Only includes walls that are:
    1. In the direction of the trade (profit direction)
    2. Far enough to cover fees + spread + min profit
    3. Close enough to be realistic target

    Returns prices sorted closest-to-entry first.
    """
    if direction not in ("LONG", "SHORT"):
        return []

    ref = entry_price
    if ref <= 0:
        best_bid = _level_price(bids[0]) if bids else 0.0
        best_ask = _level_price(asks[0]) if asks else 0.0
        ref = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
    if ref <= 0:
        return []

    tick = tick_size if tick_size > 0 else _infer_tick_size(ref)
    offset = max(1, tick_offset) * tick

    # Find walls in the profit direction
    if direction == "LONG":
        walls = _find_liquidity_walls(asks, top_n=top_n)
    else:
        walls = _find_liquidity_walls(bids, top_n=top_n)

    if not walls:
        return []

    # Cluster nearby walls
    clustered = _cluster_walls(walls, tick)

    # Filter and score walls
    candidates: List[Tuple[float, float, float]] = []
    for wall_price, wall_size, wall_count in clustered:
        if direction == "LONG":
            # Wall must be above entry (profit direction)
            if wall_price <= ref:
                continue
            distance_pct = (wall_price - ref) / ref
            # Place TP just below the wall (price slows before wall)
            tp = _round_tick(wall_price - offset, tick)
            if tp <= ref:
                continue
        else:
            # Wall must be below entry (profit direction)
            if wall_price >= ref:
                continue
            distance_pct = (ref - wall_price) / ref
            # Place TP just above the wall (price slows before wall)
            tp = _round_tick(wall_price + offset, tick)
            if tp >= ref:
                continue

        # Must be profitable after fees
        if distance_pct < MIN_PROFITABLE_MOVE_PCT:
            continue

        # Must be within max distance
        if max_distance_pct > 0 and distance_pct > max_distance_pct:
            continue

        # Score: closer walls with larger size are better
        # Prefer walls that are 0.3%-1.5% away (sweet spot)
        optimal_dist = 0.008  # 0.8%
        dist_score = 1.0 / (1.0 + abs(distance_pct - optimal_dist) / optimal_dist)
        size_score = min(wall_size / 1000, 2.0)  # Normalize size
        count_score = min(wall_count / 2, 1.5)  # Cluster bonus
        total_score = dist_score * (1 + size_score * 0.3 + count_score * 0.2)

        candidates.append((tp, total_score, distance_pct))

    if not candidates:
        return []

    # Sort by score (best first)
    candidates.sort(key=lambda x: -x[1])

    # Take top levels, ensure minimum spacing
    result: List[float] = []
    min_spacing_pct = 0.001  # 0.1% minimum between TPs

    for tp, score, dist in candidates:
        if len(result) >= max_levels:
            break
        # Check minimum spacing from existing TPs
        too_close = False
        for existing in result:
            if abs(tp - existing) / max(ref, 1e-8) < min_spacing_pct:
                too_close = True
                break
        if not too_close:
            result.append(tp)

    # Sort by distance (closest to entry first)
    if direction == "LONG":
        result.sort()
    else:
        result.sort(reverse=True)

    return result


def compute_liquidity_sl(
    direction: str,
    bids: Sequence[Any],
    asks: Sequence[Any],
    entry_price: float = 0.0,
    tick_size: float = 0.0,
    min_sl_distance_pct: float = 0.005,
    max_sl_distance_pct: float = 0.015,
) -> float:
    """Find SL level near support/resistance walls.

    For LONG: place SL below nearest support wall (bid wall below entry).
    For SHORT: place SL above nearest resistance wall (ask wall above entry).

    SL should be:
    1. Beyond a wall (so if price breaks the wall, we're out)
    2. Not too tight (avoid noise stop-outs)
    3. Not too wide (limit max loss)
    """
    ref = entry_price
    if ref <= 0:
        return 0.0

    tick = tick_size if tick_size > 0 else _infer_tick_size(ref)

    if direction == "LONG":
        # Find support walls below entry
        walls = _find_liquidity_walls(bids, top_n=30)
        candidates = []
        for wall_price, wall_size in walls:
            if wall_price >= ref:
                continue
            distance_pct = (ref - wall_price) / ref
            if distance_pct < min_sl_distance_pct:
                continue
            if distance_pct > max_sl_distance_pct:
                continue
            # Place SL below the wall (2 ticks below)
            sl = _round_tick(wall_price - tick * 2, tick)
            if sl >= ref:
                continue
            candidates.append((sl, wall_size, distance_pct))

        if candidates:
            # Prefer strongest wall that's not too far
            candidates.sort(key=lambda x: -x[1])
            return candidates[0][0]

        # Fallback: use max SL distance
        return _round_tick(ref * (1 - max_sl_distance_pct), tick)

    else:  # SHORT
        # Find resistance walls above entry
        walls = _find_liquidity_walls(asks, top_n=30)
        candidates = []
        for wall_price, wall_size in walls:
            if wall_price <= ref:
                continue
            distance_pct = (wall_price - ref) / ref
            if distance_pct < min_sl_distance_pct:
                continue
            if distance_pct > max_sl_distance_pct:
                continue
            # Place SL above the wall (2 ticks above)
            sl = _round_tick(wall_price + tick * 2, tick)
            if sl <= ref:
                continue
            candidates.append((sl, wall_size, distance_pct))

        if candidates:
            # Prefer strongest wall that's not too far
            candidates.sort(key=lambda x: -x[1])
            return candidates[0][0]

        # Fallback: use max SL distance
        return _round_tick(ref * (1 + max_sl_distance_pct), tick)
