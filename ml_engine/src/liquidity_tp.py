"""Liquidity-aware take-profit levels from order book clusters."""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple


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


def _liquidity_nodes(
    levels: Sequence[Any],
    top_n: int = 25,
    window: int = 3,
    spike_ratio: float = 1.8,
) -> List[Tuple[float, float]]:
    """Return (price, size) nodes where size exceeds local average by spike_ratio."""
    parsed: List[Tuple[float, float]] = []
    for lv in levels[:top_n]:
        p, s = _level_price(lv), _level_size(lv)
        if p > 0 and s > 0:
            parsed.append((p, s))
    if not parsed:
        return []

    nodes: List[Tuple[float, float]] = []
    for i, (price, size) in enumerate(parsed):
        start = max(0, i - window)
        end = min(len(parsed), i + window + 1)
        neighbors = [parsed[j][1] for j in range(start, end) if j != i]
        local_avg = sum(neighbors) / len(neighbors) if neighbors else size
        if local_avg <= 0:
            continue
        if size >= local_avg * spike_ratio:
            nodes.append((price, size))

    # Deduplicate nearby prices (within 2 ticks).
    tick = _infer_tick_size(parsed[0][0])
    deduped: List[Tuple[float, float]] = []
    for price, size in nodes:
        if any(abs(price - p) <= tick * 2 for p, _ in deduped):
            continue
        deduped.append((price, size))
    return deduped


def compute_liquidity_tp_prices(
    direction: str,
    bids: Sequence[Any],
    asks: Sequence[Any],
    entry_price: float = 0.0,
    tick_size: float = 0.0,
    top_n: int = 25,
    tick_offset: int = 2,
    max_levels: int = 4,
    max_distance_pct: float = 0.008,
) -> List[float]:
    """
  For SHORT: scan bids, place TP 1-2 ticks above bid liquidity nodes.
  For LONG: scan asks, place TP 1-2 ticks below ask liquidity nodes.
  Returns prices sorted closest-to-entry first.
    """
    if direction not in ("LONG", "SHORT"):
        return []

    levels = bids if direction == "SHORT" else asks
    nodes = _liquidity_nodes(levels, top_n=top_n)
    if not nodes:
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

    tps: List[float] = []
    for node_price, _ in nodes:
        if direction == "SHORT":
            if node_price >= ref:
                continue
            tp = _round_tick(node_price + offset, tick)
            if tp >= ref or tp <= 0:
                continue
            if max_distance_pct > 0 and (ref - tp) / ref > max_distance_pct:
                continue
        else:
            if node_price <= ref:
                continue
            tp = _round_tick(node_price - offset, tick)
            if tp <= ref or tp <= 0:
                continue
            if max_distance_pct > 0 and (tp - ref) / ref > max_distance_pct:
                continue
        tps.append(tp)

    if not tps:
        return []

    if direction == "SHORT":
        tps.sort(reverse=True)
    else:
        tps.sort()

    # Unique, keep order.
    unique: List[float] = []
    for p in tps:
        if not any(math.isclose(p, u, rel_tol=0, abs_tol=tick * 0.5) for u in unique):
            unique.append(p)

    return unique[:max_levels]
