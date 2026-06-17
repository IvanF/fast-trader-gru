"""Market microstructure telemetry and pattern event detection for Grafana."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .features import SymbolBuffer, _level_price, _level_size
from .liquidity_tp import _liquidity_nodes

TELEMETRY_MEASUREMENT = "market_microstructure"
EVENT_MEASUREMENT = "pattern_events"


@dataclass
class TelemetrySnapshot:
    mid: float = 0.0
    vwap: float = 0.0
    vwap_deviation_pct: float = 0.0
    obi: float = 0.0
    cvd_norm: float = 0.0
    order_flow_speed: float = 0.0
    atr_pct: float = 0.0
    volatility_pct: float = 0.0
    spread_bps: float = 0.0
    bid_wall_price: float = 0.0
    ask_wall_price: float = 0.0
    bid_wall_dist_pct: float = 0.0
    ask_wall_dist_pct: float = 0.0
    trend_15m_pct: float = 0.0
    trend_1h_pct: float = 0.0
    funding_rate: float = 0.0
    trade_intensity: float = 0.0
    liquidation_score: float = 0.0
    regime_vol_ema: float = 0.0

    def as_fields(self) -> dict[str, float]:
        return {
            "mid": self.mid,
            "vwap": self.vwap,
            "vwap_deviation_pct": self.vwap_deviation_pct,
            "obi": self.obi,
            "cvd_norm": self.cvd_norm,
            "order_flow_speed": self.order_flow_speed,
            "atr_pct": self.atr_pct,
            "volatility_pct": self.volatility_pct,
            "spread_bps": self.spread_bps,
            "bid_wall_price": self.bid_wall_price,
            "ask_wall_price": self.ask_wall_price,
            "bid_wall_dist_pct": self.bid_wall_dist_pct,
            "ask_wall_dist_pct": self.ask_wall_dist_pct,
            "trend_15m_pct": self.trend_15m_pct,
            "trend_1h_pct": self.trend_1h_pct,
            "funding_rate": self.funding_rate,
            "trade_intensity": self.trade_intensity,
            "liquidation_score": self.liquidation_score,
            "regime_vol_ema": self.regime_vol_ema,
        }


@dataclass
class PatternEvent:
    event_type: str
    price: float
    score: float
    detail: str = ""


@dataclass
class SymbolEventState:
    prev_mid: float = 0.0
    support_wall: float = 0.0
    resistance_wall: float = 0.0
    near_support_ticks: int = 0
    near_resistance_ticks: int = 0
    local_min: float = 0.0
    local_max: float = 0.0
    vol_ema: float = 0.0
    recent_trade_sizes: Deque[Tuple[float, float, str]] = field(default_factory=lambda: deque(maxlen=200))


def _mid_from_book(buf: SymbolBuffer) -> float:
    bids, asks = buf.latest_bids, buf.latest_asks
    if bids and asks:
        bid = _level_price(bids[0])
        ask = _level_price(asks[0])
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    return buf.last_mid()


def _spread_bps(buf: SymbolBuffer) -> float:
    bids, asks = buf.latest_bids, buf.latest_asks
    if not bids or not asks:
        return 0.0
    bid = _level_price(bids[0])
    ask = _level_price(asks[0])
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 0.0
    return (ask - bid) / mid * 10_000.0


def _vwap(buf: SymbolBuffer) -> tuple[float, float]:
    trades = [p for p in buf.points if p.size > 0]
    if not trades:
        mid = _mid_from_book(buf)
        return mid, 0.0
    vol = sum(t.size for t in trades)
    if vol <= 0:
        mid = trades[-1].price
        return mid, 0.0
    vwap = sum(t.price * t.size for t in trades) / vol
    last = trades[-1].price
    dev = (last - vwap) / vwap if vwap > 0 else 0.0
    return vwap, dev


def _atr_pct(buf: SymbolBuffer, period: int = 14) -> float:
    prices = [p for _, p in buf.price_history][-period - 1 :]
    if len(prices) < 3:
        return 0.0
    trs = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    atr = float(np.mean(trs))
    mid = prices[-1]
    return atr / mid if mid > 0 else 0.0


def _volatility_pct(buf: SymbolBuffer, window: int = 60) -> float:
    prices = [p for _, p in buf.price_history][-window - 1 :]
    if len(prices) < 5:
        return 0.0
    rets = np.diff(prices) / np.array(prices[:-1])
    return float(np.std(rets))


def _nearest_walls(buf: SymbolBuffer, mid: float) -> tuple[float, float]:
    bid_nodes = _liquidity_nodes(buf.latest_bids, top_n=25)
    ask_nodes = _liquidity_nodes(buf.latest_asks, top_n=25)
    support = 0.0
    resistance = 0.0
    for price, _ in bid_nodes:
        if price < mid and (support <= 0 or price > support):
            support = price
    for price, _ in ask_nodes:
        if price > mid and (resistance <= 0 or price < resistance):
            resistance = price
    return support, resistance


def _trade_intensity(buf: SymbolBuffer, window_sec: float = 60.0) -> float:
    if not buf.points:
        return 0.0
    now = buf.points[-1].ts
    trades = [p for p in buf.points if p.size > 0 and p.ts >= now - window_sec]
    if len(trades) < 2:
        return float(len(trades))
    duration = trades[-1].ts - trades[0].ts
    if duration <= 0:
        return float(len(trades))
    return len(trades) / duration


def _liquidation_score(buf: SymbolBuffer, window_sec: float = 30.0) -> tuple[float, str]:
    """Heuristic 0..1 score for aggressive one-sided liquidation flow."""
    if not buf.points:
        return 0.0, ""
    now = buf.points[-1].ts
    trades = [p for p in buf.points if p.size > 0 and p.ts >= now - window_sec]
    if len(trades) < 3:
        return 0.0, ""

    sizes = [t.size for t in trades]
    median = float(np.median(sizes))
    if median <= 0:
        return 0.0, ""

    sell_vol = sum(t.size for t in trades if t.side.upper() in ("SELL", "S"))
    buy_vol = sum(t.size for t in trades if t.side.upper() in ("BUY", "B"))
    total = sell_vol + buy_vol
    if total <= 0:
        return 0.0, ""

    large = [t for t in trades if t.size >= median * 2.5]
    if len(large) < 2:
        return 0.0, ""

    prices = [t.price for t in trades]
    move_pct = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0.0
    imbalance = abs(sell_vol - buy_vol) / total
    size_spike = sum(t.size for t in large) / total
    score = min(1.0, imbalance * 0.5 + size_spike * 0.35 + min(abs(move_pct) / 0.004, 1.0) * 0.15)

    if score < 0.55:
        return score, ""
    if sell_vol > buy_vol * 1.5 and move_pct < -0.002:
        return score, "mass_liquidation_long"
    if buy_vol > sell_vol * 1.5 and move_pct > 0.002:
        return score, "mass_liquidation_short"
    return score, ""


def compute_telemetry(buf: SymbolBuffer) -> TelemetrySnapshot:
    mid = _mid_from_book(buf)
    vwap, vwap_dev = _vwap(buf)
    support, resistance = _nearest_walls(buf, mid)
    liq_score, _ = _liquidation_score(buf)

    snap = TelemetrySnapshot(
        mid=mid,
        vwap=vwap,
        vwap_deviation_pct=vwap_dev,
        obi=buf.order_book_imbalance(),
        cvd_norm=float(np.tanh(buf.cvd / 1e6)),
        order_flow_speed=buf.order_flow_speed(),
        atr_pct=_atr_pct(buf),
        volatility_pct=_volatility_pct(buf),
        spread_bps=_spread_bps(buf),
        bid_wall_price=support,
        ask_wall_price=resistance,
        bid_wall_dist_pct=(mid - support) / mid if mid > 0 and support > 0 else 0.0,
        ask_wall_dist_pct=(resistance - mid) / mid if mid > 0 and resistance > 0 else 0.0,
        trend_15m_pct=buf.macro_trend(900),
        trend_1h_pct=buf.macro_trend(3600),
        funding_rate=buf.funding_rate,
        trade_intensity=_trade_intensity(buf),
        liquidation_score=liq_score,
    )
    return snap


def detect_pattern_events(
    buf: SymbolBuffer,
    snap: TelemetrySnapshot,
    state: SymbolEventState,
) -> List[PatternEvent]:
    events: List[PatternEvent] = []
    mid = snap.mid
    if mid <= 0:
        return events

    state.support_wall, state.resistance_wall = snap.bid_wall_price, snap.ask_wall_price
    if state.local_min <= 0 or mid < state.local_min:
        state.local_min = mid
    if state.local_max <= 0 or mid > state.local_max:
        state.local_max = mid

    near_pct = 0.0025
    bounce_pct = 0.0015
    breakout_pct = 0.0010

    # --- Bounce from bid liquidity (support) ---
    if state.support_wall > 0:
        dist = abs(mid - state.support_wall) / state.support_wall
        if dist <= near_pct:
            state.near_support_ticks += 1
        elif state.near_support_ticks >= 2 and mid > state.local_min * (1.0 + bounce_pct):
            events.append(PatternEvent(
                event_type="bounce_support",
                price=mid,
                score=min(1.0, state.near_support_ticks / 5.0),
                detail=f"wall={state.support_wall:.6f} obi={snap.obi:.3f}",
            ))
            state.near_support_ticks = 0
            state.local_min = mid
        else:
            state.near_support_ticks = 0

    # --- Bounce from ask liquidity (resistance) ---
    if state.resistance_wall > 0:
        dist = abs(mid - state.resistance_wall) / state.resistance_wall
        if dist <= near_pct:
            state.near_resistance_ticks += 1
        elif state.near_resistance_ticks >= 2 and mid < state.local_max * (1.0 - bounce_pct):
            events.append(PatternEvent(
                event_type="bounce_resistance",
                price=mid,
                score=min(1.0, state.near_resistance_ticks / 5.0),
                detail=f"wall={state.resistance_wall:.6f} obi={snap.obi:.3f}",
            ))
            state.near_resistance_ticks = 0
            state.local_max = mid
        else:
            state.near_resistance_ticks = 0

    # --- Level breakout ---
    prev = state.prev_mid
    if prev > 0 and state.support_wall > 0:
        if prev >= state.support_wall and mid < state.support_wall * (1.0 - breakout_pct):
            events.append(PatternEvent(
                event_type="breakout_down",
                price=mid,
                score=min(1.0, abs(snap.trend_15m_pct) * 50 + snap.volatility_pct * 100),
                detail=f"support={state.support_wall:.6f} vol={snap.volatility_pct:.4f}",
            ))
    if prev > 0 and state.resistance_wall > 0:
        if prev <= state.resistance_wall and mid > state.resistance_wall * (1.0 + breakout_pct):
            events.append(PatternEvent(
                event_type="breakout_up",
                price=mid,
                score=min(1.0, abs(snap.trend_15m_pct) * 50 + snap.volatility_pct * 100),
                detail=f"resistance={state.resistance_wall:.6f} vol={snap.volatility_pct:.4f}",
            ))

    # --- Mass liquidation ---
    _, liq_type = _liquidation_score(buf)
    if liq_type:
        events.append(PatternEvent(
            event_type=liq_type,
            price=mid,
            score=snap.liquidation_score,
            detail=f"intensity={snap.trade_intensity:.2f}/s cvd={snap.cvd_norm:.3f}",
        ))

    vol = snap.volatility_pct
    state.vol_ema = 0.15 * vol + 0.85 * state.vol_ema if state.vol_ema > 0 else vol
    snap.regime_vol_ema = state.vol_ema
    state.prev_mid = mid
    return events


def compute_telemetry_and_events(
    buf: SymbolBuffer,
    state: Optional[SymbolEventState],
) -> tuple[TelemetrySnapshot, List[PatternEvent]]:
    if state is None:
        state = SymbolEventState()
    snap = compute_telemetry(buf)
    events = detect_pattern_events(buf, snap, state)
    return snap, events


class EventStateStore:
    def __init__(self) -> None:
        self._states: Dict[str, SymbolEventState] = {}

    def get(self, symbol: str) -> SymbolEventState:
        if symbol not in self._states:
            self._states[symbol] = SymbolEventState()
        return self._states[symbol]
