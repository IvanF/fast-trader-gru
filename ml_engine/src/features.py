"""Rolling feature engineering for order flow and orderbook metrics."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


def _level_price(level) -> float:
    if isinstance(level, dict):
        raw = level.get("price") or level.get("Price")
        return float(raw) if raw is not None else 0.0
    if isinstance(level, (list, tuple)) and level:
        return float(level[0])
    return 0.0


def _level_size(level) -> float:
    if isinstance(level, dict):
        raw = level.get("size") or level.get("Size")
        return float(raw) if raw is not None else 0.0
    if isinstance(level, (list, tuple)) and len(level) > 1:
        return float(level[1])
    return 0.0


@dataclass
class TickPoint:
    ts: float
    price: float
    size: float
    side: str
    bid_vol: float
    ask_vol: float


@dataclass
class SymbolBuffer:
    symbol: str
    window_sec: int = 300
    points: Deque[TickPoint] = field(default_factory=deque)
    cvd: float = 0.0
    funding_rate: float = 0.0
    price_history: Deque[Tuple[float, float]] = field(default_factory=deque)
    latest_bids: list = field(default_factory=list)
    latest_asks: list = field(default_factory=list)

    def add_trade(self, ts_ms: int, price: float, size: float, side: str,
                  bid_vol: float = 0.0, ask_vol: float = 0.0) -> None:
        ts = ts_ms / 1000.0
        self.points.append(TickPoint(ts, price, size, side, bid_vol, ask_vol))
        signed = size if side.upper() in ("BUY", "B") else -size
        self.cvd += signed
        self.price_history.append((ts, price))
        self._trim(ts)

    def add_orderbook(self, ts_ms: int, bids: list, asks: list) -> None:
        ts = ts_ms / 1000.0
        self.latest_bids = bids
        self.latest_asks = asks
        bid_vol = sum(_level_size(b) for b in bids[:10])
        ask_vol = sum(_level_size(a) for a in asks[:10])
        mid = 0.0
        if bids and asks:
            mid = (_level_price(bids[0]) + _level_price(asks[0])) / 2.0
        if mid > 0:
            self.price_history.append((ts, mid))
        self.points.append(TickPoint(ts, mid, 0.0, "OB", bid_vol, ask_vol))
        self._trim(ts)

    def last_mid(self) -> float:
        if self.price_history:
            return self.price_history[-1][1]
        if self.points:
            return self.points[-1].price
        return 0.0

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self.points and self.points[0].ts < cutoff:
            self.points.popleft()
        while self.price_history and self.price_history[0][0] < now - 3600:
            self.price_history.popleft()

    def order_book_imbalance(self) -> float:
        if not self.points:
            return 0.0
        last = self.points[-1]
        total = last.bid_vol + last.ask_vol
        if total <= 0:
            return 0.0
        return (last.bid_vol - last.ask_vol) / total

    def order_flow_speed(self) -> float:
        if len(self.points) < 2:
            return 0.0
        trades = [p for p in self.points if p.side != "OB"]
        if len(trades) < 2:
            return 0.0
        duration = trades[-1].ts - trades[0].ts
        if duration <= 0:
            return 0.0
        return len(trades) / duration

    def vwap_deviation(self) -> float:
        trades = [p for p in self.points if p.size > 0]
        if not trades:
            return 0.0
        vol = sum(t.size for t in trades)
        if vol <= 0:
            return 0.0
        vwap = sum(t.price * t.size for t in trades) / vol
        last_price = trades[-1].price
        if vwap <= 0:
            return 0.0
        return (last_price - vwap) / vwap

    def macro_trend(self, horizon_sec: int) -> float:
        if not self.price_history:
            return 0.0
        now = self.price_history[-1][0]
        cutoff = now - horizon_sec
        prices = [p for t, p in self.price_history if t >= cutoff]
        if len(prices) < 2:
            return 0.0
        base = prices[0]
        if abs(base) < 1e-10:
            return 0.0
        return (prices[-1] - base) / base

    def _obi_reversal(self) -> float:
        ob_points = [p for p in self.points if p.side == "OB"]
        if len(ob_points) < 5:
            return 0.0
        recent = ob_points[-60:]
        obi_vals = []
        for p in recent:
            total = p.bid_vol + p.ask_vol
            if total > 0:
                obi_vals.append((p.bid_vol - p.ask_vol) / total)
        if len(obi_vals) < 2:
            return 0.0
        return float(max(obi_vals) - min(obi_vals))

    def _pre_entry_sweep(self) -> float:
        trade_points = [p for p in self.points if p.side == "BUY" or p.side == "SELL"]
        if len(trade_points) < 2:
            return 0.0
        recent = trade_points[-10:]
        if not recent:
            return 0.0
        sizes = [abs(p.size) for p in recent]
        if not sizes:
            return 0.0
        avg_size = np.mean(sizes)
        max_size = max(sizes)
        if max_size > avg_size * 3.0 and avg_size > 0:
            return 1.0
        return 0.0

    def _fill_delay_norm(self) -> float:
        trade_points = [p for p in self.points if p.side in ("BUY", "SELL")]
        if len(trade_points) < 2:
            return 0.5
        delays = []
        for i in range(1, len(trade_points)):
            dt = trade_points[i].ts - trade_points[i-1].ts
            if dt > 0:
                delays.append(dt)
        if not delays:
            return 0.5
        avg_delay = np.mean(delays)
        return float(np.clip(avg_delay / 60.0, 0.0, 1.0))

    def liquidity_features(self) -> np.ndarray:
        bid_vol = sum(_level_size(b) for b in self.latest_bids[:10]) if self.latest_bids else 0.0
        ask_vol = sum(_level_size(a) for a in self.latest_asks[:10]) if self.latest_asks else 0.0
        total_depth = bid_vol + ask_vol
        depth_imbalance = (bid_vol - ask_vol) / total_depth if total_depth > 0 else 0.0

        top3_bid = sum(_level_size(b) for b in self.latest_bids[:3]) if self.latest_bids else 0.0
        top3_ask = sum(_level_size(a) for a in self.latest_asks[:3]) if self.latest_asks else 0.0
        top3_total = top3_bid + top3_ask
        depth_concentration = top3_total / total_depth if total_depth > 0 else 0.0

        spread_bps = 0.0
        if self.latest_bids and self.latest_asks:
            best_bid = _level_price(self.latest_bids[0])
            best_ask = _level_price(self.latest_asks[0])
            if best_bid > 0 and best_ask > 0:
                spread_bps = (best_ask - best_bid) / best_bid * 10000

        avg_level_size = total_depth / 10.0 if total_depth > 0 else 0.0
        avg_fill = np.mean([p.size for p in self.points if p.size > 0]) if any(p.size > 0 for p in self.points) else 0.0
        fill_to_depth = avg_fill / avg_level_size if avg_level_size > 0 else 0.0

        bid_levels = sum(1 for b in self.latest_bids[:10] if _level_size(b) > 0) if self.latest_bids else 0
        ask_levels = sum(1 for a in self.latest_asks[:10] if _level_size(a) > 0) if self.latest_asks else 0
        level_density = (bid_levels + ask_levels) / 20.0

        return np.array([
            depth_imbalance,
            depth_concentration,
            spread_bps,
            fill_to_depth,
            level_density,
            np.tanh(bid_vol / 1e6),
            np.tanh(ask_vol / 1e6),
        ], dtype=np.float32)

    def feature_vector(self) -> np.ndarray:
        obi = self.order_book_imbalance()
        cvd_norm = np.tanh(self.cvd / 1e6)
        ofs = self.order_flow_speed()
        vwap_dev = self.vwap_deviation()
        trend_5m = self.macro_trend(300)
        trend_15m = self.macro_trend(900)
        trend_1h = self.macro_trend(3600)
        trend_4h = self.macro_trend(14400)
        trend_1d = self.macro_trend(86400)

        obi_reversal = self._obi_reversal()
        pre_entry_sweep = self._pre_entry_sweep()
        fill_delay_norm = self._fill_delay_norm()

        base = np.array([
            obi, cvd_norm, ofs, vwap_dev,
            trend_5m, trend_15m, trend_1h, trend_4h, trend_1d,
            self.funding_rate,
            obi_reversal, pre_entry_sweep, fill_delay_norm,
        ], dtype=np.float32)
        return np.concatenate([base, self.liquidity_features()])

    def orderbook_sequence(self, length: int = 60) -> np.ndarray:
        obs = [p for p in self.points if p.side == "OB"]
        if not obs:
            return np.zeros((length, 2), dtype=np.float32)
        seq = []
        for p in obs[-length:]:
            total = p.bid_vol + p.ask_vol
            obi = (p.bid_vol - p.ask_vol) / total if total > 0 else 0.0
            seq.append([obi, p.bid_vol - p.ask_vol])
        while len(seq) < length:
            seq.insert(0, [0.0, 0.0])
        return np.array(seq[-length:], dtype=np.float32)

    def flow_sequence(self, length: int = 60) -> np.ndarray:
        trades = [p for p in self.points if p.size > 0]
        if not trades:
            return np.zeros((length, 3), dtype=np.float32)
        seq = []
        for p in trades[-length:]:
            signed = p.size if p.side.upper() in ("BUY", "B") else -p.size
            seq.append([signed, p.price, p.size])
        while len(seq) < length:
            seq.insert(0, [0.0, 0.0, 0.0])
        return np.array(seq[-length:], dtype=np.float32)


class FeatureStore:
    def __init__(self, window_sec: int = 300) -> None:
        self.window_sec = window_sec
        self._buffers: Dict[str, SymbolBuffer] = {}

    def get(self, symbol: str) -> SymbolBuffer:
        if symbol not in self._buffers:
            self._buffers[symbol] = SymbolBuffer(symbol=symbol, window_sec=self.window_sec)
        return self._buffers[symbol]

    def set_funding_rate(self, symbol: str, rate: float) -> None:
        self.get(symbol).funding_rate = rate

    def btc_returns(self, window_sec: int = 300) -> List[float]:
        buf = self._buffers.get("BTCUSDT")
        if not buf or len(buf.price_history) < 2:
            return []
        now = buf.price_history[-1][0]
        prices = [p for t, p in buf.price_history if t >= now - window_sec]
        if len(prices) < 2:
            return []
        returns = np.diff(prices) / np.array(prices[:-1])
        return returns.tolist()

    def correlation_with_btc(self, symbol: str, window_sec: int = 300) -> float:
        if symbol == "BTCUSDT":
            return 1.0
        btc_buf = self._buffers.get("BTCUSDT")
        sym_buf = self._buffers.get(symbol)
        if not btc_buf or not sym_buf:
            return 0.0
        now = time.time()
        btc_prices = [p for t, p in btc_buf.price_history if t >= now - window_sec]
        sym_prices = [p for t, p in sym_buf.price_history if t >= now - window_sec]
        n = min(len(btc_prices), len(sym_prices))
        if n < 10:
            return 0.0
        btc_r = np.diff(btc_prices[-n:]) / np.array(btc_prices[-n:-1])
        sym_r = np.diff(sym_prices[-n:]) / np.array(sym_prices[-n:-1])
        if len(btc_r) != len(sym_r) or len(btc_r) < 5:
            return 0.0
        if np.std(btc_r) < 1e-12 or np.std(sym_r) < 1e-12:
            return 0.0
        corr = np.corrcoef(btc_r, sym_r)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0
