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
        return (prices[-1] - prices[0]) / prices[0]

    def feature_vector(self) -> np.ndarray:
        obi = self.order_book_imbalance()
        cvd_norm = np.tanh(self.cvd / 1e6)
        ofs = self.order_flow_speed()
        vwap_dev = self.vwap_deviation()
        trend_15m = self.macro_trend(900)
        trend_1h = self.macro_trend(3600)
        return np.array([
            obi, cvd_norm, ofs, vwap_dev,
            trend_15m, trend_1h, self.funding_rate,
        ], dtype=np.float32)

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
