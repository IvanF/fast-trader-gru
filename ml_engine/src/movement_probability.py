"""
P(Y = y | X_t) via softmax regression on microstructure state vector.

State X_t = [x1 OBI, x2 vol delta, x3 trade velocity, x4 OI delta, x5 liq imbalance].
Classes y ∈ {-1, 0, 1} → Short, Neutral, Long.

Bayesian context: funding-rate prior adjusts logits before softmax.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .features import SymbolBuffer, _level_size

MOVEMENT_MEASUREMENT = "movement_probability"
CLASS_LABELS = ("short", "neutral", "long")
CLASS_Y = (-1, 0, 1)

# W_y · X_t + b_y  (rows: short, neutral, long)
_DEFAULT_WEIGHTS = np.array([
    [-1.40, -1.20, 0.15, -0.50, -0.90],  # Short
    [0.00, 0.00, -0.60, 0.00, 0.00],     # Neutral — penalise high trade velocity
    [1.40, 1.20, 0.15, 0.50, 0.90],      # Long
], dtype=np.float64)
_DEFAULT_BIAS = np.array([0.0, 0.35, 0.0], dtype=np.float64)


@dataclass
class MovementProbability:
    p_short: float
    p_neutral: float
    p_long: float
    x1_obi: float
    x2_vol_delta: float
    x3_trade_vel: float
    x4_oi_delta: float
    x5_liq_imb: float
    logit_short: float = 0.0
    logit_neutral: float = 0.0
    logit_long: float = 0.0

    def as_fields(self) -> dict[str, float]:
        return {
            "p_short": self.p_short,
            "p_neutral": self.p_neutral,
            "p_long": self.p_long,
            "x1_obi": self.x1_obi,
            "x2_vol_delta": self.x2_vol_delta,
            "x3_trade_vel": self.x3_trade_vel,
            "x4_oi_delta": self.x4_oi_delta,
            "x5_liq_imb": self.x5_liq_imb,
            "logit_short": self.logit_short,
            "logit_neutral": self.logit_neutral,
            "logit_long": self.logit_long,
        }


@dataclass
class MovementState:
    prev_funding: float = 0.0
    prev_vol_pressure: float = 0.0
    initialized: bool = False


class MovementStateStore:
    def __init__(self) -> None:
        self._states: Dict[str, MovementState] = {}

    def get(self, symbol: str) -> MovementState:
        if symbol not in self._states:
            self._states[symbol] = MovementState()
        return self._states[symbol]


def _obi(buf: SymbolBuffer, depth: int = 10) -> float:
    bid_vol = sum(_level_size(b) for b in buf.latest_bids[:depth])
    ask_vol = sum(_level_size(a) for a in buf.latest_asks[:depth])
    total = bid_vol + ask_vol
    if total <= 0:
        return buf.order_book_imbalance()
    return (bid_vol - ask_vol) / total


def _vol_delta(buf: SymbolBuffer, window_sec: float = 60.0) -> float:
    if not buf.points:
        return 0.0
    now = buf.points[-1].ts
    trades = [p for p in buf.points if p.size > 0 and p.ts >= now - window_sec]
    buy = sum(p.size for p in trades if p.side.upper() in ("BUY", "B"))
    sell = sum(p.size for p in trades if p.side.upper() in ("SELL", "S"))
    total = buy + sell
    if total <= 0:
        return 0.0
    return (buy - sell) / total


def _trade_velocity(buf: SymbolBuffer, window_sec: float = 60.0) -> float:
    if not buf.points:
        return 0.0
    now = buf.points[-1].ts
    n = sum(1 for p in buf.points if p.size > 0 and p.ts >= now - window_sec)
    raw = math.log(n + 1)
    return raw / math.log(101.0)


def _oi_delta_proxy(buf: SymbolBuffer, state: MovementState) -> float:
    """ΔOI proxy: funding-rate change + signed-volume pressure delta."""
    funding = buf.funding_rate
    vol_p = _vol_delta(buf)
    if not state.initialized:
        state.prev_funding = funding
        state.prev_vol_pressure = vol_p
        state.initialized = True
        return 0.0
    fund_delta = funding - state.prev_funding
    vol_delta_change = vol_p - state.prev_vol_pressure
    state.prev_funding = funding
    state.prev_vol_pressure = vol_p
    combined = fund_delta * 100.0 + vol_delta_change * 0.5
    return float(np.clip(combined, -1.0, 1.0))


def _liquidation_imbalance(buf: SymbolBuffer, window_sec: float = 30.0, spike: float = 2.0) -> float:
    """
    x5 = (L_short - L_long) / (L_short + L_long + ε).
    Short liquidation → aggressive buy; long liquidation → aggressive sell.
    """
    if not buf.points:
        return 0.0
    now = buf.points[-1].ts
    trades = [p for p in buf.points if p.size > 0 and p.ts >= now - window_sec]
    if len(trades) < 2:
        return 0.0
    sizes = [t.size for t in trades]
    median = float(np.median(sizes))
    if median <= 0:
        return 0.0
    l_short = 0.0  # shorts liquidated → buy
    l_long = 0.0   # longs liquidated → sell
    for t in trades:
        if t.size < median * spike:
            continue
        if t.side.upper() in ("BUY", "B"):
            l_short += t.size
        else:
            l_long += t.size
    eps = 1e-8
    return (l_short - l_long) / (l_short + l_long + eps)


def build_state_vector(buf: SymbolBuffer, state: MovementState) -> np.ndarray:
    return np.array([
        _obi(buf),
        _vol_delta(buf),
        _trade_velocity(buf),
        _oi_delta_proxy(buf, state),
        _liquidation_imbalance(buf),
    ], dtype=np.float64)


def _funding_log_prior(funding_rate: float, scale: float = 50.0) -> np.ndarray:
    """Bayesian prior from funding: crowded longs → higher P(short)."""
    prior = np.zeros(3, dtype=np.float64)
    if funding_rate > 0.0003:
        prior[0] += scale * funding_rate
        prior[2] -= scale * funding_rate * 0.5
    elif funding_rate < -0.0003:
        prior[2] += scale * abs(funding_rate)
        prior[0] -= scale * abs(funding_rate) * 0.5
    return prior


def softmax_movement_probability(
    buf: SymbolBuffer,
    state: Optional[MovementState] = None,
    weights: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    funding_prior_scale: float = 50.0,
) -> MovementProbability:
    if state is None:
        state = MovementState()
    W = weights if weights is not None else _DEFAULT_WEIGHTS
    b = bias if bias is not None else _DEFAULT_BIAS

    x = build_state_vector(buf, state)
    logits = W @ x + b + _funding_log_prior(buf.funding_rate, funding_prior_scale)

    logits = logits - np.max(logits)
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum()

    return MovementProbability(
        p_short=float(probs[0]),
        p_neutral=float(probs[1]),
        p_long=float(probs[2]),
        x1_obi=float(x[0]),
        x2_vol_delta=float(x[1]),
        x3_trade_vel=float(x[2]),
        x4_oi_delta=float(x[3]),
        x5_liq_imb=float(x[4]),
        logit_short=float(logits[0]),
        logit_neutral=float(logits[1]),
        logit_long=float(logits[2]),
    )


def load_weights_from_env() -> tuple[np.ndarray, np.ndarray]:
    """Optional override: MOVEMENT_PROB_WEIGHTS=JSON 3x5, MOVEMENT_PROB_BIAS=JSON len3."""
    import json

    w_raw = os.getenv("MOVEMENT_PROB_WEIGHTS", "")
    b_raw = os.getenv("MOVEMENT_PROB_BIAS", "")
    W, b = _DEFAULT_WEIGHTS.copy(), _DEFAULT_BIAS.copy()
    if w_raw:
        parsed = np.array(json.loads(w_raw), dtype=np.float64)
        if parsed.shape == (3, 5):
            W = parsed
    if b_raw:
        parsed_b = np.array(json.loads(b_raw), dtype=np.float64)
        if parsed_b.shape == (3,):
            b = parsed_b
    return W, b
