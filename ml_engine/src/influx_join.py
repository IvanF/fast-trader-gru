"""Join trade outcomes with market microstructure features for training."""

from __future__ import annotations

import json
import math
from typing import Any, Optional

import numpy as np

from .influx_store import InfluxStore
from .models.nn_models import FLOW_DIM, MACRO_DIM, OB_DIM, SEQ_LEN, STATE_DIM

FEATURE_COLS = ["obi", "bid_vol", "ask_vol", "price", "size"]

MAX_PNL = 2.0
MIN_PNL = -2.0
MAX_PRICE = 1e9
MAX_SIZE = 1e8


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


def _direction_label(direction: str, pnl: float) -> int:
    """0=LONG, 1=SHORT, 2=HOLD — label reflects actual trade direction."""
    d = (direction or "").upper()
    if d == "LONG":
        return 0
    if d == "SHORT":
        return 1
    return 2


def _confidence_label(
    pnl: float,
    entry: float,
    direction: str,
    obi: float = 0.0,
    regime: str = "Choppy",
) -> float:
    """Confidence = how well-aligned the trade was with market microstructure.

    High confidence when: direction matches OBI sign AND trade was profitable.
    Low confidence when: direction contradicts OBI OR trade lost money.
    This teaches the model to output confidence based on market conditions, not PnL.
    """
    if entry <= 0:
        return 0.5

    dir_aligned = (
        (direction == "LONG" and obi > 0.02) or
        (direction == "SHORT" and obi < -0.02)
    )
    won = pnl >= 0

    if dir_aligned and won:
        base = 0.85
    elif dir_aligned and not won:
        base = 0.55
    elif not dir_aligned and won:
        base = 0.50
    else:
        base = 0.25

    regime_bonus = {
        "Trending": 0.10,
        "Breakout": 0.05,
        "Choppy": -0.05,
    }.get(regime, 0.0)

    return float(np.clip(base + regime_bonus, 0.1, 0.95))


def _rows_to_sequences(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ob_seq = np.zeros((SEQ_LEN, OB_DIM), dtype=np.float32)
    flow_seq = np.zeros((SEQ_LEN, FLOW_DIM), dtype=np.float32)
    macro = np.zeros(MACRO_DIM, dtype=np.float32)

    if not rows:
        return ob_seq, flow_seq, macro

    obi_vals = []
    flow_vals = []
    prices = []
    sizes = []
    bid_vols = []
    ask_vols = []
    levels_list = []

    for row in rows:
        obi = _safe_float(row.get("obi", 0))
        bid = max(_safe_float(row.get("bid_vol", 0)), 0.0)
        ask = max(_safe_float(row.get("ask_vol", 0)), 0.0)
        price = _safe_float(row.get("price", 0))
        size = _safe_float(row.get("size", 0))
        levels = _safe_float(row.get("levels", 0))
        obi_vals.append([obi, bid - ask])
        bid_vols.append(bid)
        ask_vols.append(ask)
        levels_list.append(levels)
        if size > 0 and price > 0:
            signed = size if str(row.get("side", "Buy")).upper() in ("BUY", "B") else -size
            flow_vals.append([signed, price, size])
        if price > 0:
            prices.append(price)
        if size > 0:
            sizes.append(size)

    if obi_vals:
        arr = np.array(obi_vals[-SEQ_LEN:], dtype=np.float32)
        ob_seq[-len(arr):] = arr
    if flow_vals:
        arr = np.array(flow_vals[-SEQ_LEN:], dtype=np.float32)
        flow_seq[-len(arr):] = arr

    if obi_vals:
        last = obi_vals[-1]
        total = abs(last[1]) + 1e-8
        macro[0] = last[0]
        macro[1] = np.tanh(last[1] / 1e6)
    if len(prices) >= 2:
        macro[4] = (prices[-1] - prices[0]) / max(prices[0], 1e-8)
    if len(prices) >= 10:
        macro[5] = (prices[-1] - prices[len(prices)//2]) / max(prices[len(prices)//2], 1e-8)
    if sizes and prices:
        vwap = sum(p * s for p, s in zip(prices, sizes)) / max(sum(sizes), 1e-8)
        macro[3] = (prices[-1] - vwap) / max(vwap, 1e-8)

    last_bid = bid_vols[-1] if bid_vols else 0.0
    last_ask = ask_vols[-1] if ask_vols else 0.0
    total_depth = last_bid + last_ask
    macro[7] = (last_bid - last_ask) / total_depth if total_depth > 0 else 0.0
    macro[8] = np.tanh(total_depth / 1e6)

    avg_bid = np.mean(bid_vols[-10:]) if bid_vols else 0.0
    avg_ask = np.mean(ask_vols[-10:]) if ask_vols else 0.0
    avg_total = avg_bid + avg_ask
    macro[9] = (avg_bid - avg_ask) / avg_total if avg_total > 0 else 0.0

    macro[10] = levels_list[-1] / 50.0 if levels_list else 0.0

    bid_vol_std = np.std(bid_vols[-20:]) if len(bid_vols) >= 2 else 0.0
    ask_vol_std = np.std(ask_vols[-20:]) if len(ask_vols) >= 2 else 0.0
    macro[11] = np.tanh(bid_vol_std / 1e4)
    macro[12] = np.tanh(ask_vol_std / 1e4)

    avg_fill = np.mean([s for s in sizes if s > 0]) if any(s > 0 for s in sizes) else 0.0
    macro[13] = avg_fill / (total_depth / 10.0) if total_depth > 0 else 0.0

    # Trap features (14-16)
    if len(obi_vals) >= 2:
        obi_arr = np.array([v[0] for v in obi_vals], dtype=np.float32)
        macro[14] = float(np.max(obi_arr) - np.min(obi_arr))  # obi_reversal
    else:
        macro[14] = 0.0

    pre_entry_sweep = 0.0
    if len(flow_vals) >= 5:
        recent_sizes = [abs(v[0]) for v in flow_vals[-5:]]
        avg_s = np.mean(recent_sizes) if recent_sizes else 0
        max_s = max(recent_sizes) if recent_sizes else 0
        if avg_s > 0 and max_s > avg_s * 3.0:
            pre_entry_sweep = 1.0
    macro[15] = pre_entry_sweep

    if len(flow_vals) >= 2:
        delays = [flow_vals[i][1] - flow_vals[i-1][1] for i in range(1, min(len(flow_vals), 60))]
        valid_delays = [d for d in delays if d > 0]
        if valid_delays:
            macro[16] = float(np.clip(np.mean(valid_delays) / 60.0, 0.0, 1.0))
        else:
            macro[16] = 0.5
    else:
        macro[16] = 0.5

    # Multi-timeframe trends (17-19)
    if len(prices) >= 2:
        macro[17] = (prices[-1] - prices[0]) / max(prices[0], 1e-8)  # trend_5m
    if len(prices) >= 10:
        q = len(prices) // 4
        macro[18] = (prices[-1] - prices[q]) / max(prices[q], 1e-8)  # trend_4h
    if len(prices) >= 20:
        macro[19] = (prices[-1] - prices[0]) / max(prices[0], 1e-8)  # trend_1d

    # Volume imbalance from trades (20)
    buy_vol = sum(abs(v[0]) for v in flow_vals if v[0] > 0)
    sell_vol = sum(abs(v[0]) for v in flow_vals if v[0] < 0)
    trade_total = buy_vol + sell_vol
    macro[20] = (buy_vol - sell_vol) / trade_total if trade_total > 0 else 0.0

    # Funding rate proxy = OBI change (21)
    if len(obi_vals) >= 2:
        obi_first = obi_vals[0][0]
        obi_last = obi_vals[-1][0]
        macro[21] = float(np.clip(obi_last - obi_first, -1.0, 1.0))

    return ob_seq, flow_seq, macro


def build_joined_dataset(
    store: InfluxStore,
    start: str,
    stop: str = "now()",
    symbol: Optional[str] = None,
    feature_window_sec: int = 300,
) -> dict[str, np.ndarray]:
    outcomes = store.query_trade_outcomes(start, stop, symbol)
    if not outcomes:
        return _empty_dataset()

    ob_list, flow_list, macro_list = [], [], []
    state_list, memory_list = [], []
    dir_list, conf_list, pnl_list = [], [], []
    ts_list, sym_list = [], []

    raw_trades = []

    for outcome in outcomes:
        sym = str(outcome.get("symbol", ""))
        t = outcome["_time"]
        pnl = _safe_float(outcome.get("net_pnl", 0))
        if pnl > MAX_PNL or pnl < MIN_PNL:
            continue
        direction = str(outcome.get("direction", "HOLD"))
        entry = _safe_float(outcome.get("entry_price", 0))
        if entry <= 0:
            continue

        feature_rows = store.query_market_features_window(sym, t, feature_window_sec)
        ob_seq, flow_seq, macro = _rows_to_sequences(feature_rows)

        if ob_seq.sum() == 0 and flow_seq.sum() == 0:
            continue

        state_json = outcome.get("state_vector_json", "[]")
        try:
            state_vec = np.array(json.loads(state_json), dtype=np.float32)
        except (json.JSONDecodeError, TypeError):
            state_vec = np.zeros(STATE_DIM, dtype=np.float32)
        if not np.all(np.isfinite(state_vec)):
            state_vec = np.nan_to_num(state_vec, nan=0.0, posinf=0.0, neginf=0.0)
        if state_vec.size < STATE_DIM:
            state_vec = np.pad(state_vec, (0, STATE_DIM - state_vec.size))
        state_vec = state_vec[:STATE_DIM]

        regime = outcome.get("regime", "Choppy")
        obi = macro[0] if len(macro) > 0 else 0.0

        raw_trades.append({
            "ob_seq": ob_seq,
            "flow_seq": flow_seq,
            "macro": macro,
            "state_vec": state_vec,
            "direction": direction,
            "regime": regime,
            "pnl": pnl,
            "entry": entry,
            "obi": obi,
            "ts": t.timestamp(),
            "sym": sym,
        })

    if not raw_trades:
        return _empty_dataset()

    n = len(raw_trades)

    for i, trade in enumerate(raw_trades):
        neighbor_mask = np.zeros(n, dtype=bool)
        neighbor_mask[max(0, i - 5):min(n, i + 6)] = True
        neighbor_mask[i] = True

        neighbor_indices = np.where(neighbor_mask)[0]
        if len(neighbor_indices) == 0:
            neighbor_indices = np.array([i])

        weighted_pnl = 0.0
        weighted_wins = 0.0
        total_weight = 0.0
        for j in neighbor_indices:
            w = 1.0 / (1.0 + abs(i - j))
            weighted_pnl += raw_trades[j]["pnl"] * w
            weighted_wins += (1.0 if raw_trades[j]["pnl"] >= 0 else 0.0) * w
            total_weight += w

        win_rate = weighted_wins / max(total_weight, 1e-8)
        avg_pnl = weighted_pnl / max(total_weight, 1e-8)

        memory_vec = np.array([
            win_rate,
            avg_pnl,
            float(np.tanh(avg_pnl / 100)),
            total_weight,
            float(len(neighbor_indices)),
            1.0 if trade["regime"] == "Trending" else 0.0,
            1.0 if trade["regime"] == "Breakout" else 0.0,
            1.0 if trade["regime"] == "Choppy" else 0.0,
        ], dtype=np.float32)

        ob_list.append(trade["ob_seq"])
        flow_list.append(trade["flow_seq"])
        macro_list.append(trade["macro"])
        state_list.append(trade["state_vec"])
        memory_list.append(memory_vec)
        dir_list.append(_direction_label(trade["direction"], trade["pnl"]))
        conf_list.append(_confidence_label(
            trade["pnl"], trade["entry"], trade["direction"],
            trade["obi"], trade["regime"],
        ))
        pnl_list.append(trade["pnl"])
        ts_list.append(trade["ts"])
        sym_list.append(trade["sym"])

    if not ob_list:
        return _empty_dataset()

    result = {
        "ob_seq": np.nan_to_num(np.stack(ob_list), nan=0.0, posinf=0.0, neginf=0.0),
        "flow_seq": np.nan_to_num(np.stack(flow_list), nan=0.0, posinf=0.0, neginf=0.0),
        "macro": np.nan_to_num(np.stack(macro_list), nan=0.0, posinf=0.0, neginf=0.0),
        "state_vector": np.nan_to_num(np.stack(state_list), nan=0.0, posinf=0.0, neginf=0.0),
        "memory": np.nan_to_num(np.stack(memory_list), nan=0.0, posinf=0.0, neginf=0.0),
        "direction": np.array(dir_list, dtype=np.int64),
        "confidence": np.array(conf_list, dtype=np.float32),
        "pnl": np.array(pnl_list, dtype=np.float32),
        "timestamps": np.array(ts_list, dtype=np.float64),
        "symbols": np.array(sym_list),
    }
    return result


def _empty_dataset() -> dict[str, np.ndarray]:
    return {
        "ob_seq": np.zeros((0, SEQ_LEN, OB_DIM), dtype=np.float32),
        "flow_seq": np.zeros((0, SEQ_LEN, FLOW_DIM), dtype=np.float32),
        "macro": np.zeros((0, MACRO_DIM), dtype=np.float32),
        "state_vector": np.zeros((0, STATE_DIM), dtype=np.float32),
        "memory": np.zeros((0, 8), dtype=np.float32),
        "direction": np.zeros(0, dtype=np.int64),
        "confidence": np.zeros(0, dtype=np.float32),
        "pnl": np.zeros(0, dtype=np.float32),
        "timestamps": np.zeros(0, dtype=np.float64),
        "symbols": np.array([]),
    }
