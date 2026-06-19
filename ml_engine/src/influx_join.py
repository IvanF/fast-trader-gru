"""Join trade outcomes with market microstructure features for training."""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from .influx_store import InfluxStore
from .models.nn_models import FLOW_DIM, MACRO_DIM, OB_DIM, SEQ_LEN, STATE_DIM

FEATURE_COLS = ["obi", "bid_vol", "ask_vol", "price", "size"]


def _direction_label(direction: str, pnl: float) -> int:
    """0=LONG, 1=SHORT, 2=HOLD — label reflects actual trade direction."""
    d = (direction or "").upper()
    if d == "LONG":
        return 0
    if d == "SHORT":
        return 1
    return 2


def _confidence_label(pnl: float, entry: float) -> float:
    if entry <= 0:
        return 0.5
    return float(np.clip(abs(pnl / entry), 0.0, 1.0))


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

    for row in rows:
        obi = float(row.get("obi", 0) or 0)
        bid = float(row.get("bid_vol", 0) or 0)
        ask = float(row.get("ask_vol", 0) or 0)
        price = float(row.get("price", 0) or 0)
        size = float(row.get("size", 0) or 0)
        obi_vals.append([obi, bid - ask])
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
        macro[5] = (prices[-1] - prices[0]) / max(prices[0], 1e-8)
    if sizes and prices:
        vwap = sum(p * s for p, s in zip(prices, sizes)) / max(sum(sizes), 1e-8)
        macro[3] = (prices[-1] - vwap) / max(vwap, 1e-8)

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

    for outcome in outcomes:
        sym = str(outcome.get("symbol", ""))
        t = outcome["_time"]
        pnl = float(outcome.get("net_pnl", 0) or 0)
        direction = str(outcome.get("direction", "HOLD"))
        entry = float(outcome.get("entry_price", 0) or 0)

        feature_rows = store.query_market_features_window(sym, t, feature_window_sec)
        ob_seq, flow_seq, macro = _rows_to_sequences(feature_rows)

        state_json = outcome.get("state_vector_json", "[]")
        try:
            state_vec = np.array(json.loads(state_json), dtype=np.float32)
        except (json.JSONDecodeError, TypeError):
            state_vec = np.zeros(STATE_DIM, dtype=np.float32)
        if state_vec.size < STATE_DIM:
            state_vec = np.pad(state_vec, (0, STATE_DIM - state_vec.size))
        state_vec = state_vec[:STATE_DIM]

        memory_vec = np.array([
            1.0 if pnl >= 0 else 0.0,
            pnl,
            np.tanh(pnl / 100),
            1.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)

        ob_list.append(ob_seq)
        flow_list.append(flow_seq)
        macro_list.append(macro)
        state_list.append(state_vec)
        memory_list.append(memory_vec)
        dir_list.append(_direction_label(direction, pnl))
        conf_list.append(_confidence_label(pnl, entry))
        pnl_list.append(pnl)
        ts_list.append(t.timestamp())
        sym_list.append(sym)

    return {
        "ob_seq": np.stack(ob_list),
        "flow_seq": np.stack(flow_list),
        "macro": np.stack(macro_list),
        "state_vector": np.stack(state_list),
        "memory": np.stack(memory_list),
        "direction": np.array(dir_list, dtype=np.int64),
        "confidence": np.array(conf_list, dtype=np.float32),
        "pnl": np.array(pnl_list, dtype=np.float32),
        "timestamps": np.array(ts_list, dtype=np.float64),
        "symbols": np.array(sym_list),
    }


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
