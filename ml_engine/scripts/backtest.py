#!/usr/bin/env python3
"""
Backtest ONNX models against historical trade outcomes from InfluxDB.

For each closed trade, reconstructs the state vector from market features,
runs inference through the ONNX pipeline, and compares predicted direction
and confidence with the actual outcome.

Usage (inside ml_engine container):
  python scripts/backtest.py --symbol BTCUSDT --hours 48

  # Or from pre-exported .npz:
  python scripts/backtest.py --dataset /app/data/training/BTCUSDT_joined_20260625.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import HotSwapONNXInference
from src.influx_join import build_joined_dataset
from src.influx_store import InfluxStore
from src.memory import ExperienceEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest ONNX models on historical data")
    p.add_argument("--symbol", default="", help="Symbol to backtest (e.g. BTCUSDT)")
    p.add_argument("--hours", type=int, default=48, help="Lookback hours for InfluxDB query")
    p.add_argument("--dataset", default="", help="Pre-exported .npz file (overrides InfluxDB query)")
    p.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "/app/models"))
    p.add_argument("--state-dim", type=int, default=int(os.getenv("STATE_DIM", "128")))
    p.add_argument("--long-threshold", type=float, default=float(os.getenv("LONG_CONFIDENCE_THRESHOLD", "0.45")))
    p.add_argument("--short-threshold", type=float, default=float(os.getenv("CONFIDENCE_THRESHOLD", "0.30")))
    p.add_argument("--output", default="", help="JSON output path (default: stdout)")
    return p.parse_args()


def load_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.dataset:
        raw = np.load(args.dataset, allow_pickle=True)
        return {k: raw[k] for k in raw.files}

    token = os.getenv("INFLUX_TOKEN", "")
    if not token:
        print("INFLUX_TOKEN required for InfluxDB query", file=sys.stderr)
        sys.exit(1)

    store = InfluxStore(
        os.getenv("INFLUX_URL", "http://influxdb:8086"),
        token,
        os.getenv("INFLUX_ORG", "fasttrader"),
        os.getenv("INFLUX_BUCKET_RAW", "market_raw"),
        os.getenv("INFLUX_BUCKET_FEATURES", "market_features"),
    )
    start = f"-{args.hours}h"
    symbols = [s.strip() for s in args.symbol.split(",") if s.strip()] if args.symbol else [""]
    parts = []
    for sym in symbols:
        print(f"loading {sym or 'all'} ({start})...")
        parts.append(build_joined_dataset(store, start, symbol=sym or None))
    store.close()

    if len(parts) == 1:
        return parts[0]
    keys = parts[0].keys()
    out = {}
    for k in keys:
        if k == "symbols":
            out[k] = np.concatenate([p[k] for p in parts])
        else:
            out[k] = np.concatenate([p[k] for p in parts], axis=0)
    return out


def run_backtest(
    data: dict[str, np.ndarray],
    inference: HotSwapONNXInference,
    state_dim: int,
    long_threshold: float,
    short_threshold: float,
) -> dict:
    n = len(data["direction"])
    if n == 0:
        return {"error": "empty dataset", "samples": 0}

    ob_seq = data["ob_seq"]
    flow_seq = data["flow_seq"]
    macro = data["macro"]
    state_vecs = data["state_vector"]
    memory_vecs = data["memory"]
    actual_dir = data["direction"]
    actual_conf = data["confidence"]
    actual_pnl = data["pnl"]

    pred_dirs = []
    pred_confs = []
    pred_vol_mults = []
    signals = []
    no_signal_count = 0

    for i in range(n):
        v_state = state_vecs[i]
        v_memory = memory_vecs[i]

        direction, confidence, vol_mult = inference.decide(v_state, v_memory)

        if direction == "HOLD":
            no_signal_count += 1
            pred_dirs.append(2)
            pred_confs.append(confidence)
            pred_vol_mults.append(vol_mult)
            signals.append(None)
            continue

        if direction == "SHORT" and confidence < short_threshold:
            no_signal_count += 1
            pred_dirs.append(2)
            pred_confs.append(confidence)
            pred_vol_mults.append(vol_mult)
            signals.append(None)
            continue

        if direction == "LONG" and confidence < long_threshold:
            no_signal_count += 1
            pred_dirs.append(2)
            pred_confs.append(confidence)
            pred_vol_mults.append(vol_mult)
            signals.append(None)
            continue

        dir_idx = 0 if direction == "LONG" else 1
        pred_dirs.append(dir_idx)
        pred_confs.append(confidence)
        pred_vol_mults.append(vol_mult)
        signals.append({
            "direction": direction,
            "confidence": confidence,
            "actual_direction": ["LONG", "SHORT", "HOLD"][int(actual_dir[i])],
            "actual_pnl": float(actual_pnl[i]),
        })

    pred_dirs = np.array(pred_dirs)
    pred_confs = np.array(pred_confs)

    acted_mask = pred_dirs != 2
    n_acted = int(acted_mask.sum())

    if n_acted == 0:
        return {
            "samples": n,
            "acted": 0,
            "no_signal": no_signal_count,
            "error": "model produced no actionable signals",
        }

    acted_pred = pred_dirs[acted_mask]
    acted_actual = actual_dir[acted_mask]
    acted_pnl = actual_pnl[acted_mask]

    correct = (acted_pred == acted_actual).sum()
    accuracy = correct / n_acted

    long_mask = acted_pred == 0
    short_mask = acted_pred == 1

    long_correct = ((acted_pred == 0) & (acted_actual == 0)).sum() if long_mask.sum() > 0 else 0
    short_correct = ((acted_pred == 1) & (acted_actual == 1)).sum() if short_mask.sum() > 0 else 0

    long_accuracy = long_correct / max(long_mask.sum(), 1)
    short_accuracy = short_correct / max(short_mask.sum(), 1)

    total_pnl = acted_pnl.sum()
    mean_pnl = acted_pnl.mean()
    win_mask = acted_pnl > 0
    win_rate = win_mask.sum() / n_acted

    long_pnl = acted_pnl[long_mask].sum() if long_mask.sum() > 0 else 0.0
    short_pnl = acted_pnl[short_mask].sum() if short_mask.sum() > 0 else 0.0

    winning_trades = acted_pnl[win_mask]
    losing_trades = acted_pnl[~win_mask]
    avg_win = float(winning_trades.mean()) if len(winning_trades) > 0 else 0.0
    avg_loss = float(losing_trades.mean()) if len(losing_trades) > 0 else 0.0
    profit_factor = abs(float(winning_trades.sum()) / float(losing_trades.sum())) if len(losing_trades) > 0 and losing_trades.sum() < 0 else 0.0

    pnl_std = float(acted_pnl.std())
    sharpe = mean_pnl / pnl_std if pnl_std > 0 else 0.0

    max_dd = 0.0
    cum = 0.0
    peak = 0.0
    for p in acted_pnl:
        cum += float(p)
        peak = max(peak, cum)
        dd = peak - cum
        max_dd = max(max_dd, dd)

    def _r(v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return round(v, 6)

    result = {
        "samples": n,
        "acted": n_acted,
        "no_signal": no_signal_count,
        "model_version": inference.version,
        "thresholds": {
            "long": long_threshold,
            "short": short_threshold,
        },
        "accuracy": round(float(accuracy), 4),
        "direction_accuracy": {
            "long": round(float(long_accuracy), 4),
            "long_count": int(long_mask.sum()),
            "short": round(float(short_accuracy), 4),
            "short_count": int(short_mask.sum()),
        },
        "pnl": {
            "total": _r(float(total_pnl)),
            "mean": _r(float(mean_pnl)),
            "std": _r(pnl_std),
            "long_total": _r(float(long_pnl)),
            "short_total": _r(float(short_pnl)),
        },
        "risk": {
            "win_rate": round(float(win_rate), 4),
            "avg_win": _r(float(avg_win)),
            "avg_loss": _r(float(avg_loss)),
            "profit_factor": _r(float(profit_factor)),
            "sharpe_ratio": _r(float(sharpe)),
            "max_drawdown": _r(float(max_dd)),
        },
    }
    return result


def main() -> int:
    args = parse_args()
    print(f"backtest: model_dir={args.model_dir} state_dim={args.state_dim}")

    data = load_dataset(args)
    n = len(data["direction"])
    print(f"loaded {n} samples")

    if n == 0:
        print("no data to backtest")
        return 1

    dirs = data["direction"]
    print(f"  LONG={int((dirs==0).sum())} SHORT={int((dirs==1).sum())} HOLD={int((dirs==2).sum())}")
    print(f"  mean PnL={data['pnl'].mean():.4f}")

    inference = HotSwapONNXInference(args.model_dir, args.state_dim)
    print(f"loaded model version={inference.version}")

    start = time.time()
    result = run_backtest(data, inference, args.state_dim, args.long_threshold, args.short_threshold)
    elapsed = time.time() - start
    result["duration_sec"] = round(elapsed, 2)

    output = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nresults written to {args.output}")
    else:
        print(f"\n{output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
