#!/usr/bin/env python3
"""Backtest ExitOptimizer: compare PnL with vs without optimal TP/SL prediction.

Loads FAISS metadata (MAE/MFE/pnl) and simulates trades with:
1. Fixed SL/TP (current system)
2. ExitOptimizer-predicted SL/TP

Shows whether ExitOptimizer would have improved results.

Usage:
    python3 scripts/backtest_exit_optimizer.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.nn_models import ExitOptimizer, EXIT_OPT_IN_DIM


def load_faiss_metadata(faiss_path: str) -> list[dict]:
    meta_file = f"{faiss_path}.meta.json"
    if not os.path.exists(meta_file):
        return []
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_faiss_vectors(faiss_path: str) -> np.ndarray | None:
    idx_file = f"{faiss_path}.faiss"
    if not os.path.exists(idx_file):
        return None
    import faiss
    index = faiss.read_index(idx_file)
    n, d = index.ntotal, index.d
    vecs = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        vecs[i] = index.reconstruct(i)
    return vecs


def simulate_trade(
    entry_price: float,
    direction: str,
    sl_distance_pct: float,
    tp_distance_pct: float,
    mae_pct: float,
    mfe_pct: float,
    fee_pct: float = 0.0011,
) -> dict:
    if direction == "LONG":
        sl_price = entry_price * (1 - sl_distance_pct)
        tp_price = entry_price * (1 + tp_distance_pct)
    else:
        sl_price = entry_price * (1 + sl_distance_pct)
        tp_price = entry_price * (1 - tp_distance_pct)

    hit_sl = False
    hit_tp = False

    if direction == "LONG":
        if mae_pct >= sl_distance_pct:
            hit_sl = True
        if mfe_pct >= tp_distance_pct:
            hit_tp = True
    else:
        if mae_pct >= sl_distance_pct:
            hit_sl = True
        if mfe_pct >= tp_distance_pct:
            hit_tp = True

    if hit_sl and hit_tp:
        if mae_pct < mfe_pct:
            hit_tp = True
            hit_sl = False
        else:
            hit_sl = True
            hit_tp = False

    if hit_tp:
        pnl_pct = tp_distance_pct - fee_pct
    elif hit_sl:
        pnl_pct = -sl_distance_pct - fee_pct
    else:
        if direction == "LONG":
            exit_pct = mfe_pct * 0.5 if mfe_pct > 0 else -mae_pct * 0.5
        else:
            exit_pct = mfe_pct * 0.5 if mfe_pct > 0 else -mae_pct * 0.5
        pnl_pct = exit_pct - fee_pct

    return {
        "pnl_pct": pnl_pct,
        "hit_sl": hit_sl,
        "hit_tp": hit_tp,
        "sl_distance": sl_distance_pct,
        "tp_distance": tp_distance_pct,
    }


def backtest_exit_optimizer(
    faiss_path: str,
    model_path: str,
    device: str = "cuda",
    fixed_sl: float = 0.006,
    fixed_tp: float = 0.015,
) -> dict:
    print(f"loading FAISS metadata from {faiss_path}")
    metadata = load_faiss_metadata(faiss_path)
    vectors = load_faiss_vectors(faiss_path)

    if vectors is None or not metadata:
        return {"error": "no data"}

    print(f"loaded {len(metadata)} entries")

    model = ExitOptimizer(in_dim=EXIT_OPT_IN_DIM)
    model_path_pt = os.path.join(os.path.dirname(model_path), "exit_optimizer.pt")
    if os.path.exists(model_path_pt):
        state = torch.load(model_path_pt, map_location=device, weights_only=False)
        model.load_state_dict(state)
        print(f"loaded ExitOptimizer from {model_path_pt}")
    elif os.path.exists(model_path):
        import onnxruntime as ort
        print(f"found ONNX model at {model_path}, using random PT weights for backtest")
    else:
        print(f"no model found, using random weights")
    model.eval().to(device)

    fixed_pnls = []
    optimized_pnls = []
    sl_distances_fixed = []
    tp_distances_fixed = []
    sl_distances_opt = []
    tp_distances_opt = []

    for i, meta in enumerate(metadata):
        if i >= len(vectors):
            break

        mae = abs(meta.get("mae_pct", 0.0))
        mfe = abs(meta.get("mfe_pct", 0.0))
        pnl = meta.get("pnl", 0.0)
        direction = meta.get("direction", "HOLD")
        regime = meta.get("regime", "Choppy")

        if mae < 0.0001 and mfe < 0.0001:
            continue

        if direction not in ("LONG", "SHORT"):
            continue

        vec = vectors[i]
        dir_onehot = [1.0, 0.0] if direction == "LONG" else [0.0, 1.0]
        regime_onehot = [
            1.0 if regime == "Choppy" else 0.0,
            1.0 if regime == "Trending" else 0.0,
            1.0 if regime == "Breakout" else 0.0,
        ]

        memory_features = np.array([
            1.0 if pnl > 0 else 0.0, pnl, float(np.tanh(pnl / 100)),
            5.0, 10.0,
            1.0 if regime == "Trending" else 0.0,
            1.0 if regime == "Breakout" else 0.0,
            1.0 if regime == "Choppy" else 0.0,
        ], dtype=np.float32)

        with torch.no_grad():
            x = np.concatenate([vec, memory_features, np.array(
                dir_onehot + regime_onehot + [0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                dtype=np.float32,
            )]).astype(np.float32).reshape(1, -1)
            x_tensor = torch.from_numpy(x).to(device)
            sl_pred, tp_pred, score = model(x_tensor)
            opt_sl = float(sl_pred[0])
            opt_tp = float(tp_pred[0])

        entry_price = 1.0
        fixed_result = simulate_trade(entry_price, direction, fixed_sl, fixed_tp, mae, mfe)
        optimized_result = simulate_trade(entry_price, direction,
                                          min(max(opt_sl, 0.002), 0.015),
                                          min(max(opt_tp, 0.003), 0.025),
                                          mae, mfe)

        fixed_pnls.append(fixed_result["pnl_pct"])
        optimized_pnls.append(optimized_result["pnl_pct"])
        sl_distances_fixed.append(fixed_result["sl_distance"])
        tp_distances_fixed.append(fixed_result["tp_distance"])
        sl_distances_opt.append(opt_sl)
        tp_distances_opt.append(opt_tp)

    if not fixed_pnls:
        return {"error": "no valid trades"}

    fixed_pnls = np.array(fixed_pnls)
    optimized_pnls = np.array(optimized_pnls)

    fixed_wr = (fixed_pnls > 0).sum() / len(fixed_pnls)
    opt_wr = (optimized_pnls > 0).sum() / len(optimized_pnls)

    fixed_pf = abs(fixed_pnls[fixed_pnls > 0].sum() / fixed_pnls[fixed_pnls < 0].sum()) if (fixed_pnls < 0).any() else 0
    opt_pf = abs(optimized_pnls[optimized_pnls > 0].sum() / optimized_pnls[optimized_pnls < 0].sum()) if (optimized_pnls < 0).any() else 0

    return {
        "trades": len(fixed_pnls),
        "fixed": {
            "sl_pct": round(float(np.mean(sl_distances_fixed)), 6),
            "tp_pct": round(float(np.mean(tp_distances_fixed)), 6),
            "total_pnl_pct": round(float(fixed_pnls.sum()), 6),
            "avg_pnl_pct": round(float(fixed_pnls.mean()), 6),
            "win_rate": round(float(fixed_wr), 4),
            "profit_factor": round(float(fixed_pf), 4),
        },
        "optimized": {
            "sl_pct": round(float(np.mean(sl_distances_opt)), 6),
            "tp_pct": round(float(np.mean(tp_distances_opt)), 6),
            "total_pnl_pct": round(float(optimized_pnls.sum()), 6),
            "avg_pnl_pct": round(float(optimized_pnls.mean()), 6),
            "win_rate": round(float(opt_wr), 4),
            "profit_factor": round(float(opt_pf), 4),
        },
        "improvement": {
            "pnl_delta_pct": round(float((optimized_pnls.sum() - fixed_pnls.sum()) / abs(fixed_pnls.sum()) * 100) if fixed_pnls.sum() != 0 else 0, 2),
            "wr_delta": round(float(opt_wr - fixed_wr), 4),
            "pf_delta": round(float(opt_pf - fixed_pf), 4),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest ExitOptimizer")
    parser.add_argument("--faiss-path", default=os.getenv("FAISS_PATH", "/app/data/faiss_index"))
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "/app/models/active/current"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed-sl", type=float, default=0.006)
    parser.add_argument("--fixed-tp", type=float, default=0.015)
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "exit_optimizer.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, "exit_optimizer.onnx")

    result = backtest_exit_optimizer(
        faiss_path=args.faiss_path,
        model_path=model_path,
        device=args.device,
        fixed_sl=args.fixed_sl,
        fixed_tp=args.fixed_tp,
    )
    print(json.dumps(result, indent=2))
