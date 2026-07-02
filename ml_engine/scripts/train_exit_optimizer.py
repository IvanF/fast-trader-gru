#!/usr/bin/env python3
"""Train ExitOptimizer neural network for optimal TP/SL prediction.

Loads historical trade data from FAISS metadata (MAE/MFE) and trains
a lightweight MLP to predict optimal SL/TP distances before entry.

Usage:
    python3 scripts/train_exit_optimizer.py --epochs 20 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

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


def build_exit_dataset(
    vectors: np.ndarray,
    metadata: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    sl_targets = []
    tp_targets = []

    for i, meta in enumerate(metadata):
        if i >= len(vectors):
            break

        mae = abs(meta.get("mae_pct", 0.0))
        mfe = abs(meta.get("mfe_pct", 0.0))

        if mae < 0.0001 and mfe < 0.0001:
            continue

        vec = vectors[i]

        direction = meta.get("direction", "HOLD")
        dir_onehot = [1.0, 0.0] if direction == "LONG" else [0.0, 1.0] if direction == "SHORT" else [0.0, 0.0]

        regime = meta.get("regime", "Choppy")
        regime_onehot = [
            1.0 if regime == "Choppy" else 0.0,
            1.0 if regime == "Trending" else 0.0,
            1.0 if regime == "Breakout" else 0.0,
        ]

        pnl = meta.get("pnl", 0.0)
        trade_score = 1.0 if pnl > 0 else 0.0

        opt_sl = mae * 1.1
        opt_tp = mfe * 0.9 if mfe > 0 else mae * 2.0

        won = 1.0 if pnl > 0 else 0.0
        avg_pnl = pnl
        memory_features = np.array([
            won, avg_pnl, float(np.tanh(avg_pnl / 100)), 5.0, 10.0,
            1.0 if regime == "Trending" else 0.0,
            1.0 if regime == "Breakout" else 0.0,
            1.0 if regime == "Choppy" else 0.0,
        ], dtype=np.float32)

        extras = np.array(dir_onehot + regime_onehot + [0.5, 1.0, 0.0, 0.0, 0.0, opt_sl, opt_tp, 0.5], dtype=np.float32)

        features.append(np.concatenate([vec, memory_features, extras]))
        sl_targets.append(opt_sl)
        tp_targets.append(opt_tp)

    return (
        np.array(features, dtype=np.float32),
        np.array(sl_targets, dtype=np.float32),
        np.array(tp_targets, dtype=np.float32),
    )


class ExitDataset(Dataset):
    def __init__(self, features: np.ndarray, sl: np.ndarray, tp: np.ndarray) -> None:
        self.features = features
        self.sl = sl
        self.tp = tp

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.sl[idx]),
            torch.tensor(self.tp[idx]),
        )


class ExitLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, sl_pred: torch.Tensor, tp_pred: torch.Tensor,
                score_pred: torch.Tensor, sl_true: torch.Tensor,
                tp_true: torch.Tensor) -> torch.Tensor:
        sl_loss = nn.functional.mse_loss(sl_pred, sl_true)
        tp_loss = nn.functional.mse_loss(tp_pred, tp_true)
        score_loss = nn.functional.mse_loss(score_pred, torch.ones_like(score_pred))
        return sl_loss + tp_loss + 0.5 * score_loss


def train_exit_optimizer(faiss_path: str, epochs: int = 20, lr: float = 1e-3,
                         batch_size: int = 64, device: str = "cuda",
                         model_dir: str = "/app/models/active/current") -> dict:
    print(f"loading FAISS metadata from {faiss_path}")
    metadata = load_faiss_metadata(faiss_path)
    vectors = load_faiss_vectors(faiss_path)

    if vectors is None or not metadata:
        print("no FAISS data found, aborting")
        return {"error": "no data"}

    print(f"loaded {len(metadata)} metadata entries, {vectors.shape[0]} vectors")

    features, sl_targets, tp_targets = build_exit_dataset(vectors, metadata)
    print(f"built {len(features)} training samples")

    if len(features) < 10:
        print("insufficient samples (<10), aborting")
        return {"error": "insufficient samples"}

    train_size = int(0.85 * len(features))
    val_size = len(features) - train_size

    train_ds = ExitDataset(features[:train_size], sl_targets[:train_size], tp_targets[:train_size])
    val_ds = ExitDataset(features[train_size:], sl_targets[train_size:], tp_targets[train_size:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = ExitOptimizer(in_dim=EXIT_OPT_IN_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = ExitLoss()

    best_val_loss = float("inf")
    best_state = None

    print(f"training ExitOptimizer: {len(train_ds)} train, {len(val_ds)} val, {epochs} epochs")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for x, sl, tp in train_loader:
            x, sl, tp = x.to(device), sl.to(device), tp.to(device)
            optimizer.zero_grad()
            sl_pred, tp_pred, score_pred = model(x)
            loss = loss_fn(sl_pred, tp_pred, score_pred, sl, tp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(x)
            n += len(x)
        train_loss /= max(n, 1)

        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for x, sl, tp in val_loader:
                x, sl, tp = x.to(device), sl.to(device), tp.to(device)
                sl_pred, tp_pred, score_pred = model(x)
                loss = loss_fn(sl_pred, tp_pred, score_pred, sl, tp)
                val_loss += loss.item() * len(x)
                vn += len(x)
        val_loss /= max(vn, 1)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"epoch {epoch:02d} train={train_loss:.4f} val={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    pt_path = os.path.join(model_dir, "exit_optimizer.pt")
    torch.save(best_state if best_state else model.state_dict(), pt_path)
    print(f"saved PyTorch checkpoint to {pt_path}")

    exit_model_path = os.path.join(model_dir, "exit_optimizer.onnx")
    os.makedirs(model_dir, exist_ok=True)

    dummy = torch.randn(1, EXIT_OPT_IN_DIM, device=device)
    try:
        torch.onnx.export(
            model, dummy, exit_model_path,
            input_names=["features"],
            output_names=["sl_pct", "tp_pct", "trade_score"],
            dynamic_axes={"features": {0: "batch"}, "sl_pct": {0: "batch"},
                          "tp_pct": {0: "batch"}, "trade_score": {0: "batch"}},
            opset_version=17,
        )
        print(f"exported ONNX model to {exit_model_path}")
    except Exception as exc:
        pt_path = os.path.join(model_dir, "exit_optimizer.pt")
        torch.save(model.state_dict(), pt_path)
        print(f"ONNX export failed ({exc}), saved PyTorch checkpoint to {pt_path}")

    return {
        "val_loss": best_val_loss,
        "samples": len(features),
        "epochs": epochs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ExitOptimizer")
    parser.add_argument("--faiss-path", default=os.getenv("FAISS_PATH", "/app/data/faiss_index"))
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "/app/models/active/current"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result = train_exit_optimizer(
        faiss_path=args.faiss_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        model_dir=args.model_dir,
    )
    print(json.dumps(result, indent=2))
