#!/usr/bin/env python3
"""
Rolling retrain pipeline with PnL-joined InfluxDB labels and asymmetric loss.

Pulls the last N hours (default 6) of microstructure data joined with trade_outcomes.
Runs at low OS priority (nice 15) when invoked by the retrain worker.

Writes /app/data/retrain_report.json for Prometheus bridge in ml_engine.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.influx_join import build_joined_dataset
from src.influx_store import InfluxStore
from src.models.nn_models import TradingModel
from src.onnx_deploy import export_onnx_models, promote_models, publish_reload, validate_onnx
from src.train_device import resolve_train_device

REPORT_PATH = os.getenv("RETRAIN_REPORT_PATH", "/app/data/retrain_report.json")
MIN_SAMPLES = int(os.getenv("RETRAIN_MIN_SAMPLES", "20"))
MIN_VAL_ACC = float(os.getenv("RETRAIN_MIN_VAL_ACC", "0.45"))

# Regularization defaults — prevent confidence logit saturation (sigmoid → 1.0).
DEFAULT_LABEL_SMOOTHING = float(os.getenv("TRAIN_LABEL_SMOOTHING", "0.1"))
DEFAULT_CONFIDENCE_EPS = float(os.getenv("TRAIN_CONFIDENCE_LABEL_EPS", "0.05"))
DEFAULT_WEIGHT_DECAY = float(os.getenv("TRAIN_WEIGHT_DECAY", "5e-4"))
DEFAULT_MAX_GRAD_NORM = float(os.getenv("TRAIN_MAX_GRAD_NORM", "1.0"))


def smooth_binary_targets(targets: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    """Map labels in [0, 1] to [eps, 1-eps] so BCEWithLogitsLoss does not push logits to ±inf."""
    eps = float(eps)
    return targets.clamp(0.0, 1.0) * (1.0 - 2.0 * eps) + eps


def apply_low_priority(nice_level: int = 15) -> None:
    try:
        os.nice(nice_level)
        print(f"os.nice({nice_level}) applied")
    except (AttributeError, PermissionError) as exc:
        print(f"could not set nice level: {exc}")


class DecoupledTrapLoss(nn.Module):
    """Decoupled loss: direction CE + confidence MSE + trap BCE. No penalty for losing trades."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: dict,
    ) -> torch.Tensor:
        # 1. Cross-entropy for direction (NO loss penalty!)
        dir_loss = F.cross_entropy(
            logits[:, :3], targets["direction"], label_smoothing=0.1
        )
        # 2. MSE for confidence
        conf_loss = F.mse_loss(
            torch.sigmoid(logits[:, 3]), targets["confidence"]
        )
        # 3. BCE for trap head — THIS is where the model learns about losing trades
        trap_loss = F.binary_cross_entropy_with_logits(
            logits[:, 5], targets["is_trap_label"]
        )
        return dir_loss + 0.5 * conf_loss + 1.5 * trap_loss


def stratified_split(dataset: JoinedDataset, val_ratio: float = 0.15, min_val: int = 10) -> tuple:
    """Stratified train/val split preserving direction distribution."""
    n = len(dataset)
    val_size = max(min_val, int(val_ratio * n))
    val_size = min(val_size, n - 1)
    labels = dataset.direction
    classes = np.unique(labels)
    val_indices: list[int] = []
    train_indices: list[int] = list(range(n))
    rng = np.random.RandomState(42)
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        n_cls_val = max(1, int(val_ratio * len(cls_idx)))
        n_cls_val = min(n_cls_val, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n_cls_val, replace=False)
        val_indices.extend(chosen.tolist())
    for idx in val_indices:
        train_indices.remove(idx)
    if len(val_indices) < min_val:
        extra = rng.choice(train_indices, size=min(min_val - len(val_indices), len(train_indices)), replace=False)
        for e in extra:
            val_indices.append(int(e))
            train_indices.remove(int(e))
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


class JoinedDataset(Dataset):
    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self.ob = data["ob_seq"]
        self.flow = data["flow_seq"]
        self.macro = data["macro"]
        self.memory = data["memory"]
        self.direction = data["direction"]
        self.confidence = data["confidence"]
        self.pnl = data.get("pnl", np.zeros(len(data["direction"])))
        # Trap label: 1.0 if PnL < 0 (losing trade = likely trap), else 0.0
        self.is_trap = (self.pnl < 0).astype(np.float32)
        self.n = len(self.direction)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple:
        return (
            torch.from_numpy(self.ob[idx]),
            torch.from_numpy(self.flow[idx]),
            torch.from_numpy(self.macro[idx]),
            torch.from_numpy(self.memory[idx]),
            torch.tensor(self.direction[idx], dtype=torch.long),
            torch.tensor(self.confidence[idx], dtype=torch.float32),
            torch.tensor(float(self.pnl[idx]), dtype=torch.float32),
            torch.tensor(self.is_trap[idx], dtype=torch.float32),
        )


def load_npz(path: str) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def merge_datasets(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("no datasets to merge")
    keys = parts[0].keys()
    out: dict[str, np.ndarray] = {}
    for k in keys:
        if k == "symbols":
            out[k] = np.concatenate([p[k] for p in parts])
        else:
            out[k] = np.concatenate([p[k] for p in parts], axis=0)
    return out


def load_checkpoint(model: TradingModel, model_dir: str, device: torch.device) -> None:
    ckpt_dir = Path(model_dir).parent / "data" / "checkpoints"
    if not ckpt_dir.exists():
        return
    ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        return
    try:
        state = torch.load(ckpts[0], map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpts[0], map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"], strict=False)
        print(f"loaded incremental checkpoint {ckpts[0].name}")


def train_epoch(
    model: TradingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    trap_loss: DecoupledTrapLoss,
    max_grad_norm: float,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for ob, flow, macro, memory, direction, confidence, pnl, is_trap in loader:
        ob = ob.to(device, non_blocking=True)
        flow = flow.to(device, non_blocking=True)
        macro = macro.to(device, non_blocking=True)
        memory = memory.to(device, non_blocking=True)
        direction = direction.to(device, non_blocking=True)
        confidence = confidence.to(device, non_blocking=True)
        pnl = pnl.to(device, non_blocking=True)
        is_trap = is_trap.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        _, logits = model(ob, flow, macro, memory)
        targets = {
            "direction": direction,
            "confidence": confidence,
            "is_trap_label": is_trap,
        }
        loss = trap_loss(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += float(loss.item()) * len(direction)
        n += len(direction)
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(
    model: TradingModel,
    loader: DataLoader,
    device: torch.device,
    trap_loss: DecoupledTrapLoss,
) -> tuple[float, float, float]:
    model.eval()
    total, correct, n = 0.0, 0, 0
    trap_correct = 0
    trap_total = 0
    for ob, flow, macro, memory, direction, confidence, pnl, is_trap in loader:
        ob = ob.to(device, non_blocking=True)
        flow = flow.to(device, non_blocking=True)
        macro = macro.to(device, non_blocking=True)
        memory = memory.to(device, non_blocking=True)
        direction = direction.to(device, non_blocking=True)
        pnl = pnl.to(device, non_blocking=True)
        is_trap = is_trap.to(device, non_blocking=True)
        confidence = confidence.to(device, non_blocking=True)
        _, logits = model(ob, flow, macro, memory)
        targets = {
            "direction": direction,
            "confidence": confidence,
            "is_trap_label": is_trap,
        }
        total += float(trap_loss(logits, targets).item()) * len(direction)
        correct += int((logits[:, :3].argmax(dim=1) == direction).sum().item())
        trap_pred = (torch.sigmoid(logits[:, 5]) > 0.5).float()
        trap_correct += int((trap_pred == is_trap).sum().item())
        trap_total += len(direction)
        n += len(direction)
    return total / max(n, 1), correct / max(n, 1), trap_correct / max(trap_total, 1)


def write_report(report: dict) -> None:
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_PATH)


def main() -> int:
    p = argparse.ArgumentParser(description="Rolling retrain with asymmetric PnL loss")
    p.add_argument("--symbols", default="", help="Comma-separated symbols")
    p.add_argument("--hours", type=int, default=6, help="Influx lookback hours (2-6 recommended)")
    p.add_argument("--days", type=int, default=0, help="Legacy: overrides hours if set")
    p.add_argument("--dataset", default="", help="Pre-exported joined .npz")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "/app/models"))
    p.add_argument("--no-publish", action="store_true")
    p.add_argument("--incremental", action="store_true", help="Warm-start from latest checkpoint")
    p.add_argument("--trigger", default="manual", help="interval|trade_threshold|manual")
    p.add_argument("--nice-level", type=int, default=int(os.getenv("TRAIN_NICE_LEVEL", "15")))
    p.add_argument("--loss-penalty", type=float, default=float(os.getenv("LOSS_PENALTY_WEIGHT", "2.5")))
    p.add_argument("--label-smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING)
    p.add_argument("--confidence-eps", type=float, default=DEFAULT_CONFIDENCE_EPS)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    p.add_argument("--device", default=os.getenv("TRAIN_DEVICE", "cuda"), help="Training device (cuda|cpu)")
    args = p.parse_args()

    apply_low_priority(args.nice_level)
    start_time = time.time()
    hours = max(2, min(48, args.hours)) if not args.days else args.days * 24

    device = resolve_train_device(args.device)
    print(f"lookback={hours}h trigger={args.trigger}")

    if args.dataset:
        data = load_npz(args.dataset)
    else:
        if not args.symbols:
            print("Provide --symbols or --dataset", file=sys.stderr)
            return 2
        store = InfluxStore(
            os.getenv("INFLUX_URL", "http://influxdb:8086"),
            os.getenv("INFLUX_TOKEN", ""),
            os.getenv("INFLUX_ORG", "fasttrader"),
            os.getenv("INFLUX_BUCKET_RAW", "market_raw"),
            os.getenv("INFLUX_BUCKET_FEATURES", "market_features"),
        )
        start = f"-{hours}h"
        parts = []
        for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
            print(f"joining Influx {sym} ({start})...")
            parts.append(build_joined_dataset(store, start, symbol=sym))
        data = merge_datasets(parts)
        store.close()

    n_before = len(data["direction"])
    # Oversample minority direction to balance LONG/SHORT
    # Exclude outlier PnL entries from oversampling pool to avoid duplicating extremes
    pnl_arr = data["pnl"]
    safe_mask = (pnl_arr >= -0.05) & (pnl_arr <= 0.08)
    dirs = np.unique(data["direction"])
    dir_counts = {int(d): int(np.sum(data["direction"] == d)) for d in dirs}
    if 0 in dir_counts and 1 in dir_counts:
        n_long = dir_counts.get(0, 0)
        n_short = dir_counts.get(1, 0)
        if n_long > 0 and n_short > 0:
            target = max(n_long, n_short)
            indices_long = np.where((data["direction"] == 0) & safe_mask)[0]
            indices_short = np.where((data["direction"] == 1) & safe_mask)[0]
            if len(indices_long) == 0:
                indices_long = np.where(data["direction"] == 0)[0]
            if len(indices_short) == 0:
                indices_short = np.where(data["direction"] == 1)[0]
            oversample_long = np.random.choice(indices_long, size=target - n_long, replace=True) if target > n_long else np.array([], dtype=int)
            oversample_short = np.random.choice(indices_short, size=target - n_short, replace=True) if target > n_short else np.array([], dtype=int)
            oversample = np.concatenate([oversample_long, oversample_short])
            if len(oversample) > 0:
                for key in data:
                    if isinstance(data[key], np.ndarray) and key not in ("symbols", "timestamps"):
                        data[key] = np.concatenate([data[key], data[key][oversample]], axis=0)
                    elif key == "symbols":
                        data[key] = np.concatenate([data[key], data[key][oversample]])
                print(f"oversampled: {n_before} → {len(data['direction'])} (balanced LONG/SHORT)")

    n = len(data["direction"])
    print(f"training samples: {n}")
    if n < MIN_SAMPLES:
        report = {
            "exit_code": 3,
            "error": "insufficient_samples",
            "samples": n,
            "min_required": MIN_SAMPLES,
            "timestamp": time.time(),
            "trigger": args.trigger,
        }
        write_report(report)
        print(f"insufficient training samples (need >= {MIN_SAMPLES})", file=sys.stderr)
        return 3

    dataset = JoinedDataset(data)
    train_ds, val_ds = stratified_split(dataset, val_ratio=0.15, min_val=10)
    print(f"split: train={len(train_ds)} val={len(val_ds)}")
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=use_cuda,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0, pin_memory=use_cuda)

    model = TradingModel().to(device)
    if args.incremental:
        load_checkpoint(model, args.model_dir, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trap_loss_fn = DecoupledTrapLoss()
    print(
        f"loss=DecoupledTrapLoss weight_decay={args.weight_decay:g} "
        f"max_grad_norm={args.max_grad_norm:g}"
    )

    initial_loss = 0.0
    final_loss = 0.0
    best_acc = 0.0
    best_trap_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, trap_loss_fn,
            args.max_grad_norm,
        )
        val_loss, val_acc, val_trap_acc = eval_epoch(model, val_loader, device, trap_loss_fn)
        if epoch == 1:
            initial_loss = val_loss
        final_loss = val_loss
        print(f"epoch {epoch:02d} train={train_loss:.4f} val={val_loss:.4f} dir_acc={val_acc:.3f} trap_acc={val_trap_acc:.3f}")
        # Promote if direction accuracy improved OR trap accuracy improved
        score = val_acc * 0.5 + val_trap_acc * 0.5
        if score >= best_acc:
            best_acc = score
            best_trap_acc = val_trap_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    @torch.no_grad()
    def _check_signal_rate(mdl, loader, device):
        mdl.eval()
        total, acted = 0, 0
        for ob, flow, macro, memory, direction, confidence, pnl, _ in loader:
            ob = ob.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            macro = macro.to(device, non_blocking=True)
            memory = memory.to(device, non_blocking=True)
            _, logits = mdl(ob, flow, macro, memory)
            pred = logits[:, :3].argmax(dim=1)
            conf = torch.sigmoid(logits[:, 3])
            for p, c in zip(pred.tolist(), conf.tolist()):
                total += 1
                if p != 2 and c >= 0.30:
                    acted += 1
        return acted / max(total, 1)

    signal_rate = _check_signal_rate(model, val_loader, device)
    print(f"val signal_rate={signal_rate:.3f}")

    loss_delta = final_loss - initial_loss
    loss_improved = loss_delta < -0.01

    prev_acc = 0.0
    prev_version = "none"
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                prev_report = json.load(f)
            prev_acc = float(prev_report.get("val_acc", 0))
            prev_version = str(prev_report.get("version", "unknown"))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    print(f"previous model: version={prev_version} val_acc={prev_acc:.3f}")
    print(f"new model: score={best_acc:.3f} trap_acc={best_trap_acc:.3f} loss_delta={loss_delta:.4f}")

    # NEW promotion logic: combined score > previous + 0.02
    acc_improved = best_acc > prev_acc + 0.02
    should_promote = acc_improved or loss_improved

    if not should_promote:
        reason = []
        if not acc_improved:
            reason.append(f"no improvement (prev={prev_acc:.3f}, new={best_acc:.3f})")
        if not loss_improved:
            reason.append(f"loss not improved (delta={loss_delta:.4f})")
        print(f"model NOT promoted: {'; '.join(reason)}", file=sys.stderr)
        report = {
            "exit_code": 5,
            "version": "rejected",
            "new_acc": best_acc,
            "prev_acc": prev_acc,
            "prev_version": prev_version,
            "loss_delta": loss_delta,
            "samples": n,
            "reason": "; ".join(reason),
            "timestamp": time.time(),
            "trigger": args.trigger,
        }
        write_report(report)
        return 5

    staging = os.path.join(args.model_dir, "staging")
    paths = export_onnx_models(model.fusion, model.decision, staging, device=device)
    validate_onnx(paths, prefer_gpu=use_cuda)
    manifest = promote_models(staging, args.model_dir)
    print(f"promoted version={manifest.version} acc={best_acc:.3f} (prev={prev_acc:.3f})")

    if not args.no_publish:
        redis_addr = os.getenv("REDIS_ADDR", "redis:6379")
        channel = os.getenv("MODEL_RELOAD_CHANNEL", "models:reload")
        try:
            publish_reload(redis_addr, channel, manifest, args.model_dir)
        except Exception as exc:
            print(f"redis publish failed: {exc}")

    ckpt_dir = Path(args.model_dir).parent / "data" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "version": manifest.version, "val_acc": best_acc},
               ckpt_dir / f"{manifest.version}.pt")

    duration = time.time() - start_time
    report = {
        "exit_code": 0,
        "version": manifest.version,
        "timestamp": time.time(),
        "duration_sec": duration,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_delta": loss_delta,
        "val_acc": best_acc,
        "signal_rate": signal_rate,
        "prev_acc": prev_acc,
        "prev_version": prev_version,
        "samples": n,
        "trigger": args.trigger,
        "lookback_hours": hours,
        "label_smoothing": args.label_smoothing,
        "confidence_eps": args.confidence_eps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
    }
    write_report(report)
    print(f"training complete in {duration:.1f}s loss_delta={loss_delta:.4f} signal_rate={signal_rate:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
