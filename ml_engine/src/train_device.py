"""Resolve PyTorch training device — GPU-only by default."""

from __future__ import annotations

import os
import sys

import torch


def resolve_train_device(device_name: str | None = None) -> torch.device:
    """Return training device. Default TRAIN_DEVICE=cuda; falls back to CPU if unavailable."""
    want = (device_name or os.getenv("TRAIN_DEVICE", "cuda")).lower()
    if want == "cuda":
        if not torch.cuda.is_available():
            print(
                "TRAIN_DEVICE=cuda but CUDA is not available, falling back to CPU",
                file=sys.stderr,
            )
            return torch.device("cpu")
        name = torch.cuda.get_device_name(0)
        print(f"training device=cuda ({name})")
        return torch.device("cuda")
    if want == "cpu":
        print("training device=cpu (TRAIN_DEVICE=cpu override)")
        return torch.device("cpu")
    print(f"unknown TRAIN_DEVICE={want!r}, expected cuda or cpu", file=sys.stderr)
    sys.exit(4)


def log_cuda_status(logger) -> None:
    """Log GPU availability at engine startup."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("cuda available for training: %s (%.1f GiB)", name, mem)
    else:
        logger.error(
            "cuda NOT available — retraining will fail (TRAIN_DEVICE=cuda). "
            "Install nvidia-container-toolkit and set gpus: all on ml_engine."
        )
