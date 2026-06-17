#!/usr/bin/env python3
"""Export PyTorch models to ONNX and create bootstrap manifest."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_manifest import ModelManifest
from src.models.nn_models import DecisionMLP, FusionModel
from src.onnx_deploy import export_onnx_models, validate_onnx


def export_models(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fusion = FusionModel()
    decision = DecisionMLP()
    paths = export_onnx_models(fusion, decision, output_dir)
    validate_onnx(paths)

    manifest = ModelManifest(
        version="bootstrap",
        updated_at=time.time(),
        models={
            "orderbook_cnn": "orderbook_cnn.onnx",
            "flow_gru_attention": "flow_gru_attention.onnx",
            "decision_mlp": "decision_mlp.onnx",
        },
    )
    manifest.write_atomic(output_dir)
    print(f"exported ONNX models to {output_dir} manifest={manifest.version}")


if __name__ == "__main__":
    out = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
    export_models(out)
