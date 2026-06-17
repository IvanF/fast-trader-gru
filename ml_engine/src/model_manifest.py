"""Atomic model manifest for hot-swap deployments."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_NAME = "manifest.json"


@dataclass
class ModelManifest:
    version: str
    updated_at: float
    models: dict[str, str]

    @classmethod
    def from_dir(cls, model_dir: str) -> "ModelManifest":
        path = Path(model_dir) / MANIFEST_NAME
        if not path.exists():
            return cls(
                version="bootstrap",
                updated_at=0.0,
                models={
                    "orderbook_cnn": "orderbook_cnn.onnx",
                    "flow_gru_attention": "flow_gru_attention.onnx",
                    "decision_mlp": "decision_mlp.onnx",
                },
            )
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            version=raw["version"],
            updated_at=float(raw.get("updated_at", 0)),
            models=dict(raw["models"]),
        )

    def resolve(self, model_dir: str, key: str) -> str:
        rel = self.models.get(key, f"{key}.onnx")
        return str(Path(model_dir) / rel)

    def write_atomic(self, model_dir: str) -> None:
        path = Path(model_dir) / MANIFEST_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": self.version,
            "updated_at": self.updated_at,
            "models": self.models,
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def new_version() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
