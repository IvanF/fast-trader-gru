"""Deploy trained ONNX models atomically and trigger hot-swap."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import onnx
import onnxruntime as ort
import redis
import torch

from .model_manifest import ModelManifest, new_version
from .models.nn_models import MLP_IN_DIM, DecisionMLP, FusionModel

ONNX_FILENAMES = (
    "orderbook_cnn.onnx",
    "flow_gru_attention.onnx",
    "decision_mlp.onnx",
)


def embed_onnx_self_contained(onnx_path: str) -> None:
    """Rewrite ONNX so all weights live inside the .onnx file (no .onnx.data sidecar)."""
    model = onnx.load(onnx_path, load_external_data=True)
    tmp = f"{onnx_path}.selfcontained.tmp"
    onnx.save_model(model, tmp, save_as_external_data=False)
    os.replace(tmp, onnx_path)
    sidecar = f"{onnx_path}.data"
    if os.path.exists(sidecar):
        os.remove(sidecar)


def _torch_onnx_export(module, args, path: str, **export_kw) -> None:
    """Export via torch.onnx and always produce a self-contained ONNX artifact."""
    kwargs = dict(export_kw, opset_version=17)
    try:
        torch.onnx.export(module, args, path, use_external_data_format=False, **kwargs)
    except TypeError:
        torch.onnx.export(module, args, path, **kwargs)
    embed_onnx_self_contained(path)


def _atomic_copy_onnx_bundle(src: Path, dst: Path) -> None:
    """Copy .onnx and optional .onnx.data sidecar; finalize as self-contained."""
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copy2(src, tmp)
    data_src = Path(f"{src}.data")
    if data_src.exists():
        data_tmp = Path(f"{tmp}.data")
        shutil.copy2(data_src, data_tmp)
        os.replace(data_tmp, Path(f"{dst}.data"))
    os.replace(tmp, dst)
    embed_onnx_self_contained(str(dst))


def export_onnx_models(
    fusion: FusionModel,
    decision: DecisionMLP,
    output_dir: str,
    device: torch.device | None = None,
) -> dict[str, str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dev = device or torch.device("cpu")
    fusion.eval()
    decision.eval()

    cnn_path = os.path.join(output_dir, "orderbook_cnn.onnx")
    gru_path = os.path.join(output_dir, "flow_gru_attention.onnx")
    mlp_path = os.path.join(output_dir, "decision_mlp.onnx")

    dummy_ob = torch.randn(1, 60, 2, device=dev)
    _torch_onnx_export(
        fusion.cnn, dummy_ob, cnn_path,
        input_names=["orderbook_seq"],
        output_names=["cnn_embedding"],
        dynamic_axes={"orderbook_seq": {0: "batch"}, "cnn_embedding": {0: "batch"}},
    )

    dummy_flow = torch.randn(1, 60, 3, device=dev)
    _torch_onnx_export(
        fusion.gru, dummy_flow, gru_path,
        input_names=["flow_seq"],
        output_names=["gru_embedding"],
        dynamic_axes={"flow_seq": {0: "batch"}, "gru_embedding": {0: "batch"}},
    )

    dummy_fused = torch.randn(1, MLP_IN_DIM, device=dev)
    _torch_onnx_export(
        decision, dummy_fused, mlp_path,
        input_names=["fused_vector"],
        output_names=["decision_logits"],
        dynamic_axes={"fused_vector": {0: "batch"}, "decision_logits": {0: "batch"}},
    )

    return {
        "orderbook_cnn": cnn_path,
        "flow_gru_attention": gru_path,
        "decision_mlp": mlp_path,
    }


def validate_onnx(paths: dict[str, str], prefer_gpu: bool = False) -> None:
    available = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    for path in paths.values():
        sess = ort.InferenceSession(path, providers=providers)
        del sess


def promote_models(staging_dir: str, model_dir: str) -> ModelManifest:
    """Atomic promote: copy to versioned dir, update manifest, symlink current."""
    version = new_version()
    model_path = Path(model_dir)
    active_dir = model_path / "active" / version
    active_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "orderbook_cnn": "orderbook_cnn.onnx",
        "flow_gru_attention": "flow_gru_attention.onnx",
        "decision_mlp": "decision_mlp.onnx",
    }
    rel_paths: dict[str, str] = {}
    for key, fname in files.items():
        src = Path(staging_dir) / fname
        dst = active_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"staging onnx missing: {src}")
        _atomic_copy_onnx_bundle(src, dst)
        rel_paths[key] = f"active/{version}/{fname}"

    # Symlink swap: current -> active/{version}
    current_link = model_path / "current"
    tmp_link = model_path / f".current.{version}.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(f"active/{version}")
    if current_link.is_symlink() or current_link.exists():
        current_link.unlink(missing_ok=True)
    os.replace(tmp_link, current_link)

    manifest = ModelManifest(version=version, updated_at=time.time(), models=rel_paths)
    manifest.write_atomic(model_dir)
    return manifest


def publish_reload(redis_addr: str, channel: str, manifest: ModelManifest, model_dir: str) -> None:
    rdb = redis.Redis.from_url(f"redis://{redis_addr}", decode_responses=True)
    payload = json.dumps({
        "version": manifest.version,
        "model_dir": model_dir,
        "updated_at": manifest.updated_at,
    })
    rdb.publish(channel, payload)
