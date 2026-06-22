"""Thread-safe ONNX Runtime inference with atomic hot-swap reload."""

from __future__ import annotations

import gc
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort

from .model_manifest import ModelManifest
from .onnx_deploy import _atomic_copy_onnx_bundle, embed_onnx_self_contained

logger = logging.getLogger(__name__)


@dataclass
class _SessionBundle:
    version: str
    cnn: Optional[ort.InferenceSession]
    gru: Optional[ort.InferenceSession]
    mlp: Optional[ort.InferenceSession]


class HotSwapONNXInference:
    """Loads ONNX sessions from manifest; thread-safe hot-swap without stopping event loop."""

    def __init__(self, model_dir: str, state_dim: int = 128) -> None:
        self.model_dir = model_dir
        self.state_dim = state_dim
        self._lock = threading.RLock()
        self._bundle = self._load_bundle()
        self._last_manifest_mtime = self._manifest_mtime()
        self._last_onnx_mtime = self._max_onnx_mtime()

    @property
    def version(self) -> str:
        with self._lock:
            return self._bundle.version

    def _manifest_mtime(self) -> float:
        path = os.path.join(self.model_dir, "manifest.json")
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    def _max_onnx_mtime(self) -> float:
        manifest = ModelManifest.from_dir(self.model_dir)
        mtimes = [self._manifest_mtime()]
        for key in ("orderbook_cnn", "flow_gru_attention", "decision_mlp"):
            path = manifest.resolve(self.model_dir, key)
            try:
                mtimes.append(os.path.getmtime(path))
                data_path = f"{path}.data"
                if os.path.exists(data_path):
                    mtimes.append(os.path.getmtime(data_path))
            except OSError:
                pass
        current = Path(self.model_dir) / "current"
        if current.is_symlink():
            try:
                mtimes.append(os.path.getmtime(current))
            except OSError:
                pass
        return max(mtimes) if mtimes else 0.0

    def _session_options(self) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = True
        # Explicit thread counts prevent ORT from calling pthread_setaffinity_np
        # on host CPUs outside the container cpuset (e.g. {1,19} with cpuset 3-6).
        opts.intra_op_num_threads = int(os.getenv("ORT_INTRA_OP_THREADS", "4"))
        opts.inter_op_num_threads = int(os.getenv("ORT_INTER_OP_THREADS", "2"))
        try:
            opts.add_session_config_entry("session.disable_cpu_affinitization", "1")
        except Exception:
            pass
        try:
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        except Exception:
            pass
        return opts

    def _providers(self) -> list[str]:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _providers_for(self, key: str) -> list[str]:
        # PyTorch-exported GRU has no CUDA kernel in ORT; mixed CPU/GPU in one
        # session adds Memcpy nodes and spams warnings. Run GRU graph on CPU only.
        if key == "flow_gru_attention":
            return ["CPUExecutionProvider"]
        return self._providers()

    @staticmethod
    def _dispose_bundle(bundle: _SessionBundle) -> None:
        for name in ("cnn", "gru", "mlp"):
            sess = getattr(bundle, name, None)
            if sess is not None:
                try:
                    # ONNX Runtime 1.16+ may expose session.end_profiling; del is sufficient
                    del sess
                except Exception as exc:
                    logger.debug("session dispose %s: %s", name, exc)
        gc.collect()

    def _load_bundle(self) -> _SessionBundle:
        manifest = ModelManifest.from_dir(self.model_dir)
        opts = self._session_options()

        def load_session(key: str) -> Optional[ort.InferenceSession]:
            fname = {
                "orderbook_cnn": "orderbook_cnn.onnx",
                "flow_gru_attention": "flow_gru_attention.onnx",
                "decision_mlp": "decision_mlp.onnx",
            }[key]
            path = manifest.resolve(self.model_dir, key)
            if not os.path.exists(path):
                alt = Path(self.model_dir) / "current" / fname
                if alt.exists():
                    path = str(alt)
                else:
                    logger.warning("onnx model missing: %s", path)
                    return None
            key_providers = self._providers_for(key)
            try:
                sess = ort.InferenceSession(path, opts, providers=key_providers)
                logger.info("loaded %s from %s providers=%s", key, path, key_providers)
                return sess
            except Exception as exc:
                sidecar = f"{path}.data"
                if os.path.exists(sidecar) or "onnx.data" in str(exc).lower():
                    try:
                        embed_onnx_self_contained(path)
                        logger.info("embedded external ONNX weights for hot-swap: %s", path)
                        sess = ort.InferenceSession(path, opts, providers=key_providers)
                        logger.info("loaded %s from %s providers=%s", key, path, key_providers)
                        return sess
                    except Exception as repair_exc:
                        logger.error("onnx repair failed for %s: %s", path, repair_exc)
                bootstrap = Path(self.model_dir) / fname
                if bootstrap.exists() and bootstrap.resolve() != Path(path).resolve():
                    try:
                        _atomic_copy_onnx_bundle(bootstrap, Path(path))
                        logger.warning("restored %s from bootstrap weights at %s", key, bootstrap)
                        sess = ort.InferenceSession(path, opts, providers=key_providers)
                        logger.info("loaded %s from %s providers=%s", key, path, key_providers)
                        return sess
                    except Exception as bootstrap_exc:
                        logger.error("bootstrap onnx restore failed for %s: %s", path, bootstrap_exc)
                raise exc

        return _SessionBundle(
            version=manifest.version,
            cnn=load_session("orderbook_cnn"),
            gru=load_session("flow_gru_attention"),
            mlp=load_session("decision_mlp"),
        )

    def reload(self, force: bool = False) -> bool:
        manifest_mtime = self._manifest_mtime()
        onnx_mtime = self._max_onnx_mtime()
        if not force and manifest_mtime <= self._last_manifest_mtime and onnx_mtime <= self._last_onnx_mtime:
            return False

        try:
            new_bundle = self._load_bundle()
        except Exception as exc:
            logger.error("hot-swap load failed: %s", exc)
            return False

        with self._lock:
            old = self._bundle
            self._bundle = new_bundle
            self._last_manifest_mtime = manifest_mtime
            self._last_onnx_mtime = onnx_mtime
            version = new_bundle.version

        self._dispose_bundle(old)
        logger.info("hot-swapped ONNX models -> version=%s", version)
        return True

    def infer_state_vector(
        self,
        ob_seq: np.ndarray,
        flow_seq: np.ndarray,
        macro_features: np.ndarray,
    ) -> np.ndarray:
        with self._lock:
            bundle = self._bundle
        cnn_out = self._run_cnn(bundle.cnn, ob_seq)
        gru_out = self._run_gru(bundle.gru, flow_seq)
        fused = np.concatenate([cnn_out, gru_out, macro_features], axis=-1).astype(np.float32)

        if fused.shape[-1] < self.state_dim:
            pad = np.zeros(self.state_dim - fused.shape[-1], dtype=np.float32)
            fused = np.concatenate([fused, pad])
        elif fused.shape[-1] > self.state_dim:
            fused = fused[: self.state_dim]
        return fused

    def decide(self, v_state: np.ndarray, v_memory: np.ndarray) -> Tuple[str, float, float]:
        with self._lock:
            mlp_sess = self._bundle.mlp

        if mlp_sess is not None:
            expected = mlp_sess.get_inputs()[0].shape[1]
            inp = np.concatenate([v_state, v_memory]).astype(np.float32).reshape(1, -1)
            actual = inp.shape[1]
            if actual != expected:
                if actual > expected:
                    inp = inp[:, :expected]
                else:
                    pad = np.zeros((1, expected - actual), dtype=np.float32)
                    inp = np.concatenate([inp, pad], axis=1)
            outputs = mlp_sess.run(None, {mlp_sess.get_inputs()[0].name: inp})
            logits = outputs[0][0]
            direction_idx = int(np.argmax(logits[:3]))
            confidence = float(self._sigmoid(logits[3])) if len(logits) > 3 else 0.5
            vol_mult = float(np.clip(logits[4], 0.5, 3.0)) if len(logits) > 4 else 1.0
            directions = ["LONG", "SHORT", "HOLD"]
            return directions[direction_idx], confidence, vol_mult

        return self._heuristic_decision(v_state, v_memory)

    @staticmethod
    def _run_cnn(sess: Optional[ort.InferenceSession], ob_seq: np.ndarray) -> np.ndarray:
        if sess is None:
            return ob_seq.mean(axis=0)[:32]
        inp = ob_seq.reshape(1, *ob_seq.shape).astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: inp})
        return out[0].flatten()[:32]

    @staticmethod
    def _run_gru(sess: Optional[ort.InferenceSession], flow_seq: np.ndarray) -> np.ndarray:
        if sess is None:
            return flow_seq.mean(axis=0)[:32]
        inp = flow_seq.reshape(1, *flow_seq.shape).astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: inp})
        return out[0].flatten()[:32]

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = float(x)
        if not np.isfinite(x):
            return 0.5
        # Stable form avoids exp overflow for large |x| (bootstrap MLP can emit extreme logits).
        if x >= 0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        z = np.exp(x)
        return float(z / (1.0 + z))

    @staticmethod
    def _heuristic_decision(v_state: np.ndarray, v_memory: np.ndarray) -> Tuple[str, float, float]:
        obi = v_state[0] if len(v_state) > 0 else 0.0
        cvd = v_state[1] if len(v_state) > 1 else 0.0
        win_rate = v_memory[0] if len(v_memory) > 0 else 0.5
        score = abs(obi) * 0.4 + abs(cvd) * 0.3 + win_rate * 0.3
        if obi > 0.05 and cvd > 0:
            direction = "LONG"
        elif obi < -0.05 and cvd < 0:
            direction = "SHORT"
        else:
            direction = "HOLD"
        vol_mult = 1.0 + abs(v_state[3]) * 2 if len(v_state) > 3 else 1.0
        return direction, float(np.clip(score, 0.0, 1.0)), float(np.clip(vol_mult, 0.5, 3.0))


ONNXInference = HotSwapONNXInference
