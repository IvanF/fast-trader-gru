"""Gatekeeper — Binary classifier for trade success prediction.

Supports:
- ONNX Runtime for fast inference (gatekeeper.onnx)
- CatBoost fallback for online retraining (gatekeeper_model.cbm)
- Batch training via gatekeeper_trainer.py

Fail-safe: if no model is loaded, all signals PASS through.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

GK_THRESHOLD = float(os.getenv("GATEKEEPER_THRESHOLD", "0.55"))
GK_MIN_SAMPLES = int(os.getenv("GATEKEEPER_MIN_SAMPLES", "50"))
GK_RETRAIN_EVERY = int(os.getenv("GATEKEEPER_RETRAIN_EVERY", "50"))
GK_CB_MODEL_PATH = os.getenv("GATEKEEPER_CB_MODEL_PATH", "/app/data/gatekeeper_model.cbm")
GK_ONNX_MODEL_PATH = os.getenv("GATEKEEPER_MODEL_PATH", "/app/data/gatekeeper.onnx")

FEATURE_NAMES = [
    "confidence", "pred_pnl", "spread_pct", "obi", "volume_ratio",
    "momentum", "price_velocity", "atr_pct", "funding_rate",
    "btc_correlation", "volatility_multiplier", "symbol_wr",
    "symbol_pnl_sum", "symbol_consec_losses", "symbol_trades_24h",
    "hour_of_day", "open_positions_count", "recent_wr_20",
]


@dataclass
class GatekeeperResult:
    prob: float
    prediction: int
    features: List[float]
    threshold: float


class Gatekeeper:
    """Binary classifier for trade success prediction.

    Inference priority:
    1. ONNX Runtime (gatekeeper.onnx) — fastest, from batch training
    2. CatBoost native (gatekeeper_model.cbm) — from online retraining
    3. Pass-through (no model) — fail-safe, never blocks signals
    """

    def __init__(self) -> None:
        self._cb_model = None
        self._onnx_session = None
        self._trained = False
        self._inference_backend = "none"
        self._train_count = 0
        self._trade_buffer: List[Dict[str, Any]] = []
        self._last_train_time = 0.0
        self._feature_importance: Dict[str, float] = {}

    def predict(self, features: Dict[str, float]) -> GatekeeperResult:
        """Predict trade success probability.

        Args:
            features: dict with FEATURE_NAMES keys + symbol stats

        Returns:
            GatekeeperResult with prob, prediction, features, threshold
        """
        feat_vec = [features.get(name, 0.0) for name in FEATURE_NAMES]

        if not self._trained:
            return GatekeeperResult(
                prob=0.55, prediction=1, features=feat_vec,
                threshold=GK_THRESHOLD,
            )

        try:
            if self._onnx_session is not None:
                prob = self._predict_onnx(feat_vec)
            elif self._cb_model is not None:
                prob = float(self._cb_model.predict_proba([feat_vec])[0][1])
            else:
                return GatekeeperResult(
                    prob=0.55, prediction=1, features=feat_vec,
                    threshold=GK_THRESHOLD,
                )

            prediction = 1 if prob >= GK_THRESHOLD else 0
            return GatekeeperResult(
                prob=prob, prediction=prediction,
                features=feat_vec, threshold=GK_THRESHOLD,
            )
        except Exception as e:
            logger.warning("Gatekeeper predict failed (fail-safe PASS): %s", e)
            return GatekeeperResult(
                prob=0.55, prediction=1, features=feat_vec,
                threshold=GK_THRESHOLD,
            )

    def _predict_onnx(self, feat_vec: list) -> float:
        """Run ONNX inference."""
        input_name = self._onnx_session.get_inputs()[0].name
        data = np.array([feat_vec], dtype=np.float32)
        output = self._onnx_session.run(None, {input_name: data})
        prob = float(output[0][0][1]) if len(output[0][0]) > 1 else float(output[0][0])
        return prob

    def record_trade(self, features: Dict[str, float], pnl: float) -> None:
        """Record trade outcome for online training."""
        self._trade_buffer.append({
            **features,
            "net_pnl": pnl,
            "label": 1 if pnl > 0 else 0,
            "timestamp": time.time(),
        })

        if len(self._trade_buffer) >= GK_RETRAIN_EVERY:
            self._try_retrain()

    def _try_retrain(self) -> None:
        """Online retrain with CatBoost (saves .cbm for fallback)."""
        if len(self._trade_buffer) < GK_MIN_SAMPLES:
            return

        try:
            import catboost
        except ImportError:
            logger.warning("CatBoost not installed — online retrain disabled")
            return

        try:
            X, y = [], []
            for trade in self._trade_buffer:
                row = [trade.get(name, 0.0) for name in FEATURE_NAMES]
                X.append(row)
                y.append(trade["label"])

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int32)

            pos_count = int(y.sum())
            neg_count = len(y) - pos_count
            if pos_count == 0 or neg_count == 0:
                return

            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            weights = {0: neg_count / len(y), 1: pos_count / len(y)}

            model = catboost.CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                border_count=128,
                loss_function="Logloss",
                eval_metric="Accuracy",
                early_stopping_rounds=50,
                verbose=0,
                class_weights=weights,
                random_seed=42,
            )

            model.fit(X_train, y_train, eval_set=(X_val, y_val))

            val_pred = model.predict(X_val)
            val_acc = float((val_pred == y_val).mean())

            if val_acc < 0.50:
                logger.warning("Gatekeeper online retrain: val_acc=%.3f < 0.50, skipping", val_acc)
                return

            self._cb_model = model
            self._trained = True
            self._inference_backend = "catboost"
            self._train_count += 1
            self._last_train_time = time.time()

            fi = model.get_feature_importance()
            self._feature_importance = {
                FEATURE_NAMES[i]: float(fi[i]) for i in range(len(FEATURE_NAMES))
            }

            try:
                os.makedirs(os.path.dirname(GK_CB_MODEL_PATH), exist_ok=True)
                model.save_model(GK_CB_MODEL_PATH)
                logger.info("Gatekeeper online retrain: samples=%d val_acc=%.3f top=%s",
                            len(self._trade_buffer), val_acc,
                            dict(sorted(self._feature_importance.items(),
                                         key=lambda x: x[1], reverse=True)[:5]))
            except Exception as e:
                logger.warning("Gatekeeper: failed to save CB model: %s", e)

            if len(self._trade_buffer) > 1000:
                self._trade_buffer = self._trade_buffer[-500:]

        except Exception as e:
            logger.error("Gatekeeper retrain failed: %s", e)

    def load_model(self) -> bool:
        """Load pre-trained model. Tries ONNX first, then CatBoost."""
        # Try ONNX first
        if os.path.exists(GK_ONNX_MODEL_PATH):
            try:
                import onnxruntime as ort
                self._onnx_session = ort.InferenceSession(
                    GK_ONNX_MODEL_PATH,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self._trained = True
                self._inference_backend = "onnx"
                logger.info("Gatekeeper: loaded ONNX model from %s (provider=%s)",
                            GK_ONNX_MODEL_PATH,
                            self._onnx_session.get_providers()[0])
                return True
            except Exception as e:
                logger.warning("Gatekeeper: ONNX load failed (%s), trying CatBoost", e)

        # Try CatBoost fallback
        if os.path.exists(GK_CB_MODEL_PATH):
            try:
                import catboost
                self._cb_model = catboost.CatBoostClassifier()
                self._cb_model.load_model(GK_CB_MODEL_PATH)
                self._trained = True
                self._inference_backend = "catboost"
                fi = self._cb_model.get_feature_importance()
                self._feature_importance = {
                    FEATURE_NAMES[i]: float(fi[i]) for i in range(len(FEATURE_NAMES))
                }
                logger.info("Gatekeeper: loaded CatBoost model from %s", GK_CB_MODEL_PATH)
                return True
            except Exception as e:
                logger.warning("Gatekeeper: CatBoost load failed: %s", e)

        logger.info("Gatekeeper: no saved model found (pass-through mode)")
        return False

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def sample_count(self) -> int:
        return len(self._trade_buffer)

    @property
    def train_count(self) -> int:
        return self._train_count

    @property
    def backend(self) -> str:
        return self._inference_backend

    def get_feature_importance(self) -> Dict[str, float]:
        return self._feature_importance.copy()
