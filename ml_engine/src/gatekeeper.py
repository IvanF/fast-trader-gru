"""Gatekeeper — CatBoost binary classifier for trade success prediction.

Replaces hardcoded DynamicConfCap and MIN_EDGE with data-driven decisions.
Trains online on trade outcomes stored in InfluxDB.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Gatekeeper config
GK_THRESHOLD = float(os.getenv("GATEKEEPER_THRESHOLD", "0.55"))
GK_MIN_SAMPLES = int(os.getenv("GATEKEEPER_MIN_SAMPLES", "50"))
GK_RETRAIN_EVERY = int(os.getenv("GATEKEEPER_RETRAIN_EVERY", "50"))
GK_MODEL_PATH = os.getenv("GATEKEEPER_MODEL_PATH", "/app/data/gatekeeper_model.cbm")

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
    prediction: int  # 1=PASS, 0=REJECT
    features: List[float]
    threshold: float


class Gatekeeper:
    """Online-learning CatBoost classifier for trade success prediction.

    Trains on historical trade outcomes from InfluxDB.
    Predicts probability of trade being profitable before OMS entry.
    """

    def __init__(self) -> None:
        self._model = None
        self._trained = False
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

        if self._model is None:
            # No model yet → pass through (don't block)
            return GatekeeperResult(
                prob=0.55, prediction=1, features=feat_vec,
                threshold=GK_THRESHOLD,
            )

        try:
            prob = self._model.predict_proba([feat_vec])[0][1]
            prediction = 1 if prob >= GK_THRESHOLD else 0
            return GatekeeperResult(
                prob=float(prob), prediction=prediction,
                features=feat_vec, threshold=GK_THRESHOLD,
            )
        except Exception as e:
            logger.warning("Gatekeeper predict failed: %s", e)
            return GatekeeperResult(
                prob=0.55, prediction=1, features=feat_vec,
                threshold=GK_THRESHOLD,
            )

    def record_trade(self, features: Dict[str, float], pnl: float) -> None:
        """Record trade outcome for online training.

        Called after each trade closes with its PnL.
        """
        self._trade_buffer.append({
            **features,
            "net_pnl": pnl,
            "label": 1 if pnl > 0 else 0,
            "timestamp": time.time(),
        })

        # Auto-retrain every N trades
        if len(self._trade_buffer) >= GK_RETRAIN_EVERY:
            self._try_retrain()

    def _try_retrain(self) -> None:
        """Retrain model on accumulated trade buffer."""
        if len(self._trade_buffer) < GK_MIN_SAMPLES:
            logger.debug("Gatekeeper: %d/%d samples, skipping retrain",
                        len(self._trade_buffer), GK_MIN_SAMPLES)
            return

        try:
            import catboost
        except ImportError:
            logger.warning("CatBoost not installed — gatekeeper disabled")
            return

        try:
            X = []
            y = []
            for trade in self._trade_buffer:
                row = [trade.get(name, 0.0) for name in FEATURE_NAMES]
                X.append(row)
                y.append(trade["label"])

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int32)

            # Balance classes
            pos_count = int(y.sum())
            neg_count = len(y) - pos_count
            if pos_count == 0 or neg_count == 0:
                logger.debug("Gatekeeper: one class empty (%d/%d), skipping", pos_count, neg_count)
                return

            # CatBoost with class weights
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
                class_weights={0: neg_count / len(y), 1: pos_count / len(y)},
                random_seed=42,
            )

            # 80/20 split
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            model.fit(X_train, y_train, eval_set=(X_val, y_val))

            # Evaluate
            val_pred = model.predict(X_val)
            val_acc = float((val_pred == y_val).mean())

            if val_acc < 0.50:
                logger.warning("Gatekeeper: val_acc=%.3f < 0.50, skipping retrain", val_acc)
                return

            self._model = model
            self._trained = True
            self._train_count += 1
            self._last_train_time = time.time()

            # Feature importance
            fi = model.get_feature_importance()
            self._feature_importance = {
                FEATURE_NAMES[i]: float(fi[i]) for i in range(len(FEATURE_NAMES))
            }

            # Save model
            try:
                os.makedirs(os.path.dirname(GK_MODEL_PATH), exist_ok=True)
                model.save_model(GK_MODEL_PATH)
                logger.info("Gatekeeper retrained: samples=%d val_acc=%.3f importance=%s",
                           len(self._trade_buffer), val_acc,
                           dict(sorted(self._feature_importance.items(),
                                       key=lambda x: x[1], reverse=True)[:5]))
            except Exception as e:
                logger.warning("Gatekeeper: failed to save model: %s", e)

            # Keep last 500 samples for next retrain
            if len(self._trade_buffer) > 1000:
                self._trade_buffer = self._trade_buffer[-500:]

        except Exception as e:
            logger.error("Gatekeeper retrain failed: %s", e)

    def load_model(self) -> bool:
        """Load pre-trained model from disk."""
        if not os.path.exists(GK_MODEL_PATH):
            logger.info("Gatekeeper: no saved model found")
            return False

        try:
            import catboost
            self._model = catboost.CatBoostClassifier()
            self._model.load_model(GK_MODEL_PATH)
            self._trained = True
            fi = self._model.get_feature_importance()
            self._feature_importance = {
                FEATURE_NAMES[i]: float(fi[i]) for i in range(len(FEATURE_NAMES))
            }
            logger.info("Gatekeeper: loaded model from %s", GK_MODEL_PATH)
            return True
        except Exception as e:
            logger.warning("Gatekeeper: failed to load model: %s", e)
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

    def get_feature_importance(self) -> Dict[str, float]:
        return self._feature_importance.copy()
