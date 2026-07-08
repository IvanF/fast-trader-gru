"""Gatekeeper Batch Trainer — reads from InfluxDB, trains CatBoost, exports ONNX.

Usage:
    python3 -m src.gatekeeper_trainer

Environment:
    INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET_RAW
    GATEKEEPER_MODEL_PATH (default: /app/data/gatekeeper.onnx)
    GATEKEEPER_THRESHOLD (default: 0.55)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

GK_MODEL_PATH = os.getenv("GATEKEEPER_MODEL_PATH", "/app/data/gatekeeper.onnx")
GK_CB_MODEL_PATH = os.getenv("GATEKEEPER_CB_MODEL_PATH", "/app/data/gatekeeper_model.cbm")
GK_THRESHOLD = float(os.getenv("GATEKEEPER_THRESHOLD", "0.55"))
GK_MIN_SAMPLES = int(os.getenv("GATEKEEPER_MIN_SAMPLES", "50"))

FEATURE_NAMES = [
    "confidence", "pred_pnl", "spread_pct", "obi", "volume_ratio",
    "momentum", "price_velocity", "atr_pct", "funding_rate",
    "btc_correlation", "volatility_multiplier", "symbol_wr",
    "symbol_pnl_sum", "symbol_consec_losses", "symbol_trades_24h",
    "hour_of_day", "open_positions_count", "recent_wr_20",
]

CAT_FEATURES = ["symbol", "direction", "regime"]

NUMERIC_FEATURES = [
    "confidence", "spread_pct", "obi", "volume_ratio",
    "momentum", "price_velocity", "atr_pct", "funding_rate",
    "btc_correlation", "volatility_multiplier", "symbol_wr",
    "symbol_pnl_sum", "symbol_consec_losses", "symbol_trades_24h",
    "hour_of_day", "open_positions_count", "recent_wr_20",
]


def fetch_from_influx() -> list[dict]:
    """Fetch gatekeeper_features from InfluxDB."""
    try:
        from influxdb_client import InfluxDBClient
    except ImportError:
        logger.error("influxdb-client not installed")
        return []

    url = os.getenv("INFLUX_URL", "http://influxdb:8086")
    token = os.getenv("INFLUX_TOKEN", "")
    org = os.getenv("INFLUX_ORG", "fasttrader")
    bucket = os.getenv("INFLUX_BUCKET_RAW", "market_raw")

    if not token:
        logger.error("INFLUX_TOKEN not set")
        return []

    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()

    query = f'''
    from(bucket: "{bucket}")
      |> range(start: -30d)
      |> filter(fn: (r) => r["_measurement"] == "gatekeeper_features")
      |> filter(fn: (r) => r["_field"] == "label")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''

    try:
        tables = query_api.query(query)
    except Exception as e:
        logger.error("InfluxDB query failed: %s", e)
        client.close()
        return []

    rows = []
    for table in tables:
        for record in table.records:
            row = dict(record.values)
            row["_time"] = record.get_time()
            rows.append(row)

    if not rows:
        logger.warning("No gatekeeper_features found in InfluxDB")
        client.close()
        return []

    # Second query to get all fields for matching timestamps
    field_query = f'''
    from(bucket: "{bucket}")
      |> range(start: -30d)
      |> filter(fn: (r) => r["_measurement"] == "gatekeeper_features")
      |> pivot(rowKey: ["_time", "symbol", "direction", "close_reason"], columnKey: ["_field"], valueColumn: "_value")
    '''

    try:
        tables = query_api.query(field_query)
    except Exception as e:
        logger.error("InfluxDB field query failed: %s", e)
        client.close()
        return []

    full_rows = []
    for table in tables:
        for record in table.records:
            d = dict(record.values)
            d["_time"] = record.get_time()
            full_rows.append(d)

    client.close()
    logger.info("Fetched %d gatekeeper samples from InfluxDB", len(full_rows))
    return full_rows


def prepare_data(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """Convert raw rows to feature matrix and labels.

    Returns (X_numeric, y, cat_values_per_feature, all_features_dict).
    """
    X = []
    y = []
    cat_data = {name: [] for name in CAT_FEATURES}
    dropped = 0

    for row in rows:
        label = row.get("label")
        if label is None:
            dropped += 1
            continue

        row = dict(row)
        numeric = []
        valid = True
        for fname in NUMERIC_FEATURES:
            val = row.get(fname, 0.0)
            if val is None:
                val = 0.0
            try:
                numeric.append(float(val))
            except (ValueError, TypeError):
                numeric.append(0.0)

        if not valid:
            dropped += 1
            continue

        X.append(numeric)
        y.append(int(float(label)))

        for cat in CAT_FEATURES:
            val = row.get(cat, "unknown")
            if val is None:
                val = "unknown"
            cat_data[cat].append(str(val))

    if dropped > 0:
        logger.info("Dropped %d invalid rows", dropped)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), cat_data


def train_model(X: np.ndarray, y: np.ndarray, cat_data: dict) -> object | None:
    """Train CatBoostClassifier with temporal split (no shuffle)."""
    try:
        import catboost
    except ImportError:
        logger.error("catboost not installed — cannot train")
        return None

    n = len(X)
    if n < GK_MIN_SAMPLES:
        logger.warning("Not enough samples: %d < %d", n, GK_MIN_SAMPLES)
        return None

    pos_count = int(y.sum())
    neg_count = n - pos_count
    if pos_count == 0 or neg_count == 0:
        logger.warning("One class empty: %d/%d", pos_count, neg_count)
        return None

    # Temporal split: first 80% train, last 20% test (NO shuffle!)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    logger.info("Train: %d samples (%d pos, %d neg), Test: %d samples",
                split, int(y_train.sum()), split - int(y_train.sum()),
                n - split)

    # Class weights for imbalanced data
    train_pos = int(y_train.sum())
    train_neg = split - train_pos
    weights = {0: train_neg / split, 1: train_pos / split}

    # Combine numeric + categorical columns for CatBoost
    # Categorical columns use object dtype so CatBoost recognizes them
    cat_arr_train = np.array([cat_data[c][:split] for c in CAT_FEATURES], dtype=object).T
    cat_arr_test = np.array([cat_data[c][split:] for c in CAT_FEATURES], dtype=object).T

    X_train_cb = np.hstack([X_train, cat_arr_train])
    X_test_cb = np.hstack([X_test, cat_arr_test])

    all_feature_names = NUMERIC_FEATURES + CAT_FEATURES
    cat_indices = list(range(len(NUMERIC_FEATURES), len(all_feature_names)))

    model = catboost.CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        border_count=128,
        loss_function="Logloss",
        eval_metric="AUC",
        early_stopping_rounds=50,
        verbose=100,
        class_weights=weights,
        random_seed=42,
    )

    model.fit(
        X_train_cb, y_train,
        cat_features=cat_indices,
        feature_names=all_feature_names,
        eval_set=(X_test_cb, y_test),
    )

    # Evaluate
    from sklearn.metrics import roc_auc_score, accuracy_score

    y_pred_proba = model.predict_proba(X_test_cb)[:, 1]
    y_pred = (y_pred_proba >= GK_THRESHOLD).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    logger.info("=== Gatekeeper Training Results ===")
    logger.info("  Samples: train=%d test=%d", split, n - split)
    logger.info("  AUC: %.4f (target: > 0.60)", auc)
    logger.info("  Accuracy: %.4f (target: > 0.55)", acc)
    logger.info("  Threshold: %.3f", GK_THRESHOLD)

    fi = model.get_feature_importance()
    importance = sorted(
        zip(FEATURE_NAMES, fi), key=lambda x: x[1], reverse=True
    )
    logger.info("  Top features: %s",
                {name: f"{imp:.2f}" for name, imp in importance[:8]})

    if auc < 0.60 or acc < 0.55:
        logger.warning("Model quality below threshold (AUC=%.3f, Acc=%.3f) — NOT exporting", auc, acc)
        return None

    return model, auc, acc, importance


def export_onnx(model, auc: float, acc: float) -> bool:
    """Export CatBoost model to ONNX format for fast inference."""
    try:
        os.makedirs(os.path.dirname(GK_MODEL_PATH), exist_ok=True)

        # Save CatBoost native model first (for fallback)
        model.save_model(GK_CB_MODEL_PATH)
        logger.info("Saved CatBoost model to %s", GK_CB_MODEL_PATH)

        # Export to ONNX
        try:
            model.export_model(
                GK_MODEL_PATH,
                format="onnx",
                export_parameters={
                    "onnx_domain": "ai.catboost",
                    "model_version": 0,
                    "onnx_doc_string": f"Gatekeeper AUC={auc:.4f} Acc={acc:.4f}",
                },
            )
            logger.info("Exported ONNX model to %s", GK_MODEL_PATH)
            return True
        except Exception as e:
            logger.warning("ONNX export failed (%s), keeping CatBoost model", e)
            return True  # CB model is saved

    except Exception as e:
        logger.error("Model save failed: %s", e)
        return False


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=== Gatekeeper Batch Trainer ===")
    logger.info("Reading from InfluxDB...")

    rows = fetch_from_influx()
    if not rows:
        logger.error("No data available — exiting")
        sys.exit(1)

    X, y, cat_data = prepare_data(rows)
    logger.info("Prepared %d samples, label distribution: %d/%d (pos/neg)",
                len(X), int(y.sum()), len(y) - int(y.sum()))

    result = train_model(X, y, cat_data)
    if result is None:
        logger.error("Training failed or quality too low")
        sys.exit(1)

    model, auc, acc, importance = result
    success = export_onnx(model, auc, acc)

    if success:
        logger.info("=== Training complete: AUC=%.4f Acc=%.4f ===", auc, acc)
    else:
        logger.error("Export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
