#!/usr/bin/env python3
"""
Query InfluxDB 2.x for offline training datasets with real PnL join.

Joins trade_outcomes (from execution:results) with market microstructure
features in a ±300s window around each trade close.

Usage:
    python scripts/query_influx_training_data.py \
        --symbol BTCUSDT \
        --start -30d \
        --output /app/data/training/btc_joined.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.influx_join import build_joined_dataset
from src.influx_store import InfluxStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export joined InfluxDB training datasets")
    p.add_argument("--symbol", required=True, help="Trading symbol, e.g. BTCUSDT")
    p.add_argument("--start", default="-30d", help="Flux range start, e.g. -7d, -30d")
    p.add_argument("--stop", default="now()", help="Flux range stop")
    p.add_argument("--output", default=None, help="Output .npz path")
    p.add_argument("--window", type=int, default=300, help="Feature join window in seconds")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv("INFLUX_TOKEN", "")
    if not token:
        print("INFLUX_TOKEN environment variable is required", file=sys.stderr)
        sys.exit(1)

    store = InfluxStore(
        os.getenv("INFLUX_URL", "http://influxdb:8086"),
        token,
        os.getenv("INFLUX_ORG", "fasttrader"),
        os.getenv("INFLUX_BUCKET_RAW", "market_raw"),
        os.getenv("INFLUX_BUCKET_FEATURES", "market_features"),
    )

    start = args.start if args.start.startswith("-") else f'time(v: "{args.start}")'
    stop = args.stop if args.stop == "now()" else f'time(v: "{args.stop}")'

    data = build_joined_dataset(store, start, stop, args.symbol, args.window)
    store.close()

    out = args.output
    if out is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        out = f"/app/data/training/{args.symbol}_joined_{ts}.npz"

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    n = len(data["direction"])
    print(f"exported {n} joined samples -> {out}")
    if n > 0:
        print(f"  direction distribution: LONG={np.sum(data['direction']==0)} SHORT={np.sum(data['direction']==1)} HOLD={np.sum(data['direction']==2)}")
        print(f"  mean pnl: {data['pnl'].mean():.4f}")


if __name__ == "__main__":
    main()
