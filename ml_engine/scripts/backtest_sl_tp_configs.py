#!/usr/bin/env python3
"""Backtest SL/TP optimization using FAISS metadata (MAE/MFE).

Simulates trades with:
1. OLD SL/TP: SL=1.2%, TP=0.8% (R:R = 0.67:1 — unprofitable)
2. NEW SL/TP: SL=0.8%, TP=1.5% (R:R = 1.875:1 — profitable)

Uses MAE/MFE data to determine if trade would have hit SL or TP.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def load_faiss_metadata(faiss_path: str) -> list[dict]:
    meta_file = f"{faiss_path}.meta.json"
    if not os.path.exists(meta_file):
        return []
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def simulate_trade(
    direction: str,
    sl_pct: float,
    tp_pct: float,
    mae_pct: float,
    mfe_pct: float,
    fee_pct: float = 0.00075,
) -> dict:
    abs_mae = abs(mae_pct)
    abs_mfe = abs(mfe_pct)

    hit_sl = abs_mae >= sl_pct
    hit_tp = abs_mfe >= tp_pct

    if hit_sl and hit_tp:
        if abs_mae < abs_mfe:
            hit_tp, hit_sl = True, False
        else:
            hit_sl, hit_tp = True, False

    if hit_tp:
        pnl_pct = tp_pct - fee_pct
    elif hit_sl:
        pnl_pct = -sl_pct - fee_pct
    else:
        if abs_mfe > 0:
            exit_pct = abs_mfe * 0.5
        elif abs_mae > 0:
            exit_pct = -abs_mae * 0.5
        else:
            exit_pct = 0.0
        pnl_pct = exit_pct - fee_pct

    return {"pnl_pct": pnl_pct, "hit_sl": hit_sl, "hit_tp": hit_tp}


def main() -> int:
    faiss_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("FAISS_PATH", "/app/data/faiss_index")
    metadata = load_faiss_metadata(faiss_path)
    if not metadata:
        print(f"no metadata found at {faiss_path}.meta.json")
        return 1

    print(f"loaded {len(metadata)} FAISS entries\n")

    configs = [
        ("OLD: SL=1.2% TP=0.8% (R:R=0.67:1)", 0.012, 0.008),
        ("OLD_TIGHT: SL=0.8% TP=0.8% (R:R=1:1)", 0.008, 0.008),
        ("NEW: SL=0.8% TP=1.5% (R:R=1.875:1)", 0.008, 0.015),
        ("NEW_CONSERV: SL=0.6% TP=1.2% (R:R=2:1)", 0.006, 0.012),
        ("NEW_AGGRESSIVE: SL=0.5% TP=1.5% (R:R=3:1)", 0.005, 0.015),
    ]

    print(f"{'Config':<45} {'Trades':>7} {'WR%':>6} {'PnL%':>8} {'SL%':>5} {'TP%':>5} {'PF':>6}")
    print("-" * 90)

    for label, sl, tp in configs:
        pnls = []
        sl_count = 0
        tp_count = 0
        for meta in metadata:
            direction = meta.get("direction", "HOLD")
            if direction not in ("LONG", "SHORT"):
                continue
            mae = abs(meta.get("mae_pct", 0))
            mfe = abs(meta.get("mfe_pct", 0))
            if mae < 0.0001 and mfe < 0.0001:
                continue

            result = simulate_trade(direction, sl, tp, mae, mfe)
            pnls.append(result["pnl_pct"])
            if result["hit_sl"]:
                sl_count += 1
            if result["hit_tp"]:
                tp_count += 1

        if not pnls:
            print(f"{label:<45} {'N/A':>7}")
            continue

        arr = np.array(pnls)
        n = len(arr)
        wr = (arr > 0).sum() / n * 100
        total = arr.sum()
        wins = arr[arr > 0].sum()
        losses = abs(arr[arr < 0].sum())
        pf = wins / losses if losses > 0 else 0
        sl_pct = sl_count / n * 100
        tp_pct = tp_count / n * 100

        print(f"{label:<45} {n:>7} {wr:>5.1f}% {total:>7.4f}% {sl_pct:>4.1f}% {tp_pct:>4.1f}% {pf:>5.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
