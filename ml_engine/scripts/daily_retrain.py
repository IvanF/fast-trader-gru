#!/usr/bin/env python3
"""Daily rolling retrain — delegates to train.py with worker defaults."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True)
    p.add_argument("--hours", type=int, default=int(__import__("os").getenv("RETRAIN_LOOKBACK_HOURS", "6")))
    p.add_argument("--epochs", type=int, default=int(__import__("os").getenv("RETRAIN_EPOCHS", "8")))
    args = p.parse_args()

    scripts = Path(__file__).parent
    cmd = [
        "nice", "-n", str(int(__import__("os").getenv("TRAIN_NICE_LEVEL", "15"))),
        sys.executable,
        str(scripts / "train.py"),
        "--symbols", args.symbols,
        "--hours", str(args.hours),
        "--epochs", str(args.epochs),
        "--incremental",
        "--trigger", "daily_cron",
        "--device", __import__("os").getenv("TRAIN_DEVICE", "cuda"),
    ]
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        subprocess.check_call(cmd[2:])


if __name__ == "__main__":
    main()
