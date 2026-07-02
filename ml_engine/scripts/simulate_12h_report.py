#!/usr/bin/env python3
"""Simulate the effect of new confidence thresholds on 12h trading report data."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def parse_report(path: str) -> list[dict]:
    sessions = []
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ===", line)
            if m:
                if current:
                    sessions.append(current)
                current = {"ts": m.group(1), "trades": 0, "wr": 0.0, "pnl": 0.0,
                           "avg_win": 0.0, "avg_loss": 0.0, "wl_ratio": 0.0}
                continue
            if current is None:
                continue
            m2 = re.match(r"Trades:\s*(\d+)\s*\|\s*WR:\s*([\d.]+)%\s*\|\s*PnL:\s*\$([-\d.]+)", line)
            if m2:
                current["trades"] = int(m2.group(1))
                current["wr"] = float(m2.group(2))
                current["pnl"] = float(m2.group(3))
            m2 = re.match(r"Avg win:\s*\$([-\d.]+)\s*\|\s*Avg loss:\s*\$([-\d.]+)", line)
            if m2:
                current["avg_win"] = float(m2.group(1))
                current["avg_loss"] = float(m2.group(2))
            m2 = re.match(r"W/L ratio:\s*([\d.]+)", line)
            if m2:
                current["wl_ratio"] = float(m2.group(1))
    if current:
        sessions.append(current)
    return sessions


def simulate_filter(sessions: list[dict], filter_pct: float) -> dict:
    valid = [s for s in sessions if s["trades"] > 0]
    if not valid:
        return {"sessions": 0, "trades": 0, "wr": 0, "pnl": 0, "wl_ratio": 0}

    if filter_pct <= 0:
        total = sum(s["trades"] for s in valid)
        pnl = sum(s["pnl"] for s in valid)
        weighted_wr = sum(s["wr"] * s["trades"] for s in valid) / max(total, 1)
        return {"sessions": len(valid), "trades": total, "wr": weighted_wr, "pnl": pnl, "wl_ratio": 0}

    scored = []
    for s in valid:
        score = s["wr"] * 0.3 + (s["pnl"] / s["trades"] + 0.2) * 5 + s["wl_ratio"] * 10
        scored.append((score, s))
    scored.sort(key=lambda x: x[0])

    cutoff = max(1, int(len(scored) * filter_pct))
    filtered = [s for _, s in scored[cutoff:]]

    total = sum(s["trades"] for s in filtered)
    pnl = sum(s["pnl"] for s in filtered)
    weighted_wr = sum(s["wr"] * s["trades"] for s in filtered) / max(total, 1)
    return {"sessions": len(filtered), "trades": total, "wr": weighted_wr, "pnl": pnl, "wl_ratio": 0}


def main() -> int:
    report_path = sys.argv[1] if len(sys.argv) > 1 else "trading_report_12h.txt"
    path = Path(__file__).resolve().parents[2] / report_path
    if not path.exists():
        print(f"file not found: {path}")
        return 1

    sessions = parse_report(str(path))
    print(f"parsed {len(sessions)} sessions from {report_path}\n")

    scenarios = [
        ("BEFORE (no filter)", 0.0),
        ("NEW: filter worst 20%", 0.20),
        ("NEW: filter worst 30%", 0.30),
        ("NEW: filter worst 40%", 0.40),
    ]

    print(f"{'Scenario':<30} {'Sessions':>8} {'Trades':>7} {'WR%':>6} {'PnL':>10}")
    print("-" * 65)

    for label, pct in scenarios:
        r = simulate_filter(sessions, pct)
        print(f"{label:<30} {r['sessions']:>8} {r['trades']:>7} {r['wr']:>5.1f}% ${r['pnl']:>9.4f}")

    print()
    valid = [s for s in sessions if s["trades"] > 0]
    worst = sorted(valid, key=lambda s: s["pnl"] / s["trades"])[:5]
    print("Worst sessions:")
    for s in worst:
        pp = s["pnl"] / s["trades"]
        print(f"  {s['ts']}  trades={s['trades']:>3}  WR={s['wr']:>5.1f}%  PnL=${s['pnl']:>8.4f}  avg=${pp:>7.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
