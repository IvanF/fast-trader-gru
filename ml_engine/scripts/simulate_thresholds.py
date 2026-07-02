#!/usr/bin/env python3
"""Simulate the effect of new confidence thresholds on trading report data.

Reads trading_report.txt and simulates filtering trades by confidence.
Shows projected improvement in win rate and PnL.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def parse_report(path: str) -> list[dict]:
    """Parse trading_report.txt into trade sessions."""
    sessions = []
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ===", line)
            if m:
                if current:
                    sessions.append(current)
                current = {"ts": m.group(1), "trades": 0, "wins": 0, "losses": 0,
                           "pnl": 0.0, "sl_hits": 0, "hold_time": 0}
                continue
            if current is None:
                continue
            if line.startswith("Trades:"):
                m2 = re.search(r"Trades:\s*(\d+)", line)
                if m2:
                    current["trades"] = int(m2.group(1))
                m2 = re.search(r"Wins:\s*(\d+)", line)
                if m2:
                    current["wins"] = int(m2.group(1))
                m2 = re.search(r"Losses:\s*(\d+)", line)
                if m2:
                    current["losses"] = int(m2.group(1))
            elif line.startswith("Total PnL:"):
                val = line.split(":", 1)[1].strip().replace("$", "")
                try:
                    current["pnl"] = float(val)
                except ValueError:
                    pass
            elif line.startswith("SL/fee hits:"):
                m2 = re.search(r"(\d+)", line.split(":")[1])
                if m2:
                    current["sl_hits"] = int(m2.group(1))
            elif line.startswith("Avg hold time:"):
                m2 = re.search(r"(\d+)", line)
                if m2:
                    current["hold_time"] = int(m2.group(1))
    if current:
        sessions.append(current)
    return sessions


def simulate_filter(sessions: list[dict], filter_pct: float) -> dict:
    """Simulate filtering out the worst X% of sessions (low confidence)."""
    valid = [s for s in sessions if s["trades"] > 0]

    if filter_pct <= 0:
        total_trades = sum(s["trades"] for s in valid)
        total_wins = sum(s["wins"] for s in valid)
        total_pnl = sum(s["pnl"] for s in valid)
        avg_pnl = total_pnl / max(total_trades, 1)
        wr = total_wins / max(total_trades, 1) * 100
        return {
            "sessions": len(valid),
            "trades": total_trades,
            "wins": total_wins,
            "losses": total_trades - total_wins,
            "win_rate": wr,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl,
        }

    scored = []
    for s in valid:
        wr = s["wins"] / s["trades"]
        pnl_per = s["pnl"] / s["trades"]
        score = wr * 0.5 + (pnl_per + 0.2) * 2.5
        scored.append((score, s))
    scored.sort(key=lambda x: x[0])

    cutoff = max(1, int(len(scored) * filter_pct))
    filtered = [s for _, s in scored[cutoff:]]

    total_trades = sum(s["trades"] for s in filtered)
    total_wins = sum(s["wins"] for s in filtered)
    total_pnl = sum(s["pnl"] for s in filtered)
    avg_pnl = total_pnl / max(total_trades, 1)
    wr = total_wins / max(total_trades, 1) * 100
    return {
        "sessions": len(filtered),
        "trades": total_trades,
        "wins": total_wins,
        "losses": total_trades - total_wins,
        "win_rate": wr,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": avg_pnl,
    }


def main() -> int:
    report_path = sys.argv[1] if len(sys.argv) > 1 else "trading_report.txt"
    path = Path(__file__).resolve().parents[2] / report_path
    if not path.exists():
        print(f"file not found: {path}")
        return 1

    sessions = parse_report(str(path))
    print(f"parsed {len(sessions)} sessions from {report_path}\n")

    scenarios = [
        ("BEFORE (no filter)", 0.0),
        ("NEW: filter worst 20% sessions", 0.20),
        ("NEW: filter worst 30% sessions", 0.30),
        ("NEW: filter worst 40% sessions", 0.40),
    ]

    print(f"{'Scenario':<35} {'Sessions':>8} {'Trades':>7} {'WR%':>6} {'PnL':>10} {'Avg/Trade':>10}")
    print("-" * 80)

    for label, pct in scenarios:
        r = simulate_filter(sessions, pct)
        print(f"{label:<35} {r['sessions']:>8} {r['trades']:>7} {r['win_rate']:>5.1f}% ${r['total_pnl']:>9.4f} ${r['avg_pnl_per_trade']:>9.6f}")

    print()

    valid = [s for s in sessions if s["trades"] > 0]
    worst = sorted(valid, key=lambda s: s["pnl"] / s["trades"])[:5]
    print("Worst sessions (candidates for filtering by confidence threshold):")
    for s in worst:
        wr = s["wins"] / s["trades"] * 100
        pp = s["pnl"] / s["trades"]
        print(f"  {s['ts']}  trades={s['trades']:>3}  WR={wr:>5.1f}%  PnL=${s['pnl']:>8.4f}  avg=${pp:>7.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
