#!/usr/bin/env python3
"""Clean FAISS: remove outliers, deduplicate, balance LONG/SHORT at 2:1."""

import json
import os
import sys
from collections import defaultdict

import faiss
import numpy as np

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/app/data/faiss_index")
MAX_IMBALANCE = 2.0
MAX_PNL = 0.15
MIN_PNL = -0.50


def main():
    idx_file = f"{INDEX_PATH}.faiss"
    meta_file = f"{INDEX_PATH}.meta.json"

    with open(meta_file) as f:
        meta = json.load(f)

    entries = meta if isinstance(meta, list) else meta.get("entries", [])
    print(f"loaded {len(entries)} entries")

    # Step 1: Remove outliers
    keep = []
    for e in entries:
        pnl = e.get("pnl", 0)
        if pnl > MAX_PNL:
            continue
        if pnl < MIN_PNL:
            continue
        keep.append(e)
    print(f"after outlier removal: {len(keep)} (removed {len(entries)-len(keep)})")

    # Step 2: Deduplicate — keep best PnL per unique (direction, regime) group
    groups = defaultdict(list)
    for e in keep:
        key = (e.get("direction", ""), e.get("regime", ""), round(e.get("pnl", 0), 4))
        groups[key].append(e)

    deduped = []
    removed_dupes = 0
    for key, group in groups.items():
        # Keep the one with best pnl
        best = max(group, key=lambda e: e.get("pnl", 0))
        deduped.append(best)
        if len(group) > 1:
            removed_dupes += len(group) - 1

    print(f"after dedup: {len(deduped)} (removed {removed_dupes} duplicates)")
    keep = deduped

    # Step 3: Balance LONG/SHORT
    long_entries = [e for e in keep if e.get("direction") == "LONG"]
    short_entries = [e for e in keep if e.get("direction") == "SHORT"]

    max_count = max(len(long_entries), len(short_entries), 1)
    min_count = min(len(long_entries), len(short_entries))
    cap = max(min_count, int(min_count * MAX_IMBALANCE)) if min_count > 0 else max_count

    if len(long_entries) > cap:
        long_entries.sort(key=lambda e: e.get("pnl", 0), reverse=True)
        long_entries = long_entries[:cap]
    if len(short_entries) > cap:
        short_entries.sort(key=lambda e: e.get("pnl", 0), reverse=True)
        short_entries = short_entries[:cap]

    balanced = long_entries + short_entries
    print(f"after balance: LONG={len(long_entries)} SHORT={len(short_entries)}")

    # Rebuild FAISS
    old = faiss.read_index(idx_file)
    new = faiss.IndexFlatIP(old.d)

    for i, e in enumerate(balanced):
        e["vector_id"] = i

    kept_ids = set()
    for i, e in enumerate(entries):
        if e in balanced:
            kept_ids.add(i)

    vecs = np.array([old.reconstruct(i) for i in range(old.ntotal)], dtype=np.float32)
    kept_vecs = np.array([vecs[i] for i in sorted(kept_ids)], dtype=np.float32)
    faiss.normalize_L2(kept_vecs)
    new.add(kept_vecs)

    faiss.write_index(new, idx_file)
    with open(meta_file, "w") as f:
        json.dump(balanced, f)

    lp = [e.get("pnl", 0) for e in long_entries]
    sp = [e.get("pnl", 0) for e in short_entries]
    print(f"\nResult: {new.ntotal} entries (was {old.ntotal})")
    if lp:
        lw = sum(1 for p in lp if p > 0)
        print(f"LONG:  {len(lp)} entries, WR={lw*100/len(lp):.0f}%, avg_pnl=${sum(lp)/len(lp):.4f}")
    if sp:
        sw = sum(1 for p in sp if p > 0)
        print(f"SHORT: {len(sp)} entries, WR={sw*100/len(sp):.0f}%, avg_pnl=${sum(sp)/len(sp):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
