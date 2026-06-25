#!/usr/bin/env python3
"""Clean FAISS: remove outliers (pnl > 0.15 or pnl < -0.05), balance LONG/SHORT at 2:1."""

import json
import os
import sys

import faiss
import numpy as np

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/app/data/faiss_index")
MAX_IMBALANCE = 2.0
MAX_PNL = 0.15
MIN_PNL = -0.05


def main():
    idx_file = f"{INDEX_PATH}.faiss"
    meta_file = f"{INDEX_PATH}.meta.json"

    with open(meta_file) as f:
        meta = json.load(f)

    entries = meta if isinstance(meta, list) else meta.get("entries", [])
    print(f"loaded {len(entries)} entries")

    keep = []
    removed = []
    for e in entries:
        pnl = e.get("pnl", 0)
        if pnl > MAX_PNL:
            removed.append(e)
            continue
        if pnl < MIN_PNL:
            removed.append(e)
            continue
        keep.append(e)

    print(f"removed {len(removed)} outliers (pnl > ${MAX_PNL} or < ${MIN_PNL})")
    for e in removed:
        print(f"  {e.get('direction','?'):6s} pnl=${e.get('pnl',0):.4f} regime={e.get('regime','?')}")

    long_entries = [e for e in keep if e.get("direction") == "LONG"]
    short_entries = [e for e in keep if e.get("direction") == "SHORT"]

    print(f"\nafter cleanup: LONG={len(long_entries)} SHORT={len(short_entries)}")

    max_count = max(len(long_entries), len(short_entries), 1)
    min_count = min(len(long_entries), len(short_entries))
    cap = max(min_count, int(min_count * MAX_IMBALANCE)) if min_count > 0 else max_count

    if len(long_entries) > cap:
        long_entries.sort(key=lambda e: abs(e.get("pnl", 0)))
        long_entries = long_entries[:cap]
    if len(short_entries) > cap:
        short_entries.sort(key=lambda e: abs(e.get("pnl", 0)))
        short_entries = short_entries[:cap]

    balanced = long_entries + short_entries
    print(f"after balance: LONG={len(long_entries)} SHORT={len(short_entries)}")

    old_index = faiss.read_index(idx_file)
    dim = old_index.d
    new_index = faiss.IndexFlatIP(dim)

    for i, e in enumerate(balanced):
        e["vector_id"] = i

    vecs = []
    for i in range(old_index.ntotal):
        vecs.append(old_index.reconstruct(i))
    vecs = np.array(vecs, dtype=np.float32)

    kept_ids = set()
    for i, e in enumerate(entries):
        if e in balanced:
            kept_ids.add(i)

    kept_vecs = np.array([vecs[i] for i in sorted(kept_ids)], dtype=np.float32)
    if len(kept_vecs) > 0:
        faiss.normalize_L2(kept_vecs)
        new_index.add(kept_vecs)

    faiss.write_index(new_index, idx_file)
    with open(meta_file, "w") as f:
        json.dump(balanced, f)

    print(f"\nrebuilt FAISS: {new_index.ntotal} entries (was {old_index.ntotal})")

    # Print final stats
    lp = [e.get("pnl", 0) for e in balanced if e.get("direction") == "LONG"]
    sp = [e.get("pnl", 0) for e in balanced if e.get("direction") == "SHORT"]
    if lp:
        lw = sum(1 for p in lp if p > 0)
        print(f"LONG:  {len(lp)} entries, WR={lw*100/len(lp):.0f}%, avg_pnl=${sum(lp)/len(lp):.4f}")
    if sp:
        sw = sum(1 for p in sp if p > 0)
        print(f"SHORT: {len(sp)} entries, WR={sw*100/len(sp):.0f}%, avg_pnl=${sum(sp)/len(sp):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
