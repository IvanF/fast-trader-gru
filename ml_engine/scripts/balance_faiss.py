#!/usr/bin/env python3
"""Rebalance FAISS index: remove entries with pnl < -0.05, cap imbalance at 2:1."""

import json
import os
import sys
import time

import faiss
import numpy as np

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/app/data/faiss_index")
MAX_IMBALANCE = 2.0


def main():
    idx_file = f"{INDEX_PATH}.faiss"
    meta_file = f"{INDEX_PATH}.meta.json"

    if not os.path.exists(meta_file):
        print("no FAISS metadata found", file=sys.stderr)
        return 1

    with open(meta_file) as f:
        meta = json.load(f)

    entries = meta if isinstance(meta, list) else meta.get("entries", [])
    print(f"loaded {len(entries)} entries")

    keep = []
    for e in entries:
        pnl = e.get("pnl", 0)
        if pnl < -0.05:
            continue
        keep.append(e)

    long_entries = [e for e in keep if e.get("direction") == "LONG"]
    short_entries = [e for e in keep if e.get("direction") == "SHORT"]
    other_entries = [e for e in keep if e.get("direction") not in ("LONG", "SHORT")]

    print(f"after pnl filter: LONG={len(long_entries)} SHORT={len(short_entries)} other={len(other_entries)}")

    max_count = max(len(long_entries), len(short_entries), 1)
    min_count = min(len(long_entries), len(short_entries))
    cap = max(min_count, int(min_count * MAX_IMBALANCE)) if min_count > 0 else max_count

    if len(long_entries) > cap:
        long_entries.sort(key=lambda e: e.get("pnl", 0), reverse=True)
        long_entries = long_entries[:cap]
    if len(short_entries) > cap:
        short_entries.sort(key=lambda e: e.get("pnl", 0), reverse=True)
        short_entries = short_entries[:cap]

    balanced = long_entries + short_entries + other_entries
    print(f"after balance cap: LONG={len(long_entries)} SHORT={len(short_entries)}")

    old_index = faiss.read_index(idx_file)
    dim = old_index.d
    new_index = faiss.IndexFlatIP(dim)

    for i, e in enumerate(balanced):
        e["vector_id"] = i

    vecs = []
    for i in range(old_index.ntotal):
        vec = old_index.reconstruct(i)
        vecs.append(vec)
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

    print(f"rebuilt FAISS: {new_index.ntotal} entries (was {old_index.ntotal})")
    print(f"LONG={len(long_entries)} SHORT={len(short_entries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
