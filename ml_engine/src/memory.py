"""FAISS vector memory with time-decay and regime-aware forgetting."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import faiss
import numpy as np


@dataclass
class MemoryEntry:
    vector_id: int
    pnl: float
    timestamp: float
    regime: str
    won: bool


class ExperienceEngine:
    def __init__(self, dim: int, index_path: str, decay_days: int = 14) -> None:
        self.dim = dim
        self.index_path = index_path
        self.decay_days = decay_days
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[MemoryEntry] = []
        self._load()

    def _load(self) -> None:
        idx_file = f"{self.index_path}.faiss"
        meta_file = f"{self.index_path}.meta.json"
        if os.path.exists(idx_file):
            self.index = faiss.read_index(idx_file)
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.metadata = [MemoryEntry(**e) for e in raw]

    def persist(self) -> None:
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}.meta.json", "w", encoding="utf-8") as f:
            json.dump([e.__dict__ for e in self.metadata], f)

    def add(self, vector: np.ndarray, pnl: float, regime: str) -> int:
        vec = vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        vid = self.index.ntotal
        self.index.add(vec)
        self.metadata.append(MemoryEntry(
            vector_id=vid,
            pnl=pnl,
            timestamp=time.time(),
            regime=regime,
            won=pnl >= 0,
        ))
        return vid

    def _decay_weight(self, entry: MemoryEntry, current_regime: str) -> float:
        age_days = (time.time() - entry.timestamp) / 86400.0
        time_decay = np.exp(-age_days / max(self.decay_days, 1))
        regime_decay = 1.0 if entry.regime == current_regime else 0.3
        return float(time_decay * regime_decay)

    def query(self, vector: np.ndarray, current_regime: str, k: int = 10) -> Tuple[np.ndarray, dict]:
        if self.index.ntotal == 0:
            return np.zeros(8, dtype=np.float32), {"win_rate": 0.5, "avg_pnl": 0.0, "matches": 0}

        vec = vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(vec, k)

        weighted_pnl = 0.0
        weighted_wins = 0.0
        total_weight = 0.0
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            w = self._decay_weight(entry, current_regime) * max(float(dist), 0.01)
            weighted_pnl += entry.pnl * w
            weighted_wins += (1.0 if entry.won else 0.0) * w
            total_weight += w

        if total_weight <= 0:
            return np.zeros(8, dtype=np.float32), {"win_rate": 0.5, "avg_pnl": 0.0, "matches": 0}

        win_rate = weighted_wins / total_weight
        avg_pnl = weighted_pnl / total_weight
        v_memory = np.array([
            win_rate, avg_pnl,
            float(np.tanh(avg_pnl / 100)),
            float(total_weight),
            float(k),
            1.0 if current_regime == "Trending" else 0.0,
            1.0 if current_regime == "Breakout" else 0.0,
            1.0 if current_regime == "Choppy" else 0.0,
        ], dtype=np.float32)
        return v_memory, {"win_rate": win_rate, "avg_pnl": avg_pnl, "matches": int(k)}

    @property
    def size(self) -> int:
        return self.index.ntotal
