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
    symbol: str = ""
    direction: str = "HOLD"
    optimal_sl_pct: float = 0.0
    optimal_tp_pct: float = 0.0
    is_salvageable: bool = False
    mae_pct: float = 0.0
    mfe_pct: float = 0.0
    trade_judge: str = "UNCERTAIN"
    weight: float = 1.0


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

    def add(self, vector: np.ndarray, pnl: float, regime: str, direction: str = "HOLD",
            symbol: str = "", mae_pct: float = 0.0, mfe_pct: float = 0.0,
            trade_judge: str = "UNCERTAIN", weight: float = 1.0) -> int:
        # Block extreme positive outliers (lucky wins) — allow extreme losses for learning
        if pnl > 0.50:
            return -1

        # Block near-zero noise
        if abs(pnl) < 0.001:
            return -1

        # Duplicate filter disabled — shadow trades cluster at same PnL (TP hits),
        # blocking 100% of entries. Vector similarity in FAISS already handles diversity.

        # Compute optimal SL/TP from MAE/MFE
        optimal_sl = abs(mae_pct) * 1.1 if mae_pct != 0 else 0.0
        optimal_tp = abs(mfe_pct) * 0.9 if mfe_pct != 0 else 0.0
        is_salvageable = optimal_tp > (optimal_sl * 1.5) if optimal_sl > 0 else False

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
            symbol=symbol,
            direction=direction.upper() if direction else "HOLD",
            optimal_sl_pct=optimal_sl,
            optimal_tp_pct=optimal_tp,
            is_salvageable=is_salvageable,
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
            trade_judge=trade_judge,
            weight=weight,
        ))
        return vid

    def _decay_weight(self, entry: MemoryEntry, current_regime: str) -> float:
        age_days = (time.time() - entry.timestamp) / 86400.0
        time_decay = np.exp(-age_days / max(self.decay_days, 1))
        regime_decay = 1.0 if entry.regime == current_regime else 0.3
        return float(time_decay * regime_decay)

    def query(self, vector: np.ndarray, current_regime: str, k: int = 10) -> Tuple[np.ndarray, dict]:
        if self.index.ntotal == 0:
            v = np.zeros(8, dtype=np.float32)
            v[0] = 0.5
            return v, {"win_rate": 0.5, "avg_pnl": 0.0, "matches": 0, "long_win_rate": 0.5, "short_win_rate": 0.5,
                       "long_avg_pnl": 0.0, "short_avg_pnl": 0.0, "long_matches": 0, "short_matches": 0}

        vec = vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(vec, k)

        weighted_pnl = 0.0
        weighted_wins = 0.0
        total_weight = 0.0
        long_wins = 0.0
        long_weight = 0.0
        long_pnl = 0.0
        short_wins = 0.0
        short_weight = 0.0
        short_pnl = 0.0
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            w = self._decay_weight(entry, current_regime) * max(float(dist), 0.01)
            weighted_pnl += entry.pnl * w
            weighted_wins += (1.0 if entry.won else 0.0) * w
            total_weight += w
            if entry.direction == "LONG":
                long_wins += (1.0 if entry.won else 0.0) * w
                long_weight += w
                long_pnl += entry.pnl * w
            elif entry.direction == "SHORT":
                short_wins += (1.0 if entry.won else 0.0) * w
                short_weight += w
                short_pnl += entry.pnl * w

        if total_weight <= 0:
            v = np.zeros(8, dtype=np.float32)
            v[0] = 0.5
            return v, {"win_rate": 0.5, "avg_pnl": 0.0, "matches": 0, "long_win_rate": 0.5, "short_win_rate": 0.5,
                       "long_avg_pnl": 0.0, "short_avg_pnl": 0.0, "long_matches": 0, "short_matches": 0}

        win_rate = weighted_wins / total_weight
        avg_pnl = weighted_pnl / total_weight
        long_win_rate = long_wins / max(long_weight, 1e-8) if long_weight > 0 else 0.5
        short_win_rate = short_wins / max(short_weight, 1e-8) if short_weight > 0 else 0.5
        long_avg_pnl = long_pnl / max(long_weight, 1e-8) if long_weight > 0 else 0.0
        short_avg_pnl = short_pnl / max(short_weight, 1e-8) if short_weight > 0 else 0.0
        long_matches = sum(1 for dist, idx in zip(distances[0], indices[0])
                          if 0 <= idx < len(self.metadata) and self.metadata[idx].direction == "LONG")
        short_matches = sum(1 for dist, idx in zip(distances[0], indices[0])
                           if 0 <= idx < len(self.metadata) and self.metadata[idx].direction == "SHORT")
        v_memory = np.array([
            win_rate, avg_pnl,
            float(np.tanh(avg_pnl / 100)),
            float(total_weight),
            float(k),
            1.0 if current_regime == "Trending" else 0.0,
            1.0 if current_regime == "Breakout" else 0.0,
            1.0 if current_regime == "Choppy" else 0.0,
        ], dtype=np.float32)
        return v_memory, {
            "win_rate": win_rate, "avg_pnl": avg_pnl, "matches": long_matches + short_matches,
            "long_win_rate": long_win_rate, "short_win_rate": short_win_rate,
            "long_avg_pnl": long_avg_pnl, "short_avg_pnl": short_avg_pnl,
            "long_matches": long_matches, "short_matches": short_matches,
        }

    @property
    def size(self) -> int:
        return self.index.ntotal

    def query_with_metadata(self, vector: np.ndarray, current_regime: str, k: int = 10, symbol: str = "") -> Tuple[np.ndarray, dict]:
        """Query FAISS and return v_memory + per-neighbor metadata (optimal SL/TP, salvageable)."""
        if self.index.ntotal == 0:
            v = np.zeros(8, dtype=np.float32)
            v[0] = 0.5
            return v, {"neighbors": [], "salvageable_count": 0, "unsalvageable_count": 0,
                       "dynamic_sl_pct": 0.0, "dynamic_tp_pct": 0.0}

        vec = vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(vec, k)

        neighbors = []
        salvageable_count = 0
        unsalvageable_count = 0
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            w = self._decay_weight(entry, current_regime) * max(float(dist), 0.01)
            if entry.is_salvageable:
                salvageable_count += 1
            else:
                unsalvageable_count += 1
            neighbors.append({
                "symbol": entry.symbol,
                "direction": entry.direction,
                "pnl": entry.pnl,
                "optimal_sl_pct": entry.optimal_sl_pct,
                "optimal_tp_pct": entry.optimal_tp_pct,
                "is_salvageable": entry.is_salvageable,
                "weight": w,
                "regime": entry.regime,
                "mae_pct": entry.mae_pct,
                "mfe_pct": entry.mfe_pct,
            })

        v_memory, info = self.query(vector, current_regime, k)
        info["neighbors"] = neighbors
        info["salvageable_count"] = salvageable_count
        info["unsalvageable_count"] = unsalvageable_count

        # Compute dynamic SL/TP from salvageable neighbors OF SAME SYMBOL
        salvageable = [n for n in neighbors if n["is_salvageable"]]
        same_symbol_salvageable = [n for n in salvageable if n["symbol"] == symbol]
        if same_symbol_salvageable:
            avg_optimal_sl = sum(n["optimal_sl_pct"] for n in same_symbol_salvageable) / len(same_symbol_salvageable)
            avg_optimal_tp = sum(n["optimal_tp_pct"] for n in same_symbol_salvageable) / len(same_symbol_salvageable)
            info["dynamic_sl_pct"] = avg_optimal_sl
            info["dynamic_tp_pct"] = avg_optimal_tp
        else:
            info["dynamic_sl_pct"] = 0.0
            info["dynamic_tp_pct"] = 0.0

        return v_memory, info
