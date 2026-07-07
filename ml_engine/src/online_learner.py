"""Online learning with EWC (Elastic Weight Consolidation) and pattern memory.

Provides lightweight per-trade weight updates without full retraining.
Protects against catastrophic forgetting via Fisher information regularization
and a small replay buffer of past trades.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ONLINE_LR = float(os.getenv("ONLINE_LR", "5e-6"))
EWC_LAMBDA = float(os.getenv("EWC_LAMBDA", "1000.0"))
REPLAY_BUFFER_SIZE = int(os.getenv("REPLAY_BUFFER_SIZE", "200"))
REPLAY_BATCH = int(os.getenv("REPLAY_BATCH", "16"))
PATTERN_MEMORY_SIZE = int(os.getenv("PATTERN_MEMORY_SIZE", "5000"))
PATTERN_SIMILARITY_THRESHOLD = float(os.getenv("PATTERN_SIMILARITY_THRESHOLD", "0.85"))
PATTERN_TTL_HOURS = float(os.getenv("PATTERN_TTL_HOURS", "1"))
CONSECUTIVE_LOSS_THRESHOLD = int(os.getenv("CONSECUTIVE_LOSS_THRESHOLD", "10"))
ONLINE_UPDATE_INTERVAL = int(os.getenv("ONLINE_UPDATE_INTERVAL", "5"))


@dataclass
class TradeSample:
    v_state: np.ndarray
    v_memory: np.ndarray
    direction: int
    confidence: float
    pnl: float
    regime: str
    timestamp: float


@dataclass
class LosingPattern:
    v_state: np.ndarray
    regime: str
    direction: str
    pnl: float
    timestamp: float
    symbol: str = ""
    reason: str = ""


class PatternMemory:
    """Stores specific losing patterns for pattern-based avoidance."""

    def __init__(self, max_size: int = PATTERN_MEMORY_SIZE) -> None:
        self.max_size = max_size
        self.patterns: list[LosingPattern] = []
        self._lock = threading.Lock()
        self._block_timestamps: dict[str, list[float]] = {}

    def add(self, pattern: LosingPattern) -> None:
        with self._lock:
            if len(self.patterns) >= self.max_size:
                self.patterns.pop(0)
            self.patterns.append(pattern)

    def find_similar(
        self, v_state: np.ndarray, regime: str, direction: str, k: int = 5
    ) -> list[LosingPattern]:
        """Find similar losing patterns by cosine similarity."""
        with self._lock:
            if not self.patterns:
                return []

            query = v_state.astype(np.float32)
            norm_q = np.linalg.norm(query)
            if norm_q < 1e-8:
                return []
            query = query / norm_q

            scored = []
            for p in self.patterns:
                if p.direction != direction:
                    continue
                norm_p = np.linalg.norm(p.v_state)
                if norm_p < 1e-8:
                    continue
                sim = float(np.dot(query, p.v_state / norm_p))
                if sim >= PATTERN_SIMILARITY_THRESHOLD:
                    regime_bonus = 0.1 if p.regime == regime else 0.0
                    scored.append((sim + regime_bonus, p))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [p for _, p in scored[:k]]

    def should_avoid(self, v_state: np.ndarray, regime: str, direction: str, symbol: str = "") -> Tuple[bool, float]:
        """Check if a trade pattern should be avoided based on historical losses.

        Block if same symbol has 3+ losses, or same symbol has 3+ losses in similar
        state-vector patterns with avg loss < -$0.05, or any single catastrophic loss.
        """
        now = time.time()

        similar = self.find_similar(v_state, regime, direction)

        with self._lock:
            if PATTERN_TTL_HOURS > 0:
                cutoff = now - PATTERN_TTL_HOURS * 3600
                similar = [p for p in similar if p.timestamp > cutoff]

            if not similar:
                return False, 0.0

            # Symbol-level: 3+ losses of same symbol → block
            if symbol:
                symbol_losses = [p for p in similar if p.symbol == symbol and p.pnl < 0]
                if len(symbol_losses) >= 3:
                    return True, np.mean([p.pnl for p in symbol_losses])

            # Single catastrophic loss on THIS symbol → block
            for p in similar:
                if p.pnl < -0.50 and p.symbol == symbol:
                    return True, p.pnl

            # State-vector level: 3+ losses of SAME symbol with similar vector
            if symbol:
                symbol_similar = [p for p in similar if p.symbol == symbol]
                loss_count = sum(1 for p in symbol_similar if p.pnl < 0)
                if loss_count >= 3:
                    avg_pnl = np.mean([p.pnl for p in symbol_similar])
                    if avg_pnl < -0.05:
                        return True, avg_pnl

            # Cross-symbol: only block if ANY 3+ losses with avg < -$0.10 (stricter)
            loss_count = sum(1 for p in similar if p.pnl < 0)
            if loss_count >= 3:
                avg_pnl = np.mean([p.pnl for p in similar])
                if avg_pnl < -0.10:
                    return True, avg_pnl

            return False, 0.0

    def persist(self, path: str) -> None:
        with self._lock:
            data = [
                {
                    "v_state": p.v_state.tolist(),
                    "regime": p.regime,
                    "direction": p.direction,
                    "pnl": p.pnl,
                    "timestamp": p.timestamp,
                    "symbol": p.symbol,
                    "reason": p.reason,
                }
                for p in self.patterns
            ]
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.patterns = [
                LosingPattern(
                    v_state=np.array(d["v_state"], dtype=np.float32),
                    regime=d["regime"],
                    direction=d["direction"],
                    pnl=d["pnl"],
                    timestamp=d["timestamp"],
                    symbol=d.get("symbol", ""),
                    reason=d.get("reason", ""),
                )
                for d in data
            ]
        except Exception as exc:
            logger.warning("failed to load pattern memory: %s", exc)


class ReplayBuffer:
    """Fixed-size replay buffer for online learning stability."""

    def __init__(self, max_size: int = REPLAY_BUFFER_SIZE) -> None:
        self.max_size = max_size
        self.buffer: deque[TradeSample] = deque(maxlen=max_size)

    def add(self, sample: TradeSample) -> None:
        self.buffer.append(sample)

    def sample(self, batch_size: int) -> Optional[list[TradeSample]]:
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class OnlineLearner:
    """Lightweight online learning with EWC regularization.

    Updates the DecisionMLP weights after each trade closure.
    Uses EWC to prevent catastrophic forgetting of past patterns.
    """

    def __init__(
        self,
        model_dir: str,
        state_dim: int = 128,
        memory_dim: int = 8,
        lr: float = ONLINE_LR,
        ewc_lambda: float = EWC_LAMBDA,
    ) -> None:
        self.model_dir = model_dir
        self.state_dim = state_dim
        self.memory_dim = memory_dim
        self.lr = lr
        self.ewc_lambda = ewc_lambda
        self._lock = threading.Lock()
        self._trades_since_update = 0
        self._total_updates = 0

        self.replay = ReplayBuffer()
        self.pattern_memory = PatternMemory()
        self._fisher: dict[str, torch.Tensor] = {}
        self._optimal_params: dict[str, torch.Tensor] = {}
        self._ewc_initialized = False

        pattern_path = os.path.join(model_dir, "..", "data", "pattern_memory.json")
        self._pattern_path = os.path.abspath(pattern_path)
        self.pattern_memory.load(self._pattern_path)

    def _load_decision_mlp(self) -> Optional[nn.Module]:
        """Load DecisionMLP from latest checkpoint."""
        ckpt_dir = Path(self.model_dir).parent / "data" / "checkpoints"
        if not ckpt_dir.exists():
            return None
        ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpts:
            return None
        try:
            from .models.nn_models import DecisionMLP
            state = torch.load(ckpts[0], map_location="cpu", weights_only=True)
            model = DecisionMLP(in_dim=self.state_dim + self.memory_dim)
            if "model" in state:
                mlp_keys = {k: v for k, v in state["model"].items() if k.startswith("decision.")}
                if mlp_keys:
                    stripped = {k.replace("decision.", ""): v for k, v in mlp_keys.items()}
                    model.load_state_dict(stripped, strict=False)
            return model
        except Exception as exc:
            logger.debug("failed to load DecisionMLP: %s", exc)
            return None

    def _init_ewc(self, model: nn.Module) -> None:
        """Compute Fisher information matrix for EWC regularization."""
        if self._ewc_initialized:
            return
        samples = self.replay.sample(min(32, len(self.replay)))
        if not samples:
            return

        model.eval()
        self._fisher = {}
        self._optimal_params = {}

        for name, param in model.named_parameters():
            self._optimal_params[name] = param.data.clone()
            self._fisher[name] = torch.zeros_like(param.data)

        for sample in samples:
            v_state = torch.tensor(sample.v_state, dtype=torch.float32).unsqueeze(0)
            v_memory = torch.tensor(sample.v_memory, dtype=torch.float32).unsqueeze(0)
            inp = torch.cat([v_state, v_memory], dim=-1)
            logits = model(inp)
            log_probs = F.log_softmax(logits[:, :3], dim=-1)
            target = torch.tensor([sample.direction], dtype=torch.long)
            loss = F.nll_loss(log_probs, target)
            model.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self._fisher[name] += param.grad.data.pow(2)

        for name in self._fisher:
            self._fisher[name] /= max(len(samples), 1)
        self._ewc_initialized = True

    def update(
        self,
        v_state: np.ndarray,
        v_memory: np.ndarray,
        direction: int,
        confidence: float,
        pnl: float,
        regime: str = "Choppy",
        symbol: str = "",
    ) -> Optional[float]:
        """Perform one online learning step after a trade closes.

        Returns the loss value, or None if no update was performed.
        """
        sample = TradeSample(
            v_state=v_state,
            v_memory=v_memory,
            direction=direction,
            confidence=confidence,
            pnl=pnl,
            regime=regime,
            timestamp=time.time(),
        )
        self.replay.add(sample)
        self._trades_since_update += 1

        if pnl < -0.01:
            self.pattern_memory.add(LosingPattern(
                v_state=v_state.copy(),
                regime=regime,
                direction=["LONG", "SHORT", "HOLD"][direction] if direction < 3 else "HOLD",
                pnl=pnl,
                timestamp=time.time(),
                symbol=symbol,
            ))

        if self._trades_since_update < ONLINE_UPDATE_INTERVAL:
            return None

        with self._lock:
            model = self._load_decision_mlp()
            if model is None:
                return None

            self._init_ewc(model)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            batch = self.replay.sample(REPLAY_BATCH)
            if not batch:
                return None

            v_state_b = torch.tensor(np.stack([s.v_state for s in batch]), dtype=torch.float32)
            v_memory_b = torch.tensor(np.stack([s.v_memory for s in batch]), dtype=torch.float32)
            direction_b = torch.tensor([s.direction for s in batch], dtype=torch.long)
            pnl_b = torch.tensor([s.pnl for s in batch], dtype=torch.float32)

            inp = torch.cat([v_state_b, v_memory_b], dim=-1)
            logits = model(inp)

            ce_loss = F.cross_entropy(logits[:, :3], direction_b, reduction="none")
            penalty = torch.where(pnl_b < 0, 1.5, 1.0)
            loss_cls = (ce_loss * penalty).mean()

            ewc_loss = torch.tensor(0.0)
            if self._ewc_initialized and self.ewc_lambda > 0:
                for name, param in model.named_parameters():
                    if name in self._fisher:
                        fisher = self._fisher[name]
                        optimal = self._optimal_params[name]
                        ewc_loss = ewc_loss + (fisher * (param - optimal).pow(2)).sum()
                ewc_loss = ewc_loss * self.ewc_lambda / max(sum(p.numel() for p in model.parameters()), 1)

            loss = loss_cls + ewc_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            self._trades_since_update = 0
            self._total_updates += 1

            loss_val = float(loss.item())
            if self._total_updates % 50 == 0:
                logger.info(
                    "online_update #%d loss=%.4f ewc=%.4f buffer=%d patterns=%d",
                    self._total_updates, loss_val, float(ewc_loss.item()),
                    len(self.replay), len(self.pattern_memory.patterns),
                )
            return loss_val

    def persist(self) -> None:
        """Persist pattern memory to disk."""
        self.pattern_memory.persist(self._pattern_path)

    @property
    def stats(self) -> dict:
        return {
            "total_updates": self._total_updates,
            "replay_size": len(self.replay),
            "pattern_count": len(self.pattern_memory.patterns),
        }
