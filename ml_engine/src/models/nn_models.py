"""Shared PyTorch model definitions for training and ONNX export."""

from __future__ import annotations

import torch
import torch.nn as nn

SEQ_LEN = 60
OB_DIM = 2
OB_DEPTH = 20
FLOW_DIM = 3
MACRO_DIM = 22
EMBED_DIM = 32
STATE_DIM = 128
MEMORY_DIM = 8
MLP_IN_DIM = STATE_DIM + MEMORY_DIM
NUM_DIRECTIONS = 3


class OrderbookCNN(nn.Module):
    """2D CNN for spatial orderbook tensor (batch, 1, depth, seq_len).
    
    Analyzes patterns across both depth (levels) and time simultaneously.
    Input: (batch, 1, OB_DEPTH, SEQ_LEN) — single channel, depth×time.
    """
    def __init__(self, depth: int = OB_DEPTH, seq_len: int = SEQ_LEN, out_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, depth, seq_len)
        x = self.conv(x)  # (batch, 32, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch, 32)
        return self.fc(x)  # (batch, out_dim)


class SelfAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale, dim=-1)
        return torch.bmm(attn, v)


class FlowGRUAttention(nn.Module):
    def __init__(self, input_dim: int = FLOW_DIM, hidden: int = 32, out_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True, num_layers=1)
        self.attention = SelfAttention(hidden)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        attended = self.attention(out)
        pooled = attended.mean(dim=1)
        return self.fc(pooled)


class FusionModel(nn.Module):
    """Late-fusion backbone producing the master state vector.

    ob_seq: (batch, 1, depth=20, seq_len=60) — 2D orderbook tensor for Conv2d.
    flow_seq: (batch, seq_len=60, flow_dim=3) — trade flow for GRU.
    macro: (batch, macro_dim=22) — macro features.
    """

    def __init__(self, state_dim: int = STATE_DIM) -> None:
        super().__init__()
        self.cnn = OrderbookCNN()
        self.gru = FlowGRUAttention()
        self.macro_proj = nn.Linear(MACRO_DIM, state_dim - 2 * EMBED_DIM)
        self.state_dim = state_dim

    def forward(
        self, ob_seq: torch.Tensor, flow_seq: torch.Tensor, macro: torch.Tensor
    ) -> torch.Tensor:
        cnn_out = self.cnn(ob_seq)
        gru_out = self.gru(flow_seq)
        macro_out = self.macro_proj(macro)
        fused = torch.cat([cnn_out, gru_out, macro_out], dim=-1)
        if fused.shape[-1] < self.state_dim:
            pad = torch.zeros(fused.shape[0], self.state_dim - fused.shape[-1], device=fused.device)
            fused = torch.cat([fused, pad], dim=-1)
        return fused[:, : self.state_dim]


class DecisionMLP(nn.Module):
    """Direction + confidence + vol + trap heads."""

    def __init__(self, in_dim: int = MLP_IN_DIM, out_dim: int = 6, dropout: float = 0.45) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TradingModel(nn.Module):
    """End-to-end model for joint training."""

    def __init__(self) -> None:
        super().__init__()
        self.fusion = FusionModel()
        self.decision = DecisionMLP()

    def forward(
        self,
        ob_seq: torch.Tensor,
        flow_seq: torch.Tensor,
        macro: torch.Tensor,
        v_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.fusion(ob_seq, flow_seq, macro)
        logits = self.decision(torch.cat([state, v_memory], dim=-1))
        return state, logits


EXIT_OPT_IN_DIM = STATE_DIM + MEMORY_DIM + 13


class ExitOptimizer(nn.Module):
    """Predicts optimal SL/TP distances and trade score from market state.

    Input: state_vector(128) + memory(8) + extras(12) = 148
    Output: sl_pct, tp_pct, trade_score
    """

    def __init__(self, in_dim: int = EXIT_OPT_IN_DIM, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        sl_pct = torch.sigmoid(raw[:, 0]) * 0.015
        tp_pct = torch.sigmoid(raw[:, 1]) * 0.05
        trade_score = torch.sigmoid(raw[:, 2])
        return sl_pct, tp_pct, trade_score
