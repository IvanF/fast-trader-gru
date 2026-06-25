"""Shared PyTorch model definitions for training and ONNX export."""

from __future__ import annotations

import torch
import torch.nn as nn

SEQ_LEN = 60
OB_DIM = 2
FLOW_DIM = 3
MACRO_DIM = 14
EMBED_DIM = 32
STATE_DIM = 128
MEMORY_DIM = 8
MLP_IN_DIM = STATE_DIM + MEMORY_DIM
NUM_DIRECTIONS = 3


class OrderbookCNN(nn.Module):
    def __init__(self, out_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(OB_DIM, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


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
    """Late-fusion backbone producing the master state vector."""

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
    """Direction + confidence + vol heads. Balanced architecture for medium-data regimes."""

    def __init__(self, in_dim: int = MLP_IN_DIM, out_dim: int = 5, dropout: float = 0.45) -> None:
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
