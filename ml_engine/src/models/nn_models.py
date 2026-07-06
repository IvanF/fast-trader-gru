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


class DeltaBarEncoder(nn.Module):
    """Encode 50ms delta bars into token embeddings.

    Input: (batch, seq_len, 6) — 6 features per delta bar
    Output: (batch, seq_len, d_model) — token embeddings
    """

    def __init__(self, in_features: int = 6, d_model: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultimodalCrossAttention(nn.Module):
    """Cross-Attention: TradeFlow (Q) scans Orderbook (K,V) with macro bias.

    TradeFlow tokens query spatial orderbook levels to find
    where aggressive flow meets passive liquidity.
    """

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.macro_gate = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, macro_bias: torch.Tensor) -> torch.Tensor:
        """
        query: (batch, q_len, d_model) — trade flow / delta bars
        key:   (batch, k_len, d_model) — orderbook tokens
        value: (batch, k_len, d_model) — orderbook tokens
        macro_bias: (batch, d_model) — macro context vector
        """
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # macro_gate: (batch, d_model) → (batch, 1) → broadcast to (batch, q_len, k_len)
        gate = torch.tanh(self.macro_gate(macro_bias))  # (batch, 1)
        scores = scores + gate.unsqueeze(-1)

        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)
        return self.out_proj(context)


class FusionModel(nn.Module):
    """Backbone producing the master state vector.

    Legacy mode: CNN + GRU + macro_proj → concat → 128-dim
    Cross-Attention mode: CrossAttention(Flow→OB) + DeltaBarEncoder + macro → 128-dim
    """

    def __init__(self, state_dim: int = STATE_DIM, use_cross_attention: bool = False) -> None:
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.state_dim = state_dim

        if use_cross_attention:
            self.ob_encoder = nn.Linear(2, 64)  # ob_seq → tokens
            self.delta_encoder = DeltaBarEncoder(in_features=6, d_model=64)
            self.cross_attn = MultimodalCrossAttention(d_model=64)
            self.macro_proj = nn.Linear(MACRO_DIM, 64)
            self.out_proj = nn.Linear(64, state_dim)
        else:
            self.cnn = OrderbookCNN()
            self.gru = FlowGRUAttention()
            self.macro_proj = nn.Linear(MACRO_DIM, state_dim - 2 * EMBED_DIM)

    def forward(
        self, ob_seq: torch.Tensor, flow_seq: torch.Tensor, macro: torch.Tensor,
        delta_bars: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.use_cross_attention and delta_bars is not None:
            # ob_seq: (batch, seq_len, 2) → encode to tokens
            ob_tokens = self.ob_encoder(ob_seq)  # (batch, seq, 64)
            # delta_bars: (batch, bar_len, 6) → encode to tokens
            flow_tokens = self.delta_encoder(delta_bars)  # (batch, bar_len, 64)
            # macro → context vector
            macro_out = self.macro_proj(macro)  # (batch, 64)
            # Cross-Attention: flow queries, ob keys/values, macro bias
            context = self.cross_attn(flow_tokens, ob_tokens, ob_tokens, macro_out)
            pooled = context.mean(dim=1)  # (batch, 64)
            return self.out_proj(pooled)[:, :self.state_dim]
        else:
            cnn_out = self.cnn(ob_seq)
            gru_out = self.gru(flow_seq)
            macro_out = self.macro_proj(macro)
            fused = torch.cat([cnn_out, gru_out, macro_out], dim=-1)
            if fused.shape[-1] < self.state_dim:
                pad = torch.zeros(fused.shape[0], self.state_dim - fused.shape[-1], device=fused.device)
                fused = torch.cat([fused, pad], dim=-1)
            return fused[:, : self.state_dim]


class DecisionMLP(nn.Module):
    """Direction + confidence + vol + trap + toxic heads.

    Legacy mode (out_dim=6): direction[0:3] + confidence[3] + vol_mult[4] + trap[5]
    PnL mode (out_dim=3): pred_pnl[0] + trap_logit[1] + toxic_logit[2]
    """

    def __init__(self, in_dim: int = MLP_IN_DIM, out_dim: int = 6, dropout: float = 0.45) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        return self.head(h)


class TradingModel(nn.Module):
    """End-to-end model for joint training."""

    def __init__(self, use_cross_attention: bool = False) -> None:
        super().__init__()
        self.fusion = FusionModel(use_cross_attention=use_cross_attention)
        self.decision = DecisionMLP()

    def forward(
        self,
        ob_seq: torch.Tensor,
        flow_seq: torch.Tensor,
        macro: torch.Tensor,
        v_memory: torch.Tensor,
        delta_bars: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.fusion(ob_seq, flow_seq, macro, delta_bars=delta_bars)
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
