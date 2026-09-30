"""Causal multi-head softmax attention with rotary position embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import apply_rotary, rotary_cache


class AttentionMixer(nn.Module):
    """Parameters: 4 * d_model^2 (fused QKV plus output projection, no biases)."""

    def __init__(self, d_model: int, n_heads: int, rope_base: float = 10000.0, rope_fraction: float = 1.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2:
            raise ValueError("rotary embeddings need an even head dimension")
        self.rope_base = rope_base
        # Rotary on the first rope_dim channels of each head only (Qwen3.5/3.6 rotate 64 of 256); the
        # remaining channels are position-free and can match keys by content alone.
        self.rope_dim = 2 * int(round(rope_fraction * self.head_dim / 2))
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        q, k, v = self.qkv(x).view(b, length, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.rope_dim:
            r = self.rope_dim
            cos, sin = rotary_cache(length, r, self.rope_base, x.device)
            q = torch.cat([apply_rotary(q[..., :r], cos, sin), q[..., r:]], dim=-1)
            k = torch.cat([apply_rotary(k[..., :r], cos, sin), k[..., r:]], dim=-1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, length, d))

    def state_numel(self, seq_len: int) -> int:
        """KV cache entries per sequence at length seq_len (grows with length)."""
        return 2 * seq_len * self.n_heads * self.head_dim
