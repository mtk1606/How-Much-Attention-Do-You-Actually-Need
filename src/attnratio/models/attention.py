"""Causal multi-head softmax attention with rotary position embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import apply_rotary, rotary_cache


class AttentionMixer(nn.Module):
    """Parameters: 4 * d_model^2 (fused QKV plus output projection, no biases)."""

    def __init__(self, d_model: int, n_heads: int, rope_base: float = 10000.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2:
            raise ValueError("rotary embeddings need an even head dimension")
        self.rope_base = rope_base
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        q, k, v = self.qkv(x).view(b, length, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = rotary_cache(length, self.head_dim, self.rope_base, x.device)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, length, d))

    def state_numel(self, seq_len: int) -> int:
        """KV cache entries per sequence at length seq_len (grows with length)."""
        return 2 * seq_len * self.n_heads * self.head_dim
