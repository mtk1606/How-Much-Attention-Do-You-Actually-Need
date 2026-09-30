"""Mamba-2 mixer in pure PyTorch.

Follows the Mamba-2 block of Dao & Gu (2024): one input projection producing (z, x, B, C, dt), a
causal depthwise convolution over (x, B, C), a scalar-decay-per-head selective SSM computed with the
chunked SSD algorithm, a D skip term, a gated RMSNorm and an output projection. ngroups = 1.

The chunked scan is my own implementation of the SSD decomposition. It is checked in
tests/test_mamba2.py against a token-by-token recurrence and against the official minimal
reference vendored from state-spaces/mamba. There are no fused kernels here: this code is meant to
be correct and readable on CPU and small GPUs, not fast.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .common import CausalShortConv, RMSNorm, init_dt_bias


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sums: out[..., i, j] = sum_{j < s <= i} x[..., s] for j <= i, -inf above the diagonal."""
    t = x.shape[-1]
    x = x[..., None].expand(*x.shape, t)
    strict = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~strict, 0.0)
    out = torch.cumsum(x, dim=-2)
    incl = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=0)
    return out.masked_fill(~incl, -math.inf)


def ssd_chunked(
    x: torch.Tensor, log_a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked scan of h_t = exp(log_a_t) h_{t-1} + x_t b_t^T, y_t = h_t c_t.

    x: (B, L, H, P), log_a: (B, L, H) (<= 0), b, c: (B, L, H, N). L must be a multiple of chunk.
    Returns y (B, L, H, P) and the final state (B, H, P, N).
    """
    bsz, length, h, p = x.shape
    n = b.shape[-1]
    nc = length // chunk
    x = x.reshape(bsz, nc, chunk, h, p)
    b = b.reshape(bsz, nc, chunk, h, n)
    c = c.reshape(bsz, nc, chunk, h, n)
    a = log_a.reshape(bsz, nc, chunk, h).permute(0, 3, 1, 2)  # (B, H, nc, chunk)
    a_cum = torch.cumsum(a, dim=-1)

    # Intra-chunk (diagonal blocks).
    decay = torch.exp(_segsum(a))  # (B, H, nc, l, s)
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", c, b, decay, x)

    # State contributed by each chunk at its right boundary.
    decay_to_end = torch.exp(a_cum[..., -1:] - a_cum)
    chunk_states = torch.einsum("bclhn,bhcl,bclhp->bchpn", b, decay_to_end, x)

    # Propagate states across chunks.
    init = torch.zeros_like(chunk_states[:, :1])
    chunk_states = torch.cat([init, chunk_states], dim=1)
    decay_chunks = torch.exp(_segsum(F.pad(a_cum[..., -1], (1, 0))))  # (B, H, nc+1, nc+1)
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunks, chunk_states)
    prev_states, final = states[:, :-1], states[:, -1]

    # Contribution of the incoming state to outputs inside each chunk.
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", c, prev_states, torch.exp(a_cum))
    return (y_diag + y_off).reshape(bsz, length, h, p), final


def ssd_recurrent(x: torch.Tensor, log_a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Token-by-token reference for ssd_chunked (slow; tests only)."""
    bsz, length, h, p = x.shape
    state = x.new_zeros(bsz, h, p, b.shape[-1])
    ys = []
    for t in range(length):
        state = state * torch.exp(log_a[:, t])[..., None, None] + x[:, t, :, :, None] * b[:, t, :, None, :]
        ys.append(torch.einsum("bhpn,bhn->bhp", state, c[:, t]))
    return torch.stack(ys, dim=1)


class Mamba2Mixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        head_dim: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        chunk: int = 32,
    ) -> None:
        super().__init__()
        if d_inner % head_dim:
            raise ValueError(f"d_inner={d_inner} not divisible by head_dim={head_dim}")
        self.d_inner, self.head_dim, self.d_state, self.chunk = d_inner, head_dim, d_state, chunk
        self.n_heads = d_inner // head_dim
        conv_dim = d_inner + 2 * d_state
        self.in_proj = nn.Linear(d_model, 2 * d_inner + 2 * d_state + self.n_heads, bias=False)
        self.conv = CausalShortConv(conv_dim, d_conv)
        self.dt_bias = nn.Parameter(init_dt_bias(self.n_heads))
        self.a_log = nn.Parameter(torch.log(torch.empty(self.n_heads).uniform_(1.0, 16.0)))
        self.d_skip = nn.Parameter(torch.ones(self.n_heads))
        self.norm = RMSNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = u.shape
        z, xbc, dt = torch.split(self.in_proj(u), [self.d_inner, self.d_inner + 2 * self.d_state, self.n_heads], dim=-1)
        xbc = F.silu(self.conv(xbc))
        x, b, c = torch.split(xbc, [self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)  # (B, L, H)
        a = -torch.exp(self.a_log.float())
        x = x.view(bsz, length, self.n_heads, self.head_dim)

        pad = (-length) % self.chunk
        xs = F.pad(x * dt[..., None], (0, 0, 0, 0, 0, pad))
        log_a = F.pad(dt * a, (0, 0, 0, pad))
        bb = F.pad(b, (0, 0, 0, pad))[:, :, None, :].expand(-1, -1, self.n_heads, -1)
        cc = F.pad(c, (0, 0, 0, pad))[:, :, None, :].expand(-1, -1, self.n_heads, -1)
        y, _ = ssd_chunked(xs, log_a, bb, cc, self.chunk)
        y = y[:, :length] + x * self.d_skip[:, None]
        y = y.reshape(bsz, length, self.d_inner)
        return self.out_proj(self.norm(y * F.silu(z)))

    def state_numel(self, seq_len: int) -> int:
        """Recurrent state entries per sequence (constant in length; excludes the conv buffer)."""
        del seq_len
        return self.d_inner * self.d_state
