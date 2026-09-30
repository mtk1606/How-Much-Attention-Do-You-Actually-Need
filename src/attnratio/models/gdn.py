"""Gated DeltaNet mixer in pure PyTorch.

Recurrence (Yang, Kautz & Hatamizadeh, ICLR 2025), per head, state S in R^{d_k x d_v}:

    S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,     o_t = S_t^T q_t

with alpha_t = exp(g_t) in (0, 1] a scalar decay and beta_t a scalar write strength. With
``allow_neg_eigval`` beta_t lies in (0, 2), so the transition (I - beta k k^T) can have eigenvalue
1 - beta < 0 along k_t. Grazzi et al. (ICLR 2025) show this is what lets delta-rule models solve
parity; with beta in (0, 1) they provably cannot at finite precision.

The layer layout follows the Qwen3-Next / Qwen3.5 Gated DeltaNet block: separate q, k, v projections
with SiLU short convolutions, L2-normalised q and k, Mamba-style decay g = -exp(A_log) *
softplus(a + dt_bias), a per-head RMSNorm gated by SiLU(gate), and an output projection.

The chunked algorithm below is derived in docs/DERIVATIONS.md (section "Chunked gated delta rule").
Within a chunk it solves a unit lower-triangular system instead of a Python loop, and it is checked
against the token-by-token recurrence in tests/test_gdn.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .common import CausalShortConv, RMSNorm, init_dt_bias


def gated_delta_chunked(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: (B, H, L, dk); v: (B, H, L, dv); g (log decay, <= 0), beta: (B, H, L).

    L must be a multiple of chunk. Returns outputs (B, H, L, dv) and the final state (B, H, dk, dv).
    """
    bsz, h, length, dk = q.shape
    dv = v.shape[-1]
    nc = length // chunk
    q = q.reshape(bsz, h, nc, chunk, dk)
    k = k.reshape(bsz, h, nc, chunk, dk)
    v = v.reshape(bsz, h, nc, chunk, dv)
    g = g.reshape(bsz, h, nc, chunk)
    beta = beta.reshape(bsz, h, nc, chunk)

    gam = torch.cumsum(g, dim=-1)
    incl = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
    strict = torch.tril(incl, diagonal=-1)
    decay = torch.exp((gam[..., :, None] - gam[..., None, :]).masked_fill(~incl, -math.inf))

    kb = k * beta[..., None]
    eye = torch.eye(chunk, dtype=q.dtype, device=q.device)
    m = eye + ((kb @ k.transpose(-1, -2)) * decay).masked_fill(~strict, 0.0)
    w = torch.linalg.solve_triangular(m, kb * torch.exp(gam)[..., None], upper=False, unitriangular=True)
    u0 = torch.linalg.solve_triangular(m, v * beta[..., None], upper=False, unitriangular=True)
    qk = (q @ k.transpose(-1, -2)) * decay
    q_dec = q * torch.exp(gam)[..., None]
    k_dec = k * torch.exp(gam[..., -1:] - gam)[..., None]

    state = q.new_zeros(bsz, h, dk, dv)
    outs = []
    for i in range(nc):
        u = u0[:, :, i] - w[:, :, i] @ state
        outs.append(q_dec[:, :, i] @ state + qk[:, :, i] @ u)
        state = state * torch.exp(gam[:, :, i, -1])[..., None, None] + k_dec[:, :, i].transpose(-1, -2) @ u
    return torch.stack(outs, dim=2).reshape(bsz, h, length, dv), state


def gated_delta_recurrent(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Token-by-token reference for gated_delta_chunked (slow; tests only)."""
    bsz, h, length, dk = q.shape
    state = q.new_zeros(bsz, h, dk, v.shape[-1])
    outs = []
    for t in range(length):
        kt, vt, bt = k[:, :, t], v[:, :, t], beta[:, :, t, None, None]
        state = torch.exp(g[:, :, t])[..., None, None] * (state - bt * kt[..., :, None] * (kt[..., None, :] @ state))
        state = state + bt * kt[..., :, None] * vt[..., None, :]
        outs.append((q[:, :, t, None, :] @ state).squeeze(-2))
    return torch.stack(outs, dim=2)


class GatedDeltaNetMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        d_conv: int = 4,
        chunk: int = 32,
        allow_neg_eigval: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads, self.dk, self.dv, self.chunk = n_heads, head_k_dim, head_v_dim, chunk
        self.allow_neg_eigval = allow_neg_eigval
        key_dim, value_dim = n_heads * head_k_dim, n_heads * head_v_dim
        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, value_dim, bias=False)
        self.gate_proj = nn.Linear(d_model, value_dim, bias=False)
        self.ab_proj = nn.Linear(d_model, 2 * n_heads, bias=False)
        self.q_conv = CausalShortConv(key_dim, d_conv)
        self.k_conv = CausalShortConv(key_dim, d_conv)
        self.v_conv = CausalShortConv(value_dim, d_conv)
        self.dt_bias = nn.Parameter(init_dt_bias(n_heads))
        self.a_log = nn.Parameter(torch.log(torch.empty(n_heads).uniform_(1.0, 16.0)))
        self.norm = RMSNorm(head_v_dim)
        self.o_proj = nn.Linear(value_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = x.shape
        h = self.n_heads

        def heads(t: torch.Tensor, d: int) -> torch.Tensor:
            return t.view(bsz, length, h, d).transpose(1, 2)

        q = F.normalize(heads(F.silu(self.q_conv(self.q_proj(x))), self.dk), dim=-1) * self.dk**-0.5
        k = F.normalize(heads(F.silu(self.k_conv(self.k_proj(x))), self.dk), dim=-1)
        v = heads(F.silu(self.v_conv(self.v_proj(x))), self.dv)
        a, b = self.ab_proj(x).transpose(1, 2).chunk(2, dim=1)  # (B, H, L) each
        g = -torch.exp(self.a_log)[None, :, None] * F.softplus(a + self.dt_bias[None, :, None])
        beta = torch.sigmoid(b) * (2.0 if self.allow_neg_eigval else 1.0)

        pad = (-length) % self.chunk
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
        o, _ = gated_delta_chunked(q, k, v, g, beta, self.chunk)
        o = o[:, :, :length].transpose(1, 2)  # (B, L, H, dv)
        gate = self.gate_proj(x).view(bsz, length, h, self.dv)
        o = self.norm(o) * F.silu(gate)
        return self.o_proj(o.reshape(bsz, length, h * self.dv))

    def state_numel(self, seq_len: int) -> int:
        del seq_len
        return self.n_heads * self.dk * self.dv
