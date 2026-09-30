"""Blocks shared by every architecture family: norms, MLP, rotary embeddings, short convolution."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.w_in = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(a) * b)


def mlp_hidden(d_model: int, ratio: float, multiple: int = 16) -> int:
    h = int(ratio * d_model)
    return multiple * math.ceil(h / multiple)


class CausalShortConv(nn.Module):
    """Depthwise causal 1D convolution over the sequence axis, input/output (b, l, c)."""

    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, padding=kernel - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        return self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2)


def rotary_cache(length: int, dim: int, base: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(length, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (b, h, l, d). Rotates channel pairs (i, i + d/2)."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def init_dt_bias(
    n: int, dt_min: float = 1e-3, dt_max: float = 1e-1, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Mamba-style step-size bias: dt ~ LogUniform(dt_min, dt_max), stored as softplus^-1(dt)."""
    u = torch.rand(n, generator=generator)
    dt = torch.exp(u * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
    return inverse_softplus(dt.clamp(min=1e-4))
