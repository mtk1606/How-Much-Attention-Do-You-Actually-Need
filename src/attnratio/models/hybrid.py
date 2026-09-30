"""Hybrid sequence model: a stack of pre-norm residual blocks, each a sequence mixer plus a SwiGLU MLP.

A block's mixer is either softmax attention or the family's non-attention mixer. Which layers get
attention is decided by ``attention_schedule`` from (n_layers, n_attn, placement), so the attention
ratio and the placement are independent, explicit variables.

Per-layer mixer parameter budgets are matched to 4 * d_model^2 (the attention mixer's size) by
construction; ``parameter_report`` measures the remaining mismatch rather than assuming it away.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .attention import AttentionMixer
from .common import RMSNorm, SwiGLU, mlp_hidden
from .gdn import GatedDeltaNetMixer
from .mamba2 import Mamba2Mixer

FAMILIES = ("mamba2", "gdn", "gdn_neg")
PLACEMENTS = ("block_end", "block_start", "front", "back", "middle")


def attention_schedule(n_layers: int, n_attn: int, placement: str) -> list[bool]:
    """Boolean mask over layers, True where the mixer is attention.

    block_end   split depth into n_attn equal blocks, attention closes each block (Qwen3.5/3.6 style)
    block_start same blocks, attention opens each block
    front       first n_attn layers
    back        last n_attn layers
    middle      a contiguous run centred in depth
    """
    if not 0 <= n_attn <= n_layers:
        raise ValueError(f"n_attn={n_attn} outside [0, {n_layers}]")
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown placement {placement!r}; expected one of {PLACEMENTS}")
    if n_attn == 0:
        return [False] * n_layers
    if placement == "block_end":
        idx = {math.ceil((i + 1) * n_layers / n_attn) - 1 for i in range(n_attn)}
    elif placement == "block_start":
        idx = {math.floor(i * n_layers / n_attn) for i in range(n_attn)}
    elif placement == "front":
        idx = set(range(n_attn))
    elif placement == "back":
        idx = set(range(n_layers - n_attn, n_layers))
    else:
        start = (n_layers - n_attn) // 2
        idx = set(range(start, start + n_attn))
    assert len(idx) == n_attn
    return [i in idx for i in range(n_layers)]


@dataclass(frozen=True)
class ModelConfig:
    family: str
    vocab_size: int
    d_model: int = 128
    n_layers: int = 8
    n_attn: int = 0
    placement: str = "block_end"
    attn_heads: int = 4
    rope_fraction: float = 1.0
    mlp_ratio: float = 2.0
    mamba_head_dim: int = 16
    mamba_d_state: int = 16
    gdn_heads: int = 4
    d_conv: int = 4
    chunk: int = 32
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; expected one of {FAMILIES}")
        attention_schedule(self.n_layers, self.n_attn, self.placement)

    @property
    def attention_ratio(self) -> float:
        return self.n_attn / self.n_layers

    @property
    def mamba_d_inner(self) -> int:
        """Multiple of mamba_head_dim whose Mamba-2 mixer parameter count is closest to attention's 4 d^2."""
        d, n, p, k = self.d_model, self.mamba_d_state, self.mamba_head_dim, self.d_conv

        def params(di: int) -> int:
            h = di // p
            return d * (2 * di + 2 * n + h) + (di + 2 * n) * (k + 1) + 3 * h + di + di * d

        candidates = [p * j for j in range(1, 4 * d // p + 1)]
        return min(candidates, key=lambda di: abs(params(di) - 4 * d * d))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["attention_ratio"] = self.attention_ratio
        return d


def build_mixer(cfg: ModelConfig, is_attention: bool) -> nn.Module:
    if is_attention:
        return AttentionMixer(cfg.d_model, cfg.attn_heads, rope_fraction=cfg.rope_fraction)
    if cfg.family == "mamba2":
        return Mamba2Mixer(cfg.d_model, cfg.mamba_d_inner, cfg.mamba_head_dim, cfg.mamba_d_state, cfg.d_conv, cfg.chunk)
    # Gated DeltaNet: H * d_k = d/2 and H * d_v = d gives q,k (d^2) + v, gate, o (3 d^2) = 4 d^2.
    if cfg.d_model % (2 * cfg.gdn_heads):
        raise ValueError("d_model must be divisible by 2 * gdn_heads")
    return GatedDeltaNetMixer(
        cfg.d_model,
        cfg.gdn_heads,
        head_k_dim=cfg.d_model // (2 * cfg.gdn_heads),
        head_v_dim=cfg.d_model // cfg.gdn_heads,
        d_conv=cfg.d_conv,
        chunk=cfg.chunk,
        allow_neg_eigval=cfg.family == "gdn_neg",
    )


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, is_attention: bool) -> None:
        super().__init__()
        self.is_attention = is_attention
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = build_mixer(cfg, is_attention)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, mlp_hidden(cfg.d_model, cfg.mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class HybridLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.schedule = attention_schedule(cfg.n_layers, cfg.n_attn, cfg.placement)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, a) for a in self.schedule])
        self.norm = RMSNorm(cfg.d_model)
        self.apply(self._init)
        # GPT-2 style scaling of residual-branch output projections.
        for name, p in self.named_parameters():
            if name.endswith(("out.weight", "out_proj.weight", "o_proj.weight", "w_out.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.norm(x), self.embed.weight)  # tied output head

    def layer_types(self) -> list[str]:
        return ["attention" if a else self.cfg.family for a in self.schedule]


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def parameter_report(model: HybridLM, seq_len: int) -> dict[str, Any]:
    """Parameter and state accounting used for model_match_report.csv."""
    cfg = model.cfg
    attn_mix = [count_params(b.mixer) for b in model.blocks if b.is_attention]
    other_mix = [count_params(b.mixer) for b in model.blocks if not b.is_attention]
    state = [b.mixer.state_numel(seq_len) for b in model.blocks]  # type: ignore[operator]
    total = count_params(model)
    embed = model.embed.weight.numel()
    return {
        "parameter_count": total,
        "non_embedding_parameters": total - embed,
        "attention_mixer_params_per_layer": attn_mix[0] if attn_mix else 0,
        "other_mixer_params_per_layer": other_mix[0] if other_mix else 0,
        "mixer_param_ratio_other_over_attention": (other_mix[0] / (4 * cfg.d_model**2)) if other_mix else 1.0,
        "recurrent_state_numel_per_layer": state[model.schedule.index(False)] if other_mix else 0,
        "kv_cache_numel_at_seq_len": sum(s for s, a in zip(state, model.schedule, strict=True) if a),
        "layer_types": "".join("A" if a else "S" for a in model.schedule),
    }
