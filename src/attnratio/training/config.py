"""Run configuration: one model, one probe task, one optimiser setting, one seed."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..data.tasks import get_task
from ..models.hybrid import ModelConfig


@dataclass(frozen=True)
class DataSpec:
    seq_len: int
    difficulty: dict[str, Any] = field(default_factory=dict)  # overrides on the task defaults


@dataclass(frozen=True)
class RunConfig:
    task: str
    model: ModelConfig
    train: tuple[DataSpec, ...]
    eval: tuple[DataSpec, ...]
    steps: int = 2000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.1
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    seed: int = 0
    eval_every: int = 250
    n_eval: int = 256
    n_val: int = 128
    precision: str = "fp32"
    save_checkpoint: bool = True
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        task = get_task(self.task)
        for spec in (*self.train, *self.eval):
            d = task.difficulty(spec.difficulty)
            task.generate_example(0, spec.seq_len, d)  # raises on an invalid spec
            if task.vocab_size(d) > self.model.vocab_size:
                raise ValueError(f"task vocab {task.vocab_size(d)} exceeds model vocab {self.model.vocab_size}")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model"] = self.model.to_dict()
        return d

    def config_hash(self) -> str:
        d = self.to_dict()
        d.pop("tags", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]

    @property
    def experiment_id(self) -> str:
        m = self.model
        fam = "attn" if m.n_attn == m.n_layers else m.family
        lr = f"{self.lr:.0e}".replace("-0", "-")
        return (
            f"{self.task}_{fam}_a{m.n_attn}of{m.n_layers}_{m.placement}_d{m.d_model}"
            f"_L{max(s.seq_len for s in self.train)}_lr{lr}_s{self.seed}_{self.config_hash()}"
        )


def run_config_from_dict(d: dict[str, Any]) -> RunConfig:
    d = dict(d)
    model = dict(d.pop("model"))
    model.pop("attention_ratio", None)
    return RunConfig(
        model=ModelConfig(**model),
        train=tuple(DataSpec(**s) for s in d.pop("train")),
        eval=tuple(DataSpec(**s) for s in d.pop("eval")),
        tags=tuple(d.pop("tags", ())),
        **d,
    )


def load_run_config(path: str | Path) -> RunConfig:
    return run_config_from_dict(yaml.safe_load(Path(path).read_text()))
