"""Batching and split seeding for the probe tasks.

Train and eval examples come from disjoint seed namespaces: train seeds are derived from
("train", task, run_seed, step, i); validation and test seeds from ("val" | "test", task,
difficulty, length, i). Validation picks hyperparameters; only test numbers are reported.
A dataset is therefore fully described by (task, difficulty, sequence_length, split, index range),
which is what gets stored in run metadata instead of cached tensors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .tasks import IGNORE, Task, derive_seed


@dataclass
class Batch:
    tokens: torch.Tensor  # (B, L)
    targets: torch.Tensor  # (B, L), IGNORE where no loss
    score_mask: torch.Tensor  # (B, L) bool, positions that count toward the probe metric
    meta: list[dict[str, Any]]


def difficulty_key(difficulty: dict[str, Any]) -> str:
    return json.dumps(difficulty, sort_keys=True)


def train_seeds(task: Task, run_seed: int, step: int, batch_size: int) -> list[int]:
    return [derive_seed("train", task.name, run_seed, step, i) for i in range(batch_size)]


def eval_seeds(
    task: Task, difficulty: dict[str, Any], seq_len: int, n: int, offset: int = 0, split: str = "test"
) -> list[int]:
    """split="val" is used for hyperparameter selection, split="test" for every reported number."""
    if split not in ("val", "test"):
        raise ValueError(f"unknown split {split!r}")
    key = difficulty_key(difficulty)
    return [derive_seed(split, task.name, key, seq_len, i) for i in range(offset, offset + n)]


def make_batch(task: Task, difficulty: dict[str, Any], seq_len: int, seeds: list[int]) -> Batch:
    exs = [task.generate_example(s, seq_len, difficulty) for s in seeds]
    tokens = torch.from_numpy(np.stack([e.tokens for e in exs]))
    targets = torch.from_numpy(np.stack([e.targets for e in exs]))
    if "induction_mask" in exs[0].meta:
        mask = torch.from_numpy(np.stack([e.meta["induction_mask"] for e in exs]))
    else:
        mask = targets != IGNORE
    return Batch(tokens, targets, mask, [e.meta for e in exs])
