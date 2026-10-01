"""Probe evaluation. Deterministic given (model weights, task, difficulty, length, split, n)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..data.batches import eval_seeds, make_batch
from ..data.tasks import IGNORE, Task


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    task: Task,
    difficulty: dict[str, Any],
    seq_len: int,
    n_examples: int,
    split: str = "test",
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Returns token accuracy over scored positions, exact match, first-error position and loss.

    first-error position is the index (among scored positions) of the first wrong prediction, or the
    number of scored positions if none is wrong. For state tracking it measures how long a model follows
    the rule before failing, which per-position accuracy hides.

    Also returns per-example exact match and token accuracy so the analysis can bootstrap over examples.
    """
    was_training = model.training
    model.eval()
    seeds = eval_seeds(task, difficulty, seq_len, n_examples, split=split)
    correct = total = 0
    loss_sum, loss_n = 0.0, 0
    exact: list[int] = []
    tok_acc: list[float] = []
    first_err: list[int] = []
    for i in range(0, n_examples, batch_size):
        b = make_batch(task, difficulty, seq_len, seeds[i : i + batch_size])
        logits = model(b.tokens.to(device)).float().cpu()
        pred = logits.argmax(-1)
        lossed = b.targets != IGNORE  # same positions as the training loss
        loss_sum += float(F.cross_entropy(logits[lossed], b.targets[lossed], reduction="sum"))
        loss_n += int(lossed.sum())
        hit = (pred == b.targets) & b.score_mask
        correct += int(hit.sum())
        total += int(b.score_mask.sum())
        miss = b.score_mask & ~hit
        exact.extend((~miss.any(-1)).int().tolist())
        tok_acc.extend((hit.sum(-1) / b.score_mask.sum(-1).clamp(min=1)).tolist())
        for row_mask, row_miss in zip(b.score_mask, miss, strict=True):
            scored = torch.nonzero(row_mask).flatten()
            errs = torch.nonzero(row_miss[scored]).flatten()
            first_err.append(int(errs[0]) if errs.numel() else int(scored.numel()))
    if was_training:
        model.train()
    return {
        "task": task.name,
        "split": split,
        "seq_len": seq_len,
        "difficulty": difficulty,
        "n_examples": n_examples,
        "token_accuracy": correct / max(1, total),
        "exact_match": float(np.mean(exact)),
        "mean_first_error": float(np.mean(first_err)),
        "loss": loss_sum / max(1, loss_n),
        "per_example_first_error": first_err,
        "per_example_exact": exact,
        "per_example_token_accuracy": [round(x, 6) for x in tok_acc],
    }
