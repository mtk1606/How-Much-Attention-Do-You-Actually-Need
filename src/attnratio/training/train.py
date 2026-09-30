"""Single-run training loop with periodic validation/test evaluation and full provenance."""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import UTC, datetime
from typing import Any

import torch
import torch.nn.functional as F

from ..data.batches import make_batch, train_seeds
from ..data.tasks import IGNORE, get_task
from ..evaluation.evaluate import evaluate
from ..models.hybrid import HybridLM, parameter_report
from ..tracking import registry
from .config import RunConfig

log = logging.getLogger("attnratio.train")


def lr_at(step: int, cfg: RunConfig) -> float:
    warm = max(1, int(cfg.warmup_frac * cfg.steps))
    if step < warm:
        return cfg.lr * (step + 1) / warm
    progress = (step - warm) / max(1, cfg.steps - warm)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))  # cosine to 10% of peak


def build_optimizer(model: torch.nn.Module, cfg: RunConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 and "embed" not in name else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
        betas=(0.9, 0.98),
    )


def _eval_all(model: HybridLM, cfg: RunConfig, split: str, n: int, device: str) -> list[dict[str, Any]]:
    task = get_task(cfg.task)
    out = []
    for spec in cfg.eval:
        r = evaluate(model, task, task.difficulty(spec.difficulty), spec.seq_len, n, split=split, device=device)
        out.append(r)
    return out


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        f"L{r['seq_len']}_{json.dumps(r['difficulty'], sort_keys=True)}": {
            "token_accuracy": r["token_accuracy"],
            "exact_match": r["exact_match"],
        }
        for r in results
    }


def train_run(cfg: RunConfig, device: str | None = None, resume_ok: bool = True) -> dict[str, Any]:
    """Train one configuration. Returns the final record also written to the registry.

    If a completed run with the same experiment_id exists, it is returned unchanged (idempotent sweeps).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = registry.run_dir(cfg.experiment_id)
    final_path = out / "final.json"
    if resume_ok and final_path.exists():
        log.info("skip %s (already completed)", cfg.experiment_id)
        return json.loads(final_path.read_text())

    torch.manual_seed(cfg.seed)
    task = get_task(cfg.task)
    model = HybridLM(cfg.model).to(device)
    opt = build_optimizer(model, cfg)
    max_len = max(s.seq_len for s in (*cfg.train, *cfg.eval))
    meta: dict[str, Any] = {
        "experiment_id": cfg.experiment_id,
        **registry.git_state(),
        "model_config": cfg.model.to_dict(),
        "dataset_config": {
            "task": cfg.task,
            "train": [vars(s) for s in cfg.train],
            "eval": [vars(s) for s in cfg.eval],
        },
        "attention_ratio": cfg.model.attention_ratio,
        "attention_placement": cfg.model.placement,
        "architecture": cfg.model.family,
        "layer_types": model.layer_types(),
        "seed": cfg.seed,
        **parameter_report(model, max_len),
        "training_tokens": cfg.steps * cfg.batch_size * max(s.seq_len for s in cfg.train),
        "hardware": registry.hardware(),
        "precision": cfg.precision,
        "start_time": datetime.now(UTC).isoformat(),
        "status": "running",
        "tags": list(cfg.tags),
    }
    registry.write_yaml(out / "config.yaml", cfg.to_dict())
    registry.write_yaml(out / "metadata.yaml", meta)
    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("")

    t0 = time.perf_counter()
    tokens_seen = 0
    ema = None
    diverged = False
    model.train()
    for step in range(cfg.steps):
        spec = cfg.train[step % len(cfg.train)]
        d = task.difficulty(spec.difficulty)
        b = make_batch(task, d, spec.seq_len, train_seeds(task, cfg.seed, step, cfg.batch_size))
        tokens, targets = b.tokens.to(device), b.targets.to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        with torch.autocast(
            device_type=device if device != "cpu" else "cpu", dtype=torch.bfloat16, enabled=cfg.precision == "bf16"
        ):
            logits = model(tokens)
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=IGNORE)
        if not torch.isfinite(loss):
            diverged = True
            log.warning("non-finite loss at step %d in %s", step, cfg.experiment_id)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        tokens_seen += tokens.numel()
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()

        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            val = _eval_all(model, cfg, "val", cfg.n_val, device)
            rec = {
                "step": step + 1,
                "tokens": tokens_seen,
                "train_loss_ema": ema,
                "grad_norm": float(gnorm),
                "lr": lr_at(step, cfg),
                "wall_seconds": time.perf_counter() - t0,
                "val": _summary(val),
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            log.info("%s step %d loss %.4f val %s", cfg.experiment_id, step + 1, ema, rec["val"])

    wall = time.perf_counter() - t0
    test = _eval_all(model, cfg, "test", cfg.n_eval, device) if not diverged else []
    val = _eval_all(model, cfg, "val", cfg.n_val, device) if not diverged else []
    if cfg.save_checkpoint and not diverged:
        torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out / "ckpt.pt")
    hw = meta["hardware"]
    gpu_hours = wall / 3600 * hw.get("gpu_count", 0)
    meta.update(
        {
            "end_time": datetime.now(UTC).isoformat(),
            "status": "diverged" if diverged else "completed",
            "wall_seconds": round(wall, 2),
            "tokens_processed": tokens_seen,
            "gpu_hours": round(gpu_hours, 4),
            "estimated_cost_usd": round(wall / 3600 * registry.usd_per_hour(hw["device"]), 4),
            "peak_memory_mb": round(registry.peak_memory_mb(), 1),
            "checkpoint_path": str((out / "ckpt.pt").relative_to(registry.ARTIFACTS.parent))
            if cfg.save_checkpoint and not diverged
            else None,
            "metrics_path": str(metrics_path.relative_to(registry.ARTIFACTS.parent)),
            "final_train_loss_ema": ema,
        }
    )
    registry.write_yaml(out / "metadata.yaml", meta)
    final = {"experiment_id": cfg.experiment_id, "status": meta["status"], "metadata": meta, "test": test, "val": val}
    final_path.write_text(json.dumps(final))
    registry.append_registry(
        {
            "experiment_id": cfg.experiment_id,
            "status": meta["status"],
            "task": cfg.task,
            "family": cfg.model.family,
            "n_attn": cfg.model.n_attn,
            "n_layers": cfg.model.n_layers,
            "attention_ratio": cfg.model.attention_ratio,
            "placement": cfg.model.placement,
            "d_model": cfg.model.d_model,
            "lr": cfg.lr,
            "seed": cfg.seed,
            "tags": list(cfg.tags),
            "final_path": str(final_path.relative_to(registry.ARTIFACTS.parent)),
            "git_commit": meta["git_commit"],
            "wall_seconds": meta["wall_seconds"],
        }
    )
    return final
