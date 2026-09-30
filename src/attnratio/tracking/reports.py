"""Tables derived from the registry: configs/model_match_report.csv and artifacts/compute_summary.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..models.hybrid import HybridLM, parameter_report
from ..training.config import RunConfig
from . import registry

MATCH_FIELDS = [
    "model_id",
    "architecture_family",
    "attention_ratio",
    "attention_layers",
    "non_attention_layers",
    "total_layers",
    "placement",
    "layer_types",
    "hidden_size",
    "parameter_count",
    "non_embedding_parameters",
    "param_delta_vs_attention_only_pct",
    "mixer_param_ratio_other_over_attention",
    "recurrent_state_numel_per_layer",
    "kv_cache_numel_at_seq_len",
    "training_tokens",
    "estimated_train_flops",
    "seed",
]


def model_match_rows(configs: list[RunConfig]) -> list[dict[str, Any]]:
    """One row per distinct model architecture (seed/lr/task-independent) in the given configs."""
    rows: dict[str, dict[str, Any]] = {}
    ref_params: dict[tuple[int, int, int], int] = {}
    for cfg in configs:
        m = cfg.model
        fam = "attn" if m.n_attn == m.n_layers else m.family
        model_id = f"{fam}_a{m.n_attn}of{m.n_layers}_{m.placement}_d{m.d_model}_v{m.vocab_size}"
        if model_id in rows:
            continue
        max_len = max(s.seq_len for s in (*cfg.train, *cfg.eval))
        model = HybridLM(m)
        rep = parameter_report(model, max_len)
        tokens = cfg.steps * cfg.batch_size * max(s.seq_len for s in cfg.train)
        key = (m.d_model, m.n_layers, m.vocab_size)
        if key not in ref_params:
            ref_cfg = type(m)(
                **{**{k: v for k, v in m.to_dict().items() if k != "attention_ratio"}, "n_attn": m.n_layers}
            )
            ref_params[key] = parameter_report(HybridLM(ref_cfg), max_len)["parameter_count"]
        rows[model_id] = {
            "model_id": model_id,
            "architecture_family": "attention" if m.n_attn == m.n_layers else m.family,
            "attention_ratio": m.attention_ratio,
            "attention_layers": m.n_attn,
            "non_attention_layers": m.n_layers - m.n_attn,
            "total_layers": m.n_layers,
            "placement": m.placement,
            "layer_types": rep["layer_types"],
            "hidden_size": m.d_model,
            "parameter_count": rep["parameter_count"],
            "non_embedding_parameters": rep["non_embedding_parameters"],
            "param_delta_vs_attention_only_pct": round(100 * (rep["parameter_count"] / ref_params[key] - 1), 3),
            "mixer_param_ratio_other_over_attention": round(rep["mixer_param_ratio_other_over_attention"], 4),
            "recurrent_state_numel_per_layer": rep["recurrent_state_numel_per_layer"],
            "kv_cache_numel_at_seq_len": rep["kv_cache_numel_at_seq_len"],
            "training_tokens": tokens,
            # 6 N D for the dense parameters; attention's L^2 term is excluded and reported separately.
            "estimated_train_flops": 6 * rep["non_embedding_parameters"] * tokens,
            "seed": "all",
        }
    return list(rows.values())


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0]) if rows else (fields or [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


COMPUTE_FIELDS = [
    "experiment_id",
    "status",
    "device",
    "gpu_count",
    "cpu_threads",
    "wall_seconds",
    "gpu_hours",
    "estimated_cost_usd",
    "tokens_processed",
    "peak_memory_mb",
    "precision",
]


def compute_summary() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for rec in registry.load_registry(status=None):
        final = registry.ARTIFACTS.parent / rec["final_path"]
        if not final.exists():
            continue
        meta = json.loads(final.read_text())["metadata"]
        hw = meta["hardware"]
        rows.append(
            {
                "experiment_id": rec["experiment_id"],
                "status": meta["status"],
                "device": hw["device"],
                "gpu_count": hw.get("gpu_count", 0),
                "cpu_threads": hw.get("cpu_threads", ""),
                "wall_seconds": meta["wall_seconds"],
                "gpu_hours": meta["gpu_hours"],
                "estimated_cost_usd": meta["estimated_cost_usd"],
                "tokens_processed": meta["tokens_processed"],
                "peak_memory_mb": meta["peak_memory_mb"],
                "precision": meta["precision"],
            }
        )
    total = {
        "runs": len(rows),
        "wall_hours": round(sum(r["wall_seconds"] for r in rows) / 3600, 3),
        "gpu_hours": round(sum(r["gpu_hours"] for r in rows), 3),
        "estimated_cost_usd": round(sum(r["estimated_cost_usd"] for r in rows), 2),
        "tokens_processed": sum(r["tokens_processed"] for r in rows),
    }
    return rows, total
