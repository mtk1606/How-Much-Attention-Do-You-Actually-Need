"""Registry -> selected runs -> aggregated curves -> figures and website data.

Steps:
1. Collect completed runs carrying the sweep tag.
2. For each cell (everything except LR and seed) pick the LR with the best mean *validation* token
   accuracy; keep every seed at that LR.
3. For each (cell, eval condition) aggregate *test* scores over seeds with a hierarchical bootstrap.
4. Threshold analysis per (task, family, condition) curve.
5. Write artifacts/analysis/<tag>/{curves.csv, curves.json, thresholds.json} and figures.

Every aggregated point carries the experiment_ids it came from.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..data.tasks import get_task
from ..tracking import registry
from .stats import hierarchical_bootstrap, threshold_analysis

# Categorical slots from the validated reference palette (dataviz skill), fixed order by entity.
FAMILY_COLORS = {"mamba2": "#2a78d6", "gdn": "#eb6834", "gdn_neg": "#1baf7a"}
FAMILY_LABELS = {
    "mamba2": "Mamba-2 hybrid",
    "gdn": "Gated DeltaNet hybrid",
    "gdn_neg": "Gated DeltaNet (β∈(0,2)) hybrid",
}


def condition_label(task: str, seq_len: int, difficulty: dict[str, Any]) -> str:
    t = get_task(task)
    changed = {k: v for k, v in difficulty.items() if t.defaults.get(k) != v}
    main = {
        "mqar": "num_pairs",
        "copy": "span_length",
        "niah": "n_needles",
        "state": "group",
        "induction": "segment_length",
    }
    key = main.get(task)
    parts = [f"L={seq_len}"]
    if key:
        parts.append(f"{key}={difficulty[key]}")
    parts += [f"{k}={v}" for k, v in sorted(changed.items()) if k != key]
    return ", ".join(parts)


def load_runs(tag: str) -> list[dict[str, Any]]:
    runs = []
    for rec in registry.load_registry():
        if tag not in rec.get("tags", []):
            continue
        final = json.loads((registry.ARTIFACTS.parent / rec["final_path"]).read_text())
        runs.append({**rec, "final": final})
    return runs


def _cell_key(r: dict[str, Any]) -> tuple:
    return (r["task"], r["family"], r["n_attn"], r["n_layers"], r["placement"], r["d_model"])


def select_lr(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (selected runs, selection log)."""
    by_cell: dict[tuple, dict[float, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by_cell[_cell_key(r)][r["lr"]].append(r)
    selected, log = [], []
    for cell, by_lr in sorted(by_cell.items(), key=lambda kv: str(kv[0])):
        scores = {
            lr: float(np.mean([np.mean([v["token_accuracy"] for v in x["final"]["val"]]) for x in rs]))
            for lr, rs in by_lr.items()
        }
        best = max(scores, key=lambda lr: scores[lr])
        selected.extend(by_lr[best])
        # A winner at the edge of the grid means a better LR may lie outside it (log entry 06).
        at_edge = len(scores) < 2 or best in (min(scores), max(scores))
        log.append(
            {"cell": list(cell), "val_token_accuracy_by_lr": scores, "selected_lr": best, "lr_at_grid_edge": at_edge}
        )
    return selected, log


def aggregate(selected: list[dict[str, Any]], n_boot: int = 2000) -> list[dict[str, Any]]:
    groups: dict[tuple, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for r in selected:
        for t in r["final"]["test"]:
            cond = json.dumps({"seq_len": t["seq_len"], "difficulty": t["difficulty"]}, sort_keys=True)
            groups[(*_cell_key(r), cond)].append((r, t))
    rows = []
    for (task, family, n_attn, n_layers, placement, d_model, cond), items in groups.items():
        c = json.loads(cond)
        token_chance = get_task(task).chance(c["difficulty"])
        for metric, per_ex in (("token_accuracy", "per_example_token_accuracy"), ("exact_match", "per_example_exact")):
            per_seed = []
            for _, t in items:
                if per_ex in t:
                    per_seed.append(np.asarray(t[per_ex], float))
                else:  # runs from before per-example token accuracy was stored: aggregate value only
                    per_seed.append(np.asarray([t[metric]], float))
            mean, lo, hi = hierarchical_bootstrap(per_seed, n_boot=n_boot)
            chance = token_chance if metric == "token_accuracy" else 0.0  # exact match: effectively 0
            rows.append(
                {
                    "task": task,
                    "family": "attention" if n_attn == n_layers else family,
                    "n_attn": n_attn,
                    "n_layers": n_layers,
                    "attention_ratio": n_attn / n_layers,
                    "placement": placement,
                    "d_model": d_model,
                    "seq_len": c["seq_len"],
                    "difficulty": c["difficulty"],
                    "condition": condition_label(task, c["seq_len"], c["difficulty"]),
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "chance": chance,
                    "normalized": (mean - chance) / (1 - chance),
                    "n_seeds": len(items),
                    "seed_values": [float(t[metric]) for _, t in items],
                    "experiment_ids": sorted(r["experiment_id"] for r, _ in items),
                    "lr": items[0][0]["lr"],
                }
            )
    return sorted(rows, key=lambda x: (x["task"], x["condition"], x["metric"], x["family"], x["attention_ratio"]))


def curves_by_family(rows: list[dict[str, Any]], metric: str) -> dict[tuple, dict[str, list[dict[str, Any]]]]:
    """(task, condition, placement, d, n_layers) -> family -> points, with r = 1 shared by every family."""
    out: dict[tuple, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    shared: dict[tuple, dict[str, Any]] = {}
    fams = {r["family"] for r in rows if r["family"] != "attention"}
    for r in rows:
        if r["metric"] != metric:
            continue
        key = (r["task"], r["condition"], r["d_model"], r["n_layers"])
        if r["family"] == "attention":
            shared[key] = r
        elif r["n_attn"] == 0 or r["placement"] == "block_end":
            out[key][r["family"]].append(r)
    for key, r in shared.items():
        for f in fams:
            out[key][f].append(r)
    for key in out:
        for f in out[key]:
            out[key][f].sort(key=lambda x: x["attention_ratio"])
    return out


def thresholds(rows: list[dict[str, Any]], metric: str = "token_accuracy") -> list[dict[str, Any]]:
    res = []
    for key, fams in curves_by_family(rows, metric).items():
        for fam, pts in fams.items():
            if len(pts) < 3:
                continue
            per = {p["attention_ratio"]: p["seed_values"] for p in pts}
            a = threshold_analysis(list(per), per, n_boot=500)
            res.append({"task": key[0], "condition": key[1], "d_model": key[2], "n_layers": key[3], "family": fam, **a})
    return res


def plot_curves(rows: list[dict[str, Any]], tag: str, out_dir: Path, metric: str = "token_accuracy") -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = curves_by_family(rows, metric)
    tasks = sorted({k[0] for k in curves})
    paths = []
    for task in tasks:
        keys = sorted((k for k in curves if k[0] == task), key=lambda k: _sort_condition(k[1]))
        n = len(keys)
        cols = min(n, 5)
        nrows = int(np.ceil(n / cols))
        width = max(5.2, 2.6 * cols + 0.4)
        fig, axes = plt.subplots(nrows, cols, figsize=(width, 2.5 * nrows + 1.1), squeeze=False, sharey=True)
        for ax, key in zip(axes.flat, keys, strict=False):
            for fam, pts in sorted(curves[key].items()):
                x = [p["attention_ratio"] for p in pts]
                y = [p["mean"] for p in pts]
                lo = [p["ci_low"] for p in pts]
                hi = [p["ci_high"] for p in pts]
                col = FAMILY_COLORS.get(fam, "#52514e")
                ax.fill_between(x, lo, hi, color=col, alpha=0.15, linewidth=0)
                ax.plot(x, y, "-o", color=col, lw=2, ms=4.5, label=FAMILY_LABELS.get(fam, fam))
            ax.axhline(pts[0]["chance"], color="#9b9a95", lw=1, ls=":")
            ax.set_title(", ".join(key[1].split(", ")[:2]), fontsize=9, color="#0b0b0b")
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, color="#e6e5e0", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8, colors="#52514e")
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel(metric.replace("_", " "), fontsize=9)
        for ax in axes[-1, :]:
            ax.set_xlabel("attention ratio r", fontsize=9)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 2), frameon=False, fontsize=9)
        shared = sorted(set.intersection(*(set(k[1].split(", ")[2:]) for k in keys))) if keys else []
        note = f"{task}, d={keys[0][2]}, {keys[0][3]} layers" + (f"; {', '.join(shared)}" if shared else "")
        note += "; band = 95% bootstrap interval; dotted = chance"
        import textwrap

        note = "\n".join(textwrap.wrap(note, width=int(width * 17)))
        fig.text(0.01, 0.005, note, fontsize=7, color="#52514e", ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.07, 1, 0.88))
        stem = out_dir / f"{tag}_{task}_{metric}"
        for ext in ("pdf", "png"):
            fig.savefig(f"{stem}.{ext}", dpi=160)
        plt.close(fig)
        paths.append(Path(f"{stem}.png"))
    return paths


def _sort_condition(label: str) -> tuple:
    import re

    nums = [float(x) for x in re.findall(r"=(\d+(?:\.\d+)?)", label)]
    return (tuple(nums), label)


def analyze_tag(tag: str, out_dir: Path) -> dict[str, Any]:
    runs = load_runs(tag)
    if not runs:
        raise SystemExit(f"no completed runs tagged {tag!r} in {registry.ARTIFACTS}")
    selected, lr_log = select_lr(runs)
    rows = aggregate(selected)
    thr = thresholds(rows)
    adir = registry.ARTIFACTS / "analysis" / tag
    adir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (adir / "curves.json").write_text(json.dumps(rows, indent=1))
    (adir / "lr_selection.json").write_text(json.dumps(lr_log, indent=1))
    (adir / "thresholds.json").write_text(json.dumps(thr, indent=1))
    flat = [{k: v for k, v in r.items() if k not in ("seed_values", "experiment_ids", "difficulty")} for r in rows]
    for r, f in zip(rows, flat, strict=True):
        f["experiment_ids"] = ";".join(r["experiment_ids"])
    with open(adir / "curves.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    figs = plot_curves(rows, tag, out_dir) + plot_curves(rows, tag, out_dir, "exact_match")
    summary = {
        "tag": tag,
        "runs": len(runs),
        "selected_runs": len(selected),
        "cells_with_lr_at_grid_edge": [c["cell"] for c in lr_log if c["lr_at_grid_edge"]],
        "figures": [str(p) for p in figs],
        "analysis_dir": str(adir),
    }
    (adir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary
