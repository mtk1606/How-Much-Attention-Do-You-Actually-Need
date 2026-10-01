"""G0: measure real training-step throughput for every G1a configuration and project G1a time and cost.

Times the actual training step (batch generation on CPU + transfer + forward + backward + AdamW) for the exact
G1a model and data configs (configs/sweeps/gpu/g1a), separately reports the share of time spent generating
batches (a CPU data bottleneck would show up there), samples GPU utilisation with nvidia-smi, records peak VRAM,
and projects G1a wall time and dollars from the measured numbers and the hourly price you pass.

Exit code 0 = G0 gate passes, 3 = gate fails (projected cost or time too high, or a likely bottleneck).

    python scripts/g0_benchmark.py --usd-per-hour 0.79 --provider RunPod --workers 3
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from attnratio.data.batches import make_batch, train_seeds
from attnratio.data.tasks import IGNORE, get_task
from attnratio.models import HybridLM, parameter_report
from attnratio.training.sweep import load_sweep
from attnratio.training.train import build_optimizer

G1A_DIR = Path("configs/sweeps/gpu/g1a")
MAX_G1A_USD = 15.0
MAX_G1A_HOURS = 24.0
MAX_DATA_FRACTION = 0.5  # more than half the step spent generating batches = fix the data path first
MIN_GPU_UTIL = 30.0  # mean GPU utilisation below this with a single worker = launch-bound; consider fixes


def sample_gpu_util(stop: threading.Event, out: list[float]) -> None:
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            out.append(float(r.stdout.strip().splitlines()[0]))
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            pass
        stop.wait(0.5)


def bench_config(cfg, device: str, steps: int, warmup: int) -> dict:
    task = get_task(cfg.task)
    model = HybridLM(cfg.model).to(device)
    opt = build_optimizer(model, cfg)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    data_s = step_s = 0.0
    for i in range(warmup + steps):
        spec = cfg.train[i % len(cfg.train)]
        t0 = time.perf_counter()
        b = make_batch(task, task.difficulty(spec.difficulty), spec.seq_len, train_seeds(task, 0, i, cfg.batch_size))
        t1 = time.perf_counter()
        tokens, targets = b.tokens.to(device), b.targets.to(device)
        loss = F.cross_entropy(model(tokens).float().flatten(0, 1), targets.flatten(), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= warmup:
            data_s += t1 - t0
            step_s += t2 - t0
    seq_len = max(s.seq_len for s in cfg.train)
    per_step = step_s / steps
    return {
        "experiment_id": cfg.experiment_id,
        "family": "attention" if cfg.model.n_attn == cfg.model.n_layers else cfg.model.family,
        "seq_len": seq_len,
        "batch_size": cfg.batch_size,
        "params": parameter_report(model, seq_len)["parameter_count"],
        "seconds_per_step": per_step,
        "data_fraction": data_s / step_s,
        "tokens_per_second": cfg.batch_size * seq_len / per_step,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
        "projected_run_hours_single": per_step * cfg.steps / 3600,
        "finite_loss": bool(torch.isfinite(loss).item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd-per-hour", type=float, required=True, help="actual checkout price of the instance")
    ap.add_argument("--provider", required=True)
    ap.add_argument("--workers", type=int, default=3, help="concurrent runs planned for G1a")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="artifacts/g0_benchmark.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.perf_counter()

    # One config per (family, L): LR and seed do not change step cost.
    configs = {}
    for f in sorted(G1A_DIR.glob("*.yaml")):
        for c in load_sweep(f):
            configs.setdefault(
                (c.model.family if c.model.n_attn < c.model.n_layers else "attention", c.train[0].seq_len), c
            )
    g1a_runs = [c for f in sorted(G1A_DIR.glob("*.yaml")) for c in load_sweep(f)]

    util: list[float] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu_util, args=(stop, util), daemon=True)
    if device == "cuda":
        sampler.start()
    rows = [bench_config(c, device, args.steps, args.warmup) for _, c in sorted(configs.items())]
    stop.set()

    by_key = {(r["family"], r["seq_len"]): r for r in rows}
    single_hours = sum(
        by_key[("attention" if c.model.n_attn == c.model.n_layers else c.model.family, c.train[0].seq_len)][
            "projected_run_hours_single"
        ]
        for c in g1a_runs
    )
    # Concurrency is assumed to give at most a linear speedup and is discounted by half until measured.
    effective_speedup = 1 + (args.workers - 1) * 0.5
    g1a_hours = single_hours / effective_speedup
    eval_overhead = 1.15  # periodic validation + final test, from CPU runs
    g1a_hours *= eval_overhead
    g1a_usd = g1a_hours * args.usd_per_hour
    mean_util = sum(util) / len(util) if util else None
    worst_data = max(r["data_fraction"] for r in rows)

    reasons = []
    if not all(r["finite_loss"] for r in rows):
        reasons.append("non-finite loss in benchmark")
    if g1a_usd > MAX_G1A_USD:
        reasons.append(f"projected G1a cost ${g1a_usd:.2f} > ${MAX_G1A_USD}")
    if g1a_hours > MAX_G1A_HOURS:
        reasons.append(f"projected G1a time {g1a_hours:.1f} h > {MAX_G1A_HOURS} h")
    if worst_data > MAX_DATA_FRACTION:
        reasons.append(f"batch generation is {worst_data:.0%} of a step (CPU data bottleneck)")
    if device == "cuda" and mean_util is not None and mean_util < MIN_GPU_UTIL:
        reasons.append(f"mean GPU utilisation {mean_util:.0f}% with one process (launch-bound)")

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "provider": args.provider,
        "usd_per_hour": args.usd_per_hour,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30 if device == "cuda" else None,
        "host": platform.node(),
        "cpu_count": __import__("os").cpu_count(),
        "ram_gb": round(
            __import__("os").sysconf("SC_PAGE_SIZE") * __import__("os").sysconf("SC_PHYS_PAGES") / 2**30, 1
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "per_config": rows,
        "mean_gpu_util_pct": mean_util,
        "benchmark_wall_seconds": time.perf_counter() - t_start,
        "g1a_runs": len(g1a_runs),
        "g1a_projected_hours": g1a_hours,
        "g1a_projected_usd": g1a_usd,
        "assumptions": {
            "workers": args.workers,
            "concurrency_speedup": effective_speedup,
            "eval_overhead": eval_overhead,
        },
        "gate_pass": not reasons,
        "gate_fail_reasons": reasons,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_config"}, indent=1))
    for r in rows:
        print(
            f"{r['family']:9s} L={r['seq_len']:4d} {r['tokens_per_second']:>12,.0f} tok/s  "
            f"data {r['data_fraction']:.0%}  VRAM {r['peak_vram_gb'] or 0:.2f} GB  "
            f"run {r['projected_run_hours_single']:.2f} h"
        )
    sys.exit(0 if not reasons else 3)


if __name__ == "__main__":
    main()
