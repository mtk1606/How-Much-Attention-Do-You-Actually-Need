"""Measure training-step throughput (tokens/s) of each mixer family on the current machine.

Used to size the pilot and the compute plan. Writes artifacts/throughput_<host>.csv.
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

import torch

from attnratio.models import HybridLM, ModelConfig, parameter_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--seq-len", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default=f"artifacts/throughput_{platform.node()}.csv")
    args = ap.parse_args()
    rows = []
    for family in ("mamba2", "gdn"):
        for n_attn in (0, args.n_layers // 2, args.n_layers):
            cfg = ModelConfig(
                family=family, vocab_size=512, d_model=args.d_model, n_layers=args.n_layers, n_attn=n_attn
            )
            model = HybridLM(cfg)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            for seq_len in args.seq_len:
                x = torch.randint(0, 512, (args.batch, seq_len))
                for i in range(args.steps + 1):
                    if i == 1:
                        t0 = time.perf_counter()
                    loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), x.flatten())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                dt = (time.perf_counter() - t0) / args.steps
                rep = parameter_report(model, seq_len)
                row = {
                    "device": "cpu" if not torch.cuda.is_available() else torch.cuda.get_device_name(),
                    "threads": torch.get_num_threads(),
                    "family": family,
                    "n_attn": n_attn,
                    "n_layers": args.n_layers,
                    "d_model": args.d_model,
                    "seq_len": seq_len,
                    "batch": args.batch,
                    "params": rep["parameter_count"],
                    "step_seconds": round(dt, 4),
                    "tokens_per_second": round(args.batch * seq_len / dt),
                }
                print(row, flush=True)
                rows.append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
