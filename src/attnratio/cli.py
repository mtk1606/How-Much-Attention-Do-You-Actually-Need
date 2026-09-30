"""Command-line entry point: ``attnratio <command>`` or ``python -m attnratio.cli <command>``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="attnratio")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="train one run config (YAML)")
    p.add_argument("config")
    p.add_argument("--device")

    p = sub.add_parser("sweep", help="expand and run a sweep file")
    p.add_argument("sweep")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--filter", help="substring an experiment_id must contain")
    p.add_argument("--attention-first", action="store_true", help="run high-attention configs first")

    p = sub.add_parser("match-report", help="write configs/model_match_report.csv for sweep files")
    p.add_argument("sweeps", nargs="+")
    p.add_argument("--out", default="configs/model_match_report.csv")

    p = sub.add_parser("compute-summary", help="write artifacts/compute_summary.csv from the registry")
    p.add_argument("--out", default="artifacts/compute_summary.csv")

    p = sub.add_parser("analyze", help="aggregate results and write figures/tables for a sweep tag")
    p.add_argument("tag")
    p.add_argument("--out-dir", default="figures")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.cmd == "train":
        from .training.config import load_run_config
        from .training.train import train_run

        r = train_run(load_run_config(args.config), device=args.device)
        print(
            json.dumps(
                {
                    "experiment_id": r["experiment_id"],
                    "status": r["status"],
                    "test": [{k: t[k] for k in ("seq_len", "token_accuracy", "exact_match")} for t in r["test"]],
                },
                indent=1,
            )
        )

    elif args.cmd == "sweep":
        from .training.sweep import load_sweep, run_sweep

        configs = load_sweep(args.sweep)
        if args.filter:
            configs = [c for c in configs if args.filter in c.experiment_id]
        if args.attention_first:
            configs.sort(key=lambda c: -c.model.attention_ratio)
        tokens = sum(c.steps * c.batch_size * max(s.seq_len for s in c.train) for c in configs)
        print(f"{len(configs)} runs, {tokens / 1e6:.1f}M training tokens")
        if args.dry_run:
            for c in configs:
                print(c.experiment_id)
            return
        for eid, status in run_sweep(configs, args.workers, args.threads):
            print(eid, status)

    elif args.cmd == "match-report":
        from .tracking.reports import MATCH_FIELDS, model_match_rows, write_csv
        from .training.sweep import load_sweep

        configs = [c for s in args.sweeps for c in load_sweep(s)]
        rows = model_match_rows(configs)
        write_csv(Path(args.out), rows, MATCH_FIELDS)
        print(f"wrote {len(rows)} rows to {args.out}")

    elif args.cmd == "compute-summary":
        from .tracking.reports import COMPUTE_FIELDS, compute_summary, write_csv

        rows, total = compute_summary()
        write_csv(Path(args.out), rows, COMPUTE_FIELDS)
        Path(args.out).with_suffix(".total.json").write_text(json.dumps(total, indent=1))
        print(json.dumps(total, indent=1))

    elif args.cmd == "analyze":
        from .analysis.pipeline import analyze_tag

        analyze_tag(args.tag, Path(args.out_dir))


if __name__ == "__main__":
    main()
