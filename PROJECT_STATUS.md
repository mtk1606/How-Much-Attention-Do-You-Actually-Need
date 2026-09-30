# Project status

Updated: 2026-09-30

## Current phase

Phase 3 (infrastructure) complete on CPU; Phase 4 (pilot) running.

## Current hypothesis

See `docs/EXPERIMENT_DESIGN.md` §2. Revised from the brief: state tracking is expected to be flat or inverted in r for vanilla Mamba-2 / Gated DeltaNet (they provably cannot do parity; literature L18), so a negative-eigenvalue Gated DeltaNet arm was added. Competing explanation to test: "attention layer count, not ratio".

## Completed

- Phase 0/1: literature verification (`research/literature/verification.md`, `sources.yaml`). Corrections: Nemotron 3 Super is Mamba-heavy with 8 attention anchors among 48 mixers (r ≈ 1/6), not alternating; Qwen3.6 is r = 1/4 (every 4th layer); prior ratio sweeps exist (arXiv 2507.06457), which narrows the novelty claim.
- Phase 2: experimental design, decisions, derivations (`docs/`).
- Phase 3: Mamba-2 and Gated DeltaNet mixers in pure PyTorch, verified against recurrences and the upstream SSD reference; hybrid stack with ratio and placement; five probe generators; training loop, registry, compute accounting; statistics; CLI; Makefile; CI workflow. Tests pass locally.

## In progress

- Stage 4 pilot: `configs/sweeps/pilot_mqar.yaml` (10 CPU runs).

## Blocked

- Paid GPU compute for the main sweep (Stage 5+): needs the owner's budget approval. Estimate in `docs/EXPERIMENT_DESIGN.md` §8.
- Full-text reading of arXiv PDFs (arxiv.org and huggingface.co are blocked by this container's network policy). Several literature rows are VERIFIED-EXCERPT only.

## Next three actions

1. Read the pilot; decide whether MQAR difficulty and training length are informative at d = 64, L = 128.
2. Pilot state tracking (parity, S3) with the gdn_neg arm and copy.
3. Recalibrate the GPU compute estimate on real hardware and ask for budget approval.

## Measured results

- CPU training throughput (MEASURED, `artifacts/throughput_cpu_container.csv`).
- Parameter matching across families and ratios within 3% (MEASURED, tested).
- No capability result yet.

## Unverified assumptions

- Exact placement of Nemotron 3 Super's 8 attention layers.
- Layer counts L3, L15 and the Jamba ablation L14 (excerpt only).
- GPU throughput of the pure-PyTorch kernels (drives the cost estimate).

## Compute spent

CPU only, in this container; $0 billed. Totals: `make compute-summary`.

## Major decisions

See `docs/DECISIONS.md` (S1 to S7).
