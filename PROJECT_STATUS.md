# Project status

Updated: 2026-09-30

## Current phase

Phase 4 (pilot). First interpretable pilot (v4, MQAR) complete; LR-grid extension running.

## Current hypothesis

See `docs/EXPERIMENT_DESIGN.md` §2. Revised from the brief: state tracking is expected to be flat or inverted in r for vanilla Mamba-2 / Gated DeltaNet (they provably cannot do parity; literature L18), so a negative-eigenvalue Gated DeltaNet arm was added. Competing explanation to test: "attention layer count, not ratio".

## Completed

- Phase 0/1: literature verification (`research/literature/verification.md`, `sources.yaml`). Corrections: Nemotron 3 Super is Mamba-heavy with 8 attention anchors among 48 mixers (r ≈ 1/6), not alternating; Qwen3.6 is r = 1/4 (every 4th layer); prior ratio sweeps exist (arXiv 2507.06457), which narrows the novelty claim.
- Phase 2: experimental design, decisions, derivations (`docs/`).
- Phase 3: Mamba-2 and Gated DeltaNet mixers in pure PyTorch, verified against recurrences and the upstream SSD reference; hybrid stack with ratio and placement; five probe generators; training loop, registry, compute accounting; statistics; CLI; Makefile; CI workflow. Tests pass locally.

## In progress

- Pure Mamba-2 at LR 2e-2 and 3e-2 (1e-2 was again the grid top).

## Blocked

- Paid GPU compute for the main sweep (Stage 5+): needs the owner's budget approval. Estimate in `docs/EXPERIMENT_DESIGN.md` §8.
- Full-text reading of arXiv PDFs (arxiv.org and huggingface.co are blocked by this container's network policy). Several literature rows are VERIFIED-EXCERPT only.

## Next three actions

1. Finish the LR extension; re-run the analysis with three LRs.
2. Seeds 1 and 2 for pilot v4, then a harder MQAR regime (P up to 16, L = 64, 4 layers) where Gated DeltaNet may need attention.
3. State-tracking and copy pilots at 4 layers; GPU cost recalibration and budget request.

## Measured results

- CPU training throughput (MEASURED, `artifacts/throughput_cpu_container.csv`).
- Parameter matching across families and ratios within 3% (MEASURED, tested).
- PILOT, one seed (log entry 06, corrected): at L = 32, P ≤ 8, d = 64, 4 layers, pure Gated DeltaNet matches pure attention on MQAR (≥ 0.995 test token accuracy); pure Mamba-2 degrades mildly with P (0.998 → 0.883) once its LR is raised to 1e-2. The earlier "Mamba-2 fails" reading (0.26 at P = 8) was an LR-grid artifact. Higher LRs for pure Mamba-2 are running.
- Methodological finding: good LRs differ by architecture by ~3x in both directions (attention collapses at 1e-2; pure Mamba-2 needs it), so a shared LR grid manufactures architecture effects.
- Trainability finding (log entries 01 to 05): at d = 64, 8-layer pure attention does not leave the MQAR plateau within ~2,000 steps on mixed pair counts, while 4 layers does; learning rate 1e-3 vs 3e-3 decides whether a Mamba-2 hybrid learns MQAR at all.

## Unverified assumptions

- Exact placement of Nemotron 3 Super's 8 attention layers.
- Layer counts L3, L15 and the Jamba ablation L14 (excerpt only).
- GPU throughput of the pure-PyTorch kernels (drives the cost estimate).

## Compute spent

CPU only, in this container; $0 billed. Registered runs so far: 10 (1.15 wall-hours, 61.4M tokens; `artifacts/compute_summary.total.json`). Abandoned pilots and unregistered diagnosis runs add roughly 2 CPU-hours, not itemised.

## Major decisions

See `docs/DECISIONS.md` (S1 to S8).
