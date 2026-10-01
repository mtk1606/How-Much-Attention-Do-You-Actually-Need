# Project status

Updated: 2026-10-01

## Current phase

CPU phase frozen (log entry 08). Waiting for GPU budget approval (`docs/GPU_GO_NO_GO.md`).

## Current hypothesis

See `docs/EXPERIMENT_DESIGN.md` §2. Revised from the brief: state tracking is expected to be flat or inverted in r for vanilla Mamba-2 / Gated DeltaNet (they provably cannot do parity; literature L18), so a negative-eigenvalue Gated DeltaNet arm was added. Competing explanation to test: "attention layer count, not ratio".

## Completed

- Phase 0/1: literature verification (`research/literature/verification.md`, `sources.yaml`). Corrections: Nemotron 3 Super is Mamba-heavy with 8 attention anchors among 48 mixers (r ≈ 1/6), not alternating; Qwen3.6 is r = 1/4 (every 4th layer); prior ratio sweeps exist (arXiv 2507.06457), which narrows the novelty claim.
- Phase 2: experimental design, decisions, derivations (`docs/`).
- Phase 3: Mamba-2 and Gated DeltaNet mixers in pure PyTorch, verified against recurrences and the upstream SSD reference; hybrid stack with ratio and placement; five probe generators; training loop, registry, compute accounting; statistics; CLI; Makefile; CI workflow. Tests pass locally.

## In progress

- Nothing running.

## Blocked

- Paid GPU compute: needs the owner's approval ($25 for G0 + G1; $150 total cap) and GPU access (`docs/GPU_GO_NO_GO.md`).
- Full-text reading of arXiv PDFs (arxiv.org and huggingface.co are blocked by this container's network policy). Several literature rows are VERIFIED-EXCERPT only.

## Next three actions

1. On approval: G0 throughput benchmark on the rented GPU; replace the assumed throughput in the go/no-go doc.
2. G1 calibration (36 runs) and its gate.
3. G2 ratio map only if the G1 gate passes.

## Measured results

- CPU training throughput (MEASURED, `artifacts/throughput_cpu_container.csv`).
- Parameter matching across families and ratios within 3% (MEASURED, tested).
- PILOT, one seed (log entry 06, corrected twice): at L = 32, P ≤ 8, d = 64, 4 layers, every architecture reaches ≥ 0.97 MQAR test token accuracy once its LR is tuned (pure Mamba-2 needs 3e-2, attention 3e-3). This regime cannot measure attention need. The earlier "Mamba-2 fails" reading was an LR-grid artifact.
- PILOT, one seed (log entry 07): on parity, only negative-eigenvalue Gated DeltaNet (r = 0 and r = 1/2) extrapolates exactly to 4x the training length; every other model fits L = 64 given enough steps and fails beyond it; attention layers did not help any family extrapolate.
- Methodological finding: good LRs differ by architecture by ~3x in both directions (attention collapses at 1e-2; pure Mamba-2 needs it), so a shared LR grid manufactures architecture effects.
- Trainability finding (log entries 01 to 05): at d = 64, 8-layer pure attention does not leave the MQAR plateau within ~2,000 steps on mixed pair counts, while 4 layers does; learning rate 1e-3 vs 3e-3 decides whether a Mamba-2 hybrid learns MQAR at all.

## Unverified assumptions

- Exact placement of Nemotron 3 Super's 8 attention layers.
- Layer counts L3, L15 and the Jamba ablation L14 (excerpt only).
- GPU throughput of the pure-PyTorch kernels (drives the cost estimate).

## Compute spent

CPU only, in this container; $0 billed. Registered runs: 53 (7.5 CPU wall-hours, 448.5M tokens; `artifacts/compute_summary.total.json`). Unregistered diagnosis runs and abandoned pilots add roughly 3 CPU-hours.

## Major decisions

See `docs/DECISIONS.md` (S1 to S8).
