# 2026-09-30 · 05 · Pilot v3 stalls at depth 8; depth 4 learns; pilot v4

## Observed result (MEASURED)

Pilot v3 (`configs/sweeps/pilot_mqar_v3.yaml`, L = 32, key/value vocab 64, P ∈ {2, 4, 6, 8}, 8 layers, 2 heads),
pure attention, validation token accuracy P = 2 / 4 / 6 / 8:

| run | step | loss | accuracy |
|---|---|---|---|
| attn, lr 1e-3 | 1,000 | 1.61 | 0.51 / 0.27 / 0.21 / 0.18 |
| attn, lr 3e-3 | 1,000 | 1.62 | 0.52 / 0.27 / 0.19 / 0.16 |

Diagnosis on the exact v3 data (`scripts/diagnose_mqar_plateau.py`, unregistered, 1 seed, val accuracy at P = 4):

| variant | step 250 | 500 | 750 | 1,000 | 1,500 |
|---|---|---|---|---|---|
| v3_task (8 layers, plain AdamW) | 0.26 | 0.27 | 0.26 | – | – |
| v3_task_recipe (8 layers, training-loop optimiser) | 0.28 | 0.27 | 0.27 | – | – |
| v3_vocab32 (8 layers, vocab 32) | – | 0.30 | – | 0.28 | – |
| v3_depth4 (4 layers, vocab 64) | – | 0.96 | – | 0.98 | 0.99 |

The recipe runs were stopped at step 750 and v3_vocab32 after step 1,000, once the comparison was clear.

## Interpretation

At d = 64, an 8-layer pure-attention stack trained on a mix of pair counts stays on the 1/P plateau for at least
1,000 steps regardless of optimiser recipe or vocabulary size; the same data at 4 layers is solved by step 500.
(Depth 8 with a single pair count did learn in entry 04.) This is an optimisation effect of depth at small width,
and it matters for the main experiment: at depth 8 the r = 1 baseline itself can fail to train on a CPU budget,
which would masquerade as an architecture effect. The GPU plan must check that every r = 1 cell leaves the
plateau, and report trainability separately from capability.

## Decision

Pilot v4 uses 4 layers (r ∈ {0, 1/2, 1} = 0, 2, 4 attention layers). r = 1/8 and the depth-decoupling arm need
8 or 16 layers and move to the GPU plan, where longer training is affordable. Recorded as S8 in docs/DECISIONS.md.
