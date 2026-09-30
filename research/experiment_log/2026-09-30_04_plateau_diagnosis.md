# 2026-09-30 · 04 · Why pure attention stays on the MQAR 1/P plateau (one-factor diagnosis)

Script: `scripts/diagnose_mqar_plateau.py` (unregistered, scratch runs; 1 seed; validation token accuracy on P = 4).
Base = the entry-02 setting that learns: 2 attention layers, d = 64, 2 heads (head dim 32), full RoPE,
L = 32, key/value vocab 32, P = 4, batch 64, AdamW defaults, lr 1e-3. Each variant changes one factor
(except `v2_task` and `pilot_like*`, which combine the pilot-v2 factors). Chance at P = 4 with the "any
in-context value" strategy ≈ 0.25.

## Observed result (MEASURED, unregistered)

| variant | change from base | step 250 | 500 | 1,000 | 1,500 |
|---|---|---|---|---|---|
| depth8 | 8 layers | 0.80 | 0.97 | 0.95 | – |
| heads4 | 4 heads (head dim 16) | 0.29 | 0.29 | 0.87 | – |
| heads4_rope025 | 4 heads, RoPE on 1/4 of each head | 0.29 | 0.38 | 0.97 | 1.00 |
| vocab64 | key/value vocab 64 | 0.27 | 0.85 | 0.96 | 0.96 |
| pairmix | train P ∈ {2, 4, 8} round-robin | 0.28 | 0.66 | 0.99 | 1.00 |
| len64 | L = 64 | 0.31 | 0.29 | 0.30 | 0.30 |
| v2_task | L = 64, vocab 64, P ∈ {2..16} | 0.28 | 0.28 | 0.27 | – |
| pilot_like | 8 layers, 4 heads, L = 64, vocab 64, P ∈ {2..16} | 0.26 | 0.27 | 0.27 | 0.28 |
| pilot_like_rope025 | pilot_like with RoPE on 1/4 of each head | 0.26 | 0.28 | 0.28 | 0.28 |

## Interpretation

Sequence length is the blocking factor within this step budget: L = 64 alone keeps the model on the plateau
for 1,500 steps, while depth, vocabulary size and mixing pair counts do not. More heads (smaller head
dimension) delays the transition, and partial RoPE shortens that delay, but partial RoPE does not unblock
L = 64. The most likely explanation is the step/example budget (Zoology-style MQAR training uses millions of
examples; this CPU budget is ~0.1M), not an implementation defect: the same code learns L = 32 in a few
hundred steps. I have not tested L = 64 for longer, so "blocked" means "not within 1,500 steps".

## Decision

Pilot v3 at L = 32 (the learnable regime on CPU): key/value vocab 64, P ∈ {2, 4, 6, 8}, 2 attention heads,
full RoPE (no non-standard attention needed). Longer contexts move to the GPU plan. If recurrent models are
also at ceiling at L = 32, P ≤ 8, the CPU pilot cannot separate architectures on MQAR and says so.
