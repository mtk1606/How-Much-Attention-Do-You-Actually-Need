# 2026-09-30 · 02 · Sanity check: can my attention stack learn MQAR at all?

## Reason for running

At step 1,000 to 1,500 of the pilot (entry 01), pure attention sat at token accuracy ≈ 1/P on every P
(0.26 at P = 4, 0.14 at P = 8, ...), with loss falling toward the "uniform over in-context values"
level (mean ln P ≈ 2.35 for the P ∈ {4, 8, 16, 24} mix). That is either the known MQAR plateau or a bug.

## Configuration

Ad-hoc script (not a registered run; scratch only): 2 attention layers, d = 64, 2 heads, MQAR with P = 4,
key/value vocab 32, L = 32, batch 64, AdamW lr 1e-3, 1 CPU thread, seed 0. Eval on 256 test-split examples.

## Observed result (MEASURED, unregistered)

| step | train loss | test token accuracy |
|---|---|---|
| 0 | 4.43 | 0.00 |
| 250 | 1.59 | 0.29 |
| 500 | 0.15 | 0.94 |

Stopped at step 500 to free CPU for the pilot.

## Interpretation

The attention mixer, RoPE, masking and task labels are sound: a 2-layer model leaves the 1/P plateau
between steps 250 and 500 on the easy setting. The pilot's plateau is a difficulty/schedule issue
(8 layers, L = 128, P up to 24, key vocab 128), not a bug.

## Decision

Let the pilot's two attention-only runs finish 3,000 steps. If they do not leave the plateau, recalibrate
the pilot before spending CPU on the recurrent runs.
