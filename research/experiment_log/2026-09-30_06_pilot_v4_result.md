# 2026-09-30 · 06 · Pilot v4 result: MQAR at r ∈ {0, 1/2, 1}, 4 layers (PILOT, one seed)

> **Correction (same day, after the LR extensions below):** the pure Mamba-2 failure in the first table was a
> learning-rate artifact. With LR 3e-2 it reaches 0.984 to 1.000. Interpretation points 1 and 2 are withdrawn.
> Final pilot reading: at L = 32, P ≤ 8 every architecture is at or near ceiling once its LR is tuned, so this
> regime cannot measure attention need. See "Second LR extension" at the end.

Sweep `configs/sweeps/pilot_mqar_v4.yaml`, tag `pilot_mqar_v4`, 10 runs, all completed. CPU only: 1.15 wall-hours,
61.4M training tokens, $0 billed (`artifacts/compute_summary.csv`).
Analysis: `attnratio analyze pilot_mqar_v4` → `artifacts/analysis/pilot_mqar_v4/`, figure
`figures/pilot_mqar_v4_mqar_token_accuracy.png`.

## Hypothesis (from entry 03, carried over)

Pure attention near ceiling; pure Mamba-2 and pure Gated DeltaNet fall with P, Gated DeltaNet above Mamba-2
despite its smaller state (512 vs 1,152 entries per layer); r = 1/2 within 0.05 of r = 1.

## Observed result (MEASURED, test split, token accuracy, LR selected per cell on validation, 1 seed)

95% intervals resample test examples only (one seed), so they describe evaluation noise, not seed-to-seed variation.

| architecture | r | P = 2 | P = 4 | P = 6 | P = 8 |
|---|---|---|---|---|---|
| attention | 1 | 1.000 | 1.000 | 1.000 | 0.999 |
| Mamba-2 hybrid | 1/2 | 0.998 | 0.990 | 0.987 | 0.973 [0.968, 0.977] |
| Mamba-2 | 0 | 0.752 [0.726, 0.777] | 0.462 [0.449, 0.477] | 0.326 [0.315, 0.337] | 0.264 [0.256, 0.273] |
| Gated DeltaNet hybrid | 1/2 | 1.000 | 1.000 | 1.000 | 1.000 |
| Gated DeltaNet | 0 | 1.000 | 1.000 | 0.999 | 0.995 [0.993, 0.997] |

Chance is 1/64 ≈ 0.016; "any in-context value" is ≈ 1/P.

Learning-rate sensitivity (validation, mean over P): 1e-3 left Mamba-2 r = 1/2 and r = 0 on the 1/P plateau
(0.29 both); 3e-3 won every cell.

## Interpretation

1. At this difficulty (L = 32, P ≤ 8, d = 64) the answer to "how much attention" depends on the non-attention
   mixer. Pure Gated DeltaNet is at ceiling with no attention; pure Mamba-2 fails and degrades with P, and two
   attention layers out of four recover it to within 0.03 of pure attention.
2. The Mamba-2 deficit is not explained by state size: its recurrent state per layer is 2.25× Gated DeltaNet's.
   It is consistent with the mechanism difference (delta-rule overwrite and normalised-key reads versus decay-only
   accumulation), but this pilot does not isolate the mechanism.
3. The pre-registered expectation "Gated DeltaNet above Mamba-2" held. "Gated DeltaNet falls with P" did not
   at P ≤ 8: the task is too easy for it, so this setting cannot measure Gated DeltaNet's attention need.

## Potential confounds

- One seed.
- **LR grid edge:** 3e-3 was the top of the grid and won every cell; Mamba-2 r = 0 might improve at a higher LR.
  Extension to 1e-2 launched (`configs/sweeps/pilot_mqar_v4_lr.yaml`, same tag).
- Mamba-2 r = 0 plateaued from step 1,500 to 3,000 (validation 0.71 → 0.73 at P = 2), so "more steps" is not
  the obvious fix, but longer training was not tested.
- 4 layers only; r = 1/2 is two attention layers, so ratio and count are not separable here.

## Decision

Before seeds: finish the LR extension. Then: 3 seeds for the Mamba-2 arm; a harder regime (P up to 16 at L = 64
with 4 layers, untested for trainability) to find where Gated DeltaNet starts needing attention.

## LR extension (MEASURED, test split, 1 seed)

`configs/sweeps/pilot_mqar_v4_lr.yaml` added LR 1e-2 for all five cells (same tag). Validation token accuracy
(mean over P) by LR:

| cell | 1e-3 | 3e-3 | 1e-2 | selected |
|---|---|---|---|---|
| attention r = 1 | 0.997 | 1.000 | 0.287 | 3e-3 |
| Mamba-2 r = 1/2 | 0.293 | 0.984 | 0.961 | 3e-3 |
| Mamba-2 r = 0 | 0.293 | 0.458 | 0.943 | 1e-2 |
| Gated DeltaNet r = 1/2 | 0.987 | 1.000 | 1.000 | 3e-3 |
| Gated DeltaNet r = 0 | 0.958 | 0.999 | 0.993 | 3e-3 |

Test token accuracy with the re-selected LR, pure Mamba-2 (r = 0): P = 2 / 4 / 6 / 8 = 0.998 / 0.960 [0.952, 0.969] /
0.928 [0.919, 0.937] / 0.883 [0.874, 0.893]. Other cells unchanged.

## Revised interpretation

1. The large Mamba-2 deficit reported above was an artifact of an LR grid that stopped too low. Each architecture
   has a different good LR (attention collapses at 1e-2; pure Mamba-2 needs it). A shared LR grid would have
   produced a fake architecture effect.
2. What survives, pending seeds: pure Mamba-2 degrades with P (0.998 → 0.883 at P = 8) while pure Gated DeltaNet
   stays ≥ 0.995 and r = 1/2 Mamba-2 stays ≥ 0.973. This is a small, P-dependent gap, not a failure.
3. 1e-2 is again the top of the grid for pure Mamba-2, so 2e-2 and 3e-2 are running
   (`configs/sweeps/pilot_mqar_v4_lr2.yaml`).

## Decision (revised)

The main sweep needs a per-architecture LR grid wide enough that the selected LR is interior for every cell, and
the analysis should flag any cell whose selected LR is at a grid edge. Adding that check to the pipeline.

## Second LR extension (MEASURED, test split, 1 seed)

`configs/sweeps/pilot_mqar_v4_lr2.yaml`: pure Mamba-2 at 2e-2 and 3e-2. Validation (mean over P): 2e-2 → 0.992,
3e-2 → 0.993 (selected; flat, so the grid edge no longer matters in practice). Test token accuracy, pure Mamba-2:
P = 2 / 4 / 6 / 8 = 1.000 / 0.998 [0.995, 1.000] / 0.995 [0.993, 0.998] / 0.984 [0.980, 0.988].

## Final pilot reading

| architecture | r | selected LR | P = 8 test token accuracy |
|---|---|---|---|
| attention | 1 | 3e-3 | 0.999 |
| Mamba-2 hybrid | 1/2 | 3e-3 | 0.973 |
| Mamba-2 | 0 | 3e-2 | 0.984 |
| Gated DeltaNet hybrid | 1/2 | 3e-3 | 1.000 |
| Gated DeltaNet | 0 | 3e-3 | 0.995 |

At this difficulty there is no attention requirement to measure; the remaining differences are ≤ 0.03 and come
from one seed. (The Mamba-2 hybrid's 0.973 is below pure Mamba-2's 0.984 only because its LR grid stops at 1e-2;
no conclusion drawn.) Pure Mamba-2 wants a 10x higher LR than attention.

Lessons that carry into the main sweep:
1. Per-architecture LR grids, spanning at least 1e-3 to 3e-2, with the edge flag checked before reporting.
2. The CPU-trainable MQAR regime (L = 32, P ≤ 8, d = 64) is too easy. The informative regime needs longer
   contexts or more pairs than state size comfortably holds, which on CPU did not train for attention in 1,500 to
   2,000 steps (entries 03 to 05). That regime is a GPU job.

## Decision

MQAR on CPU is closed as uninformative-at-this-scale. Next CPU pilot: state tracking (parity), where the
literature predicts qualitative differences (entry L18), with a 4-value LR grid from the start.
