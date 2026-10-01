# 2026-10-01 · 07 · Parity pilot result (PILOT, one seed, 4 layers, d = 64)

Sweeps: `configs/sweeps/pilot_state_v4.yaml` (28 runs), `pilot_state_v4_lr.yaml` (6 runs, LR grid extended
after the headline gate failed at the lower edge), same tag `pilot_state_v4`; `pilot_state_v4_long.yaml`
(2 convergence checks, separate tag, not used for selection). Analysis `artifacts/analysis/pilot_state_v4/`
(`headline_eligible: true` after the extension). Train L = 64; test L = 64 (in distribution), 128 and 256.
LR selected on validation (mean token accuracy over all three lengths); numbers below are test split.

## Observed result (MEASURED)

| model | r | LR | exact seq. 64 / 128 / 256 | first error 64 / 128 / 256 | token acc. 64 |
|---|---|---|---|---|---|
| Gated DeltaNet, β ∈ (0, 2) | 0 | 3e-3 | 1.00 / 1.00 / 0.996 | 64.0 / 128.0 / 255.8 | 1.000 |
| Gated DeltaNet, β ∈ (0, 2), hybrid | 1/2 | 1e-2 | 1.00 / 1.00 / 1.00 | 64.0 / 128.0 / 256.0 | 1.000 |
| Mamba-2 | 0 | 1e-3 | 0.945 / 0.055 / 0.000 | 63.7 / 92.2 / 94.2 | 0.997 |
| Gated DeltaNet | 0 | 3e-3 | 0.770 / 0.008 / 0.000 | 62.1 / 77.2 / 76.6 | 0.981 |
| Gated DeltaNet hybrid | 1/2 | 3e-3 | 0.840 / 0.000 / 0.000 | 63.0 / 74.8 / 74.4 | 0.991 |
| Mamba-2 hybrid | 1/2 | 1e-3 | 0.000 / 0.000 / 0.000 | 37.7 / 38.4 / 38.2 | 0.790 |
| attention | 1 | 3e-4 | 0.000 / 0.000 / 0.000 | 20.4 / 23.1 / 21.0 | 0.708 |

Convergence checks (LR 1e-3, 8,000 instead of 2,000 steps):

| model | r | exact seq. 64 / 128 / 256 | first error 64 / 128 / 256 | train loss at 2k → 8k |
|---|---|---|---|---|
| attention | 1 | 0.64 / 0.00 / 0.00 | 44.9 / 54.2 / 50.1 | 0.553 → 0.022 |
| Mamba-2 hybrid | 1/2 | 0.99 / 0.00 / 0.00 | 64.0 / 73.8 / 73.6 | 0.564 → 0.000 |

Exact-match intervals (bootstrap over test examples, one seed) are narrow, e.g. Mamba-2 r = 0 at L = 64:
[0.91, 0.97]. They do not include seed variation.

## Failure classification

- **Optimisation failure (at the 2,000-step budget):** pure attention and the Mamba-2 hybrid. Training loss was
  still falling at 2,000 steps; with 8,000 steps both fit the training length (attention 0.99 token accuracy,
  64% exact; hybrid 99% exact). Their headline 2,000-step numbers understate in-distribution ability.
- **Extrapolation failure:** every model except negative-eigenvalue Gated DeltaNet. Each fits L = 64 (given
  enough steps) and breaks at a roughly fixed position beyond it: first error ≈ 74 to 94 for the recurrent and
  hybrid models, ≈ 50 for long-trained attention, independent of whether the test length is 128 or 256.
- **No failure:** negative-eigenvalue Gated DeltaNet, with or without attention layers: exact on 4x the training
  length.
- **Representational failure:** not demonstrated by this experiment. The extrapolation failures are consistent
  with the theoretical limit (below), but a finite-length experiment cannot show it.

## What was measured, what the literature predicts, what I infer

- Measured: allowing β ∈ (0, 2) in the delta rule turns a model that fails beyond its training length into one
  that extrapolates parity exactly to 4x, at both r = 0 and r = 1/2. Adding attention layers did not help any
  family extrapolate; long-trained pure attention extrapolated worst.
- Literature (L18, Grazzi et al. 2025, excerpt only): linear RNNs whose transition eigenvalues lie in [0, 1]
  cannot represent parity at arbitrary length in finite precision; eigenvalues in [-1, 1] can.
- Inference (not tested mechanistically): the gdn_neg models extrapolate because they use the negative-eigenvalue
  range, and the others fit a finite-length approximation. I did not inspect learned β values or state
  trajectories, so this remains the literature's mechanism applied to my result, not a measured mechanism.

## Framing

Parity is a counterexample capability, not a measurement of "how much attention is needed". In this pilot,
attention is neither necessary nor sufficient for length-general parity; the recurrent update rule decides it.

## Limitations

One seed; 4 layers; d = 64; only Z2 (no S3/S5); attention uses RoPE, so its extrapolation failure may partly be a
positional-encoding effect; LR chosen on a validation mix that includes extrapolation lengths.
