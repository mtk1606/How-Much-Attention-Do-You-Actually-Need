# Public claim audit

This file tracks the claims used on the project website in `site/`. The website is intentionally conservative: pilot results are labelled as pilots, one-seed results are not presented as final capability curves, and planned GPU work is not described as completed.

## Hero and design

| Public claim | Repository evidence |
|---|---|
| Attention ratios are 0, 1/8, 1/4, 1/2, 1 | `docs/EXPERIMENT_DESIGN.md` §3 and §5.2 |
| Primary recurrent families are Mamba-2 and Gated DeltaNet | `docs/EXPERIMENT_DESIGN.md` §3; `docs/DECISIONS.md` S1–S3 |
| Parameter counts are matched within 3% | `README.md`, `PROJECT_STATUS.md`, experiment design controlled variables |
| Five probes are MQAR, copy, NIAH, state tracking and induction | `docs/EXPERIMENT_DESIGN.md` §6 |
| LR is selected on validation and reported numbers use a separate test split | `docs/DECISIONS.md` S7; `docs/EXPERIMENT_DESIGN.md` §5.2 |
| Threshold is reported only when logistic beats linear by ΔAIC > 4 in ≥80% of bootstrap resamples | `docs/EXPERIMENT_DESIGN.md` §7 |
| Main capability sweep has not run | `PROJECT_STATUS.md` current phase / blocked / next actions |

## MQAR pilot

| Public claim | Repository evidence |
|---|---|
| Initial pure Mamba-2 P=8 token accuracy was 0.264 | `research/experiment_log/2026-09-30_06_pilot_v4_result.md`, first observed-results table |
| After LR expansion, pure Mamba-2 P=8 accuracy was 0.984 at LR 3e-2 | same log, second LR extension / final pilot reading |
| Pure attention selected LR 3e-3 and collapsed at 1e-2 | same log, LR extension table |
| Final P=8 accuracies are all ≥0.973 across the five architecture cells | same log, final pilot reading |
| The regime is too easy to measure attention need | same log, final pilot reading and decision |
| Pilot is one seed | log title and limitations |

The website deliberately does not repeat the withdrawn interpretation that “Mamba-2 fails MQAR.”

## Parity pilot

| Public claim | Repository evidence |
|---|---|
| Train length 64; test lengths 64, 128 and 256 | `research/experiment_log/2026-10-01_07_parity_result.md` |
| Negative-eigenvalue Gated DeltaNet r=0 exact sequence accuracy: 1.00 / 1.00 / 0.996 | measured-results table |
| Negative-eigenvalue Gated DeltaNet r=1/2: 1.00 / 1.00 / 1.00 | measured-results table |
| Other tested model families fail to extrapolate beyond the training length | failure-classification section |
| Adding attention did not help any family extrapolate in this pilot | “What was measured” section |
| Pilot is one seed, four layers, d=64, Z2 only | limitations section |

The website uses the repository's framing: parity is a counterexample capability, not a final measurement of “how much attention is needed.”

## Engineering and compute

| Public claim | Repository evidence |
|---|---|
| Mamba-2 is checked against token recurrence and upstream minimal SSD reference | `README.md`; `PROJECT_STATUS.md`; mixer tests |
| Deterministic probe generators and train/val/test separation are tested | `README.md`; `tests/test_tasks.py`; `docs/EXPERIMENT_DESIGN.md` |
| 53 registered CPU runs | `PROJECT_STATUS.md` |
| 7.5 registered CPU wall-hours | `PROJECT_STATUS.md` |
| 448.5M tokens | `PROJECT_STATUS.md` |
| $0 billed compute so far | `PROJECT_STATUS.md` |
| GPU calibration has a $25 hard maximum and is blocked on GPU access | `PROJECT_STATUS.md` |

## Scope boundaries

The site does **not** claim:
- a final critical attention ratio;
- production-LM behavior from the small CPU pilots;
- a main-sweep result;
- GPU throughput that has not been measured;
- that attention is universally unnecessary for parity or any other capability.

Any later website update that adds a main result should update this file at the same time.
