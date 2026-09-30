# Experimental design

Status: **PLANNED** unless a section says otherwise. Written 2026-09-30, before any pilot result existed.

## 1. Question

How much softmax attention does a hybrid sequence model need, and does the answer differ by capability?

Prior work already shows the coarse version: in 340M and 1.3B hybrids, language-model loss is flat across linear-to-full ratios while recall improves with more full attention (Wang et al. 2025, arXiv 2507.06457, **LITERATURE**, excerpt only). What is not established, and what this project measures, is the **shape** of each capability's response to the attention ratio under controlled conditions, whether it has a transition, and how that transition moves with sequence length L and memory load (number of bindings P, key cardinality K).

## 2. Hypotheses

- **H1 (HYPOTHESIS).** Content-addressable retrieval (MQAR, NIAH) shows a threshold-like response in r, and the critical ratio r_c rises with memory load P and with L.
- **H2 (HYPOTHESIS, revised).** Exact copying behaves like retrieval, with r_c rising with span length.
- **H3 (HYPOTHESIS, revised after verification L18).** On state tracking, vanilla Mamba-2 and Gated DeltaNet fail parity at any ratio below 1 beyond the training length, and attention does not fix length generalization either. Gated DeltaNet with negative eigenvalues (β ∈ (0, 2)) solves parity at r = 0. So the state-tracking curve should be *flat or inverted* in r, not a smoother version of the retrieval curve. The original brief said SSM layers would "remain strong" here; the literature says that is only true for the negative-eigenvalue variant.
- **H4 (HYPOTHESIS).** Induction needs at least one attention layer at small scale, but its emergence step is insensitive to r above that.
- **H0 (null, what would falsify the project thesis).** All capabilities share one response shape: once a small fixed number of attention layers is present, every probe is at ceiling, and remaining differences are within seed noise.

A competing explanation I will test directly: **the relevant variable is the number of attention layers, not the ratio.** A single attention layer can implement lookup. If r_c is really "one layer", then an 8-layer and a 16-layer model with one attention layer each (r = 1/8 vs 1/16) will perform the same. The depth-decoupling arm (section 5.4) separates the two.

## 3. Variables

| Kind | Variable | Values |
|---|---|---|
| Independent | attention ratio r = attention mixers / all mixers | 0, 1/8, 1/4, 1/2, 1 (8 layers); refinements with 16 layers (1/16, 3/16, ...) |
| Independent | placement | block_end (Qwen-style), block_start, front, back, middle |
| Independent | non-attention family | Mamba-2; Gated DeltaNet; Gated DeltaNet with negative eigenvalues (state tracking only) |
| Independent | sequence length L | 128 (CPU pilot); 256, 512, 1024 (GPU) |
| Independent | memory load | MQAR pairs P; NIAH needles; copy span length |
| Independent | width d (scale) | 64, 128, 256 |
| Dependent | probe metrics | token accuracy on scored positions, per-example exact match, first-error position, emergence step |
| Controlled | params | matched within 3% across families and ratios (tested) |
| Controlled | optimiser | AdamW (0.9, 0.98), wd 0.1 on matrices, cosine to 10%, 5% warmup, clip 1.0 |
| Controlled | data | identical deterministic train streams per seed across all architectures |
| Controlled | tokens | same steps × batch × L within a comparison |
| Measured, not controlled | recurrent state size | Mamba-2: d_inner·N; GDN: H·d_k·d_v. Differ between families (e.g. d=512: 10,752 vs 32,768). Handled by the state-matched control in 5.6 |
| Measured, not controlled | wall-clock and FLOPs | attention's L² term is not in the 6ND estimate; reported separately |

## 4. Two tracks

**Track A (primary, this repository now).** Each model is trained on one probe distribution and evaluated in and out of distribution. This measures what each architecture can represent and learn when the capability is trained, which is the cleanest attribution to architecture. It is the Zoology/Jelassi protocol. Models are 0.35M to 5M parameters. This is a deliberate departure from the brief's 100M to 500M default: for directly trained mechanistic probes, the literature's evidence is at this scale, the cost is two orders of magnitude lower, and the multi-seed, LR-tuned grid is affordable only here.

**Track B (PLANNED, GPU only, conditional on Track A).** Pretrain 30M and ~125M hybrids on a text corpus at 3 ratios and evaluate the same probes zero-shot / in-context. This tests whether Track A's curves predict emergent behaviour. It is not started until Track A shows capability-specific differences, per the stopping rules.

Trained capability and emergent capability are reported separately and never merged into one curve.

## 5. Arms

5.1 **Pilot (Stage 4).** r ∈ {0, 1/2, 1}, both families, MQAR, 2 LRs, 1 seed, d = 64, 8 layers, L = 128. Exit condition: the probe separates r = 0 from r = 1 at some difficulty, and r = 1 is near ceiling.

5.2 **Main ratio sweep (Stage 5).** r ∈ {0, 1/8, 1/4, 1/2, 1} × 2 families × 5 probes, LR ∈ {3e-4, 1e-3, 3e-3} on seed 0, best LR per cell by *validation* split, then seeds 1 and 2 at that LR. All reported numbers come from the *test* split.

5.3 **Load and length scaling (MQAR, NIAH).** P ∈ {8, 16, 32, 64} and L ∈ {256, 512, 1024} at the ratios bracketing the transition found in 5.2.

5.4 **Depth decoupling.** n_attn = 1 and 2 at depth 8 and 16. Distinguishes "ratio" from "count".

5.5 **Placement.** At the two most informative ratios from 5.2 (expected 1/8 and 1/4), 5 placements × 2 families × 3 seeds.

5.6 **State-matched control.** Rerun the transition cells with d_state chosen so both families have equal recurrent state per layer.

5.7 **Width.** d ∈ {64, 128, 256} at transition cells: does r_c move with state size?

## 6. Probes

Implemented in `src/attnratio/data/tasks.py`, tested in `tests/test_tasks.py`.

| Probe | Capability | Main difficulty knobs | Metric |
|---|---|---|---|
| MQAR | content-addressable recall | P, K, L, noise density | token accuracy over queries |
| Copy | information preservation | span length, n spans (interference), gap | token accuracy, exact match, first error |
| NIAH | long-range single retrieval with distractors | L, depth, n needles, successor transform | accuracy |
| State | finite-group state tracking | group (Z2, Z_m, S3, S5), train vs test length | per-position accuracy, exact match, length generalization |
| Induction | in-context bigram copying | segment length, vocab | accuracy on repeat positions; emergence step across checkpoints |

## 7. Statistics

- Unit of replication: training seed. Examples within a run are not independent replicates of the architecture effect; they give per-run precision only.
- Per cell: mean over 3 seeds, 95% interval from a hierarchical bootstrap (resample seeds, then examples within seed).
- **Threshold definition.** For each probe and condition, fit accuracy(r) with (a) a monotone smooth model (isotonic, for reference), (b) a 4-parameter logistic in r with floor = chance and a free ceiling, (c) a linear model. Report r_c = the logistic midpoint, with a bootstrap CI, **only if** the logistic beats linear by ΔAIC > 4 across bootstrap resamples at least 80% of the time. Otherwise report "no threshold detected" and the slope.
- With 5 ratio points, a midpoint between two grid points is interpolation. When a transition is detected between two ratios I add refinement points (16-layer depth) inside the interval instead of reporting the grid ratio as r_c.
- Scaling of r_c with P or L: compare constant, linear in log P, and linear in P/L by AIC; report the preferred model and the others.

## 8. Compute envelope

**MEASURED** (this container, 4 CPU threads, fp32, `artifacts/throughput_cpu_container.csv`): d = 128, 8 layers, batch 32: 6.3k to 16k tokens/s depending on family and ratio; d = 64, L = 128: 9k to 34k tokens/s.

**ESTIMATE, not measured** (GPU, to be recalibrated with `scripts/bench_throughput.py` on the rented machine before launch):

| Arm | Runs | Tokens/run | Notes |
|---|---|---|---|
| 5.2 main sweep | 5 probes × 9 architectures × (3 LR + 2 seeds) = 225 | ~160M (5k steps × 64 × 512) | r = 1 is shared across families |
| 5.3 scaling | ~90 | ~160M | |
| 5.4 depth | ~24 | ~160M | |
| 5.5 placement | ~48 | ~160M | |
| 5.6, 5.7 controls | ~40 | ~160M | |
| **Total** | **~430** | **~70B tokens** | |

At an assumed 0.5M to 1.5M tokens/s for a 1 to 5M-parameter model in pure PyTorch on one A100-class GPU, that is 13 to 40 GPU-hours, or about **$25 to $100** at $1.5 to $2.5 per GPU-hour. The dominant uncertainty is my kernels' GPU throughput; the estimate can be wrong by 3x. Track B (125M LM pretraining at 3 ratios, 2.5B tokens each) would add roughly 20 to 60 GPU-hours. I will not launch any paid compute without confirming the budget with the owner.

## 9. Stopping rules

- A run whose loss is non-finite is recorded as diverged and not retried at the same LR.
- If the pilot shows r = 0 already at ceiling on a probe at every difficulty I can afford, that probe is too easy: raise difficulty before the sweep; if impossible, drop it and say so.
- If r = 1 cannot solve a probe at a difficulty where r = 0 fails, the probe cannot measure attention need at that difficulty.
- The placement arm runs only at ratios where 5.2 found the response to be changing.
- Track B starts only if Track A shows at least one capability-specific difference that survives LR tuning and seeds.

## 10. Largest validity risks

1. **Optimisation and implementation confound.** A pure-PyTorch recurrent mixer at small scale can underperform because of LR sensitivity or a subtle bug, and that would look like an attention requirement. Mitigations: numerical tests against recurrences and the upstream SSD reference (done), overfit tests per family (done), per-cell LR selection on a validation split, loss curves stored for every run.
2. **State size, not ratio.** Recall capacity of a recurrent layer scales with its state. Families differ in state size at matched parameters, and r_c may be a property of state size. Mitigations: state size recorded per model, state-matched control (5.6), width sweep (5.7).
3. **Scale and training-regime transfer.** Thresholds measured on 1M-parameter models trained on the probe may not predict production LMs. Mitigations: Track B; explicit scope statements; comparison with published 340M/1.3B sweeps (L13) rather than claims about production models.

## 11. What counts as success, and what falsifies the thesis

Success: at least two probes whose critical ratios have non-overlapping 95% intervals (or where one has a detected transition and another does not), replicated in both families, surviving LR tuning and the state-matched control, plus a measured dependence of r_c on P or L that a constant model fits worse.

Falsification: every probe reaches ceiling by the same small number of attention layers with overlapping intervals (H0); or the differences vanish under LR tuning or state matching; or r_c does not depend on P or L. Any of these is reported as the result.
