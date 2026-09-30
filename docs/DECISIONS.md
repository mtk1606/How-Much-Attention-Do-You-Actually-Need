# Decisions and substitutions

Every change to the brief's plan, with the reason. Newest last.

## S1. Family A is Mamba-2, not Mamba-3 (2026-09-30)

The brief targets the Mamba family and names Mamba-3. I implement Mamba-2 because (a) Nemotron 3 Super, the production Mamba hybrid I compare against, uses Mamba-2 (verification L2/L3); (b) Mamba-3's complex-valued, trapezoidal, MIMO update would need its own verified pure-PyTorch implementation and I cannot read the paper's full text from this environment (arXiv blocked), so I cannot check an implementation against it. Mamba-3 is a candidate replication arm once the paper can be read.

## S2. Family B is Gated DeltaNet (v1), with Gated DeltaNet-2 deferred (2026-09-30)

Qwen3.5/3.6 use Gated DeltaNet v1 (L7). Matching the production mixer makes the production comparison cleaner. Gated DeltaNet-2's channel-wise decay and decoupled erase/write gates (L11) do not admit the same scalar-decay chunked algorithm I verified here; a recurrent-only implementation would be too slow for the sweep on CPU. GDN-2 is a planned extension.

## S3. Negative-eigenvalue Gated DeltaNet added for state tracking (2026-09-30)

Literature (L18) shows delta-rule and Mamba layers with transition eigenvalues in [0, 1] cannot solve parity at finite precision. Testing only vanilla layers would measure a known impossibility. I add `gdn_neg` (β ∈ (0, 2)) as a third family used on the state-tracking probe.

## S4. Track A models are 0.35M to 5M parameters, not 100M to 500M (2026-09-30)

The brief's default range is for language-model pretraining. The mechanistic-probe protocol (train on the probe, evaluate in and out of distribution) is established at small width (Zoology, Repeat After Me) and lets me afford LR tuning and three seeds per cell. Larger LM-pretrained models are Track B, conditional on Track A results. See `docs/EXPERIMENT_DESIGN.md` §4.

## S5. Attention ratio counts sequence mixers only (2026-09-30)

MLP and MoE blocks are excluded from both numerator and denominator, so r is comparable across production models that interleave MoE layers differently (L3: Nemotron 3 Super is r = 1/6 by this definition and 9.1% if MoE layers are counted).

## S6. Pure-attention configurations are run once per sweep (2026-09-30)

With n_attn = n_layers the non-attention family is irrelevant. The sweep expander canonicalises such configs, so r = 1 appears once and is shared by both family curves.

## S7. Hyperparameters are selected on a validation split; results are reported on a disjoint test split (2026-09-30)

Selecting the best LR on the same examples that are reported would inflate every cell by the max over LRs of the noise. Seeds for val and test come from separate namespaces (`src/attnratio/data/batches.py`).
