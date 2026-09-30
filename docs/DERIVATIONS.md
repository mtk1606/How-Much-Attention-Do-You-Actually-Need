# Derivations

## Chunked gated delta rule

Per head, with state S_t ∈ R^{d_k × d_v}, unit-norm keys, decay α_t = exp(g_t) and write strength β_t:

    S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ,        o_t = S_tᵀ q_t.

Expand the update as S_t = α_t S_{t−1} + k_t u_tᵀ with

    u_t = β_t (v_t − α_t S_{t−1}ᵀ k_t).

Inside a chunk that starts from state S_0, let γ_t = Σ_{s≤t} g_s (so e^{γ_t − γ_s} is the decay from s to t). Unrolling,

    S_t = e^{γ_t} S_0 + Σ_{s≤t} e^{γ_t − γ_s} k_s u_sᵀ.

Substituting S_{t−1} into u_t:

    u_t = β_t v_t − β_t e^{γ_t} S_0ᵀ k_t − Σ_{s<t} β_t e^{γ_t − γ_s} (k_tᵀ k_s) u_s.

Stacking rows over the chunk, with A_{ts} = β_t e^{γ_t − γ_s} k_tᵀ k_s for s < t and 0 otherwise,

    (I + A) U = diag(β) V − diag(β e^{γ}) K S_0,

so with T = (I + A)^{−1} (unit lower triangular, solved by forward substitution):

    U = U_0 − W S_0,    U_0 = T diag(β) V,    W = T diag(β e^{γ}) K.

Outputs and the carried state:

    O = diag(e^{γ}) Q S_0 + (Q Kᵀ ⊙ Γ) U,        Γ_{ts} = e^{γ_t − γ_s} for s ≤ t, else 0,
    S_c = e^{γ_c} S_0 + (diag(e^{γ_c − γ}) K)ᵀ U.

U_0, W, QKᵀ ⊙ Γ are computed for all chunks in parallel; only the S_0-dependent terms run sequentially over chunks. This is what `gated_delta_chunked` in `src/attnratio/models/gdn.py` computes, and `tests/test_mixers.py` checks it against the token-level recurrence to 1e-8 in float64, with β ∈ (0, 1) and β ∈ (0, 2).

## Parameter matching

Attention mixer: fused QKV (3d²) + output (d²) = 4d².

Gated DeltaNet: with H·d_k = d/2 and H·d_v = d, q and k projections cost d²/2 each, v, gate and output cost d² each: 4d² plus 2Hd for the decay and β projections and small conv/norm terms.

Mamba-2: in_proj d(2 d_inner + 2N + H), conv (d_inner + 2N)(k + 1), out d_inner·d, plus per-head scalars. `ModelConfig.mamba_d_inner` picks the multiple of the head dimension that brings this closest to 4d². Whole-model mismatches are reported in `configs/model_match_report.csv` and tested to be under 3%.
