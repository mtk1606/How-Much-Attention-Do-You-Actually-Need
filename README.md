# How Much Attention Do You Actually Need?

*Mohamed El Khoudimi*

How many softmax-attention layers does a hybrid sequence model need, and does the answer change with the capability being measured?

**Status: no final capability curve yet.** The infrastructure is built and tested; measured CPU pilots are recorded, and the main GPU sweep is pending. Live state: [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

**Research explainer:** https://mtk1606.github.io/How-Much-Attention-Do-You-Actually-Need/

## Why this question

Production models now mix a few attention layers into stacks of recurrent or state-space layers. By the definition used here (attention mixers / all sequence mixers, MLP and MoE blocks excluded), Qwen3.6-35B-A3B uses r = 1/4 with Gated DeltaNet, and Nemotron 3 Super uses about r = 1/6 with Mamba-2 (the second from a report excerpt; see [`research/literature/verification.md`](research/literature/verification.md)). Published ratio sweeps show that recall degrades as attention is removed while language-model loss barely moves. I want the finer picture: the response curve of each capability as a function of r, whether it has a transition, and how that transition moves with sequence length and memory load.

## What I am measuring

- **Architectures.** 8- and 16-layer stacks in which each layer's sequence mixer is either causal softmax attention or a non-attention mixer. Family A: Mamba-2. Family B: Gated DeltaNet, plus a negative-eigenvalue variant for state tracking. Parameter counts are matched within 3% across families and ratios (tested). Placement of the attention layers is an explicit variable.
- **Ratios.** r ∈ {0, 1/8, 1/4, 1/2, 1}, with 16-layer refinements where a transition appears.
- **Probes.** Multi-query associative recall, exact copying, needle in a haystack, finite-group state tracking (parity, Z_m, S3, S5) and induction. All synthetic, generated deterministically from seeds, with tests for label correctness, answer leakage and train/val/test disjointness.
- **Statistics.** Seeds are the replication unit; hierarchical bootstrap intervals; a threshold is claimed only when a logistic transition beats linear and constant fits by AICc in at least 80% of bootstrap resamples. LRs are chosen on a validation split and every reported number comes from a separate test split.

Full design, hypotheses, stopping rules and the compute estimate: [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md). Every departure from the original plan: [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Reproduce

```bash
make setup          # uv venv + editable install
make test           # unit, numerical and pipeline tests
make smoke          # toy sweep end to end on CPU: train, evaluate, register, analyse
make train-small    # CPU pilot sweep (about 1.5 h on 4 cores)
make figure-main    # figures from the registry
make match-report   # configs/model_match_report.csv
make compute-summary
```

The Mamba-2 and Gated DeltaNet mixers are pure PyTorch (no custom kernels), so everything runs on CPU. Their chunked algorithms are tested against token-by-token recurrences, and the Mamba-2 scan against the official minimal SSD reference from `state-spaces/mamba`.

## Repository

```
src/attnratio/models      attention, Mamba-2, Gated DeltaNet mixers; hybrid stack; placement schedules
src/attnratio/data        probe generators and seeded batching
src/attnratio/training    run config, training loop, sweeps
src/attnratio/evaluation  deterministic probe evaluation
src/attnratio/analysis    bootstrap, threshold model comparison, figures
src/attnratio/tracking    experiment registry, compute accounting, match report
configs/                  run and sweep configs
artifacts/                per-run metadata, metrics and results (registry.jsonl)
research/                 literature verification, experiment log, paper drafts
docs/                     design, decisions, derivations
```

## Limitations known before any result

- Track A models are small (0.35M to 5M parameters) and trained on the probe itself. They measure what an architecture can learn when a capability is trained, not what emerges from language-model pretraining.
- The two families have different recurrent state sizes at matched parameters; a state-matched control is planned.
- The main sweep needs a GPU; it has not been run.

## License

MIT.
