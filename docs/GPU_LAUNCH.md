# GPU launch plan

Status: **PLANNED**. Nothing here has been run on a GPU. Do not launch paid compute before the owner approves a budget.

## 1. Machine

One Ampere-or-newer GPU with ≥ 24 GB (A10G, L4, A100, H100). The models are 0.35M to 5M parameters; memory is dominated by activations of the chunked scans at L = 1024 and batch 64. No multi-GPU code is needed: sweeps parallelise across runs, not within a run.

## 2. Setup

```bash
git clone https://github.com/mtk1606/How-Much-Attention-Do-You-Actually-Need
cd How-Much-Attention-Do-You-Actually-Need
make setup && make test
export ATTNRATIO_USD_PER_HOUR=<actual hourly price>   # recorded in every run's cost estimate
```

## 3. Calibrate before spending

```bash
python scripts/bench_throughput.py --d-model 128 --seq-len 256 512 1024 --batch 64 --out artifacts/throughput_gpu.csv
```

Replace the throughput assumption in `docs/EXPERIMENT_DESIGN.md` §8 with the measured value, recompute the envelope, and get approval for the number. If the pure-PyTorch recurrent mixers are the bottleneck (expected at L ≥ 1024), try `torch.compile` on the mixer forward before optimising anything by hand; re-run `make test` after any kernel change.

## 4. Order of launches (each gated on the previous)

1. `attnratio sweep configs/sweeps/<main>.yaml --dry-run` to list runs and tokens.
2. Main ratio sweep, seed 0, three LRs. Analyse; pick LRs on validation.
3. Seeds 1 and 2 at the selected LRs.
4. Scaling, depth-decoupling, placement and state-matched arms, only for cells where step 2 showed a changing response.

Several runs fit on one GPU at these sizes; run `--workers 3` or `4` and watch memory.

## 5. After each stage

```bash
make compute-summary
attnratio analyze <tag> --out-dir figures
```

Commit `artifacts/registry.jsonl`, `artifacts/runs/*/{config.yaml,metadata.yaml,metrics.jsonl,final.json}` and `artifacts/compute_summary.*`. Checkpoints are not committed (see `.gitignore`); keep them on the machine or object storage and record where in `research/experiment_log/`.
