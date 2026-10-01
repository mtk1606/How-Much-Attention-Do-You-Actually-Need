# GPU calibration runbook (G0 + G1a)

Authorised: calibration only, **$25 hard maximum** including rental, G0, G1a and any LR-edge extension (owner,
2026-10-01). No G2.

## Why this is a runbook

The development container cannot rent a GPU: provider APIs (api.runpod.io, rest.runpod.io, cloud.lambdalabs.com,
console.vast.ai) are blocked by its network policy and no provider key is configured. Either the owner runs the
script below on a rented instance, or the environment is given (a) the provider API host in its allowed domains and
(b) a provider API key as `RUNPOD_API_KEY`, after which the same script can be launched from the session.

## Choosing the instance (compare before renting)

At checkout, note for L40S and A100 80GB: hourly price, vCPUs, RAM, VRAM. Rent the L40S if its price is at most
~60% of the A100's (the workload is expected to be launch-bound, so the A100 is unlikely to be > 1.6x faster);
otherwise rent the A100. G0 then measures the chosen GPU; if G0 fails its gate on cost or time, do not switch GPUs
and retry without reporting back.

## Run

```bash
git clone -b claude/zen-dirac-waz0tr https://github.com/mtk1606/How-Much-Attention-Do-You-Actually-Need
cd How-Much-Attention-Do-You-Actually-Need
USD_PER_HOUR=<checkout price> PROVIDER=<provider> SPENT_USD=<billed so far, e.g. 0.2> bash scripts/gpu_calibration.sh
```

The script, in order, stopping at the first failure:

1. Prints `nvidia-smi`, torch/CUDA versions, CPU and RAM.
2. Runs the full test suite on the GPU, including `tests/test_gpu.py` (chunked scans vs recurrences on CUDA;
   CPU vs GPU forward agreement).
3. Runs the toy smoke sweep on the GPU.
4. **G0**: `scripts/g0_benchmark.py` times the real training step for each G1a configuration, records tokens/s,
   peak VRAM, batch-generation share and GPU utilisation, projects G1a hours and dollars, and **exits non-zero if
   projected G1a cost > $15, projected time > 24 h, batch generation > 50% of a step, or GPU utilisation < 30%**.
   Output: `artifacts/g0_benchmark.json`.
5. **Stops here by default.** Only with `RUN_G1A=1` (after G0 is reviewed): **G1a**, 18 runs, under a wall-clock limit computed from the remaining budget minus a $1 margin.
6. Calibration analysis (`attnratio analyze gpu_g1`, not a headline), compute summary, commit and push.
7. Reminds you to terminate the instance (or stops a RunPod pod with `AUTO_STOP=1`).

Then send back `artifacts/g0_benchmark.json` and `artifacts/logs/gpu_calibration.log` (or the pushed commit).
