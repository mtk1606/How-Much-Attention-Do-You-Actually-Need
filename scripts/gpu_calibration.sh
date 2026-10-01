#!/usr/bin/env bash
# G0 + G1a calibration on one rented GPU, with a hard dollar guard. See docs/GPU_RUNBOOK.md.
#
#   USD_PER_HOUR=0.79 PROVIDER=RunPod bash scripts/gpu_calibration.sh
#
# Optional: BUDGET_USD (default 25), SPENT_USD (already billed before this script, default 0.5 for boot/setup),
#           WORKERS (concurrent runs, default 3), AUTO_STOP=1 (stop a RunPod pod when done, needs runpodctl).
# Stops (exit != 0) at the first failed gate: GPU tests, smoke run, G0 benchmark gate, or the dollar guard.
set -euo pipefail

: "${USD_PER_HOUR:?set USD_PER_HOUR to the actual checkout price}"
: "${PROVIDER:?set PROVIDER}"
BUDGET_USD="${BUDGET_USD:-25}"
SPENT_USD="${SPENT_USD:-0.5}"
WORKERS="${WORKERS:-3}"
START=$(date +%s)
LOG=artifacts/logs/gpu_calibration.log
mkdir -p artifacts/logs
exec > >(tee -a "$LOG") 2>&1

spent_now() { python3 -c "import time; print(round($SPENT_USD + (time.time() - $START) / 3600 * $USD_PER_HOUR, 2))"; }
finish() {
  echo "== total estimated spend: \$$(spent_now) of \$$BUDGET_USD"
  if [ "${AUTO_STOP:-0}" = "1" ] && command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID"
  else
    echo "== REMEMBER: stop/terminate the GPU instance now; billing continues until you do."
  fi
}
trap finish EXIT

echo "== environment"; nvidia-smi; python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nproc; free -g

echo "== setup"
python3 -m pip install -q -e ".[dev]"

echo "== G0 step 1: tests on GPU (numerical, generators, training, analysis)"
python3 -m pytest tests -o addopts="" -q

echo "== G0 step 2: smoke sweep on GPU"
ATTNRATIO_ARTIFACTS=artifacts/smoke_gpu python3 -m attnratio.cli sweep configs/sweeps/smoke.yaml

echo "== G0 step 3: throughput benchmark and gate"
python3 scripts/g0_benchmark.py --usd-per-hour "$USD_PER_HOUR" --provider "$PROVIDER" --workers "$WORKERS"
# (exits 3 if the projected G1a cost > $15, time > 24 h, or a data/launch bottleneck is detected)

echo "== G1a: 18 runs (pure attention / Mamba-2 / Gated DeltaNet x L in {64, 256, 512} x 2 LRs, seed 0)"
REMAINING=$(python3 -c "print($BUDGET_USD - $(spent_now) - 1.0)")  # keep $1 margin for analysis and shutdown
MAX_SECONDS=$(python3 -c "print(max(0, int($REMAINING / $USD_PER_HOUR * 3600)))")
echo "dollar guard: \$$REMAINING left -> G1a wall-clock limit ${MAX_SECONDS}s"
timeout --signal=INT "$MAX_SECONDS" python3 - <<EOF || echo "== G1a stopped by the dollar guard or an error; completed runs are kept"
import logging
from pathlib import Path
from attnratio.training.sweep import load_sweep, run_sweep
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
configs = [c for f in sorted(Path("configs/sweeps/gpu/g1a").glob("*.yaml")) for c in load_sweep(f)]
configs.sort(key=lambda c: -max(s.seq_len for s in c.train))  # longest runs first so the guard cuts short ones
print(len(configs), "G1a runs")
for eid, status in run_sweep(configs, workers=$WORKERS, threads_per_worker=2):
    print(eid, status)
EOF

echo "== analysis (calibration only, not a headline)"
python3 -m attnratio.cli analyze gpu_g1 --out-dir figures/gpu_g1 || true
python3 -m attnratio.cli compute-summary

echo "== commit results"
for d in artifacts/runs/*; do [ -f "$d/final.json" ] && git add -f "$d"/{config.yaml,metadata.yaml,metrics.jsonl,final.json}; done
git add -f artifacts/g0_benchmark.json artifacts/analysis/gpu_g1 artifacts/compute_summary.* artifacts/registry.jsonl "$LOG" figures/gpu_g1 2>/dev/null || true
git commit -m "G0 benchmark and G1a calibration results ($PROVIDER, \$$USD_PER_HOUR/h)" || true
git push || echo "push failed: copy artifacts/ back manually"
