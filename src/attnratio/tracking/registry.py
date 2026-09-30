"""Experiment registry and compute accounting.

Each run writes artifacts/runs/<experiment_id>/ with config.yaml, metadata.yaml, metrics.jsonl,
final.json and (optionally) ckpt.pt, and appends one line to artifacts/registry.jsonl. Figures and
tables are built only from these files, so every plotted number traces to an experiment_id.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(os.environ.get("ATTNRATIO_ARTIFACTS", ROOT / "artifacts"))

# Hourly prices used for cost estimates. The CPU container this project was developed in is not
# billed per hour to me, so it is recorded as 0. GPU prices are placeholders for the launch plan and
# are overridden by ATTNRATIO_USD_PER_HOUR when a real run is billed.
USD_PER_HOUR = {"cpu": 0.0}


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {"git_commit": run("rev-parse", "HEAD") or None, "git_dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def hardware() -> dict[str, Any]:
    if torch.cuda.is_available():
        return {"device": torch.cuda.get_device_name(), "gpu_count": 1, "host": platform.node()}
    return {"device": "cpu", "cpu_threads": torch.get_num_threads(), "gpu_count": 0, "host": platform.node()}


def peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KiB


def usd_per_hour(device: str) -> float:
    env = os.environ.get("ATTNRATIO_USD_PER_HOUR")
    if env is not None:
        return float(env)
    return USD_PER_HOUR.get(device, float("nan"))


def run_dir(experiment_id: str) -> Path:
    return ARTIFACTS / "runs" / experiment_id


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def append_registry(record: dict[str, Any]) -> None:
    path = ARTIFACTS / "registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_registry(status: str | None = "completed") -> list[dict[str, Any]]:
    """Latest record per experiment_id, optionally filtered by status."""
    path = ARTIFACTS / "registry.jsonl"
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["experiment_id"]] = r
    return [r for r in latest.values() if status is None or r.get("status") == status]
