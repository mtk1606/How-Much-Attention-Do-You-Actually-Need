"""Sweep expansion and execution.

A sweep file has a ``base`` run config and a ``grid`` of dotted keys to lists of values; the sweep is
the Cartesian product, optionally extended by explicit ``extra`` override dicts. Pure-attention
configurations (n_attn == n_layers) do not depend on the non-attention family, so they are
canonicalised to one family and run once.
"""

from __future__ import annotations

import copy
import itertools
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from ..models.hybrid import FAMILIES
from .config import RunConfig, run_config_from_dict

log = logging.getLogger("attnratio.sweep")


def _set(d: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def expand_sweep(spec: dict[str, Any]) -> list[RunConfig]:
    base = spec["base"]
    grid = spec.get("grid", {})
    keys = list(grid)
    overrides = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))]
    overrides += spec.get("extra", [])
    configs: dict[str, RunConfig] = {}
    for ov in overrides or [{}]:
        d = copy.deepcopy(base)
        for k, v in ov.items():
            _set(d, k, v)
        m = d["model"]
        if m.get("n_attn", 0) == m.get("n_layers", 8):
            m["family"] = FAMILIES[0]
            m["placement"] = "block_end"
        if m.get("n_attn", 0) == 0:
            m["placement"] = "block_end"
        d.setdefault("tags", [])
        d["tags"] = list(d["tags"]) + [spec["name"]]
        cfg = run_config_from_dict(d)
        configs.setdefault(cfg.experiment_id, cfg)
    return list(configs.values())


def load_sweep(path: str | Path) -> list[RunConfig]:
    return expand_sweep(yaml.safe_load(Path(path).read_text()))


def _worker_init(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _run(cfg_dict: dict[str, Any]) -> tuple[str, str]:
    from .train import train_run

    cfg = run_config_from_dict(cfg_dict)
    try:
        r = train_run(cfg)
        return cfg.experiment_id, r["status"]
    except Exception as e:  # recorded, the sweep continues
        log.exception("run failed: %s", cfg.experiment_id)
        return cfg.experiment_id, f"failed: {e!r}"


def run_sweep(
    configs: list[RunConfig], workers: int = 1, threads_per_worker: int | None = None
) -> list[tuple[str, str]]:
    import os

    threads = threads_per_worker or max(1, (os.cpu_count() or 1) // workers)
    dicts = [c.to_dict() | {"tags": list(c.tags)} for c in configs]
    if workers == 1:
        _worker_init(threads)
        return [_run(d) for d in dicts]
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_worker_init, initargs=(threads,)) as ex:
        futs = [ex.submit(_run, d) for d in dicts]
        for f in as_completed(futs):
            results.append(f.result())
            log.info("finished %s -> %s (%d/%d)", *results[-1], len(results), len(futs))
    return results
