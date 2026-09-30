"""Uncertainty and threshold analysis for capability-vs-ratio curves.

The replication unit is the training seed. Intervals come from a hierarchical bootstrap: resample
seeds, then examples within each resampled seed. Threshold claims require a logistic transition to
beat linear and constant fits by AICc on seed-level points, consistently across bootstrap resamples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def hierarchical_bootstrap(
    per_seed_examples: list[np.ndarray], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and (1 - alpha) percentile interval of the grand mean.

    per_seed_examples: one array of per-example scores for each training seed. With a single seed the
    interval reflects example sampling only, and the caller must say so.
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a, dtype=float) for a in per_seed_examples]
    point = float(np.mean([a.mean() for a in arrays]))
    k = len(arrays)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        seeds = rng.integers(0, k, size=k) if k > 1 else [0]
        stats[b] = np.mean([arrays[s][rng.integers(0, arrays[s].size, size=arrays[s].size)].mean() for s in seeds])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def logistic(r: np.ndarray, floor: float, ceil: float, rc: float, width: float) -> np.ndarray:
    return floor + (ceil - floor) / (1.0 + np.exp(np.clip(-(r - rc) / width, -60, 60)))


def _aicc(rss: float, n: int, k: int) -> float:
    rss = max(rss, 1e-12)
    aic = n * math.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1)) / max(1, n - k - 1)


@dataclass
class FitResult:
    model: str
    params: dict[str, float]
    aicc: float


def fit_models(r: np.ndarray, y: np.ndarray) -> dict[str, FitResult]:
    """Least-squares fits of constant, linear and 4-parameter logistic models of y against r."""
    r, y = np.asarray(r, float), np.asarray(y, float)
    n = r.size
    out = {"constant": FitResult("constant", {"c": float(y.mean())}, _aicc(float(((y - y.mean()) ** 2).sum()), n, 1))}
    slope, intercept = np.polyfit(r, y, 1)
    rss = float(((y - (slope * r + intercept)) ** 2).sum())
    out["linear"] = FitResult("linear", {"slope": float(slope), "intercept": float(intercept)}, _aicc(rss, n, 2))
    # Profile fit: for fixed (rc, width) the logistic is linear in (floor, ceil), so those are solved in
    # closed form (clipped to [0, 1]) over a dense grid of rc and log-spaced widths. Deterministic.
    lo_r, hi_r = float(r.min()), float(r.max())
    rcs = np.linspace(lo_r, hi_r, 101)
    widths = np.geomspace(0.005, 1.0, 16)
    z = (r[None, None, :] - rcs[:, None, None]) / widths[None, :, None]
    sig = 1.0 / (1.0 + np.exp(np.clip(-z, -60, 60)))  # (rc, w, n)
    a = 1.0 - sig  # design: y = floor * a + ceil * sig
    saa, sss, sas = (a * a).sum(-1), (sig * sig).sum(-1), (a * sig).sum(-1)
    say, ssy = (a * y).sum(-1), (sig * y).sum(-1)
    det = saa * sss - sas**2
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = np.where(det > 1e-12, (sss * say - sas * ssy) / det, y.mean())
        ceil = np.where(det > 1e-12, (saa * ssy - sas * say) / det, y.mean())
    floor, ceil = np.clip(floor, 0.0, 1.0), np.clip(ceil, 0.0, 1.0)
    pred = floor[..., None] * a + ceil[..., None] * sig
    rss_grid = ((pred - y) ** 2).sum(-1)
    i, j = np.unravel_index(int(np.argmin(rss_grid)), rss_grid.shape)
    params = {"floor": float(floor[i, j]), "ceil": float(ceil[i, j]), "rc": float(rcs[i]), "width": float(widths[j])}
    out["logistic"] = FitResult("logistic", params, _aicc(float(rss_grid[i, j]), n, 4))
    return out


def threshold_analysis(
    ratios: list[float], per_seed_scores: dict[float, list[float]], n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Decide between smooth and threshold-like response for one capability curve.

    per_seed_scores maps each ratio to its seed-level scores. Returns the preferred model on the full
    data, the fraction of seed-bootstrap resamples in which the logistic wins by ΔAICc > 4 over both
    alternatives, a bootstrap interval for r_c over those resamples, and the largest adjacent jump on
    the measured grid (always reported, model-free).
    """
    rng = np.random.default_rng(seed)
    ratios = sorted(ratios)
    r = np.concatenate([[x] * len(per_seed_scores[x]) for x in ratios])
    y = np.concatenate([per_seed_scores[x] for x in ratios])
    full = fit_models(r, y)
    means = np.array([np.mean(per_seed_scores[x]) for x in ratios])
    jumps = np.diff(means)
    j = int(np.argmax(np.abs(jumps))) if len(jumps) else 0

    wins, rcs = 0, []
    for _ in range(n_boot):
        rb, yb = [], []
        for x in ratios:
            s = np.asarray(per_seed_scores[x])
            rb.extend([x] * s.size)
            yb.extend(s[rng.integers(0, s.size, size=s.size)])
        fits = fit_models(np.array(rb), np.array(yb))
        if "logistic" in fits:
            la = fits["logistic"].aicc
            if la + 4 < fits["linear"].aicc and la + 4 < fits["constant"].aicc:
                wins += 1
                rcs.append(fits["logistic"].params["rc"])
    frac = wins / n_boot
    n_seeds_min = int(min(len(per_seed_scores[x]) for x in ratios))
    preferred = min(full.values(), key=lambda f: f.aicc)
    res: dict[str, Any] = {
        "preferred_model": preferred.model,
        "fits": {k: {"params": v.params, "aicc": v.aicc} for k, v in full.items()},
        "logistic_win_fraction": frac,
        # Seed-level resampling is meaningless with one seed; no threshold claim is made then.
        "threshold_detected": (frac >= 0.8) if n_seeds_min >= 2 else None,
        "largest_jump": {
            "from_ratio": ratios[j] if len(jumps) else None,
            "to_ratio": ratios[j + 1] if len(jumps) else None,
            "delta": float(jumps[j]) if len(jumps) else 0.0,
        },
        "n_points": int(r.size),
        "n_seeds_min": n_seeds_min,
    }
    if res["threshold_detected"]:
        lo, hi = np.quantile(rcs, [0.025, 0.975])
        res["rc"] = full["logistic"].params["rc"] if "logistic" in full else float(np.median(rcs))
        res["rc_ci95"] = [float(lo), float(hi)]
    return res
