"""Uncertainty and threshold analysis for capability-vs-ratio curves.

The replication unit is the training seed. Intervals come from a hierarchical bootstrap: resample
seeds, then examples within each resampled seed. Threshold claims require a logistic transition to
beat linear and constant fits by AICc on seed-level points, consistently across bootstrap resamples.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


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
    best = None
    lo_r, hi_r = float(r.min()), float(r.max())
    for rc0 in np.linspace(lo_r, hi_r, 7):
        for w0 in (0.02, 0.08, 0.25):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p, _ = curve_fit(
                        logistic,
                        r,
                        y,
                        p0=[max(0.0, y.min()), min(1.0, y.max()), rc0, w0],
                        bounds=([0.0, 0.0, lo_r, 1e-3], [1.0, 1.0, hi_r, 1.0]),
                        maxfev=5000,
                    )
            except (RuntimeError, ValueError):
                continue
            rss = float(((y - logistic(r, *p)) ** 2).sum())
            if best is None or rss < best[1]:
                best = (p, rss)
    if best is not None:
        p, rss = best
        params = dict(zip(("floor", "ceil", "rc", "width"), map(float, p), strict=True))
        out["logistic"] = FitResult("logistic", params, _aicc(rss, n, 4))
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
    preferred = min(full.values(), key=lambda f: f.aicc)
    res: dict[str, Any] = {
        "preferred_model": preferred.model,
        "fits": {k: {"params": v.params, "aicc": v.aicc} for k, v in full.items()},
        "logistic_win_fraction": frac,
        "threshold_detected": frac >= 0.8,
        "largest_jump": {
            "from_ratio": ratios[j] if len(jumps) else None,
            "to_ratio": ratios[j + 1] if len(jumps) else None,
            "delta": float(jumps[j]) if len(jumps) else 0.0,
        },
        "n_points": int(r.size),
        "n_seeds_min": int(min(len(per_seed_scores[x]) for x in ratios)),
    }
    if res["threshold_detected"]:
        lo, hi = np.quantile(rcs, [0.025, 0.975])
        res["rc"] = full["logistic"].params["rc"] if "logistic" in full else float(np.median(rcs))
        res["rc_ci95"] = [float(lo), float(hi)]
    return res
