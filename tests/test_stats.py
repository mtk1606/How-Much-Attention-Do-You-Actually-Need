"""The threshold detector must find planted steps, reject planted smooth curves, and bootstrap sensibly."""

import numpy as np

from attnratio.analysis.stats import fit_models, hierarchical_bootstrap, logistic, threshold_analysis

RATIOS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0]


def _scores(fn, noise, seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    return {r: list(np.clip(fn(r) + rng.normal(0, noise, seeds), 0, 1)) for r in RATIOS}


def test_detects_planted_threshold():
    res = threshold_analysis(RATIOS, _scores(lambda r: logistic(np.array(r), 0.05, 0.98, 0.3, 0.02), 0.02), n_boot=200)
    assert res["threshold_detected"]
    lo, hi = res["rc_ci95"]
    assert lo <= 0.3 <= hi + 0.07  # grid resolution: the step lies between 0.25 and 0.375
    assert res["largest_jump"]["from_ratio"] == 0.25 and res["largest_jump"]["to_ratio"] == 0.375


def test_rejects_linear_curve():
    res = threshold_analysis(RATIOS, _scores(lambda r: 0.2 + 0.7 * r, 0.03), n_boot=200)
    assert not res["threshold_detected"]


def test_rejects_flat_curve():
    res = threshold_analysis(RATIOS, _scores(lambda r: 0.9, 0.02), n_boot=200)
    assert not res["threshold_detected"]
    assert res["preferred_model"] in ("constant", "linear")


def test_fit_models_recovers_line():
    r = np.linspace(0, 1, 20)
    fits = fit_models(r, 0.1 + 0.5 * r)
    assert abs(fits["linear"].params["slope"] - 0.5) < 1e-9


def test_bootstrap_interval_contains_mean_and_shrinks_with_data():
    rng = np.random.default_rng(0)
    small = [rng.binomial(1, 0.7, 50) for _ in range(3)]
    large = [rng.binomial(1, 0.7, 5000) for _ in range(3)]
    m1, lo1, hi1 = hierarchical_bootstrap(small, n_boot=500)
    m2, lo2, hi2 = hierarchical_bootstrap(large, n_boot=500)
    assert lo1 <= m1 <= hi1 and lo2 <= m2 <= hi2
    assert hi2 - lo2 < hi1 - lo1


def test_bootstrap_between_seed_variance_dominates():
    """Seeds that disagree must give a wide interval even with many examples per seed."""
    per_seed = [np.full(5000, 0.2), np.full(5000, 0.9), np.full(5000, 0.5)]
    _, lo, hi = hierarchical_bootstrap(per_seed, n_boot=500)
    assert hi - lo > 0.3


def test_lr_selection_flags_grid_edge():
    from attnratio.analysis.pipeline import select_lr

    def run(lr, acc):
        return {
            "task": "mqar",
            "family": "gdn",
            "n_attn": 0,
            "n_layers": 4,
            "placement": "block_end",
            "d_model": 64,
            "lr": lr,
            "experiment_id": f"x{lr}",
            "final": {"val": [{"token_accuracy": acc}]},
        }

    _, log = select_lr([run(1e-3, 0.5), run(3e-3, 0.9), run(1e-2, 0.7)])
    assert log[0]["selected_lr"] == 3e-3 and not log[0]["lr_at_grid_edge"]
    _, log = select_lr([run(1e-3, 0.5), run(3e-3, 0.9), run(1e-2, 0.95)])
    assert log[0]["selected_lr"] == 1e-2 and log[0]["lr_at_grid_edge"]
