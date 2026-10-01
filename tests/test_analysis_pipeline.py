"""Frontier estimation and the headline grid-edge gate."""

import pytest

from attnratio.analysis.pipeline import HeadlineBlocked, check_headline_eligible, frontier


def _row(family, ratio, seeds, condition="L=64, num_pairs=8", n_layers=8):
    return {
        "task": "mqar",
        "family": "attention" if ratio == 1 else family,
        "n_attn": int(ratio * n_layers),
        "n_layers": n_layers,
        "attention_ratio": ratio,
        "placement": "block_end",
        "d_model": 64,
        "condition": condition,
        "metric": "token_accuracy",
        "mean": sum(seeds) / len(seeds),
        "seed_values": seeds,
        "lr_at_grid_edge": False,
    }


def test_frontier_minimum_ratio_reaching_tau():
    rows = [
        _row("gdn", 0.0, [0.5, 0.55, 0.52]),
        _row("gdn", 0.125, [0.80, 0.85, 0.82]),
        _row("gdn", 0.25, [0.95, 0.96, 0.94]),
        _row("gdn", 0.5, [0.99, 0.99, 0.99]),
        _row("gdn", 1.0, [1.0, 1.0, 1.0]),
    ]
    (f,) = frontier(rows, tau=0.9, n_boot=200)
    assert f["family"] == "gdn" and f["rc"] == 0.25
    assert f["rc_bootstrap_distribution"] == {"0.25": 1.0}
    assert not f["non_monotone"]


def test_frontier_none_when_never_reached_and_flags_non_monotone():
    rows = [_row("mamba2", r, [a]) for r, a in [(0.0, 0.3), (0.25, 0.5), (0.5, 0.6), (1.0, 0.7)]]
    assert frontier(rows, tau=0.9, n_boot=50)[0]["rc"] is None
    rows = [_row("mamba2", r, [a]) for r, a in [(0.0, 0.95), (0.25, 0.5), (1.0, 1.0)]]
    f = frontier(rows, tau=0.9, n_boot=50)[0]
    assert f["rc"] == 0.0 and f["non_monotone"]


def test_frontier_uncertainty_spreads_when_seeds_disagree():
    rows = [
        _row("gdn", 0.0, [0.2, 0.2, 0.2]),
        _row("gdn", 0.25, [0.95, 0.70, 0.98]),
        _row("gdn", 0.5, [0.99, 0.99, 0.99]),
        _row("gdn", 1.0, [1.0, 1.0, 1.0]),
    ]
    dist = frontier(rows, tau=0.9, n_boot=500)[0]["rc_bootstrap_distribution"]
    assert set(dist) == {"0.25", "0.5"} and 0.1 < dist["0.5"] < 0.9


def test_headline_gate_blocks_edge_cells():
    check_headline_eligible([{"cell": ["a"], "lr_at_grid_edge": False}])
    with pytest.raises(HeadlineBlocked):
        check_headline_eligible([{"cell": ["a"], "lr_at_grid_edge": False}, {"cell": ["b"], "lr_at_grid_edge": True}])


def test_headline_gate_requires_three_seeds():
    ok = [{"cell": ["a"], "lr_at_grid_edge": False}]
    check_headline_eligible(ok, [_row("gdn", 0.0, [0.9, 0.9, 0.9]) | {"n_seeds": 3}])
    with pytest.raises(HeadlineBlocked):
        check_headline_eligible(ok, [_row("gdn", 0.0, [0.9]) | {"n_seeds": 1}])
