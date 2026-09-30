"""Model assembly, attention placement, parameter matching, overfitting, checkpoints, registry, sweeps."""

import json

import pytest
import torch
import torch.nn.functional as F

from attnratio.data.batches import make_batch, train_seeds
from attnratio.data.tasks import IGNORE, get_task
from attnratio.models import FAMILIES, PLACEMENTS, HybridLM, ModelConfig, attention_schedule, parameter_report
from attnratio.training.config import load_run_config
from attnratio.training.sweep import expand_sweep


@pytest.mark.parametrize("n_layers", [8, 16])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_schedule_counts(n_layers, placement):
    for n_attn in range(n_layers + 1):
        s = attention_schedule(n_layers, n_attn, placement)
        assert len(s) == n_layers and sum(s) == n_attn


def test_schedule_layouts():
    assert attention_schedule(8, 2, "block_end") == [False, False, False, True, False, False, False, True]
    assert attention_schedule(8, 2, "block_start") == [True, False, False, False, True, False, False, False]
    assert attention_schedule(8, 2, "front") == [True, True] + [False] * 6
    assert attention_schedule(8, 2, "back") == [False] * 6 + [True, True]
    assert attention_schedule(8, 2, "middle") == [False, False, False, True, True, False, False, False]
    # Qwen3.5/3.6 layout: every 4th layer is full attention.
    assert attention_schedule(40, 10, "block_end") == [(i + 1) % 4 == 0 for i in range(40)]


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ModelConfig(family="mamba3", vocab_size=10)
    with pytest.raises(ValueError):
        ModelConfig(family="gdn", vocab_size=10, n_attn=9, n_layers=8)


@pytest.mark.parametrize("d_model", [64, 128, 256])
def test_parameter_match_within_three_percent(d_model):
    """Whole-model parameter counts across families and ratios stay within 3% of attention-only."""
    kw = {"mamba_head_dim": 8 if d_model <= 64 else 16}
    ref = parameter_report(HybridLM(ModelConfig("gdn", 256, d_model=d_model, n_attn=8, **kw)), 256)["parameter_count"]
    for fam in FAMILIES:
        for n_attn in (0, 1, 2, 4):
            m = HybridLM(ModelConfig(fam, 256, d_model=d_model, n_attn=n_attn, **kw))
            p = parameter_report(m, 256)["parameter_count"]
            assert abs(p / ref - 1) < 0.03, (fam, n_attn, p, ref)


def test_layer_types_follow_schedule():
    m = HybridLM(ModelConfig("mamba2", 32, d_model=32, n_layers=8, n_attn=2, placement="front", mamba_head_dim=8))
    assert m.layer_types() == ["attention", "attention"] + ["mamba2"] * 6


@pytest.mark.parametrize("family,n_attn", [("mamba2", 0), ("gdn", 0), ("gdn_neg", 0), ("gdn", 2), ("mamba2", 4)])
def test_overfits_fixed_batch(family, n_attn):
    """Stage 2 of the pilot protocol: every architecture must memorise one small batch."""
    torch.manual_seed(0)
    task = get_task("mqar")
    d = task.difficulty({"num_pairs": 4, "key_vocab": 16, "value_vocab": 16})
    b = make_batch(task, d, 32, train_seeds(task, 0, 0, 8))
    cfg = ModelConfig(
        family,
        64,
        d_model=32,
        n_layers=4,
        n_attn=n_attn,
        attn_heads=2,
        gdn_heads=2,
        mamba_head_dim=8,
        mamba_d_state=8,
        chunk=16,
    )
    model = HybridLM(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss = F.cross_entropy(model(b.tokens).flatten(0, 1), b.targets.flatten(), ignore_index=IGNORE)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()


def test_checkpoint_roundtrip_and_registry(tmp_path, monkeypatch):
    from attnratio.tracking import registry
    from attnratio.training.train import train_run

    monkeypatch.setattr(registry, "ARTIFACTS", tmp_path)
    cfg = load_run_config("configs/training/smoke.yaml")
    r1 = train_run(cfg)
    assert r1["status"] == "completed"
    ckpt = torch.load(tmp_path / "runs" / cfg.experiment_id / "ckpt.pt", weights_only=False)
    model = HybridLM(cfg.model)
    model.load_state_dict(ckpt["model"])
    from attnratio.evaluation.evaluate import evaluate

    spec = cfg.eval[0]
    task = get_task(cfg.task)
    again = evaluate(model, task, task.difficulty(spec.difficulty), spec.seq_len, cfg.n_eval)
    assert again["token_accuracy"] == r1["test"][0]["token_accuracy"]  # deterministic evaluation
    assert again["per_example_exact"] == r1["test"][0]["per_example_exact"]
    reg = registry.load_registry()
    assert [x["experiment_id"] for x in reg] == [cfg.experiment_id]
    # Re-running is a no-op that returns the stored result.
    r2 = train_run(cfg)
    assert json.dumps(r2, sort_keys=True) == json.dumps(r1, sort_keys=True)


def test_training_is_seed_deterministic(tmp_path, monkeypatch):
    from attnratio.tracking import registry
    from attnratio.training.train import train_run

    cfg = load_run_config("configs/training/smoke.yaml")
    losses = []
    for i in range(2):
        monkeypatch.setattr(registry, "ARTIFACTS", tmp_path / str(i))
        losses.append(train_run(cfg)["metadata"]["final_train_loss_ema"])
    assert losses[0] == losses[1]


def test_sweep_dedupes_pure_attention():
    import yaml

    with open("configs/sweeps/smoke.yaml") as f:
        spec = yaml.safe_load(f)
    configs = expand_sweep(spec)
    assert len(configs) == 5  # 2 families x {0, 1/2} + one shared r = 1
    ids = [c.experiment_id for c in configs]
    assert len(set(ids)) == len(ids)
    assert sum(c.model.n_attn == c.model.n_layers for c in configs) == 1


def test_experiment_id_changes_with_any_config_field():
    cfg = load_run_config("configs/training/smoke.yaml")
    import dataclasses

    other = dataclasses.replace(cfg, weight_decay=0.0)
    assert other.experiment_id != cfg.experiment_id
