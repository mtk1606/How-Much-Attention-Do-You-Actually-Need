"""One-factor-at-a-time diagnosis of the MQAR plateau for pure attention (research log entry 04).

Starts from the configuration that learned in entry 02 (2 layers, 2 heads, L=32, vocab 32, P=4) and
changes one factor per variant. Prints validation token accuracy on P=4 every 250 steps. Not a
registered experiment; used to choose the pilot configuration.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from attnratio.data.batches import eval_seeds, make_batch, train_seeds
from attnratio.data.tasks import IGNORE, get_task
from attnratio.models import HybridLM, ModelConfig

BASE = {
    "n_layers": 2,
    "heads": 2,
    "L": 32,
    "vocab": 32,
    "pairs": [4],
    "lr": 1e-3,
    "rope": 1.0,
    "recipe": "adamw_default",
}
VARIANTS = {
    "base": {},
    "depth8": {"n_layers": 8},
    "heads4": {"heads": 4},
    "v2_task": {"L": 64, "vocab": 64, "pairs": [2, 4, 8, 16]},
    "len64": {"L": 64},
    "vocab64": {"vocab": 64},
    "pairmix": {"pairs": [2, 4, 8]},  # 16 pairs do not fit in L = 32
    "heads4_rope025": {"heads": 4, "rope": 0.25},
    "v3_task": {"n_layers": 8, "vocab": 64, "pairs": [2, 4, 6, 8]},
    "v3_vocab32": {"n_layers": 8, "vocab": 32, "pairs": [2, 4, 6, 8]},
    "v3_depth4": {"n_layers": 4, "vocab": 64, "pairs": [2, 4, 6, 8]},
    "v3_task_recipe": {"n_layers": 8, "vocab": 64, "pairs": [2, 4, 6, 8], "recipe": "pilot"},
    "pilot_like": {"n_layers": 8, "heads": 4, "L": 64, "vocab": 64, "pairs": [2, 4, 8, 16]},
    "pilot_like_rope025": {"n_layers": 8, "heads": 4, "L": 64, "vocab": 64, "pairs": [2, 4, 8, 16], "rope": 0.25},
}


def run(name: str, steps: int) -> list[dict]:
    v = {**BASE, **VARIANTS[name]}
    torch.manual_seed(0)
    task = get_task("mqar")
    diffs = [task.difficulty({"num_pairs": p, "key_vocab": v["vocab"], "value_vocab": v["vocab"]}) for p in v["pairs"]]
    d_eval = task.difficulty({"num_pairs": 4, "key_vocab": v["vocab"], "value_vocab": v["vocab"]})
    cfg = ModelConfig(
        "gdn",
        1 + 2 * v["vocab"] + 15,
        d_model=64,
        n_layers=v["n_layers"],
        n_attn=v["n_layers"],
        attn_heads=v["heads"],
        rope_fraction=v["rope"],
    )
    model = HybridLM(cfg)
    if (
        v["recipe"] == "pilot"
    ):  # the training loop's optimiser: wd 0.1 on matrices, betas (0.9, 0.98), warmup+cosine, clip 1
        from attnratio.training.config import DataSpec, RunConfig
        from attnratio.training.train import build_optimizer, lr_at

        rc = RunConfig(
            task="mqar",
            model=cfg,
            train=(DataSpec(v["L"], {"num_pairs": 2, "key_vocab": v["vocab"], "value_vocab": v["vocab"]}),),
            eval=(DataSpec(v["L"], {"num_pairs": 2, "key_vocab": v["vocab"], "value_vocab": v["vocab"]}),),
            steps=3000,
            lr=v["lr"],
        )
        opt = build_optimizer(model, rc)
    else:
        rc = None
        opt = torch.optim.AdamW(model.parameters(), lr=v["lr"])
    out, t0 = [], time.time()
    for s in range(steps + 1):
        if s % 250 == 0:
            e = make_batch(task, d_eval, v["L"], eval_seeds(task, d_eval, v["L"], 256, split="val"))
            with torch.no_grad():
                p = model(e.tokens).argmax(-1)
            acc = float(((p == e.targets) & e.score_mask).sum() / e.score_mask.sum())
            out.append({"variant": name, "step": s, "val_acc_P4": round(acc, 3), "seconds": round(time.time() - t0)})
            print(json.dumps(out[-1]), flush=True)
        if s == steps:
            break
        b = make_batch(task, diffs[s % len(diffs)], v["L"], train_seeds(task, 0, s, 64))
        loss = F.cross_entropy(model(b.tokens).flatten(0, 1), b.targets.flatten(), ignore_index=IGNORE)
        opt.zero_grad()
        loss.backward()
        if rc is not None:
            for g in opt.param_groups:
                g["lr"] = lr_at(s, rc)
            torch.nn.utils.clip_grad_norm_(model.parameters(), rc.grad_clip)
        opt.step()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    torch.set_num_threads(2)
    for name in args.variants:
        run(name, args.steps)
