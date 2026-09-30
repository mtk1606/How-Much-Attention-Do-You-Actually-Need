"""Probe generators: determinism, label correctness, no answer leakage, difficulty controls, split disjointness."""

import numpy as np
import pytest

from attnratio.data.batches import eval_seeds, make_batch, train_seeds
from attnratio.data.tasks import IGNORE, TASKS, _group_table, get_task

LENGTHS = {"mqar": 128, "copy": 96, "niah": 128, "state": 64, "induction": 96}


@pytest.mark.parametrize("name", sorted(TASKS))
def test_deterministic(name):
    t = get_task(name)
    d = t.difficulty()
    a = t.generate_example(123, LENGTHS[name], d)
    b = t.generate_example(123, LENGTHS[name], d)
    c = t.generate_example(124, LENGTHS[name], d)
    assert np.array_equal(a.tokens, b.tokens) and np.array_equal(a.targets, b.targets)
    assert not np.array_equal(a.tokens, c.tokens)


@pytest.mark.parametrize("name", sorted(TASKS))
def test_tokens_in_vocab_and_lengths(name):
    t = get_task(name)
    d = t.difficulty()
    v = t.vocab_size(d)
    for s in range(20):
        e = t.generate_example(s, LENGTHS[name], d)
        assert e.tokens.shape == e.targets.shape == (LENGTHS[name],)
        assert e.tokens.min() >= 0 and e.tokens.max() < v
        scored = e.targets[e.targets != IGNORE]
        assert scored.size > 0 and scored.min() >= 1 and scored.max() < v


@pytest.mark.parametrize("name", ["mqar", "copy", "niah", "induction", "state"])
def test_split_disjoint(name):
    t = get_task(name)
    d = t.difficulty()
    tr = set(train_seeds(t, run_seed=0, step=s, batch_size=64)[i] for s in range(20) for i in range(64))
    ev = set(eval_seeds(t, d, LENGTHS[name], 1000))
    val = set(eval_seeds(t, d, LENGTHS[name], 1000, split="val"))
    assert not tr & ev and not val & ev and not tr & val
    if name != "state":  # short state sequences can coincide by chance; the seeds above are still disjoint
        tr_x = {t.generate_example(s, LENGTHS[name], d).tokens.tobytes() for s in list(tr)[:500]}
        ev_x = {t.generate_example(s, LENGTHS[name], d).tokens.tobytes() for s in list(ev)[:500]}
        assert not tr_x & ev_x


def test_mqar_labels_are_bound_values_and_not_leaked():
    t = get_task("mqar")
    d = t.difficulty({"num_pairs": 8, "noise_vocab": 16, "noise_density": 0.5})
    for s in range(50):
        e = t.generate_example(s, 128, d)
        binding = dict(zip(e.tokens[0:16:2].tolist(), e.tokens[1:16:2].tolist(), strict=True))
        qpos = np.flatnonzero(e.targets != IGNORE)
        assert len(qpos) == 8
        for q in qpos:
            assert e.targets[q] == binding[int(e.tokens[q])]
            # The value appears before the query only inside its own binding, never at the query itself.
            assert e.tokens[q] != e.targets[q]
        noise = (e.tokens[16:] > 1 + d["key_vocab"] + d["value_vocab"]).sum()
        assert noise > 0


def test_mqar_difficulty_controls_pair_count():
    t = get_task("mqar")
    for p in (4, 16, 32):
        e = t.generate_example(0, 256, t.difficulty({"num_pairs": p}))
        assert (e.targets != IGNORE).sum() == p


def test_copy_targets_equal_selected_span():
    t = get_task("copy")
    d = t.difficulty({"span_length": 10, "n_spans": 3})
    for s in range(30):
        e = t.generate_example(s, 80, d)
        start = e.meta["answer_start"]
        q_id = e.tokens[start]
        where = int(np.flatnonzero(e.tokens[:start] == q_id)[0])
        span = e.tokens[where + 1 : where + 11]
        assert np.array_equal(e.targets[start : start + 10], span)
        # Teacher forcing: the input at start+i+1 is the previous answer token, never the one to predict.
        assert np.array_equal(e.tokens[start + 1 : start + 10], span[:-1])


def test_niah_answer_and_depth():
    t = get_task("niah")
    for depth in (0.0, 0.5, 1.0):
        d = t.difficulty({"depth": depth, "n_needles": 3})
        e = t.generate_example(7, 200, d)
        key = e.tokens[-1]
        pos = e.meta["needle_position"]
        assert e.tokens[pos] == 1 and e.tokens[pos + 1] == key
        assert e.targets[-1] == e.tokens[pos + 2]
        assert abs(e.meta["needle_depth"] - depth) < 0.02
    d = t.difficulty({"transform": "successor"})
    e = t.generate_example(1, 200, d)
    pos = e.meta["needle_position"]
    v0 = 3 + d["filler_vocab"] + d["key_vocab"]
    assert e.targets[-1] == v0 + (e.tokens[pos + 2] - v0 + 1) % d["value_vocab"]


@pytest.mark.parametrize("group", ["parity", "cyclic", "s3", "s5"])
def test_state_labels_are_running_products(group):
    t = get_task("state")
    d = t.difficulty({"group": group, "modulus": 7})
    table = _group_table(d)
    e = t.generate_example(3, 50, d)
    s = 0
    for x, y in zip(e.tokens - 1, e.targets - 1, strict=True):
        s = table[s, x]
        assert s == y
    if group == "parity":
        assert np.array_equal(e.targets - 1, np.cumsum(e.tokens - 1) % 2)


def test_s5_is_a_group_table():
    table = _group_table({"group": "s5"})
    assert table.shape == (120, 120)
    assert all(sorted(row) == list(range(120)) for row in table)  # Latin square
    assert np.array_equal(table[0], np.arange(120))  # index 0 is the identity


def test_induction_scored_positions_are_copies():
    t = get_task("induction")
    d = t.difficulty({"segment_length": 8})
    for s in range(30):
        e = t.generate_example(s, 64, d)
        a, b = e.meta["first_start"], e.meta["repeat_start"]
        idx = np.flatnonzero(e.meta["induction_mask"])
        assert len(idx) == 7
        for i in idx:
            off = i - b
            assert e.targets[i] == e.tokens[a + off + 1]


def test_make_batch_shapes():
    t = get_task("mqar")
    d = t.difficulty()
    b = make_batch(t, d, 128, eval_seeds(t, d, 128, 4))
    assert b.tokens.shape == b.targets.shape == b.score_mask.shape == (4, 128)
    assert int(b.score_mask.sum()) == 4 * d["num_pairs"]
