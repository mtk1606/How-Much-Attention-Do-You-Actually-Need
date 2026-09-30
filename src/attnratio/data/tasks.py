"""Deterministic generators for the mechanistic probes.

Every task implements

    generate_example(seed: int, sequence_length: int, difficulty: dict) -> Example

and ``vocab_size(difficulty)``. ``Example.targets`` holds -100 wherever no prediction is scored;
``targets[i]`` is the token the model must emit after reading ``tokens[: i + 1]``. Answer tokens
never appear in ``tokens`` at or before the position that predicts them (tested in
tests/test_tasks.py). Randomness comes only from ``numpy.random.default_rng(seed)``.

Token 0 is padding in every task.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

IGNORE = -100


@dataclass
class Example:
    tokens: np.ndarray  # (L,) int64
    targets: np.ndarray  # (L,) int64, IGNORE where unscored
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    name: str
    generate_example: Callable[[int, int, dict[str, Any]], Example]
    vocab_size: Callable[[dict[str, Any]], int]
    defaults: dict[str, Any]
    capability: str

    def difficulty(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        d = dict(self.defaults)
        d.update(overrides or {})
        return d


def derive_seed(*parts: Any) -> int:
    """Stable 63-bit seed from arbitrary labelled parts (never Python's salted hash())."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") >> 1


# ---------------------------------------------------------------------------------------------
# Probe 1: multi-query associative recall (MQAR, Arora et al. 2023)
# ---------------------------------------------------------------------------------------------
# Layout: k1 v1 k2 v2 ... kP vP | filler with the P keys queried at random positions.
# At a query position the input is the key and the target is its bound value.
# difficulty: num_pairs (P, bindings held at once), key_vocab (K), value_vocab,
#             noise_vocab (0 = pad filler), noise_density (fraction of non-query filler that is noise)


def _mqar_vocab(d: dict[str, Any]) -> int:
    return 1 + d["key_vocab"] + d["value_vocab"] + d["noise_vocab"]


def mqar_example(seed: int, sequence_length: int, difficulty: dict[str, Any]) -> Example:
    d = difficulty
    p, kv, vv = d["num_pairs"], d["key_vocab"], d["value_vocab"]
    if p > kv:
        raise ValueError("num_pairs cannot exceed key_vocab (keys are distinct)")
    if 4 * p > sequence_length:
        raise ValueError(f"sequence_length={sequence_length} too short for {p} pairs and queries")
    rng = np.random.default_rng(seed)
    keys = rng.choice(kv, size=p, replace=False) + 1
    vals = rng.integers(0, vv, size=p) + 1 + kv
    tokens = np.zeros(sequence_length, dtype=np.int64)
    targets = np.full(sequence_length, IGNORE, dtype=np.int64)
    tokens[0 : 2 * p : 2], tokens[1 : 2 * p : 2] = keys, vals
    free = np.arange(2 * p, sequence_length)
    qpos = np.sort(rng.choice(free, size=p, replace=False))
    order = rng.permutation(p)
    tokens[qpos], targets[qpos] = keys[order], vals[order]
    filler = np.setdiff1d(free, qpos)
    if d["noise_vocab"] and d["noise_density"] > 0:
        noisy = filler[rng.random(filler.size) < d["noise_density"]]
        tokens[noisy] = rng.integers(0, d["noise_vocab"], size=noisy.size) + 1 + kv + vv
    return Example(tokens, targets, {"query_positions": qpos.tolist(), "query_gap": (qpos - 2 * order).tolist()})


# ---------------------------------------------------------------------------------------------
# Probe 2: exact copying (Jelassi et al. 2024)
# ---------------------------------------------------------------------------------------------
# Layout: [ID_1 span_1] ... [ID_n span_n] pad*gap GO ID_q s_0 ... s_{m-1}
# The model must reproduce span_q after GO ID_q. With n_spans > 1 the other spans interfere.
# Target positions: ID_q -> s_0, s_0 -> s_1, ..., s_{m-2} -> s_{m-1} (teacher forcing).
# difficulty: span_length (m), vocab (content tokens), n_spans; the gap absorbs the remaining length.

COPY_SPECIAL = 2  # GO + first ID offset


def _copy_vocab(d: dict[str, Any]) -> int:
    return 1 + COPY_SPECIAL + d["n_spans"] + d["vocab"]


def copy_example(seed: int, sequence_length: int, difficulty: dict[str, Any]) -> Example:
    d = difficulty
    m, n, v = d["span_length"], d["n_spans"], d["vocab"]
    need = n * (m + 1) + 2 + m
    if need > sequence_length:
        raise ValueError(f"sequence_length={sequence_length} < {need} needed for copy task")
    rng = np.random.default_rng(seed)
    go, id0, content0 = 1, 1 + COPY_SPECIAL, 1 + COPY_SPECIAL + n
    spans = rng.integers(0, v, size=(n, m)) + content0
    q = int(rng.integers(0, n))
    tokens = np.zeros(sequence_length, dtype=np.int64)
    targets = np.full(sequence_length, IGNORE, dtype=np.int64)
    pos = 0
    for i in rng.permutation(n):
        tokens[pos] = id0 + i
        tokens[pos + 1 : pos + 1 + m] = spans[i]
        pos += m + 1
    gap = sequence_length - need
    start = pos + gap
    tokens[start] = go
    tokens[start + 1] = id0 + q
    tokens[start + 2 : start + 2 + m - 1] = spans[q][:-1]
    targets[start + 1 : start + 1 + m] = spans[q]
    return Example(tokens, targets, {"answer_start": start + 1, "span_length": m, "copy_distance": start + 1 - pos})


# ---------------------------------------------------------------------------------------------
# Probe 3: needle in a haystack
# ---------------------------------------------------------------------------------------------
# Haystack of filler tokens. n_needles triples [MARK key value] are inserted; the query at the end is
# [ASK key] and the target is the value bound to that key (or its successor with transform="successor").
# depth in [0, 1] fixes where the queried needle sits; depth < 0 samples it uniformly.
# difficulty: filler_vocab, key_vocab, value_vocab, n_needles, depth, transform

NIAH_MARK, NIAH_ASK = 1, 2


def _niah_vocab(d: dict[str, Any]) -> int:
    return 3 + d["filler_vocab"] + d["key_vocab"] + d["value_vocab"]


def niah_example(seed: int, sequence_length: int, difficulty: dict[str, Any]) -> Example:
    d = difficulty
    fv, kv, vv, nn_ = d["filler_vocab"], d["key_vocab"], d["value_vocab"], d["n_needles"]
    if nn_ > kv:
        raise ValueError("n_needles cannot exceed key_vocab")
    body = sequence_length - 2
    if 3 * nn_ > body:
        raise ValueError("sequence too short for the needles")
    rng = np.random.default_rng(seed)
    k0, v0 = 3 + fv, 3 + fv + kv
    tokens = rng.integers(0, fv, size=sequence_length).astype(np.int64) + 3
    targets = np.full(sequence_length, IGNORE, dtype=np.int64)
    keys = rng.choice(kv, size=nn_, replace=False) + k0
    vals = rng.integers(0, vv, size=nn_) + v0
    slots = body // 3  # non-overlapping needle slots of width 3
    if d["depth"] >= 0:
        q_slot = min(slots - 1, int(round(d["depth"] * (slots - 1))))
        others = rng.choice(np.setdiff1d(np.arange(slots), [q_slot]), size=nn_ - 1, replace=False)
        slot_idx = np.concatenate([[q_slot], others])
    else:
        slot_idx = rng.choice(slots, size=nn_, replace=False)
    for s, kk, vv_ in zip(slot_idx, keys, vals, strict=True):
        tokens[3 * s : 3 * s + 3] = [NIAH_MARK, kk, vv_]
    answer = vals[0] if d["transform"] == "identity" else v0 + (vals[0] - v0 + 1) % vv
    tokens[-2], tokens[-1] = NIAH_ASK, keys[0]
    targets[-1] = answer
    depth = (3 * slot_idx[0]) / max(1, body - 3)
    return Example(tokens, targets, {"needle_depth": float(depth), "needle_position": int(3 * slot_idx[0])})


# ---------------------------------------------------------------------------------------------
# Probe 4: state tracking over a finite group
# ---------------------------------------------------------------------------------------------
# Inputs are group elements g_1..g_L; the target at every position t is g_1 * ... * g_t.
# group: "parity" (Z_2), "cyclic" (Z_m, modular addition), "s3", "s5" (permutation composition).
# Z_2 and Z_m are abelian and solvable; S_5 is non-solvable (not in TC^0 unless TC^0 = NC^1).


def _perm_table(n: int) -> tuple[np.ndarray, np.ndarray]:
    import itertools

    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
    index = {tuple(p): i for i, p in enumerate(perms)}
    size = len(perms)
    table = np.zeros((size, size), dtype=np.int64)
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            table[i, j] = index[tuple(a[b])]  # (a o b)(x) = a(b(x)); running product s_t = s_{t-1} o g_t
    return perms, table


_TABLES: dict[str, np.ndarray] = {}


def _group_table(d: dict[str, Any]) -> np.ndarray:
    key = f"{d['group']}:{d.get('modulus')}"
    if key not in _TABLES:
        if d["group"] == "parity":
            m = 2
            _TABLES[key] = (np.arange(m)[:, None] + np.arange(m)[None]) % m
        elif d["group"] == "cyclic":
            m = d["modulus"]
            _TABLES[key] = (np.arange(m)[:, None] + np.arange(m)[None]) % m
        elif d["group"] in ("s3", "s5"):
            _TABLES[key] = _perm_table(int(d["group"][1]))[1]
        else:
            raise ValueError(f"unknown group {d['group']!r}")
    return _TABLES[key]


def _state_vocab(d: dict[str, Any]) -> int:
    return 1 + len(_group_table(d))


def state_example(seed: int, sequence_length: int, difficulty: dict[str, Any]) -> Example:
    table = _group_table(difficulty)
    rng = np.random.default_rng(seed)
    g = rng.integers(0, len(table), size=sequence_length)
    run = np.empty(sequence_length, dtype=np.int64)
    s = 0  # identity is index 0 for every table above
    for t in range(sequence_length):
        s = table[s, g[t]]
        run[t] = s
    return Example(g.astype(np.int64) + 1, run + 1, {})


# ---------------------------------------------------------------------------------------------
# Probe 5: induction
# ---------------------------------------------------------------------------------------------
# Uniform random tokens with one segment of length m repeated later. Every position is trained with
# next-token loss (so the model is an LM on this distribution); the metric is accuracy on the
# 2nd..m-th tokens of the repeat, which are predictable only by looking back at the first copy.


def _induction_vocab(d: dict[str, Any]) -> int:
    return 1 + d["vocab"]


def induction_example(seed: int, sequence_length: int, difficulty: dict[str, Any]) -> Example:
    m, v = difficulty["segment_length"], difficulty["vocab"]
    if 2 * m + 2 > sequence_length:
        raise ValueError("sequence too short for induction segment")
    rng = np.random.default_rng(seed)
    x = rng.integers(0, v, size=sequence_length + 1).astype(np.int64) + 1
    a = int(rng.integers(0, sequence_length // 2 - m + 1))
    b = int(rng.integers(a + m, sequence_length + 1 - m))
    x[b : b + m] = x[a : a + m]
    tokens, nxt = x[:-1], x[1:]
    scored = np.zeros(sequence_length, dtype=bool)
    scored[b : b + m - 1] = True  # position i predicts x[i+1]; x[b+1..b+m-1] are copies
    return Example(tokens, nxt.copy(), {"induction_mask": scored, "repeat_start": b, "first_start": a})


TASKS: dict[str, Task] = {
    "mqar": Task(
        "mqar",
        mqar_example,
        _mqar_vocab,
        {"num_pairs": 16, "key_vocab": 128, "value_vocab": 128, "noise_vocab": 0, "noise_density": 0.0},
        "content-addressable retrieval",
    ),
    "copy": Task("copy", copy_example, _copy_vocab, {"span_length": 32, "vocab": 32, "n_spans": 1}, "exact copying"),
    "niah": Task(
        "niah",
        niah_example,
        _niah_vocab,
        {
            "filler_vocab": 64,
            "key_vocab": 64,
            "value_vocab": 64,
            "n_needles": 4,
            "depth": -1.0,
            "transform": "identity",
        },
        "needle retrieval",
    ),
    "state": Task("state", state_example, _state_vocab, {"group": "parity", "modulus": 5}, "state tracking"),
    "induction": Task(
        "induction", induction_example, _induction_vocab, {"segment_length": 16, "vocab": 128}, "induction"
    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; expected one of {sorted(TASKS)}")
    return TASKS[name]
