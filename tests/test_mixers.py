"""Numerical correctness of the non-attention mixers.

The chunked algorithms are the ones used in training; the recurrences are the definitions. The
Mamba-2 scan is also checked against the minimal SSD reference vendored from state-spaces/mamba.
"""

import pytest
import torch
import torch.nn.functional as F

from attnratio.models.attention import AttentionMixer
from attnratio.models.gdn import GatedDeltaNetMixer, gated_delta_chunked, gated_delta_recurrent
from attnratio.models.mamba2 import Mamba2Mixer, ssd_chunked, ssd_recurrent
from tests.reference.ssd_minimal_upstream import ssd_minimal_discrete


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _ssd_inputs(seed: int, length: int = 64):
    g = torch.Generator().manual_seed(seed)
    b, h, p, n = 2, 3, 4, 5
    x = torch.randn(b, length, h, p, generator=g)
    dt = F.softplus(torch.randn(b, length, h, generator=g) - 1)
    a = -torch.exp(torch.rand(h, generator=g))
    bb = torch.randn(b, length, h, n, generator=g)
    cc = torch.randn(b, length, h, n, generator=g)
    return x * dt[..., None], dt * a, bb, cc


def test_ssd_chunked_matches_recurrence():
    x, log_a, b, c = _ssd_inputs(0)
    y_chunk, _ = ssd_chunked(x, log_a, b, c, chunk=16)
    y_rec = ssd_recurrent(x, log_a, b, c)
    torch.testing.assert_close(y_chunk, y_rec, rtol=1e-9, atol=1e-9)


def test_ssd_chunk_size_invariance():
    x, log_a, b, c = _ssd_inputs(1)
    y8, s8 = ssd_chunked(x, log_a, b, c, chunk=8)
    y32, s32 = ssd_chunked(x, log_a, b, c, chunk=32)
    torch.testing.assert_close(y8, y32)
    torch.testing.assert_close(s8, s32)


def test_ssd_matches_upstream_minimal_reference():
    x, log_a, b, c = _ssd_inputs(2)
    y_mine, s_mine = ssd_chunked(x, log_a, b, c, chunk=16)
    y_ref, s_ref = ssd_minimal_discrete(x, log_a, b, c, block_len=16)
    torch.testing.assert_close(y_mine, y_ref)
    torch.testing.assert_close(s_mine, s_ref)


def _gdn_inputs(seed: int, length: int = 64, neg: bool = False):
    g = torch.Generator().manual_seed(seed)
    b, h, dk, dv = 2, 3, 8, 6
    q = F.normalize(torch.randn(b, h, length, dk, generator=g), dim=-1)
    k = F.normalize(torch.randn(b, h, length, dk, generator=g), dim=-1)
    v = torch.randn(b, h, length, dv, generator=g)
    logdecay = -F.softplus(torch.randn(b, h, length, generator=g))
    beta = torch.sigmoid(torch.randn(b, h, length, generator=g)) * (2.0 if neg else 1.0)
    return q, k, v, logdecay, beta


def test_gated_delta_chunked_matches_recurrence():
    for neg in (False, True):
        q, k, v, g, beta = _gdn_inputs(3, neg=neg)
        o_chunk, _ = gated_delta_chunked(q, k, v, g, beta, chunk=16)
        o_rec = gated_delta_recurrent(q, k, v, g, beta)
        torch.testing.assert_close(o_chunk, o_rec, rtol=1e-8, atol=1e-8)


def test_gated_delta_chunk_size_invariance():
    q, k, v, g, beta = _gdn_inputs(4)
    o8, s8 = gated_delta_chunked(q, k, v, g, beta, chunk=8)
    o64, s64 = gated_delta_chunked(q, k, v, g, beta, chunk=64)
    torch.testing.assert_close(o8, o64)
    torch.testing.assert_close(s8, s64)


def test_delta_rule_retrieves_stored_value():
    """With beta = 1, no decay and orthonormal keys, reading key i returns value i exactly."""
    dk = dv = 8
    keys = torch.eye(dk)[None, None]  # (1, 1, 8, 8): key t = e_t
    vals = torch.randn(1, 1, dk, dv)
    g = torch.zeros(1, 1, dk)
    beta = torch.ones(1, 1, dk)
    _, state = gated_delta_chunked(keys, keys, vals, g, beta, chunk=8)
    torch.testing.assert_close(keys[0, 0] @ state[0, 0], vals[0, 0])


def _causality(mixer: torch.nn.Module, d: int):
    torch.manual_seed(0)
    x = torch.randn(2, 40, d)
    y = mixer(x)
    x2 = x.clone()
    x2[:, 25:] += torch.randn_like(x2[:, 25:])
    y2 = mixer(x2)
    torch.testing.assert_close(y[:, :25], y2[:, :25])
    assert not torch.allclose(y[:, 25:], y2[:, 25:])


def test_mixers_are_causal_and_shape_preserving():
    d = 32
    for mixer in (
        AttentionMixer(d, 4),
        Mamba2Mixer(d, d_inner=48, head_dim=16, d_state=8, chunk=16),
        GatedDeltaNetMixer(d, 4, 4, 8, chunk=16),
        GatedDeltaNetMixer(d, 4, 4, 8, chunk=16, allow_neg_eigval=True),
    ):
        _causality(mixer, d)


def test_mixer_gradients_are_finite_and_nonzero():
    d = 32
    for mixer in (
        AttentionMixer(d, 4),
        Mamba2Mixer(d, d_inner=48, head_dim=16, d_state=8, chunk=16),
        GatedDeltaNetMixer(d, 4, 4, 8, chunk=16),
    ):
        x = torch.randn(2, 37, d, requires_grad=True)  # length not a multiple of chunk
        mixer(x).pow(2).sum().backward()
        assert torch.isfinite(x.grad).all()
        for name, p in mixer.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name
