"""GPU numerical checks (skipped without CUDA). Run as the first step of G0 on the rented machine."""

import pytest
import torch
import torch.nn.functional as F

from attnratio.models import HybridLM, ModelConfig
from attnratio.models.gdn import gated_delta_chunked, gated_delta_recurrent
from attnratio.models.mamba2 import ssd_chunked, ssd_recurrent

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_chunked_scans_match_recurrence_on_gpu():
    g = torch.Generator(device="cuda").manual_seed(0)
    b, h, length = 2, 3, 128
    x = torch.randn(b, length, h, 4, device="cuda", generator=g)
    log_a = -F.softplus(torch.randn(b, length, h, device="cuda", generator=g))
    bb = torch.randn(b, length, h, 5, device="cuda", generator=g)
    cc = torch.randn(b, length, h, 5, device="cuda", generator=g)
    y, _ = ssd_chunked(x, log_a, bb, cc, chunk=32)
    torch.testing.assert_close(y, ssd_recurrent(x, log_a, bb, cc), rtol=1e-4, atol=1e-4)

    q = F.normalize(torch.randn(b, h, length, 8, device="cuda", generator=g), dim=-1)
    k = F.normalize(torch.randn(b, h, length, 8, device="cuda", generator=g), dim=-1)
    v = torch.randn(b, h, length, 6, device="cuda", generator=g)
    gl = -F.softplus(torch.randn(b, h, length, device="cuda", generator=g))
    beta = 2 * torch.sigmoid(torch.randn(b, h, length, device="cuda", generator=g))
    o, _ = gated_delta_chunked(q, k, v, gl, beta, chunk=32)
    torch.testing.assert_close(o, gated_delta_recurrent(q, k, v, gl, beta), rtol=1e-4, atol=1e-4)


@cuda
@pytest.mark.parametrize("family,n_attn", [("mamba2", 0), ("gdn", 0), ("gdn", 8), ("gdn_neg", 2)])
def test_gpu_forward_matches_cpu(family, n_attn):
    torch.backends.cudnn.allow_tf32 = False  # the short convolutions would otherwise run in TF32 on Ampere/Ada
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    cfg = ModelConfig(family, 528, d_model=128, n_layers=8, n_attn=n_attn, mamba_head_dim=16, chunk=64)
    model = HybridLM(cfg).eval()
    x = torch.randint(0, 528, (2, 256))
    with torch.no_grad():
        cpu = model(x)
        gpu = model.cuda()(x.cuda()).cpu()
    torch.testing.assert_close(gpu, cpu, rtol=2e-3, atol=2e-3)
