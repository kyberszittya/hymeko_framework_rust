"""Tests for the fused Triton CR activation kernel (parity vs the eager spline; CPU fallback)."""
from __future__ import annotations

import pytest
import torch

from hymeko_rl.cr_kernel import _HAVE_TRITON, FusedCatmullRom, fused_catmull_rom
from hymeko_rl.policy import _catmull_rom

_GPU = _HAVE_TRITON and torch.cuda.is_available()


def test_cpu_fallback_matches_eager() -> None:
    """On CPU the fused entry point falls back to the eager spline — identical values."""
    torch.manual_seed(0)
    coef = torch.randn(8, 5)
    x = torch.randn(4, 3, 8) * 1.5
    assert torch.allclose(fused_catmull_rom(x, coef), _catmull_rom(coef, x, 5), atol=1e-6)


def test_module_builds_and_runs_cpu() -> None:
    act = FusedCatmullRom(8, grid=5)
    out = act(torch.randn(4, 3, 8))
    assert out.shape == (4, 3, 8) and torch.isfinite(out).all()


def test_fused_rejects_bad_grid() -> None:
    with pytest.raises(ValueError):
        FusedCatmullRom(8, grid=3)


@pytest.mark.skipif(not _GPU, reason="needs CUDA + Triton")
def test_fused_parity_fwd_and_grads() -> None:
    """The Triton kernel matches the eager spline on forward AND both gradients (x and coef)."""
    torch.manual_seed(0)
    coef = torch.randn(16, 5, device="cuda", requires_grad=True)
    x = (torch.randn(32, 4, 16, device="cuda") * 1.5).requires_grad_(True)
    out_f = fused_catmull_rom(x, coef)
    gx_f, gc_f = torch.autograd.grad(out_f.sum(), [x, coef])
    c2 = coef.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    out_e = _catmull_rom(c2, x2, 5)
    gx_e, gc_e = torch.autograd.grad(out_e.sum(), [x2, c2])
    assert torch.allclose(out_f, out_e, atol=1e-4)
    assert torch.allclose(gx_f, gx_e, atol=1e-4)
    assert torch.allclose(gc_f, gc_e, atol=1e-3)   # atomic-add + fp32 -> looser tol
