"""Parity test: CRActivation's fast (stacked-indexing) forward must produce
the same output AND the same gradient w.r.t. control points as the legacy
4-separate-gathers implementation.

Rationale: 2026-05-29 torch.profiler probe attributed 76 % of HSiKAN's
training-time CUDA to `aten::_index_put_impl_` (the backward of the 4
separate `cp[ch_idx, i{0..3}]` gathers in CRActivation). The fix stacks
the indices into a single advanced-indexing op; backward becomes one
index_put_ instead of four. Mathematically equivalent. This test pins it.
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.vision.hsikan_vision import CRActivation


@pytest.mark.parametrize("branch", [0, 1])
@pytest.mark.parametrize("shape", [(4, 8), (2, 16, 8), (3, 5, 7, 8)])
def test_cractivation_forward_parity(branch, shape):
    """Output of forward() == output of _forward_legacy() on any shape."""
    torch.manual_seed(0)
    act = CRActivation(channels=shape[-1], n_branches=2, m=8)
    x = torch.randn(*shape, requires_grad=False)
    y_fast = act.forward(x, branch_idx=branch)
    y_legacy = act._forward_legacy(x, branch_idx=branch)
    assert torch.allclose(y_fast, y_legacy, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("branch", [0, 1])
def test_cractivation_gradient_parity(branch):
    """grad w.r.t. cpts is identical under both paths."""
    torch.manual_seed(0)
    shape = (4, 16, 8)
    # Two independent activations with the SAME initialised cpts.
    act_fast = CRActivation(channels=shape[-1], n_branches=2, m=8)
    act_legacy = CRActivation(channels=shape[-1], n_branches=2, m=8)
    with torch.no_grad():
        act_legacy.cpts.copy_(act_fast.cpts)
    x = torch.randn(*shape)

    y_fast = act_fast.forward(x, branch_idx=branch).sum()
    y_fast.backward()

    y_legacy = act_legacy._forward_legacy(x, branch_idx=branch).sum()
    y_legacy.backward()

    assert torch.allclose(act_fast.cpts.grad, act_legacy.cpts.grad,
                          atol=1e-6, rtol=1e-5)


def test_x_range_extrapolation_parity():
    """Inputs outside [-3, 3] (clamped to boundary) still parity-match."""
    torch.manual_seed(1)
    act = CRActivation(channels=8, n_branches=2, m=8)
    x = torch.linspace(-5.0, 5.0, 50).unsqueeze(0).repeat(8, 1).t()  # (50,8)
    for b in (0, 1):
        y_fast = act.forward(x, branch_idx=b)
        y_legacy = act._forward_legacy(x, branch_idx=b)
        assert torch.allclose(y_fast, y_legacy, atol=1e-7)
