"""Structural-preservation contract tests (§3-§5): zero-init residual, bounded transform, gate=0 ⇒ composite==base
bit-identically for arbitrary residual output, gradient isolation, and frozen-base assertions."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from hymeko_rl.coin_delivery.coin_residual_controller import (
    BoundedResidualTransform,
    CompositeResidualController,
    ZeroInitResidualActor,
    assert_base_absent_from_optimizer,
    assert_frozen_base,
)


class _Base(nn.Module):
    """Minimal frozen base actor with an ``action_mean`` in ±4."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(48, 4)

    def action_mean(self, obs):
        return torch.clamp(self.lin(obs), -4, 4)


def _frozen_base():
    b = _Base()
    for p in b.parameters():
        p.requires_grad_(False)
    return b


def _obs(n=8):
    torch.manual_seed(0)
    return torch.randn(n, 48)


def test_residual_zero_at_init():
    r = ZeroInitResidualActor()
    assert torch.allclose(r.residual_exec(_obs()), torch.zeros(8, 4), atol=1e-12)


def test_bounded_transform_range_and_grad():
    tr = BoundedResidualTransform(0.25)
    big = torch.tensor([[100.0, -100.0, 1000.0, -1000.0]])
    out = tr(big)
    assert out.abs().max() <= 0.25 + 1e-6
    assert torch.allclose(out, torch.tensor([[0.25, -0.25, 0.25, -0.25]]), atol=1e-4)
    assert torch.allclose(tr.grad_at(torch.zeros(1, 4)), torch.full((1, 4), 0.25))   # learnable slope at 0


def test_composite_gate0_equals_base_for_arbitrary_residual():
    base = _frozen_base()
    r = ZeroInitResidualActor()
    # force the residual net to emit large / saturated / random values
    with torch.no_grad():
        r.net[4].weight.copy_(torch.randn_like(r.net[4].weight) * 50)
        r.net[4].bias.copy_(torch.tensor([100.0, -100.0, 5.0, -5.0]))
    ctrl = CompositeResidualController(base, r)
    obs = _obs()
    comp0 = ctrl.composite_action(obs, 0.0)
    assert torch.equal(comp0, ctrl.base_action(obs))          # BIT-identical with gate off


def test_composite_gate1_at_init_equals_base():
    base = _frozen_base(); r = ZeroInitResidualActor()
    ctrl = CompositeResidualController(base, r)
    obs = _obs()
    assert torch.allclose(ctrl.composite_action(obs, 1.0), ctrl.base_action(obs), atol=1e-12)


def test_residual_gradient_zero_when_gate_off():
    base = _frozen_base(); r = ZeroInitResidualActor()
    with torch.no_grad():                                     # make residual nonzero so any leak would show
        r.net[4].weight.copy_(torch.randn_like(r.net[4].weight))
        r.net[4].bias.copy_(torch.randn_like(r.net[4].bias))
    ctrl = CompositeResidualController(base, r)
    obs = _obs()
    loss = ctrl.composite_action(obs, 0.0).sum()
    loss.backward()
    for n, p in r.named_parameters():
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)), n


def test_residual_gradient_nonzero_when_gate_on():
    base = _frozen_base(); r = ZeroInitResidualActor()
    ctrl = CompositeResidualController(base, r)
    obs = _obs()
    ctrl.composite_action(obs, 1.0).sum().backward()
    gnorm = float(r.net[4].weight.grad.norm())               # last layer gets gradient even at zero init
    assert gnorm > 0


def test_frozen_base_assertions():
    base = _frozen_base()
    assert_frozen_base(base)                                  # passes
    base.lin.weight.requires_grad_(True)
    with pytest.raises(AssertionError):
        assert_frozen_base(base)


def test_base_absent_from_optimizer():
    base = _frozen_base(); r = ZeroInitResidualActor()
    opt = torch.optim.Adam(r.parameters(), lr=3e-4)
    assert_base_absent_from_optimizer(base, opt)             # residual-only optimizer passes
    bad = torch.optim.Adam(list(r.parameters()) + [torch.nn.Parameter(torch.zeros(1))])
    # a param group holding a base tensor must raise
    base.lin.weight.requires_grad_(True)
    bad2 = torch.optim.Adam([base.lin.weight])
    with pytest.raises(AssertionError):
        assert_base_absent_from_optimizer(base, bad2)


def test_residual_contract_sha_deterministic():
    a = ZeroInitResidualActor().contract_sha256()
    b = ZeroInitResidualActor().contract_sha256()
    c = ZeroInitResidualActor(bound=0.5).contract_sha256()
    assert a == b and a != c and len(a) == 64


def test_act_numpy_gate_off_equals_base():
    base = _frozen_base(); r = ZeroInitResidualActor()
    with torch.no_grad():
        r.net[4].bias.copy_(torch.tensor([10.0, -10.0, 10.0, -10.0]))
    ctrl = CompositeResidualController(base, r)
    o = np.random.default_rng(0).standard_normal(48).astype(np.float32)
    off = ctrl.act(o, 0.0)
    base_a = ctrl.base_action(torch.as_tensor(o[None], dtype=torch.float32))[0].numpy()
    assert np.array_equal(off, base_a)
