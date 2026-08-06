"""The in-loop mirror-equivariant actor is exactly equivariant (before and after a gradient step).

Locks the contract the culmination relies on: the wrapped actor's deterministic mean satisfies
``a(g·s) == g·a(s)`` to float precision for any base weights (so the trained policy is two-sided by
construction), it adds no parameters, and equivariance survives an optimiser step.
"""

from __future__ import annotations

import torch

from hymeko_rl.train.sac import build_sac
from scenarios.aibo.equivariant_actor import (
    MirrorEquivariantActor,
    equivariance_residual,
    mirror_obs_flat,
    mirror_pre_act,
)


def _mirror_act(a: torch.Tensor) -> torch.Tensor:
    return -a[..., [1, 0, 3, 2]]


def _mlp_equivariant() -> MirrorEquivariantActor:
    torch.manual_seed(0)
    base, _ = build_sac("mlp", obs_dim=9, flat_dim=9, action_dim=4, action_scale=1.0, hidden=64)
    return MirrorEquivariantActor(base, mirror_obs_flat, mirror_pre_act)


def test_flat_mirror_is_involution() -> None:
    s = torch.randn(6, 9)
    assert torch.allclose(mirror_obs_flat(mirror_obs_flat(s)), s)


def test_pre_act_mirror_is_involution() -> None:
    mu = torch.randn(6, 4)
    assert torch.allclose(mirror_pre_act(mirror_pre_act(mu)), mu)


def test_actor_mean_exactly_equivariant() -> None:
    actor = _mlp_equivariant()
    s = torch.randn(8, 9)
    assert equivariance_residual(actor, mirror_obs_flat, _mirror_act, s) < 1e-6


def test_wrapper_adds_no_parameters() -> None:
    actor = _mlp_equivariant()
    assert sum(p.numel() for p in actor.parameters()) == sum(p.numel() for p in actor.base.parameters())


def test_equivariance_survives_a_gradient_step() -> None:
    actor = _mlp_equivariant()
    s = torch.randn(16, 9)
    opt = torch.optim.SGD(actor.parameters(), lr=1e-2)
    a, logp = actor.sample(s)                              # some differentiable objective through the mean
    (a.pow(2).mean() + logp.mean()).backward()
    opt.step()
    assert equivariance_residual(actor, mirror_obs_flat, _mirror_act, s) < 1e-6   # still exactly equivariant
