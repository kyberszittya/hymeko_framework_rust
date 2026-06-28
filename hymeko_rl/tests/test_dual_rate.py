"""Rate-asymmetric dual-loop controller: fused reflex+deliberation forward, gradient to both streams, and the
two-rate cadence (deliberation recomputed every N steps, held otherwise)."""
from __future__ import annotations

import pytest
import torch

from hymeko_rl.dual_rate import DualRateController, RateAsymmetricLoop, build_dual_rate


class _HG:
    n_vertices = 6

    def dense_signed_adj(self, device: object = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        a = (torch.rand(6, 6) > 0.6).float()
        return a, a * 0.3


def _build() -> DualRateController:
    torch.manual_seed(0)
    return build_dual_rate(obs_feat=4, action_dim=3, hg_state=_HG(), n_vertices=6,
                           reflex_hidden=16, delib_hidden=16, skip="highway")


def test_shapes_and_deliberate() -> None:
    c = _build()
    obs = torch.randn(8, 6, 4)
    ctx = c.deliberate(obs)
    assert ctx.shape == (8, 16)
    assert c.action_mean(obs, ctx).shape == (8, 3)
    assert c.value(obs, ctx).shape == (8,)
    a, lp, v = c.act(obs, ctx)
    assert a.shape == (8, 3) and lp.shape == (8,) and v.shape == (8,)
    lp2, ent, _v = c.evaluate(obs, a, ctx)
    assert lp2.shape == (8,) and ent.shape == (8,) and c.n_parameters() > 0
    # context-optional (ActorCritic-compatible: deliberate every step) — drop-in for BC/PPO trainers.
    assert c.action_mean(obs).shape == (8, 3) and c.value(obs).shape == (8,)


def test_gradient_flows_to_both_streams() -> None:
    c = _build()
    obs = torch.randn(4, 6, 4)
    ctx = c.deliberate(obs)
    (c.action_mean(obs, ctx).pow(2).mean() + c.value(obs, ctx).pow(2).mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in c.reflex.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in c.deliberative.parameters())
    assert c.actor_mean.weight.grad is not None and c.actor_mean.weight.grad.abs().sum() > 0


def test_held_context_conditions_the_reflex() -> None:
    c = _build()
    obs = torch.randn(4, 6, 4)
    with torch.no_grad():
        a1 = c.action_mean(obs, torch.zeros(4, 16))
        a2 = c.action_mean(obs, torch.ones(4, 16))
    assert (a1 - a2).abs().max() > 1e-4   # the slow deliberation context actually conditions the fast reflex


def test_rate_asymmetric_cadence() -> None:
    c = _build()
    for n, t_steps, expected in [(4, 10, 3), (1, 10, 10), (3, 9, 3)]:
        loop = RateAsymmetricLoop(c, deliberate_every=n)
        obs = torch.randn(1, 6, 4)
        for _ in range(t_steps):
            loop.act_mean(obs)
        assert loop.n_deliberations == expected, (n, t_steps, loop.n_deliberations)
    with pytest.raises(ValueError):
        RateAsymmetricLoop(c, deliberate_every=0)
