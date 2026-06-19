"""Actor-critic contract + the agent-spec + the build_policy Strategy.

Run: pytest -p no:randomly hymeko_rl/tests/test_policy.py
"""
from __future__ import annotations

import pytest
import torch

from hymeko_rl.agent import AgentSpec
from hymeko_rl.policy import POLICY_KINDS, ActorCritic, build_policy, mlp_backbone


def test_mlp_actor_critic_shapes_and_shared_backbone() -> None:
    torch.manual_seed(0)
    ac = build_policy("mlp", obs_dim=8, action_dim=4, hidden=32)
    assert isinstance(ac, ActorCritic) and ac.n_parameters() > 0
    obs = torch.randn(5, 8)
    action, log_prob, value = ac.act(obs)
    assert action.shape == (5, 4) and log_prob.shape == (5,) and value.shape == (5,)
    # evaluate scores a *given* action and shares the backbone forward.
    lp2, entropy, value2 = ac.evaluate(obs, action)
    assert lp2.shape == (5,) and entropy.shape == (5,) and value2.shape == (5,)
    # differential entropy of a Gaussian is finite but may be negative (σ < 0.24).
    assert torch.isfinite(entropy).all() and torch.isfinite(log_prob).all()
    # act() is no-grad (rollout); evaluate() carries grad (the PPO update).
    assert not log_prob.requires_grad and lp2.requires_grad


def test_evaluate_logprob_matches_distribution_for_the_mean_action() -> None:
    torch.manual_seed(1)
    ac = build_policy("mlp", obs_dim=3, action_dim=2)
    obs = torch.randn(4, 3)
    # the action mean is the deterministic actor output; its log-prob must be finite.
    mean = ac.action_mean(obs)
    lp, _, _ = ac.evaluate(obs, mean)
    assert torch.isfinite(lp).all()


def test_actor_and_critic_backbones_are_separate() -> None:
    """The fix for PPO degrading the actor: separate networks. The actor/critic backbones
    share no parameters, and a value-loss backward updates the critic backbone but leaves
    the actor backbone untouched (so the value loss can never corrupt the policy)."""
    torch.manual_seed(0)
    ac = build_policy("mlp", obs_dim=6, action_dim=2, hidden=16)
    actor_ids = {id(p) for p in ac.actor_backbone.parameters()}
    critic_ids = {id(p) for p in ac.critic_backbone.parameters()}
    assert actor_ids and critic_ids and actor_ids.isdisjoint(critic_ids)

    actor_before = [p.clone() for p in ac.actor_backbone.parameters()]
    opt = torch.optim.SGD(ac.parameters(), lr=0.1)
    loss = ac.value(torch.randn(4, 6)).pow(2).mean()   # a pure value-head loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert all(torch.equal(b, p)
               for b, p in zip(actor_before, ac.actor_backbone.parameters()))


def test_build_policy_errors() -> None:
    with pytest.raises(TypeError):
        build_policy("hsikan", obs_dim=4, action_dim=2)   # hg_state is mandatory
    with pytest.raises(ValueError):
        build_policy("not_a_kind", obs_dim=4, action_dim=2)
    assert "mlp" in POLICY_KINDS and "hsikan" in POLICY_KINDS


def test_mlp_backbone_validates_dims() -> None:
    with pytest.raises(ValueError):
        mlp_backbone(0)
    bb, feat = mlp_backbone(6, hidden=16, depth=1)
    assert feat == 16 and bb(torch.randn(2, 6)).shape == (2, 16)


def test_actor_critic_rejects_bad_dims() -> None:
    bb, _ = mlp_backbone(4)
    bb2, _ = mlp_backbone(4)
    with pytest.raises(ValueError):
        ActorCritic(bb, bb2, feat_dim=0, action_dim=2)


@pytest.mark.parametrize(
    "kw", [dict(obs_dim=0, action_dim=1), dict(obs_dim=1, action_dim=0),
           dict(obs_dim=1, action_dim=1, action_low=1.0, action_high=1.0),
           dict(obs_dim=1, action_dim=1, max_steps=0)])
def test_agent_spec_rejects_invalid(kw: dict) -> None:
    base = dict(obs_dim=4, action_dim=2, action_low=-1.0, action_high=1.0,
                reward="reach")
    base.update(kw)
    with pytest.raises(ValueError):
        AgentSpec(**base)


def test_agent_spec_valid() -> None:
    spec = AgentSpec(obs_dim=12, action_dim=4, action_low=-2.0, action_high=2.0,
                     reward="reach_grasp_lift", max_steps=300)
    assert spec.action_dim == 4 and spec.max_steps == 300
