"""Unit tests for the bounded phase-gated ResidualActor + train_residual (frozen base + frozen critic)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from hymeko_rl.agents.residual_actor import ResidualActor, build_residual_net, contact_gate
from hymeko_rl.train.critic_repair import ResidualTrainConfig, train_residual

LO = np.array([-1, -1, -1, -1], np.float32)
HI = np.array([1, 1, 1, 1], np.float32)


class BaseActor(nn.Module):
    """A frozen base policy exposing action_mean / action_dim / action_scale (the DAgger-actor contract)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(48, 4)
        self.action_dim = 4
        self.action_scale = 1.0

    def action_mean(self, obs):
        return torch.tanh(self.lin(obs.reshape(obs.shape[0], -1)))

    def forward(self, obs):
        return self.action_mean(obs)


def _residual_actor(eps=0.03):
    return ResidualActor(BaseActor(), build_residual_net(48, 4, hidden=32),
                         epsilon=eps, action_lo=LO, action_hi=HI)


def test_init_action_exactly_matches_base():
    ra = _residual_actor()
    obs = torch.randn(8, 6, 8)
    base = ra.base.action_mean(obs)
    gate = torch.ones(8)
    assert torch.allclose(ra.action_mean(obs, gate), base, atol=1e-7)   # zero-init residual → a ≡ base


def test_residual_is_bounded_by_epsilon():
    ra = _residual_actor(eps=0.03)
    with torch.no_grad():                                   # force a non-zero residual net
        ra.residual[-1].weight.normal_(0, 5.0)
        ra.residual[-1].bias.normal_(0, 5.0)
    obs = torch.randn(64, 6, 8)
    r = ra.raw_residual(obs)
    assert float(r.detach().abs().max()) <= 0.03 * ra.action_scale + 1e-6   # |eps*scale*tanh| ≤ eps*scale


def test_phase_gate_zeros_residual_in_approach():
    ra = _residual_actor()
    with torch.no_grad():
        ra.residual[-1].weight.normal_(0, 5.0)
        ra.residual[-1].bias.normal_(0, 5.0)
    obs = torch.randn(8, 6, 8)
    gate_off = torch.zeros(8)
    assert torch.allclose(ra.action_mean(obs, gate_off), ra.base.action_mean(obs), atol=1e-7)  # gate 0 → base
    gate_on = torch.ones(8)
    assert not torch.allclose(ra.action_mean(obs, gate_on), ra.base.action_mean(obs))          # gate 1 → corrected


def test_contact_gate_from_z():
    z = torch.tensor([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1.0]])
    g = contact_gate(z)
    assert g.tolist() == [1.0, 1.0, 0.0, 0.0]      # contact on either fingertip → 1; no contact → 0


def test_saturation_metric():
    ra = _residual_actor()
    obs = torch.randn(16, 6, 8)
    assert ra.saturation(obs) == pytest.approx(0.0, abs=1e-6)          # zero-init → no saturation
    with torch.no_grad():
        ra.residual[-1].bias.fill_(20.0)                              # huge pre-activation → tanh saturates
    assert ra.saturation(obs) > 0.99


def test_epsilon_must_be_positive():
    with pytest.raises(ValueError, match="epsilon must be positive"):
        ResidualActor(BaseActor(), build_residual_net(48, 4), epsilon=0.0, action_lo=LO, action_hi=HI)


class FrozenCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(48 + 4 + 5, 1)

    def forward(self, s, a, z):
        return self.lin(torch.cat([s.reshape(s.shape[0], -1), a, z], -1)).squeeze(-1)


def test_train_residual_updates_residual_keeps_base_and_critic_frozen():
    from hymeko_rl.eval.task_monitor import param_hash
    from hymeko_rl.train.replay import ReplayBuffer

    torch.manual_seed(0)
    ra = _residual_actor()
    critic = FrozenCritic()
    base_h, critic_h = param_hash(ra.base), param_hash(critic)
    res_h0 = param_hash(ra.residual)
    buf = ReplayBuffer(300, (6, 8), 4, priv_dim=5)
    rng = np.random.default_rng(0)
    for _ in range(64):
        buf.add(rng.standard_normal((6, 8)).astype(np.float32), (rng.random(4) * 2 - 1).astype(np.float32),
                float(rng.standard_normal()), rng.standard_normal((6, 8)).astype(np.float32), False,
                priv=np.array([1, 1, 0, 0, 0], np.float32), priv_next=np.array([1, 1, 0, 0, 0], np.float32))
    out = train_residual(ra, critic, buf, ResidualTrainConfig(steps=30, batch_size=16, log_every=0),
                         gate_fn=contact_gate)
    assert out["aborted"] is None
    assert param_hash(ra.base) == base_h and param_hash(critic) == critic_h    # base + critic frozen
    assert param_hash(ra.residual) != res_h0                                   # residual trained
    assert out["final_residual_normalized"] <= 0.03 + 1e-6                     # bounded by epsilon
