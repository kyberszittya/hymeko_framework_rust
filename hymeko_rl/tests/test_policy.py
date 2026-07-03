"""Actor-critic contract + the agent-spec + the build_policy Strategy.

Run: pytest -p no:randomly hymeko_rl/tests/test_policy.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.agents.agent import AgentSpec
from hymeko_rl.agents.hypergraph_state import HypergraphState
from hymeko_rl.agents.policy import (
    POLICY_KINDS,
    ActorCritic,
    build_policy,
    hsikan_backbone,
    mlp_backbone,
)


def _toy_hg() -> HypergraphState:
    """A 3-vertex signed kinematic chain (l0-l1-l2) — no MuJoCo needed."""
    return HypergraphState(
        ("l0", "l1", "l2"),
        np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64),
        np.array([1, -1, 1, -1], dtype=np.int64),
        "test-topo")


def test_catmull_rom_parity_with_signedkan() -> None:
    """The self-contained CR activation matches hymeko_neuro's canonical ``_catmull_rom_eval``
    (so the HSiKAN backbone's KAN nonlinearity is consistent with the rest of the framework)."""
    from hymeko_rl.agents.policy import _catmull_rom
    torch.manual_seed(0)
    n_channels, grid = 6, 5
    coef = torch.randn(n_channels, grid)
    x = torch.randn(4, n_channels) * 2.0   # outside [-1,1] — exercises the clamp
    mine = _catmull_rom(coef, x, grid)
    try:
        from hymeko_neuro.hyperedge.splines import _catmull_rom_eval
    except Exception:  # noqa: BLE001 - cross-package import is environment-dependent
        pytest.skip("hymeko_neuro not importable in this environment")
    ref = _catmull_rom_eval(coef.unsqueeze(0).expand(x.shape[0], -1, -1), x, grid)
    assert torch.allclose(mine, ref, atol=1e-6)


def test_gomb_actor_critic_shared_trunk() -> None:
    """AC-Gömb prototype: ONE HSiKAN trunk feeding both heads (design note Idea-1 config c).

    Passing a single backbone instance as both heads shares the structural trunk (PyTorch dedups the
    params), so the actor and critic reason over one signed hypergraph. Smoke: it builds, the trunk is
    genuinely shared (≈half the params of two backbones), and forward act/evaluate work."""
    torch.manual_seed(0)
    hg = _toy_hg()
    feat, action_dim = 4, 2
    trunk, fdim = hsikan_backbone(feat, hg_state=hg, hidden=16)
    shared = ActorCritic(trunk, trunk, fdim, action_dim)        # one Gömb -> both heads
    separate = build_policy("hsikan", obs_dim=feat, action_dim=action_dim, hg_state=hg, hidden=16)
    assert shared.actor_backbone is shared.critic_backbone      # genuinely one trunk
    assert shared.n_parameters() < separate.n_parameters()      # sharing removes a whole backbone
    obs = torch.randn(5, hg.n_vertices, feat)
    action, log_prob, value = shared.act(obs)
    assert action.shape == (5, action_dim) and value.shape == (5,)
    lp2, entropy, _ = shared.evaluate(obs, action)
    assert lp2.requires_grad and torch.isfinite(entropy).all()


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
    assert "mlp" in POLICY_KINDS and "hsikan" in POLICY_KINDS and "signedkan" in POLICY_KINDS


class _FakeHG:
    """Minimal hg_state for backbone tests — a 3-vertex signed adjacency, no MuJoCo needed."""

    n_vertices = 3

    def dense_signed_adj(self, device: object = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        return a, a * 0.3


def test_signedkan_learns_the_incidence_hsikan_keeps_it_fixed() -> None:
    """``signedkan`` makes the signed incidence trainable (the learned star edges); ``hsikan`` keeps it a
    fixed buffer (the kinematic structure). Both forward to the same shape — only the registration differs."""
    torch.manual_seed(0)
    hg = _FakeHG()
    sk = build_policy("signedkan", obs_dim=2, action_dim=1, hg_state=hg, hidden=8)
    hs = build_policy("hsikan", obs_dim=2, action_dim=1, hg_state=hg, hidden=8)
    # signedkan: incidence is a trainable parameter, initialised to the kinematic adjacency.
    assert isinstance(sk.actor_backbone.a_pos, torch.nn.Parameter)
    assert sk.actor_backbone.a_pos.requires_grad and sk.actor_backbone.a_neg.requires_grad
    assert torch.allclose(sk.actor_backbone.a_pos.detach(), hg.dense_signed_adj()[0])
    # hsikan: incidence is a buffer (not in parameters()) — the regression guard.
    assert not isinstance(hs.actor_backbone.a_pos, torch.nn.Parameter)
    hs_param_ids = {id(p) for p in hs.actor_backbone.parameters()}
    assert id(hs.actor_backbone.a_pos) not in hs_param_ids
    # signedkan has strictly more params (the incidence) and both forward identically-shaped.
    assert sk.n_parameters() > hs.n_parameters()
    obs = torch.randn(4, 3, 2)
    assert sk.act(obs)[0].shape == hs.act(obs)[0].shape == (4, 1)


def test_signedkan_incidence_receives_gradient() -> None:
    """A loss through the signedkan backbone produces a gradient on the learned incidence — so training
    actually updates the star edges."""
    torch.manual_seed(0)
    sk = build_policy("signedkan", obs_dim=2, action_dim=1, hg_state=_FakeHG(), hidden=8)
    loss = sk.value(torch.randn(4, 3, 2)).pow(2).mean()
    loss.backward()
    g = sk.critic_backbone.a_pos.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_weighted_incidence_real_arc_weights_init_parity() -> None:
    """``incidence="weighted"``: the structural mask is a fixed buffer, but each REAL arc gets a free
    real-valued weight parameter (init 1.0 → the effective adjacency equals the fixed binary structure,
    i.e. parity with plain HSiKAN at init). This lifts the binary {0,±1} incidence to free real weights —
    the point of a *signed* hypergraph — without ``signedkan``'s loss of the structural sparsity prior."""
    torch.manual_seed(0)
    hg = _FakeHG()
    w = build_policy("hsikan", obs_dim=2, action_dim=1, hg_state=hg, hidden=8, incidence="weighted")
    bb = w.actor_backbone
    # the structural mask is a buffer (not a parameter); the per-arc weights are parameters.
    assert not isinstance(bb.a_pos, torch.nn.Parameter)
    assert isinstance(bb.w_pos_arc, torch.nn.Parameter) and isinstance(bb.w_neg_arc, torch.nn.Parameter)
    # init 1.0 → effective adjacency == the fixed structural adjacency (parity with hsikan at init).
    eff_pos, eff_neg = bb._effective_adj()
    assert torch.allclose(eff_pos, hg.dense_signed_adj()[0])
    assert torch.allclose(eff_neg, hg.dense_signed_adj()[1])
    # strictly more parameters than fixed hsikan (the arc weights); same forward shape.
    hs = build_policy("hsikan", obs_dim=2, action_dim=1, hg_state=hg, hidden=8)
    assert w.n_parameters() > hs.n_parameters()
    assert w.act(torch.randn(4, 3, 2))[0].shape == (4, 1)


def test_weighted_incidence_arc_weights_stay_on_real_arcs() -> None:
    """A loss updates the per-arc weights, and the gradient is non-zero ONLY where the structural mask is
    non-zero — so the learned real weights live on the real hyperedge arcs (the sparsity prior holds)."""
    torch.manual_seed(0)
    hg = _FakeHG()
    w = build_policy("hsikan", obs_dim=2, action_dim=1, hg_state=hg, hidden=8, incidence="weighted")
    w.value(torch.randn(4, 3, 2)).pow(2).mean().backward()
    g = w.critic_backbone.w_pos_arc.grad
    mask = hg.dense_signed_adj()[0] != 0
    assert g is not None and torch.isfinite(g).all()
    assert g[mask].abs().sum() > 0                                  # real arcs receive gradient
    assert torch.allclose(g[~mask], torch.zeros_like(g[~mask]))     # non-arcs stay inert (masked off)


def test_invalid_incidence_rejected() -> None:
    with pytest.raises(ValueError, match="incidence must be one of"):
        build_policy("hsikan", obs_dim=2, action_dim=1, hg_state=_FakeHG(), hidden=8, incidence="bogus")


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
