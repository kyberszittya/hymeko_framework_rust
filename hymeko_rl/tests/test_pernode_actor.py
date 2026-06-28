"""Tests for the per-node multidimensional actor head (the RL payoff test for the multidim-readout finding).

Pins the actuator→vertex map, the per-node head shape/finiteness, the per-node SAC actor's duck-typing of
the pooled actor's interface, and the build_sac selection.
"""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.env.arm_world import actuator_vertices
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.policy import PerNodeActionHead
from hymeko_rl.sac import PerNodeSquashedGaussianActor, SquashedGaussianActor, build_sac


def _galambos():  # type: ignore[no-untyped-def]
    env = PlanarGraspEnv.from_hymeko(max_steps=10, difficulty=0.3)
    obs, _ = env.reset(seed=0)
    nv, feat = obs.shape
    return env, obs, nv, feat


def test_actuator_vertices_maps_each_actuator_to_its_link() -> None:
    env, *_ = _galambos()
    av = actuator_vertices(env.model)
    assert av.shape == (env.model.nu,)
    assert all(0 <= int(v) < env.hg.n_vertices for v in av)
    # galambos: 4 actuators drive the upper/lower links of each arm (bases are unactuated).
    assert [env.hg.vertex_labels[int(v)] for v in av] == \
        ["upper_left", "lower_left", "upper_right", "lower_right"]


def test_per_node_head_shape_and_responds() -> None:
    head = PerNodeActionHead(8, act_vertices=[1, 2, 4])
    assert head.action_dim == 3
    h = torch.randn(5, 6, 8)
    out = head(h)
    assert out.shape == (5, 3) and torch.isfinite(out).all()
    # changing a referenced vertex's activation changes its output row (gather + pool both feed it).
    h2 = h.clone()
    h2[:, 2, :] += 3.0
    assert not torch.allclose(head(h)[:, 1], head(h2)[:, 1])


def test_per_node_actor_ducktypes_pooled_actor() -> None:
    env, obs, nv, feat = _galambos()
    av = actuator_vertices(env.model)
    ad = int(np.prod(env.action_space.shape))
    scale = float(np.max(np.abs(env.action_space.high)))
    actor, critics = build_sac("hsikan", obs_dim=feat, flat_dim=nv * feat, action_dim=ad,
                               action_scale=scale, hidden=32, hg_state=env.hg,
                               actor_head="per_node", act_vertices=av)
    assert isinstance(actor, PerNodeSquashedGaussianActor)
    assert actor.action_dim == ad and actor.action_scale == scale
    x = torch.as_tensor(obs[None], dtype=torch.float32)
    a, logp = actor.sample(x)
    mean = actor.action_mean(x)
    assert a.shape == (1, ad) and mean.shape == (1, ad) and logp.shape == (1,)
    assert torch.isfinite(a).all() and bool((a.abs() <= scale + 1e-4).all())


def test_per_node_with_highway_builds_and_runs() -> None:
    """The per-node actor composes with the HSiKAN highway alpha-gate (skip='highway'): it builds, has more
    params than skip='none' (the gate), and forwards a finite action."""
    env, obs, nv, feat = _galambos()
    av = actuator_vertices(env.model)
    ad = int(np.prod(env.action_space.shape))
    none_actor, _ = build_sac("hsikan", obs_dim=feat, flat_dim=nv * feat, action_dim=ad, action_scale=1.0,
                              hidden=32, hg_state=env.hg, actor_head="per_node", act_vertices=av, skip="none")
    hw_actor, _ = build_sac("hsikan", obs_dim=feat, flat_dim=nv * feat, action_dim=ad, action_scale=1.0,
                            hidden=32, hg_state=env.hg, actor_head="per_node", act_vertices=av, skip="highway")
    n_none = sum(p.numel() for p in none_actor.parameters())
    n_hw = sum(p.numel() for p in hw_actor.parameters())
    assert n_hw > n_none, "the highway gate adds parameters"
    x = torch.as_tensor(obs[None], dtype=torch.float32)
    assert torch.isfinite(hw_actor.action_mean(x)).all() and hw_actor.action_mean(x).shape == (1, ad)


def test_build_sac_pooled_default_and_per_node_guards() -> None:
    env, _obs, nv, feat = _galambos()
    av = actuator_vertices(env.model)
    ad = int(np.prod(env.action_space.shape))
    pooled, _ = build_sac("hsikan", obs_dim=feat, flat_dim=nv * feat, action_dim=ad, action_scale=1.0,
                          hidden=32, hg_state=env.hg)
    assert isinstance(pooled, SquashedGaussianActor)        # default is the pooled actor (back-compat)
    import pytest
    with pytest.raises(ValueError):                          # per_node needs a per-vertex backbone
        build_sac("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=ad, action_scale=1.0, hidden=32,
                  actor_head="per_node", act_vertices=av)
    with pytest.raises(ValueError):                          # per_node needs act_vertices
        build_sac("hsikan", obs_dim=feat, flat_dim=nv * feat, action_dim=ad, action_scale=1.0, hidden=32,
                  hg_state=env.hg, actor_head="per_node")
