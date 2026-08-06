"""Collaborative (cooperative CTDE) Galambos reframe: per-arm action partition + merge, the multi-agent env
view (shared obs, team reward), and the CTDE policy (shared backbone → per-arm heads + centralized critic)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from hymeko_rl.agents.collaborative import (
    CTDEActorCritic,
    CollaborativeGalambos,
    arm_action_partition,
    build_collaborative,
)
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv


def _fake_backbone(feat_dim: int = 16) -> tuple[nn.Module, int]:
    return nn.Sequential(nn.Flatten(), nn.Linear(6 * 4, feat_dim)), feat_dim   # obs (B, 6, 4)


def test_ctde_policy_shapes_and_grad() -> None:
    bb, fd = _fake_backbone()
    ac = CTDEActorCritic(bb, fd, (2, 2))
    obs = torch.randn(8, 6, 4)
    means = ac.agent_means(obs)
    assert len(means) == 2 and all(m.shape == (8, 2) for m in means)
    assert ac.action_mean(obs).shape == (8, 4) and ac.value(obs).shape == (8,)
    a, lp, v = ac.act(obs)
    assert a.shape == (8, 4) and lp.shape == (8,) and v.shape == (8,)
    lp2, ent, _v = ac.evaluate(obs, a)
    assert lp2.shape == (8,) and ent.shape == (8,)
    (ac.action_mean(obs).pow(2).mean() + ac.value(obs).pow(2).mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ac.backbone.parameters())
    assert all(h.weight.grad is not None and h.weight.grad.abs().sum() > 0 for h in ac.action_heads)
    assert ac.critic.weight.grad is not None and ac.n_parameters() > 0


def test_ctde_validation() -> None:
    bb, fd = _fake_backbone()
    with pytest.raises(ValueError):
        CTDEActorCritic(bb, 0, (2, 2))
    with pytest.raises(ValueError):
        CTDEActorCritic(bb, fd, ())
    with pytest.raises(ValueError):
        CTDEActorCritic(bb, fd, (2, 0))


def test_partition_and_merge() -> None:
    env = PlanarGraspEnv(robot=None, max_steps=50)
    covered: list[int] = []
    for s in arm_action_partition(env):
        covered += list(range(s.start, s.stop))
    assert covered == list(range(env.n_actions))           # tiles the action vector, contiguous, no overlap
    coll = CollaborativeGalambos(env)
    assert coll.n_agents == 2 and coll.agent_action_dims == (2, 2)
    merged = coll.merge([np.array([0.1, 0.2], np.float32), np.array([0.3, 0.4], np.float32)])
    assert np.allclose(merged, [0.1, 0.2, 0.3, 0.4])       # merge is the inverse of the partition
    with pytest.raises(ValueError):
        coll.merge([np.zeros(2, np.float32)])              # wrong agent count


def test_collab_env_rollout_shared_reward() -> None:
    coll = CollaborativeGalambos(PlanarGraspEnv(robot=None, max_steps=50))
    obs_list, _info = coll.reset(seed=0)
    assert len(obs_list) == coll.n_agents
    acts = [np.zeros(d, np.float32) for d in coll.agent_action_dims]
    obs_list, rew_list, _term, _trunc, info = coll.step(acts)
    assert len(obs_list) == 2 and len(rew_list) == 2
    assert rew_list[0] == rew_list[1]                      # cooperative: shared team reward
    assert "in_zone" in info


def test_build_collaborative_bc_compatible() -> None:
    env = PlanarGraspEnv(robot=None, max_steps=50)
    ac = build_collaborative(env, hidden=16)
    feat = int(env.observation_space.shape[1])             # type: ignore[index]
    obs = torch.randn(4, env.hg.n_vertices, feat)
    assert ac.action_mean(obs).shape == (4, env.n_actions)  # drop-in for BC/eval: full action vector
