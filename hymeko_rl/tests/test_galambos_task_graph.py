"""Galambos `task_graph` variant: coin+zone enter the graph as vertices (grasp/goal hyperedges).

Baseline (robot-only) must be byte-identical; the augmented variant adds coin/zone/hub rows and feeds HSiKAN.
Uses the hand-authored arms (``robot=None``) so the test needs no hymeko CLI."""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.policy import HSiKANBackbone


def test_baseline_unchanged() -> None:
    """robot-only graph: 6 vertices, (6,8) obs — the augmentation must not perturb the default path."""
    env = PlanarGraspEnv(robot=None, max_steps=10)
    obs, _ = env.reset(seed=1)
    assert env.hg.n_vertices == 6 and obs.shape == (6, 8)


def test_task_graph_adds_coin_zone_and_hubs() -> None:
    """task_graph: 6 robot + {coin,zone} + {grasp_hub,goal_hub} = 10 vertices; (10,8) obs."""
    env = PlanarGraspEnv(robot=None, max_steps=10, task_graph=True)
    obs, _ = env.reset(seed=1)
    assert env.hg.n_vertices == 10 and obs.shape == (10, 8)
    assert env.hg.vertex_labels[6:] == ("coin", "zone", "grasp_hub", "goal_hub")
    # coin vertex carries the coin's xy; zone vertex carries the zone's xy.
    assert np.allclose(obs[env._coin_vtx, 2:4], env._planar_metrics.disk_pos, atol=1e-5)
    assert np.allclose(obs[env._zone_vtx, 2:4], [env._zone_x, env._zone_y], atol=1e-5)


def test_robot_rows_identical_to_baseline() -> None:
    """The 6 robot rows are the SAME in both variants (same seed) — the augmentation only ADDS structure, so
    the HSiKAN(augmented) vs HSiKAN(baseline) comparison isolates the coin/zone hyperedges."""
    base = PlanarGraspEnv(robot=None, max_steps=10)
    aug = PlanarGraspEnv(robot=None, max_steps=10, task_graph=True)
    ob, _ = base.reset(seed=7)
    oa, _ = aug.reset(seed=7)
    assert np.allclose(ob, oa[:6], atol=1e-6)


def test_control_interface_is_actuated() -> None:
    """Regression: the action space must be non-degenerate and a command must move the arm. The hand-authored
    position actuators previously lacked ``ctrlrange`` → ``ctrllimited=False`` → action space ``[0,0,0,0]``, so
    the policy could only command zero (uncontrollable — the flat-curve failure looked like an RL problem)."""
    env = PlanarGraspEnv(robot=None, max_steps=160)
    lo, hi = env.action_space.low, env.action_space.high
    assert np.all(hi > lo), f"degenerate action space: low={lo} high={hi}"
    env.reset(seed=1)
    q0 = env.data.qpos.copy()
    for _ in range(40):
        env.step(hi.astype(np.float32))
    assert float(np.abs(env.data.qpos - q0).max()) > 0.5, "a full command barely moved the arm"


def test_task_graph_feeds_hsikan_and_steps() -> None:
    """The augmented graph is a drop-in for the HSiKAN backbone, and the env steps under the gym contract."""
    env = PlanarGraspEnv(robot=None, max_steps=5, task_graph=True)
    obs, _ = env.reset(seed=2)
    backbone = HSiKANBackbone(in_feat=8, hidden=16, n_layers=2, hg_state=env.hg)
    acts = backbone.node_activations(torch.as_tensor(obs[None], dtype=torch.float32))
    assert acts.shape == (1, 10, 16)
    obs2, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs2.shape == (10, 8) and np.isfinite(r)
