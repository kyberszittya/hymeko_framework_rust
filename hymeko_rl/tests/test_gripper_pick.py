"""Tests for the floating-gripper pick-and-place de-risk testbed + its BC loop (Phase 1, path B)."""
from __future__ import annotations

import numpy as np

from hymeko_rl.env.gripper_pick_env import GripperPickEnv
from hymeko_rl.experiments.gripper_pick_bc import run_gripper_bc


def test_env_builds_and_shapes() -> None:
    env = GripperPickEnv(max_steps=20)
    assert env.hg.n_vertices == 5                       # carriage_x/y, palm, finger_l/r
    obs, info = env.reset(seed=0)
    assert obs.shape == (5, 8) and np.isfinite(obs).all()
    assert env.action_space.shape == (4,)
    o, r, term, trunc, step_info = env.step(env.action_space.sample())
    assert o.shape == (5, 8) and np.isfinite(r)
    assert {"lifted", "both_contact", "reached", "death"} <= set(step_info)


def test_expert_grasps_and_lifts() -> None:
    """The direct-Cartesian expert reliably grasps and lifts the box (control works — no IK)."""
    env = GripperPickEnv(max_steps=250)
    lifts = 0
    for seed in range(4):
        env.reset(seed=seed)
        lifted = False
        for _ in range(250):
            _o, _r, term, _tr, info = env.step(env.expert_action)
            lifted = lifted or bool(info["lifted"] > 0.04)
            if term:
                break
        lifts += int(lifted)
    assert lifts >= 3, f"expert lifted on only {lifts}/4 seeds"


def test_bc_learns_to_grasp_beating_untrained_floor() -> None:
    """The Phase-1 acceptance: a behaviour-cloned policy grasps far above the untrained floor (~0)."""
    res = run_gripper_bc(kind="hsikan", n_demos=24, n_epochs=80, n_eval=8, seed=0)
    assert res["untrained_lift"] == 0.0
    assert float(res["lift_rate"]) >= 0.25, f"BC lift_rate too low: {res}"
