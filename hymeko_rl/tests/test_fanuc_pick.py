"""Tests for the FANUC LR Mate-config arm pick-and-place (the arm that grasps top-down)."""
from __future__ import annotations

import mujoco
import numpy as np

from hymeko_rl.viz.render_pick_place import fanuc_pick_env


def test_fanuc_env_builds_and_shapes() -> None:
    """The FANUC arm+gripper composite emits and gives the same 9-vertex hypergraph / (9,8) obs contract."""
    env = fanuc_pick_env(max_steps=10)
    obs, info = env.reset(seed=0)
    assert env.hg.n_vertices == 9
    assert env.action_space.shape == (7,)              # 6 arm joints + grip
    assert obs.shape == (9, 10) and np.isfinite(obs).all()
    assert "obj_to_target" in info


def test_fanuc_top_down_grasp_pose_is_collision_free() -> None:
    """The structural win: the FANUC arm reaches a top-down grasp pose at the object radius WITHOUT
    self-collision — the regression the anthropomorphic arm failed (it could only point the tool down by
    folding into ``link_0↔link_3`` self-collision). Spherical wrist + slim links."""
    env = fanuc_pick_env(max_steps=10)
    env.reset(seed=0)
    target = np.array([0.32, 0.0, env._surf + 0.18])   # hover above the table top, clear of the floor
    q = env._ik.solve_collision_free(env.data.qpos[:6], target, env._ik_pose_valid, down=True, n_starts=40)
    d = mujoco.MjData(env.model)
    d.qpos[:6] = q
    mujoco.mj_forward(env.model, d)
    ee = np.asarray(d.xpos[env._b_tool])
    down = -float(d.xmat[env._b_tool].reshape(3, 3)[2, 2])
    assert float(np.linalg.norm(ee - target)) < 3e-2
    assert down > 0.9                                  # tool points (nearly) straight down
    assert env._ik_pose_valid(d)                       # AND collision-free — the regression vs the old arm


def test_fanuc_expert_grasps_lifts_and_places() -> None:
    """The scripted expert does the full pick-and-place — grasps the box, lifts it clear of the table, and
    transports it into the target zone — on a known-good seed. Deterministic (fixed seed)."""
    env = fanuc_pick_env()
    env.reset(seed=0)
    grasped = placed = floor_hit = False
    lifted_max = 0.0
    for _ in range(env.max_steps):
        _o, _r, term, trunc, info = env.step(env.expert_action)
        grasped = grasped or bool(info["both_contact"])
        placed = placed or bool(info["reached"])
        lifted_max = max(lifted_max, float(info["lifted"]))
        # contact with the table while GRASPING is fine (the box rests on it); the floor is never touched.
        floor_hit = floor_hit or bool(env.data.xpos[env._b_tool][2] < env.table_z)
        if term or trunc:
            break
    assert grasped, "expert never achieved a two-finger grasp"
    assert lifted_max > 0.035, f"box not lifted clear of the table (max {lifted_max:.3f} m)"
    assert placed, "box not transported into the target zone"
    assert not floor_hit, "tool dropped below the ground plane"


def test_v2_expert_path_runs_and_is_deterministic() -> None:
    """The clearance-aware v2 waypoint expert produces valid, deterministic actions and advances its IK seed
    chain. It does NOT yet pass the clearance gate (retraction stall from the extended home pose; see
    reports/2026-07-07-pick-place-v2-waypoint-controller.md) — this guards the CONTROL PATH, not task success."""
    env = fanuc_pick_env(expert_version=2, max_steps=30)
    env.reset(seed=0)
    assert env._v2_seed is None                         # seed chain starts empty
    a0 = env.expert_action
    assert a0.shape == (env.n_actions,) and np.isfinite(a0).all()
    for _ in range(30):
        a = env.expert_action
        assert a.shape == (7,) and np.isfinite(a).all()
        env.step(a)
    assert env._v2_seed is not None and env._v2_seed.shape == (6,)   # seed chain advanced (un-folded commands)

    env2 = fanuc_pick_env(expert_version=2, max_steps=30)
    env2.reset(seed=0)
    assert np.allclose(env2.expert_action, a0)          # deterministic: same seed → same first action


def test_v2_preserves_v1_default_and_dispatch() -> None:
    """``expert_version`` defaults to 1 and dispatches correctly; v1's action is unchanged by the v2 addition
    (deterministic, same seed → same v1 action across instances)."""
    e1a = fanuc_pick_env(max_steps=10)
    e1b = fanuc_pick_env(expert_version=1, max_steps=10)
    e1a.reset(seed=0)
    e1b.reset(seed=0)
    assert e1a._expert_version == 1                     # default is the dirty v1 baseline
    assert np.allclose(e1a.expert_action, e1b.expert_action)
