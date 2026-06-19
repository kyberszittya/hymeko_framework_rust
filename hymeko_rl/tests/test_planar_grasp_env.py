"""Tests for the Galambos top-down planar grasping env (hymeko_rl/env/planar_grasp_env.py).

The arms are hand-authored MJCF (connected capsule links), so no CLI/emit is needed. The coin is a
planar table body placed in reach (not dropped); the policy reads the 6-vertex two-arm hypergraph.
"""
from __future__ import annotations

import mujoco
import numpy as np

from hymeko_rl.env.planar_grasp_env import (
    PlanarGraspEnv, compose_planar_scene, make_planar_arms_mjcf,
)
from hymeko_rl.env.reward import _REWARD_TERMS, RewardSpec

_TASK = "data/robotics/galambos_task.hymeko"


def test_planar_reward_terms_registered() -> None:
    assert "both_contact" in _REWARD_TERMS
    assert "in_zone" in _REWARD_TERMS


def test_galambos_task_parses() -> None:
    spec = RewardSpec.from_hymeko(_TASK)
    assert [k for k, _ in spec.terms] == ["reach_distance", "both_contact", "in_zone", "action_cost"]
    assert dict(spec.terms)["in_zone"] == 10.0


def test_scene_is_connected_planar_table() -> None:
    m = mujoco.MjModel.from_xml_string(compose_planar_scene(make_planar_arms_mjcf()))
    assert m.nu == 4                                  # arm actuators only (coin unactuated)
    assert m.nq == 7                                  # 4 arm + 3 planar coin DOF
    # connected 2-link arms: base/link1/link2 per side.
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    for nm in ("link1_left", "link2_left", "link1_right", "link2_right"):
        assert nm in bodies
    assert sum(int(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE) for g in range(m.ngeom)) == 1
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_zone") >= 0
    # the coin is confined to the plane: z never changes.
    d = mujoco.MjData(m)
    disk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "disk")
    mujoco.mj_forward(m, d)
    z0 = float(d.xpos[disk][2])
    for _ in range(150):
        mujoco.mj_step(m, d)
    assert abs(float(d.xpos[disk][2]) - z0) < 1e-4


def test_env_shapes_and_coin_placed_in_reach() -> None:
    env = PlanarGraspEnv(max_steps=50)
    assert env.n_actions == 4 and env.hg.n_vertices == 6
    disk = env._disk_body
    for seed in range(8):
        obs, info = env.reset(seed=seed)
        assert obs.shape == (6, 8)
        assert info["disk_to_zone"] >= env._zone_half          # outside the zone
        x, y = float(env.data.xpos[disk][0]), float(env.data.xpos[disk][1])
        assert abs(x) <= 0.12 and 0.07 <= y <= 0.23            # within the (reachable) spawn box


def test_coin_at_fingertip_registers_contact() -> None:
    """Reach + contact detection align: a coin placed at the left distal link touches the left arm."""
    env = PlanarGraspEnv()
    env.reset(seed=0)
    mujoco.mj_forward(env.model, env.data)
    lower_left = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "lower_left")
    g = next(gg for gg in range(env.model.ngeom)
             if int(env.model.geom_bodyid[gg]) == lower_left)
    pos = env.data.geom_xpos[g]            # the distal link's geom centroid (fingertip region)
    # place the coin overlapping the link off-centre (concentric placement is a degenerate contact).
    env.data.qpos[env._disk_x_adr] = float(pos[0]) + 0.025
    env.data.qpos[env._disk_y_adr] = float(pos[1])
    left = False
    for _ in range(5):                     # step a few frames so the contact registers
        mujoco.mj_step(env.model, env.data)
        m = env._metrics()
        left = left or (m.left_contact and not m.right_contact)
    assert left, "a coin overlapping the left distal link should register a left-only contact"


def test_coin_planted_in_zone_terminates() -> None:
    env = PlanarGraspEnv(max_steps=120, success_steps=3)
    env.reset(seed=0)
    env.data.qpos[env._disk_x_adr] = env._zone_x
    env.data.qpos[env._disk_y_adr] = env._zone_y
    mujoco.mj_forward(env.model, env.data)
    zero = np.zeros(env.n_actions, dtype=np.float32)
    terminated = False
    for _ in range(120):
        _o, _r, terminated, truncated, info = env.step(zero)
        if terminated:
            assert info["in_zone"]
            break
        if truncated:
            break
    assert terminated, "coin at the zone centre should be in-zone and terminate"
