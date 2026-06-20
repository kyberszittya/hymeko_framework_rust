"""Tests for the Galambos top-down planar grasping env (hymeko_rl/env/planar_grasp_env.py).

The arms are hand-authored MJCF (connected capsule links), so no CLI/emit is needed. The coin is a
planar table body placed in reach (not dropped); the policy reads the 6-vertex two-arm hypergraph.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from hymeko_rl.env.env_spec import EnvSpec
from hymeko_rl.env.planar_grasp_env import (
    PlanarGraspEnv, compose_planar_scene, make_planar_arms_mjcf,
)
from hymeko_rl.env.reward import _REWARD_TERMS, RewardSpec

_TASK = "data/robotics/galambos_task.hymeko"


def test_planar_reward_terms_registered() -> None:
    for k in ("both_contact", "in_zone", "grasp_approach", "settle", "arm_motion", "center_bonus",
              "arm_collision", "out_of_bounds"):
        assert k in _REWARD_TERMS


def test_galambos_task_parses() -> None:
    spec = RewardSpec.from_hymeko(_TASK)
    # action_cost dropped (rewarded stationarity); center_bonus (precision), arm_motion (anti-stall),
    # arm_collision (no arm-arm crash) and out_of_bounds (no knock-out) added.
    assert [k for k, _ in spec.terms] == [
        "grasp_approach", "reach_distance", "both_contact", "in_zone", "center_bonus", "arm_motion",
        "arm_collision", "out_of_bounds"]
    assert dict(spec.terms)["in_zone"] == 10.0
    assert dict(spec.terms)["center_bonus"] == 5.0
    assert dict(spec.terms)["arm_collision"] == 1.0
    assert dict(spec.terms)["out_of_bounds"] == 5.0
    assert "action_cost" not in dict(spec.terms)


def _metrics(*, tipl: float = 0.0, tipr: float = 0.0, speed: float = 0.0, arm_speed: float = 0.0,
             self_contact: bool = False, dz: float = 0.1):  # type: ignore[no-untyped-def]
    from hymeko_rl.env.planar_grasp_env import PlanarGraspMetrics
    return PlanarGraspMetrics(
        np.zeros(2, np.float32), dz, False, False, False, tipl, tipr, speed, arm_speed, self_contact)


class _Stub:
    def __init__(self, m, zone_half: float = 0.055) -> None:  # type: ignore[no-untyped-def]
        self._planar_metrics = m
        self._zone_half = zone_half


def test_grasp_approach_term_rewards_closing_on_coin() -> None:
    """The dense approach term is more negative when the arms are far from the coin than near,
    and inert (0) on an env without planar metrics — the directed regression that would fail
    against the pre-shaping reward (which had no such gradient)."""
    from hymeko_rl.env.reward import _term_grasp_approach

    z = np.zeros(4, np.float32)
    far_r = _term_grasp_approach(_Stub(_metrics(tipl=0.20, tipr=0.18)), 0.0, z)
    near_r = _term_grasp_approach(_Stub(_metrics(tipl=0.02, tipr=0.03)), 0.0, z)
    assert far_r < near_r < 0.0, "closing on the coin must raise the approach reward toward 0"
    assert _term_grasp_approach(object(), 0.0, z) == 0.0   # inert on a non-planar env


def test_settle_term_brakes_only_inside_zone() -> None:
    """The overshoot brake penalises coin speed only once the coin is inside the zone
    (``dist < zone_half``), so the coin moves freely during approach (braking there was measured to
    cause undershoot), scales with speed, and is inert on a non-planar env."""
    from hymeko_rl.env.reward import _term_settle

    z = np.zeros(4, np.float32)  # zone_half default 0.055
    fast_in = _term_settle(_Stub(_metrics(speed=0.8)), dist=0.04, action=z)    # inside the zone
    fast_out = _term_settle(_Stub(_metrics(speed=0.8)), dist=0.09, action=z)   # approaching, not in
    slow_in = _term_settle(_Stub(_metrics(speed=0.1)), dist=0.04, action=z)
    assert fast_in < slow_in < 0.0, "inside the zone, faster coin = larger brake penalty"
    assert fast_out == 0.0, "during approach (outside the zone) the coin may move freely"
    assert _term_settle(object(), 0.0, z) == 0.0          # inert on a non-planar env


def test_arm_motion_penalizes_an_idle_arm() -> None:
    """Anti-stall: a frozen arm is penalised, the penalty vanishes once the arm moves at v_min, and
    it is inert on a non-planar env."""
    from hymeko_rl.env.reward import _ARM_STALL_VMIN, _term_arm_motion

    z = np.zeros(4, np.float32)
    frozen = _term_arm_motion(_Stub(_metrics(arm_speed=0.0)), 0.0, z)
    slow = _term_arm_motion(_Stub(_metrics(arm_speed=0.5)), 0.0, z)
    moving = _term_arm_motion(_Stub(_metrics(arm_speed=_ARM_STALL_VMIN + 0.2)), 0.0, z)
    assert frozen < slow < 0.0, "a more frozen arm is penalised more"
    assert moving == 0.0, "an arm moving at >= v_min is not penalised"
    assert _term_arm_motion(object(), 0.0, z) == 0.0


def test_center_bonus_grades_toward_the_zone_centre() -> None:
    """The centring bonus rises from 0 at the zone edge to 1 at the exact centre, and is 0 outside
    the zone / on a non-planar env."""
    from hymeko_rl.env.reward import _term_center_bonus

    z = np.zeros(4, np.float32)
    stub = _Stub(_metrics(), zone_half=0.04)
    centre = _term_center_bonus(stub, dist=0.0, action=z)
    halfway = _term_center_bonus(stub, dist=0.02, action=z)
    edge = _term_center_bonus(stub, dist=0.04, action=z)
    outside = _term_center_bonus(stub, dist=0.08, action=z)
    assert centre == 1.0 and abs(halfway - 0.5) < 1e-9 and edge == 0.0 and outside == 0.0
    assert _term_center_bonus(object(), dist=0.0, action=z) == 0.0


def test_arm_collision_penalises_arm_arm_contact() -> None:
    """The arm-arm collision term is -1 while the two arms touch each other, 0 otherwise, and inert
    on a non-planar env."""
    from hymeko_rl.env.reward import _term_arm_collision

    z = np.zeros(4, np.float32)
    assert _term_arm_collision(_Stub(_metrics(self_contact=True)), 0.0, z) == -1.0
    assert _term_arm_collision(_Stub(_metrics(self_contact=False)), 0.0, z) == 0.0
    assert _term_arm_collision(object(), 0.0, z) == 0.0


def test_out_of_bounds_penalises_knocking_the_disk_off() -> None:
    """The out-of-bounds term is -1 on the step the disk leaves the table (env `_disk_out`), 0
    otherwise / on a non-planar env."""
    from hymeko_rl.env.reward import _term_out_of_bounds

    class _S:
        def __init__(self, out: bool) -> None:
            self._disk_out = out

    z = np.zeros(4, np.float32)
    assert _term_out_of_bounds(_S(True), 0.0, z) == -1.0
    assert _term_out_of_bounds(_S(False), 0.0, z) == 0.0
    assert _term_out_of_bounds(object(), 0.0, z) == 0.0


def test_planar_metrics_expose_tip_distances() -> None:
    env = PlanarGraspEnv(max_steps=10)
    env.reset(seed=0)
    m = env._metrics()
    assert m.left_tip_dist >= 0.0 and m.right_tip_dist >= 0.0
    # the reward now responds to the arms' proximity to the coin.
    assert "grasp_approach" in [k for k, _ in env.reward_spec.terms]


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
    saw_outside_band = False
    for seed in range(16):
        obs, info = env.reset(seed=seed)
        assert obs.shape == (6, 8)
        assert info["disk_to_zone"] >= env._zone_half          # outside the zone
        x, y = float(env.data.xpos[disk][0]), float(env.data.xpos[disk][1])
        assert env._reachable_by_any(x, y)                     # an arm can fetch it
        saw_outside_band = saw_outside_band or abs(x) > 0.11   # can spawn outside the arm band
    assert saw_outside_band, "the harder task must sometimes spawn the coin outside the arm band"


def test_zone_randomizes_within_both_arm_reach_and_is_observed() -> None:
    env = PlanarGraspEnv(max_steps=20)
    zones = set()
    for seed in range(12):
        env.reset(seed=seed)
        zx, zy = env._zone_x, env._zone_y
        zones.add((round(zx, 3), round(zy, 3)))
        # within reach of BOTH arms (deliverable target).
        for cx, cy in env._reach_centers:
            assert np.hypot(zx - cx, zy - cy) <= 0.28 + 1e-9
        # the zone is observed: the coin->zone channel matches the live coin/zone gap.
        feat = env.node_features()
        cx = float(env._planar_metrics.disk_pos[0])
        assert abs(feat[0, 6] - (cx - zx)) < 1e-5
    assert len(zones) >= 5, "the zone must actually vary across episodes"


def test_fixed_zone_mode_keeps_zone_put() -> None:
    env = PlanarGraspEnv(max_steps=20, env=EnvSpec(randomize_zone=False, zone_x=0.0, zone_y=0.16))
    for seed in range(4):
        env.reset(seed=seed)
        assert (env._zone_x, env._zone_y) == (0.0, 0.16)


def test_env_spec_parses_galambos_env() -> None:
    s = EnvSpec.from_hymeko("data/robotics/galambos_env.hymeko")
    assert s.zone_half == 0.04 and s.success_steps == 5 and s.randomize_zone is True
    assert s.zone_region == (-0.05, 0.05, 0.10, 0.18)
    assert s.coin_region == (-0.20, 0.20, 0.05, 0.23)
    assert s.coin_clearance == 0.03
    assert (s.out_bound, s.y_min, s.y_max) == (0.40, -0.08, 0.45)
    assert s.disk_radius == 0.02   # a small disk, not a coin


def test_smaller_disk_geom_uses_spec_radius() -> None:
    env = PlanarGraspEnv.from_hymeko(max_steps=10)
    g = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    assert abs(float(env.model.geom_size[g][0]) - 0.02) < 1e-9   # declarative disk radius


def test_env_spec_missing_term_raises(tmp_path: Path) -> None:
    # an env_spec that bundles a member which is not declared → ValueError (the reader's contract).
    bad = tmp_path / "bad_env.hymeko"
    bad.write_text(
        "bad_description {\n  @\"../../data/robotics/meta_env.hymeko\";\n  using env as e;\n}\n"
        "bad: e {\n  @env_spec: e.env_spec { (+ nonesuch); }\n}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        EnvSpec.from_hymeko(bad)


def test_planar_env_from_hymeko_matches_spec() -> None:
    """The three-path from_hymeko builds an env whose geometry equals the .hymeko declaration —
    the MDP (robot + environment + reward) all from source."""
    env = PlanarGraspEnv.from_hymeko(max_steps=40)
    assert env._zone_half == 0.04 and env.success_steps == 5 and env._randomize_zone is True
    assert env._zone_region == (-0.05, 0.05, 0.10, 0.18)
    assert env._coin_region == (-0.20, 0.20, 0.05, 0.23)
    assert (env._out_bound, env._y_min, env._y_max) == (0.40, -0.08, 0.45)
    assert [k for k, _ in env.reward_spec.terms][0] == "grasp_approach"   # reward also from .hymeko
    # and it actually runs: a reset yields a zone in the declared region and a reachable coin.
    obs, info = env.reset(seed=0)
    assert obs.shape == (6, 8) and info["disk_to_zone"] >= env._zone_half


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


def test_shoulder_joint_is_not_frozen_by_self_contact() -> None:
    """Regression for the emitter base/first-link overlap that pinned the shoulder: holding a
    constant inward shoulder target must actually rotate the shoulder joints (j1), not leave them
    frozen near 0 (measured tracking error 1.19 rad before the adjacent-link contact excludes)."""
    env = PlanarGraspEnv(max_steps=400)
    env.reset(seed=0)
    target = np.array([1.2, -1.6, -1.2, 1.6], dtype=np.float32)[: env.n_actions]
    for _ in range(300):
        env.step(target)
    q = env.data.qpos[: env.n_actions]
    # shoulder joints are indices 0 and 2 (j1 of each arm); they must track toward the |1.2| target.
    assert abs(float(q[0])) > 0.5 and abs(float(q[2])) > 0.5, f"shoulder joints frozen: qpos={q}"


def test_difficulty_pulls_spawn_toward_zone() -> None:
    """Lower curriculum difficulty caps the coin nearer the zone; at difficulty 1 it spans the full
    reachable table. The coin is always reachable and outside the zone."""
    def mean_dist(diff: float) -> float:
        env = PlanarGraspEnv(max_steps=10, difficulty=diff)
        ds = []
        for s in range(60):
            _o, info = env.reset(seed=s)
            ds.append(info["disk_to_zone"])
        return float(np.mean(ds))

    near, full = mean_dist(0.0), mean_dist(1.0)
    assert near < full, f"difficulty 0 should spawn nearer the zone ({near:.3f} vs {full:.3f})"
    env = PlanarGraspEnv(max_steps=10, difficulty=1.0)
    for s in range(8):
        _o, info = env.reset(seed=s)
        x, y = float(env.data.xpos[env._disk_body][0]), float(env.data.xpos[env._disk_body][1])
        assert env._reachable_by_any(x, y) and info["disk_to_zone"] >= env._zone_half
        assert info["disk_to_zone"] >= env._zone_half


def test_difficulty_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        PlanarGraspEnv(difficulty=1.5)


def test_coin_planted_in_zone_terminates() -> None:
    env = PlanarGraspEnv(max_steps=120, env=EnvSpec(success_steps=3))
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
