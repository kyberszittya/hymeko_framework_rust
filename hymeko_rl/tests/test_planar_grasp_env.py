"""Tests for the Galambos top-down planar grasping env (hymeko_rl/env/planar_grasp_env.py).

The arms are hand-authored MJCF (connected capsule links), so no CLI/emit is needed. The coin is a
planar table body placed in reach (not dropped); the policy reads the 6-vertex two-arm hypergraph.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from hymeko_rl.env.constants import Collision
from hymeko_rl.env.env_spec import EnvSpec
from hymeko_rl.env.planar_grasp_env import (
    PlanarGraspEnv, compose_planar_scene, make_planar_arms_mjcf,
)
from hymeko_rl.env.reward import _REWARD_TERMS, RewardSpec


# ── collision for BOTH arm geometries: fingertip sphere collides with the coin, arm-link capsule does NOT ──
# (the sim shows the coin passing through the arm bodies — this pins that behaviour at the compiled-model +
# physics level, one geometry per assertion, above the bitmask-predicate test in test_env_constants.py.)

def _geom_channel(model: object, g: int) -> tuple[int, int]:
    return (int(model.geom_contype[g]), int(model.geom_conaffinity[g]))   # type: ignore[attr-defined]


def _capsule_geoms(model: object) -> list[int]:
    return [g for g in range(int(model.ngeom))                            # type: ignore[attr-defined]
            if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)]   # type: ignore[attr-defined]


def test_compiled_model_coin_collides_with_fingertip_not_arm_capsule() -> None:
    """On the actual compiled model: the coin geom's channel collides with the fingertip geom's channel and
    with NONE of the arm-link capsule channels (the fingertip-only manipulation invariant, at model level)."""
    env = PlanarGraspEnv(robot=None, max_steps=10, difficulty=0.3)
    env.reset(seed=0)
    model = env.model
    disk = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "disk"))
    ft = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left"))
    assert disk >= 0 and ft >= 0
    coin_ch = _geom_channel(model, disk)
    assert Collision.collide(coin_ch, _geom_channel(model, ft))           # FINGERTIP geometry: collides
    caps = _capsule_geoms(model)
    assert caps, "expected arm-link capsule geoms in the compiled model"
    for g in caps:                                                        # ARM geometry: never collides
        assert not Collision.collide(coin_ch, _geom_channel(model, g)), \
            f"arm capsule geom {g} would collide with the coin (mask leak)"
    env.close()


def test_coin_passes_through_arm_capsule_but_contacts_fingertip() -> None:
    """Dynamic (mj_forward at overlapping poses): the coin placed ON a fingertip generates a coin-fingertip
    contact, but the coin placed ON an arm capsule generates NO coin-arm contact — it passes through, exactly
    what the simulation showed. Tests the collision for both geometries via real MuJoCo contact generation."""
    env = PlanarGraspEnv(robot=None, max_steps=10, difficulty=0.3)
    env.reset(seed=0)
    model, data = env.model, env.data
    disk = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "disk"))
    ft = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left"))
    caps = set(_capsule_geoms(model))
    data.qpos[:] = 0.0                                                    # home the arms
    mujoco.mj_forward(model, data)
    ft_xy = data.geom_xpos[ft][:2].copy()
    cap_geom = min(caps)
    cap_xy = data.geom_xpos[cap_geom][:2].copy()

    def _place_coin(xy: np.ndarray) -> None:
        data.qpos[:] = 0.0
        data.qpos[env._disk_x_adr] = float(xy[0])
        data.qpos[env._disk_y_adr] = float(xy[1])
        mujoco.mj_forward(model, data)

    def _pairs(pred: object) -> int:
        return sum(1 for i in range(int(data.ncon))
                   if pred(int(data.contact[i].geom1), int(data.contact[i].geom2)))   # type: ignore[operator]

    _place_coin(ft_xy)                                                    # coin ON the fingertip
    assert _pairs(lambda a, b: {a, b} == {disk, ft}) >= 1, "coin should contact the fingertip"

    _place_coin(cap_xy)                                                   # coin ON an arm capsule
    assert _pairs(lambda a, b: disk in (a, b) and (a in caps or b in caps)) == 0, \
        "coin must pass THROUGH the arm capsule (no coin-arm contact)"
    env.close()


_TASK = "data/robotics/galambos_task.hymeko"


def test_planar_reward_terms_registered() -> None:
    for k in ("both_contact", "in_zone", "grasp_approach", "settle", "arm_motion", "center_bonus",
              "arm_collision", "out_of_bounds"):
        assert k in _REWARD_TERMS


def test_only_fingertip_can_touch_the_coin() -> None:
    """Galambos 2026-07-03: ONLY the yellow fingertip may contact the cylinder — the arm links must NOT be able
    to. Enforced by collision bitmasks: coin on bit 2 (arm links are MuJoCo-default 1/1 → cannot touch it), a
    fingertip geom on conaffinity 3 (→ touches the coin), floor on conaffinity 3 (→ the coin still rests). Two
    geoms collide iff (contype_a & conaffinity_b) | (contype_b & conaffinity_a)."""
    env = PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3)
    m = env.model

    def mask(name: str) -> tuple[int, int]:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, f"geom {name} missing"
        return int(m.geom_contype[gid]), int(m.geom_conaffinity[gid])

    def collide(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return bool((a[0] & b[1]) or (b[0] & a[1]))

    disk, tip, floor = mask("disk"), mask("fingertip_left"), mask("floor")
    arm = (1, 1)                                                    # arm-link capsules use the MuJoCo default
    assert disk == (2, 2) and tip == (1, 3)
    assert not collide(arm, disk), "arm links must NOT be able to touch the coin (Galambos: a kar ne ütközzön)"
    assert collide(tip, disk), "the yellow fingertip must contact the coin"
    assert collide(disk, floor), "the coin must still rest on the table"
    assert collide(mask("fingertip_left"), mask("fingertip_right")), "fingers still collide (fingers_collision)"


def test_coin_frictionloss_is_opt_in_and_sets_joint_threshold() -> None:
    """The two-arm-force lever (Galambos): dry friction on the coin's slide joints — a FORCE threshold below
    which the coin will not move. 0 (default) = the original free-sliding coin; > 0 raises a jointer frictionloss."""
    free = PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3, coin_frictionloss=0.0)
    heavy = PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3, coin_frictionloss=8.0)

    def joint_frictionloss(env: PlanarGraspEnv) -> float:
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "disk_x")
        return float(env.model.dof_frictionloss[env.model.jnt_dofadr[jid]])

    assert joint_frictionloss(free) == 0.0
    assert joint_frictionloss(heavy) == pytest.approx(8.0)         # the declared force threshold reaches the joint


def test_galambos_task_parses() -> None:
    spec = RewardSpec.from_hymeko(_TASK)
    # 4-core (approach·both·zone·oob, grasp-fraction 0.615) + the COLLAB extension 2026-07-03: finger_contact
    # (denser per-fingertip reward) and arm_body_collision (upper-arm-only collision, collision-forward at 2.0).
    assert [k for k, _ in spec.terms] == [
        "grasp_approach", "both_contact", "finger_contact", "in_zone", "out_of_bounds", "arm_body_collision"]
    assert dict(spec.terms)["both_contact"] == 5.0
    assert dict(spec.terms)["in_zone"] == 10.0
    assert dict(spec.terms)["out_of_bounds"] == 2.0
    assert dict(spec.terms)["finger_contact"] == 1.5      # denser per-fingertip contact reward
    assert dict(spec.terms)["arm_body_collision"] == 0.5  # LIGHT upper-arm collision (A/B: 2.0 suppressed delivery)
    assert "action_cost" not in dict(spec.terms)
    assert "arm_collision" not in dict(spec.terms)        # the grasp-killing WHOLE-arm penalty stays dropped


def _metrics(*, tipl: float = 0.0, tipr: float = 0.0, speed: float = 0.0, arm_speed: float = 0.0,
             self_contact: bool = False, dz: float = 0.1):  # type: ignore[no-untyped-def]
    from hymeko_rl.env.planar_grasp_env import PlanarGraspMetrics
    return PlanarGraspMetrics(
        np.zeros(2, np.float32), dz, False, False, False, tipl, tipr, speed, arm_speed, self_contact,
        np.zeros(2, np.float32))


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


def test_fingertip_sites_injected_and_idempotent() -> None:
    """The emitted planar arm ships NO fingertip site; ``with_fingertip_sites`` adds ``tip_left`` /
    ``tip_right``, is idempotent, and leaves the hand-authored scene (which declares its own) intact."""
    from hymeko_rl.env.arm_world import emit_arm_mjcf
    from hymeko_rl.env.planar_grasp_env import with_fingertip_sites

    emitted = emit_arm_mjcf("data/robotics/galambos_planar.hymeko", name="galambos",
                            control_mode="position")
    assert "tip_left" not in emitted and "tip_right" not in emitted   # ships no fingertip site
    fixed = with_fingertip_sites(emitted)
    m = mujoco.MjModel.from_xml_string(fixed)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip_left") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip_right") >= 0
    assert with_fingertip_sites(fixed).count('name="tip_left"') == 1   # idempotent
    assert with_fingertip_sites(make_planar_arms_mjcf()).count('name="tip_left"') == 1  # not duplicated


def test_fingertip_injection_preserves_dynamics() -> None:
    """A site is massless and collisionless: injecting the tips changes only ``nsite`` — bodies,
    DOFs, and actuators (the compiled dynamics) are bit-identical, so training physics is unchanged."""
    from hymeko_rl.env.arm_world import emit_arm_mjcf
    from hymeko_rl.env.planar_grasp_env import with_fingertip_sites

    raw = emit_arm_mjcf("data/robotics/galambos_planar.hymeko", name="galambos", control_mode="position")
    m0 = mujoco.MjModel.from_xml_string(raw)
    m1 = mujoco.MjModel.from_xml_string(with_fingertip_sites(raw))
    assert (m1.nq, m1.nv, m1.nu, m1.nbody) == (m0.nq, m0.nv, m0.nu, m0.nbody)
    assert m1.nsite == m0.nsite + 2


def test_approach_distance_is_tip_dominant_blend_not_body_min() -> None:
    """Regression: each arm's approach distance is ``0.75·fingertip + 0.25·elbow`` (the true tool
    point), NOT the min over body origins. On seed 2 the fingertip is far while the elbow is near the
    coin, so the blend strictly exceeds the per-arm body-min — this assertion fails against the
    pre-2026-06-26 ``_nearest`` metric (which returned the body-min and ignored the tip)."""
    env = PlanarGraspEnv.from_hymeko(max_steps=10, difficulty=0.3)
    env.reset(seed=2)
    m = env._metrics()
    disk = np.asarray(env.data.xpos[env._disk_body][:2], np.float64)

    def pdist(p: np.ndarray) -> float:
        return float(np.hypot(p[0] - disk[0], p[1] - disk[1]))

    for tip_site, elbow_body, bodies, got in (
            (env._tip_sites[0], env._elbow_bodies[0], env._left_bodies, m.left_tip_dist),
            (env._tip_sites[1], env._elbow_bodies[1], env._right_bodies, m.right_tip_dist)):
        d_tip = pdist(env.data.site_xpos[tip_site])
        d_elbow = pdist(env.data.xpos[elbow_body])
        assert abs(got - (0.75 * d_tip + 0.25 * d_elbow)) < 1e-6   # the exact tip-dominant blend
        body_min = min(pdist(env.data.xpos[b]) for b in bodies)
        assert got >= body_min - 1e-9                              # blend never below the body-min
    # the right arm's elbow is the nearest body at seed 2, so its tip-true blend strictly exceeds it:
    rbody_min = min(pdist(env.data.xpos[b]) for b in env._right_bodies)
    assert m.right_tip_dist > rbody_min + 1e-3, "the fix must be observable (tip farther than elbow)"


def test_extract_arms_resolves_fingertip_site() -> None:
    """Demonstrator regression: ``_extract_arms`` resolves a real tip site per arm. Pre-fix it fell
    back to ``tip_site=-1`` on the emitted arm, so ``_tip_xy`` silently read ``target_zone``."""
    from hymeko_rl.experiments.galambos_demo import _extract_arms

    env = PlanarGraspEnv.from_hymeko(max_steps=10, difficulty=0.3)
    env.reset(seed=0)
    arms = _extract_arms(env.model)
    assert arms["left"].tip_site >= 0 and arms["right"].tip_site >= 0


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
    """Reach + contact detection align: a coin at the left FINGERTIP registers a left-only contact — and ONLY
    the fingertip does (Galambos 2026-07-03: arm links can no longer touch the coin, so contact = a real
    fingertip touch, not an arm knock)."""
    env = PlanarGraspEnv()
    env.reset(seed=0)
    mujoco.mj_forward(env.model, env.data)
    tip = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    assert tip >= 0, "the emitted arm must carry a fingertip collision geom"
    pos = env.data.geom_xpos[tip]          # the yellow fingertip — the only geom that can touch the coin now
    env.data.qpos[env._disk_x_adr] = float(pos[0]) + 0.02
    env.data.qpos[env._disk_y_adr] = float(pos[1])
    left = False
    for _ in range(5):                     # step a few frames so the contact registers
        mujoco.mj_step(env.model, env.data)
        m = env._metrics()
        left = left or (m.left_contact and not m.right_contact)
    assert left, "a coin at the left fingertip should register a left-only contact"


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


def test_terminate_on_success_false_holds_the_episode() -> None:
    """De-farm (oracle-verified): with ``terminate_on_success=False`` a coin held in the zone does NOT end the
    episode (so holding, not oscillating, is optimal). Regression against the default, which DOES terminate."""
    env = PlanarGraspEnv(max_steps=40, env=EnvSpec(success_steps=3), terminate_on_success=False)
    env.reset(seed=0)
    env.data.qpos[env._disk_x_adr] = env._zone_x
    env.data.qpos[env._disk_y_adr] = env._zone_y
    mujoco.mj_forward(env.model, env.data)
    zero = np.zeros(env.n_actions, dtype=np.float32)
    seen_in_zone = False
    for _ in range(40):
        _o, _r, terminated, truncated, info = env.step(zero)
        seen_in_zone = seen_in_zone or bool(info["in_zone"])
        assert not terminated, "terminate_on_success=False must not end the episode on sustained delivery"
        if truncated:
            break
    assert seen_in_zone, "coin planted at the zone centre should register in_zone"


# ── privileged state z(s) for the asymmetric-CTDE (MADDPG) critic + coin velocity in the metrics ──


def test_privileged_state_shape_and_phase_onehot() -> None:
    env = PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3)
    env.reset(seed=0)
    z = env.privileged_state()
    assert z.shape == (env.privileged_dim,) == (5,) and z.dtype == np.float32
    assert set(np.unique(z[:2]).tolist()).issubset({0.0, 1.0})   # contact bits are 0/1
    assert float(z[2:].sum()) == 1.0                             # exactly one phase bit set (one-hot)
    assert float(z[2]) == 1.0                                    # fresh reset: not grasped, not in zone -> reaching


def test_privileged_state_phase_transitions() -> None:
    env = PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3)
    env.reset(seed=1)
    env._ever_grasped = True                                     # carrying: latch grasp, coin still outside zone
    assert not env._planar_metrics.in_zone
    assert float(env.privileged_state()[3]) == 1.0              # phase = carrying
    env.data.qpos[env._disk_x_adr] = env._zone_x                # in_zone dominates: plant coin at the zone centre
    env.data.qpos[env._disk_y_adr] = env._zone_y
    mujoco.mj_forward(env.model, env.data)
    env._planar_metrics = env._metrics()
    assert env._planar_metrics.in_zone
    assert float(env.privileged_state()[4]) == 1.0             # phase = in_zone takes precedence over carrying


def test_privileged_state_contact_bits_track_metrics() -> None:
    from dataclasses import replace
    env = PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3)
    env.reset(seed=2)
    env._planar_metrics = replace(env._planar_metrics, left_contact=True, right_contact=False)
    z = env.privileged_state()
    assert float(z[0]) == 1.0 and float(z[1]) == 0.0           # z carries per-arm contact


def test_disk_vel_in_metrics_consistent_with_speed() -> None:
    env = PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3)
    env.reset(seed=3)
    m = env._planar_metrics
    assert np.asarray(m.disk_vel).shape == (2,)                # in-plane coin velocity vector
    assert abs(float(m.disk_speed) - float(np.hypot(m.disk_vel[0], m.disk_vel[1]))) < 1e-6


# ── v2 contact-legality (graded contact-quality model) ───────────────────────────────────────────────

def _park_coin_in_zone_one_short(env: PlanarGraspEnv) -> None:
    """Park the coin at the zone centre and set ``_success`` one short of a held delivery, so the NEXT
    in-zone step completes it — a deterministic delivery probe independent of physics/control."""
    env.data.qpos[env._disk_x_adr] = env._zone_x
    env.data.qpos[env._disk_y_adr] = env._zone_y
    mujoco.mj_forward(env.model, env.data)
    env._success = env.success_steps - 1


def test_v2_arm_capsules_collide_with_coin() -> None:
    """v2 (contact_legality on): the arm-link capsules DO collide with the coin at the mask level — the
    opposite of the v1 passthrough (test_compiled_model_coin_collides_with_fingertip_not_arm_capsule)."""
    env = PlanarGraspEnv(robot=None, max_steps=10, difficulty=0.3, contact_legality=True)
    env.reset(seed=0)
    m = env.model
    coin_ch = _geom_channel(m, int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "disk")))
    caps = _capsule_geoms(m)
    assert caps
    for g in caps:
        assert Collision.collide(coin_ch, _geom_channel(m, g)), f"v2: arm capsule {g} must collide with the coin"
    env.close()


def test_v2_graded_arm_body_does_not_void_delivery() -> None:
    """GRADED mode (default): an arm-body↔coin contact is tracked but does NOT void a held delivery."""
    env = PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3, contact_legality=True)
    env.reset(seed=1)
    env._arm_body_ever = True                                  # simulate an earlier arm-body contact
    _park_coin_in_zone_one_short(env)
    _o, _r, term, _tr, info = env.step(np.zeros(env.n_actions, np.float32))
    assert info["in_zone"] is True
    assert info["arm_body_contact"] is True                    # tracked ...
    assert info["delivered"] is True                           # ... but the delivery still counts (graded)
    env.close()


def test_v2_strict_arm_body_invalidates_and_terminates() -> None:
    """STRICT mode: the same arm-body contact voids the delivery and terminates the episode."""
    from dataclasses import replace

    from hymeko_rl.env.env_spec import DEFAULT_ENV
    env = PlanarGraspEnv(robot=None, env=replace(DEFAULT_ENV, contact_mode="strict"),
                         max_steps=60, difficulty=0.3, contact_legality=True)
    env.reset(seed=1)
    env._arm_body_ever = True
    _park_coin_in_zone_one_short(env)
    _o, _r, term, _tr, info = env.step(np.zeros(env.n_actions, np.float32))
    assert info["delivered"] is False                          # STRICT: arm-body contact voids the delivery
    assert term is True                                        # ... and terminates
    env.close()


def test_v1_legality_none_and_info_defaults() -> None:
    """v1 (default): no contact spec, ``legality`` is None, and the arm-body info columns default off."""
    env = PlanarGraspEnv(robot=None, max_steps=10, difficulty=0.3)
    env.reset(seed=0)
    assert env._contact_spec is None
    _o, _r, _t, _tr, info = env.step(np.zeros(env.n_actions, np.float32))
    assert env._planar_metrics.legality is None
    assert info["arm_body_contact"] is False and info["arm_body_contact_steps"] == 0
    env.close()


def test_v2_tracks_arm_body_duration_and_impulse() -> None:
    """The env accumulates per-episode arm-body contact duration (steps) and summed impulse from the state."""
    import dataclasses

    from hymeko_rl.env.contact_legality import ContactLegalityState
    env = PlanarGraspEnv(robot=None, max_steps=10, difficulty=0.3, contact_legality=True)
    env.reset(seed=0)
    forced = dataclasses.replace(
        env._metrics(), legality=ContactLegalityState(arm_body_contact=True, arm_body_contact_count=2,
                                                      arm_body_contact_impulse=1.5))
    env._metrics = lambda: forced                              # type: ignore[method-assign]
    info: dict = {}
    for _ in range(3):
        _o, _r, term, trunc, info = env.step(np.zeros(env.n_actions, np.float32))
        if term or trunc:
            break
    assert env._arm_body_ever is True
    assert env._arm_body_steps == 3 and info["arm_body_contact_steps"] == 3
    assert env._arm_body_impulse_sum == pytest.approx(4.5)     # 3 steps x 1.5
    env.close()


def test_arm_body_coin_contact_reward_penalizes_body_only_not_grasp() -> None:
    """The graded reward term penalises a body-only push (arm-body contact, NO fingertip) but not a hand touch
    during a fingertip grasp — so it does not fight the only feasible grasp."""
    import dataclasses

    from hymeko_rl.env.contact_legality import ContactLegalityState
    from hymeko_rl.env.reward import _REWARD_TERMS
    term = _REWARD_TERMS["arm_body_coin_contact"]
    a = np.zeros(4, np.float32)

    def env_with(legality: object) -> object:
        return _Stub(dataclasses.replace(_metrics(), legality=legality))

    assert term(env_with(ContactLegalityState(arm_body_contact=True)), 0.1, a) == -1.0            # body-only push
    assert term(env_with(ContactLegalityState(arm_body_contact=True,                              # hand during grasp
                                              left_fingertip_contact=True)), 0.1, a) == 0.0
    assert term(env_with(ContactLegalityState()), 0.1, a) == 0.0                                  # no contact
    assert term(env_with(None), 0.1, a) == 0.0                                                    # v1: no legality
