"""Integration tests (§7) for the schema-aware wrist/closure delivery motor path."""
from __future__ import annotations

import numpy as np

from hymeko_rl.env.pad_actuation import Phase, WristCloseController, actuator_groups, pad_joint_qpos_addrs
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv, with_fingertip_shape, with_fingertip_clamp
from hymeko_rl.coin_delivery.scenarios.kinematic_variant import with_distal_pad_orientation, with_pad_closure

_BOX = "0.004 0.016 0.02"


def _planar(tf):
    return PlanarGraspEnv(robot=None, max_steps=120, difficulty=0.5, arm_mjcf_transform=tf)


def _box(m):
    return with_fingertip_shape(m, "box", _BOX)


_E = {"E0": with_fingertip_clamp,
      "E1": lambda m: with_distal_pad_orientation(_box(m)),
      "E2": lambda m: with_pad_closure(_box(m)),
      "E3": lambda m: with_pad_closure(with_distal_pad_orientation(_box(m)))}


def test_actuator_groups_typed_by_name() -> None:
    g0 = actuator_groups(_planar(_E["E0"]).model)
    assert len(g0["ARM"]) == 4 and not g0["WRIST_YAW"] and not g0["PAD_CLOSURE"]
    g3 = actuator_groups(_planar(_E["E3"]).model)
    assert len(g3["ARM"]) == 4 and len(g3["WRIST_YAW"]) == 2 and len(g3["PAD_CLOSURE"]) == 2


def test_e0_motor_override_is_byte_identical() -> None:
    """E0 (no wrist/closure actuators) → the controller is a no-op; the arm motor passes through unchanged."""
    env = _planar(_E["E0"])
    ctrl = WristCloseController(env)
    arm = np.array([0.1, -0.2, 0.3, -0.4], np.float32)
    out = ctrl.motor_override(arm, Phase.TRANSPORT)
    assert np.array_equal(out, arm)                                # byte-identical


def test_e1_drives_only_wrist_actuators() -> None:
    env = _planar(_E["E1"])
    env.reset(seed=1)
    ctrl = WristCloseController(env)
    g = actuator_groups(env.model)
    out = ctrl.motor_override(np.zeros(env.model.nu, np.float32), Phase.WRIST_ALIGN)
    assert out.shape[0] == env.model.nu
    assert np.all(out[g["ARM"]] == 0.0)                            # arm untouched


def test_e3_drives_all_eight_actuators_finite() -> None:
    env = _planar(_E["E3"])
    env.reset(seed=1)
    ctrl = WristCloseController(env)
    out = ctrl.motor_override(np.zeros(8, np.float32), Phase.PAD_CLOSE)
    assert out.shape[0] == 8 and np.isfinite(out).all()


def test_pad_joint_qpos_addrs_and_layout() -> None:
    addrs = pad_joint_qpos_addrs(_planar(_E["E3"]).model)
    assert len(addrs) == 4 and addrs == sorted(addrs)              # 2 hinge + 2 slide, sorted for insertion


def test_wristed_restore_round_trip_exact() -> None:
    """The generalized restore pads a canonical 7-qpos snapshot into the E3 11-qpos layout and round-trips exactly."""
    from hymeko_rl.env.pad_actuation import build_wristed_contact_env
    from hymeko_rl.env.planar_snapshot import snapshot_planar
    from hymeko_rl.experiments.pedc_selection import _C1_HORIZON, _ctx, _load_pkl_bank, c1_config
    planar = _planar(_E["E3"])
    bank = _load_pkl_bank("c1_heldseed_bank.pkl", holdout=False)
    cf = build_wristed_contact_env(planar, bank, _ctx()["contract"], horizon=_C1_HORIZON, cfg=c1_config())
    cf.reset(seed=3)
    snap = snapshot_planar(planar)
    assert len(snap.qpos) == int(planar.model.nq)                  # E3 layout (11 qpos)
    assert np.isfinite(planar.data.qpos).all() and np.isfinite(planar.data.ctrl).all()


def test_closure_force_bounded_and_release_ramps_down() -> None:
    """Force target rises under contact and ramps to ~0 in RELEASE; slide command stays within the position limit."""
    env = _planar(_E["E3"])
    env.reset(seed=1)
    ctrl = WristCloseController(env)
    for _ in range(20):
        ctrl.motor_override(np.zeros(8, np.float32), Phase.PAD_CLOSE)
    hi = max(ctrl._slide_cmd.values())
    for _ in range(30):
        ctrl.motor_override(np.zeros(8, np.float32), Phase.RELEASE)
    lo = max(ctrl._slide_cmd.values())
    assert 0.0 <= hi <= ctrl.lim.slide_range                      # bounded
    assert lo <= hi                                               # release ramps the closure down
