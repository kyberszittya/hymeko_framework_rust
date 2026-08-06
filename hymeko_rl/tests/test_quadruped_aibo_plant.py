"""P0 regression tests for the Aibo ERS-1000 dog plant (``data/robotics/quadruped.hymeko``).

Guards the *structure* of the 22-DOF faithful Aibo (distinct from ``test_quadruped_standing.py``, which covers the
standing reward/obs/metric): the exact DOF layout (legs 12 / head 3 / neck 1 / waist 1 / mouth 1 / ears 2 / tail 2),
that every leg joint is actuated AND leg-damped (the ``_tune_legs`` regex fix for the ``hip_abduct``/``hip_flex``/
``knee`` names), that the cute cosmetic parts (eyes/nose/muzzle/paws/tail-tip/mid-belly) are *fixed* welds carrying
no DOF, that the rest pose is upright at nominal height, and that the task is **non-trivial** (a null policy sinks
and cannot hold the dwell). Each test would fail against the prior 8-DOF box quadruped.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv

# The faithful ERS-1000 DOF layout, by group (excluding the env-promoted floating base).
LEG_JOINTS = tuple(f"{j}_{leg}" for leg in ("fl", "fr", "bl", "br")
                   for j in ("hip_abduct", "hip_flex", "knee"))          # 12
EXPRESSIVE_JOINTS = ("head_yaw", "head_pitch", "head_roll", "neck_j", "waist",
                     "mouth", "ear_l_j", "ear_r_j", "tail_pan", "tail_tilt")  # 10
COSMETIC_BODIES = ("mid_body", "muzzle", "nose", "eye_l", "eye_r", "tail_tip",
                   "paw_fl", "paw_fr", "paw_bl", "paw_br")


def _stand_env() -> QuadrupedGoalEnv:
    return QuadrupedGoalEnv(base="free", task="stand", init_noise=0.0)


def _joint_id(model: "mujoco.MjModel", name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))


def test_plant_has_22_articulated_dof_controls_12_legs() -> None:
    """The plant has the faithful ERS-1000 count (22 articulated hinge joints), and the standing action space is
    the 12 legs — the 10 expressive DOF are passively spring-held, not RL-driven."""
    env = _stand_env()
    hinges = int(np.sum(env.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
    assert hinges == 22, f"expected 22 articulated hinge joints (ERS-1000), got {hinges}"
    assert env.n_actions == 12, f"standing must control the 12 legs, got {env.n_actions}"
    assert int(env.model.nu) == 12, "only the 12 leg motors remain (expressive joints are passive springs)"


def test_leg_joints_present_and_actuated() -> None:
    """All 12 leg joints (3-axis × 4) exist and are driven by a motor (else a passive leg the policy can't use)."""
    env = _stand_env()
    actuated = {int(env.model.actuator_trnid[a, 0]) for a in range(env.model.nu)}
    for name in LEG_JOINTS:
        jid = _joint_id(env.model, name)
        assert jid >= 0, f"leg joint {name!r} missing from the emitted plant"
        assert jid in actuated, f"leg joint {name!r} has no motor (under-actuated leg)"


def test_expressive_joints_present_and_held() -> None:
    """The 10 expressive joints (head 3 / neck / waist / mouth / ears 2 / tail 2) exist but are PASSIVE — no
    motor, spring-held at neutral (so RL cannot flail them and the Aibo posture is kept)."""
    env = _stand_env()
    actuated = {int(env.model.actuator_trnid[a, 0]) for a in range(env.model.nu)}
    for name in EXPRESSIVE_JOINTS:
        jid = _joint_id(env.model, name)
        assert jid >= 0, f"expressive joint {name!r} missing"
        assert jid not in actuated, f"expressive joint {name!r} must be passive (held), not actuated"
        assert float(env.model.jnt_stiffness[jid]) > 0.0, f"expressive joint {name!r} not spring-held"
    assert int(env.model.njnt) == 23, "22 articulated hinges + 1 promoted freejoint base"


def test_cosmetic_parts_are_fixed_welds() -> None:
    """The cute face/paws/belly parts are bodies but carry NO joint (fixed welds → cosmetic, no extra DOF)."""
    env = _stand_env()
    for name in COSMETIC_BODIES:
        bid = int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name))
        assert bid >= 0, f"cosmetic body {name!r} missing"
        assert int(env.model.body_jntnum[bid]) == 0, f"cosmetic body {name!r} must be welded (0 joints)"


def test_leg_joints_are_damped() -> None:
    """The ``_tune_legs`` regex must damp all three leg-joint families (the fix for the new
    ``hip_abduct``/``hip_flex``/``knee`` names); an un-damped leg folds instead of holding a stance."""
    env = _stand_env()
    for name in LEG_JOINTS:
        dof = int(env.model.jnt_dofadr[_joint_id(env.model, name)])
        assert float(env.model.dof_damping[dof]) == pytest.approx(0.8, abs=1e-6), \
            f"leg joint {name!r} not damped — regex missed it"


def test_rest_pose_upright_at_nominal_height() -> None:
    """PRECONDITION: at rest the torso is upright (cos ≈ 1) and at its nominal standing height — else the
    uprightness sign / height reference (and thus the whole standing reward) is wrong."""
    env = _stand_env()
    env.reset(seed=0)
    assert env._torso_uprightness() > 0.95
    assert abs(float(env.data.xpos[env.torso, 2]) - env._stand_height) < 1e-3


def test_unactuated_policy_cannot_hold_standing() -> None:
    """NON-TRIVIALITY: a null (zero-torque) policy must NOT satisfy the standing dwell — the damped bent stance
    holds briefly then sinks under gravity, so holding requires active balance. Would fail if the plant were
    trivially self-stable (the failure mode of the old sagittal-only box, where survival saturated)."""
    env = _stand_env()
    env.reset(seed=0)
    dwell = best = 0
    min_z = float("inf")
    for _ in range(env.max_steps):
        _o, _r, term, trunc, info = env.step(np.zeros(env.n_actions, np.float32))
        dwell = dwell + 1 if info["standing"] else 0
        best = max(best, dwell)
        min_z = min(min_z, float(env.data.xpos[env.torso, 2]))
        if term or trunc:
            break
    assert best < 200, f"zero-control held standing {best}≥200 frames — task is trivially stable"
    assert min_z < env._stand_height - env.stand_height_tol, "torso never sank — expected an unactuated collapse"
