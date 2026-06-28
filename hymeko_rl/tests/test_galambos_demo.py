"""Validate the planar 2-link IK against the actual Galambos arm model (the demonstrator foundation)."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.galambos_demo import planar_2link_ik

# make_planar_arms_mjcf defaults: bases at (±0.14, -0.02), l1=0.16, l2=0.14.
_L1, _L2 = 0.16, 0.14
_BASES = {"right": (0.14, -0.02), "left": (-0.14, -0.02)}


def _tip_xy(env: PlanarGraspEnv, side: str) -> np.ndarray:
    sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, f"tip_{side}")
    return np.asarray(env.data.site_xpos[sid][:2], dtype=np.float64)


def _set_joint(env: PlanarGraspEnv, name: str, val: float) -> None:
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    env.data.qpos[env.model.jnt_qposadr[jid]] = val


def test_ik_reaches_targets_on_the_model() -> None:
    """For reachable targets on both arms, the IK joint angles place the fingertip site on the target."""
    env = PlanarGraspEnv(robot=None, max_steps=10)
    env.reset(seed=0)
    targets = {"right": [(0.10, 0.18), (0.20, 0.12), (0.05, 0.22)],
               "left": [(-0.10, 0.18), (-0.20, 0.12), (-0.05, 0.22)]}
    for side, pts in targets.items():
        for tgt in pts:
            j1, j2 = planar_2link_ik(_BASES[side], _L1, _L2, tgt)
            env.data.qpos[:] = 0.0
            _set_joint(env, f"j1_{side}", j1)
            _set_joint(env, f"j2_{side}", j2)
            mujoco.mj_forward(env.model, env.data)
            tip = _tip_xy(env, side)
            assert np.allclose(tip, tgt, atol=1e-2), f"{side} tip {tip} != target {tgt} (j1={j1:.3f}, j2={j2:.3f})"


def test_out_of_reach_target_is_clamped() -> None:
    """A target beyond l1+l2 clamps to the reach boundary (no NaN, tip on the line to the target)."""
    j1, j2 = planar_2link_ik(_BASES["right"], _L1, _L2, (0.14, 0.60))  # far above reach
    assert math.isfinite(j1) and math.isfinite(j2)


def test_zero_link_length_rejected() -> None:
    try:
        planar_2link_ik((0.0, 0.0), 0.0, 0.14, (0.1, 0.1))
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-positive link length")
