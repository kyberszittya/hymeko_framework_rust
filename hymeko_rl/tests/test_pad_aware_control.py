"""Tests for PAD-AWARE-COOPERATIVE-CONTROL-0 (closed-loop distal pad-orientation control).

Covers: K0 action output unchanged; K1 exposes exactly two distal actuators; neutral P0 holds the pads; geometry-aligned
P1 rotates the pad toward the object (reduces misalignment); scramble P3 is capacity-matched (sign-inverted, same dim);
distal targets within limits; the bilateral-balance metric; the common monitor runs on the pad-aware env; determinism.
"""
from __future__ import annotations

import mujoco
import numpy as np

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.experiments.exp_v3_handoff_gate import _DIFF, _MAX_STEPS
from hymeko_rl.experiments.pedc_selection import _C1_HORIZON, _ctx, _load_pkl_bank, c1_config
from hymeko_rl.train.coin_delivery_actor import DeliveryActor, rollout_delivery
from hymeko_rl.train.pad_aware_control import (
    PadAwareContactFormationEnv,
    PadParams,
    PadVariant,
    _pad_facing,
    _signed_angle,
    bilateral_balance,
    desired_pad_dir,
    distal_targets,
)


def _k1_planar():
    from hymeko_rl.coin_delivery.scenarios.kinematic_variant import with_distal_pad_orientation
    return PlanarGraspEnv(robot=None, max_steps=_MAX_STEPS, difficulty=_DIFF, task_graph=True,
                          arm_mjcf_transform=with_distal_pad_orientation)


def _pad_env(variant):
    return PadAwareContactFormationEnv(_k1_planar(), _load_pkl_bank("c1_heldseed_bank.pkl", holdout=False),
                                       _ctx()["contract"], horizon=_C1_HORIZON, cfg=c1_config(), variant=variant, seed=0)


# --- K0 / K1 kinematics ------------------------------------------------------------------------------------------
def test_k0_action_output_unchanged() -> None:
    m = PlanarGraspEnv(robot=None, max_steps=_MAX_STEPS, difficulty=_DIFF, task_graph=True).model
    assert m.nu == 4                                                  # K0: 4 arm servos, no distal DOF


def test_k1_exposes_exactly_two_distal_actuators() -> None:
    m = _k1_planar().model
    pads = [i for i in range(m.nu) if str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)).startswith("a_pad_")]
    assert m.nu == 6 and len(pads) == 2                              # 4 arm + exactly 2 distal


# --- pad-orientation control -------------------------------------------------------------------------------------
def test_neutral_p0_holds_pads_at_zero() -> None:
    env = _pad_env(PadVariant.P0_NEUTRAL)
    env.reset(seed=64_000)
    assert np.allclose(distal_targets(env._env, PadVariant.P0_NEUTRAL), 0.0)


def test_geometry_aligned_reduces_misalignment() -> None:
    # P1 drives the pad toward the coin: after stepping under P1, the pad-normal misalignment to the coin is small.
    env = _pad_env(PadVariant.P1_GEOMETRY)
    env.reset(seed=64_003)
    inner = env._env
    from hymeko_rl.train.coin_delivery_actor import actor_action
    for t in range(40):
        env.step(np.clip(actor_action(inner, t, DeliveryActor.A4_RECOVERY), -1, 1).astype(np.float32))
    ang = abs(_signed_angle(_pad_facing(inner, "left"), desired_pad_dir(inner, "left", PadVariant.P1_GEOMETRY, PadParams())))
    assert ang < 0.2                                                 # the controller aligns the pad (not saturated away)


def test_scramble_is_capacity_matched_sign_inverted() -> None:
    env = _pad_env(PadVariant.P1_GEOMETRY)
    env.reset(seed=64_005)
    for _ in range(20):
        env._env.step(np.zeros(6, np.float32))
    p1 = distal_targets(env._env, PadVariant.P1_GEOMETRY)
    p3 = distal_targets(env._env, PadVariant.P3_SCRAMBLE)
    assert p1.shape == p3.shape == (2,) and np.allclose(p3, -p1)     # same dimensionality, sign-inverted


def test_distal_targets_within_limits() -> None:
    env = _pad_env(PadVariant.P2_PUSH_AWARE)
    env.reset(seed=64_007)
    for _ in range(30):
        env._env.step(np.zeros(6, np.float32))
    assert np.all(np.abs(distal_targets(env._env, PadVariant.P2_PUSH_AWARE)) <= 1.2 + 1e-6)


# --- metric + monitor + determinism ------------------------------------------------------------------------------
def test_bilateral_balance_metric() -> None:
    assert bilateral_balance(0.5, 0.5) == 1.0                        # perfectly shared
    assert bilateral_balance(1.0, 0.0) == 0.0                        # one-finger
    assert abs(bilateral_balance(0.75, 0.25) - 0.5) < 1e-6


def test_common_monitor_runs_on_pad_env() -> None:
    env = _pad_env(PadVariant.P1_GEOMETRY)
    r = rollout_delivery(env, 64_000, DeliveryActor.A1_VPLOW)
    assert isinstance(r.strict_delivery, bool)
    for k in ("alpha_L", "alpha_R", "alpha_body", "alpha_support", "alpha_free"):
        assert k in r.attribution


def test_pad_env_reset_is_deterministic() -> None:
    env = _pad_env(PadVariant.P1_GEOMETRY)
    a = rollout_delivery(env, 64_002, DeliveryActor.A4_RECOVERY)
    b = rollout_delivery(env, 64_002, DeliveryActor.A4_RECOVERY)
    assert a.progress == b.progress and a.mechanism == b.mechanism    # same seed → same rollout
