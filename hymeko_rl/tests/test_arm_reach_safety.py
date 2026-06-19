"""Tests for the reaching MDP's safety/config penalties (env + reward + model).

Covers: the safe task parses (and the plain task is unchanged); the env has exactly one floor;
a clean home pose is safe; joint_margin drops near a limit; the reward-term extractors map a
planted SafetyState to the right signs; zero-torque free-fall terminates as death with a bounded
penalty; the reach target is sampled outside the robot column.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from hymeko_rl.env.arm_reach_env import ArmReachEnv
from hymeko_rl.env.arm_world import _hymeko_cli
from hymeko_rl.env.reward import _REWARD_TERMS, RewardSpec
from hymeko_rl.env.safety import CLEAN_SAFETY, SafetyState
from hymeko_rl.env.termination import TerminationSpec

_SAFE = "data/robotics/arm_reach_safe_task.hymeko"
_PLAIN = "data/robotics/arm_reach_task.hymeko"
_ARM6 = "data/robotics/anthropomorphic_arm.hymeko"


def _cli() -> bool:
    try:
        _hymeko_cli()
        return True
    except FileNotFoundError:
        return False


requires_cli = pytest.mark.skipif(not _cli(), reason="hymeko CLI not built")


def _n_planes(env: ArmReachEnv) -> int:
    return sum(int(env.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)
               for g in range(env.model.ngeom))


def test_safe_task_parses_all_terms() -> None:
    spec = RewardSpec.from_hymeko(_SAFE)
    assert [k for k, _ in spec.terms] == [
        "reach_distance", "ground_penalty", "self_collision_penalty",
        "joint_limit_penalty", "below_ground_penalty"]
    w = dict(spec.terms)
    assert w["ground_penalty"] == 5.0 and w["self_collision_penalty"] == 5.0


def test_plain_task_unchanged_regression() -> None:
    assert RewardSpec.from_hymeko(_PLAIN).terms == (("reach_distance", 1.0),)


def test_env_has_exactly_one_floor() -> None:
    assert _n_planes(ArmReachEnv()) == 1
    assert ArmReachEnv()._floor_geom >= 0


def test_home_pose_is_clean() -> None:
    env = ArmReachEnv()
    env.reset(seed=0)
    env.data.qpos[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    ss = env.safety_state()
    assert not ss.ground_contact
    assert not ss.self_collision


def test_joint_margin_drops_near_a_limit() -> None:
    env = ArmReachEnv()
    env.reset(seed=0)
    env.data.qpos[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    assert env.safety_state().joint_margin > 0.9          # mid-range
    env.data.qpos[0] = float(env.model.jnt_range[0, 1])   # joint 0 at its upper limit
    mujoco.mj_forward(env.model, env.data)
    assert env.safety_state().joint_margin < 0.05


def test_reward_terms_map_planted_safety_to_signs() -> None:
    env = ArmReachEnv()
    zero = np.zeros(env.n_actions, dtype=np.float32)
    env._last_safety = SafetyState(ground_contact=True, self_collision=True,
                                   joint_margin=0.0, below_ground=True)
    assert _REWARD_TERMS["ground_penalty"](env, 0.0, zero) == -1.0
    assert _REWARD_TERMS["self_collision_penalty"](env, 0.0, zero) == -1.0
    assert _REWARD_TERMS["joint_limit_penalty"](env, 0.0, zero) == -1.0   # -(1-0)²
    assert _REWARD_TERMS["below_ground_penalty"](env, 0.0, zero) == -1.0
    env._last_safety = CLEAN_SAFETY                                       # all inert when clean
    for k in ("ground_penalty", "self_collision_penalty", "below_ground_penalty",
              "joint_limit_penalty"):
        assert _REWARD_TERMS[k](env, 0.0, zero) == 0.0


def test_safety_off_by_default_no_death() -> None:
    """Plain env (safety off): no floor injected on an emitted-style scene, no death termination."""
    env = ArmReachEnv()
    assert env.enable_safety is False
    env.reset(seed=0)
    _obs, _r, _term, _trunc, info = env.step(np.zeros(env.n_actions, dtype=np.float32))
    assert info["death"] is False


def test_safety_death_terminates_with_penalty() -> None:
    """With safety on, driving the arm hard into the floor is a death: terminate + bounded penalty."""
    env = ArmReachEnv(reward_spec=RewardSpec.from_hymeko(_SAFE), enable_safety=True)
    env.reset(seed=1)
    drive = (-env.action_space.high).astype(np.float32)   # max negative torque → slam down
    died = False
    for _ in range(env.max_steps):
        _obs, reward, terminated, truncated, info = env.step(drive)
        if info["death"]:
            died = True
            assert terminated
            assert reward < 0.0          # the bounded penalty is applied
            break
        if terminated or truncated:
            break
    assert died, "driving max torque into the floor should be a death"


def test_target_sampled_outside_the_robot() -> None:
    env = ArmReachEnv(reach_min_radius=0.15)
    for seed in range(10):
        _obs, info = env.reset(seed=seed)
        r = float(np.hypot(info["target"][0], info["target"][1]))
        assert r >= 0.15, f"seed {seed}: target radius {r:.3f} < 0.15"


# ── termination as a model-declared predicate ────────────────────────────────
def test_termination_spec_parses_from_model() -> None:
    ts = TerminationSpec.from_hymeko(_SAFE)
    assert set(ts.conditions) == {"ground_contact", "self_collision"}
    assert ts.is_terminal(SafetyState(True, False, 1.0, False))   # ground → death
    assert ts.is_terminal(SafetyState(False, True, 1.0, False))   # self-collision → death
    assert not ts.is_terminal(CLEAN_SAFETY)


def test_termination_spec_unknown_condition_raises() -> None:
    with pytest.raises(ValueError, match="termination condition"):
        TerminationSpec(("ground_contact", "not_a_condition"))


def test_empty_termination_spec_never_terminal() -> None:
    assert TerminationSpec(()).is_terminal(SafetyState(True, True, 0.0, True)) is False


@requires_cli
def test_emitted_6dof_safe_home_is_clean_and_reads_model_predicate() -> None:
    """The slim-collision + parent-child exclusion + compile-time floor seating make the 6-DOF
    arm's home pose contact-free (the spurious-self-collision blocker is resolved), and the env
    takes its death predicate from the .hymeko task profile."""
    env = ArmReachEnv.from_hymeko(_ARM6, task_profile=_SAFE, control_mode="position",
                                  ee_body="tool", enable_safety=True)
    assert set(env.termination_spec.conditions) == {"ground_contact", "self_collision"}
    env.reset(seed=0)
    env.data.qpos[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    ss = env.safety_state()
    assert not ss.ground_contact
    assert not ss.self_collision
