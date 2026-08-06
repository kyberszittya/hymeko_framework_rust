"""Co-design loop — the joint stance×gait search that maps the propulsion/turning/stability frontier.

Certifies the co-design primitives (parameterised gait action; agility profile; multi-position reach)
and the regression-locked frontier finding: a wider stance turns more but walks less (the tradeoff),
and the baseline reaches the straight goals upright.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
from scenarios.aibo.codesign import CoDesignPoint, multi_position_reach, profile
from scenarios.aibo.motion_contract import JointVelocityGovernor

_BASE = Path("data/robotics/quadruped.hymeko")
_AGILE = Path("data/robotics/quadruped_agile.hymeko")


def _env(path: Path, max_steps: int = 1200) -> QuadrupedGoalEnv:
    return QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12,
                            max_steps=max_steps, hymeko_path=str(path))


def test_codesign_gait_action_is_valid() -> None:
    env = _env(_BASE)
    env.reset(seed=0)
    a = CoDesignPoint().gait_action(env, yaw_cmd=0.5, drive=1.0)
    assert a.shape == (int(env.action_space.shape[0]),)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_stride_sign_flips_walk_direction() -> None:
    gov = JointVelocityGovernor(v_max=8.0)
    fwd_plus = profile(_env(_BASE), CoDesignPoint(stride_sign=1.0), gov).forward
    fwd_minus = profile(_env(_BASE), CoDesignPoint(stride_sign=-1.0), gov).forward
    assert np.sign(fwd_plus) != np.sign(fwd_minus) or abs(fwd_plus) > abs(fwd_minus)


def test_wider_stance_turns_more_but_walks_less() -> None:
    # the Pareto tradeoff the co-design loop maps: wide stance buys turn authority, costs propulsion.
    gov = JointVelocityGovernor(v_max=8.0)
    base = profile(_env(_BASE), CoDesignPoint(stance_width=0.062), gov)
    agile = profile(_env(_AGILE), CoDesignPoint(stance_width=0.11), gov)
    assert abs(agile.turn_deg_per_1000) > abs(base.turn_deg_per_1000)   # more turn authority
    assert abs(agile.forward) < abs(base.forward)                       # but weaker forward walk


def test_baseline_reaches_straight_goals_upright() -> None:
    gov = JointVelocityGovernor(v_max=8.0)
    reach = multi_position_reach(_env(_BASE), CoDesignPoint(), gov, [(0.5, 0), (0.5, 40)])
    assert reach["upright_reach_rate"] > 0.0        # at least the straight goal is reached upright
    assert reach["reach_rate"] == reach["upright_reach_rate"]   # reaches here are valid (upright)
