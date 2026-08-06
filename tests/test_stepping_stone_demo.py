"""Stepping-stone plan-then-execute — A* corridor planning (always) + trained-walker execution (gated).

The planner unit tests run everywhere (pure Python). The execution test needs mujoco + a trained
footstep policy fixture and skips when either is absent.
"""

from __future__ import annotations

import pytest

from scenarios.humanoid.footstep_planner import Backend
from scenarios.humanoid.stepping_stone_demo import Corridor, plan_stones


def test_plan_skips_a_gap() -> None:
    # Stones 0..9, gap at 4: the plan must double-step 3 -> 5 to cross it.
    corr = Corridor(dx=0.06, n_stones=10, gaps=frozenset({4}))
    plan = plan_stones(corr, 0, 9, max_stride=2, backend=Backend.PYTHON)
    assert plan is not None
    assert 4 not in plan                                       # never lands in the gap
    assert plan[-1] == 9                                       # reaches the goal stone
    assert all(plan[i] < plan[i + 1] for i in range(len(plan) - 1))  # strictly forward
    assert all(corr.is_stone(i) for i in plan)                 # every step is a valid stone


def test_plan_unreachable_when_gap_too_wide() -> None:
    # Two adjacent gaps (4,5) need a stride-3 jump; max_stride=2 cannot cross ⇒ None.
    corr = Corridor(dx=0.06, n_stones=10, gaps=frozenset({4, 5}))
    assert plan_stones(corr, 0, 9, max_stride=2, backend=Backend.PYTHON) is None
    assert plan_stones(corr, 0, 9, max_stride=3, backend=Backend.PYTHON) is not None  # stride 3 clears it


def test_plan_prefers_natural_single_strides() -> None:
    # No gaps: cost = stride, so the optimum is all single steps (never a needless double).
    corr = Corridor(dx=0.06, n_stones=6, gaps=frozenset())
    assert plan_stones(corr, 0, 5, max_stride=2, backend=Backend.PYTHON) == [1, 2, 3, 4, 5]


def test_execute_plan_runs_and_reports() -> None:
    """``execute_plan`` drives the target-conditioned env over a plan and returns a well-formed report.

    A mechanical exercise (a trivial zero-action policy), NOT a following claim: measurements
    (``reports/2026-08-06-stepping-stone-plan-execute.md``) show the WBC executor + this humanoid model
    do not realise a commanded foothold, so no policy makes the feet track the plan — the honest wall
    this thread reached. This test pins the plan→execute *plumbing*, not a demo that does not exist.
    """
    pytest.importorskip("mujoco")
    import numpy as np

    from scenarios.humanoid.footstep_env import HumanoidFootstepEnv
    from scenarios.humanoid.stepping_stone_demo import execute_plan
    from scenarios.humanoid.train_target_footstep import target_cfg

    env = HumanoidFootstepEnv(target_cfg(6), seed=0)
    env.reset(seed=0)
    x0 = float(env.data.xpos[env._fl if env._stance == "L" else env._fr, 0])
    corr = Corridor(dx=0.02, n_stones=6, gaps=frozenset())
    plan = plan_stones(corr, 0, 5, max_stride=2, backend=Backend.PYTHON)
    assert plan is not None
    stone_xs = [x0 + i * corr.dx for i in plan]

    ad = env.action_space.shape[0]
    report = execute_plan(env, lambda _obs: np.zeros(ad, np.float32), stone_xs)
    assert isinstance(report.upright, bool)
    assert len(report.target_xs) == len(report.foot_xs) <= len(stone_xs)
    assert 0.0 <= report.on_stone_fraction <= 1.0
    assert report.target_xs == stone_xs[: len(report.target_xs)]  # commanded the planned stones in order
