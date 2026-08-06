r"""Trained walker + shared-A\* stepping-stone: plan-then-execute forward locomotion.

The end of the "wire the planned footholds into the WBC executor" thread with a demo that *executes*.
A learned footstep-walk policy (``train_footstep_walk``, CEM) gives a walker that truly moves forward;
the shared A\* planner (``footstep_planner.solve_astar`` → ``hymeko.astar_plan`` or its fallback) plans
which forward stones to land on across a corridor with gaps; the trained walker executes the plan,
stepping onto the planned stones.

Forward-only by design: a direct fidelity measurement (``reports/2026-08-06-…``) showed the marching WBC
executor does not realise *lateral* foothold commands (the swing foot lands at the nominal stance
regardless), while *forward* progress is realised by the trained gait. So the A\* routes over the forward
stone layout (which stone / stride), where the executor is functional.

# Preconditions: a trained ``policy_fn`` and the built ``footstep_env`` (mujoco + CLI). # Postconditions:
#   ``execute_plan`` commands each planned stone and reports where the feet landed + upright.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scenarios.humanoid.footstep_planner import Backend, solve_astar


@dataclass(frozen=True)
class Corridor:
    """A forward corridor of stepping stones at ``x = i·dx``; ``gaps`` are forbidden stone indices.

    # Preconditions: ``dx > 0``, ``n_stones >= 2``, ``gaps ⊂ range(n_stones)`` and excludes 0 / n−1.
    """

    dx: float
    n_stones: int
    gaps: frozenset  # of int (forbidden stone indices)

    def __post_init__(self) -> None:
        assert self.dx > 0 and self.n_stones >= 2, "need a forward corridor of >= 2 stones"
        assert 0 not in self.gaps and (self.n_stones - 1) not in self.gaps, "start/goal must be stones"

    def is_stone(self, i: int) -> bool:
        return 0 <= i < self.n_stones and i not in self.gaps

    def neighbours_id(self, i: int, max_stride: int) -> "list[tuple[int, float]]":
        """Forward steps of 1..``max_stride`` onto a valid stone; cost = stride² so single (natural)
        steps are preferred and a longer stride is spent only to clear a gap."""
        return [(i + s, float(s * s)) for s in range(1, max_stride + 1) if self.is_stone(i + s)]


def plan_stones(corridor: Corridor, start_i: int, goal_i: int, *, max_stride: int = 2,
                backend: Backend = Backend.AUTO, max_expansions: int = 100_000) -> "list[int] | None":
    r"""Plan the stone-index sequence from ``start_i`` to ``goal_i`` via the shared A\* engine.

    Returns the stone indices to land on (``start_i`` excluded), or ``None`` if the goal is unreachable
    (a gap wider than ``max_stride``). # Postconditions: every returned index is a valid stone; strictly
    increasing; the last is ``goal_i``.
    """
    assert corridor.is_stone(start_i) and corridor.is_stone(goal_i), "start/goal must be stones"
    return solve_astar(
        start_i,
        lambda i: corridor.neighbours_id(i, max_stride),
        lambda i: i == goal_i,
        lambda i: float(goal_i - i),  # remaining stones ≤ remaining strides ⇒ admissible
        max_expansions=max_expansions,
        backend=backend,
    )


@dataclass
class ExecReport:
    """Outcome of executing a stone plan with the trained walker."""

    upright: bool
    reached_x: float                 # net forward pelvis progress (m)
    on_stone_fraction: float         # fraction of steps whose landed foot hit its target stone (± tol)
    foot_xs: "list[float]" = field(default_factory=list)      # landed swing-foot x per step
    target_xs: "list[float]" = field(default_factory=list)    # commanded stone x per step


def execute_plan(env, policy_fn, stone_world_xs: "list[float]", *, tol: float = 0.04) -> ExecReport:
    r"""Drive ``env`` (a reset ``HumanoidFootstepEnv``) to step onto ``stone_world_xs`` in turn.

    Each footstep commands the next stone as the forward foothold target (``env._plan_forward_x``); the
    trained ``policy_fn(obs) -> action`` supplies the stabilising residual. Records the landed swing-foot
    x per step and whether it hit the target stone within ``tol``.

    # Preconditions: ``env`` reset; ``policy_fn`` maps an obs to a bounded action. # Postconditions: the
    #   report's ``on_stone_fraction`` and ``upright`` reflect the executed crossing.
    """
    pel_x0 = float(env.data.xpos[env._pel, 0])
    foot_xs: "list[float]" = []
    target_xs: "list[float]" = []
    fell = False
    for target_x in stone_world_xs:
        swing = "R" if env._stance == "L" else "L"
        swing_b = env._fr if swing == "R" else env._fl
        env._plan_forward_x = float(target_x)
        obs = env._obs()                                 # recompute AFTER the target is set (target-conditioned)
        _o, _r, done, trunc, _info = env.step(np.asarray(policy_fn(obs), np.float32))
        foot_xs.append(float(env.data.xpos[swing_b, 0]))
        target_xs.append(float(target_x))
        if done:
            fell = True
            break
        if trunc:
            break
    env._plan_forward_x = None
    hits = sum(abs(fx - tx) <= tol for fx, tx in zip(foot_xs, target_xs))
    frac = hits / len(foot_xs) if foot_xs else 0.0
    return ExecReport(upright=not fell, reached_x=float(env.data.xpos[env._pel, 0]) - pel_x0,
                      on_stone_fraction=frac, foot_xs=foot_xs, target_xs=target_xs)
