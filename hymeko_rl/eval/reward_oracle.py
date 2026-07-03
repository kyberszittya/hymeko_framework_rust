"""Reward-Alignment Planner-Oracle — certify a declared reward's OPTIMUM achieves the task, at planning speed.

A reward is task-aligned *iff* its optimal trajectory achieves the goal. RL finds that optimum slowly and
stochastically (~27 min/seed on galambos, variance-swamped); a planner finds it **exactly, in milliseconds**. This
scores a small time-expanded abstract MDP of the grasp-and-deliver task with the **same declarative**
:class:`~hymeko_rl.env.reward.RewardSpec` (duck-typed — no reward re-implementation, the whole point: the reward is
a separable artifact), solves it by backward DP, and reports whether the reward-optimal trajectory **delivers**.

The abstraction is kinematic/phase-level (no contact dynamics): it certifies the reachability / hold / termination
structure — exactly where the *farming-vs-delivering* perverse incentive lives (grasping is a per-step annuity while
success terminates the episode) — not the full contact-rich grasp. It is a reward **certifier**, not a simulator.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from hymeko_rl.env.planar_grasp_env import CLEAN_PLANAR
from hymeko_rl.env.reward import RewardSpec


class Phase(IntEnum):
    """The reward-relevant phases of a planar grasp-and-deliver."""
    FAR = 0
    NEAR = 1
    GRASPED_OUT = 2   # both fingers on the coin, coin OUTSIDE the zone (the farming trap)
    GRASPED_IN = 3    # both fingers on the coin, coin IN the zone (delivering)


# Per-phase abstract metrics: override the clean template with the values the reward terms read.
_PHASE_METRICS = {
    Phase.FAR:         dataclasses.replace(CLEAN_PLANAR, disk_to_zone=0.30, left_tip_dist=0.30, right_tip_dist=0.30),
    Phase.NEAR:        dataclasses.replace(CLEAN_PLANAR, disk_to_zone=0.15, left_tip_dist=0.02, right_tip_dist=0.02),
    Phase.GRASPED_OUT: dataclasses.replace(CLEAN_PLANAR, disk_to_zone=0.10, left_contact=True, right_contact=True,
                                           left_tip_dist=0.0, right_tip_dist=0.0),
    Phase.GRASPED_IN:  dataclasses.replace(CLEAN_PLANAR, disk_to_zone=0.0, in_zone=True, left_contact=True,
                                           right_contact=True, left_tip_dist=0.0, right_tip_dist=0.0),
}


class _AbstractEnv:
    """Duck-typed env exposing exactly what :meth:`RewardSpec.evaluate` reads for one abstract phase."""

    def __init__(self, phase: Phase, ever_grasped: bool, zone_half: float = 0.055) -> None:
        self._planar_metrics = _PHASE_METRICS[phase]
        self._ever_grasped = ever_grasped
        self._zone_half = zone_half


@dataclass(frozen=True)
class AlignmentReport:
    """The oracle's verdict on a declared reward."""

    delivers: bool            # does the reward-OPTIMAL trajectory reach sustained delivery?
    optimal_return: float     # the optimal discounted return under the reward
    trajectory: "list[str]"   # the arg-max action sequence (phase moves)

    def summary(self) -> str:
        verb = "DELIVERS" if self.delivers else "does NOT deliver (farms the per-step reward)"
        head = " -> ".join(self.trajectory[:6]) + (" ..." if len(self.trajectory) > 6 else "")
        return f"reward-optimal {verb} | return={self.optimal_return:.1f} | {head}"


def _phase_reward(spec: RewardSpec, phase: Phase, ever_grasped: bool, n_actions: int) -> float:
    """The declared reward evaluated on an abstract phase (the SAME RewardSpec that scores the live env)."""
    env = _AbstractEnv(phase, ever_grasped)
    return spec.evaluate(env, _PHASE_METRICS[phase].disk_to_zone, np.zeros(n_actions, dtype=np.float32))


def _actions(phase: Phase, dwell: int) -> "list[tuple[Phase, int, str]]":
    """Available ``(next_phase, next_dwell, name)`` — advance toward delivery, or hold/farm in place."""
    if phase == Phase.FAR:
        return [(Phase.FAR, 0, "wait"), (Phase.NEAR, 0, "approach")]
    if phase == Phase.NEAR:
        return [(Phase.NEAR, 0, "wait"), (Phase.GRASPED_OUT, 0, "grasp")]
    if phase == Phase.GRASPED_OUT:
        return [(Phase.GRASPED_OUT, 0, "farm"), (Phase.GRASPED_IN, 0, "carry_in")]
    return [(Phase.GRASPED_IN, dwell + 1, "hold"), (Phase.GRASPED_OUT, 0, "release")]   # GRASPED_IN


def certify(spec: RewardSpec, *, n_actions: int = 4, gamma: float = 0.99, horizon: int = 300,
            success_steps: int = 5, terminal_bonus: float = 0.0) -> AlignmentReport:
    """Solve the abstract MDP for the reward-optimal trajectory; report whether it **delivers**.

    Backward DP over the time-expanded DAG (``|phase|·|dwell|·horizon`` states — milliseconds). ``terminal_bonus`` is
    a one-time reward on sustained delivery (the candidate fix: make delivering beat farming).

    # Preconditions ``horizon >= 1``, ``success_steps >= 1``, ``0 < gamma <= 1``.
    # Postconditions ``delivers`` is True iff the arg-max trajectory reaches ``dwell >= success_steps``."""
    if horizon < 1 or success_steps < 1 or not (0.0 < gamma <= 1.0):
        raise ValueError("need horizon>=1, success_steps>=1, 0<gamma<=1")
    reward = {(p, eg): _phase_reward(spec, p, eg, n_actions) for p in Phase for eg in (False, True)}
    memo: dict[tuple[Phase, int, bool, int], tuple[float, list[str]]] = {}

    def value(phase: Phase, dwell: int, eg: bool, t: int) -> tuple[float, list[str]]:
        if dwell >= success_steps:                     # sustained delivery — terminal (episode ends)
            return terminal_bonus, ["DELIVER"]
        if t >= horizon:
            return 0.0, []
        key = (phase, dwell, eg, t)
        cached = memo.get(key)
        if cached is not None:
            return cached
        r = reward[(phase, eg)]
        best: tuple[float, list[str]] = (-1e18, [])
        for nphase, ndwell, name in _actions(phase, dwell):
            neg = eg or nphase in (Phase.GRASPED_OUT, Phase.GRASPED_IN)
            vn, tn = value(nphase, ndwell, neg, t + 1)
            val = r + gamma * vn
            if val > best[0]:
                best = (val, [name, *tn])
        memo[key] = best
        return best

    ret, traj = value(Phase.FAR, 0, False, 0)
    # "Delivers" = the optimum reaches sustained delivery WITHOUT farming: no staying in GRASPED_OUT (`farm`) and
    # no leaving the zone (`release`) before success. A trajectory that oscillates in/out to milk the in-zone
    # annuity and only DELIVERs when forced at the horizon is NOT aligned — that is the farmable-reward signature.
    clean = "DELIVER" in traj and "farm" not in traj and "release" not in traj
    return AlignmentReport(delivers=clean, optimal_return=ret, trajectory=traj)


def tune_terminal_bonus(spec: RewardSpec, *, lo: float = 0.0, hi: float = 5000.0, tol: float = 1.0,
                        **kw: object) -> "float | None":
    """Smallest terminal delivery bonus that makes the reward-optimal **deliver** (binary search), or ``None`` if
    even ``hi`` does not — meaning the per-step grasp annuity is unbeatable by a terminal bonus and must be
    *de-annuitised* instead (bound/one-shot the grasp reward, or add a per-step living cost). # Preconditions
    ``lo < hi``, ``tol > 0``."""
    if lo >= hi or tol <= 0:
        raise ValueError("need lo < hi and tol > 0")
    if not certify(spec, terminal_bonus=hi, **kw).delivers:   # type: ignore[arg-type]
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if certify(spec, terminal_bonus=mid, **kw).delivers:  # type: ignore[arg-type]
            hi = mid
        else:
            lo = mid
    return hi
