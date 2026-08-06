"""Planner-traced reward — potential-based shaping (PBRS) from a planner's cost-to-go.

The PPO-collapse on pick-place is a *reward* failure: the dense shaping has a farmable detour whose return beats
the sparse success, so PPO walks off the BC solution. The fix (Hajdu, 2026-06-30): trace the reward from a
**planner that solves the problem**. A planner's *cost-to-go* is a correct "distance to solved", so use it as a
potential ``Φ(s) = −cost_to_go(s)`` and shape the reward as ``F = γ·Φ(s′) − Φ(s)``.

This is **potential-based reward shaping** (Ng, Harada & Russell 1999): it is *policy-invariant* — it cannot
create a new farm, it only densifies the gradient toward the planner's solution. So the optimal policy is still
"reach success" (the sparse term), but PPO/DDPG/TD3 now get a dense signal that provably points at it.

Here the "planner" is the pick task's own subgoal decomposition (reach → grasp → lift → transport) — the same
structure the scripted demonstrator and an A*/RRT planner (``hymeko_graph::traversal``) both realise. The
kinematic limit is honest (see the plan): the cost-to-go guides the *arm*; the contact-rich *grasp* is still
learned. Phase-2 productionisation declares ``Φ`` in the ``.hymeko`` reward substrate; this module is the engine.
"""
from __future__ import annotations

from typing import Any

from hymeko_rl.env.reward import PickMetrics

# A one-unit cost charged for the (instantaneous) grasp event, so the not-grasped→grasped transition is a small
# positive potential jump (a shaping reward for grasping), not a discontinuous penalty.
_GRASP_STEP_COST = 1.0


def planner_cost_to_go(m: PickMetrics) -> float:
    """Remaining path cost to "solved" through the planner's subgoals (reach, grasp, lift, transport). Monotone
    non-increasing along the demonstrator's trajectory; 0 at success. # Postconditions ``>= 0``."""
    if m.reached:
        return 0.0
    grasping = m.left > 0.5 and m.right > 0.5
    if not grasping:                                     # reach the box, grasp it, lift it, carry it
        return m.approach + _GRASP_STEP_COST + m.lift_thresh + m.to_target
    return max(0.0, m.lift_thresh - m.lifted) + m.to_target   # grasp done: lift the rest, then carry


class PlannerTracedReward:
    """Env wrapper: ``reward = success_bonus·[reached] + (γ·Φ(s′) − Φ(s))``, ``Φ = −planner_cost_to_go``.

    A drop-in for the single-agent BC/PPO/eval machinery (delegates everything but ``reset``/``step``). The base
    objective is the sparse success; the PBRS term is policy-invariant dense guidance from the planner.

    # Preconditions ``env`` is a ``PickPlaceEnv`` (exposes ``_pick_metrics`` after ``step`` + ``info['reached']``).
    # Invariants the first step of an episode contributes no shaping (``Φ`` baseline is set there).
    """

    def __init__(self, env: Any, *, gamma: float = 0.99, success_bonus: float = 20.0) -> None:
        if not (0.0 < gamma <= 1.0) or success_bonus < 0.0:
            raise ValueError("require 0 < gamma <= 1 and success_bonus >= 0")
        self.env = env
        self.gamma = float(gamma)
        self.success_bonus = float(success_bonus)
        self._prev_phi: "float | None" = None

    def _phi(self) -> float:
        return -planner_cost_to_go(self.env._pick_metrics)

    def reset(self, *, seed: "int | None" = None, options: "dict[str, Any] | None" = None,
              ) -> "tuple[Any, dict[str, Any]]":
        self._prev_phi = None                            # baseline set on the first step
        out: tuple[Any, dict[str, Any]] = self.env.reset(seed=seed, options=options)
        return out

    def step(self, action: Any) -> "tuple[Any, float, bool, bool, dict[str, Any]]":
        obs, _r, term, trunc, info = self.env.step(action)
        phi = self._phi()
        if self._prev_phi is None:
            self._prev_phi = phi                         # first step: no shaping (F = 0)
        shaped = (self.success_bonus if info.get("reached") else 0.0) + self.gamma * phi - self._prev_phi
        self._prev_phi = phi
        return obs, float(shaped), term, trunc, info

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)                   # delegate obs space / hg / n_actions / demonstrator / …
