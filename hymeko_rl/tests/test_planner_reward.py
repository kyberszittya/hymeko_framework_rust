"""Planner-traced PBRS reward: cost-to-go monotonicity + the shaping wrapper."""
from typing import Any

from hymeko_rl.env.reward import PickMetrics
from hymeko_rl.control.planner_reward import PlannerTracedReward, planner_cost_to_go


def _m(approach: float, lifted: float, to_target: float, left: float, right: float, reached: bool,
       lift_thresh: float = 0.06) -> PickMetrics:
    return PickMetrics(approach=approach, left=left, right=right, lifted=lifted, lift_thresh=lift_thresh,
                       to_target=to_target, reached=reached, approach_contact=False, pre_grasp_disturb=0.0)


def test_cost_to_go_decreases_along_the_solution() -> None:
    far = planner_cost_to_go(_m(0.30, 0.0, 0.20, 0, 0, False))     # not grasped, far from box
    near = planner_cost_to_go(_m(0.05, 0.0, 0.20, 0, 0, False))    # not grasped, at the box
    grasped = planner_cost_to_go(_m(0.0, 0.0, 0.20, 1, 1, False))  # grasped, not lifted
    lifted = planner_cost_to_go(_m(0.0, 0.06, 0.10, 1, 1, False))  # lifted, carrying
    done = planner_cost_to_go(_m(0.0, 0.06, 0.0, 1, 1, True))      # reached
    assert far > near > grasped > lifted > done
    assert done == 0.0                                             # success = zero cost-to-go


class _ProgressEnv:
    """Fake pick env whose cost-to-go falls each step (approach shrinks), then 'reaches' on the last step."""
    def __init__(self) -> None:
        self.n_actions = 2
        self._k = 0
        self._pick_metrics = _m(0.30, 0.0, 0.20, 0, 0, False)

    def reset(self, *, seed: "int | None" = None, options: "dict[str, Any] | None" = None,
              ) -> "tuple[list[int], dict[str, Any]]":
        self._k = 0
        self._pick_metrics = _m(0.30, 0.0, 0.20, 0, 0, False)
        return [0], {}

    def step(self, _a: Any) -> "tuple[list[int], float, bool, bool, dict[str, Any]]":
        self._k += 1
        reached = self._k >= 4
        self._pick_metrics = _m(max(0.0, 0.30 - 0.1 * self._k), 0.0, 0.20, 0, 0, reached)
        return [0], 0.0, False, reached, {"reached": reached}


def test_pbrs_rewards_progress_and_success() -> None:
    w = PlannerTracedReward(_ProgressEnv(), gamma=0.99, success_bonus=20.0)
    w.reset(seed=0)
    _o, r1, _t, _tr, _i = w.step([0])           # first step: baseline set, ~no shaping
    _o, r2, _t, _tr, _i = w.step([0])           # progress (approach shrinks) → positive shaping
    assert abs(r1) < 0.5 and r2 > 0.0
    _o, r3, _t, _tr3, _i3 = w.step([0])
    _o, r4, _t, _tr4, i4 = w.step([0])          # reached → success bonus dominates
    assert i4["reached"] and r4 > 15.0


def test_wrapper_rejects_bad_config() -> None:
    for bad in ({"gamma": 0.0}, {"gamma": 1.5}, {"success_bonus": -1.0}):
        try:
            PlannerTracedReward(_ProgressEnv(), **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
