"""The unified pluggable-metric rollout (evaluate.eval_metric) the eval_* family collapses into."""
from typing import Any

import numpy as np

from hymeko_rl.eval.evaluate import (
    FinalScalarMetric,
    FlagSeenMetric,
    StepCountMetric,
    eval_metric,
)


class _FakeEnv:
    """4-step episodes; sets in_zone on step 3, dist decreasing, so each metric has a known answer."""
    max_steps = 10

    def __init__(self) -> None:
        self._t = 0

    def reset(self, *, seed: int = 0) -> "tuple[np.ndarray, dict[str, Any]]":
        self._t = 0
        return np.zeros(2, np.float32), {}

    def step(self, _a: np.ndarray) -> "tuple[np.ndarray, float, bool, bool, dict[str, Any]]":
        self._t += 1
        info = {"in_zone": self._t >= 3, "dist": 0.5 - 0.1 * self._t}
        return np.zeros(2, np.float32), 1.0, self._t >= 4, False, info


def _af(_env: Any, _obs: np.ndarray) -> np.ndarray:
    return np.zeros(2, np.float32)


def test_flag_seen_metric_rate() -> None:
    res = eval_metric(_FakeEnv(), _af, FlagSeenMetric("in_zone"), n_episodes=5)
    assert res == [1, 1, 1, 1, 1] and sum(res) / 5 == 1.0      # in_zone fires by step 3 every episode


def test_step_count_metric() -> None:
    res = eval_metric(_FakeEnv(), _af, StepCountMetric(), n_episodes=3)
    assert res == [4, 4, 4]                                     # terminates at step 4


def test_final_scalar_metric() -> None:
    res = eval_metric(_FakeEnv(), _af, FinalScalarMetric("dist"), n_episodes=2)
    assert all(abs(r - (0.5 - 0.4)) < 1e-6 for r in res)        # last dist at step 4 = 0.1


def test_early_stop_on_metric() -> None:
    class _Stop(FlagSeenMetric):
        def on_step(self, env: Any, info: "dict[str, Any]", r: float, done: bool) -> bool:
            return True                                         # stop immediately

    res = eval_metric(_FakeEnv(), _af, _Stop("in_zone"), n_episodes=2)
    assert res == [0, 0]                                        # stopped before in_zone ever fired


def test_dwell_metric_requires_sustained_presence() -> None:
    from hymeko_rl.eval.evaluate import DwellMetric
    # _FakeEnv: in_zone fires at steps 3 and 4 (2 consecutive) then the episode ends.
    held2 = eval_metric(_FakeEnv(), _af, DwellMetric("in_zone", 2), n_episodes=3)
    assert held2 == [1, 1, 1]                                  # held for 2 steps -> a real delivery
    held3 = eval_metric(_FakeEnv(), _af, DwellMetric("in_zone", 3), n_episodes=3)
    assert held3 == [0, 0, 0]                                  # only 2 consecutive -> a graze, NOT counted
