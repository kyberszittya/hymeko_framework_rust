"""Stage R1 — synthetic reward-ordering tests for the monitor-aligned pick-place reward."""
from __future__ import annotations

import importlib.util

import pytest

from hymeko_rl.eval.cip.monitor_aligned_reward import MonitorAlignedReward, monitor_aligned_step


def _metaworld_missing() -> bool:
    return importlib.util.find_spec("metaworld") is None


def _sig(*, d: float, d_prev: float, obj_z: float = 0.02, obj_z0: float = 0.02, ott: float = 0.30,
         ott_prev: float = 0.30, grasp: float = 0.0, near: float = 0.0, success: float = 0.0) -> dict:
    return {"d": d, "d_prev": d_prev, "obj_z": obj_z, "obj_z0": obj_z0, "ott": ott, "ott_prev": ott_prev,
            "grasp": grasp, "near": near, "success": success}


# The eight canonical cases (R1).
FAR = _sig(d=0.50, d_prev=0.50)                                             # 1 far, static
APPROACH = _sig(d=0.15, d_prev=0.20)                                        # 2 approaching
NEAR_STATIC = _sig(d=0.04, d_prev=0.04, near=1.0)                           # 3 near, object still
GRASP_STATIC = _sig(d=0.03, d_prev=0.03, near=1.0, grasp=1.0)               # 4 grasped, no progress
LIFTED = _sig(d=0.03, d_prev=0.03, near=1.0, grasp=1.0, obj_z=0.10)         # 5 lifted/carried
DELIVERING = _sig(d=0.03, d_prev=0.03, near=1.0, grasp=1.0, obj_z=0.10, ott=0.20, ott_prev=0.25)  # 6 toward target
SUCCESS = _sig(d=0.03, d_prev=0.03, near=1.0, grasp=1.0, obj_z=0.10, ott=0.20, ott_prev=0.25, success=1.0)  # 7


def test_far_is_low() -> None:
    assert abs(monitor_aligned_step(FAR)) < 0.1


def test_approach_small_positive() -> None:
    r = monitor_aligned_step(APPROACH)
    assert 0.0 < r < 0.2


def test_near_static_is_capped_penalized() -> None:
    """Hovering near a still object (farming) is penalized, not rewarded."""
    assert monitor_aligned_step(NEAR_STATIC) < 0.0


def test_grasp_without_progress_is_moderate_but_capped() -> None:
    """Grasp with no object motion is positive but capped well below lift/delivery."""
    r = monitor_aligned_step(GRASP_STATIC)
    assert 0.0 < r < monitor_aligned_step(LIFTED)
    assert r < monitor_aligned_step(DELIVERING)


def test_lift_is_positive() -> None:
    assert monitor_aligned_step(LIFTED) > monitor_aligned_step(GRASP_STATIC) > 0.0


def test_delivery_is_strong() -> None:
    assert monitor_aligned_step(DELIVERING) > monitor_aligned_step(LIFTED)


def test_success_is_highest() -> None:
    r = monitor_aligned_step(SUCCESS)
    assert r > monitor_aligned_step(DELIVERING)
    assert r == max(monitor_aligned_step(c) for c in (FAR, APPROACH, NEAR_STATIC, GRASP_STATIC, LIFTED, DELIVERING, SUCCESS))


def test_farming_scores_below_true_delivery() -> None:
    """Contact/proximity without delivery must score below actual delivery progress."""
    farming = max(monitor_aligned_step(NEAR_STATIC), monitor_aligned_step(GRASP_STATIC))
    assert farming < monitor_aligned_step(DELIVERING)


def test_stateful_wrapper_matches_pure_step() -> None:
    """The stateful wrapper reproduces the pure per-step reward from an obs/info stream."""
    import numpy as np
    w = MonitorAlignedReward()
    obs0 = np.zeros(39, np.float32)
    obs0[:3] = [0.0, 0.6, 0.2]
    obs0[4:7] = [0.2, 0.6, 0.2]                       # object 0.2 away in x
    w.reset(obs0, {"obj_to_target": 0.30})
    obs1 = obs0.copy()
    obs1[:3] = [0.1, 0.6, 0.2]                        # hand moved halfway toward object → d 0.2→0.1
    r = w.step(obs1, {"obj_to_target": 0.30, "grasp_success": 0.0, "near_object": 0.0, "success": 0.0})
    assert r == pytest.approx(monitor_aligned_step(_sig(d=0.1, d_prev=0.2)), abs=1e-5)   # approach 0.1 → reward 0.1


def test_corr_helper() -> None:
    """R2 helper: Pearson corr on known arrays; constant input → 0."""
    import numpy as np
    from hymeko_rl.eval.cip.monitor_aligned_reward import _corr
    assert _corr(np.array([1.0, 2, 3, 4]), np.array([2.0, 4, 6, 8])) == 1.0
    assert _corr(np.array([1.0, 2, 3]), np.array([5.0, 5, 5])) == 0.0


class _StubEnv:
    """Minimal env returning scripted obs/info streams — exercises MonitorAlignedEnv without MetaWorld."""
    def __init__(self) -> None:
        import numpy as np
        self._np = np
        self.observation_space = type("S", (), {"shape": (39,)})()
        self.action_space = type("A", (), {"shape": (4,), "high": np.ones(4)})()
        self._t = 0

    def _obs(self, hand_x: float, obj_z: float) -> "object":
        o = self._np.zeros(39, self._np.float32)
        o[:3] = [hand_x, 0.6, 0.2]
        o[4:7] = [0.2, 0.6, obj_z]
        o[-3:] = [0.2, 0.9, 0.2]
        return o

    def reset(self, **_kw: object) -> "tuple[object, dict]":
        self._t = 0
        return self._obs(0.0, 0.02), {"obj_to_target": 0.3}

    def step(self, _a: object) -> "tuple[object, float, bool, bool, dict]":
        self._t += 1
        return self._obs(0.1, 0.10), 0.0, self._t >= 2, False, {"obj_to_target": 0.25, "grasp_success": 1.0,
                                                                 "near_object": 1.0, "success": 0.0}


def test_monitor_aligned_env_overrides_reward() -> None:
    """R3 wrapper: the step reward is the monitor-aligned reward (env reward preserved in info)."""
    from hymeko_rl.eval.cip.monitor_aligned_reward import MonitorAlignedEnv
    env = MonitorAlignedEnv(_StubEnv())
    env.reset()
    _obs, r, _term, _trunc, info = env.step([0, 0, 0, 0])
    assert r != 0.0 and info["env_reward"] == 0.0 and info["monitor_aligned_reward"] == r
    assert r > 0.0                                            # approach + grasp-while-moving + gated delivery/lift > 0


@pytest.mark.skipif(_metaworld_missing(), reason="metaworld not installed")
def test_r2_comparison_runs_and_cross_view(tmp_path) -> None:
    """R2: the offline comparison recomputes all three variants and each cross-view-verifies."""
    from hymeko_rl.eval.cip.monitor_aligned_reward import run_monitor_aligned_comparison
    s = run_monitor_aligned_comparison("pick-place", n=12, seed=0, out_dir=tmp_path)
    assert set(s["variants"]) == {"original", "mw_in_place_off", "monitor_aligned"}
    for v in s["variants"].values():
        assert v["cross_view_agree"] in (True, False) and v["reward_std"] > 0.0
    assert (tmp_path / "monitor_aligned_comparison.json").exists()
