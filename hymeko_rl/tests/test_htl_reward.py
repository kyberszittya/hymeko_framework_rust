"""Tests for the HTL-robustness reward adapter (hymeko_rl/htl_reward.py).

The adapter reuses the non-core HTL evaluator in signedkan_wip/src/htl; these tests pin the galambos
spec, the metrics→signals mapping, the robustness sign, the RewardSpec duck-typing (drop-in, no env
change), and the per-episode delivery verdict.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.htl_reward import HtlRewardSpec, signals_from_planar


def _metrics(*, disk_to_zone: float = 0.1, tipl: float = 0.2, tipr: float = 0.2,
             speed: float = 0.0, in_zone: bool = False, lc: bool = False, rc: bool = False,
             arm_self: bool = False):  # type: ignore[no-untyped-def]
    from hymeko_rl.env.planar_grasp_env import PlanarGraspMetrics
    return PlanarGraspMetrics(np.zeros(2, np.float32), disk_to_zone, lc, rc, in_zone,
                              tipl, tipr, speed, 0.0, arm_self)


class _Stub:
    def __init__(self, m, disk_out: bool = False) -> None:  # type: ignore[no-untyped-def]
        self._planar_metrics = m
        self._disk_out = disk_out


def test_htl_spec_parses_and_is_nonempty() -> None:
    spec = HtlRewardSpec()                       # reads data/robotics/galambos_spec.htl
    assert spec.formula_text                     # comments stripped, formula remains
    assert "disk_to_zone" in spec.formula_text and "approach_l" in spec.formula_text
    assert spec._node is not None                # parsed AST


def test_signals_extractor_complete_and_finite() -> None:
    sig = signals_from_planar(_Stub(_metrics(in_zone=True, lc=True, rc=True)))
    for k in ("disk_to_zone", "approach_l", "approach_r", "disk_speed", "in_zone",
              "both_contact", "disk_oob", "arm_self_contact"):
        assert k in sig and np.isfinite(sig[k])
    assert sig["in_zone"] == 1.0 and sig["both_contact"] == 1.0  # binaries are 0/1
    assert sig["disk_oob"] == 0.0


def test_robustness_sign_far_negative_delivered_positive() -> None:
    """ρ < 0 when the arms are far and the coin is out of the zone; ρ > 0 once the coin is in the zone,
    both fingertips are on it, and it is slow — the directed assertion that the formula shapes the task."""
    spec = HtlRewardSpec()
    z = np.zeros(4, np.float32)
    far = spec.evaluate(_Stub(_metrics(disk_to_zone=0.2, tipl=0.25, tipr=0.25)), 0.2, z)
    delivered = spec.evaluate(
        _Stub(_metrics(disk_to_zone=0.02, tipl=0.04, tipr=0.04, speed=0.05, in_zone=True)), 0.02, z)
    assert far < 0.0 < delivered, f"far={far} should be <0< delivered={delivered}"


def test_robustness_focuses_on_worst_subgoal() -> None:
    """AND = min, so closing the approach (the early-binding margin) raises ρ until disk_to_zone becomes
    the worst conjunct — the natural-curriculum property the dense leaves were chosen for."""
    spec = HtlRewardSpec()
    z = np.zeros(4, np.float32)
    arms_far = spec.evaluate(_Stub(_metrics(disk_to_zone=0.1, tipl=0.3, tipr=0.3)), 0.1, z)
    arms_near = spec.evaluate(_Stub(_metrics(disk_to_zone=0.1, tipl=0.04, tipr=0.04)), 0.1, z)
    assert arms_near > arms_far, "closing the fingertips must raise ρ when approach is the worst margin"


def test_ducktypes_into_env_no_change() -> None:
    """HtlRewardSpec drops into the env's reward_spec seam (duck-typed RewardSpec.evaluate); the env
    steps with a finite reward and no env edit — the regression that the seam accepts any .evaluate."""
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv

    env = PlanarGraspEnv(reward_spec=HtlRewardSpec(), max_steps=10)
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(5):
        _o, r, _term, _trunc, _info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        total += r
    assert np.isfinite(total)


def test_episode_monitor_delivery_verdict() -> None:
    """The temporal verdict F[0,T](in_zone>0.5) flips satisfied once the coin is ever in the zone, and
    stays unsatisfied on a trace that never delivers."""
    spec = HtlRewardSpec()
    delivered = spec.episode_monitor(horizon=50)
    for t in range(10):
        delivered.observe(spec.event(_Stub(_metrics(in_zone=(t >= 7))), t))
    assert delivered.satisfied()

    never = spec.episode_monitor(horizon=50)
    for t in range(10):
        never.observe(spec.event(_Stub(_metrics(in_zone=False)), t))
    assert not never.satisfied()
