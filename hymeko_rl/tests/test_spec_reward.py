"""Tests for the spec_bench → CIP reward bridge (arbitrated HTL spec as a per-step MetaWorld reward).

Unit + integration coverage of :mod:`hymeko_rl.eval.spec_bench.spec_reward`: the info→signal extractor, the
``SpecRewardEnv`` reward-override (per-step robustness, potential shaping, info passthrough), the offline
reward-quality metric (separation / point-biserial / AUC), and the discriminating **thesis regression** — the
arbitrated coffee-push spec separates native success from failure far better than the raw over-constrained one.
The MetaWorld integration test is skipped when ``metaworld`` is unavailable (CI-portable)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hymeko_neuro.eval.htl import parse, robustness_at
from hymeko_rl.eval.spec_bench.spec_bench import Rollout, synth_rollouts
from hymeko_rl.eval.spec_bench.spec_reward import (
    ARBITRATED_COFFEE_SPEC,
    RAW_COFFEE_SPEC,
    SpecRewardEnv,
    SpecRewardQuality,
    _auc,
    _episode_spec_return,
    signals_from_metaworld_info,
    spec_reward_separation,
)

_REPO = Path(__file__).resolve().parents[2]
_REAL_ROLLOUTS = _REPO / "reports" / "figures" / "2026_07_13_coffee_push" / "coffee_push_rollouts.json"


# ── a scripted stub env (no MuJoCo) that emits chosen info dicts, gym-duck-typed ─────────────────────────────
class _StubEnv:
    """Minimal env: replays a queued list of ``info`` dicts as steps, with gym-like spaces."""

    def __init__(self, infos: "list[dict[str, float]]") -> None:
        self._infos = infos
        self._i = 0
        self.observation_space = SimpleNamespace(shape=(3,))
        self.action_space = SimpleNamespace(shape=(4,), high=np.ones(4), low=-np.ones(4))

    def reset(self, **_kw: Any) -> "tuple[np.ndarray, dict[str, float]]":
        self._i = 0
        return np.zeros(3, np.float32), {"obj_to_target": 1.0}

    def step(self, _action: Any) -> "tuple[np.ndarray, float, bool, bool, dict[str, float]]":
        info = self._infos[min(self._i, len(self._infos) - 1)]
        self._i += 1
        done = self._i >= len(self._infos)
        return np.zeros(3, np.float32), 0.5, done, False, dict(info)


def test_signals_from_metaworld_info_maps_in_place_reward() -> None:
    sig = signals_from_metaworld_info({"in_place_reward": 0.8, "obj_to_target": 0.05, "near_object": 1.0})
    assert sig["in_place"] == pytest.approx(0.8)          # in_place_reward → in_place alias
    assert sig["obj_to_target"] == pytest.approx(0.05)
    assert sig["near_object"] == pytest.approx(1.0)
    assert sig["grasp_success"] == 0.0                     # absent key defaults to 0.0
    assert all(np.isfinite(v) for v in sig.values())


def test_scalar_pred_robustness_is_geometric_margin() -> None:
    # rho of F(ott <= 0.071) at one event collapses to the leaf margin (0.071 - ott).
    node = parse(ARBITRATED_COFFEE_SPEC)
    from hymeko_neuro.eval.htl import HypergraphEvent
    close = robustness_at(node, HypergraphEvent(t=0.0, scalar_signals={"obj_to_target": 0.05}))
    far = robustness_at(node, HypergraphEvent(t=0.0, scalar_signals={"obj_to_target": 0.5}))
    assert close == pytest.approx(0.071 - 0.05)
    assert far == pytest.approx(0.071 - 0.5)
    assert close > 0 > far                                  # sign is the monitor verdict


def test_spec_reward_env_reward_equals_rho_and_passthrough() -> None:
    infos = [{"obj_to_target": 0.05}, {"obj_to_target": 0.5}]
    env = SpecRewardEnv(_StubEnv(infos), ARBITRATED_COFFEE_SPEC)
    env.reset()
    _o, r0, _t, _tr, i0 = env.step(np.zeros(4))
    assert r0 == pytest.approx(0.071 - 0.05)               # reward == instantaneous rho
    assert i0["env_reward"] == pytest.approx(0.5)          # original env reward preserved
    assert i0["spec_reward"] == pytest.approx(0.071 - 0.05)
    assert i0["spec_satisfied"] is True
    _o, r1, _t, _tr, i1 = env.step(np.zeros(4))
    assert r1 == pytest.approx(0.071 - 0.5)
    assert i1["spec_satisfied"] is False
    assert env.observation_space.shape == (3,)             # spaces delegate
    assert env.action_space.shape == (4,)


def test_spec_reward_env_potential_shaping_telescopes() -> None:
    # sum of potential-based rewards (gamma=1) telescopes to rho_T - rho_0.
    infos = [{"obj_to_target": 0.5}, {"obj_to_target": 0.2}, {"obj_to_target": 0.05}]
    env = SpecRewardEnv(_StubEnv(infos), ARBITRATED_COFFEE_SPEC, potential=True, gamma=1.0)
    env.reset()                                            # rho_0 from reset info (ott=1.0)
    node = parse(ARBITRATED_COFFEE_SPEC)
    from hymeko_neuro.eval.htl import HypergraphEvent
    rho0 = robustness_at(node, HypergraphEvent(0.0, {"obj_to_target": 1.0}))
    total = 0.0
    rho_last = rho0
    for info in infos:
        _o, r, _t, _tr, _i = env.step(np.zeros(4))
        total += r
        rho_last = robustness_at(node, HypergraphEvent(0.0, info))
    assert total == pytest.approx(rho_last - rho0)


def test_auc_extremes() -> None:
    assert _auc([1.0, 2.0, 3.0], [-1.0, -2.0]) == pytest.approx(1.0)     # perfect separation
    assert _auc([-1.0, -2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)     # reversed
    assert _auc([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)           # all ties
    assert _auc([], [1.0]) == pytest.approx(0.5)                         # empty guard


def test_episode_spec_return_sums_margins() -> None:
    node = parse(ARBITRATED_COFFEE_SPEC)
    roll = Rollout(trace=[{"obj_to_target": 0.05}, {"obj_to_target": 0.171}], success=True)
    # (0.071-0.05) + (0.071-0.171) = 0.021 - 0.1
    assert _episode_spec_return(node, roll) == pytest.approx(0.021 - 0.1)


def test_separation_on_synthetic_target_vs_wrong_spec() -> None:
    rolls = synth_rollouts(40, seed=0)
    # the synthetic ground-truth predicate is F(in_place >= 0.9)
    good = spec_reward_separation("F(in_place >= 0.9)", rolls)
    assert isinstance(good, SpecRewardQuality)
    assert good.auc > 0.9 and good.separation > 0 and good.point_biserial > 0.5
    # a spec on a distractor constant (grasp_success) should not separate the classes
    bad = spec_reward_separation("F(grasp_success >= 0.5)", rolls)
    assert abs(bad.auc - 0.5) < 0.2
    assert good.auc > bad.auc


def test_thesis_regression_arbitrated_beats_raw_on_real_rollouts() -> None:
    """The discriminating result: on real coffee-push rollouts the arbitrated spec's reward separates
    success from failure; the raw over-constrained spec's reward is near-flat (dead-conjunct offset)."""
    if not _REAL_ROLLOUTS.exists():
        pytest.skip(f"real rollouts not on disk: {_REAL_ROLLOUTS}")
    data = json.loads(_REAL_ROLLOUTS.read_text())
    rolls = [Rollout(trace=d["trace"], success=bool(d["success"])) for d in data]
    arb = spec_reward_separation(ARBITRATED_COFFEE_SPEC, rolls)
    raw = spec_reward_separation(RAW_COFFEE_SPEC, rolls)
    assert arb.separation > 5.0                            # arbitrated: strong success/failure gap
    assert abs(raw.separation) < 2.0                       # raw: offset-dominated, near-flat
    assert arb.auc > raw.auc                               # arbitrated ranks success far better
    assert arb.point_biserial > raw.point_biserial


def test_stageb_policy_decoupling() -> None:
    """The Stage-B scripted-policy lookup is injectable (coffee-push has no GENERIC_TASKS entry)."""
    from hymeko_rl.experiments.exp_metaworld_reward_stageb import StageBConfig, _scripted_policy_name
    assert _scripted_policy_name(StageBConfig(task="pick-place")) == "SawyerPickPlaceV3Policy"   # default path
    assert _scripted_policy_name(
        StageBConfig(task="coffee-push", policy_name="SawyerCoffeePushV3Policy")) == "SawyerCoffeePushV3Policy"


def test_coffee_cfg_pins_coffee_push() -> None:
    """The preservation config pins coffee-push via the injectable policy and carries the weak-BC head-room knobs."""
    from hymeko_rl.experiments.exp_metaworld_spec_reward_ab import _coffee_cfg
    cfg = _coffee_cfg(steps=12_000, hidden=128, seed=1, bc_demos=8, n_eval=20, bc_epochs=60, explore_std=0.05)
    assert cfg.task == "coffee-push" and cfg.policy_name == "SawyerCoffeePushV3Policy"
    assert cfg.env_id == "coffee-push-v3-goal-observable"
    assert cfg.optimizer == "ppo" and cfg.warm_start and cfg.total_env_steps == 12_000
    assert cfg.bc_demos == 8 and cfg.bc_epochs == 60 and cfg.explore_std == 0.05


def test_rl_reward_env_dispatch() -> None:
    """The Phase-2 arm→reward-env dispatch (no MuJoCo needed for construction)."""
    from hymeko_rl.eval.cip.monitor_aligned_reward import MonitorAlignedEnv
    from hymeko_rl.experiments.exp_metaworld_spec_reward_ab import _reward_env
    base = _StubEnv([{"obj_to_target": 0.05}])
    assert _reward_env("native", base) is base                       # native = untouched env
    assert isinstance(_reward_env("spec_arbitrated", base), SpecRewardEnv)
    assert isinstance(_reward_env("spec_raw", base), SpecRewardEnv)
    assert isinstance(_reward_env("monitor_aligned", base), MonitorAlignedEnv)
    with pytest.raises(ValueError, match="unknown arm"):
        _reward_env("bogus", base)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("metaworld") is None, reason="metaworld not installed")
def test_rl_arm_plumbing_runs_end_to_end() -> None:
    """A tiny from-scratch SAC run on the arbitrated arm returns a finite native-success curve (§3 pre-queue)."""
    from hymeko_rl.experiments.exp_metaworld_spec_reward_ab import (
        _certify_arm,
        _fit_coffee_obs_norm,
        run_rl_arm,
    )
    cert = _certify_arm("spec_arbitrated", n=4)
    assert set(cert) >= {"delivers", "n_success"}
    mean, std = _fit_coffee_obs_norm(n=2)
    res = run_rl_arm("spec_arbitrated", seed=0, steps=800, mean=mean, std=std, hidden=32, n_eval=2)
    assert res["arm"] == "spec_arbitrated"
    assert res["success_curve"] and all(0.0 <= c <= 1.0 for c in res["success_curve"])
    assert res["final_success"] is not None


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("metaworld") is None, reason="metaworld not installed")
def test_spec_reward_env_live_metaworld_finite_and_certifies() -> None:
    """Integration: SpecRewardEnv over a real coffee-push scripted episode gives a finite dense reward."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]
        import metaworld.policies as mp
        base = ENVS["coffee-push-v3-goal-observable"](render_mode=None)
        env = SpecRewardEnv(base, ARBITRATED_COFFEE_SPEC)
        obs, _ = env.reset(seed=0)
        pol = mp.SawyerCoffeePushV3Policy()
        rewards = []
        for _ in range(60):
            obs, r, term, trunc, info = env.step(np.clip(np.asarray(pol.get_action(obs), np.float32), -1, 1))
            rewards.append(r)
            assert {"env_reward", "spec_reward", "spec_satisfied"} <= set(info)
            if term or trunc:
                break
    assert np.all(np.isfinite(rewards))
    assert rewards[-1] > rewards[0]                        # object approaches target → rho increases
