"""Quadruped STANDING (balance) task: the ``task="stand"`` env mode, the four standing reward terms, the
``info["standing"]`` success signal, and the ``quadruped_stand`` registry wiring (env factory + DwellMetric).

Complements ``test_quadruped_env.py`` (goal-reach) — here the robot must hold upright at its nominal height on
a *free* base (it can fall). Covers: the new reward terms' sign/value + their 0-on-absent-attr guard, the
task-dispatched torso obs, the ``standing`` flag, that the rest pose is actually upright (the risk-anticipation
precondition: a mis-signed uprightness would break the whole task), and goal-mode regression.
"""
from __future__ import annotations

import time
import types

import numpy as np
import pytest

from hymeko_rl.env.quadruped_env import GOAL_REWARD, STAND_REWARD, QuadrupedGoalEnv
from hymeko_rl.env.reward import (
    _term_alive,
    _term_stand_still,
    _term_standing,
    _term_torso_height,
    _term_upright,
)
from hymeko_rl.eval.evaluate import DwellMetric, eval_metric
from hymeko_rl.eval.tasks import best_arch, get_task


def _zero_action(env: object, _obs: np.ndarray) -> np.ndarray:
    return np.zeros(env.n_actions, dtype=np.float32)   # type: ignore[attr-defined]


# ── reward terms: sign/value on a stub, and the 0-on-absent-attr guard ────────
def _stub_torso(*, up: float, z: float, h: float, vx: float, standing: bool = False) -> object:
    """A minimal duck for the standing terms: exposes just what each reads."""
    stub = types.SimpleNamespace(
        torso=0, _stand_height=h, _vx=vx, _standing=standing,
        data=types.SimpleNamespace(xpos=np.array([[0.0, 0.0, z]], dtype=np.float64)))
    stub._torso_uprightness = lambda: up   # type: ignore[attr-defined]
    return stub


def test_stand_terms_values() -> None:
    env = _stub_torso(up=0.87, z=0.42, h=0.50, vx=-0.30)
    a = np.zeros(8, dtype=np.float32)
    assert _term_upright(env, 0.0, a) == pytest.approx(0.87)
    assert _term_torso_height(env, 0.0, a) == pytest.approx(-abs(0.42 - 0.50))
    assert _term_alive(env, 0.0, a) == pytest.approx(1.0)          # constant survival bonus
    assert _term_stand_still(env, 0.0, a) == pytest.approx(-0.30)  # -|vx|


def test_stand_terms_guarded_on_foreign_env() -> None:
    """On an env with no torso/height/vx (e.g. the arm), the standing terms contribute 0 (not raise) — the
    same duck-typed contract as the pick terms. ``alive`` is the intentional exception (always +1)."""
    foreign = object()
    a = np.zeros(4, dtype=np.float32)
    assert _term_upright(foreign, 0.0, a) == 0.0
    assert _term_torso_height(foreign, 0.0, a) == 0.0
    assert _term_stand_still(foreign, 0.0, a) == 0.0
    assert _term_alive(foreign, 0.0, a) == 1.0


def test_stand_reward_terms_present() -> None:
    kinds = [k for k, _ in STAND_REWARD.terms]
    for required in ("standing", "torso_height", "upright", "stand_still", "joint_velocity"):
        assert required in kinds, f"stand reward missing {required}"
    # The unconditional `alive` bonus is DELIBERATELY gone (2026-07-03): it rewarded a crouched/collapsed-but-
    # not-inverted robot for merely existing, so the reward did not require standing. The gated `standing` term
    # (the success predicate) replaces it — reward ≡ metric.
    assert "alive" not in kinds, "unconditional `alive` must not be in STAND_REWARD (it rewards not-falling)"
    assert dict(STAND_REWARD.terms)["standing"] >= dict(STAND_REWARD.terms)["upright"], \
        "the standing (success-predicate) term must dominate the level-only shaping"


def test_stand_reward_strongly_prefers_standing_over_crouch() -> None:
    """REGRESSION (2026-07-03): the reward must score a genuine stand (upright AND at height) FAR above a level
    crouch (upright but below nominal height) — the crouch is NOT standing. The old spec (unconditional `alive`
    +0.5, upright +1, torso_height −|·|·2) separated them by only ~0.3 (crouch ≈ stand → stand_rate stayed ~0);
    the metric-aligned spec separates them by the full gated `standing` bonus. Would fail against the old reward.
    """
    a = np.zeros(8, dtype=np.float32)
    stand = _stub_torso(up=1.0, z=0.50, h=0.50, vx=0.0, standing=True)     # upright, at height → standing
    crouch = _stub_torso(up=1.0, z=0.35, h=0.50, vx=0.0, standing=False)   # level but 0.15 below nominal
    collapse = _stub_torso(up=0.2, z=0.20, h=0.50, vx=0.0, standing=False)  # tipped + low
    r_stand = STAND_REWARD.evaluate(stand, 0.0, a)
    r_crouch = STAND_REWARD.evaluate(crouch, 0.0, a)
    r_collapse = STAND_REWARD.evaluate(collapse, 0.0, a)
    assert r_stand - r_crouch > 4.0, f"standing must dominate crouch; got {r_stand:.2f} vs {r_crouch:.2f}"
    assert r_stand - r_collapse > 4.0, f"standing must dominate collapse; got {r_stand:.2f} vs {r_collapse:.2f}"
    assert _term_standing(stand, 0.0, a) == 1.0 and _term_standing(crouch, 0.0, a) == 0.0   # the gated predicate


# ── env task mode: builds, obs, standing flag, default reward, validation ─────
def test_stand_mode_default_reward() -> None:
    """task='stand' with no explicit spec uses STAND_REWARD; task='goal' keeps GOAL_REWARD (regression)."""
    assert QuadrupedGoalEnv(base="free", task="stand").reward_spec is STAND_REWARD
    assert QuadrupedGoalEnv(base="free", task="goal").reward_spec is GOAL_REWARD


def test_stand_validation() -> None:
    with pytest.raises(ValueError, match="stand_cos"):
        QuadrupedGoalEnv(base="free", task="stand", stand_cos=0.0)
    with pytest.raises(ValueError, match="stand_cos"):
        QuadrupedGoalEnv(base="free", task="stand", stand_height_tol=-0.1)


def test_rest_pose_is_upright() -> None:
    """PRECONDITION (plan risk-anticipation): the rest pose must be genuinely upright, else the uprightness
    cosine is mis-signed and the whole standing reward/metric is wrong. Guard it before any run."""
    env = QuadrupedGoalEnv(base="free", task="stand")
    env.reset(seed=0)
    assert env._torso_uprightness() > 0.9, "rest pose is not upright — standing task would be mis-signed"


def test_stand_obs_carries_height_and_upright() -> None:
    """In stand mode the torso vertex carries [z - stand_height, upright] (≈[0, 1] at the rest pose)."""
    env = QuadrupedGoalEnv(base="free", task="stand")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.hg.n_vertices, 2)
    assert float(obs[env._torso_vtx, 0]) == pytest.approx(0.0, abs=0.05)   # height error ~0 at rest
    assert float(obs[env._torso_vtx, 1]) > 0.9                             # upright ~1 at rest


def test_goal_obs_unchanged_regression() -> None:
    """Regression: task='goal' torso obs is still [dx_to_goal, vx] — the stand branch didn't alter goal mode."""
    env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=2.0)
    obs, _ = env.reset(seed=0)
    assert float(obs[env._torso_vtx, 0]) == pytest.approx(2.0, abs=0.1)


def test_standing_flag_and_positive_reward_at_rest() -> None:
    """One zero-control step from the (upright, at-height) rest pose: info['standing'] is True and the reward
    is positive (upright + alive dominate — surviving upright pays)."""
    env = QuadrupedGoalEnv(base="free", task="stand")
    env.reset(seed=0)
    _o, r, _t, _tr, info = env.step(np.zeros(env.n_actions, dtype=np.float32))
    assert isinstance(info["standing"], bool) and info["standing"]
    assert r > 0.0, f"upright rest step should pay positively, got {r}"


def test_fall_scores_not_standing() -> None:
    """A flipped torso is not 'standing' (a blow-up cannot fake success — inverse of the pick explosion)."""
    env = QuadrupedGoalEnv(base="free", task="stand")
    env.reset(seed=0)
    saw_not_standing = False
    for _ in range(env.max_steps):
        _o, _r, term, trunc, info = env.step(env.action_space.high.astype(np.float32))  # max torque → tip
        if not info["standing"]:
            saw_not_standing = True
        if term or trunc:
            break
    assert saw_not_standing, "driving the legs to tip never dropped the standing flag"


# ── registry wiring: quadruped_stand dispatches through TaskSpec + DwellMetric ─
def test_quadruped_stand_registered() -> None:
    spec = get_task("quadruped_stand")
    env = spec.make_env()
    assert env.task == "stand" and env.base == "free"
    rec = best_arch("quadruped_stand")
    assert rec.backbone == "sa_hsikan" and rec.algorithm == "td3"


def test_quadruped_stand_eval_end_to_end() -> None:
    """The scenario's declared DwellMetric runs through the shared eval loop and yields per-episode 0/1."""
    spec = get_task("quadruped_stand")
    vals = eval_metric(spec.make_env(), _zero_action, spec.metric(), n_episodes=2, seed0=0)
    assert len(vals) == 2 and all(v in (0, 1) for v in vals)
    assert isinstance(spec.metric(), DwellMetric)


# ── performance: the new stand path stays under budget (§3) ───────────────────
def test_stand_step_throughput_and_memory() -> None:
    import tracemalloc
    env = QuadrupedGoalEnv(base="free", task="stand", max_steps=200)
    rng = np.random.default_rng(0)
    times = []
    tracemalloc.start()
    for _ in range(5):
        env.reset(seed=0)
        t = time.perf_counter()
        for _ in range(200):
            env.step(rng.uniform(-1, 1, env.n_actions).astype(np.float32))
        times.append(time.perf_counter() - t)
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    times.sort()
    assert times[2] < 2.0, f"200-step median {times[2]:.3f}s over budget"
    assert peak_mb < 256.0, f"tracked peak {peak_mb:.1f} MB unexpectedly large"
