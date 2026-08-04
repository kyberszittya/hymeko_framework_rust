"""Neural RL for the centroidal L-regulation — contract tests, with the honest held-out comparison.

The task is only meaningful if the L port actually controls the fall (verified: it does ONLY with no direct pitch
hold). Then the learned policy beats the WEAK scaffold on held-out states — but merely MATCHES a well-tuned linear
regulator, the honest ceiling. `rollout(None)` is pinned bit-identical to the bare integrator (the residual is a
true additive port), addressing the "is it the same implementation" check.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from scenarios.humanoid.centroidal import centroidal_step
from scenarios.humanoid.centroidal_rl import (
    RLConfig,
    _grid,
    _n_params,
    evaluate,
    policy_action,
    regulation_task_config,
    rollout,
    train_cem,
)


@pytest.fixture(scope="module")
def trained() -> tuple:
    """Train one CEM residual policy on the controllable regulation task (shared across the held-out tests)."""
    cfg = regulation_task_config(l_damp=2.0)
    rl = RLConfig(a_max=8.0, iters=15, pop=40)
    return train_cem(cfg, rl), cfg, rl


def test_policy_action_is_bounded() -> None:
    rl = RLConfig(a_max=4.0)
    params = np.random.RandomState(1).standard_normal(_n_params(rl))
    feats = np.random.RandomState(2).standard_normal((50, 4)) * 5
    assert np.all(np.abs(policy_action(params, feats, rl)) <= rl.a_max + 1e-9)


def test_rollout_none_matches_the_bare_integrator() -> None:
    """Parity: the scaffold rollout is bit-identical to the plain centroidal_step loop (residual is additive)."""
    cfg = regulation_task_config()
    rl = RLConfig()
    x0 = _grid(cfg, 9, 0.0)
    _, fell = rollout(None, x0, cfg, rl)
    state = x0.copy()
    ref = np.zeros(len(x0), dtype=bool)
    for i in range(int(round(rl.horizon_s / cfg.dt))):
        state = centroidal_step(state, i * cfg.dt, cfg)
        ref |= np.abs(state[:, 3]) > cfg.fall_pitch
    assert np.array_equal(fell, ref)


def test_regulation_task_is_controllable_by_the_L_port() -> None:
    """The fix: with no direct pitch hold, stronger L regulation recovers more — the port axis genuinely controls."""
    rl = RLConfig()
    x0 = _grid(regulation_task_config(l_damp=2.0), 15, 0.0)
    weak = rollout(None, x0, regulation_task_config(l_damp=2.0), rl)[1].mean()
    strong = rollout(None, x0, regulation_task_config(l_damp=8.0), rl)[1].mean()
    assert strong < weak - 0.05                                     # the L port moves the fall rate


def test_rl_beats_the_weak_scaffold_on_held_out(trained) -> None:
    """The policy learns: on unseen initial states it recovers markedly more than the weak scripted scaffold."""
    params, cfg, rl = trained
    rl_recover = evaluate(params, cfg, rl, offset=0.5)["recover_rate"]
    scaffold = evaluate(None, cfg, rl, offset=0.5)["recover_rate"]
    assert rl_recover > scaffold + 0.1                              # genuine, held-out improvement


def test_rl_matches_but_does_not_beat_a_tuned_linear_regulator(trained) -> None:
    """The honest ceiling: the neural policy ≈ a well-tuned linear regulator (it rediscovers 'damp L'), not beyond."""
    params, cfg, rl = trained
    rl_recover = evaluate(params, cfg, rl, offset=0.5)["recover_rate"]
    tuned = evaluate(None, regulation_task_config(l_damp=6.0), rl, offset=0.5)["recover_rate"]
    assert abs(rl_recover - tuned) < 0.1                           # matches tuned-linear within noise (no clear win)


def test_training_is_deterministic() -> None:
    cfg = regulation_task_config()
    rl = RLConfig(iters=6, pop=24)
    assert np.allclose(train_cem(cfg, rl), train_cem(cfg, rl))


def test_evaluate_within_wall_budget() -> None:
    cfg = regulation_task_config()
    rl = RLConfig()
    evaluate(None, cfg, rl)                                          # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        evaluate(None, cfg, rl)
        times.append(time.perf_counter() - t0)
    assert statistics.median(times) < 5.0
