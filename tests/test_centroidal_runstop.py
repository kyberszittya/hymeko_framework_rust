"""Run-and-stop — the hard task where neural RL beats tuned-linear, with a HSTL-monitor-robustness reward.

The task genuinely has headroom (the best single linear gain tops out ~0.75 on the mixed set), the reward is
verified to BE the HSTL monitor robustness of ``G(fall_margin ≥ 0)``, and the learned policy beats the best tuned
linear controller on held-out stop-success without falling.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from scenarios.humanoid.centroidal_runstop import (
    PolicyConfig,
    RunStopConfig,
    _n_params,
    episode,
    evaluate,
    mixed_set,
    policy_actions,
    runstop_step,
    target_speed,
    train_cem,
)
from scenarios.humanoid.hstl_monitor import make_monitor


def _best_linear(cfg: RunStopConfig, pc: PolicyConfig, states: np.ndarray) -> float:
    best = 0.0
    for kp in (2, 4, 6, 8, 10, 12):
        for kd in (1, 2, 3, 4, 5):
            best = max(best, episode(None, states, cfg, pc, gains=(kp, kd))[0].mean())
    return best


@pytest.fixture(scope="module")
def trained() -> tuple:
    cfg = RunStopConfig()
    pc = PolicyConfig(iters=18)
    return train_cem(cfg, pc), cfg, pc


def test_actions_are_bounded() -> None:
    cfg, pc = RunStopConfig(), PolicyConfig()
    params = np.random.RandomState(1).standard_normal(_n_params(pc))
    feats = np.random.RandomState(2).standard_normal((40, 5)) * 4
    fx, a = policy_actions(params, feats, pc, cfg)
    assert np.all(np.abs(fx) <= cfg.fx_max + 1e-9) and np.all(np.abs(a) <= cfg.a_max + 1e-9)


def test_task_has_headroom_for_a_nonlinear_policy() -> None:
    """No single linear gain solves the run-stop across speeds — the best tops out well below 1 (room to learn)."""
    cfg, pc = RunStopConfig(), PolicyConfig()
    assert _best_linear(cfg, pc, mixed_set(cfg, offset=0.5)) < 0.85


def test_reward_is_the_hstl_monitor_robustness() -> None:
    """(2) The training reward IS the HSTL robustness of G(fall_margin≥0) = min_t (fall_pitch−|pitch|)."""
    cfg, pc = RunStopConfig(), PolicyConfig()
    x0 = np.array([[2.5, 0.5, 0.2]])
    mon = make_monitor("G(fall_margin >= 0)", "python")
    state, v_run, margins = x0.copy(), x0[:, 0].copy(), []
    for i in range(int(round(cfg.horizon / cfg.dt))):
        t = i * cfg.dt
        targ = target_speed(t, v_run, cfg)
        fx = np.clip(-4.0 * (state[:, 0] - targ), -cfg.fx_max, cfg.fx_max)
        a = np.clip(-1.0 * state[:, 1], -cfg.a_max, cfg.a_max)
        state = runstop_step(state, t, fx, a, cfg)
        margin = cfg.fall_pitch - abs(float(state[0, 2]))
        margins.append(margin)
        mon.observe(i, {"fall_margin": margin})
    ep_min = float(episode(None, x0, cfg, pc, gains=(4.0, 1.0))[2][0])
    assert abs(mon.robustness() - min(margins)) < 1e-9           # HSTL G robustness = min margin
    assert abs(ep_min - min(margins)) < 1e-9                     # the episode reward IS that robustness


def test_neural_rl_beats_tuned_linear_on_held_out(trained) -> None:
    """The positive: the neural policy beats the BEST single linear gain on held-out stop-success."""
    params, cfg, pc = trained
    rl = evaluate(params, cfg, pc, offset=0.5)
    best_lin = _best_linear(cfg, pc, mixed_set(cfg, offset=0.5))
    assert rl["stop_success"] > best_lin + 0.05                  # a genuine, held-out nonlinear-control win
    assert rl["fall_rate"] <= 0.05                               # and it stops without falling


def test_training_is_deterministic() -> None:
    cfg, pc = RunStopConfig(), PolicyConfig(iters=5, pop=24)
    assert np.allclose(train_cem(cfg, pc), train_cem(cfg, pc))


def test_evaluate_within_wall_budget() -> None:
    cfg, pc = RunStopConfig(), PolicyConfig()
    evaluate(None, cfg, pc)                                       # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        evaluate(None, cfg, pc)
        times.append(time.perf_counter() - t0)
    assert statistics.median(times) < 5.0
