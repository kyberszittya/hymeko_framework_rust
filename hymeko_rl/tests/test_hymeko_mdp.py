"""HyMeKo-declared control MDP (the substrate toy): state, observation, dynamics, and objective all read from
one ``.hymeko`` (``data/robotics/toy_reach.hymeko``), solved by the same off-policy stack as the robots."""
from __future__ import annotations

import numpy as np

from hymeko_rl.env.hymeko_mdp import HymekoReachEnv

TOY = "data/robotics/toy_reach.hymeko"


def test_from_hymeko_reads_everything() -> None:
    env = HymekoReachEnv.from_hymeko(TOY)
    assert env.action_dim == 2 and env.state_dim == 4                 # state/action dims from HyMeKo
    assert env.observation_space.shape == (4, 1)                      # observed channels from HyMeKo
    assert env.action_space.shape == (2,) and env.action_scale == 1.0
    assert tuple(env.target) == (0.8, -0.6) and env.reach_radius == 0.1   # objective geometry from HyMeKo
    assert [k for k, _ in env.reward_spec.terms] == ["reach_distance", "action_cost"]   # objective from HyMeKo


def test_reset_step_contract_and_obs_shape() -> None:
    env = HymekoReachEnv.from_hymeko(TOY)
    obs, info = env.reset(seed=0)
    assert obs.shape == (4, 1) and isinstance(info, dict)
    obs, r, term, trunc, info = env.step(np.zeros(2, np.float32))
    assert obs.shape == (4, 1) and np.isfinite(r) and isinstance(term, bool) and isinstance(trunc, bool)
    assert "dist" in info


def test_reward_is_declared_spec_closer_is_better() -> None:
    # reward = 1.0·(−dist) + 0.02·(−‖f‖²): moving the mass nearer the target must raise the reward (the objective
    # really is "reach the target", read from the .hymeko — not a bespoke reward baked into the env).
    env = HymekoReachEnv.from_hymeko(TOY)
    env.reset(seed=1)
    env._x = np.array([0.8, -0.6, 0.0, 0.0])          # exactly at target
    _o, r_at, _t, _tr, info_at = env.step(np.zeros(2, np.float32))
    env._x = np.array([-0.8, 0.6, 0.0, 0.0])          # far from target
    _o, r_far, _t, _tr, info_far = env.step(np.zeros(2, np.float32))
    assert info_at["dist"] < info_far["dist"]
    assert r_at > r_far                                # closer → higher reward (the declared reach objective)


def test_objective_needs_control_but_is_reachable() -> None:
    # Zero control never reaches (the objective is non-trivial); a crude proportional push reaches often (it is
    # learnable). Guards against a degenerate toy that is solved by doing nothing or unsolvable in principle.
    env = HymekoReachEnv.from_hymeko(TOY)
    zero_hits = 0
    for ep in range(20):
        env.reset(seed=ep)
        for _ in range(env.max_steps):
            _o, _r, term, trunc, _i = env.step(np.zeros(2, np.float32))
            if term or trunc:
                zero_hits += int(term)
                break
    assert zero_hits == 0, "objective must require control (zero action should not reach)"

    p_hits = 0
    for ep in range(20):
        obs, _ = env.reset(seed=ep)
        for _ in range(env.max_steps):
            f = np.clip(3.0 * (env.target - env._x[:2]), -1, 1)
            _o, _r, term, trunc, _i = env.step(f.astype(np.float32))
            if term or trunc:
                p_hits += int(term)
                break
    assert p_hits >= 5, f"a proportional controller should reach often (task learnable); got {p_hits}/20"


def test_out_of_range_observed_index_rejected() -> None:
    import pytest
    with pytest.raises(ValueError):
        HymekoReachEnv(action_dim=2, dt=0.1, damping=0.2, force_gain=1.0, action_scale=1.0, target=(0.0, 0.0),
                       reach_radius=0.1, max_steps=60, observed=(0, 1, 2, 9),   # 9 >= state_dim 4
                       reward_spec=HymekoReachEnv.from_hymeko(TOY).reward_spec)
