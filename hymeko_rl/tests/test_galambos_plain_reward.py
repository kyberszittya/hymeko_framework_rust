from __future__ import annotations

import numpy as np

from hymeko_rl.experiments.galambos_plain_reward import PlainPythonDeliverReward, make_plain_env


def test_plain_env_uses_python_reward_and_hand_authored_robot() -> None:
    env = make_plain_env(reward_kind="dense")
    assert isinstance(env.reward_spec, PlainPythonDeliverReward)
    assert env.n_actions == 4
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert "disk_to_zone" in info


def test_plain_reward_terminal_and_dense_are_finite() -> None:
    env = make_plain_env(reward_kind="dense")
    env.reset(seed=1)
    action = np.zeros(env.n_actions, dtype=np.float32)
    _obs, reward, _term, _trunc, info = env.step(action)
    assert np.isfinite(reward)
    assert {"disk_to_zone", "in_zone", "both_contact"} <= info.keys()

    terminal = PlainPythonDeliverReward(kind="terminal")
    assert terminal.evaluate(env, env._planar_metrics.disk_to_zone, action) in {0.0, 30.0}
