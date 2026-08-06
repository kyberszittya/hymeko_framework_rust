"""§5 receding-horizon feedback-expert unit tests: obs-history equivalence, failure taxonomy, planner state-safety."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.coin_v3_receding_horizon import _classify_failure
from hymeko_rl.coin_delivery.full_action_obs_history import (
    FEATURE_DIM, ObsHistoryV1, build_history_features, contract_spec,
)


def test_obs_contract_shape_and_hash_stable():
    spec = contract_spec()
    assert spec["feature_dim"] == FEATURE_DIM == 152
    assert spec["action_dim"] == 4
    assert spec["semantic_graph_fp"].startswith("sem:469094de")
    assert spec["sha256"] == contract_spec()["sha256"]          # deterministic


def test_obs_history_streaming_equals_batch():
    rng = np.random.default_rng(0)
    t = 9
    obs = rng.standard_normal((t, 48)).astype(np.float32)
    act = rng.standard_normal((t, 4)).astype(np.float32)
    traj = np.zeros(t, np.int32)
    batch = build_history_features(obs, act, traj)
    h = ObsHistoryV1()
    h.reset(obs[0])
    stream = []
    for i in range(t):
        stream.append(h.feature())
        h.push(obs[i + 1] if i + 1 < t else obs[i], act[i])
    assert np.allclose(batch, np.asarray(stream))
    assert batch.shape[1] == FEATURE_DIM


def test_obs_history_never_crosses_trajectory_boundary():
    obs = np.arange(6 * 48, dtype=np.float32).reshape(6, 48)
    act = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    traj = np.array([0, 0, 0, 1, 1, 1], np.int32)               # two trajectories
    feats = build_history_features(obs, act, traj)
    # first row of trajectory 1 (index 3) must pad from ITS first frame (obs[3]), not the previous trajectory
    assert np.allclose(feats[3, :48], obs[3])
    assert np.allclose(feats[3, 48:96], obs[3])                 # obs_{t-1} padded to obs[3]
    assert np.allclose(feats[3, 144:148], 0.0)                  # a_{t-1} padded to zero


@pytest.mark.parametrize("kw,expected", [
    (dict(strict=True, fc=True, bil=True, grasped=True, entered=True, centered=True, max_dwell=6,
          contact_loss=False, exit_after=False, start_dtz=0.2, end_dtz=0.0), None),
    (dict(strict=False, fc=False, bil=False, grasped=False, entered=False, centered=False, max_dwell=0,
          contact_loss=False, exit_after=False, start_dtz=0.2, end_dtz=0.2), "approach_failure"),
    (dict(strict=False, fc=True, bil=False, grasped=True, entered=False, centered=False, max_dwell=0,
          contact_loss=False, exit_after=False, start_dtz=0.2, end_dtz=0.19), "no_bilateral_grasp"),
    (dict(strict=False, fc=True, bil=True, grasped=True, entered=True, centered=False, max_dwell=0,
          contact_loss=False, exit_after=False, start_dtz=0.2, end_dtz=0.03), "target_entry_failure"),
    (dict(strict=False, fc=True, bil=True, grasped=True, entered=True, centered=True, max_dwell=0,
          contact_loss=False, exit_after=True, start_dtz=0.2, end_dtz=0.1), "target_exit_failure"),
    (dict(strict=False, fc=True, bil=True, grasped=True, entered=True, centered=True, max_dwell=4,
          contact_loss=False, exit_after=False, start_dtz=0.2, end_dtz=0.01), "dwell_recovery_failure"),
    (dict(strict=False, fc=True, bil=True, grasped=True, entered=False, centered=False, max_dwell=0,
          contact_loss=True, exit_after=False, start_dtz=0.2, end_dtz=0.1), "contact_loss_recovery_failure"),
])
def test_failure_classifier(kw, expected):
    assert _classify_failure(**kw) == expected


def test_plan_first_action_leaves_env_state_unchanged():
    pytest.importorskip("mujoco")
    from hymeko_rl.coin_delivery.coin_v3_receding_horizon import plan_first_action
    from hymeko_rl.experiments.coin_neutral_start import _clearance, neutral_env
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    env.set_stage(0)
    env.reset(seed=1011)
    for _ in range(30):                                          # advance into a contact/transport-ish state
        inner.step(np.zeros(4, np.float32))
    qpos0, qvel0 = inner.data.qpos.copy(), inner.data.qvel.copy()
    a, res = plan_first_action(inner, cf, _clearance(inner), True, 0, 0, horizon=6, pop=8, iters=2, elite=3)
    assert a.shape == (4,)                                       # first action in the 4-dim inner.step space
    assert np.all(np.abs(a) <= 4.0 + 1e-5)                      # respects the ctrl range
    assert np.allclose(inner.data.qpos, qpos0) and np.allclose(inner.data.qvel, qvel0)  # snapshot/restore invariant
