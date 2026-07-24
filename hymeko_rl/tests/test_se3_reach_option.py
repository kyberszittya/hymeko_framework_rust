"""6D-0 — the SE(3) reach bound to the FROZEN runtime: the real-MuJoCo integration test of
OPTION_RL_STRUCTURED_TEMPORAL_RUNTIME_V1 (the ToyReach analogue with genuine physics).

Covers: the StructuredStateAdapter (6-D pose-error geometry), the full deploy decision through MultimodalBudgetSearch
delivering a pose certificate, side-effect-freedom of scoring (env restored), the LSTM streaming front-end, and
checkpoint/replay reproduction of the next fused state.
"""
import numpy as np
import torch

from hymeko_rl.env.se3_reach_env import SE3ReachEnv
from hymeko_rl.env.se3_reach_option import (
    SE3OptionScorer, SE3ReachAdapter, encode_state, ik_estimate, solve_reach)
from hymeko_rl.option_rl import LSTMTemporalEncoder, MultimodalProvenance, StructuredState


def _env(**kw):
    d = dict(control_mode="position", max_steps=200, reach_thresh=0.06, ang_thresh=0.35,
             start_perturb=0.2, expert_gain=0.5)
    d.update(kw)
    return SE3ReachEnv(**d)


def test_adapter_builds_structured_state_with_pose_error_geometry():
    env = _env()
    env.reset(seed=0)
    s = SE3ReachAdapter().structured(env)
    assert isinstance(s, StructuredState)
    assert s.n_nodes == env.hg.n_vertices and s.node_features.shape[1] == 18
    assert s.geometry.shape == (6,)                              # [target−ee(3) ; orientation rotvec(3)]
    assert s.metadata["task"] == "se3_reach"
    assert all(0 <= i < s.n_nodes and 0 <= j < s.n_nodes for e in s.edges for i, j in [(e[0], e[-1])])


def test_solve_reach_returns_provenance_and_does_not_mutate_env():
    env = _env()
    env.reset(seed=3)
    q_before, v_before = env.data.qpos.copy(), env.data.qvel.copy()
    prov = solve_reach(env, np.random.default_rng(0), budget=8, horizon=120)
    assert isinstance(prov, MultimodalProvenance)
    assert prov.selected.shape == (env.n_actions,)
    assert set(("reached", "pos_err", "ang_err")).issubset(prov.outcome)
    assert np.array_equal(env.data.qpos, q_before) and np.array_equal(env.data.qvel, v_before)  # scoring restored state


def test_scorer_restores_state_between_candidates():
    env = _env()
    env.reset(seed=1)
    sc = SE3OptionScorer(env, horizon=60)
    q0 = env.data.qpos.copy()
    sc.score(ik_estimate(env), np.random.default_rng(0))
    assert np.array_equal(env.data.qpos, q0)                     # candidate rollout left no trace on the shared env


def test_runtime_deploy_delivers_pose_certificate():
    """The whole frozen pipeline delivers pose certificates on the basic difficulty (proposal → search → execute)."""
    env = _env()
    reached = 0
    for s in range(12):
        env.reset(seed=s)
        prov = solve_reach(env, np.random.default_rng(100 + s), budget=10, horizon=140)
        for _ in range(140):
            _o, _r, term, trunc, info = env.step(prov.selected)
            if term and not info["death"]:
                reached += 1
                break
            if term or trunc:
                break
    assert reached >= 7                                          # ≈14/20 IK ceiling; ≥7/12 here


def test_encode_state_streaming_pipeline_shape():
    env = _env()
    env.reset(seed=0)
    enc = LSTMTemporalEncoder(in_dim=env.hg.n_vertices * 18, hidden=16, out_dim=8).eval()
    fused, hidden = encode_state(env, SE3ReachAdapter(), enc, enc.initial_hidden(1))
    assert fused.shape[0] == (env.hg.n_vertices * 18) + 6 + 1 + 8   # nodes ⊕ geom6 ⊕ phase1 ⊕ temporal8
    assert hidden[0].shape[-1] == 16


def test_checkpoint_replay_identical_next_fused_state(tmp_path):
    env = _env()
    env.reset(seed=2)
    adapter = SE3ReachAdapter()
    enc = LSTMTemporalEncoder(env.hg.n_vertices * 18, 16, 8).eval()
    import mujoco
    _f, hidden = encode_state(env, adapter, enc, enc.initial_hidden(1))
    env.step(env.expert_action)                                 # advance physics one step
    mujoco.mj_forward(env.model, env.data)                      # canonicalise kinematics (mj_step leaves xpos stale)
    expected, _ = encode_state(env, adapter, enc, hidden)
    ck = tmp_path / "se3.pt"
    torch.save({"w": enc.state_dict(), "hidden": (hidden[0], hidden[1]),
                "qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(),
                "target": env._target.copy(), "target_quat": env._target_quat.copy()}, ck)
    blob = torch.load(ck, weights_only=False)
    enc2 = LSTMTemporalEncoder(env.hg.n_vertices * 18, 16, 8)
    enc2.load_state_dict(blob["w"])
    enc2.eval()
    env2 = _env()
    env2.reset(seed=2)
    env2.data.qpos[:], env2.data.qvel[:] = blob["qpos"], blob["qvel"]
    env2._target, env2._target_quat = blob["target"], blob["target_quat"]
    mujoco.mj_forward(env2.model, env2.data)
    got, _ = encode_state(env2, adapter, enc2, blob["hidden"])
    assert np.allclose(got, expected, atol=1e-5)                # restore → identical next fused state
