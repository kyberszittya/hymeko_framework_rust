"""RECEDING_HORIZON_ACTION_CHUNK_TD3_V1 contract tests (no TD3): chunk actor/critic (A), semi-MDP M-step target (B),
receding-horizon execution incl. gate-off=pi_0 (C), supervised chunk regression (E)."""
import numpy as np
import pytest
import torch

from hymeko_rl.coin_delivery.coin_chunk_td3 import (
    ACT_DIM,
    CHUNK_DIM,
    K,
    M,
    STATE_DIM,
    ChunkActor,
    ChunkTwinCritic,
    chunk_metrics,
    execute_chunk,
    semi_mdp_target,
    train_supervised_chunk,
)
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"


# ── A. chunk actor + twin critic ──
def test_chunk_actor_shape_and_bounds():
    a = ChunkActor()
    s = torch.randn(5, STATE_DIM)
    out = a(s); ch = a.chunk(s)
    assert out.shape == (5, CHUNK_DIM) and ch.shape == (5, K, ACT_DIM)
    assert out.abs().max().item() <= 4.0 + 1e-6 and K == 8 and M == 2 and CHUNK_DIM == 32 and STATE_DIM == 62


def test_chunk_twin_critic_independent():
    c = ChunkTwinCritic()
    s, ch = torch.randn(4, STATE_DIM), torch.randn(4, CHUNK_DIM)
    q1, q2 = c(s, ch)
    assert q1.shape == (4,) and not torch.allclose(q1, q2)
    assert torch.allclose(c.min_q(s, ch), torch.min(q1, q2))


# ── B. semi-MDP M-step target ──
def test_semi_mdp_no_termination_bootstraps():
    c = ChunkTwinCritic(); bs = torch.randn(1, STATE_DIM); bc = torch.randn(1, CHUNK_DIM)
    y = semi_mdp_target([1.0, 1.0], [False, False], [False, False], bs, bc, c, gamma=1.0, m=2)
    with torch.no_grad():
        boot = c.min_q(bs, bc)
    assert abs(float(y) - (2.0 + float(boot))) < 1e-5              # r0+r1 + gamma^2*min_q


def test_semi_mdp_terminated_no_bootstrap():
    c = ChunkTwinCritic(); bs = torch.randn(1, STATE_DIM); bc = torch.randn(1, CHUNK_DIM)
    y = semi_mdp_target([1.0, 1.0], [False, True], [False, False], bs, bc, c, gamma=1.0, m=2)
    assert abs(float(y) - 2.0) < 1e-5                              # stop at termination, no bootstrap


def test_semi_mdp_truncated_keeps_bootstrap():
    c = ChunkTwinCritic(); bs = torch.randn(1, STATE_DIM); bc = torch.randn(1, CHUNK_DIM)
    y = semi_mdp_target([1.0], [False], [True], bs, bc, c, gamma=0.5, m=2)
    with torch.no_grad():
        boot = c.min_q(bs, bc)
    assert abs(float(y) - (1.0 + 0.5 * float(boot))) < 1e-5        # truncation bootstraps


# ── C. receding-horizon execution ──
class _FakeGate:
    def __init__(self, g):
        self.gate = float(g)

    def update(self, *a, **k):
        pass


@pytest.mark.slow
def test_execute_chunk_gate_on_runs_chunk_prefix():
    from hymeko_rl.coin_delivery.coin_chunk_td3 import EventStateDetector, state_vec
    from hymeko_rl.coin_delivery.coin_late_start import build_late_start_bank, reconstruct_handoff
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    ls = build_late_start_bank(pi0, range(6000, 6020), per_family=1)[0]
    actor = ChunkActor()
    rl, gate, _h, rec = reconstruct_handoff(pi0, ls)
    det = EventStateDetector(); cm, cf, ev = det.state_of(rl)
    st = state_vec(rec.obs, cm, cf, ev, rec.base)
    with torch.no_grad():
        req = actor.chunk(torch.tensor(st[None]))[0].numpy()
    chunk, executed, recs, _o = execute_chunk(rl, pi0, actor, gate, st, m=M)
    assert len(recs) <= M and len(executed) == len(recs)
    if recs and recs[0]["gate_on"]:                               # gate-on ⇒ executed == requested chunk prefix (clipped)
        assert np.allclose(executed[0], np.clip(req[0], -4, 4), atol=1e-6)
    assert np.max(np.abs(executed)) <= 4.0 + 1e-6


@pytest.mark.slow
def test_execute_chunk_gate_off_is_pi0():
    from hymeko_rl.coin_delivery.coin_chunk_td3 import EventStateDetector, _pi0_action, state_vec
    from hymeko_rl.coin_delivery.coin_late_start import build_late_start_bank, reconstruct_handoff
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    ls = build_late_start_bank(pi0, range(6000, 6020), per_family=1)[0]
    actor = ChunkActor()
    rl, _gate, _h, rec = reconstruct_handoff(pi0, ls)
    det = EventStateDetector(); cm, cf, ev = det.state_of(rl)
    st = state_vec(rec.obs, cm, cf, ev, rec.base)
    expected0 = _pi0_action(pi0, rl.obs())                        # pi_0 at the current state
    _chunk, executed, recs, _o = execute_chunk(rl, pi0, actor, _FakeGate(0.0), st, m=1)   # forced gate-off
    assert recs[0]["gate_on"] is False and np.array_equal(executed[0], expected0)   # bit-identical to pi_0


# ── E. supervised chunk regression ──
def test_supervised_chunk_reduces_mse():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, STATE_DIM)).astype(np.float32)
    Y = np.clip(rng.standard_normal((64, CHUNK_DIM)), -4, 4).astype(np.float32)
    actor = train_supervised_chunk(X, Y, steps=400, log=lambda *a: None)
    m = chunk_metrics(actor, X, Y)
    assert set(m) == {"sequence_mse", "first_action_mse", "two_step_prefix_mse", "per_index_mse"}
    assert len(m["per_index_mse"]) == K and m["sequence_mse"] < float(np.mean(Y ** 2))   # learned below trivial-zero
