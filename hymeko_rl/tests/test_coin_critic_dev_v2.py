"""Unit tests for the corrected V2 critic-development modules (§6-§12): causal twin critic, grouped counterfactual
labels + gated transition collector, and the actor-relevant audit. Env-touching tests use tiny seed sets (fast)."""
import numpy as np
import pytest
import torch

from hymeko_rl.coin_delivery.coin_counterfactual_labels import (
    MAGNITUDES,
    RESIDUAL_BOUND,
    capture_state_panel,
    collect_critic_transitions,
    composite,
    counterfactual_return,
    family_of,
    frozen_behavior_delta,
    residual_candidates,
)
from hymeko_rl.coin_delivery.coin_critic_audit import _bootstrap_ci, _within_pair_acc, audit_family
from hymeko_rl.coin_delivery.coin_residual_critic_causal import (
    RESIDUAL_CRITIC_STATE_DIM,
    CausalCompositeTwinCritic,
    q1_grad_wrt_action,
)
from hymeko_rl.coin_delivery.coin_counterfactual_labels import StateGroup

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"


# ── causal twin critic ──
def test_causal_critic_shapes_and_independent_heads():
    c = CausalCompositeTwinCritic()
    s = torch.randn(5, RESIDUAL_CRITIC_STATE_DIM); a = torch.randn(5, 4)
    q1, q2 = c(s, a)
    assert q1.shape == (5,) and q2.shape == (5,)
    assert not torch.allclose(q1, q2)                       # independent params ⇒ generically different
    assert torch.allclose(c.min_q(s, a), torch.min(q1, q2))


def test_causal_critic_contract_sha_stable():
    assert CausalCompositeTwinCritic().contract_sha256() == CausalCompositeTwinCritic().contract_sha256()


def test_q1_grad_wrt_action_causal_shape():
    c = CausalCompositeTwinCritic()
    g = q1_grad_wrt_action(c, np.zeros(RESIDUAL_CRITIC_STATE_DIM, np.float32), np.zeros(4, np.float32), causal=True)
    assert g.shape == (4,)


# ── candidate set / directions ──
def test_candidate_set_magnitudes_and_zero():
    cs = residual_candidates(np.zeros(4, np.float32), np.random.default_rng(0), n_random=3)
    assert cs[0][0] == "zero" and np.allclose(cs[0][1], 0.0)
    assert sum(np.allclose(d, 0.0) for _n, d, _m in cs) == 1          # exactly one zero candidate
    assert sorted({m["magnitude"] for _n, _d, m in cs}) == sorted(MAGNITUDES)
    assert len(cs) == 1 + (len(MAGNITUDES) - 1) * (8 + 3)             # basis(8) + random(3)
    for _n, d, _m in cs:
        assert np.max(np.abs(d)) <= RESIDUAL_BOUND + 1e-6             # bounded


def test_no_task_space_direction_labels():
    """Regression: directions must NOT be labeled 'toward'/'away' (a joint-space sign is not a task-space effect)."""
    cs = residual_candidates(np.ones(4, np.float32), np.random.default_rng(1))
    for name, _d, meta in cs:
        assert "toward" not in name and "away" not in name
        assert meta["kind"] in ("zero", "basis", "rand")


# ── composite / gate identity ──
def test_composite_gate_off_is_pi0_bitidentical():
    base = np.array([1.0, -2.0, 3.5, -0.5], np.float32)
    for d in (np.full(4, 0.25, np.float32), np.full(4, -0.25, np.float32), np.array([9, 9, 9, 9.], np.float32)):
        assert np.array_equal(composite(base, 0.0, d), np.clip(base, -4, 4))


def test_frozen_behavior_delta_gate_off_zero():
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert np.array_equal(frozen_behavior_delta(rng, gate_active=False), np.zeros(4, np.float32))


def test_family_of():
    assert family_of(0.01, True, True) == "settling"
    assert family_of(0.10, True, False) == "contact_retention"
    assert family_of(0.20, True, True) == "transport"
    assert family_of(0.03, True, True) == "entry"


# ── audit primitives ──
def test_within_pair_acc_perfect_and_inverted():
    q = np.array([0.0, 1.0, 2.0, 3.0]); G = np.array([0.0, 1.0, 2.0, 3.0])
    ok, tot = _within_pair_acc(q, G, 0.0)
    assert ok == tot and tot == 6
    ok2, _ = _within_pair_acc(-q, G, 0.0)
    assert ok2 == 0


def test_bootstrap_ci_bounds():
    lo, hi = _bootstrap_ci([1, 1, 1, 1], n_boot=200)
    assert lo == hi == 1.0
    lo, hi = _bootstrap_ci([0, 1, 0, 1, 0, 1], n_boot=500)
    assert 0.0 <= lo <= hi <= 1.0


def _fake_group(gid, dg_sorted):
    n = len(dg_sorted)
    # distinct deltas so composite candidate actions are distinguishable (index-recoverable by the fake critic)
    deltas = [np.zeros(4, np.float32)] + [np.full(4, 0.02 * i, np.float32) for i in range(1, n)]
    return StateGroup(group_id=gid, seed=gid, family="transport", t=10,
                      obs=np.zeros(48, np.float32), base=np.zeros(4, np.float32),
                      causal_state=np.zeros(RESIDUAL_CRITIC_STATE_DIM, np.float32), cstate={"gate": 1.0},
                      snap=None, contact0=(True, True),
                      cand_delta=deltas,
                      cand_meta=[{"magnitude": 0.0 if i == 0 else 0.05, "kind": "basis+", "dir": "d"} for i in range(n)],
                      G=list(dg_sorted), G0=dg_sorted[0],
                      outcomes=[{"contact_persist": True} for _ in range(n)])


def test_audit_family_perfect_critic_ranks_high():
    groups = [_fake_group(i, [0.0, 5.0, 10.0, -5.0]) for i in range(4)]
    # perfect critic: Q == G
    q_of = lambda g, a, h: float(g.G[[np.allclose(a, composite(g.base, 1.0, d)) for d in g.cand_delta].index(True)])
    res = audit_family(groups, q_of, lambda g: None, None, None, run_grad=False)
    t = res["transport"]
    assert t["allpair_acc"] == 1.0 and t["centered_corr_Q1_vs_dG"] > 0.99
    assert t["harmful_rej"] == 1.0 and t["prob_sel_worse_than_zero"] == 0.0


# ── env-touching (small, deterministic) ──
@pytest.mark.slow
def test_counterfactual_return_deterministic():
    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor
    from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    groups = capture_state_panel(pi0, range(6000, 6008), per_family=1, n_random=1)
    assert groups
    rl = CoinRL4Dof()
    g = groups[0]
    a = composite(g.base, 1.0, g.cand_delta[1])
    r1, _ = counterfactual_return(rl, pi0, g.snap, a)
    r2, _ = counterfactual_return(rl, pi0, g.snap, a)
    assert abs(r1 - r2) < 1e-9                              # deterministic restoration


@pytest.mark.slow
def test_collect_transitions_gate_off_is_pi0_and_keeps_truncated():
    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    trs = collect_critic_transitions(pi0, range(6000, 6003), seed=0)
    assert trs and all(("terminated" in t and "truncated" in t) for t in trs)
    assert any(t["truncated"] for t in trs)                 # truncated stored, not dropped
    for t in trs:
        assert t["cs_t"].shape == (RESIDUAL_CRITIC_STATE_DIM,) and t["enc_t"].shape == (11,)
