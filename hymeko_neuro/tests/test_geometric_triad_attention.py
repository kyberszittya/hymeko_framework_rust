"""Tests for the geometric (quaternion + Clifford) triad attention pool.

Plan: docs/plans/2026-06-17-geometric-attention-head/.
Run: pytest -p no:randomly hymeko_neuro/tests/test_geometric_triad_attention.py
"""
from __future__ import annotations

import pytest
import torch

from hymeko_neuro.hyperedge.geometric_triad_attention import (
    GeometricTriadAttentionPool,
    build_vertex_triad_pairs,
    summarise_gate,
)


def _toy(channel="both", seed=0):
    torch.manual_seed(seed)
    pool = GeometricTriadAttentionPool(d_node=8, d_triad=8, hidden=8, d_out=8,
                                       channel=channel)
    h_node = torch.randn(5, 8)
    h_triad = torch.randn(3, 8)
    triad_v = torch.tensor([[0, 1, 2], [1, 2, 3], [0, 2, 3]])
    inc_v, inc_t = build_vertex_triad_pairs(triad_v)
    return pool, h_node, h_triad, inc_v, inc_t


# ── helpers ──────────────────────────────────────────────────────────
def test_build_vertex_triad_pairs():
    triad_v = torch.tensor([[0, 1, 2], [3, 4, 5]])
    inc_v, inc_t = build_vertex_triad_pairs(triad_v)
    assert inc_v.tolist() == [0, 1, 2, 3, 4, 5]
    assert inc_t.tolist() == [0, 0, 0, 1, 1, 1]


def test_hidden_not_div4_raises():
    with pytest.raises(ValueError):
        GeometricTriadAttentionPool(d_node=8, d_triad=8, hidden=6, d_out=8)


def test_bad_channel_raises():
    with pytest.raises(ValueError):
        GeometricTriadAttentionPool(d_node=8, d_triad=8, hidden=8, d_out=8,
                                    channel="octonion")


# ── shapes / pooling correctness ─────────────────────────────────────
def test_output_shape():
    pool, h_node, h_triad, inc_v, inc_t = _toy()
    out = pool(h_node, h_triad, inc_v, inc_t)
    assert out.shape == (5, 8)


def test_isolated_vertex_is_zero():
    """A vertex in no triad must get an exact zero row (den=0)."""
    pool = GeometricTriadAttentionPool(d_node=8, d_triad=8, hidden=8, d_out=8)
    h_node = torch.randn(6, 8)            # vertex 5 appears in no triad
    h_triad = torch.randn(2, 8)
    inc_v, inc_t = build_vertex_triad_pairs(torch.tensor([[0, 1, 2], [2, 3, 4]]))
    out = pool(h_node, h_triad, inc_v, inc_t)
    assert torch.allclose(out[5], torch.zeros(8))


def test_scatter_matches_reference_loop():
    """The index_add pool + magnitude-norm must equal a per-vertex loop."""
    pool, h_node, h_triad, inc_v, inc_t = _toy(channel="quaternion", seed=1)
    pool.eval()
    with torch.no_grad():
        out = pool(h_node, h_triad, inc_v, inc_t)
        q = pool.W_q(h_node)[inc_v]
        k = pool.W_k(h_triad)[inc_t]
        w = pool._score(q, k)
        v = pool.W_v(h_triad)[inc_t]
        num = torch.zeros(5, 8)
        den = torch.zeros(5)
        for p in range(inc_v.shape[0]):
            num[inc_v[p]] += w[p] * v[p]
            den[inc_v[p]] += w[p].abs()
        ref = num / (den.unsqueeze(-1) + pool.eps)
    assert torch.allclose(out, ref, atol=1e-5)


# ── the two channels are genuinely distinct geometries ───────────────
def test_quaternion_and_clifford_channels_differ():
    """(+,-,-,-) Hamilton real part vs (+,+,+,-) Cl(2,0) scalar must differ."""
    _, h_node, h_triad, inc_v, inc_t = _toy(seed=2)
    outs = {}
    for ch in ("quaternion", "clifford"):
        torch.manual_seed(7)             # identical projection init per channel
        p = GeometricTriadAttentionPool(d_node=8, d_triad=8, hidden=8, d_out=8,
                                        channel=ch)
        outs[ch] = p(h_node, h_triad, inc_v, inc_t)
    assert not torch.allclose(outs["quaternion"], outs["clifford"], atol=1e-4)


def test_weights_in_unit_range():
    """tanh score keeps per-incidence signed weights in [-1, 1]."""
    pool, h_node, h_triad, inc_v, inc_t = _toy(seed=3)
    q = pool.W_q(h_node)[inc_v]
    k = pool.W_k(h_triad)[inc_t]
    w = pool._score(q, k)
    assert w.abs().max() <= 1.0 + 1e-6


# ── trainability ─────────────────────────────────────────────────────
def test_params_receive_gradient():
    pool, h_node, h_triad, inc_v, inc_t = _toy(channel="both", seed=4)
    out = pool(h_node, h_triad, inc_v, inc_t)
    out.pow(2).sum().backward()
    assert pool.W_q.weight.grad is not None
    assert pool.W_q.weight.grad.abs().sum() > 0
    assert pool.gate.grad is not None          # gate is live in "both" mode


def test_one_step_decreases_loss():
    pool, h_node, h_triad, inc_v, inc_t = _toy(channel="both", seed=5)
    target = torch.zeros(5, 8)
    opt = torch.optim.SGD(pool.parameters(), lr=0.5)
    losses = []
    for _ in range(3):
        out = pool(h_node, h_triad, inc_v, inc_t)
        loss = (out - target).pow(2).mean()
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]


# ── gate diagnostic (summarise_gate) ─────────────────────────────────
def test_summarise_gate_keys_ranges_and_init():
    """Default gate=0 -> σ=0.5; fractions in [0,1]; norms non-negative."""
    pool, h_node, h_triad, inc_v, inc_t = _toy(channel="both", seed=6)
    s = summarise_gate(pool, h_node, h_triad, inc_v, inc_t)
    assert s["channel"] == "both"
    assert abs(s["gate_sigma"] - 0.5) < 1e-6          # σ(0)
    for k in ("w_frac_dead", "w_frac_saturated"):
        assert 0.0 <= s[k] <= 1.0
    assert s["w_abs_mean"] >= 0.0 and s["w_abs_std"] >= 0.0
    assert s["pool_norm_mean"] >= 0.0 and s["hv_norm_mean"] > 0.0
    assert s["pool_to_hv"] >= 0.0
    assert s["n_incidences"] == inc_v.shape[0]


def test_summarise_gate_zero_query_kills_weights():
    """Zeroing W_q makes every geometric score 0 -> all signed weights dead.

    Both channels are bilinear in q, so q_flat=0 vanishes the quaternion real
    part and the Clifford scalar alike, independent of the gate."""
    pool, h_node, h_triad, inc_v, inc_t = _toy(channel="both", seed=7)
    with torch.no_grad():
        pool.W_q.weight.zero_()
    s = summarise_gate(pool, h_node, h_triad, inc_v, inc_t)
    assert s["w_abs_mean"] < 1e-6
    assert s["w_frac_dead"] == 1.0
    assert s["w_frac_saturated"] == 0.0


def test_summarise_gate_sigma_tracks_param():
    """gate_sigma is σ(gate): large +ve -> quaternion-only (→1), -ve -> (→0)."""
    pool, h_node, h_triad, inc_v, inc_t = _toy(channel="both", seed=8)
    with torch.no_grad():
        pool.gate.fill_(8.0)
    assert summarise_gate(pool, h_node, h_triad, inc_v, inc_t)["gate_sigma"] > 0.999
    with torch.no_grad():
        pool.gate.fill_(-8.0)
    assert summarise_gate(pool, h_node, h_triad, inc_v, inc_t)["gate_sigma"] < 0.001


# ── waking the score: init scale / learn scale / sign-aware ───────────
def _balance(inc_t, seed=0):
    """A deterministic ±1 per-incidence balance vector aligned with inc_t."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, inc_t.shape, generator=g) * 2 - 1).float()


def test_default_knobs_reproduce_legacy_weight():
    """sign_aware=False + defaults: _weights is exactly the legacy tanh _score."""
    pool, h_node, h_triad, inc_v, inc_t = _toy(channel="both", seed=10)
    q = pool.W_q(h_node)[inc_v]
    k = pool.W_k(h_triad)[inc_t]
    assert torch.allclose(pool._weights(q, k, None), pool._score(q, k))


def test_score_init_scale_controls_projection_magnitude():
    """score_init_scale=1.0 leaves W_q at full init; 0.1 shrinks it ~10x."""
    torch.manual_seed(11)
    full = GeometricTriadAttentionPool(8, 8, 8, 8, score_init_scale=1.0)
    torch.manual_seed(11)
    shrunk = GeometricTriadAttentionPool(8, 8, 8, 8, score_init_scale=0.1)
    ratio = (full.W_q.weight.detach().norm() / shrunk.W_q.weight.detach().norm())
    assert abs(float(ratio) - 10.0) < 1e-3


def test_learn_scale_param_is_live():
    """learn_scale adds a log_scale parameter that receives gradient; off => none."""
    off = GeometricTriadAttentionPool(8, 8, 8, 8)
    assert not hasattr(off, "log_scale")
    pool, h_node, h_triad, inc_v, inc_t = _toy(channel="both", seed=12)
    on = GeometricTriadAttentionPool(8, 8, 8, 8, learn_scale=True,
                                     score_init_scale=1.0)
    on(h_node, h_triad, inc_v, inc_t).pow(2).sum().backward()
    assert on.log_scale.grad is not None and on.log_scale.grad.abs() >= 0.0


def test_sign_aware_requires_inc_balance():
    pool, h_node, h_triad, inc_v, inc_t = _toy(channel="both", seed=13)
    pool.sign_aware = True
    with pytest.raises(ValueError):
        pool(h_node, h_triad, inc_v, inc_t)            # inc_balance missing


def test_sign_aware_weight_is_balance_times_relevance():
    """w = b_t · σ(s̃): pins the woken formula against an explicit reference."""
    pool = GeometricTriadAttentionPool(8, 8, 8, 8, sign_aware=True,
                                       score_init_scale=1.0)
    _, h_node, h_triad, inc_v, inc_t = _toy(seed=14)
    b = _balance(inc_t, seed=14)
    q = pool.W_q(h_node)[inc_v]
    k = pool.W_k(h_triad)[inc_t]
    ref = b * torch.sigmoid(pool._raw_score(q, k))
    assert torch.allclose(pool._weights(q, k, b), ref)


def test_sign_aware_flipping_balance_flips_pool():
    """|w| is invariant to b_t's sign, so negating every b_t negates the pool."""
    pool = GeometricTriadAttentionPool(8, 8, 8, 8, sign_aware=True,
                                       score_init_scale=1.0)
    _, h_node, h_triad, inc_v, inc_t = _toy(seed=15)
    b = _balance(inc_t, seed=15)
    pos = pool(h_node, h_triad, inc_v, inc_t, b)
    neg = pool(h_node, h_triad, inc_v, inc_t, -b)
    assert torch.allclose(pos, -neg, atol=1e-6)


def test_summarise_gate_woken_pool_has_live_weights():
    """The woken pool's weights are alive at init (σ(0)=0.5), unlike the legacy
    dead-score pool (the diagnostic's w_frac_dead=1.0)."""
    pool = GeometricTriadAttentionPool(8, 8, 8, 8, sign_aware=True,
                                       score_init_scale=1.0)
    _, h_node, h_triad, inc_v, inc_t = _toy(seed=16)
    b = _balance(inc_t, seed=16)
    s = summarise_gate(pool, h_node, h_triad, inc_v, inc_t, b)
    assert s["w_abs_mean"] > 0.3            # ≈ σ(small) ≈ 0.5, not ≈ 0
    assert s["w_frac_dead"] == 0.0
