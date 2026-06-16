"""Tests for the geometric (quaternion + Clifford) triad attention pool.

Plan: docs/plans/2026-06-17-geometric-attention-head/.
Run: pytest -p no:randomly signedkan_wip/tests/test_geometric_triad_attention.py
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.core.geometric_triad_attention import (
    GeometricTriadAttentionPool,
    build_vertex_triad_pairs,
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
