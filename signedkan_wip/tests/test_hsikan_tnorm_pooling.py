"""Tests for HSiKAN t-norm pooling (2026-05-30, Kóczy fuzzy-signature
work). Verifies:

1. The default ``pooling="sum"`` path is unchanged (parity vs current
   behavior — covered by the existing test suite which assumes sum).
2. ``build_rf_edge_members`` is consistent with ``build_rf_position_incidence``
   (the same RF tiling, expressed two ways).
3. Each t-norm satisfies the classical fuzzy-logic boundary conditions:
   - T(1, 1, ..., 1) = 1
   - T(0, a_1, ..., a_n) = 0
   - T is non-increasing in any argument (monotonic)
4. Smoke: HSiKAN with each pooling mode builds, forwards, and trains
   one step without NaN.

Boundary tests use the t-norm functions in isolation (extracted from the
forward path) so we can assert classical fuzzy-logic identities. The
forward integration is covered by the smoke tests at the end.
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.vision.hsikan_vision import (
    HSiKANVisionClassifier,
    SignedBranchConv,
    build_rf_edge_members,
    build_rf_incidence,
    build_rf_position_incidence,
)


# ─── edge_members consistency with the existing incidence ──────────


def test_edge_members_matches_incidence():
    """Each (e, k) pair must point to a pixel that is in RF e per the
    binary incidence."""
    H, W = 28, 28
    for k, s in [(5, 2), (8, 4), (12, 4)]:
        inc, n_e = build_rf_incidence(H, W, kernel=k, stride=s)
        em, n_e2 = build_rf_edge_members(H, W, kernel=k, stride=s)
        assert n_e == n_e2
        assert em.shape == (n_e, k * k)
        for e in range(n_e):
            for kk in range(k * k):
                v = int(em[e, kk])
                assert inc[v, e] == 1.0


def test_edge_members_matches_position_incidence():
    """em[e, k] == v iff pos_inc[v, e] == k."""
    H, W = 28, 28
    for k, s in [(5, 2), (8, 4)]:
        _inc, pos_inc, n_e = build_rf_position_incidence(H, W, kernel=k, stride=s)
        em, _ = build_rf_edge_members(H, W, kernel=k, stride=s)
        for e in range(n_e):
            for kk in range(k * k):
                v = int(em[e, kk])
                assert int(pos_inc[v, e]) == kk


# ─── Classical t-norm boundary conditions ───────────────────────────


def _gödel(x):     return x.amin(dim=-1)
def _product(x):   return torch.exp(torch.log(x.clamp(min=1e-7)).sum(dim=-1))
def _luk(x):
    K = x.shape[-1]
    return (x.sum(dim=-1) - (K - 1)).clamp(min=0.0)


@pytest.mark.parametrize(
    "T,name",
    [(_gödel, "min"), (_product, "product"), (_luk, "lukasiewicz")],
)
def test_tnorm_boundary_T_of_ones_is_one(T, name):
    """T(1, 1, ..., 1) = 1 for any t-norm (identity element)."""
    K = 25
    ones = torch.ones(1, K)
    out = T(ones)
    assert torch.allclose(out, torch.tensor([1.0]), atol=1e-5), (
        f"{name}: T(1,...,1) = {out.item():.6f}, expected 1.0"
    )


@pytest.mark.parametrize(
    "T,name",
    [(_gödel, "min"), (_product, "product"), (_luk, "lukasiewicz")],
)
def test_tnorm_boundary_T_with_zero_is_zero(T, name):
    """T(0, a_1, ..., a_n) = 0 for any t-norm (annihilator)."""
    K = 25
    x = torch.rand(1, K)
    x[0, 0] = 0.0
    out = T(x)
    assert torch.allclose(out, torch.tensor([0.0]), atol=1e-5), (
        f"{name}: T(0, ...) = {out.item():.6f}, expected 0.0"
    )


@pytest.mark.parametrize(
    "T,name",
    [(_gödel, "min"), (_product, "product"), (_luk, "lukasiewicz")],
)
def test_tnorm_monotonicity(T, name):
    """T is non-decreasing in each argument: x ≤ y elementwise ⇒ T(x) ≤ T(y)."""
    K = 25
    torch.manual_seed(0)
    x = torch.rand(1, K)
    y = x + torch.rand_like(x) * (1.0 - x)  # y >= x, still in [0, 1]
    out_x = T(x)
    out_y = T(y)
    assert (out_y >= out_x - 1e-5).all(), (
        f"{name}: monotonicity violated, T(x)={out_x.item():.4f}, T(y)={out_y.item():.4f}"
    )


def test_tnorm_ordering():
    """Classical: Łukasiewicz ≤ product ≤ Gödel (well-known fuzzy
    ordering — Łukasiewicz is the strictest, Gödel the most permissive)."""
    torch.manual_seed(1)
    x = torch.rand(20, 8)  # batch of 8-arity inputs in [0, 1]
    god = _gödel(x)
    prod = _product(x)
    luk = _luk(x)
    assert (luk <= prod + 1e-5).all(), "Łukasiewicz must be ≤ product"
    assert (prod <= god + 1e-5).all(), "product must be ≤ Gödel"


# ─── SignedBranchConv pooling shape/error tests ─────────────────────


def test_signedbranchconv_pooling_default_is_sum():
    c = SignedBranchConv(d_in=4, d_out=8, n_edges=144)
    assert c.pooling == "sum"


def test_signedbranchconv_pooling_invalid_raises():
    with pytest.raises(ValueError, match="pooling"):
        SignedBranchConv(d_in=4, d_out=8, n_edges=144, pooling="bogus")


def test_signedbranchconv_tnorm_requires_edge_members():
    c = SignedBranchConv(d_in=4, d_out=8, n_edges=144, kernel=5,
                         pooling="min")
    x = torch.randn(2, 28 * 28, 4)
    inc = torch.zeros(28 * 28, 144)
    D_v = torch.ones(28 * 28)
    D_e = torch.ones(144)
    with pytest.raises(ValueError, match="edge_members"):
        c.forward(x, inc, D_v, D_e, pos_inc=None, edge_members=None)


# ─── End-to-end smoke for each pooling mode ────────────────────────


@pytest.mark.parametrize("pooling", ["sum", "min", "product", "lukasiewicz"])
def test_hsikan_classifier_pooling_smoke(pooling):
    """Each pooling mode builds, forwards, and trains one step without NaN."""
    torch.manual_seed(0)
    model = HSiKANVisionClassifier(
        H=28, W=28, n_classes=10, hidden=8, n_layers=2, pooling=pooling,
    )
    x = torch.randn(2, 1, 28, 28)
    y = torch.randint(0, 10, (2,))
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all(), f"{pooling}: non-finite forward output"
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(out, y)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
