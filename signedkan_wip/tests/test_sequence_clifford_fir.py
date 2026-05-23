"""Tests for the Clifford-algebra primitives and CliffordFIR module.

Coverage (per plan §7):
  - geometric product: associativity, distributivity, scalar identity,
    explicit basis-element multiplication checks
  - multivector norm
  - CliffordFIR: shape, causality, gradient flow, parameter count

Plan: docs/plans/2026-05-17-sequential-hsikan-clifford-fir/.
"""
from __future__ import annotations

import torch

from signedkan_wip.src.sequence.clifford import (
    CL2_DIM,
    geometric_product,
    multivector_norm,
    scalar_multivector,
)
from signedkan_wip.src.sequence.clifford_fir import CliffordFIR


# ─── Geometric product primitive tests ───────────────────────────────


def test_geometric_product_scalar_identity():
    """The scalar 1 multivector (1, 0, 0, 0) acts as identity."""
    torch.manual_seed(0)
    one = torch.tensor([1.0, 0.0, 0.0, 0.0])
    x = torch.randn(7, CL2_DIM)
    y_left = geometric_product(one.expand_as(x), x)
    y_right = geometric_product(x, one.expand_as(x))
    assert torch.allclose(y_left, x, atol=1e-6)
    assert torch.allclose(y_right, x, atol=1e-6)


def test_geometric_product_basis_e1_squared_is_one():
    """e1 * e1 = 1 (scalar)."""
    e1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    out = geometric_product(e1, e1)
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(out, expected, atol=1e-6), out


def test_geometric_product_basis_e2_squared_is_one():
    e2 = torch.tensor([0.0, 0.0, 1.0, 0.0])
    out = geometric_product(e2, e2)
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(out, expected, atol=1e-6), out


def test_geometric_product_e12_squared_is_minus_one():
    """e12 * e12 = -1 (the bivector squares to -1, as in C/quaternions)."""
    e12 = torch.tensor([0.0, 0.0, 0.0, 1.0])
    out = geometric_product(e12, e12)
    expected = torch.tensor([-1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(out, expected, atol=1e-6), out


def test_geometric_product_e1_times_e2_is_e12():
    e1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 0.0, 1.0, 0.0])
    out = geometric_product(e1, e2)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(out, expected, atol=1e-6), out


def test_geometric_product_anticommutes_for_basis_vectors():
    """e1 * e2 = - (e2 * e1)."""
    e1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 0.0, 1.0, 0.0])
    ab = geometric_product(e1, e2)
    ba = geometric_product(e2, e1)
    assert torch.allclose(ab, -ba, atol=1e-6)


def test_geometric_product_associative():
    """(a*b)*c == a*(b*c) for random multivectors."""
    torch.manual_seed(0)
    a = torch.randn(5, CL2_DIM)
    b = torch.randn(5, CL2_DIM)
    c = torch.randn(5, CL2_DIM)
    left = geometric_product(geometric_product(a, b), c)
    right = geometric_product(a, geometric_product(b, c))
    assert torch.allclose(left, right, atol=1e-5)


def test_geometric_product_distributive_left():
    """a*(b+c) == a*b + a*c."""
    torch.manual_seed(0)
    a = torch.randn(4, CL2_DIM)
    b = torch.randn(4, CL2_DIM)
    c = torch.randn(4, CL2_DIM)
    lhs = geometric_product(a, b + c)
    rhs = geometric_product(a, b) + geometric_product(a, c)
    assert torch.allclose(lhs, rhs, atol=1e-6)


def test_geometric_product_distributive_right():
    """(a+b)*c == a*c + b*c."""
    torch.manual_seed(0)
    a = torch.randn(4, CL2_DIM)
    b = torch.randn(4, CL2_DIM)
    c = torch.randn(4, CL2_DIM)
    lhs = geometric_product(a + b, c)
    rhs = geometric_product(a, c) + geometric_product(b, c)
    assert torch.allclose(lhs, rhs, atol=1e-6)


def test_geometric_product_broadcasts():
    """Leading dims must broadcast — (B, L, 4) and (4,)."""
    torch.manual_seed(0)
    a = torch.randn(3, 7, CL2_DIM)
    b = torch.randn(CL2_DIM)
    out = geometric_product(a, b)
    assert out.shape == (3, 7, CL2_DIM)


def test_geometric_product_bad_shape_raises():
    import pytest
    a = torch.randn(3, 5)  # trailing dim is 5, not 4
    b = torch.randn(3, 5)
    with pytest.raises(ValueError, match="trailing dim"):
        geometric_product(a, b)


def test_multivector_norm_positive_definite():
    torch.manual_seed(0)
    x = torch.randn(10, CL2_DIM)
    n = multivector_norm(x)
    assert n.shape == (10,)
    assert (n >= 0).all()
    # The zero multivector has norm zero.
    assert multivector_norm(torch.zeros(CL2_DIM)).item() == 0.0


def test_scalar_multivector_lifts_correctly():
    s = torch.tensor([1.5, -2.0, 0.0])
    m = scalar_multivector(s)
    assert m.shape == (3, CL2_DIM)
    assert torch.allclose(m[:, 0], s)
    assert (m[:, 1:] == 0).all()


# ─── CliffordFIR tests ───────────────────────────────────────────────


def test_clifford_fir_default_shape():
    """K=4 single-channel filter on (B=2, L=16, 4) input."""
    fir = CliffordFIR(K=4)
    x = torch.randn(2, 16, CL2_DIM)
    y = fir(x)
    assert y.shape == (2, 16, CL2_DIM)


def test_clifford_fir_finite_at_init():
    torch.manual_seed(0)
    fir = CliffordFIR(K=4)
    x = torch.randn(2, 16, CL2_DIM)
    y = fir(x)
    assert torch.isfinite(y).all()


def test_clifford_fir_param_count():
    """K=4, c_in=c_out=1: 4*1*1*4 = 16 scalar params."""
    fir = CliffordFIR(K=4)
    n = sum(p.numel() for p in fir.parameters())
    assert n == 16


def test_clifford_fir_causal():
    """Output at position t depends only on inputs at positions ≤ t.

    Build (B=1, L=8, 4) input that is zero everywhere except a single
    spike at position t=3. The output should be zero at positions
    0..2 (no input has reached them yet) and non-zero at 3, 4, 5, 6
    (for K=4 the spike's influence persists for K positions).
    """
    torch.manual_seed(0)
    fir = CliffordFIR(K=4)
    x = torch.zeros(1, 8, CL2_DIM)
    x[0, 3, 0] = 1.0  # scalar spike at position 3
    with torch.no_grad():
        y = fir(x)
    # Positions 0,1,2 must be exactly zero (causality).
    assert torch.allclose(y[0, 0:3], torch.zeros(3, CL2_DIM), atol=1e-6)
    # Position 3 must be non-zero (the spike has just arrived).
    assert y[0, 3].abs().sum() > 0


def test_clifford_fir_gradient_flows_to_taps():
    torch.manual_seed(0)
    fir = CliffordFIR(K=4)
    x = torch.randn(2, 16, CL2_DIM)
    y = fir(x)
    loss = y.pow(2).mean()
    loss.backward()
    g = fir.taps.grad
    assert g is not None
    assert g.abs().sum() > 0
    assert torch.isfinite(g).all()


def test_clifford_fir_identity_taps_passes_input_through():
    """A filter with identity-only tap (b_0 = scalar 1, b_{k>0} = 0)
    should reproduce its input bit-for-bit at every position."""
    fir = CliffordFIR(K=3)
    with torch.no_grad():
        fir.taps.zero_()
        # b_0 = scalar 1 multivector
        fir.taps[0, 0, 0, 0] = 1.0
    x = torch.randn(2, 8, CL2_DIM)
    y = fir(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_clifford_fir_multichannel_shapes():
    """(B, L, c_in, 4) in, (B, L, c_out, 4) out."""
    fir = CliffordFIR(K=3, c_in=2, c_out=3)
    x = torch.randn(2, 10, 2, CL2_DIM)
    y = fir(x)
    assert y.shape == (2, 10, 3, CL2_DIM)
    n = sum(p.numel() for p in fir.parameters())
    assert n == 3 * 2 * 3 * 4


def test_clifford_fir_rejects_wrong_c_in():
    import pytest
    fir = CliffordFIR(K=3, c_in=2, c_out=2)
    x = torch.randn(2, 10, 3, CL2_DIM)  # c_in=3 != layer's 2
    with pytest.raises(ValueError, match="c_in"):
        fir(x)


def test_clifford_fir_receptive_field():
    assert CliffordFIR(K=4).receptive_field() == 4
    assert CliffordFIR(K=8).receptive_field() == 8
