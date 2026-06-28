"""Spline library tests: Catmull-Rom + B-spline activations, the `make_activation` Strategy, and **parity with
`signedkan_wip`** for both evaluators (the gate that lets `signedkan_wip` import them from here — single source).
"""
from __future__ import annotations

import pytest
import torch

from signed_kan.splines import (
    BSplineActivation,
    CatmullRomActivation,
    ChebyshevCRActivation,
    catmull_rom,
    cox_de_boor,
    make_activation,
    make_uniform_knots,
)


# ── CR-Chebyshev (the Chebyshev-control-point view on Catmull-Rom) ────────────
def test_cr_chebyshev_shapes_cached_basis_and_grad() -> None:
    act = ChebyshevCRActivation(8, grid=12, k=5)
    x = torch.randn(4, 3, 8) * 5.0                       # large inputs clamp into [-1, 1]
    y = act(x)
    assert y.shape == x.shape                            # (..., C) -> (..., C)
    assert act.control_points().shape == (8, 12)         # band-limited points P = coef @ T_knots^T
    assert sum(p.numel() for p in act.parameters()) == 8 * 5   # k coeffs/channel, NOT grid
    # T_knots is the cached Chebyshev-at-knots constant (not a parameter), grid x k.
    assert act.t_knots.shape == (12, 5) and not act.t_knots.requires_grad
    y.sum().backward()
    assert act.coef.grad is not None and torch.isfinite(act.coef.grad).all()


def test_cr_chebyshev_deploy_bridge_approximates_train() -> None:
    """The deploy path (direct Chebyshev) approximates the train path (CR interp of the Chebyshev points)."""
    act = ChebyshevCRActivation(16, grid=16, k=5)
    x = torch.linspace(-1, 1, 200).view(-1, 1).expand(-1, 16)
    train, deploy = act(x), act.chebyshev_forward(x)
    assert deploy.shape == train.shape
    assert (train - deploy).abs().mean() < 0.05          # close (interpolation error), the bridge tolerance


def test_make_activation_cr_cheby() -> None:
    act = make_activation("cr_cheby", 6)
    assert isinstance(act, ChebyshevCRActivation)
    assert act(torch.randn(2, 6)).shape == (2, 6)


# ── B-spline activation ──────────────────────────────────────────────────────
def test_bspline_shape_zero_coef_and_grad() -> None:
    act = BSplineActivation(8, grid=5, k=3)
    x = torch.randn(4, 3, 8) * 5.0          # large inputs clamp into [-1, 1]
    assert act(x).shape == (4, 3, 8) and torch.isfinite(act(x)).all()
    with torch.no_grad():
        act.coef.zero_()
    assert act(x).abs().max() < 1e-7        # zero coefficients → identically zero
    act2 = BSplineActivation(4, grid=5)
    xv = torch.randn(16, 4, requires_grad=True)
    act2(xv).sum().backward()
    assert act2.coef.grad is not None and act2.coef.grad.abs().sum() > 0 and xv.grad is not None
    with pytest.raises(ValueError):
        BSplineActivation(0)


def test_cox_de_boor_partition_of_unity() -> None:
    knots = make_uniform_knots(5, 3)
    basis = cox_de_boor(torch.linspace(-0.95, 0.95, 200), knots, 3)
    assert (basis.sum(dim=-1) - 1.0).abs().max() < 1e-5    # B-spline basis sums to 1 in the active region


# ── activation Strategy ──────────────────────────────────────────────────────
def test_make_activation_all_kinds() -> None:
    for kind in ("cr", "bspline", "relu", "tanh"):
        out = make_activation(kind, 8)(torch.randn(2, 8))
        assert out.shape == (2, 8) and torch.isfinite(out).all()
    assert isinstance(make_activation("cr", 8), CatmullRomActivation)
    assert isinstance(make_activation("bspline", 8), BSplineActivation)
    with pytest.raises(ValueError, match="activation must be"):
        make_activation("nope", 8)


# ── parity with signedkan_wip (the dedup gate) ───────────────────────────────
def test_cox_de_boor_parity_with_signedkan() -> None:
    """signed_kan's B-spline basis must match signedkan_wip's bit-closely — the precondition for signedkan_wip to
    import it from here (single source) without moving the OTC benchmark."""
    try:
        from signedkan_wip.src.core.splines import cox_de_boor as ref_cdb
        from signedkan_wip.src.core.splines import make_uniform_knots as ref_knots
    except Exception:
        pytest.skip("signedkan_wip not importable")
    x = torch.linspace(-1.0, 1.0, 128)
    assert torch.allclose(cox_de_boor(x, make_uniform_knots(5, 3), 3), ref_cdb(x, ref_knots(5, 3), 3), atol=1e-7)


def test_catmull_rom_parity_with_signedkan() -> None:
    try:
        from signedkan_wip.src.core.splines import _catmull_rom_eval
    except Exception:
        pytest.skip("signedkan_wip not importable")
    coef = torch.randn(6, 5)
    x = torch.linspace(-1.5, 1.5, 64).unsqueeze(-1).expand(64, 6).contiguous()
    assert torch.allclose(catmull_rom(coef, x, 5), _catmull_rom_eval(coef, x, 5), atol=1e-6)
