"""Tests for the 2026-05-16-evening HyMeYOLO Stage B levers.

Two new backbones + a learnable activation primitive:

* :class:`ResNetTinyBackbone` — residual-block backbone, drop-in
  replacement for :class:`TinyBackbone`.
* :class:`CatmullRomActivation` — per-channel learnable univariate
  function (the HSiKAN basis-function primitive). Replaces ReLU in
  :class:`HSiKANConvBackbone`.
* :class:`HSiKANConvBackbone` — ResNet-tiny topology + CR activations.

Tests cover: shape contract, param-count expectations, gradient
flow, dispatch by name, and the CR-activation identity-init
property (the function is `φ(x) = x` at init, so the model at
init behaves like a no-nonlinearity skip).
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.vision.hymeyolo_backbones import (
    CatmullRomActivation,
    HSiKANConvBackbone,
    ResNetTinyBackbone,
    build_backbone,
)


# ─── CatmullRomActivation ─────────────────────────────────────────────


def test_cr_activation_output_shape() -> None:
    a = CatmullRomActivation(num_channels=8)
    x = torch.randn(2, 8, 4, 4)
    y = a(x)
    assert y.shape == x.shape


def test_cr_activation_param_count() -> None:
    """G knots × C channels = total learnable params."""
    a = CatmullRomActivation(num_channels=8, n_knots=8)
    assert sum(p.numel() for p in a.parameters()) == 8 * 8


def test_cr_activation_init_is_near_identity() -> None:
    """Initialised control points are exactly the knot x-positions,
    so φ(x) = x for x in the knot grid. Test on inputs at the knot
    positions."""
    a = CatmullRomActivation(num_channels=4, n_knots=8,
                              x_range=(-3.0, 3.0))
    # Inputs at the knots.
    x_at_knot = a.theta.view(1, 1, 1, -1).expand(2, 4, 1, 8).contiguous()
    y = a(x_at_knot)
    # At knot positions, CR interpolation returns the control point
    # exactly. With identity init, that equals the knot x-value.
    assert torch.allclose(y, x_at_knot, atol=1e-5), (
        f"CR at knots not identity at init: max diff "
        f"{(y - x_at_knot).abs().max().item()}"
    )


def test_cr_activation_outside_domain_clamps() -> None:
    """Inputs beyond [x_min, x_max] are clamped before interpolation,
    so the output is bounded by the endpoint control-point values."""
    a = CatmullRomActivation(num_channels=4, n_knots=8,
                              x_range=(-2.0, 2.0))
    x_extreme = torch.full((1, 4, 1, 1), 100.0)
    y = a(x_extreme)
    # At init, control points span -2..2, so y should ≈ +2 (the
    # upper-end identity value).
    assert torch.allclose(y, torch.full_like(y, 2.0), atol=1e-3)


def test_cr_activation_rejects_wrong_channels() -> None:
    a = CatmullRomActivation(num_channels=8)
    with pytest.raises(ValueError, match="channel mismatch"):
        a(torch.randn(2, 16, 4, 4))


def test_cr_activation_gradient_flows() -> None:
    a = CatmullRomActivation(num_channels=4, n_knots=8)
    x = torch.randn(2, 4, 4, 4, requires_grad=True)
    y = a(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Gradient flow into control points too.
    assert a.cp.grad is not None
    assert torch.isfinite(a.cp.grad).all()


# ─── ResNetTinyBackbone ───────────────────────────────────────────────


def test_resnet_backbone_output_shape_matches_tiny() -> None:
    """Same input shape contract: (B, 3, 64, 64) → (B, c_out, 8, 8)."""
    bb = ResNetTinyBackbone(c_in=3, c_out=32)
    bb.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y = bb(x)
    assert y.shape == (2, 32, 8, 8)


def test_resnet_backbone_parameter_count_in_range() -> None:
    """Design target: ~107k params at c_out=32. Sanity check the
    actual count lands in 80k-130k (allow for BN params variance)."""
    bb = ResNetTinyBackbone(c_in=3, c_out=32)
    n_params = sum(p.numel() for p in bb.parameters())
    assert 80_000 < n_params < 130_000, (
        f"ResNet-tiny params {n_params} outside design range "
        f"[80k, 130k]"
    )


def test_resnet_backbone_gradient_flow() -> None:
    bb = ResNetTinyBackbone()
    bb.train()
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    y = bb(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Some backbone params should have non-zero gradient.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in bb.parameters()
    )
    assert has_grad


# ─── HSiKANConvBackbone ───────────────────────────────────────────────


def test_hsikan_backbone_output_shape_matches_tiny() -> None:
    bb = HSiKANConvBackbone(c_in=3, c_out=32)
    bb.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y = bb(x)
    assert y.shape == (2, 32, 8, 8)


def test_hsikan_backbone_param_count_slightly_more_than_resnet() -> None:
    """HSiKAN backbone has the same convs/BN as ResNet-tiny plus the
    CR-activation control points per layer (replacing ReLU's zero
    params). The diff should be small (< 5k total)."""
    rn = ResNetTinyBackbone()
    hs = HSiKANConvBackbone()
    n_rn = sum(p.numel() for p in rn.parameters())
    n_hs = sum(p.numel() for p in hs.parameters())
    assert n_hs > n_rn, "HSiKAN should have more params than ResNet (CR ctrls)"
    delta = n_hs - n_rn
    assert delta < 5_000, (
        f"HSiKAN CR-activation param overhead {delta} > 5k; check "
        f"that CR controls aren't unintentionally large"
    )


def test_hsikan_backbone_gradient_flow() -> None:
    bb = HSiKANConvBackbone()
    bb.train()
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    y = bb(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_hsikan_backbone_cr_activations_train_via_backprop() -> None:
    """At least one CR-activation control-point tensor must receive
    a non-zero gradient (otherwise the learnable-activation claim
    is vacuous)."""
    bb = HSiKANConvBackbone()
    bb.train()
    x = torch.randn(2, 3, 64, 64)
    bb(x).sum().backward()
    cr_params = [
        m.cp for m in bb.modules() if isinstance(m, CatmullRomActivation)
    ]
    assert len(cr_params) > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in cr_params
    )


# ─── Dispatch ─────────────────────────────────────────────────────────


def test_build_backbone_dispatches_three_names() -> None:
    """The build_backbone helper supports the three canonical names."""
    from signedkan_wip.src.vision.hymeyolo_q_smoke import TinyBackbone
    assert isinstance(build_backbone("tiny"), TinyBackbone)
    assert isinstance(build_backbone("resnet"), ResNetTinyBackbone)
    assert isinstance(build_backbone("hsikan"), HSiKANConvBackbone)


def test_build_backbone_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown backbone"):
        build_backbone("convnext")


# ─── RicciHyMeYOLOMulti integration ───────────────────────────────────


def test_ricci_hymeyolo_dispatches_on_backbone_flag() -> None:
    """The backbone= kwarg is honoured."""
    from signedkan_wip.src.vision.hymeyolo_circles_ricci import (
        RicciHyMeYOLOMulti,
    )
    from signedkan_wip.src.vision.hymeyolo_q_smoke import TinyBackbone

    m_tiny = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, d_hidden=32,
        ricci_modulation=True, backbone="tiny",
    )
    m_resnet = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, d_hidden=32,
        ricci_modulation=True, backbone="resnet",
    )
    m_hsikan = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, d_hidden=32,
        ricci_modulation=True, backbone="hsikan",
    )
    assert isinstance(m_tiny.backbone, TinyBackbone)
    assert isinstance(m_resnet.backbone, ResNetTinyBackbone)
    assert isinstance(m_hsikan.backbone, HSiKANConvBackbone)


def test_ricci_hymeyolo_default_backbone_is_tiny() -> None:
    """Default backbone="tiny" preserves pre-Stage-B behaviour."""
    from signedkan_wip.src.vision.hymeyolo_circles_ricci import (
        RicciHyMeYOLOMulti,
    )
    from signedkan_wip.src.vision.hymeyolo_q_smoke import TinyBackbone

    m = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, d_hidden=32,
        ricci_modulation=True,
    )
    assert isinstance(m.backbone, TinyBackbone)
    assert m.backbone_kind == "tiny"


def test_ricci_hymeyolo_resnet_forward_finite() -> None:
    from signedkan_wip.src.vision.hymeyolo_circles_ricci import (
        RicciHyMeYOLOMulti,
    )
    torch.manual_seed(0)
    m = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, d_hidden=32,
        ricci_modulation=True, backbone="resnet",
    )
    m.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = m(x)
    for k in ("box_corners", "box_cls", "circle_corners", "circle_cls"):
        assert torch.isfinite(out[k]).all(), f"non-finite {k}"


def test_ricci_hymeyolo_hsikan_forward_finite() -> None:
    from signedkan_wip.src.vision.hymeyolo_circles_ricci import (
        RicciHyMeYOLOMulti,
    )
    torch.manual_seed(0)
    m = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, d_hidden=32,
        ricci_modulation=True, backbone="hsikan",
    )
    m.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = m(x)
    for k in ("box_corners", "box_cls", "circle_corners", "circle_cls"):
        assert torch.isfinite(out[k]).all(), f"non-finite {k}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
