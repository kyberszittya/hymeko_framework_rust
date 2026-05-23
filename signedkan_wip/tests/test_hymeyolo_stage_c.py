"""Tests for HyMeYOLO Stage C — FPN multi-scale heads.

Coverage (per the Stage C plan §5 test strategy):
  - FPN2Level output shapes.
  - ResNet/HSiKAN backbones expose multi_scale_features correctly.
  - RicciHyMeYOLOMulti with fpn="2level" forwards to finite outputs.
  - Gradient flows through the FPN's lateral convs.
  - fpn="none" produces no FPN-related state_dict keys (Stage B
    byte-identical pin).
  - FPN parameter count in the 8-15k expected range at c_out=32.
  - fpn="2level" rejects backbone="tiny" (no /4 tap).

Plan: docs/plans/2026-05-16-hymeyolo-stage-c-fpn/.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from signedkan_wip.src.vision.hymeyolo_fpn import FPN2Level
from signedkan_wip.src.vision.hymeyolo_backbones import (
    ResNetTinyBackbone,
    HSiKANConvBackbone,
)
from signedkan_wip.src.vision.hymeyolo_circles_ricci import RicciHyMeYOLOMulti


# ─── FPN2Level shape / param tests ──────────────────────────────────


def test_fpn2level_output_shapes():
    fpn = FPN2Level(c_in_4=32, c_in_8=32, c_out=32)
    p_in_4 = torch.randn(2, 32, 16, 16)
    p_in_8 = torch.randn(2, 32, 8, 8)
    p_4, p_8 = fpn(p_in_4, p_in_8)
    assert p_4.shape == (2, 32, 16, 16), p_4.shape
    assert p_8.shape == (2, 32, 8, 8), p_8.shape


def test_fpn2level_output_finite():
    torch.manual_seed(0)
    fpn = FPN2Level(c_in_4=32, c_in_8=32, c_out=32)
    p_in_4 = torch.randn(2, 32, 16, 16)
    p_in_8 = torch.randn(2, 32, 8, 8)
    p_4, p_8 = fpn(p_in_4, p_in_8)
    assert torch.isfinite(p_4).all()
    assert torch.isfinite(p_8).all()


def test_fpn2level_parameter_count_in_expected_range():
    """The Stage C plan estimates FPN at ~11k params for c_out=32."""
    fpn = FPN2Level(c_in_4=32, c_in_8=32, c_out=32)
    n = sum(p.numel() for p in fpn.parameters())
    assert 8_000 <= n <= 15_000, (
        f"FPN2Level has {n} params at c_out=32; expected 8k–15k"
    )


def test_fpn2level_gradient_flows_to_laterals():
    torch.manual_seed(0)
    fpn = FPN2Level(c_in_4=32, c_in_8=32, c_out=32)
    p_in_4 = torch.randn(2, 32, 16, 16, requires_grad=True)
    p_in_8 = torch.randn(2, 32, 8, 8, requires_grad=True)
    p_4, p_8 = fpn(p_in_4, p_in_8)
    loss = p_4.pow(2).mean() + p_8.pow(2).mean()
    loss.backward()
    # Gradients should reach both lateral conv weights.
    assert fpn.lateral_p4.weight.grad is not None
    assert fpn.lateral_p4.weight.grad.abs().sum() > 0
    assert torch.isfinite(fpn.lateral_p4.weight.grad).all()
    assert fpn.lateral_p8.weight.grad is not None
    assert fpn.lateral_p8.weight.grad.abs().sum() > 0
    assert torch.isfinite(fpn.lateral_p8.weight.grad).all()


# ─── Backbone multi_scale_features tests ─────────────────────────────


def test_resnet_backbone_multi_scale_returns_two_maps():
    bb = ResNetTinyBackbone(c_in=3, c_out=32)
    x = torch.randn(2, 3, 64, 64)
    p_4, p_8 = bb.multi_scale_features(x)
    assert p_4.shape == (2, 32, 16, 16), p_4.shape
    assert p_8.shape == (2, 32, 8, 8), p_8.shape


def test_hsikan_backbone_multi_scale_returns_two_maps():
    bb = HSiKANConvBackbone(c_in=3, c_out=32)
    x = torch.randn(2, 3, 64, 64)
    p_4, p_8 = bb.multi_scale_features(x)
    assert p_4.shape == (2, 32, 16, 16), p_4.shape
    assert p_8.shape == (2, 32, 8, 8), p_8.shape


def test_resnet_ms_matches_single_scale_at_p8():
    """Calling multi_scale_features then taking P_8 must equal
    calling backbone(x) (state-dict-identical path)."""
    torch.manual_seed(0)
    bb = ResNetTinyBackbone(c_in=3, c_out=32).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        single = bb(x)
        _, p_8 = bb.multi_scale_features(x)
    assert torch.allclose(single, p_8, atol=1e-6), \
        "multi_scale_features /8 must equal backbone(x) bit-for-bit"


# ─── RicciHyMeYOLOMulti integration tests ────────────────────────────


def test_ricci_hymeyolo_fpn_none_no_fpn_state_dict_keys():
    """fpn='none' (default) must not introduce FPN state-dict keys.

    Stage B byte-identical pin: the b_resnet checkpoint loaded into
    an fpn='none' model must have zero key mismatches.
    """
    model = RicciHyMeYOLOMulti(backbone="resnet", fpn="none")
    keys = set(model.state_dict().keys())
    fpn_keys = {k for k in keys if k.startswith("fpn.") or k.startswith("ms_proj.")}
    assert fpn_keys == set(), (
        f"fpn='none' leaked FPN state_dict keys: {sorted(fpn_keys)}"
    )


def test_ricci_hymeyolo_fpn_2level_adds_fpn_keys():
    model = RicciHyMeYOLOMulti(backbone="resnet", fpn="2level")
    keys = set(model.state_dict().keys())
    assert any(k.startswith("fpn.") for k in keys), keys
    assert any(k.startswith("ms_proj.") for k in keys), keys


def test_ricci_hymeyolo_fpn_2level_forward_finite():
    torch.manual_seed(0)
    model = RicciHyMeYOLOMulti(backbone="resnet", fpn="2level").eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    # Stage B b_resnet output contract preserved.
    assert out["box_corners"].shape == (2, 4, 4, 2)
    assert out["box_cls"].shape == (2, 4, 11)
    assert out["circle_corners"].shape == (2, 2, 8, 2)
    assert out["circle_cls"].shape == (2, 2, 11)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k}: non-finite"


def test_ricci_hymeyolo_fpn_2level_gradient_flows_to_fpn():
    torch.manual_seed(0)
    model = RicciHyMeYOLOMulti(backbone="resnet", fpn="2level")
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    loss = (out["box_corners"].pow(2).mean()
            + out["circle_corners"].pow(2).mean()
            + out["box_cls"].pow(2).mean())
    loss.backward()
    # The FPN's lateral_p4 weights must receive gradient.
    g = model.fpn.lateral_p4.weight.grad
    assert g is not None, "FPN lateral_p4 saw no gradient"
    assert g.abs().sum() > 0, "FPN lateral_p4 gradient is all zero"
    assert torch.isfinite(g).all(), "FPN lateral_p4 gradient is NaN/inf"
    # ms_proj must also receive gradient.
    g = model.ms_proj.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_ricci_hymeyolo_fpn_rejects_tinybackbone():
    """TinyBackbone has only 3 conv stages; no /4 multi-scale tap.
    Constructing with fpn='2level' + backbone='tiny' must raise."""
    with pytest.raises(ValueError, match="multi_scale_features"):
        RicciHyMeYOLOMulti(backbone="tiny", fpn="2level")


def test_ricci_hymeyolo_fpn_unknown_raises():
    with pytest.raises(ValueError, match="unknown fpn"):
        RicciHyMeYOLOMulti(backbone="resnet", fpn="bogus")


def test_ricci_hymeyolo_fpn_none_equals_stage_b_signature():
    """fpn='none' has no FPN module; self.fpn is None."""
    model = RicciHyMeYOLOMulti(backbone="resnet", fpn="none")
    assert model.fpn is None
    assert model.ms_proj is None
    assert model.fpn_kind == "none"


def test_ricci_hymeyolo_fpn_2level_works_with_hsikan_backbone():
    torch.manual_seed(0)
    model = RicciHyMeYOLOMulti(backbone="hsikan", fpn="2level").eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out["box_corners"].shape == (2, 4, 4, 2)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k}: non-finite"


def test_ricci_hymeyolo_fpn_total_param_count():
    """Sanity: FPN delta over Stage B b_resnet should be ~10-15k.

    Measured 2026-05-17: b_resnet total=119,092; c_fpn total=132,564;
    FPN delta=13,472 (FPN 11,392 + ms_proj 2,080). The original Stage
    B plan estimated ~1.1M total params; that estimate was wrong (it
    likely confused the model with a much-larger aggregator
    configuration). The actual count is what we pin against.
    """
    m_b = RicciHyMeYOLOMulti(backbone="resnet", fpn="none")
    m_c = RicciHyMeYOLOMulti(backbone="resnet", fpn="2level")
    nb = sum(p.numel() for p in m_b.parameters())
    nc = sum(p.numel() for p in m_c.parameters())
    delta = nc - nb
    assert 10_000 <= delta <= 16_000, (
        f"FPN delta = {delta} params; expected 10k-16k. "
        f"b_resnet total={nb}, c_fpn total={nc}"
    )
