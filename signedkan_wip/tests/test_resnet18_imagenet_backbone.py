"""Regression tests for the Stage D-1 ResNet18-ImageNet backbone."""
from __future__ import annotations

import pytest
import torch


def test_backbone_builds_and_has_expected_shape():
    from signedkan_wip.src.vision.hymeyolo_backbones import build_backbone
    bb = build_backbone("resnet18_imagenet", c_in=3, c_out=32)
    x = torch.rand(2, 3, 224, 224)
    y = bb(x)
    assert y.shape == (2, 32, 28, 28), y.shape


def test_backbone_multi_scale_features():
    from signedkan_wip.src.vision.hymeyolo_backbones import build_backbone
    bb = build_backbone("resnet18_imagenet", c_in=3, c_out=32)
    x = torch.rand(2, 3, 224, 224)
    p4, p8 = bb.multi_scale_features(x)
    assert p4.shape == (2, 32, 56, 56), p4.shape
    assert p8.shape == (2, 32, 28, 28), p8.shape


def test_backbone_forward_backward_finite():
    from signedkan_wip.src.vision.hymeyolo_backbones import build_backbone
    bb = build_backbone("resnet18_imagenet", c_in=3, c_out=32)
    x = torch.rand(1, 3, 224, 224, requires_grad=True)
    y = bb(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_backbone_integrates_with_RicciHyMeYOLOMulti_voc_config():
    from signedkan_wip.src.vision.hymeyolo_circles_ricci import RicciHyMeYOLOMulti
    m = RicciHyMeYOLOMulti(
        n_box_queries=12, n_circle_queries=0,
        n_classes=20, d_hidden=32,
        ricci_modulation=True, ricci_scale=1.0,
        use_layernorm=False,
        backbone="resnet18_imagenet", fpn="2level",
    )
    x = torch.rand(2, 3, 224, 224)
    pred = m(x)
    assert "box_corners" in pred
    assert "box_cls" in pred
    assert pred["box_corners"].shape == (2, 12, 4, 2)
    assert pred["box_cls"].shape == (2, 12, 21)  # n_classes + 1 (no-object)


def test_rejects_non_rgb_input_channels():
    from signedkan_wip.src.vision.hymeyolo_backbones import ResNet18ImageNetBackbone
    with pytest.raises(ValueError, match="c_in=3"):
        ResNet18ImageNetBackbone(c_in=1)


def test_l3_backbone_deeper_scales_and_more_capacity():
    # Capacity lever (2026-06-10): the L3 variant carries layer3 (/16,
    # 256 ch), so multi_scale_features taps /8 and /16 (deeper than the
    # layer2 backbone's /4 + /8) and has materially more parameters.
    from signedkan_wip.src.vision.hymeyolo_backbones import build_backbone
    bb = build_backbone("resnet18_imagenet_l3", c_in=3, c_out=32)
    x = torch.rand(2, 3, 224, 224)
    assert bb(x).shape == (2, 32, 14, 14)               # stride 16
    p8, p16 = bb.multi_scale_features(x)
    assert p8.shape == (2, 32, 28, 28)                  # /8
    assert p16.shape == (2, 32, 14, 14)                 # /16
    shallow = build_backbone("resnet18_imagenet", c_in=3, c_out=32)
    n_l3 = sum(p.numel() for p in bb.parameters())
    n_l2 = sum(p.numel() for p in shallow.parameters())
    assert n_l3 > 2 * n_l2, (n_l3, n_l2)


def test_l3_backbone_drops_into_2level_fpn_unchanged():
    from signedkan_wip.src.vision.hymeyolo_circles_ricci import RicciHyMeYOLOMulti
    m = RicciHyMeYOLOMulti(
        n_box_queries=6, n_circle_queries=0, n_classes=20, d_hidden=32,
        ricci_modulation=True, ricci_scale=1.0, use_layernorm=False,
        backbone="resnet18_imagenet_l3", fpn="2level",
        query_head_kind="nodelet",
    )
    out = m(torch.rand(2, 3, 320, 320))
    assert out["box_corners"].shape == (2, 6, 4, 2)
    assert out["box_cls"].shape == (2, 6, 20)  # nodelet: no +1 slot
