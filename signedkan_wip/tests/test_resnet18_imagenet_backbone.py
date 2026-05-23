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
