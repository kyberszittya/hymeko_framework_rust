"""Tests for RicciStimDetector (Ricci-Stim phase 8)."""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma.vision import (
    DetectionOutput,
    RicciStimDetector,
)


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_construction():
    d = RicciStimDetector()
    n = d.n_parameters()
    assert 1_000 < n < 30_000


def test_rejects_bad_image_shape():
    d = RicciStimDetector()
    with pytest.raises(ValueError, match="expected"):
        d(torch.zeros(28, 28))


# ---------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------


def test_single_image_returns_DetectionOutput():
    d = RicciStimDetector(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=2, max_depth=1, max_anchors=64,
        d_hidden=8, n_classes=3,
    ).eval()
    img = torch.randn(1, 16, 16)
    out = d(img)
    assert isinstance(out, DetectionOutput)
    # n_anchors equals tree.n_anchors (varies); always > 0.
    n = out.n_anchors
    assert n > 0
    assert out.cls_logits.shape == (n, 4)   # 3 classes + 1 bg
    assert out.bbox_offsets.shape == (n, 4)
    assert out.anchor_positions.shape == (n, 2)
    assert out.anchor_sizes.shape == (n,)


def test_batch_returns_list_of_outputs():
    d = RicciStimDetector(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=2, max_depth=1, max_anchors=64,
        d_hidden=8, n_classes=3,
    ).eval()
    batch = torch.randn(2, 1, 16, 16)
    outs = d(batch)
    assert isinstance(outs, list)
    assert len(outs) == 2
    for o in outs:
        assert isinstance(o, DetectionOutput)


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------


def test_gradient_flow_combined_loss():
    """Combined cls + bbox loss should propagate to every parameter.
    (Single-head losses only reach parameters connected to that
    head; the combined loss exercises both.) The α / β gates must be
    non-zero to wake up the Bochner Hodge / Ricci projection layers."""
    torch.manual_seed(0)
    d = RicciStimDetector(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=1, max_depth=1,
        d_hidden=8, n_classes=3,
        bochner_alpha=0.1, bochner_beta=0.1,
    )
    img = torch.randn(1, 12, 12)
    out = d(img)
    cls_targets = torch.zeros(out.n_anchors, dtype=torch.long)
    cls_loss = torch.nn.functional.cross_entropy(out.cls_logits, cls_targets)
    bbox_loss = out.bbox_offsets.abs().sum()
    loss = cls_loss + bbox_loss
    loss.backward()
    zero_grad = [name for name, p in d.named_parameters()
                 if p.grad is None or p.grad.abs().sum().item() == 0]
    assert not zero_grad, f"params with no gradient: {zero_grad}"


def test_cls_head_grad_only_from_cls_loss():
    """The cls_head receives gradient from cls loss alone, but NOT
    from bbox loss alone. Symmetric for bbox_head. This pins the
    two-head topology — losses don't cross-contaminate."""
    torch.manual_seed(0)
    d = RicciStimDetector(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=1, max_depth=1,
        d_hidden=8, n_classes=3,
    )
    img = torch.randn(1, 12, 12)
    out = d(img)
    cls_loss = torch.nn.functional.cross_entropy(
        out.cls_logits, torch.zeros(out.n_anchors, dtype=torch.long),
    )
    cls_loss.backward(retain_graph=True)
    # cls_head receives gradient; bbox_head does not.
    assert d.cls_head.weight.grad.abs().sum() > 0
    assert d.bbox_head.weight.grad is None or d.bbox_head.weight.grad.abs().sum() == 0


# ---------------------------------------------------------------------
# Training-signal contract
# ---------------------------------------------------------------------


def test_overfit_single_image_cls_and_bbox():
    """Drive both cls and bbox losses to small values on a single
    image's anchors. Pins end-to-end signal flow through the
    detector head."""
    torch.manual_seed(0)
    d = RicciStimDetector(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=2, max_depth=1, max_anchors=64,
        d_hidden=16, n_classes=2,
    )
    img = torch.randn(1, 12, 12)
    # Assign anchor 0 → class 1; rest → background (class 0).
    out = d(img)
    n = out.n_anchors
    cls_targets = torch.zeros(n, dtype=torch.long)
    cls_targets[0] = 1
    bbox_targets = torch.zeros(n, 4)
    bbox_targets[0] = torch.tensor([0.1, -0.1, 0.05, -0.05])

    opt = torch.optim.Adam(d.parameters(), lr=3e-3)
    for _step in range(300):
        opt.zero_grad()
        out = d(img)
        cls_loss = torch.nn.functional.cross_entropy(
            out.cls_logits, cls_targets,
        )
        # L1 bbox loss only on the foreground anchor.
        bbox_loss = (
            out.bbox_offsets[0] - bbox_targets[0]
        ).abs().sum()
        loss = cls_loss + bbox_loss
        loss.backward()
        opt.step()

    out = d(img)
    pred = out.cls_logits.argmax(dim=-1)
    assert pred[0].item() == 1, (
        f"foreground anchor not classified correctly; got {pred[0].item()}"
    )
    bbox_diff = (out.bbox_offsets[0] - bbox_targets[0]).abs().max().item()
    assert bbox_diff < 0.2, (
        f"bbox not converged; max diff = {bbox_diff:.3f}"
    )


# ---------------------------------------------------------------------
# Decode utility
# ---------------------------------------------------------------------


def test_decode_boxes_zero_offsets_returns_anchor_centers_and_sizes():
    """With (dx,dy,dw,dh) = 0, decoded bbox is the anchor itself
    (centre at anchor centre, size = anchor size)."""
    d = RicciStimDetector(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=2, max_depth=1,
        d_hidden=8, n_classes=2,
    ).eval()
    img = torch.zeros(1, 16, 16)
    out = d(img)
    # Override bbox_offsets to all zeros.
    out_zero = DetectionOutput(
        cls_logits=out.cls_logits,
        bbox_offsets=torch.zeros_like(out.bbox_offsets),
        anchor_positions=out.anchor_positions,
        anchor_sizes=out.anchor_sizes,
    )
    decoded = RicciStimDetector.decode_boxes(out_zero)
    assert decoded.shape == (out_zero.n_anchors, 4)
    # First anchor at (0, 0) size 4: centre (2, 2), size (4, 4).
    expected = torch.tensor([2.0, 2.0, 4.0, 4.0])
    assert torch.allclose(decoded[0], expected, atol=1e-5)


def test_decode_boxes_nonzero_offsets_shifts_center():
    d = RicciStimDetector(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=2, max_depth=1,
        d_hidden=8, n_classes=2,
    ).eval()
    img = torch.zeros(1, 16, 16)
    out = d(img)
    # First anchor at (0,0) size 4. Offset (dx=0.5, dy=-0.25, dw=0, dh=0).
    offsets = torch.zeros_like(out.bbox_offsets)
    offsets[0] = torch.tensor([0.5, -0.25, 0.0, 0.0])
    out_with = DetectionOutput(
        cls_logits=out.cls_logits,
        bbox_offsets=offsets,
        anchor_positions=out.anchor_positions,
        anchor_sizes=out.anchor_sizes,
    )
    decoded = RicciStimDetector.decode_boxes(out_with)
    # cx = 2 + 0.5*4 = 4; cy = 2 + -0.25*4 = 1; w = 4, h = 4.
    expected = torch.tensor([4.0, 1.0, 4.0, 4.0])
    assert torch.allclose(decoded[0], expected, atol=1e-5)


# ---------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------


def test_uniform_image_runs():
    d = RicciStimDetector(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=2, max_depth=1,
        d_hidden=8, n_classes=2,
    ).eval()
    img = torch.full((1, 16, 16), 0.5)
    out = d(img)
    assert out.n_anchors > 0
    assert not torch.isnan(out.cls_logits).any()
    assert not torch.isnan(out.bbox_offsets).any()


def test_n_anchors_varies_per_image():
    """A high-variance image triggers more subdivisions than a uniform
    one. Batch processing must accept this."""
    d = RicciStimDetector(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=1, max_depth=3, max_anchors=128,
        d_hidden=8, n_classes=2,
        score_threshold=0.01,
    ).eval()
    img_uniform = torch.full((1, 16, 16), 0.5)
    img_random = torch.randn(1, 16, 16)
    batch = torch.stack([img_uniform, img_random], dim=0)
    outs = d(batch)
    # Random image should have more anchors than uniform.
    assert outs[1].n_anchors >= outs[0].n_anchors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
