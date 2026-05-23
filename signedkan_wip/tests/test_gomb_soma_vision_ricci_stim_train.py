"""Tests for the Ricci-Stim training infrastructure (Phase 8-bench)."""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma.vision import RicciStimDetector
from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_train import (
    assign_anchors_to_gt,
    detection_loss,
    _iou_boxes_xyxy,
    _greedy_nms,
)


# ---------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------


def test_iou_identical_boxes_returns_one():
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    iou = _iou_boxes_xyxy(box, box)
    assert torch.allclose(iou, torch.tensor([[1.0]]))


def test_iou_non_overlapping_returns_zero():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
    iou = _iou_boxes_xyxy(a, b)
    assert iou.item() == 0.0


def test_iou_half_overlap():
    """Two equal-area boxes overlapping in half their area
    have IoU = (overlap_area) / (sum - overlap) = 50 / (100+100-50) = 1/3."""
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[5.0, 0.0, 15.0, 10.0]])
    iou = _iou_boxes_xyxy(a, b)
    assert abs(iou.item() - 1 / 3) < 1e-5


# ---------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------


def test_assignment_no_gt_all_background():
    """When there are no GT boxes, every anchor is background."""
    pos = torch.tensor([[0, 0], [0, 8], [8, 0], [8, 8]], dtype=torch.long)
    siz = torch.tensor([4, 4, 4, 4], dtype=torch.long)
    a = assign_anchors_to_gt(
        pos, siz, torch.zeros(0, 4), torch.zeros(0, dtype=torch.long),
    )
    assert (a.cls_targets == 0).all()
    assert (a.positive_mask == False).all()
    assert a.bbox_targets.abs().sum().item() == 0


def test_assignment_high_iou_anchor_gets_foreground():
    """An anchor that overlaps significantly with a GT should be assigned."""
    pos = torch.tensor([[0, 0], [0, 50]], dtype=torch.long)
    siz = torch.tensor([10, 10], dtype=torch.long)
    # GT in upper-left, matching anchor 0 closely.
    gt = torch.tensor([[1.0, 1.0, 11.0, 11.0]])
    labels = torch.tensor([3], dtype=torch.long)
    a = assign_anchors_to_gt(pos, siz, gt, labels, iou_pos=0.3)
    # Anchor 0 should be foreground with label 3 + 1 = 4.
    assert a.cls_targets[0].item() == 4
    assert a.positive_mask[0].item() is True
    # Anchor 1 (no overlap with GT) is background.
    assert a.cls_targets[1].item() == 0


def test_assignment_bbox_offsets_encode_gt():
    """For an anchor exactly matching a GT, the regression target is (0, 0, 0, 0)."""
    pos = torch.tensor([[0, 0]], dtype=torch.long)
    siz = torch.tensor([10], dtype=torch.long)
    # GT at (cx=5, cy=5, w=10, h=10) = (x1=0, y1=0, x2=10, y2=10).
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    labels = torch.tensor([1], dtype=torch.long)
    a = assign_anchors_to_gt(pos, siz, gt, labels, iou_pos=0.3)
    # bbox_target should be (0, 0, 0, 0) since GT matches anchor exactly.
    assert torch.allclose(a.bbox_targets[0], torch.zeros(4), atol=1e-5)


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------


def test_detection_loss_runs():
    """Compute loss on a synthetic detector output."""
    m = RicciStimDetector(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=2, max_depth=1, max_anchors=64,
        d_hidden=8, n_classes=3,
    ).eval()
    img = torch.randn(1, 16, 16)
    out = m(img)
    n = out.n_anchors
    # All background.
    assignment = assign_anchors_to_gt(
        out.anchor_positions, out.anchor_sizes,
        torch.zeros(0, 4), torch.zeros(0, dtype=torch.long),
    )
    losses = detection_loss(out, assignment)
    assert "loss" in losses
    assert losses["loss"].item() >= 0
    assert losses["n_pos"] == 0


def test_detection_loss_gradient_flow():
    """Loss should propagate gradient when at least one positive
    assignment exists."""
    m = RicciStimDetector(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=2, max_depth=1, max_anchors=64,
        d_hidden=8, n_classes=3,
        bochner_alpha=0.1, bochner_beta=0.1,
    )
    img = torch.randn(1, 16, 16)
    out = m(img)
    # Plant a GT that overlaps the first anchor.
    pos = out.anchor_positions
    siz = out.anchor_sizes
    r0, c0 = pos[0, 0].item(), pos[0, 1].item()
    s0 = siz[0].item()
    gt = torch.tensor(
        [[float(c0), float(r0), float(c0 + s0), float(r0 + s0)]],
    )
    labels = torch.tensor([2], dtype=torch.long)
    assignment = assign_anchors_to_gt(pos, siz, gt, labels, iou_pos=0.3)
    losses = detection_loss(out, assignment)
    losses["loss"].backward()
    n_zero_grad = sum(
        1 for p in m.parameters()
        if p.grad is None or p.grad.abs().sum().item() == 0
    )
    # Detector head + backbone should receive gradient (with α, β > 0).
    assert n_zero_grad == 0


# ---------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------


def test_greedy_nms_empty():
    assert _greedy_nms(
        torch.zeros((0, 4)), torch.zeros(0), iou_thresh=0.5,
    ).numel() == 0


def test_greedy_nms_keeps_top_overlapping_pair():
    """Three boxes: A and B overlap; C is far away. NMS keeps A and C."""
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],   # A (highest score)
        [1.0, 1.0, 11.0, 11.0],   # B (overlaps A)
        [50.0, 50.0, 60.0, 60.0], # C (far away)
    ])
    scores = torch.tensor([1.0, 0.9, 0.8])
    kept = _greedy_nms(boxes, scores, iou_thresh=0.5)
    assert sorted(kept.tolist()) == [0, 2]


# ---------------------------------------------------------------------
# Smoke train
# ---------------------------------------------------------------------


def test_smoke_train_single_step():
    """One forward + backward + opt step on synthetic data — pipeline
    glue check."""
    torch.manual_seed(0)
    m = RicciStimDetector(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=2, max_depth=1, max_anchors=64,
        d_hidden=8, n_classes=3,
    )
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    img = torch.randn(1, 16, 16)
    gt = torch.tensor([[2.0, 2.0, 6.0, 6.0]])
    labels = torch.tensor([1], dtype=torch.long)
    opt.zero_grad()
    out = m(img)
    assignment = assign_anchors_to_gt(
        out.anchor_positions, out.anchor_sizes, gt, labels, iou_pos=0.3,
    )
    losses = detection_loss(out, assignment)
    losses["loss"].backward()
    opt.step()
    # Loss is finite.
    assert torch.isfinite(losses["loss"]).item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
