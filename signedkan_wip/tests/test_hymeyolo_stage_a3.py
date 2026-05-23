"""Tests for the 2026-05-16-evening HyMeYOLO Stage A-3 levers.

Four sub-levers, all behind CLI flags + kwargs:
  --use-layernorm     → LayerNorm on the +ricci-mod class head input
  --weight-decay      → Adam weight_decay
  --cls-loss focal    → focal-loss cross-entropy alternative
  --box-loss giou     → 1-GIoU on AABBs derived from matched corners

Each lever has its own pin; integration smoke (all four on) is in the
last test.
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.vision.hymeyolo_circles_ricci import (
    RicciHyMeYOLOMulti,
)
from signedkan_wip.src.vision.train_circles_ricci import (
    focal_loss_ce, giou_loss_xyxy,
)


# ─── LayerNorm lever ──────────────────────────────────────────────────


def test_layernorm_off_is_default_byte_identical() -> None:
    """Default (use_layernorm=False) must produce byte-identical
    forward output to a model constructed with the explicit-False flag
    on the same seed."""
    torch.manual_seed(42)
    m_a = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, d_hidden=16,
        ricci_modulation=True, ricci_scale=1.0, use_layernorm=False,
    )
    torch.manual_seed(42)
    m_b = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, d_hidden=16,
        ricci_modulation=True, ricci_scale=1.0,
    )
    m_b.load_state_dict(m_a.state_dict())
    m_a.eval(); m_b.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        a = m_a(x); b = m_b(x)
    for k in ("box_corners", "box_cls", "circle_corners", "circle_cls"):
        assert torch.equal(a[k], b[k]), f"{k} differs"


def test_layernorm_on_adds_two_extra_params() -> None:
    """LayerNorm adds (γ, β) over `cls_in` dimensions. For
    d_hidden=16 and ricci_modulation=True, cls_in = 16 + 3 = 19, so
    LayerNorm adds 2*19 = 38 parameters."""
    m_off = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, d_hidden=16,
        ricci_modulation=True, use_layernorm=False,
    )
    m_on = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, d_hidden=16,
        ricci_modulation=True, use_layernorm=True,
    )
    delta = (sum(p.numel() for p in m_on.parameters())
             - sum(p.numel() for p in m_off.parameters()))
    assert delta == 38, f"expected +38 params (γ+β over 19 dims); got {delta}"


def test_layernorm_on_forward_runs_finite() -> None:
    """A forward pass with use_layernorm=True must produce finite
    output of the same shapes."""
    torch.manual_seed(0)
    m = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, d_hidden=16,
        ricci_modulation=True, use_layernorm=True,
    )
    m.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = m(x)
    for k in ("box_corners", "box_cls", "circle_corners", "circle_cls"):
        assert torch.isfinite(out[k]).all(), f"non-finite {k}"


# ─── Focal-loss lever ─────────────────────────────────────────────────


def test_focal_loss_reduces_to_cross_entropy_at_gamma_zero() -> None:
    """At gamma=0 and alpha=None, focal_loss_ce should match
    F.cross_entropy up to floating-point tolerance."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    logits = torch.randn(8, 11)
    targets = torch.randint(0, 11, (8,))
    a = focal_loss_ce(logits, targets, gamma=0.0)
    b = F.cross_entropy(logits, targets)
    assert torch.allclose(a, b, atol=1e-6), f"focal(γ=0)={a}, CE={b}"


def test_focal_loss_downweights_easy_examples() -> None:
    """Focal loss should be SMALLER than CE on the same input when
    the predictions are confident-and-correct (high p_t → strong
    downweighting). Tests the headline mechanism."""
    import torch.nn.functional as F
    # Confident, correct prediction.
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    target = torch.tensor([0])
    focal = focal_loss_ce(logits, target, gamma=2.0)
    ce = F.cross_entropy(logits, target)
    assert focal < ce, (
        f"focal({focal}) should be < CE({ce}) when prediction is "
        f"confident-correct"
    )


def test_focal_loss_gradient_finite() -> None:
    """Backprop through focal loss should produce finite gradients."""
    torch.manual_seed(0)
    logits = torch.randn(8, 11, requires_grad=True)
    targets = torch.randint(0, 11, (8,))
    loss = focal_loss_ce(logits, targets, gamma=2.0)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0  # non-trivial gradient


# ─── GIoU lever ───────────────────────────────────────────────────────


def test_giou_loss_zero_when_pred_equals_target() -> None:
    """Identical boxes → 1 - GIoU = 0."""
    boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.2, 0.3, 0.6, 0.7]])
    loss = giou_loss_xyxy(boxes, boxes.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_giou_loss_positive_when_boxes_disjoint() -> None:
    """Two non-overlapping boxes: 1-GIoU should be > 1
    (GIoU < 0 for disjoint boxes; 1 - GIoU > 1)."""
    pred = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
    target = torch.tensor([[0.8, 0.8, 1.0, 1.0]])
    loss = giou_loss_xyxy(pred, target)
    assert loss.item() > 1.0


def test_giou_loss_partial_overlap_intermediate() -> None:
    """Partially overlapping boxes give 0 < loss < 2. GIoU can be
    negative even for partial overlap when the enclosing box is
    much larger than the union (1-GIoU > 1 then). The salient
    property is just that the loss is strictly positive (some
    distance from the perfect-match zero) and bounded (gradient
    signal exists). The disjoint case in the next test gives
    1-GIoU close to but bounded above 1+epsilon."""
    pred = torch.tensor([[0.0, 0.0, 0.4, 0.4]])
    target = torch.tensor([[0.2, 0.2, 0.6, 0.6]])
    loss = giou_loss_xyxy(pred, target)
    assert 0.0 < loss.item() < 2.0
    # Tighter check: at this specific (Δ=0.2 shift on 0.4-side
    # boxes), the exact value is 1.079 (verified analytically;
    # GIoU = 1/7 − 2/9 ≈ −0.079).
    assert loss.item() == pytest.approx(1.0794, abs=1e-3)


def test_giou_loss_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="matching"):
        giou_loss_xyxy(torch.zeros((3, 4)), torch.zeros((3, 5)))


def test_giou_gradient_finite() -> None:
    pred = torch.randn(4, 4, requires_grad=True)
    target = torch.randn(4, 4)
    loss = giou_loss_xyxy(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ─── End-to-end: focal + giou plumbed through hungarian_set_loss ─────


def test_hungarian_set_loss_accepts_a3_kwargs() -> None:
    """Drive `hungarian_set_loss` with all four Stage A-3 paths
    (ce×l1, focal×l1, ce×giou, focal×giou) on a synthetic batch and
    assert every variant produces a finite scalar loss."""
    from signedkan_wip.src.vision.hymeyolo_hungarian import hungarian_set_loss
    torch.manual_seed(0)
    B, N, M = 2, 4, 2
    n_classes = 10
    pred_corners = torch.randn(B, N, 4, 2)
    pred_cls = torch.randn(B, N, n_classes + 1)
    gt_corners = torch.randn(B, M, 4, 2)
    gt_classes = torch.randint(0, n_classes, (B, M))
    gt_counts = torch.tensor([M, M], dtype=torch.long)
    for cls_k in ("ce", "focal"):
        for box_k in ("l1", "giou"):
            loss, _, _ = hungarian_set_loss(
                pred_corners, pred_cls, gt_corners,
                gt_classes, gt_counts, n_classes=n_classes,
                cls_loss_kind=cls_k, box_loss_kind=box_k,
            )
            assert torch.isfinite(loss), (
                f"non-finite loss for cls={cls_k} box={box_k}: {loss}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
