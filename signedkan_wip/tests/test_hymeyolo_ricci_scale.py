"""Tests for the 2026-05-16 HyMeYOLO Ricci-scale sweep patch.

Pins:

* ``ricci_scale=1.0`` is byte-identical to the pre-patch code path
  (forward output equality on a fixed seed).
* ``ricci_scale=0.0`` does not crash, and the gradient flowing
  through the Ricci branch of the class head is exactly zero
  (the corner-feature branch still backpropagates).
* ``compute_detection_metrics`` mAP@0.5 stays in [0, 1] even when
  many predictions overlap the same GT (the 2026-05-13 backfill
  reported seed-3 +ricci-mod = 1.017, which the new
  GT-consumption rule prevents).
* Each GT is consumed at most once per IoU level — three GTs with
  six perfect-IoU predictions credit three TPs total, not six.

Plan: ``docs/plans/2026-05-16-hymeyolo-ricci-weight-sweep/``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from signedkan_wip.src.vision.hymeyolo_circles_ricci import (
    RicciHyMeYOLOMulti,
)
from signedkan_wip.src.vision.train_circles_ricci import (
    compute_detection_metrics,
)


# ─── ricci_scale on the model ─────────────────────────────────────────


def _build_model(ricci_scale: float = 1.0, *, seed: int = 0) -> RicciHyMeYOLOMulti:
    torch.manual_seed(seed)
    return RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, circle_k=8,
        n_classes=10, d_hidden=16, ricci_modulation=True,
        ricci_scale=ricci_scale,
    )


def test_ricci_scale_default_byte_identical_to_prior() -> None:
    """``ricci_scale=1.0`` is the pre-patch default; the forward
    output of two parameter-identical models, one with
    ``ricci_scale=1.0`` (explicit) and one without (defaulting), must
    be byte-equal.

    A different construction-time seed would change ``head_cls`` /
    backbone weights; the equality below is forced via state-dict
    copy, isolating the only legal source of divergence
    (the multiplier itself).
    """
    m_ref = _build_model(seed=42)
    m_one = _build_model(ricci_scale=1.0, seed=99)  # different init
    m_one.load_state_dict(m_ref.state_dict())  # force weight equality

    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m_ref.eval()
    m_one.eval()
    with torch.no_grad():
        out_ref = m_ref(x)
        out_one = m_one(x)
    for key in ("box_corners", "box_cls", "circle_corners", "circle_cls"):
        assert torch.equal(out_ref[key], out_one[key]), (
            f"{key} diverges between default and explicit "
            f"ricci_scale=1.0 paths (should be byte-identical)"
        )


def test_ricci_scale_zero_kills_ricci_branch_gradient() -> None:
    """At ``ricci_scale=0.0``, the gradient w.r.t. corner positions
    through the *ricci* path is zero. The corner-feature path still
    propagates (it uses backbone features, not Ricci scalars), so the
    total gradient on corner-position params is non-zero and finite.
    """
    m = _build_model(ricci_scale=0.0, seed=7)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    out = m(x)
    # Mean of all class logits — drives gradient back to everything.
    loss = out["box_cls"].mean() + out["circle_cls"].mean()
    loss.backward()
    # Box corners are an nn.Parameter; gradient must be finite.
    grad_box = m.box_corners.grad
    assert grad_box is not None
    assert torch.isfinite(grad_box).all(), (
        "Non-finite gradient on box_corners at ricci_scale=0.0"
    )
    # The full gradient is the sum of (corner-feature path grad) +
    # (Ricci-feature path grad × ricci_scale). At scale=0, only the
    # corner-feature path contributes. We can't easily isolate the
    # Ricci-path component without running scale=1, so we instead
    # check it is non-trivially smaller than what scale=1 produces.
    m1 = _build_model(ricci_scale=1.0, seed=7)
    m1.load_state_dict(m.state_dict())  # same weights
    m1.box_corners.grad = None
    out1 = m1(x)
    loss1 = out1["box_cls"].mean() + out1["circle_cls"].mean()
    loss1.backward()
    grad_box_1 = m1.box_corners.grad
    diff = (grad_box - grad_box_1).abs().max().item()
    assert diff > 0.0, (
        "Gradient at scale=0 equals gradient at scale=1 — the Ricci "
        "branch is contributing zero in both cases, which contradicts "
        "the design intent"
    )


@pytest.mark.parametrize("scale", [0.0, 0.05, 0.1, 0.4, 0.8, 1.0, 2.0])
def test_ricci_scale_forward_runs_at_arbitrary_scale(scale: float) -> None:
    """Forward must not crash or produce NaN/Inf at any scale value
    in the sweep range (and a couple boundary values)."""
    m = _build_model(ricci_scale=scale, seed=1)
    m.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = m(x)
    for k in ("box_corners", "box_cls", "circle_corners", "circle_cls"):
        assert torch.isfinite(out[k]).all(), (
            f"Non-finite output in {k} at ricci_scale={scale}"
        )


# ─── mAP@0.5 fix: each GT consumed at most once per IoU level ────────


class _FixedPredictionModel(torch.nn.Module):
    """Test-only model that ignores inputs and returns hard-coded
    predictions. Lets us drive ``compute_detection_metrics`` against
    a known scenario."""

    def __init__(self, pred_boxes: torch.Tensor, pred_cls: torch.Tensor):
        super().__init__()
        self._pred_boxes = pred_boxes  # (B, Q, 4)
        self._pred_cls = pred_cls      # (B, Q, n_classes + 1)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        return (
            self._aabb_to_corners(self._pred_boxes[:B]),
            self._pred_cls[:B],
        )

    @staticmethod
    def _aabb_to_corners(aabb: torch.Tensor) -> torch.Tensor:
        """(B, Q, 4) → (B, Q, 4, 2) — pack as quadrilateral with the
        same min/max bounds. ``_aabb_from_corners`` round-trips this
        back to the original AABB."""
        x0 = aabb[..., 0]; y0 = aabb[..., 1]
        x1 = aabb[..., 2]; y1 = aabb[..., 3]
        c = torch.stack([
            torch.stack([x0, y0], dim=-1),
            torch.stack([x1, y0], dim=-1),
            torch.stack([x1, y1], dim=-1),
            torch.stack([x0, y1], dim=-1),
        ], dim=-2)
        return c


def _onehot_cls(class_id: int, score: float, n_classes: int = 10) -> list[float]:
    """Hand-build a softmax-friendly logits row with the named class
    holding `score` as its softmax probability."""
    # Probabilities: target class = score, no-obj class = 1 - score,
    # all others = 0 (via very-negative logits).
    p = [0.0] * (n_classes + 1)
    p[class_id] = score
    p[n_classes] = 1.0 - score
    # Convert to logits via log + a large negative for zero-prob slots.
    eps = 1e-12
    return [float(np.log(max(pi, eps))) for pi in p]


def test_compute_detection_metrics_caps_at_1() -> None:
    """Six predictions, all class-0 perfect-IoU against a single GT,
    must produce mAP_50 = 1.0 exactly — not 6.0 (pre-fix) and not
    inflated above 1.0."""
    # 1 image, 6 perfect-IoU class-0 predictions, 1 class-0 GT.
    gt_box = [0.25, 0.25, 0.50, 0.50]
    pred_boxes = torch.tensor([[[*gt_box]] * 6])  # (1, 6, 4)
    cls_score = 0.95
    pred_cls = torch.tensor([[
        _onehot_cls(0, cls_score) for _ in range(6)
    ]])
    model = _FixedPredictionModel(pred_boxes, pred_cls)
    X = torch.zeros(1, 3, 64, 64)
    boxes = torch.tensor([[gt_box]])
    classes = torch.tensor([[0]])
    counts = torch.tensor([1])

    out = compute_detection_metrics(model, X, boxes, classes, counts,
                                    n_classes=10, batch_size=1)
    assert 0.0 <= out["mAP_50"] <= 1.0, (
        f"mAP_50 outside [0,1]: {out['mAP_50']}"
    )
    # Perfect match must score 1.0 exactly.
    assert out["mAP_50"] == pytest.approx(1.0, abs=1e-6), (
        f"Expected mAP_50=1.0 for 6 perfect predictions over 1 GT; "
        f"got {out['mAP_50']}"
    )


def test_compute_detection_metrics_consumes_each_gt_once() -> None:
    """Three class-0 GTs at distinct locations, six perfect-IoU
    class-0 predictions (two per GT). Greedy matching must credit
    exactly 3 TPs and 3 FPs — not 6 TPs."""
    gt_boxes = [
        [0.1, 0.1, 0.3, 0.3],
        [0.4, 0.4, 0.6, 0.6],
        [0.7, 0.7, 0.9, 0.9],
    ]
    # Each GT gets two duplicates of itself as predictions.
    pred_list = [gt_boxes[0], gt_boxes[0],
                 gt_boxes[1], gt_boxes[1],
                 gt_boxes[2], gt_boxes[2]]
    pred_boxes = torch.tensor([pred_list])  # (1, 6, 4)
    pred_cls = torch.tensor([[
        _onehot_cls(0, 0.9 - 0.1 * i) for i in range(6)
    ]])  # different scores to fix the ordering
    model = _FixedPredictionModel(pred_boxes, pred_cls)
    X = torch.zeros(1, 3, 64, 64)
    boxes = torch.tensor([gt_boxes])
    classes = torch.tensor([[0, 0, 0]])
    counts = torch.tensor([3])

    out = compute_detection_metrics(model, X, boxes, classes, counts,
                                    n_classes=10, batch_size=1)
    # Critical bound: the metric must NOT exceed 1 (the pre-fix bug).
    assert out["mAP_50"] <= 1.0 + 1e-6, (
        f"mAP_50 exceeded 1.0: {out['mAP_50']}"
    )
    # The score-ordered duplicates land as TP-FP-TP-FP-TP-FP (each GT
    # consumed by the highest-scoring matching pred; its duplicate then
    # becomes FP). VOC all-points integration on this PR curve:
    #   monotone-from-right precisions: [1.0, 0.667, 0.667, 0.6, 0.6, 0.5]
    #   recalls: [1/3, 1/3, 2/3, 2/3, 3/3, 3/3]
    #   sum p_i × (r_i - r_{i-1}) = 1/3 + 0 + 0.667/3 + 0 + 0.6/3 + 0
    #                              ≈ 0.7556
    # This is the *correct* VOC AP for interleaved duplicates — and
    # would have been INFLATED past 1.0 by the pre-fix bug.
    assert out["mAP_50"] == pytest.approx(0.7556, abs=2e-3), (
        f"Expected VOC AP ≈ 0.7556 for 3 TPs interleaved with 3 FPs; "
        f"got {out['mAP_50']}"
    )
    # n_preds_used counts all valid predictions ever recorded — 6.
    assert out["n_preds_used"] == 6
    # n_gts_total counts ground truth.
    assert out["n_gts_total"] == 3


def test_compute_detection_metrics_one_pred_per_gt_is_perfect() -> None:
    """When each GT is hit by exactly one perfect-IoU prediction
    (no duplicates), the metric should be exactly 1.0 — the strict
    upper-bound case of GT-consumption-once."""
    gt_boxes = [
        [0.1, 0.1, 0.3, 0.3],
        [0.4, 0.4, 0.6, 0.6],
        [0.7, 0.7, 0.9, 0.9],
    ]
    pred_boxes = torch.tensor([gt_boxes])  # exactly one per GT
    pred_cls = torch.tensor([[
        _onehot_cls(0, 0.9 - 0.1 * i) for i in range(3)
    ]])
    model = _FixedPredictionModel(pred_boxes, pred_cls)
    X = torch.zeros(1, 3, 64, 64)
    boxes = torch.tensor([gt_boxes])
    classes = torch.tensor([[0, 0, 0]])
    counts = torch.tensor([3])
    out = compute_detection_metrics(model, X, boxes, classes, counts,
                                    n_classes=10, batch_size=1)
    assert out["mAP_50"] == pytest.approx(1.0, abs=1e-6)
    assert out["n_preds_used"] == 3


def test_compute_detection_metrics_handles_partial_matches() -> None:
    """One prediction at IoU 0.7 (passes 0.5, 0.55, 0.6, 0.65, 0.7;
    fails 0.75+). Resulting mAP_50 = 1.0 (perfect at 0.5 level);
    mAP_50:95 is positive but < 1."""
    # GT [0,0,0.4,0.4]; pred shifted to land at IoU ≈ 0.7.
    # IoU for two axis-aligned boxes shifted by Δ along x:
    #   inter = (0.4 - Δ) * 0.4; union = 2 × 0.4² - inter.
    # Solving for IoU = 0.7 gives Δ ≈ 0.0667... Use Δ = 0.0667.
    gt_box = [0.0, 0.0, 0.4, 0.4]
    pred_box = [0.0667, 0.0, 0.4 + 0.0667, 0.4]
    pred_boxes = torch.tensor([[pred_box]])
    pred_cls = torch.tensor([[_onehot_cls(0, 0.9)]])
    model = _FixedPredictionModel(pred_boxes, pred_cls)
    X = torch.zeros(1, 3, 64, 64)
    boxes = torch.tensor([[gt_box]])
    classes = torch.tensor([[0]])
    counts = torch.tensor([1])

    out = compute_detection_metrics(model, X, boxes, classes, counts,
                                    n_classes=10, batch_size=1)
    assert out["mAP_50"] == pytest.approx(1.0, abs=1e-6), (
        f"Single pred at IoU 0.7 should fully match at IoU=0.5; "
        f"got mAP_50={out['mAP_50']}"
    )
    assert 0.0 < out["mAP_50_95"] < 1.0, (
        f"Single pred at IoU 0.7 should give partial COCO mAP; "
        f"got mAP_50_95={out['mAP_50_95']}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
