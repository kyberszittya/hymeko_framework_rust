"""Tests for the MiniDETR budget-matched baseline.

The load-bearing one is `test_detr_overfits_small_set`: a competent DETR must be
able to overfit a handful of images. If it can't, the baseline is a strawman and
any "RicciStim is at parity with DETR" comparison built on it is invalid. The
metric-parity tests pin that DETR and RicciStim are scored by identical code.

Run: pytest -p no:randomly hymeko_neuro/tests/test_detr_baseline.py
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from hymeko_neuro.models.hymeko_gomb.soma.vision.ricci_stim_train import (
    match_f1_at_iou50,
)
from hymeko_neuro.experiments.vision.cluttered_mnist import (
    ClutteredMNIST,
    collate_cluttered,
)
from hymeko_neuro.experiments.vision.detr_baseline import (
    PRESETS,
    build_detr,
    evaluate_detr_map50,
)


# ── model shape + size ────────────────────────────────────────────────

def test_forward_shapes_both_presets() -> None:
    for size in ("tiny", "standard"):
        m = build_detr(size, canvas=64, n_classes=10).eval()
        corners, cls = m(torch.randn(2, 1, 64, 64))
        n = PRESETS[size].n_queries
        assert corners.shape == (2, n, 4, 2)
        assert cls.shape == (2, n, 11)         # 10 classes + no-object
        # boxes are normalised into [0, 1]
        assert float(corners.min()) >= -1e-4 and float(corners.max()) <= 1.0 + 1e-4


def test_tiny_is_parameter_matched_order() -> None:
    tiny = build_detr("tiny").n_parameters()
    standard = build_detr("standard").n_parameters()
    # tiny is ~10^4 (same order as RicciStim's ~5.9k), standard is much larger.
    assert tiny < 50_000, f"tiny has {tiny} params — not parameter-matched"
    assert standard > 5 * tiny


# ── shared metric parity ──────────────────────────────────────────────

def test_match_f1_perfect_wrong_class_and_no_overlap() -> None:
    gt_boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    gt_lbl = torch.tensor([3])
    # exact box + right class → F1 = 1.0
    assert match_f1_at_iou50(gt_boxes.clone(), torch.tensor([3]),
                             gt_boxes, gt_lbl) == 1.0
    # exact box, WRONG class → no TP → 0.0
    assert match_f1_at_iou50(gt_boxes.clone(), torch.tensor([5]),
                             gt_boxes, gt_lbl) == 0.0
    # right class, disjoint box (IoU 0) → 0.0
    far = torch.tensor([[90.0, 90.0, 99.0, 99.0]])
    assert match_f1_at_iou50(far, torch.tensor([3]), gt_boxes, gt_lbl) == 0.0


def test_match_f1_partial_precision() -> None:
    # one correct pred + one spurious pred against one GT:
    # precision 1/2, recall 1/1 → F1 = 2*0.5*1/(0.5+1) = 0.6667
    gt = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    preds = torch.tensor([[10.0, 10.0, 20.0, 20.0], [50.0, 50.0, 60.0, 60.0]])
    f1 = match_f1_at_iou50(preds, torch.tensor([1, 1]), gt, torch.tensor([1]))
    assert abs(f1 - 2 / 3) < 1e-6


# ── FAIRNESS GUARD: the model must be able to overfit ─────────────────

@pytest.mark.slow
def test_detr_overfits_small_set() -> None:
    """A correct DETR overfits a tiny train set to a high F1. A strawman
    (broken loss / heads / matching) would stay near zero — which would make the
    parity comparison dishonest, so this is a hard gate before any headline.

    Recipe (diagnosed 2026-06-16, reports/2026-06-16-detr-overfit-fix.md): the
    box loss must be ``l1giou`` (pure GIoU saturates at IoU≈0.46, just under the
    0.5 the metric needs), the lr must be 1e-3 (3e-3 diverges past ~step 400),
    and grad-norm clipping at 0.1 stabilises it. Under this recipe the model
    reaches mAP50_proxy=1.0 by ~step 450; the gate below is 0.4 with margin.
    This is a deliberately heavy guard (≈600 train steps) — hence ``slow``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    # Single-object images (easiest to memorise → a clean pipeline-correctness
    # proof), evaluated on the SAME 16, with enough steps to overfit.
    ds = ClutteredMNIST(n_samples=16, canvas=64, max_digits=1, seed=0, train=True)
    loader = DataLoader(ds, batch_size=16, shuffle=True,
                        collate_fn=collate_cluttered)
    model = build_detr("standard", canvas=64, n_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    from hymeko_neuro.experiments.vision.detr_baseline import _pad_targets
    from hymeko_neuro.experiments.vision.hymeyolo_hungarian import hungarian_set_loss

    init = evaluate_detr_map50(model, loader, device, canvas=64)
    for _ in range(600):
        for images, bb, lb in loader:
            images = images.to(device)
            gc, gcl, gcnt = _pad_targets(bb, lb, 64, 10, device)
            corners, cls = model(images)
            loss, _, _ = hungarian_set_loss(
                corners, cls, gc, gcl, gcnt, n_classes=10, box_loss_kind="l1giou")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
    final = evaluate_detr_map50(model, loader, device, canvas=64)
    assert final > init + 0.1, f"DETR did not learn: init={init:.3f} final={final:.3f}"
    assert final > 0.4, (
        f"DETR could not overfit 16 images (final mAP50_proxy={final:.3f}) — "
        "the baseline would be a strawman; fix before any comparison."
    )
