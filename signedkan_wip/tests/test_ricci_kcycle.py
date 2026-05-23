"""Unit tests for `RicciKCycleHyMeYOLOMulti`.

Covers:
  - Construction at small d, no CUDA needed.
  - Forward returns the correct dict shape contract (drop-in with
    Hungarian matcher in `train_circles_ricci.py`).
  - **Bug-fix regression:** gradient from `out["box_corners"]` reaches
    `box_aggregator` parameters. The +kcycle localization bug at
    `reports/2026-05-13-hymeyolo-kcycle-localization-bug.md` was that
    the aggregator never participated in offset prediction; this test
    will fail against the buggy implementation.
  - Ricci-pathway gradient: backward from cls reaches the path
    that depends on corner positions through the Ricci scalars
    (geometric pathway).
  - `ricci_modulation=False` still constructs and runs.
"""
from __future__ import annotations

import math

import pytest
import torch

from signedkan_wip.src.vision.hymeyolo_ricci_kcycle import (
    RicciKCycleHyMeYOLOMulti,
)


def _make_model(ricci: bool = True, d: int = 8) -> RicciKCycleHyMeYOLOMulti:
    torch.manual_seed(0)
    return RicciKCycleHyMeYOLOMulti(
        n_box_queries=2, n_circle_queries=2,
        box_k=4, circle_k=6,
        n_classes=10, d_hidden=d,
        ricci_modulation=ricci,
    )


def test_construct_and_forward_shapes():
    """Forward returns the expected dict shape contract."""
    model = _make_model()
    x = torch.randn(3, 3, 32, 32)
    out = model(x)
    assert set(out.keys()) >= {
        "box_corners", "box_cls", "circle_corners", "circle_cls",
    }
    assert out["box_corners"].shape == (3, 2, 4, 2)
    assert out["circle_corners"].shape == (3, 2, 6, 2)
    assert out["box_cls"].shape == (3, 2, 11)   # n_classes + 1
    assert out["circle_cls"].shape == (3, 2, 11)


def test_offset_uses_aggregator_kcycle_bug_regression():
    """The +kcycle bug from 2026-05-13 was that KCycleSignedAggregator
    parameters didn't receive gradient when only `box_corners` was used
    as loss. After the fix, backward through ``out["box_corners"]`` must
    reach ``box_aggregator`` weights. Without the fix this assertion
    fails."""
    model = _make_model()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    # Loss depends ONLY on corner positions (localization), NOT on cls.
    loss = out["box_corners"].sum()
    loss.backward()
    grad_norm = sum(
        (p.grad.norm().item() if p.grad is not None else 0.0)
        for p in model.box_aggregator.parameters()
    )
    assert grad_norm > 0.0, (
        "box_aggregator parameters received no gradient from box_corners loss "
        "— +kcycle localization bug is back."
    )


def test_offset_uses_ricci_path():
    """Backward through offsets must reach corner positions via the
    Ricci scalar path. The Ricci scalars are computed from corner
    positions; if they feed offset prediction, gradient on the corners
    will be non-zero even when no F_map signal reaches them
    (verified by checking that corner gradients are non-zero in a
    Ricci-on model)."""
    model = _make_model(ricci=True)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    loss = out["box_corners"].sum()
    loss.backward()
    assert model.box_corners.grad is not None
    assert model.box_corners.grad.norm().item() > 0.0


def test_cls_path_reaches_all_three_signals():
    """Backward through cls must reach: aggregator (cycle path),
    corner positions (Ricci geometric path), and backbone (mean-pool
    pixel-feature path)."""
    model = _make_model(ricci=True)
    x = torch.randn(2, 3, 32, 32).requires_grad_(True)
    out = model(x)
    loss = out["box_cls"].sum()
    loss.backward()
    # Aggregator path.
    agg_grad = sum(
        (p.grad.norm().item() if p.grad is not None else 0.0)
        for p in model.box_aggregator.parameters()
    )
    assert agg_grad > 0.0, "cls did not propagate gradient into aggregator"
    # Corner-position path (Ricci feeds cls and is corner-derived).
    assert model.box_corners.grad is not None
    assert model.box_corners.grad.norm().item() > 0.0
    # Backbone path (mean-pool reads bilinear-sampled F_map features).
    assert x.grad is not None
    assert x.grad.norm().item() > 0.0


def test_ricci_off_still_works():
    """`ricci_modulation=False` should construct and run; cls input
    becomes [mean_pool, cycle_desc] only; gradient still flows from cls
    into the aggregator."""
    model = _make_model(ricci=False, d=8)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out["box_cls"].shape == (2, 2, 11)
    loss = out["box_cls"].sum()
    loss.backward()
    agg_grad = sum(
        (p.grad.norm().item() if p.grad is not None else 0.0)
        for p in model.box_aggregator.parameters()
    )
    assert agg_grad > 0.0


def test_param_count_is_finite_and_alpha_uniform_at_init():
    model = _make_model()
    assert isinstance(model.n_params(), int)
    assert model.n_params() > 0
    # α_κ logits init at 0 → softmax = uniform.
    a_box = model.alpha_box()
    a_circ = model.alpha_circle()
    assert torch.allclose(
        a_box, torch.full_like(a_box, 1.0 / a_box.shape[0]), atol=1e-5,
    )
    assert torch.allclose(
        a_circ, torch.full_like(a_circ, 1.0 / a_circ.shape[0]), atol=1e-5,
    )


def test_circles_only_or_boxes_only_dont_crash():
    """Degenerate configurations (one query type at zero count) work."""
    m1 = RicciKCycleHyMeYOLOMulti(
        n_box_queries=2, n_circle_queries=0, d_hidden=8,
    )
    x = torch.randn(1, 3, 32, 32)
    out = m1(x)
    assert out["box_corners"].shape == (1, 2, 4, 2)
    assert out["circle_corners"].shape == (1, 0, 8, 2)
    assert out["circle_cls"].shape == (1, 0, 11)

    m2 = RicciKCycleHyMeYOLOMulti(
        n_box_queries=0, n_circle_queries=2, d_hidden=8,
    )
    out = m2(x)
    assert out["box_corners"].shape == (1, 0, 4, 2)
    assert out["circle_corners"].shape == (1, 2, 8, 2)
