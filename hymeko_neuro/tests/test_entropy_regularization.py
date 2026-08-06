"""Tests for the confidence-penalty entropy regularisation in the
combined set loss (added 2026-06-11 to regularise the high-capacity
L3 backbone, which overfits VOC trainval).

Confidence penalty (Pereyra et al. 2017): total <- total - lam * H,
where H is the mean per-query class entropy. Maximising H curbs the
over-confident predictions a memorising model produces.
"""
from __future__ import annotations

import math

import torch

from hymeko_neuro.experiments.vision.hymeyolo_circles_ricci import RicciHyMeYOLOMulti
from hymeko_neuro.experiments.vision.train_circles_ricci import combined_set_loss


def _toy():
    torch.manual_seed(0)
    m = RicciHyMeYOLOMulti(
        n_box_queries=6, n_circle_queries=0, n_classes=20, d_hidden=32,
        ricci_modulation=True, backbone="resnet", fpn="2level",
        query_head_kind="nodelet",
    )
    out = m(torch.rand(2, 3, 64, 64))
    gtb, _ = torch.rand(2, 3, 4).clamp(0, 1).sort(-1)
    gtc = torch.randint(0, 20, (2, 3))
    gtk = torch.tensor([2, 1])
    return out, gtb, gtc, gtk


def test_entropy_default_is_noop():
    out, gtb, gtc, gtk = _toy()
    a, _ = combined_set_loss(out, gtb, gtc, gtk, n_classes=20,
                             lam_gate_neg_override=2.0)
    b, _ = combined_set_loss(out, gtb, gtc, gtk, n_classes=20,
                             lam_gate_neg_override=2.0, lam_entropy=0.0)
    assert torch.allclose(a, b)


def test_entropy_penalty_subtracts_lambda_times_entropy():
    out, gtb, gtc, gtk = _toy()
    base, _ = combined_set_loss(out, gtb, gtc, gtk, n_classes=20,
                                lam_gate_neg_override=2.0, lam_entropy=0.0)
    pen, _ = combined_set_loss(out, gtb, gtc, gtk, n_classes=20,
                               lam_gate_neg_override=2.0, lam_entropy=0.3)
    # diff = -lam * H; recover H and check it is a valid entropy in
    # [0, ln C] for C = 20 classes.
    h = (base - pen).item() / 0.3
    assert 0.0 <= h <= math.log(20) + 1e-4, h
    # at init the head is near-uniform, so H should be close to ln(20).
    assert h > 0.5 * math.log(20)


def test_entropy_penalty_is_differentiable():
    out, gtb, gtc, gtk = _toy()
    loss, _ = combined_set_loss(out, gtb, gtc, gtk, n_classes=20,
                                lam_gate_neg_override=2.0, lam_entropy=0.3)
    loss.backward()
    assert torch.isfinite(loss)
