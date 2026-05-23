"""Tests for RicciStimClassifier (Ricci-Stim phase 7).

Pins the end-to-end pipeline:

  * construction at MNIST dimensions;
  * forward shape (single image + batch);
  * gradient flow through every component (quadtree → encoder →
    StimulusGraphBuilder → 3 Bochner-wrapped layers → head);
  * overfit-2-samples sanity (the central training-signal contract);
  * Bochner coupling: α > 0 changes output vs α = 0;
  * graceful handling when a primitive family is empty (e.g. no triangles).
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma.vision import RicciStimClassifier


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_construction_mnist_defaults():
    m = RicciStimClassifier()
    # Sanity: bounded parameter count for the MNIST-defaults config.
    n_params = m.n_parameters()
    assert 1_000 < n_params < 20_000, (
        f"unexpected parameter count: {n_params}"
    )


def test_rejects_bad_image_shape():
    m = RicciStimClassifier()
    with pytest.raises(ValueError, match="expected"):
        m(torch.zeros(28, 28))  # missing channel dim AND no batch


# ---------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------


def test_forward_shape_single_image():
    m = RicciStimClassifier().eval()
    img = torch.randn(1, 28, 28)
    logits = m(img)
    assert logits.shape == (10,)


def test_forward_shape_batch():
    m = RicciStimClassifier().eval()
    batch = torch.randn(3, 1, 28, 28)
    logits = m(batch)
    assert logits.shape == (3, 10)


def test_forward_smaller_image():
    m = RicciStimClassifier(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=1, max_depth=2,
        d_hidden=8, n_classes=4,
    ).eval()
    img = torch.randn(1, 16, 16)
    logits = m(img)
    assert logits.shape == (4,)


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------


def test_gradient_flow_all_components():
    """Backward populates non-zero gradients on every learnable
    parameter: patch_encoder, walk_layer (incl. Bochner α/β + projs +
    inner Walk weights), poly_layer (same), tri_layer (same), head."""
    m = RicciStimClassifier(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=1, max_depth=1,
        d_hidden=8, n_classes=3,
        bochner_alpha=0.1, bochner_beta=0.1,
    )
    img = torch.randn(1, 12, 12, requires_grad=False)
    logits = m(img)
    loss = logits.pow(2).sum()
    loss.backward()
    # Check every named parameter received gradient.
    zero_grad_params = []
    for name, p in m.named_parameters():
        if p.grad is None or p.grad.abs().sum().item() == 0:
            zero_grad_params.append(name)
    assert not zero_grad_params, (
        f"params with zero gradient: {zero_grad_params}"
    )


# ---------------------------------------------------------------------
# Training-signal contract
# ---------------------------------------------------------------------


def test_overfit_two_samples():
    """End-to-end overfit: 2 random images, 2 labels, 200 optimizer
    steps → 100 % accuracy. Verifies the entire pipeline carries
    training signal end-to-end."""
    torch.manual_seed(0)
    m = RicciStimClassifier(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=1, max_depth=2,
        d_hidden=16, n_classes=2,
    )
    x = torch.randn(2, 1, 12, 12)
    y = torch.tensor([0, 1])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _step in range(250):
        opt.zero_grad()
        logits = m(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    final_pred = m(x).argmax(dim=-1)
    assert (final_pred == y).all(), (
        f"failed to overfit 2 samples; final preds = {final_pred.tolist()}, "
        f"final loss = {loss.item():.4f}"
    )


# ---------------------------------------------------------------------
# Bochner coupling
# ---------------------------------------------------------------------


def test_bochner_alpha_changes_output():
    """With α = 0 vs α > 0 (same seeds), the output should differ."""
    torch.manual_seed(0)
    m0 = RicciStimClassifier(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=1, max_depth=1,
        d_hidden=8, n_classes=3,
        bochner_alpha=0.0, bochner_beta=0.0,
    ).eval()
    torch.manual_seed(0)
    m1 = RicciStimClassifier(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=1, max_depth=1,
        d_hidden=8, n_classes=3,
        bochner_alpha=0.5, bochner_beta=0.0,
    ).eval()
    img = torch.randn(1, 12, 12)
    y0 = m0(img)
    y1 = m1(img)
    diff = (y0 - y1).abs().max().item()
    assert diff > 1e-3, (
        f"α=0.5 did not change output vs α=0; max diff = {diff:.2e}"
    )


def test_bochner_beta_changes_output():
    torch.manual_seed(0)
    m0 = RicciStimClassifier(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=1, max_depth=1,
        d_hidden=8, n_classes=3,
        bochner_alpha=0.0, bochner_beta=0.0,
    ).eval()
    torch.manual_seed(0)
    m1 = RicciStimClassifier(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=1, max_depth=1,
        d_hidden=8, n_classes=3,
        bochner_alpha=0.0, bochner_beta=0.5,
    ).eval()
    img = torch.randn(1, 12, 12)
    y0 = m0(img)
    y1 = m1(img)
    diff = (y0 - y1).abs().max().item()
    assert diff > 1e-3, (
        f"β=0.5 did not change output vs β=0; max diff = {diff:.2e}"
    )


# ---------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------


def test_uniform_image_runs():
    """A constant image causes no quadtree subdivision and likely few
    or no triangles. The classifier should still produce a valid output."""
    m = RicciStimClassifier(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=1, max_depth=2,
        d_hidden=8, n_classes=4,
    ).eval()
    img = torch.full((1, 16, 16), 0.5)
    logits = m(img)
    assert logits.shape == (4,)
    assert not torch.isnan(logits).any()


def test_n_parameters_breakdown():
    """Verify rough parameter breakdown: backbone (patch_encoder + 3
    branches) + head, all small."""
    m = RicciStimClassifier(
        image_h=28, image_w=28, patch_size_initial=4,
        patch_size_min=1, max_depth=2,
        d_hidden=16, n_classes=10,
    )
    n_pe = sum(p.numel() for p in m.backbone.patch_encoder.parameters())
    n_walk = sum(p.numel() for p in m.backbone.walk_layer.parameters())
    n_poly = sum(p.numel() for p in m.backbone.poly_layer.parameters())
    n_tri = sum(p.numel() for p in m.backbone.tri_layer.parameters())
    n_head = sum(p.numel() for p in m.head.parameters())
    total = n_pe + n_walk + n_poly + n_tri + n_head
    assert n_pe == 272                            # 16 * 16 + 16
    assert n_head == 170                          # 16 * 10 + 10
    assert n_walk == 1568 + 2 + 2 * (16 * 16 + 16)
    assert n_poly == 2 * (16 * 16) + 2 * 16 + 2 + 2 * (16 * 16 + 16)
    assert n_tri == n_poly
    assert total == m.n_parameters()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
