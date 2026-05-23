"""Tests for WalkConvImageClassifier (GömbSoma vision phase 3-V)."""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma.vision import (
    WalkConvImageClassifier,
)


def test_construction_smoke():
    """Build an MNIST-shaped classifier; verify the documented shapes."""
    m = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4,
        in_channels=1, d_hidden=16, n_classes=10,
    )
    assert m.builder.n_patches == 49
    # n_walks: for a 7×7 grid each interior vertex (deg=4) is the
    # middle of 4×3 = 12 walks; border vertices (deg=3) middle of 6;
    # corners (deg=2) middle of 2. Total = 4 corners × 2 + 20 borders
    # × 6 + 25 interior × 12 = 8 + 120 + 300 = 428.
    assert m.builder.walks.shape[0] == 428
    # n_params: patch_embed (16×16 + 16) + walk_conv (2×3×16×16 + 2×16) + head (16×10 + 10)
    # = 272 + 1568 + 170 = 2010
    assert m.n_parameters() == 2010


def test_forward_shape_batched():
    m = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4,
        in_channels=1, d_hidden=16, n_classes=10,
    ).eval()
    batch = torch.randn(4, 1, 28, 28)
    logits = m(batch)
    assert logits.shape == (4, 10)


def test_forward_shape_single_image():
    """A bare (C, H, W) image (no batch axis) is accepted."""
    m = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4,
        in_channels=1, d_hidden=16, n_classes=10,
    ).eval()
    img = torch.randn(1, 28, 28)
    logits = m(img)
    assert logits.shape == (10,)


def test_gradient_flow_through_all_components():
    """Backward populates gradients on patch_embed, walk_conv, head."""
    m = WalkConvImageClassifier(
        image_h=12, image_w=12, patch_size=4,
        in_channels=1, d_hidden=8, n_classes=3,
    )
    batch = torch.randn(2, 1, 12, 12)
    logits = m(batch)
    loss = logits.pow(2).sum()
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, (
            f"parameter {name!r} has no gradient"
        )


def test_brightness_inverted_input_changes_output():
    """Inverting an image's brightness changes the edge sign pattern
    everywhere, hence changes the output. This is a soft sanity test
    that the layer actually reads the signed structure."""
    m = WalkConvImageClassifier(
        image_h=12, image_w=12, patch_size=4,
        in_channels=1, d_hidden=8, n_classes=4,
        use_sign_branching=True,
    ).eval()
    img = torch.rand(1, 12, 12)
    img_inverted = 1.0 - img
    out_a = m(img)
    out_b = m(img_inverted)
    diff = (out_a - out_b).abs().max().item()
    assert diff > 1e-3, (
        f"sign-branching path is dead — inverted image gave nearly "
        f"identical output (max diff = {diff:.2e})"
    )


def test_no_sign_branching_falls_back():
    """With sign-branching off, the classifier still trains and produces
    valid output (just without the sign-routing benefit)."""
    m = WalkConvImageClassifier(
        image_h=12, image_w=12, patch_size=4,
        in_channels=1, d_hidden=8, n_classes=4,
        use_sign_branching=False,
    )
    batch = torch.randn(2, 1, 12, 12)
    logits = m(batch)
    assert logits.shape == (2, 4)
    loss = logits.pow(2).sum()
    loss.backward()  # must not raise


def test_can_overfit_two_samples():
    """End-to-end signal-flow check: the classifier should be able to
    perfectly overfit a 2-sample 2-class dataset within 200 steps.
    Fails if the architecture has a learning-blocker bug."""
    m = WalkConvImageClassifier(
        image_h=12, image_w=12, patch_size=4,
        in_channels=1, d_hidden=16, n_classes=2,
    )
    torch.manual_seed(0)
    x = torch.randn(2, 1, 12, 12)
    y = torch.tensor([0, 1])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for step in range(300):
        opt.zero_grad()
        logits = m(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    final_pred = m(x).argmax(dim=1)
    assert (final_pred == y).all(), (
        f"failed to overfit 2 samples; final pred = {final_pred.tolist()}, "
        f"loss = {loss.item():.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
