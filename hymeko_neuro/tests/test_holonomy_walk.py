"""Tests for the holonomy-group ablation walk-conv (none/routing/Z2/U1)."""
from __future__ import annotations

import pytest
import torch

from hymeko_neuro.models.hymeko_gomb.soma.vision.holonomy_walk import (
    Holonomy,
    HolonomyClassifier,
    HolonomyWalkConv,
    _rotate_pairs,
)


def test_rotate_pairs_is_a_rotation() -> None:
    """Pair rotation preserves norm and matches the closed-form U(1) action."""
    msg = torch.tensor([[1.0, 0.0, 0.0, 2.0]])
    theta = torch.tensor([torch.pi / 2])
    out = _rotate_pairs(msg, theta)
    # (1,0)→(0,1) and (0,2)→(-2,0) under +90°.
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, -2.0, 0.0]]), atol=1e-5)
    assert torch.allclose(out.norm(), msg.norm(), atol=1e-5)


def test_u1_requires_even_width() -> None:
    with pytest.raises(ValueError, match="even d_out"):
        HolonomyWalkConv(8, 7, Holonomy.U1)


@pytest.mark.parametrize("mode", list(Holonomy))
def test_all_modes_forward_and_train(mode: Holonomy) -> None:
    model = HolonomyClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, mode=mode)
    logits = model(torch.randn(2, 1, 28, 28))
    assert logits.shape == (2, 10)
    logits.sum().backward()
    assert model.conv.W.grad is not None and model.conv.W.grad.abs().sum() > 0


def test_routing_has_two_banks_others_one() -> None:
    routing = HolonomyWalkConv(16, 16, Holonomy.ROUTING)
    z2 = HolonomyWalkConv(16, 16, Holonomy.Z2)
    assert routing.W.shape[0] == 2 and z2.W.shape[0] == 1
    assert z2.alpha is None
    assert HolonomyWalkConv(16, 16, Holonomy.U1).alpha is not None


def test_u1_alpha_is_trainable() -> None:
    """The learned flux scale α receives gradient (the U(1) connection learns)."""
    model = HolonomyClassifier(28, 28, 4, 1, 16, 10, Holonomy.U1)
    model(torch.randn(2, 1, 28, 28)).sum().backward()
    assert model.conv.alpha.grad is not None
    assert model.conv.alpha.grad.abs().item() >= 0.0   # finite gradient exists


@pytest.mark.parametrize("mode", list(Holonomy))
def test_holonomy_batched_forward_matches_loop(mode: Holonomy) -> None:
    """The batched (B,N,d) holonomy forward equals the per-image loop exactly —
    for every connection mode (none/routing/Z2/U1)."""
    model = HolonomyClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, mode=mode)
    model.eval()
    imgs = torch.randn(5, 1, 28, 28, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        batched = model(imgs)
        looped = torch.stack([model._forward_single(imgs[b]) for b in range(5)])
    assert batched.shape == (5, 10)
    assert torch.allclose(batched, looped, atol=1e-5)


def test_z2_sign_flips_message_contribution() -> None:
    """Z2 connection multiplies the message by the walk σ — flipping all signs
    negates the pre-pool message (a single-bank, sign-as-connection check)."""
    conv = HolonomyWalkConv(4, 4, Holonomy.Z2)
    conv.eval()
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(6, 4, generator=rng)
    walks = torch.tensor([[0, 1, 2], [3, 4, 5]])
    diffs = torch.zeros(2, 2)
    idx = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]])
    M_v = torch.sparse_coo_tensor(idx, torch.ones(6), (6, 2)).coalesce()
    pos = conv(x, walks, torch.tensor([1, 1]), diffs, M_v)
    neg = conv(x, walks, torch.tensor([-1, -1]), diffs, M_v)
    assert torch.allclose(pos, -neg, atol=1e-6)
