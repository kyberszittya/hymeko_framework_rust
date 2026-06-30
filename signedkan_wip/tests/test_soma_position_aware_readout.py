"""Tests for the scalable position-aware readout (attention pool) and the
single-digit Cluttered-MNIST classification adapter — Phase 1 of the
position-aware-readout program
(``docs/plans/2026-06-29-soma-position-aware-readout-program/``).
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma.vision.walk_conv_classifier import (
    Readout,
    WalkConvImageClassifier,
    _AttentionReadout,
    _PosAttentionReadout,
    _SpatialTreeReadout,
    _build_readout,
)


# ---------------------------------------------------------------------
# Attention readout
# ---------------------------------------------------------------------


def test_attention_weights_are_a_softmax() -> None:
    ro = _AttentionReadout(n_patches=6, d_hidden=4)
    x = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
    w = ro.attention_weights(x)
    assert w.shape == (6,)
    assert torch.all(w >= 0)
    assert abs(w.sum().item() - 1.0) < 1e-6


def test_attention_out_dim_is_scale_free() -> None:
    """out_dim == d_hidden regardless of n_patches (unlike flatten's n*d) —
    the property that lets the attention pool scale to large grids."""
    assert _build_readout(Readout.ATTENTION, 7, 7, 16).out_dim == 16
    assert _build_readout(Readout.ATTENTION, 99, 101, 16).out_dim == 16


def test_attention_output_is_a_convex_combination() -> None:
    """Each output dim lies within [min, max] of that dim across patches —
    a softmax-weighted (convex) pool, never an extrapolation."""
    ro = _AttentionReadout(n_patches=5, d_hidden=4)
    x = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
    y = ro(x)
    assert torch.all(y <= x.max(dim=0).values + 1e-6)
    assert torch.all(y >= x.min(dim=0).values - 1e-6)


def test_attention_is_permutation_invariant() -> None:
    """The attention pool is a set function (content-weighted, not
    position-indexed): permuting patches leaves the output unchanged. This is
    the contrast with flatten — attention attends to *where content is*, it does
    not encode an absolute patch index."""
    ro = _AttentionReadout(n_patches=6, d_hidden=4)
    x = torch.randn(6, 4, generator=torch.Generator().manual_seed(2))
    perm = torch.tensor([3, 1, 5, 0, 4, 2])
    assert torch.allclose(ro(x), ro(x[perm]), atol=1e-6)


def test_attention_classifier_forward_and_head_width() -> None:
    model = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=Readout.ATTENTION)
    assert model.head.in_features == 16            # scale-free
    assert model(torch.randn(3, 1, 28, 28)).shape == (3, 10)


def test_attention_lighter_than_flatten() -> None:
    """Attention head is O(d); flatten head is O(n_patches·d). A later win for
    attention is therefore not a capacity artifact."""
    attn = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=Readout.ATTENTION)
    flat = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=Readout.FLATTEN)
    assert attn.n_parameters() < flat.n_parameters()


def test_attention_score_is_trainable() -> None:
    model = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=Readout.ATTENTION)
    model(torch.randn(2, 1, 28, 28)).sum().backward()
    g = model.readout.score.weight.grad
    assert g is not None and g.abs().sum().item() > 0.0


# ---------------------------------------------------------------------
# Positional attention readout (Phase 1.5)
# ---------------------------------------------------------------------


def test_pos_attention_out_dim_is_scale_free() -> None:
    assert _build_readout(Readout.POS_ATTENTION, 7, 7, 16).out_dim == 16
    assert _build_readout(Readout.POS_ATTENTION, 30, 40, 16).out_dim == 16


def test_pos_attention_is_permutation_sensitive() -> None:
    """Unlike pure attention, the positional embedding makes the pool
    position-aware: permuting patches changes the output."""
    ro = _PosAttentionReadout(n_patches=6, d_hidden=4)
    x = torch.randn(6, 4, generator=torch.Generator().manual_seed(4))
    perm = torch.tensor([3, 1, 5, 0, 4, 2])
    assert not torch.allclose(ro(x), ro(x[perm]), atol=1e-5)


def test_pos_attention_head_is_lighter_than_flatten() -> None:
    """Pos-attention keeps an O(d) head (+ an n_patches·d position table);
    flatten's head is n_patches·d·n_classes — pos-attention must be lighter."""
    pos = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=Readout.POS_ATTENTION)
    flat = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=Readout.FLATTEN)
    assert pos.head.in_features == 16
    assert pos.n_parameters() < flat.n_parameters()


def test_pos_attention_classifier_forward_and_trainable() -> None:
    model = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=Readout.POS_ATTENTION)
    model(torch.randn(2, 1, 28, 28)).sum().backward()
    assert model(torch.randn(3, 1, 28, 28)).shape == (3, 10)
    assert model.readout.pos.grad is not None
    assert model.readout.pos.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------
# Batched forward parity (GPU optimization: one (B,N,d) pass)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("readout", [
    Readout.MEAN_POOL, Readout.FLATTEN, Readout.ATTENTION,
    Readout.POS_ATTENTION, Readout.SPATIAL_TREE,
])
def test_batched_forward_matches_per_image_loop(readout) -> None:
    """The batched ``(B,N,d)`` forward equals looping the per-image path — the
    speedup must not change the numerics."""
    model = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=readout)
    model.eval()
    imgs = torch.randn(5, 1, 28, 28, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        batched = model(imgs)                                  # (5,10), batched path
        looped = torch.stack([model._forward_single(imgs[b]) for b in range(5)])
    assert batched.shape == (5, 10)
    assert torch.allclose(batched, looped, atol=1e-5)


def test_batched_training_step_gradients_match_loop() -> None:
    """The batched *backward* matches the per-image loop too: a training step's
    parameter gradients are identical (so batched training is numerically the
    same as per-image, just faster)."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    model = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=Readout.SPATIAL_TREE)
    imgs = torch.randn(4, 1, 28, 28, generator=torch.Generator().manual_seed(1))
    y = torch.tensor([1, 3, 5, 7])

    model.zero_grad()
    F.cross_entropy(model(imgs), y, reduction="sum").backward()
    g_batched = {n: p.grad.clone() for n, p in model.named_parameters()}

    model.zero_grad()
    loss = sum(F.cross_entropy(model._forward_single(imgs[i]).unsqueeze(0),
                               y[i:i + 1], reduction="sum") for i in range(4))
    loss.backward()
    for n, p in model.named_parameters():
        assert torch.allclose(g_batched[n], p.grad, atol=1e-4), f"grad mismatch: {n}"


def test_batched_forward_parity_with_cheby_and_holonomy() -> None:
    """Parity also holds with the Chebyshev-CR cell + holonomy aggregation."""
    from signedkan_wip.src.hymeko_gomb.soma.hg_conv import (
        Aggregation,
        MessageActivation,
    )
    model = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1, d_hidden=16,
        n_classes=10, use_sign_branching=False, aggregation=Aggregation.HOLONOMY,
        message_activation=MessageActivation.CHEBY_CR, readout=Readout.SPATIAL_TREE)
    model.eval()
    imgs = torch.randn(4, 1, 28, 28, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        batched = model(imgs)
        looped = torch.stack([model._forward_single(imgs[b]) for b in range(4)])
    assert torch.allclose(batched, looped, atol=1e-5)


# ---------------------------------------------------------------------
# Dynamic spatial-tree readout
# ---------------------------------------------------------------------


def test_spatial_tree_out_dim_is_grid_independent() -> None:
    """out_dim = (Σ levelᵢ²)·d, independent of the patch-grid size (scalable),
    unlike flatten's n_patches·d."""
    small = _SpatialTreeReadout(7, 7, 16)
    big = _SpatialTreeReadout(64, 64, 16)
    assert small.out_dim == big.out_dim == (1 + 4 + 16) * 16


def test_spatial_tree_forward_shape() -> None:
    ro = _SpatialTreeReadout(12, 12, 16)
    out = ro(torch.randn(144, 16))
    assert out.shape == (ro.out_dim,)


def test_spatial_tree_is_position_sensitive() -> None:
    """Permuting patches changes the multi-scale pooled descriptor (it keeps
    spatial layout, unlike mean-pool)."""
    ro = _SpatialTreeReadout(8, 8, 4, dynamic=False)
    x = torch.randn(64, 4, generator=torch.Generator().manual_seed(0))
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(1))
    assert not torch.allclose(ro(x), ro(x[perm]), atol=1e-5)


def test_spatial_tree_classifier_lighter_than_flatten_on_large_grid() -> None:
    """On a 12×12 grid the tree head (21·d) is far smaller than flatten (144·d)."""
    tree = WalkConvImageClassifier(
        image_h=48, image_w=48, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=Readout.SPATIAL_TREE)
    flat = WalkConvImageClassifier(
        image_h=48, image_w=48, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=Readout.FLATTEN)
    assert tree.head.in_features == (1 + 4 + 16) * 16
    assert tree.head.in_features < flat.head.in_features
    assert tree(torch.randn(2, 1, 48, 48)).shape == (2, 10)


def test_spatial_tree_dynamic_gate_trains() -> None:
    model = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=Readout.SPATIAL_TREE)
    model(torch.randn(2, 1, 28, 28)).sum().backward()
    assert model.readout.pool.gate.weight.grad is not None
    assert model.readout.pool.gate.weight.grad.abs().sum().item() > 0.0


def test_spatial_tree_static_has_no_gate() -> None:
    """The static variant is the same pyramid with no learned gate (the gate
    ablation control); same out_dim as the dynamic one."""
    dyn = _SpatialTreeReadout(12, 12, 16, dynamic=True)
    sta = _SpatialTreeReadout(12, 12, 16, dynamic=False)
    assert sta.pool.gate is None and dyn.pool.gate is not None
    assert sta.out_dim == dyn.out_dim
    assert sta(torch.randn(144, 16)).shape == (sta.out_dim,)


def test_anchor_spatial_tree_readout_variable_count() -> None:
    """The RicciStim anchor tree pools a *variable* anchor set into 21·d (the
    case flatten can't serve); empty cells contribute zero, not NaN."""
    from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_classifier import (
        _AnchorSpatialTreeReadout,
    )
    ro = _AnchorSpatialTreeReadout(8)
    for n in (3, 50):                                   # different anchor counts
        feats = torch.randn(n, 8)
        centers = torch.rand(n, 2)
        out = ro(feats, centers)
        assert out.shape == (ro.out_dim,) == (21 * 8,)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------
# RicciStim readout + structural ablation (Phase 2)
# ---------------------------------------------------------------------


def _ricci(readout: Readout, *, ablate: bool):
    from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_classifier import (
        RicciStimClassifier,
    )
    return RicciStimClassifier(
        image_h=28, image_w=28, d_hidden=16, n_classes=10, max_depth=1,
        ablate_structural_branches=ablate, readout=readout)


def test_ricci_mean_and_attention_readouts_forward() -> None:
    img = torch.randn(28, 28).clamp(0, 1).unsqueeze(0)  # (1,28,28)
    for ro in (Readout.MEAN_POOL, Readout.ATTENTION):
        for ablate in (False, True):
            logits = _ricci(ro, ablate=ablate)(img)
            assert logits.shape == (10,)


def test_ricci_spatial_tree_readout_forwards_and_head_width() -> None:
    """RicciStim with the anchor spatial-tree readout: head is 21·d, forwards
    over the variable anchor set."""
    model = _ricci(Readout.SPATIAL_TREE, ablate=False)
    assert model.head.in_features == 21 * 16
    img = torch.randn(28, 28).clamp(0, 1).unsqueeze(0)
    logits = model(img)
    assert logits.shape == (10,)
    logits.sum().backward()
    assert model.head.weight.grad is not None


def test_ricci_flatten_readout_rejected() -> None:
    """Flatten is ill-defined on RicciStim's variable anchor count — must raise,
    not silently misbehave."""
    with pytest.raises(ValueError, match="FLATTEN readout is unsupported"):
        _ricci(Readout.FLATTEN, ablate=False)


def test_ricci_attention_pool_is_used_when_selected() -> None:
    assert _ricci(Readout.ATTENTION, ablate=False).readout is not None
    assert _ricci(Readout.MEAN_POOL, ablate=False).readout is None


# ---------------------------------------------------------------------
# Cluttered single-digit classification adapter
# ---------------------------------------------------------------------


def test_cluttered_classification_returns_image_and_label() -> None:
    from signedkan_wip.src.vision.cluttered_classification import (
        ClutteredMNISTClassification,
    )
    ds = ClutteredMNISTClassification(n_samples=4, canvas=48, seed=0)
    img, label = ds[0]
    assert img.shape == (1, 48, 48)
    assert img.dtype == torch.float32 and 0.0 <= float(img.max()) <= 1.0
    assert isinstance(label, int) and 0 <= label <= 9


def test_cluttered_classification_is_deterministic() -> None:
    from signedkan_wip.src.vision.cluttered_classification import (
        ClutteredMNISTClassification,
    )
    ds = ClutteredMNISTClassification(n_samples=4, canvas=48, seed=0)
    img0, lab0 = ds[1]
    img1, lab1 = ds[1]
    assert lab0 == lab1 and torch.equal(img0, img1)


def test_cluttered_classification_rejects_small_canvas() -> None:
    from signedkan_wip.src.vision.cluttered_classification import (
        ClutteredMNISTClassification,
    )
    with pytest.raises(ValueError, match="canvas must be >= 28"):
        ClutteredMNISTClassification(n_samples=2, canvas=20, seed=0)
