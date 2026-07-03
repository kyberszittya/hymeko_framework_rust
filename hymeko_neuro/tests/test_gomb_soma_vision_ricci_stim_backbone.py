"""Tests for RicciStimBackbone (Ricci-Stim phase 9 consolidation).

Pins the shared backbone contract:

  * forward returns (features, tree) with consistent shapes;
  * features have shape (tree.n_anchors, d_hidden);
  * deterministic;
  * gradient flow through every backbone component (with α / β > 0);
  * empty primitive families degrade gracefully (no NaN, valid output);
  * α=β=0 ⇒ Bochner branches reduce to flat-connection only.
"""
from __future__ import annotations

import pytest
import torch

from hymeko_neuro.models.hymeko_gomb.soma.vision import (
    AnchorTree,
    RicciStimBackbone,
)


def test_forward_returns_features_and_tree():
    bb = RicciStimBackbone(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=1, max_depth=1,
        d_hidden=8,
    ).eval()
    img = torch.randn(1, 12, 12)
    features, tree = bb(img)
    assert isinstance(tree, AnchorTree)
    assert features.shape == (tree.n_anchors, 8)


def test_rejects_bad_input_shape():
    bb = RicciStimBackbone(
        image_h=12, image_w=12, patch_size_initial=4, d_hidden=8,
    )
    with pytest.raises(ValueError, match="expected"):
        bb(torch.zeros(12, 12))  # missing channel dim


def test_features_consistent_with_tree():
    """features.shape[0] must equal tree.n_anchors for every call."""
    bb = RicciStimBackbone(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=1, max_depth=2, max_anchors=64,
        d_hidden=8,
    ).eval()
    for seed in range(3):
        torch.manual_seed(seed)
        img = torch.randn(1, 16, 16)
        features, tree = bb(img)
        assert features.shape[0] == tree.n_anchors


def test_deterministic():
    bb = RicciStimBackbone(
        image_h=12, image_w=12, patch_size_initial=4, d_hidden=8,
    ).eval()
    img = torch.randn(1, 12, 12)
    f1, t1 = bb(img)
    f2, t2 = bb(img)
    assert torch.equal(f1, f2)
    assert torch.equal(t1.positions, t2.positions)
    assert torch.equal(t1.scales, t2.scales)


def test_gradient_flow_with_bochner_coupling():
    """With α / β > 0, gradients must flow to every backbone param,
    including the Bochner Hodge / Ricci projections."""
    torch.manual_seed(0)
    bb = RicciStimBackbone(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=1, max_depth=1,
        d_hidden=8,
        bochner_alpha=0.1, bochner_beta=0.1,
    )
    img = torch.randn(1, 12, 12)
    features, _ = bb(img)
    loss = features.pow(2).sum()
    loss.backward()
    zero_grad = [name for name, p in bb.named_parameters()
                 if p.grad is None or p.grad.abs().sum().item() == 0]
    assert not zero_grad, f"params with no gradient: {zero_grad}"


def test_alpha_beta_zero_zeros_bochner_projections():
    """With α=β=0, the Bochner Hodge / Ricci projection layers don't
    contribute to the output; their gradients are zero through this
    pass."""
    torch.manual_seed(0)
    bb = RicciStimBackbone(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=1, max_depth=1,
        d_hidden=8,
        bochner_alpha=0.0, bochner_beta=0.0,
    )
    img = torch.randn(1, 12, 12)
    features, _ = bb(img)
    loss = features.pow(2).sum()
    loss.backward()
    # hodge_proj / ricci_proj receive zero gradient (gated by α/β=0).
    for layer in (bb.walk_layer, bb.poly_layer, bb.tri_layer):
        assert (
            layer.hodge_proj.weight.grad is None
            or layer.hodge_proj.weight.grad.abs().sum().item() == 0
        )
        assert (
            layer.ricci_proj.weight.grad is None
            or layer.ricci_proj.weight.grad.abs().sum().item() == 0
        )


def test_uniform_image_no_nan():
    bb = RicciStimBackbone(
        image_h=16, image_w=16, patch_size_initial=4,
        patch_size_min=2, max_depth=2,
        d_hidden=8,
    ).eval()
    img = torch.full((1, 16, 16), 0.5)
    features, tree = bb(img)
    assert not torch.isnan(features).any()
    assert features.shape == (tree.n_anchors, 8)


def test_param_count_excludes_head():
    """Backbone params should NOT include the classifier / detector
    heads — that's the whole point of the consolidation."""
    bb = RicciStimBackbone(
        image_h=28, image_w=28, patch_size_initial=4,
        patch_size_min=1, max_depth=2,
        d_hidden=16,
    )
    n_bb = bb.n_parameters()
    # patch_encoder + walk + poly + tri = 272 + 2114 + 1090 + 1090 = 4566
    assert n_bb == 4566


def test_ablate_structural_branches_ignores_branch_params():
    """With ``ablate_structural_branches=True`` the walk/poly/tri branches are
    zeroed, so the head input must be invariant to the branch sub-modules'
    parameters (it depends only on the encoder path via the highway skip). This
    is the Step-1 diagnostic toggle (plan 2026-06-16-soma-structural-highway):
    a regression that would fail if the zeroing were dropped while the branches
    produce a non-zero signal."""
    torch.manual_seed(0)
    bb = RicciStimBackbone(
        image_h=24, image_w=24, patch_size_initial=8,
        patch_size_min=2, max_depth=2, max_anchors=64,
        d_hidden=8, bochner_alpha=0.1, bochner_beta=0.1,
        use_highway=True, ablate_structural_branches=True,
    ).eval()
    # A structured image so the stim-graph (and thus the branches, pre-ablation)
    # is non-trivial — the test would be vacuous on a near-empty graph.
    img = torch.zeros(1, 24, 24)
    img[0, 4:12, 4:12] = 1.0
    img[0, 14:20, 14:20] = 1.0
    with torch.no_grad():
        h0, _ = bb(img)
        n_touched = 0
        for name, p in bb.named_parameters():
            if any(k in name for k in ("walk_layer", "poly_layer", "tri_layer")):
                p.add_(torch.randn_like(p))
                n_touched += 1
        h1, _ = bb(img)
    assert n_touched > 0, "no structural-branch params matched — test is vacuous"
    assert torch.allclose(h0, h1), "ablated output changed when branch params did"


def test_ablate_default_off_preserves_behaviour():
    """The flag defaults off, so a backbone built without it is byte-identical to
    one with ``ablate_structural_branches=False`` (contract preservation)."""
    kw = dict(image_h=16, image_w=16, patch_size_initial=4, patch_size_min=1,
              max_depth=2, max_anchors=64, d_hidden=8)
    a = RicciStimBackbone(**kw).eval()
    b = RicciStimBackbone(ablate_structural_branches=False, **kw).eval()
    b.load_state_dict(a.state_dict())
    torch.manual_seed(3)
    img = torch.randn(1, 16, 16)
    with torch.no_grad():
        ha, _ = a(img)
        hb, _ = b(img)
    assert torch.equal(ha, hb)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
