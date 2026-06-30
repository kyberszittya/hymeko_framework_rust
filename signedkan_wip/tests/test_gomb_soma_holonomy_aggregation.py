"""Tests for HOLONOMY (sign-as-connection) aggregation in GömbSoma.

The 2026-06-15 MNIST walk-vision falsification tested *sign-as-routing*
(dual banks + sign-blind sum-pool). This module pins the new
*sign-as-connection* operator added to ``HypergraphConv``:

    HOLONOMY:  y = M_v @ (σ ⊙ messages)

where σ is each primitive's σ-product (its Z₂ holonomy). The sign enters
as a multiplicative parallel-transport factor — the Soma-side realisation
of the signed operator that ``hymeko_rl/structural_actor.py`` reproduces as
``Bᴸ`` and that a sign-blind sum cannot.

Reuses the ``MeanConv`` fixture + ``_build_inputs`` helper from
``test_gomb_soma_hg_conv`` (no duplicate scaffolding — CLAUDE.md §6.1).
"""
from __future__ import annotations

import torch

from signedkan_wip.src.hymeko_gomb.soma.hg_conv import (
    Aggregation,
    HypergraphConvConfig,
)
from signedkan_wip.src.hymeko_gomb.soma.vision import WalkConvImageClassifier

# Reuse the ABC fixtures (pytest prepend-import puts this dir on sys.path).
from test_gomb_soma_hg_conv import MeanConv, _build_inputs


# ---------------------------------------------------------------------
# Aggregation contract
# ---------------------------------------------------------------------


def _sign_blind_cfg(aggregation: Aggregation) -> HypergraphConvConfig:
    """MeanConv config with sign-blind messages, so the only place sign can
    enter is the aggregation step (isolates the holonomy effect)."""
    return HypergraphConvConfig(
        in_features=4, out_features=4, k_arity=3,
        use_sign_branching=False, aggregation=aggregation,
    )


def test_holonomy_equals_manual_signed_pool() -> None:
    """HOLONOMY aggregate == M_v @ (σ ⊙ messages), exactly. This is the
    signed-operator mechanism at the Soma aggregation layer."""
    layer = MeanConv(_sign_blind_cfg(Aggregation.HOLONOMY))
    layer.eval()
    x, prims, signs, M_v = _build_inputs(n_nodes=8, k=3, n_prim=10, d_in=4, seed=3)

    y = layer(x, prims, signs, M_v)

    msgs = layer._forward_messages(x, prims, signs)
    expected = torch.sparse.mm(M_v, signs.to(msgs.dtype).unsqueeze(-1) * msgs)
    assert torch.allclose(y, expected, atol=1e-6), (
        f"holonomy pool disagrees with manual σ⊙m pool; "
        f"max diff = {(y - expected).abs().max().item():.2e}"
    )


def test_holonomy_differs_from_sum_when_signs_mixed() -> None:
    """Regression: HOLONOMY ≠ SUM on a sign-blind message with mixed signs.
    Fails against the prior implementation, which had no holonomy branch."""
    x, prims, signs, M_v = _build_inputs(n_nodes=8, k=3, n_prim=10, d_in=4, seed=5)
    assert (signs == 1).any() and (signs == -1).any(), "fixture needs mixed signs"

    sum_layer = MeanConv(_sign_blind_cfg(Aggregation.SUM))
    hol_layer = MeanConv(_sign_blind_cfg(Aggregation.HOLONOMY))
    # Share weights so the only difference is the aggregation mode.
    hol_layer.load_state_dict(sum_layer.state_dict())
    sum_layer.eval()
    hol_layer.eval()

    y_sum = sum_layer(x, prims, signs, M_v)
    y_hol = hol_layer(x, prims, signs, M_v)
    assert not torch.allclose(y_sum, y_hol, atol=1e-4), (
        "holonomy collapsed to sum — sign is not entering the pool"
    )


def test_flipping_a_sign_negates_its_holonomy_contribution() -> None:
    """A single primitive's pooled contribution flips sign when its σ flips.
    Single-primitive construction makes the effect exact."""
    layer = MeanConv(_sign_blind_cfg(Aggregation.HOLONOMY))
    layer.eval()
    # One primitive over a 3-vertex graph; M_v scatters its message to all 3.
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(11))
    prims = torch.tensor([[0, 1, 2]], dtype=torch.long)
    idx = torch.tensor([[0, 1, 2], [0, 0, 0]])
    M_v = torch.sparse_coo_tensor(
        idx, torch.ones(3), (3, 1),
    ).coalesce()

    y_pos = layer(x, prims, torch.tensor([1], dtype=torch.int64), M_v)
    y_neg = layer(x, prims, torch.tensor([-1], dtype=torch.int64), M_v)
    assert torch.allclose(y_pos, -y_neg, atol=1e-6), (
        "flipping the lone primitive's sign should negate every pooled row"
    )


def test_holonomy_preserves_sparse_invariant() -> None:
    """HOLONOMY keeps the no-dense-|V|×|P| invariant: it is an elementwise
    scale then the *same* torch.sparse.mm; equals the manually-densified path."""
    layer = MeanConv(_sign_blind_cfg(Aggregation.HOLONOMY))
    layer.eval()
    x, prims, signs, M_v = _build_inputs(n_nodes=8, k=3, n_prim=10, d_in=4, seed=7)
    assert M_v.is_sparse, "fixture must produce a sparse M_v"

    y_sparse = layer(x, prims, signs, M_v)
    msgs = layer._forward_messages(x, prims, signs)
    y_dense = M_v.to_dense() @ (signs.to(msgs.dtype).unsqueeze(-1) * msgs)
    assert torch.allclose(y_sparse, y_dense, atol=1e-6)


def test_default_aggregation_is_sum() -> None:
    """Backward-compat guard: the default config aggregation stays SUM, so
    every existing call site is unchanged in behaviour."""
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=3)
    assert cfg.aggregation is Aggregation.SUM


# ---------------------------------------------------------------------
# Classifier integration
# ---------------------------------------------------------------------


def test_holonomy_classifier_forward_shape() -> None:
    """The holonomy/single-bank WalkConv classifier runs end-to-end and
    returns (B, n_classes) on MNIST-shaped input."""
    model = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10,
        use_sign_branching=False, aggregation=Aggregation.HOLONOMY,
    )
    model.eval()
    images = torch.randn(3, 1, 28, 28)
    logits = model(images)
    assert logits.shape == (3, 10)


def test_holonomy_single_bank_has_fewer_params_than_routing() -> None:
    """Sign-as-connection uses one message bank; sign-as-routing uses two.
    The holonomy model must therefore be the lighter of the A/B pair."""
    routing = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10,
        use_sign_branching=True, aggregation=Aggregation.SUM,
    )
    holonomy = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10,
        use_sign_branching=False, aggregation=Aggregation.HOLONOMY,
    )
    assert holonomy.n_parameters() < routing.n_parameters()


def test_holonomy_classifier_is_trainable() -> None:
    """One backward pass populates a gradient on the walk-conv bank — guards
    against a dead holonomy branch detaching the message from the graph."""
    model = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10,
        use_sign_branching=False, aggregation=Aggregation.HOLONOMY,
    )
    logits = model(torch.randn(2, 1, 28, 28))
    logits.sum().backward()
    assert model.walk_conv.W.grad is not None
    assert model.walk_conv.W.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------
# Bochner delegation (the one _aggregate override)
# ---------------------------------------------------------------------


def test_bochner_delegates_holonomy_aggregation() -> None:
    """The Bochner wrapper forwards primitive_signs to the inner aggregator,
    so a holonomy-configured inner still pools with sign through the wrapper."""
    from signedkan_wip.src.hymeko_gomb.soma.hg_conv_bochner import (
        BochnerHypergraphConv,
    )

    inner = MeanConv(_sign_blind_cfg(Aggregation.HOLONOMY))
    # No Hodge / Ricci terms: the wrapper is a pure aggregation passthrough.
    wrapped = BochnerHypergraphConv(inner)
    wrapped.eval()
    x, prims, signs, M_v = _build_inputs(n_nodes=8, k=3, n_prim=10, d_in=4, seed=9)

    msgs = wrapped._forward_messages(x, prims, signs)
    y = wrapped._aggregate(msgs, signs, M_v)
    expected = torch.sparse.mm(M_v, signs.to(msgs.dtype).unsqueeze(-1) * msgs)
    assert torch.allclose(y, expected, atol=1e-6)
