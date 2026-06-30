"""Tests for the Chebyshev-CR cell × readout axes in GömbSoma vision.

Pins the two axes added to discriminate the MNIST 0.52 ceiling:

  * cell — the walk-message / patch nonlinearity (GELU vs the HSiKAN CR /
    Chebyshev-CR spline);
  * readout — mean-pool (position-blind) vs flatten (position-preserving).

Both are Strategy modules built at construction (§6.5 #8), not forward-time
flags. Reuses ``_build_inputs`` from ``test_gomb_soma_hg_conv`` (§6.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from signed_kan import CatmullRomActivation, ChebyshevCRActivation

from signedkan_wip.src.hymeko_gomb.soma.hg_conv import (
    HypergraphConvConfig,
    MessageActivation,
)
from signedkan_wip.src.hymeko_gomb.soma.walk_layer import (
    WalkConvLayer,
    _build_message_activation,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.walk_conv_classifier import (
    PatchEncoder,
    Readout,
    WalkConvImageClassifier,
    _build_readout,
)

from test_gomb_soma_hg_conv import _build_inputs


# ---------------------------------------------------------------------
# Message activation (cell axis)
# ---------------------------------------------------------------------


def test_message_activation_builds_correct_module() -> None:
    assert isinstance(_build_message_activation(MessageActivation.GELU, 8), nn.GELU)
    assert isinstance(
        _build_message_activation(MessageActivation.CR, 8), CatmullRomActivation)
    assert isinstance(
        _build_message_activation(MessageActivation.CHEBY_CR, 8), ChebyshevCRActivation)


def test_default_activation_is_gelu_module() -> None:
    """Regression: the default walk-conv activation stays GELU — guards against
    the default silently becoming a spline."""
    layer = WalkConvLayer(HypergraphConvConfig(
        in_features=4, out_features=4, k_arity=3))
    assert isinstance(layer.activation, nn.GELU)


def test_cheby_cr_walk_layer_forward_is_finite_and_shaped() -> None:
    cfg = HypergraphConvConfig(
        in_features=4, out_features=4, k_arity=3,
        message_activation=MessageActivation.CHEBY_CR)
    layer = WalkConvLayer(cfg)
    layer.eval()
    x, prims, signs, M_v = _build_inputs(n_nodes=8, k=3, n_prim=10, d_in=4, seed=2)
    y = layer(x, prims, signs, M_v)
    assert y.shape == (8, 4)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------
# Readout axis
# ---------------------------------------------------------------------


def test_flatten_readout_preserves_position_meanpool_does_not() -> None:
    """The property that operationalises H2: flatten is permutation-sensitive
    (keeps which patch held what); mean-pool is permutation-invariant."""
    x = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
    perm = torch.tensor([2, 0, 1, 5, 4, 3])

    flat = _build_readout(Readout.FLATTEN, 2, 3, 4)        # 6 patches
    mean = _build_readout(Readout.MEAN_POOL, 2, 3, 4)
    assert not torch.allclose(flat(x), flat(x[perm]))      # position-sensitive
    assert torch.allclose(mean(x), mean(x[perm]), atol=1e-6)  # position-blind


def test_readout_out_dims() -> None:
    assert _build_readout(Readout.MEAN_POOL, 7, 7, 16).out_dim == 16
    assert _build_readout(Readout.FLATTEN, 7, 7, 16).out_dim == 49 * 16


def test_flatten_classifier_head_width_matches_patches() -> None:
    flat = WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10, readout=Readout.FLATTEN)
    assert flat.head.in_features == flat.builder.n_patches * 16


# ---------------------------------------------------------------------
# 2x2 integration
# ---------------------------------------------------------------------


def _arm(cheby: bool, flat: bool) -> WalkConvImageClassifier:
    return WalkConvImageClassifier(
        image_h=28, image_w=28, patch_size=4, in_channels=1,
        d_hidden=16, n_classes=10,
        message_activation=(MessageActivation.CHEBY_CR if cheby
                            else MessageActivation.GELU),
        patch_encoder=(PatchEncoder.CHEBY_CR if cheby else PatchEncoder.LINEAR),
        readout=(Readout.FLATTEN if flat else Readout.MEAN_POOL),
    )


def test_all_four_arms_forward() -> None:
    images = torch.randn(2, 1, 28, 28)
    for cheby in (False, True):
        for flat in (False, True):
            logits = _arm(cheby, flat)(images)
            assert logits.shape == (2, 10)


def test_cheby_cell_adds_params_over_baseline() -> None:
    """The Chebyshev-CR cell adds spline coefficients (patch + message), so it
    is heavier than the GELU baseline at the same readout."""
    base = _arm(cheby=False, flat=False)
    cheby = _arm(cheby=True, flat=False)
    assert cheby.n_parameters() > base.n_parameters()


def test_cheby_classifier_is_trainable() -> None:
    """One backward populates the patch Chebyshev-CR coefficients — guards
    against a detached HSiKAN cell."""
    model = _arm(cheby=True, flat=False)
    model(torch.randn(2, 1, 28, 28)).sum().backward()
    # patch_embed is Sequential(Linear, ChebyshevCRActivation); the spline coef
    # is the last module's parameter.
    cheby_coef = model.patch_embed[-1].coef
    assert cheby_coef.grad is not None and cheby_coef.grad.abs().sum() > 0.0
