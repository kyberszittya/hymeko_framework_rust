"""Tests for arc-weight-modulated Catmull-Rom highway gate.

Pins:
- ``SignedNTuple.arc_weights`` extension is backward-compat.
- ``inner_skip="cr_highway"`` forward path produces the right shape.
- With ``arc_weights=None``, the new mode is independent of the
  ``gate_W_arc`` parameter (it never reads arc weights).
- With ``arc_weights`` provided but ``gate_W_arc=0`` (init state),
  the output matches the no-arc-weight forward exactly.
- Backward reaches BOTH ``gate_coef`` and ``gate_W_arc`` parameters.
- Existing ``inner_skip="highway"`` still works untouched.
"""
from __future__ import annotations

import numpy as np
import torch

from signedkan_wip.src.core.n_tuples import SignedNTuple
from signedkan_wip.src.core.signedkan import (
    SignedKANConfig, SignedKANLayer,
    MultiLayerSignedKANConfig, MultiLayerSignedKAN,
    SignedKAN,
)


def _toy_triads(n_nodes=6, n_triads=5):
    rng = np.random.default_rng(0)
    triad_v_np = rng.integers(0, n_nodes, size=(n_triads, 3))
    triad_v_np.sort(axis=1)
    for i in range(n_triads):
        while len(set(triad_v_np[i])) < 3:
            triad_v_np[i] = sorted(rng.integers(0, n_nodes, size=3))
    triad_sigma = torch.tensor(
        rng.choice([-1, 1], size=(n_triads, 3)), dtype=torch.long,
    )
    return torch.from_numpy(triad_v_np).long(), triad_sigma


def _layer_cfg(d=4, k=3, inner_skip="highway", grid=5):
    return SignedKANConfig(
        n_nodes=6, hidden_dim=d, grid=grid, k=k,
        use_minus_branch=True, init_scale=0.1,
        spline_kind="catmull_rom",
        inner_skip=inner_skip, outer_skip="none",
    )


def test_signed_ntuple_arc_weights_optional():
    """``SignedNTuple.arc_weights`` defaults to None — existing
    code paths that don't pass it stay backward compatible."""
    t = SignedNTuple(v=(0, 1, 2), sigma=(1, -1, 1),
                      edge_signs=(1, -1, 1), balanced=True, arity=3)
    assert t.arc_weights is None
    t2 = SignedNTuple(v=(0, 1, 2), sigma=(1, -1, 1),
                       edge_signs=(1, -1, 1), balanced=True, arity=3,
                       arc_weights=(0.8, -0.5, 0.3))
    assert t2.arc_weights == (0.8, -0.5, 0.3)


def test_cr_highway_forward_shape():
    """``inner_skip="cr_highway"`` produces the same output shape
    as ``"highway"`` (one-layer SignedKAN)."""
    cfg = _layer_cfg(d=4, inner_skip="cr_highway")
    layer = SignedKANLayer(cfg)
    triad_v, triad_sigma = _toy_triads()
    h_v = torch.randn(6, 4)
    out = layer(h_v, triad_v, triad_sigma)
    assert out.shape == (triad_v.shape[0], 4)


def test_cr_highway_accepts_arc_weights():
    """``cr_highway`` forward accepts arc_weights without error."""
    cfg = _layer_cfg(d=4, inner_skip="cr_highway")
    layer = SignedKANLayer(cfg)
    triad_v, triad_sigma = _toy_triads()
    h_v = torch.randn(6, 4)
    arc_w = torch.tensor(
        np.random.default_rng(0).uniform(-1, 1, size=triad_v.shape),
        dtype=torch.float32,
    )
    out = layer(h_v, triad_v, triad_sigma, arc_weights=arc_w)
    assert out.shape == (triad_v.shape[0], 4)


def test_cr_highway_gate_W_arc_zero_init_matches_no_arc():
    """With ``gate_W_arc`` init at zero, providing arc_weights should
    not change the output vs. omitting them — the perturbation
    term is exactly zero."""
    torch.manual_seed(42)
    cfg = _layer_cfg(d=4, inner_skip="cr_highway")
    layer = SignedKANLayer(cfg)
    # gate_W_arc is init at zero by construction; confirm.
    assert torch.allclose(layer.gate_W_arc, torch.zeros_like(layer.gate_W_arc))
    triad_v, triad_sigma = _toy_triads()
    h_v = torch.randn(6, 4)
    arc_w = torch.rand(triad_v.shape) * 2 - 1  # [-1, 1]
    out_no_arc = layer(h_v, triad_v, triad_sigma)
    out_with_arc = layer(h_v, triad_v, triad_sigma, arc_weights=arc_w)
    max_diff = (out_no_arc - out_with_arc).abs().max().item()
    assert max_diff < 1e-6, (
        f"with W_arc=0 the arc-weight path must be a no-op; "
        f"max diff = {max_diff:.2e}"
    )


def test_cr_highway_lifts_with_W_arc_nonzero():
    """Once ``gate_W_arc`` is non-zero, the output DOES depend on
    arc weights — pinning that the perturbation path is wired."""
    torch.manual_seed(42)
    cfg = _layer_cfg(d=4, inner_skip="cr_highway")
    layer = SignedKANLayer(cfg)
    with torch.no_grad():
        layer.gate_W_arc.normal_(std=0.5)
    triad_v, triad_sigma = _toy_triads()
    h_v = torch.randn(6, 4)
    arc_w = torch.rand(triad_v.shape) * 2 - 1
    out_no_arc = layer(h_v, triad_v, triad_sigma)
    out_with_arc = layer(h_v, triad_v, triad_sigma, arc_weights=arc_w)
    max_diff = (out_no_arc - out_with_arc).abs().max().item()
    assert max_diff > 1e-4, (
        f"with W_arc non-zero, arc weights must perturb the output; "
        f"max diff = {max_diff:.2e}"
    )


def test_cr_highway_backward_reaches_gate_W_arc():
    """Gradients must flow to ``gate_W_arc`` when arc_weights are
    provided."""
    torch.manual_seed(0)
    cfg = _layer_cfg(d=4, inner_skip="cr_highway")
    layer = SignedKANLayer(cfg)
    # Pull W_arc away from zero so the gradient is meaningful.
    with torch.no_grad():
        layer.gate_W_arc.uniform_(-0.1, 0.1)
    triad_v, triad_sigma = _toy_triads()
    h_v = torch.randn(6, 4, requires_grad=True)
    arc_w = torch.rand(triad_v.shape) * 2 - 1
    out = layer(h_v, triad_v, triad_sigma, arc_weights=arc_w)
    out.sum().backward()
    g = layer.gate_W_arc.grad
    assert g is not None and g.abs().sum().item() > 0, \
        "gate_W_arc got no gradient"
    g_base = layer.gate_coef.grad
    assert g_base is not None and g_base.abs().sum().item() > 0, \
        "gate_coef got no gradient"


def test_highway_mode_untouched_by_cr_highway_additions():
    """Existing ``inner_skip="highway"`` mode still works exactly."""
    torch.manual_seed(0)
    cfg_hw = _layer_cfg(d=4, inner_skip="highway")
    layer_hw = SignedKANLayer(cfg_hw)
    triad_v, triad_sigma = _toy_triads()
    h_v = torch.randn(6, 4)
    # arc_weights kwarg is accepted but ignored under "highway".
    out_a = layer_hw(h_v, triad_v, triad_sigma)
    out_b = layer_hw(h_v, triad_v, triad_sigma,
                       arc_weights=torch.rand(triad_v.shape) * 2 - 1)
    assert torch.allclose(out_a, out_b), \
        "arc_weights must be ignored under inner_skip='highway'"


def test_multi_layer_cr_highway_runs_end_to_end():
    """`MultiLayerSignedKAN` with cr_highway runs end-to-end through
    ``encode_triads``, including the M_vt pooling step."""
    from signedkan_wip.src.core.signedkan import build_vertex_triad_incidence
    cfg = MultiLayerSignedKANConfig(
        n_nodes=6, n_layers=2, hidden_dim=4, grid=5, k=3,
        spline_kinds=["catmull_rom"] * 2,
        init_scale=0.05,
        pool_mode="sum",
        jk_mode="concat",
        layer_norm_between=True,
        share_weights=True,
        inner_skip="cr_highway",
        outer_skip="none",
        use_residual=True,
    )
    model = MultiLayerSignedKAN(cfg)
    triad_v, triad_sigma = _toy_triads()
    M_vt = build_vertex_triad_incidence(
        triad_v.numpy(), 6, torch.device("cpu"), mode="sum",
    )
    arc_w = torch.rand(triad_v.shape) * 2 - 1
    out = model.encode_triads(triad_v, triad_sigma, M_vt,
                                arc_weights=arc_w)
    # jk_mode="concat" → (T, L*d) = (5, 8)
    assert out.shape == (triad_v.shape[0], 8)


def test_single_layer_signedkan_wrapper_accepts_arc_weights():
    """Single-layer ``SignedKAN`` wrapper threads arc_weights through."""
    torch.manual_seed(0)
    cfg = SignedKANConfig(
        n_nodes=6, hidden_dim=4, grid=5, k=3,
        use_minus_branch=True, init_scale=0.1,
        spline_kind="catmull_rom",
        inner_skip="cr_highway", outer_skip="none",
    )
    model = SignedKAN(cfg)
    triad_v, triad_sigma = _toy_triads()
    arc_w = torch.rand(triad_v.shape) * 2 - 1
    out = model.encode_triads(triad_v, triad_sigma, arc_weights=arc_w)
    assert out.shape == (triad_v.shape[0], 4)
