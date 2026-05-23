"""Unit tests for the stacked Gömb-HSIKAN backbone (Phase 2026-05-20).

Pins:
- ``GombConfig.middle_n_layers <= 1`` reproduces the existing
  ``MiddleHSiKAN`` path exactly (dispatch + same output shape).
- ``middle_n_layers >= 2`` swaps in ``StackedMiddleHSiKAN``.
- Forward shapes are correct at depths 2 and 4 across jk modes.
- ``initial_h_v`` on ``MultiLayerSignedKAN.encode_triads``
  bypasses the built-in node embedding.
- Backward reaches every layer of the stacked middle.
- The Gömb fuzzy signature still works on the stacked variant.
"""
from __future__ import annotations

import numpy as np
import torch

from signedkan_wip.src.hymeko_gomb.cascade import HymeKoGomb, GombConfig
from signedkan_wip.src.hymeko_gomb.shells import (
    MiddleHSiKAN, StackedMiddleHSiKAN,
)
from signedkan_wip.src.core.signedkan import (
    MultiLayerSignedKAN, MultiLayerSignedKANConfig,
    build_vertex_triad_incidence,
)
from signedkan_wip.src.interpret import extract_gomb_signature


def _toy_gomb_cfg(n_nodes=12, middle_n_layers=1, middle_jk_mode="last",
                    middle_inner_skip="highway"):
    return GombConfig(
        n_nodes=n_nodes,
        d_embed=4,
        d_outer=4, M_outer=2,
        d_middle=4,
        d_core=4, n_tiers=2,
        cycle_k=3, middle_grid=5,
        middle_n_layers=middle_n_layers,
        middle_jk_mode=middle_jk_mode,
        middle_inner_skip=middle_inner_skip,
    )


def _toy_inputs(n_nodes=12, n_cycles=8, cycle_k=3, n_query=4,
                 device=torch.device("cpu")):
    rng = np.random.default_rng(0)
    cycles_np = rng.integers(0, n_nodes, size=(n_cycles, cycle_k))
    cycles_np.sort(axis=1)
    for i in range(n_cycles):
        while len(set(cycles_np[i])) < cycle_k:
            cycles_np[i] = sorted(rng.integers(0, n_nodes, size=cycle_k))
    signs_np = rng.choice([-1, 1], size=(n_cycles, cycle_k))
    queries = [(int(cycles_np[i, 0]), int(cycles_np[i, 1]))
               for i in range(min(n_query, n_cycles))]
    while len(queries) < n_query:
        queries.append((int(cycles_np[0, 0]), int(cycles_np[0, 1])))

    cycles = torch.from_numpy(cycles_np).long().to(device)
    signs = torch.from_numpy(signs_np).float().to(device)
    tier_of = torch.zeros(n_nodes, dtype=torch.long, device=device)
    edges_to_score = torch.tensor(queries, dtype=torch.long,
                                    device=device)
    return cycles, signs, tier_of, edges_to_score, cycles_np, signs_np


def test_middle_n_layers_1_uses_existing_class():
    """Backward compat: ``middle_n_layers=1`` (default) dispatches
    to the existing ``MiddleHSiKAN``, NOT the stacked variant."""
    cfg = _toy_gomb_cfg(middle_n_layers=1)
    model = HymeKoGomb(cfg)
    assert isinstance(model.middle, MiddleHSiKAN)
    assert not isinstance(model.middle, StackedMiddleHSiKAN)


def test_middle_n_layers_2_uses_stacked():
    """``middle_n_layers=2`` dispatches to ``StackedMiddleHSiKAN``."""
    cfg = _toy_gomb_cfg(middle_n_layers=2)
    model = HymeKoGomb(cfg)
    assert isinstance(model.middle, StackedMiddleHSiKAN)


def test_stacked_forward_shape_jk_last():
    """At ``jk_mode='last'`` the stacked output dim matches
    ``d_middle`` (= the original single-layer dim)."""
    cfg = _toy_gomb_cfg(middle_n_layers=4, middle_jk_mode="last")
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, *_ = _toy_inputs()
    out = model(cycles, signs, tier_of, edges)
    assert out.shape == (edges.shape[0],)
    assert model.middle.d_out == cfg.d_middle


def test_stacked_forward_shape_jk_concat():
    """At ``jk_mode='concat'`` the stacked output dim widens to
    ``L * d_middle``."""
    cfg = _toy_gomb_cfg(middle_n_layers=4, middle_jk_mode="concat")
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, *_ = _toy_inputs()
    out = model(cycles, signs, tier_of, edges)
    assert out.shape == (edges.shape[0],)
    assert model.middle.d_out == cfg.d_middle * cfg.middle_n_layers


def test_initial_h_v_overrides_node_embed():
    """``MultiLayerSignedKAN.encode_triads(initial_h_v=...)`` uses
    the passed tensor instead of ``self.node_embed.weight``."""
    torch.manual_seed(0)
    mcfg = MultiLayerSignedKANConfig(
        n_nodes=8, n_layers=2, hidden_dim=4, grid=5, k=3,
        spline_kinds=["catmull_rom"] * 2, init_scale=0.05,
        pool_mode="sum", jk_mode="last",
        share_weights=False, inner_skip="highway",
        outer_skip="none", use_residual=True,
    )
    stack = MultiLayerSignedKAN(mcfg)
    cycles_np = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5]],
                          dtype=np.int64)
    cycles = torch.from_numpy(cycles_np).long()
    signs = torch.ones_like(cycles)
    M_vt = build_vertex_triad_incidence(
        cycles_np, 8, torch.device("cpu"), mode="sum",
    )
    # Default: uses node_embed.
    out_default = stack.encode_triads(cycles, signs, M_vt)
    # Override with a different h_v.
    custom = torch.full((8, 4), 0.5)
    out_custom = stack.encode_triads(
        cycles, signs, M_vt, initial_h_v=custom,
    )
    # Outputs must differ because the initial vertex features differ.
    diff = (out_default - out_custom).abs().max().item()
    assert diff > 1e-3, (
        f"initial_h_v must change the output; diff = {diff:.2e}"
    )


def test_backward_reaches_every_middle_layer():
    """Gradients reach every layer of the stacked middle stack."""
    cfg = _toy_gomb_cfg(middle_n_layers=3, middle_jk_mode="last")
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, *_ = _toy_inputs()
    out = model(cycles, signs, tier_of, edges)
    loss = out.sum()
    loss.backward()
    middle: StackedMiddleHSiKAN = model.middle
    for layer in middle.stack.layers:
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in layer.parameters()
        )
        assert has_grad, "a stacked-middle layer got no gradient"


def test_stacked_jk_sum_runs():
    """``jk_mode='sum'`` runs end-to-end without error."""
    cfg = _toy_gomb_cfg(middle_n_layers=2, middle_jk_mode="sum")
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, *_ = _toy_inputs()
    out = model(cycles, signs, tier_of, edges)
    assert out.shape == (edges.shape[0],)


def test_gomb_signature_works_on_stacked_middle():
    """``extract_gomb_signature`` still produces a valid signature
    when the middle shell is the stacked variant — the
    middle_per_cycle capture remains populated."""
    torch.manual_seed(0)
    cfg = _toy_gomb_cfg(middle_n_layers=2)
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, *_ = _toy_inputs()
    sig = extract_gomb_signature(
        model, cycles, signs, tier_of, edges, query_idx=0,
    )
    assert "middle" in sig.shells
    assert "outer" in sig.shells
    for c in sig.contributions:
        assert c.per_shell_magnitude["middle"] > 0
        assert c.per_shell_magnitude["outer"] > 0
