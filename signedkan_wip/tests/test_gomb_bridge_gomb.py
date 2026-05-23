"""Unit tests for ``GombBridgeGomb`` — two Gömb cortices joined
through an HSIKAN bridge."""
from __future__ import annotations

import numpy as np
import torch

from signedkan_wip.src.hymeko_gomb.cascade import (
    GombBridgeGomb, GombConfig, HymeKoGomb,
)


def _cfg(n_nodes=12, bridge_n_layers=2):
    return GombConfig(
        n_nodes=n_nodes,
        d_embed=8,
        d_outer=4, M_outer=2,
        d_middle=4,
        d_core=4, n_tiers=2,
        cycle_k=3, middle_grid=5,
        outer_hsikan_n_layers=bridge_n_layers,
        outer_hsikan_inner_skip="highway",
        outer_hsikan_jk_mode="last",
        outer_hsikan_share_weights=False,
    )


def _inputs(n_nodes=12, n_cycles=8, k=3, n_query=4):
    rng = np.random.default_rng(0)
    cycles_np = rng.integers(0, n_nodes, size=(n_cycles, k))
    cycles_np.sort(axis=1)
    for i in range(n_cycles):
        while len(set(cycles_np[i])) < k:
            cycles_np[i] = sorted(rng.integers(0, n_nodes, size=k))
    signs_np = rng.choice([-1, 1], size=(n_cycles, k))
    queries = [(int(cycles_np[i, 0]), int(cycles_np[i, 1]))
                for i in range(min(n_query, n_cycles))]
    while len(queries) < n_query:
        queries.append((int(cycles_np[0, 0]), int(cycles_np[0, 1])))
    cycles = torch.from_numpy(cycles_np).long()
    signs = torch.from_numpy(signs_np).float()
    tier_of = torch.zeros(n_nodes, dtype=torch.long)
    edges = torch.tensor(queries, dtype=torch.long)
    return cycles, signs, tier_of, edges


def test_requires_bridge_depth():
    """outer_hsikan_n_layers=0 (the marker for no bridge) must
    raise — caller should use HymeKoGomb directly."""
    cfg = _cfg(bridge_n_layers=0)
    try:
        GombBridgeGomb(cfg)
    except ValueError as e:
        assert "GombBridgeGomb" in str(e) or "bridge" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on bridge depth=0")


def test_forward_shape():
    cfg = _cfg(bridge_n_layers=2)
    model = GombBridgeGomb(cfg)
    cycles, signs, tier_of, edges = _inputs()
    out = model(cycles, signs, tier_of, edges)
    assert out.shape == (edges.shape[0],)


def test_encode_per_vertex_on_hymekogomb():
    """The new ``HymeKoGomb.encode_per_vertex`` method returns
    the (N, d_for_core) feature that the CPML would consume."""
    cfg = _cfg()
    g = HymeKoGomb(cfg)
    cycles, signs, *_ = _inputs()
    x = g.encode_per_vertex(cycles, signs)
    expected_dim = (cfg.d_embed
                     + cfg.M_outer * cfg.d_outer
                     + cfg.d_middle)
    assert x.shape == (cfg.n_nodes, expected_dim)


def test_gate_starts_low():
    """Bridge gate inits at sigmoid(-3) ≈ 0.05 per channel so the
    composite starts ≈ plain Gömb_2 (bridge contribution is small)."""
    cfg = _cfg()
    model = GombBridgeGomb(cfg)
    g = torch.sigmoid(model.bridge_gate_logit)
    assert g.max().item() < 0.10
    assert g.min().item() > 0.01


def test_backward_reaches_all_four_param_groups():
    """The composite must train end-to-end: gradient must reach
    (a) Gömb_1 outer/middle, (b) bridge HSIKAN, (c) Gömb_2 base
    embedding + gate, (d) Gömb_2 shells."""
    torch.manual_seed(0)
    cfg = _cfg(bridge_n_layers=2)
    model = GombBridgeGomb(cfg)
    cycles, signs, tier_of, edges = _inputs()
    out = model(cycles, signs, tier_of, edges)
    out.sum().backward()
    assert model.g1_outer.pre_projs[0].weight.grad.abs().sum().item() > 0
    assert model.g1_middle.pre_proj.weight.grad.abs().sum().item() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.bridge_hsikan.parameters()
    ), "bridge HSIKAN got no gradient"
    assert model.bridge_gate_logit.grad is not None
    assert model.bridge_gate_logit.grad.abs().sum().item() > 0
    assert model.g2_base_node_embed.weight.grad.abs().sum().item() > 0
    assert model.g2_outer.pre_projs[0].weight.grad.abs().sum().item() > 0
    assert model.g2_middle.pre_proj.weight.grad.abs().sum().item() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.g2_core.parameters()
    ), "Gömb_2 inner CPML got no gradient"


def test_param_count_strictly_more_than_single_gomb():
    """Two Gömbs + bridge has strictly more params than a single
    Gömb (sanity that the second cortex + bridge were actually
    allocated; the exact ratio depends on n_nodes — at small
    n_nodes the embedding term shrinks vs the CPML routing
    term, so the ratio is dataset-dependent, not 2× universally)."""
    cfg = _cfg(bridge_n_layers=2)
    g_single = HymeKoGomb(cfg)
    g_bridge = GombBridgeGomb(cfg)
    assert g_bridge.n_params() > g_single.n_params() * 1.2, (
        f"bridge {g_bridge.n_params()} not enough larger than "
        f"single {g_single.n_params()}"
    )


def test_node_embed_exposes_g2_base():
    """``model.node_embed`` exposes Gömb_2's base embedding (the
    one that drives the downstream cascade)."""
    cfg = _cfg()
    model = GombBridgeGomb(cfg)
    assert model.node_embed is model.g2_base_node_embed


def test_deeper_bridge_runs():
    """Bridge depth = 4 forward succeeds."""
    cfg = _cfg(bridge_n_layers=4)
    model = GombBridgeGomb(cfg)
    cycles, signs, tier_of, edges = _inputs()
    out = model(cycles, signs, tier_of, edges)
    assert out.shape == (edges.shape[0],)
