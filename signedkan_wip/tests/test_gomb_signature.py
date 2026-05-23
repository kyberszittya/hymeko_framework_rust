"""Unit tests for the Gömb fuzzy-signature view."""
from __future__ import annotations

import numpy as np
import torch

from signedkan_wip.src.hymeko_gomb.cascade import HymeKoGomb, GombConfig
from signedkan_wip.src.interpret import (
    GombCycleContribution, GombFuzzySignature,
    extract_gomb_signature, plot_gomb_signature,
)


def _toy_gomb_config(n_nodes: int = 12, cycle_k: int = 3):
    """Minimal Gömb config — fast forward, deterministic init."""
    return GombConfig(
        n_nodes=n_nodes,
        d_embed=4,
        d_outer=4, M_outer=2,
        d_middle=4,
        d_core=4, n_tiers=2,
        cycle_k=cycle_k,
        middle_grid=5,
    )


def _toy_inputs(n_nodes=12, n_cycles=8, cycle_k=3, n_query=4,
                  device=torch.device("cpu")):
    """Random cycle/sign/query data on a small graph."""
    rng = np.random.default_rng(0)
    cycles_np = rng.integers(0, n_nodes, size=(n_cycles, cycle_k))
    cycles_np.sort(axis=1)
    for i in range(n_cycles):
        while len(set(cycles_np[i])) < cycle_k:
            cycles_np[i] = sorted(rng.integers(0, n_nodes, size=cycle_k))
    signs_np = rng.choice([-1, 1], size=(n_cycles, cycle_k))

    # Pick query edges that DO share endpoints with at least one cycle,
    # so the signature gets non-trivial contributions.
    queries = []
    for ci in range(min(n_query, n_cycles)):
        v = cycles_np[ci]
        queries.append((int(v[0]), int(v[1])))
    # Pad if needed.
    while len(queries) < n_query:
        queries.append((int(cycles_np[0, 0]), int(cycles_np[0, 1])))

    cycles = torch.from_numpy(cycles_np).long().to(device)
    signs = torch.from_numpy(signs_np).float().to(device)
    tier_of = torch.zeros(n_nodes, dtype=torch.long, device=device)
    edges_to_score = torch.tensor(queries, dtype=torch.long,
                                    device=device)
    return cycles, signs, tier_of, edges_to_score, cycles_np, signs_np


def test_gomb_signature_returns_correct_shape():
    """Smoke: extract_gomb_signature returns a GombFuzzySignature with
    non-empty contributions on a toy Gömb instance."""
    torch.manual_seed(0)
    cfg = _toy_gomb_config()
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, *_ = _toy_inputs()
    sig = extract_gomb_signature(
        model, cycles, signs, tier_of, edges, query_idx=0,
    )
    assert isinstance(sig, GombFuzzySignature)
    assert sig.query_idx == 0
    assert len(sig.contributions) > 0
    for c in sig.contributions:
        assert isinstance(c, GombCycleContribution)


def test_per_shell_capture_populated():
    """Both outer and middle shells leave per-cycle features in the
    capture — neither shell silently empty."""
    torch.manual_seed(0)
    cfg = _toy_gomb_config()
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, *_ = _toy_inputs()
    sig = extract_gomb_signature(
        model, cycles, signs, tier_of, edges, query_idx=0,
    )
    assert "outer" in sig.shells
    assert "middle" in sig.shells
    for c in sig.contributions:
        assert "outer" in c.per_shell_magnitude
        assert "middle" in c.per_shell_magnitude
        assert c.per_shell_magnitude["outer"] > 0
        assert c.per_shell_magnitude["middle"] > 0


def test_contribution_count_matches_incident_cycles():
    """``len(contributions)`` matches the number of cycles whose
    vertex set contains BOTH endpoints of the query edge."""
    torch.manual_seed(0)
    cfg = _toy_gomb_config()
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, cycles_np, _ = _toy_inputs()
    q = 1
    u, v = int(edges[q, 0]), int(edges[q, 1])
    expected = 0
    for ci in range(cycles_np.shape[0]):
        verts = set(int(x) for x in cycles_np[ci])
        if u in verts and v in verts:
            expected += 1
    sig = extract_gomb_signature(
        model, cycles, signs, tier_of, edges, query_idx=q,
    )
    assert len(sig.contributions) == expected, (
        f"got {len(sig.contributions)} contribs, expected {expected}"
    )


def test_sigma_prod_from_edge_signs_when_provided():
    """When ``edge_signs`` is passed, ``sigma_prod`` is the
    Cartwright-Harary product of edge signs."""
    torch.manual_seed(0)
    cfg = _toy_gomb_config()
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, _, _ = _toy_inputs()
    n_cycles = int(cycles.shape[0])
    es = np.ones((n_cycles, 3), dtype=np.int64)
    es[1::2, 0] = -1  # every other cycle: one neg → unbalanced
    sig = extract_gomb_signature(
        model, cycles, signs, tier_of, edges, query_idx=0,
        edge_signs=es,
    )
    for c in sig.contributions:
        expected = int(np.prod(np.asarray(c.edge_signs,
                                            dtype=np.int64)))
        expected = 1 if expected > 0 else -1
        assert c.sigma_prod == expected
        assert c.balanced == (c.sigma_prod == 1)


def test_arc_weights_populated_when_provided():
    """``arc_weights`` is carried into each contribution when
    passed to the extractor."""
    torch.manual_seed(0)
    cfg = _toy_gomb_config()
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, _, _ = _toy_inputs()
    n_cycles = int(cycles.shape[0])
    aw = np.tile(np.array([[0.2, 0.7, -0.4]]),
                  (n_cycles, 1)).astype(np.float32)
    sig = extract_gomb_signature(
        model, cycles, signs, tier_of, edges, query_idx=0,
        arc_weights=aw,
    )
    for c in sig.contributions:
        assert len(c.arc_weights) == 3
        assert abs(c.arc_weights[1] - 0.7) < 1e-5


def test_shell_dominance_and_cross_shell_consistency():
    """``shell_dominance`` and ``cross_shell_consistency`` return
    deterministic finite values on a fixed-init model."""
    torch.manual_seed(0)
    cfg = _toy_gomb_config()
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, *_ = _toy_inputs()
    sig = extract_gomb_signature(
        model, cycles, signs, tier_of, edges, query_idx=0,
    )
    dom = sig.shell_dominance()
    assert set(dom.keys()) == set(sig.shells)
    for v in dom.values():
        assert np.isfinite(v)
    r = sig.cross_shell_consistency()
    # Two shells, finite correlation in [-1, 1].
    assert -1.0 - 1e-6 <= r <= 1.0 + 1e-6


def test_plot_gomb_signature_smoke_no_arc():
    """``plot_gomb_signature`` returns 3 axes on a signature without
    arc weights."""
    import matplotlib
    matplotlib.use("Agg")
    torch.manual_seed(0)
    cfg = _toy_gomb_config()
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, *_ = _toy_inputs()
    sig = extract_gomb_signature(
        model, cycles, signs, tier_of, edges, query_idx=0,
    )
    axes = plot_gomb_signature(sig)
    assert len(axes) == 3


def test_plot_gomb_signature_smoke_with_arc():
    """``plot_gomb_signature`` returns 4 axes when arc weights are
    present."""
    import matplotlib
    matplotlib.use("Agg")
    torch.manual_seed(0)
    cfg = _toy_gomb_config()
    model = HymeKoGomb(cfg)
    cycles, signs, tier_of, edges, *_ = _toy_inputs()
    n_cycles = int(cycles.shape[0])
    aw = np.random.default_rng(0).uniform(-1, 1, size=(n_cycles, 3))
    sig = extract_gomb_signature(
        model, cycles, signs, tier_of, edges, query_idx=0,
        arc_weights=aw,
    )
    axes = plot_gomb_signature(sig)
    assert len(axes) == 4
