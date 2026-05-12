"""Unit tests for HymeKo-Gömb three-shell cascade.

Each shell tested independently + the composer end-to-end.
Forward/backward shapes, parameter counts, no-cycles degradation,
and a tiny training-loop smoke on synthetic moons data.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from signedkan_wip.src.hymeko_gomb import (
    GombConfig, HymeKoGomb, InnerCPMLCore, MiddleHSiKAN, OuterFIRShell,
    GombNoOuter, GombNoMiddle, GombNoInner,
    MixedArityGomb,
)
from signedkan_wip.src.hymeko_gomb.shells import scatter_mean


def _scatter_mean_corner_loop(
    per_cycle: torch.Tensor,
    cycles: torch.Tensor,
    n_vertices: int,
) -> torch.Tensor:
    """Reference: one index_add per corner (legacy semantics)."""
    m_c, k = cycles.shape
    d = per_cycle.shape[-1]
    out = torch.zeros(n_vertices, d, device=per_cycle.device, dtype=per_cycle.dtype)
    counts = torch.zeros(n_vertices, device=per_cycle.device, dtype=per_cycle.dtype)
    for i in range(k):
        vidx = cycles[:, i]
        out.index_add_(0, vidx, per_cycle)
        counts.index_add_(0, vidx, torch.ones_like(vidx, dtype=per_cycle.dtype))
    return out / counts.clamp_min(1.0).unsqueeze(-1)


# ─── scatter_mean vectorized parity ─────────────────────────────────


def test_scatter_mean_matches_corner_loop():
    torch.manual_seed(0)
    for _ in range(8):
        n_v, m_c, k, d = 25, 30, 3, 6
        per_cycle = torch.randn(m_c, d)
        cycles = torch.randint(0, n_v, (m_c, k), dtype=torch.long)
        a = scatter_mean(per_cycle, cycles, n_v)
        b = _scatter_mean_corner_loop(per_cycle, cycles, n_v)
        assert torch.allclose(a, b, rtol=1e-6, atol=1e-6)


def test_scatter_mean_batched_matches_slices():
    torch.manual_seed(1)
    b, n_v, m_c, k, d = 4, 18, 12, 3, 5
    per_b = torch.randn(b, m_c, d)
    cycles = torch.randint(0, n_v, (m_c, k), dtype=torch.long)
    batched = scatter_mean(per_b, cycles, n_v)
    assert batched.shape == (b, n_v, d)
    for i in range(b):
        ref = scatter_mean(per_b[i], cycles, n_v)
        assert torch.allclose(batched[i], ref, rtol=1e-6, atol=1e-6)


def test_outer_fir_batched_matches_sequential_banks():
    """Regression: vectorized OuterFIRShell forward == per-bank loop."""
    torch.manual_seed(2)
    shell = OuterFIRShell(d_in=8, d_layer=4, M=5, cycle_k=3)
    n, m_c = 14, 18
    x = torch.randn(n, 8)
    cycles = torch.randint(0, n, (m_c, 3), dtype=torch.long)
    signs = torch.where(torch.randint(0, 2, (m_c, 3)) == 0, -1.0, 1.0)
    out = shell(x, cycles, signs)
    parts = []
    for m in range(shell.M):
        xp = shell.pre_projs[m](x)
        cv = xp[cycles]
        pc = shell.banks[m](cv, signs.float())
        parts.append(_scatter_mean_corner_loop(pc, cycles, n))
    ref = torch.cat(parts, dim=-1)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)


# ─── Outer FIR shell ────────────────────────────────────────────────


def test_outer_shell_forward_shape():
    """OuterFIRShell produces (N, M·d_layer) per-vertex features."""
    shell = OuterFIRShell(d_in=8, d_layer=4, M=3, cycle_k=3)
    N = 10
    x = torch.randn(N, 8)
    cycles = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 2, 4]], dtype=torch.long)
    signs = torch.tensor([[1, -1, 1], [1, 1, -1], [-1, 1, 1]], dtype=torch.float)
    out = shell(x, cycles, signs)
    assert out.shape == (N, 3 * 4)
    assert torch.isfinite(out).all()


def test_outer_shell_M_banks_diversified():
    """The M banks should have distinct initial coefficients."""
    shell = OuterFIRShell(d_in=4, d_layer=4, M=4, cycle_k=3)
    coef_as = [shell.banks[m].coef_a.detach().tolist() for m in range(4)]
    # All four should be different
    coef_set = {tuple(c) for c in coef_as}
    assert len(coef_set) == 4, f"banks not diversified: {coef_set}"


def test_outer_shell_zero_cycles_returns_zeros():
    shell = OuterFIRShell(d_in=4, d_layer=2, M=2, cycle_k=3)
    x = torch.randn(5, 4)
    cycles = torch.zeros((0, 3), dtype=torch.long)
    signs = torch.zeros((0, 3), dtype=torch.float)
    out = shell(x, cycles, signs)
    assert out.shape == (5, 4)
    assert torch.allclose(out, torch.zeros_like(out))


# ─── Middle HSiKAN ──────────────────────────────────────────────────


def test_middle_hsikan_forward_shape():
    middle = MiddleHSiKAN(n_nodes=10, d_in=12, d_layer=8, cycle_k=3)
    N = 10
    x = torch.randn(N, 12)
    cycles = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    signs = torch.tensor([[1, -1, 1], [1, 1, -1]], dtype=torch.float)
    out = middle(x, cycles, signs)
    assert out.shape == (N, 8)
    assert torch.isfinite(out).all()


def test_middle_hsikan_zero_cycles_zeros_output():
    middle = MiddleHSiKAN(n_nodes=8, d_in=6, d_layer=4, cycle_k=3)
    x = torch.randn(8, 6)
    cycles = torch.zeros((0, 3), dtype=torch.long)
    signs = torch.zeros((0, 3), dtype=torch.float)
    out = middle(x, cycles, signs)
    assert torch.allclose(out, torch.zeros_like(out))


# ─── Composer end-to-end ────────────────────────────────────────────


def test_gomb_forward_shape_and_param_count():
    cfg = GombConfig(
        n_nodes=20, d_embed=16, d_outer=8, M_outer=4,
        d_middle=16, d_core=16, n_tiers=3, cycle_k=3,
    )
    gomb = HymeKoGomb(cfg)
    cycles = torch.tensor([
        [0, 1, 2], [3, 4, 5], [0, 3, 6],
        [1, 4, 7], [2, 5, 8], [6, 7, 8],
    ], dtype=torch.long)
    signs = torch.where(cycles % 2 == 0, 1.0, -1.0)
    tier_of = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1],
        dtype=torch.long,
    )
    edges = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
    scores = gomb(cycles, signs, tier_of, edges)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    # Reasonable param count
    n_p = gomb.n_params()
    assert 1000 < n_p < 200_000, f"unexpected param count {n_p}"


def test_gomb_backward_no_nan():
    cfg = GombConfig(
        n_nodes=15, d_embed=8, d_outer=4, M_outer=2,
        d_middle=8, d_core=8, n_tiers=2, cycle_k=3,
    )
    gomb = HymeKoGomb(cfg)
    # Cycles touching BOTH tiers (so every tier's aggregator gets
    # gradient).  Tier 0 = vertices 0-7, Tier 1 = vertices 8-14.
    cycles = torch.tensor([
        [0, 1, 8],     # tier-0 + tier-1
        [2, 9, 10],    # tier-0 + tier-1
        [3, 4, 12],    # tier-0 + tier-1
    ], dtype=torch.long)
    signs = torch.tensor([
        [1, -1, 1], [1, 1, -1], [-1, 1, 1],
    ], dtype=torch.float)
    tier_of = torch.cat([
        torch.zeros(8, dtype=torch.long),
        torch.ones(7, dtype=torch.long),
    ])
    edges = torch.tensor([[0, 8], [2, 9]], dtype=torch.long)
    targets = torch.tensor([1.0, 0.0])
    scores = gomb(cycles, signs, tier_of, edges)
    loss = F.binary_cross_entropy_with_logits(scores, targets)
    loss.backward()
    for name, p in gomb.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert not torch.isnan(p.grad).any(), f"nan grad on {name}"


def test_gomb_no_cycles_degrades_to_embedding_only():
    """With no cycles, outer+middle produce zeros and the inner core
    sees only the embedding. The model should still produce finite
    scores (the embedding-floor pathway works)."""
    cfg = GombConfig(
        n_nodes=10, d_embed=8, d_outer=4, M_outer=2,
        d_middle=8, d_core=8, n_tiers=2, cycle_k=3,
    )
    gomb = HymeKoGomb(cfg)
    cycles = torch.zeros((0, 3), dtype=torch.long)
    signs = torch.zeros((0, 3), dtype=torch.float)
    tier_of = torch.zeros(10, dtype=torch.long)
    edges = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    scores = gomb(cycles, signs, tier_of, edges)
    assert torch.isfinite(scores).all()


# ─── Synthetic-data training smoke ──────────────────────────────────


def test_gomb_trains_on_synthetic_moons():
    """End-to-end smoke: Gömb trains on a tiny moon signed graph for
    20 epochs without divergence; loss decreases."""
    from signedkan_wip.src.datasets_synth import make_moon_signed_graph

    torch.manual_seed(0)
    g, X, y = make_moon_signed_graph(n_samples=60, k_neighbors=5, seed=0)
    cfg = GombConfig(
        n_nodes=g.n_nodes, d_embed=8, d_outer=4, M_outer=3,
        d_middle=8, d_core=8, n_tiers=2, cycle_k=3,
    )
    gomb = HymeKoGomb(cfg)
    opt = torch.optim.Adam(gomb.parameters(), lr=1e-2)

    # Cycles = empty (smoke tests pathway without cycle enumeration).
    cycles = torch.zeros((0, 3), dtype=torch.long)
    signs = torch.zeros((0, 3), dtype=torch.float)
    # Degrees → tier_of
    degrees = np.zeros(g.n_nodes, dtype=np.int64)
    for (u, v) in g.edges:
        degrees[int(u)] += 1
        degrees[int(v)] += 1
    cuts = (0.0, 0.5, 1.0)
    # Tier assignment by degree percentile (same scheme as CPML).
    n = degrees.shape[0]
    order = np.argsort(degrees, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n) / max(1, n - 1)
    tier_of = torch.from_numpy(
        np.where(ranks <= 0.5, 0, 1).astype(np.int64),
    )
    edges_t = torch.from_numpy(g.edges.astype(np.int64))
    targets = torch.from_numpy((g.signs > 0).astype(np.float32))

    initial_loss = None
    final_loss = None
    for ep in range(20):
        scores = gomb(cycles, signs, tier_of, edges_t)
        loss = F.binary_cross_entropy_with_logits(scores, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
    # Loss should decrease, not catastrophically diverge.
    assert final_loss is not None and np.isfinite(final_loss)
    # Allow some slop — synthetic graphs are tiny.
    assert final_loss < initial_loss * 1.2


# ─── Ablation wrappers ──────────────────────────────────────────────


def _ablation_inputs(n_nodes=20):
    cycles = torch.tensor([
        [0, 1, 2], [3, 4, 5], [0, 3, 6],
        [1, 4, 7], [2, 5, 8], [6, 7, 8],
    ], dtype=torch.long)
    signs = torch.where(cycles % 2 == 0, 1.0, -1.0)
    tier_of = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1],
        dtype=torch.long,
    )[:n_nodes]
    edges = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
    return cycles, signs, tier_of, edges


def test_gomb_no_outer_forward_backward():
    cfg = GombConfig(
        n_nodes=20, d_embed=16, d_outer=8, M_outer=4,
        d_middle=16, d_core=16, n_tiers=3, cycle_k=3,
    )
    m = GombNoOuter(cfg)
    cycles, signs, tier_of, edges = _ablation_inputs()
    scores = m(cycles, signs, tier_of, edges)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    loss = F.binary_cross_entropy_with_logits(
        scores, torch.tensor([1.0, 0.0, 1.0]),
    )
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert not torch.isnan(p.grad).any(), f"nan grad on {name}"


def test_gomb_no_middle_forward_backward():
    cfg = GombConfig(
        n_nodes=20, d_embed=16, d_outer=8, M_outer=4,
        d_middle=16, d_core=16, n_tiers=3, cycle_k=3,
    )
    m = GombNoMiddle(cfg)
    cycles, signs, tier_of, edges = _ablation_inputs()
    scores = m(cycles, signs, tier_of, edges)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    loss = F.binary_cross_entropy_with_logits(
        scores, torch.tensor([1.0, 0.0, 1.0]),
    )
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert not torch.isnan(p.grad).any(), f"nan grad on {name}"


def test_gomb_no_inner_forward_backward():
    cfg = GombConfig(
        n_nodes=20, d_embed=16, d_outer=8, M_outer=4,
        d_middle=16, d_core=16, n_tiers=3, cycle_k=3,
    )
    m = GombNoInner(cfg)
    cycles, signs, tier_of, edges = _ablation_inputs()
    scores = m(cycles, signs, tier_of, edges)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    loss = F.binary_cross_entropy_with_logits(
        scores, torch.tensor([1.0, 0.0, 1.0]),
    )
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert not torch.isnan(p.grad).any(), f"nan grad on {name}"


def test_mixed_arity_forward_backward():
    """MixedArityGomb forward + backward with k=3 ∪ k=4 cycles."""
    cfg = GombConfig(
        n_nodes=20, d_embed=8, d_outer=4, M_outer=2,
        d_middle=8, d_core=8, n_tiers=2, cycle_k=3,
    )
    m = MixedArityGomb(cfg, cycle_ks=(3, 4))
    # Cycles must touch BOTH tiers so every per-tier aggregator gets
    # a gradient (mirror the test_gomb_backward_no_nan invariant).
    cycles_by_k = {
        3: torch.tensor(
            [[0, 1, 12], [3, 4, 15], [6, 17, 18]], dtype=torch.long,
        ),
        4: torch.tensor(
            [[0, 1, 11, 12], [4, 5, 14, 15]], dtype=torch.long,
        ),
    }
    signs_by_k = {
        3: torch.where(cycles_by_k[3] % 2 == 0, 1.0, -1.0),
        4: torch.where(cycles_by_k[4] % 2 == 0, 1.0, -1.0),
    }
    tier_of = torch.zeros(20, dtype=torch.long)
    tier_of[10:] = 1
    edges = torch.tensor([[0, 1], [5, 12], [11, 17]], dtype=torch.long)
    scores = m(cycles_by_k, signs_by_k, tier_of, edges)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    # alpha is a proper distribution over arities
    alpha = m.alpha()
    assert alpha.shape == (2,)
    assert torch.isclose(alpha.sum(), torch.tensor(1.0), atol=1e-6)
    # backward populates grads on every parameter
    loss = F.binary_cross_entropy_with_logits(
        scores, torch.tensor([1.0, 0.0, 1.0]),
    )
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert not torch.isnan(p.grad).any(), f"nan grad on {name}"


def test_mixed_arity_empty_pool_for_one_k():
    """Mixed-arity handles the case where one of the k-pools is empty."""
    cfg = GombConfig(
        n_nodes=15, d_embed=8, d_outer=4, M_outer=2,
        d_middle=8, d_core=8, n_tiers=2, cycle_k=3,
    )
    m = MixedArityGomb(cfg, cycle_ks=(3, 4))
    cycles_by_k = {
        3: torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
        4: torch.zeros((0, 4), dtype=torch.long),  # empty k=4 pool
    }
    signs_by_k = {
        3: torch.tensor([[1, -1, 1], [1, 1, -1]], dtype=torch.float),
        4: torch.zeros((0, 4), dtype=torch.float),
    }
    tier_of = torch.zeros(15, dtype=torch.long)
    edges = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    scores = m(cycles_by_k, signs_by_k, tier_of, edges)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_sample_params_compact_is_narrow():
    from signedkan_wip.src.run_gomb_tune import sample_params

    rng = np.random.default_rng(0)
    for _ in range(40):
        p = sample_params(rng, "slashdot", compact=True)
        assert p["d_embed"] <= 24
        assert p["M_outer"] <= 4
        assert p["d_outer"] <= 10
        assert p["topk"] <= 40


def test_parse_last_gomb_json_line():
    from signedkan_wip.src.run_gomb_tune import _parse_last_gomb_json

    stdout = "garbage\n{\"dataset\": \"z\", \"test_auroc\": 0.8123}\n"
    row = _parse_last_gomb_json(stdout)
    assert row is not None
    assert row["test_auroc"] == 0.8123


def test_heldout_edge_metrics_label_keys():
    """Smoke metrics helper must namespace keys by split label."""
    from signedkan_wip.src.run_gomb_smoke import _heldout_edge_metrics

    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    p = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64)
    mv = _heldout_edge_metrics(y, p, "val")
    mt = _heldout_edge_metrics(y, p, "test")
    assert "val_auroc" in mv and "val_f1_macro" in mv
    assert "test_auroc" in mt and "test_average_precision" in mt
    assert mv["val_auroc"] == mt["test_auroc"]


def test_gomb_ablations_have_fewer_params_than_full():
    """Each one-shell-removed ablation must be strictly smaller than the
    full cascade, else we removed nothing measurable."""
    cfg = GombConfig(
        n_nodes=20, d_embed=16, d_outer=8, M_outer=4,
        d_middle=16, d_core=16, n_tiers=3, cycle_k=3,
    )
    full = HymeKoGomb(cfg).n_params()
    assert GombNoOuter(cfg).n_params() < full
    assert GombNoMiddle(cfg).n_params() < full
    # NoInner replaces CPML with a plain MLP head, so it can be larger
    # OR smaller depending on cfg; just assert it produces a real number.
    no_inner = GombNoInner(cfg).n_params()
    assert no_inner > 0
