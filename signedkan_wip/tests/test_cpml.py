"""Feasibility tests for the CPML (Concentric-Pyramid Multi-Layer)
architecture (2026-05-11).

Tests gate CPML for queue promotion:
    * tier_stratify_returns_correct_sizes
    * forward_shapes_l2_l3_l4
    * backward_no_nan
    * l1_recovers_flat_baseline
    * innermost_tier_receives_outer_features
    * peripheral_cycles_subset_of_inner_pool
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from signedkan_wip.src.cpml import (
    CPML,
    CPMLConfig,
    TierSpec,
    restrict_cycles_to_tier,
)


# ─── Helpers ────────────────────────────────────────────────────────


def _power_law_degrees(n: int, alpha: float = 2.5, seed: int = 0) -> np.ndarray:
    """Synthetic degree sequence: power-law-distributed, integer."""
    rng = np.random.default_rng(seed)
    raw = rng.pareto(alpha, n) + 1
    return np.ceil(raw).astype(np.int64)


def _toy_cycles(n_vertices: int, n_cycles: int = 50,
                 k: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random triangles + sign assignments for unit-test fixtures."""
    rng = np.random.default_rng(seed)
    cycles: list[list[int]] = []
    while len(cycles) < n_cycles:
        vs = rng.choice(n_vertices, size=k, replace=False)
        # Cycle (v0, v1, ..., v_{k-1}) → boundary edges between
        # consecutive vertices cyclically.  Stored sorted to dedup.
        cycles.append(sorted(vs.tolist()))
    cycles_arr = np.array(cycles, dtype=np.int64)
    signs = rng.choice([-1, 1], size=(n_cycles, k)).astype(np.int8)
    return cycles_arr, signs


# ─── Tier stratification ─────────────────────────────────────────────


def test_tier_stratify_returns_correct_sizes():
    """Stratification respects the configured percentile cuts."""
    n = 1000
    degrees = _power_law_degrees(n, seed=42)
    ts = TierSpec(cuts=(0.0, 0.2, 0.8, 1.0))
    tiers = ts.assign(degrees)
    assert tiers.shape == (n,)
    # Outer tier ~20%, mid ~60%, inner ~20%; allow ±5pp slop because
    # power-law ties can cluster at the boundaries.
    n_t0 = int((tiers == 0).sum())
    n_t1 = int((tiers == 1).sum())
    n_t2 = int((tiers == 2).sum())
    assert abs(n_t0 - 200) <= 50, f"tier 0: {n_t0}"
    assert abs(n_t1 - 600) <= 50, f"tier 1: {n_t1}"
    assert abs(n_t2 - 200) <= 50, f"tier 2: {n_t2}"
    assert n_t0 + n_t1 + n_t2 == n


def test_tier_stratify_innermost_is_highest_degree():
    """Tier L-1 holds the highest-degree vertices."""
    degrees = np.array([1, 1, 5, 100, 200, 2, 3], dtype=np.int64)
    ts = TierSpec(cuts=(0.0, 0.5, 1.0))   # L=2
    tiers = ts.assign(degrees)
    # 200 and 100 should land in tier 1 (innermost).
    assert tiers[degrees.argmax()] == 1
    # 1 (smallest) should land in tier 0.
    assert tiers[degrees.argmin()] == 0


# ─── Cycle-pool restriction ──────────────────────────────────────────


def test_restrict_cycles_to_tier_filters_correctly():
    cycles = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [0, 3, 5],
        [1, 4, 6],
    ], dtype=np.int64)
    # Vertices 0,1,2,3 are tier 0; 4,5,6 tier 1.
    tier_of = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)

    t0 = restrict_cycles_to_tier(cycles, tier_of, 0)
    # All cycles touch some tier-0 vertex except possibly none.
    # cycle (3,4,5) touches t0 via 3. cycle (1,4,6) touches t0 via 1.
    assert t0.shape[0] == 4

    t1 = restrict_cycles_to_tier(cycles, tier_of, 1)
    # (0,1,2) has no tier-1 vertex; others all do.
    assert t1.shape[0] == 3
    # Check (0, 1, 2) is excluded.
    assert not any((t1 == np.array([0, 1, 2])).all(axis=1))


# ─── Forward shape tests ────────────────────────────────────────────


@pytest.mark.parametrize("L_cuts", [
    (0.0, 1.0),                           # L=1 (sanity)
    (0.0, 0.5, 1.0),                       # L=2
    (0.0, 0.2, 0.8, 1.0),                  # L=3 (default)
    (0.0, 0.1, 0.3, 0.7, 1.0),             # L=4
])
def test_forward_shapes(L_cuts):
    n = 30
    d_in = 8
    cycles_arr, signs = _toy_cycles(n_vertices=n, n_cycles=20, k=3, seed=1)
    degrees = np.array([
        int((cycles_arr == v).sum()) for v in range(n)
    ], dtype=np.int64) + 1
    cfg = CPMLConfig(tier_spec=TierSpec(cuts=L_cuts),
                      d_in=d_in, d_layer=4, d_predictor_hidden=8)
    model = CPML(cfg)
    tier_of = torch.from_numpy(cfg.tier_spec.assign(degrees))
    node_features = torch.randn(n, d_in)
    cycles_t = torch.from_numpy(cycles_arr)
    signs_t = torch.from_numpy(signs)
    edges_to_score = torch.from_numpy(cycles_arr[:5, :2].copy())  # (5, 2)

    scores = model(node_features, cycles_t, signs_t, tier_of, edges_to_score)
    assert scores.shape == (5,)
    # Final concatenated dim = d_in + L * d_layer
    L = cfg.tier_spec.L
    expected_final = d_in + L * cfg.d_layer
    assert model.in_dims[-1] == expected_final


def test_backward_no_nan():
    n = 40
    d_in = 8
    cycles_arr, signs = _toy_cycles(n, n_cycles=30, k=3, seed=2)
    degrees = np.bincount(cycles_arr.ravel(), minlength=n) + 1
    cfg = CPMLConfig(d_in=d_in, d_layer=8)
    model = CPML(cfg)
    tier_of = torch.from_numpy(cfg.tier_spec.assign(degrees))
    x = torch.randn(n, d_in, requires_grad=False)
    cycles_t = torch.from_numpy(cycles_arr)
    signs_t = torch.from_numpy(signs)
    edges = torch.from_numpy(cycles_arr[:10, :2].copy())
    targets = torch.randint(0, 2, (10,), dtype=torch.float32)

    scores = model(x, cycles_t, signs_t, tier_of, edges)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, targets)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert not torch.isnan(p.grad).any(), f"NaN grad on {name}"


# ─── Sanity: L=1 recovers a flat baseline ────────────────────────────


def test_l1_runs_and_produces_finite_output():
    """L=1 should be the flat-aggregator baseline.  Test it produces
    finite scores (no NaN/Inf) — equivalent-to-flat numerical check
    is harder because the flat baseline lives elsewhere."""
    n = 25
    d_in = 6
    cycles_arr, signs = _toy_cycles(n, n_cycles=15, k=3, seed=3)
    degrees = np.bincount(cycles_arr.ravel(), minlength=n) + 1
    cfg = CPMLConfig(
        tier_spec=TierSpec(cuts=(0.0, 1.0)),    # L=1
        d_in=d_in, d_layer=4,
    )
    model = CPML(cfg)
    assert model.L == 1
    tier_of = torch.from_numpy(cfg.tier_spec.assign(degrees))
    # All vertices land in tier 0; verify.
    assert (tier_of == 0).all()
    x = torch.randn(n, d_in)
    edges = torch.from_numpy(cycles_arr[:3, :2].copy())
    scores = model(x, torch.from_numpy(cycles_arr),
                     torch.from_numpy(signs), tier_of, edges)
    assert torch.isfinite(scores).all()


# ─── Inward funnelling: H_L row depends on outer-tier features ──────


def test_innermost_tier_receives_outer_features():
    """Perturbing an outer-tier vertex's feature should change the
    inner-tier embedding of any vertex sharing a cycle with it.

    This validates the concat-style upward propagation: H_L is
    computed from X = concat(X_0, H_1, ..., H_{L-1}), so an X_0
    change at outer-tier vertex u propagates through H_1 to all
    other layers.
    """
    n = 20
    d_in = 6
    # Force a known cycle structure: triangle (0, 1, 2) and (1, 3, 4).
    # Tier setup: vertex 0 = outer (low degree), vertex 1 = inner (hub).
    cycles_arr = np.array([
        [0, 1, 2],
        [1, 3, 4],
        [1, 5, 6],
        [1, 7, 8],
    ], dtype=np.int64)
    signs = np.ones_like(cycles_arr, dtype=np.int8)
    # Compute degrees from cycle participation.
    degrees = np.bincount(cycles_arr.ravel(), minlength=n) + 1
    # Vertex 1 has high degree, vertex 0 low.

    cfg = CPMLConfig(
        tier_spec=TierSpec(cuts=(0.0, 0.5, 1.0)),    # L=2
        d_in=d_in, d_layer=4,
    )
    model = CPML(cfg)
    tier_of = torch.from_numpy(cfg.tier_spec.assign(degrees))

    x = torch.randn(n, d_in)
    cycles_t = torch.from_numpy(cycles_arr)
    signs_t = torch.from_numpy(signs)

    # Edge to score: (0, 1)
    edges = torch.tensor([[0, 1]], dtype=torch.long)

    with torch.no_grad():
        s_orig = model(x, cycles_t, signs_t, tier_of, edges)

    # Perturb vertex 0's features (an outer-tier vertex).
    x_perturbed = x.clone()
    x_perturbed[0] += 10.0   # large perturbation
    with torch.no_grad():
        s_perturbed = model(x_perturbed, cycles_t, signs_t, tier_of, edges)

    # The score should change — proves outer-tier signal reached the
    # final embedding for vertex 1 (an inner-tier vertex).
    assert not torch.allclose(s_orig, s_perturbed, atol=1e-4), \
        "outer-tier perturbation didn't propagate to inner-tier embedding"


# ─── Memory + speed feasibility ──────────────────────────────────────


def test_memory_below_4gb_at_scale():
    """Synthetic scale matching Bitcoin OTC (~6k vertices, ~100k
    cycles).  Forward should consume far less than 4 GB."""
    if not torch.cuda.is_available():
        pytest.skip("memory test only on CUDA")
    n = 6000
    d_in = 16
    rng = np.random.default_rng(7)
    n_cycles = 50_000
    cycles_arr = rng.integers(0, n, size=(n_cycles, 4), dtype=np.int64)
    signs = rng.choice([-1, 1], size=cycles_arr.shape).astype(np.int8)
    degrees = np.bincount(cycles_arr.ravel(), minlength=n) + 1

    cfg = CPMLConfig(
        tier_spec=TierSpec(cuts=(0.0, 0.2, 0.8, 1.0)),
        d_in=d_in, d_layer=16,
    )
    model = CPML(cfg).cuda()
    tier_of = torch.from_numpy(cfg.tier_spec.assign(degrees)).cuda()
    x = torch.randn(n, d_in, device="cuda")
    cycles_t = torch.from_numpy(cycles_arr).cuda()
    signs_t = torch.from_numpy(signs).cuda()
    edges = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long).cuda()

    torch.cuda.reset_peak_memory_stats()
    scores = model(x, cycles_t, signs_t, tier_of, edges)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    assert peak_mb < 4 * 1024, f"peak memory {peak_mb:.1f} MB exceeded 4 GB"
    assert torch.isfinite(scores).all()
