"""Feasibility tests for the CPML (Concentric-Pyramid Multi-Layer)
architecture (2026-05-11).

Tests gate CPML for queue promotion:
    * tier_stratify_returns_correct_sizes
    * forward_shapes (L ∈ {1,2,3,4}, route×{structural,capsule_soft}, pyramid×structural)
    * backward_no_nan (same grid)
    * capsule_soft EM routing (forward + backward)
    * capsule_soft hypergraph_conv **signed** routing + validation tests
    * capsule_soft router / iteration validation
    * l1_runs_and_produces_finite_output
    * innermost_tier_receives_outer_features (pyramid)
    * memory_below_4gb_at_scale (pyramid, CUDA)
    * route_vs_pyramid_second_tier_mlp_contract
    * capsule_soft_rejects_pyramid_topology
    * capsule_routing_iterations_invalid
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from signedkan_wip.src.cpml import (
    CPML,
    CPMLConfig,
    CapsuleHypergraphRouter,
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
@pytest.mark.parametrize("topology,tier_org", [
    ("route", "structural"),
    ("route", "capsule_soft"),
    ("pyramid", "structural"),
])
def test_forward_shapes(L_cuts, topology, tier_org):
    n = 30
    d_in = 8
    cycles_arr, signs = _toy_cycles(n_vertices=n, n_cycles=20, k=3, seed=1)
    degrees = np.array([
        int((cycles_arr == v).sum()) for v in range(n)
    ], dtype=np.int64) + 1
    cfg = CPMLConfig(tier_spec=TierSpec(cuts=L_cuts),
                      d_in=d_in, d_layer=4, d_predictor_hidden=8,
                      topology=topology, tier_organization=tier_org)
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


@pytest.mark.parametrize("topology,tier_org", [
    ("route", "structural"),
    ("route", "capsule_soft"),
    ("pyramid", "structural"),
])
def test_backward_no_nan(topology, tier_org):
    n = 40
    d_in = 8
    cycles_arr, signs = _toy_cycles(n, n_cycles=30, k=3, seed=2)
    degrees = np.bincount(cycles_arr.ravel(), minlength=n) + 1
    cfg = CPMLConfig(
        d_in=d_in, d_layer=8, topology=topology, tier_organization=tier_org,
    )
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
        topology="pyramid",
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
        topology="pyramid",
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


def test_route_vs_pyramid_second_tier_mlp_in_features():
    """Route: every MLP tier sees corner dim ``d_in``.
    Pyramid: tier 1+ sees widening concat ``d_in + ell * d_layer``."""
    d_in, d_layer = 8, 4
    cuts = (0.0, 0.5, 1.0)
    cfg_route = CPMLConfig(
        tier_spec=TierSpec(cuts=cuts),
        d_in=d_in,
        d_layer=d_layer,
        aggregator_kind="mlp",
        topology="route",
    )
    cfg_pyramid = CPMLConfig(
        tier_spec=TierSpec(cuts=cuts),
        d_in=d_in,
        d_layer=d_layer,
        aggregator_kind="mlp",
        topology="pyramid",
    )
    m_route = CPML(cfg_route)
    m_pyramid = CPML(cfg_pyramid)
    assert m_route.aggregators[0].proj[0].in_features == d_in
    assert m_pyramid.aggregators[0].proj[0].in_features == d_in
    assert m_route.aggregators[1].proj[0].in_features == d_in
    assert m_pyramid.aggregators[1].proj[0].in_features == d_in + d_layer


def test_capsule_soft_rejects_pyramid_topology():
    with pytest.raises(ValueError, match="capsule_soft"):
        CPML(
            CPMLConfig(
                tier_spec=TierSpec(cuts=(0.0, 0.5, 1.0)),
                d_in=8,
                d_layer=4,
                topology="pyramid",
                tier_organization="capsule_soft",
            ),
        )


def test_capsule_routing_iterations_invalid():
    with pytest.raises(ValueError, match="capsule_routing_iterations"):
        CPML(
            CPMLConfig(
                tier_spec=TierSpec(cuts=(0.0, 0.5, 1.0)),
                d_in=8,
                d_layer=4,
                topology="route",
                tier_organization="capsule_soft",
                capsule_routing_iterations=0,
            ),
        )


def test_capsule_soft_em_routing_forward_backward():
    """Multi-step routing touches cycle batch repeatedly; grads must flow."""
    n = 40
    d_in = 8
    cycles_arr, signs = _toy_cycles(n, n_cycles=30, k=3, seed=11)
    degrees = np.bincount(cycles_arr.ravel(), minlength=n) + 1
    cfg = CPMLConfig(
        d_in=d_in,
        d_layer=8,
        topology="route",
        tier_organization="capsule_soft",
        capsule_soft_router="em_agreement",
        capsule_routing_iterations=3,
    )
    model = CPML(cfg)
    assert model.capsule_router is None
    assert model.capsule_init_logits is not None
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


def test_capsule_soft_single_step_uses_router_mlp_not_init():
    cfg = CPMLConfig(
        d_in=8,
        d_layer=4,
        topology="route",
        tier_organization="capsule_soft",
        capsule_routing_iterations=1,
    )
    m = CPML(cfg)
    assert m.capsule_router is not None
    assert m.capsule_init_logits is None
    assert m.capsule_hg_vertex_proj is None


def test_capsule_soft_hypergraph_conv_forward_backward():
    n = 40
    d_in = 8
    cycles_arr, signs = _toy_cycles(n, n_cycles=30, k=3, seed=13)
    degrees = np.bincount(cycles_arr.ravel(), minlength=n) + 1
    cfg = CPMLConfig(
        d_in=d_in,
        d_layer=8,
        topology="route",
        tier_organization="capsule_soft",
        capsule_soft_router="hypergraph_conv",
        capsule_routing_iterations=1,
        capsule_hg_hidden=32,
    )
    model = CPML(cfg)
    assert model._capsule_soft_router_resolved == "hypergraph_conv"
    assert model.capsule_router is None
    assert model.capsule_init_logits is None
    assert model.capsule_hg_block is not None
    assert model.capsule_hg_vertex_proj is not None
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


@pytest.mark.parametrize(
    ("router", "nit"),
    [
        ("hypergraph_conv", 3),
        ("mlp_softmax", 2),
        ("em_agreement", 1),
    ],
)
def test_capsule_soft_router_iteration_mismatch_raises(router, nit):
    with pytest.raises(ValueError, match="capsule_soft_router"):
        CPML(
            CPMLConfig(
                tier_spec=TierSpec(cuts=(0.0, 0.5, 1.0)),
                d_in=8,
                d_layer=4,
                topology="route",
                tier_organization="capsule_soft",
                capsule_soft_router=router,
                capsule_routing_iterations=nit,
            ),
        )


def test_capsule_soft_unknown_router_raises():
    with pytest.raises(ValueError, match="unknown capsule_soft_router"):
        CPML(
            CPMLConfig(
                tier_spec=TierSpec(cuts=(0.0, 0.5, 1.0)),
                d_in=8,
                d_layer=4,
                topology="route",
                tier_organization="capsule_soft",
                capsule_soft_router="not_a_router",  # type: ignore[arg-type]
                capsule_routing_iterations=1,
            ),
        )


def test_capsule_hypergraph_routing_mismatched_sign_shape_raises():
    cfg = CPMLConfig(
        tier_spec=TierSpec(cuts=(0.0, 0.5, 1.0)),
        d_in=8,
        d_layer=4,
        topology="route",
        tier_organization="capsule_soft",
        capsule_soft_router="hypergraph_conv",
        capsule_routing_iterations=1,
    )
    m = CPML(cfg)
    x = torch.randn(6, 8)
    cycles = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    bad = torch.ones(2, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="cycle_signs"):
        m._capsule_hypergraph_routing_logits(x, cycles, bad)


def test_capsule_hypergraph_routing_logits_depend_on_cycle_signs():
    """Regression: σ scales node↔hyperedge messages in the HG router."""
    torch.manual_seed(42)
    cfg = CPMLConfig(
        tier_spec=TierSpec(cuts=(0.0, 0.5, 1.0)),
        d_in=6,
        d_layer=4,
        topology="route",
        tier_organization="capsule_soft",
        capsule_soft_router="hypergraph_conv",
        capsule_routing_iterations=1,
        capsule_hg_hidden=16,
    )
    m = CPML(cfg)
    x = torch.randn(5, 6)
    cycles = torch.tensor([[0, 1, 2]], dtype=torch.long)
    s_pos = torch.ones(1, 3, dtype=torch.float32)
    s_neg = s_pos.clone()
    s_neg[0, 0] = -1.0
    with torch.no_grad():
        logits_pos = m._capsule_hypergraph_routing_logits(x, cycles, s_pos)
        logits_neg = m._capsule_hypergraph_routing_logits(x, cycles, s_neg)
    assert logits_pos.shape == (1, cfg.tier_spec.L)
    assert not torch.allclose(
        logits_pos, logits_neg, atol=1e-5, rtol=1e-5,
    ), "signed HG routing logits should change when σ flips"


def test_capsule_hypergraph_cache_degrees_parity_with_uncached():
    """``capsule_hg_cache_degrees`` must not change numerics vs full recompute."""
    torch.manual_seed(7)
    n, d_in = 28, 8
    cycles_arr, signs = _toy_cycles(n, n_cycles=35, k=3, seed=101)
    degrees = np.bincount(cycles_arr.ravel(), minlength=n) + 1
    base = dict(
        tier_spec=TierSpec(cuts=(0.0, 0.5, 1.0)),
        d_in=d_in,
        d_layer=8,
        topology="route",
        tier_organization="capsule_soft",
        capsule_soft_router="hypergraph_conv",
        capsule_routing_iterations=1,
        capsule_hg_hidden=24,
    )
    m_on = CPML(CPMLConfig(**base, capsule_hg_cache_degrees=True)).eval()
    m_off = CPML(CPMLConfig(**base, capsule_hg_cache_degrees=False)).eval()
    m_off.load_state_dict(m_on.state_dict())

    cfg_ref = CPMLConfig(**base)
    tier_of = torch.from_numpy(cfg_ref.tier_spec.assign(degrees))
    x = torch.randn(n, d_in)
    cycles = torch.from_numpy(cycles_arr)
    signs_t = torch.from_numpy(signs)
    edges = torch.from_numpy(cycles_arr[:8, :2].copy())

    with torch.no_grad():
        y_on = m_on(x, cycles, signs_t, tier_of, edges)
        y_off = m_off(x, cycles, signs_t, tier_of, edges)
    assert torch.allclose(y_on, y_off, rtol=1e-6, atol=1e-6)

    # Second forward reuses ``cycles`` object → cache path on m_on
    with torch.no_grad():
        y_on2 = m_on(x, cycles, signs_t, tier_of, edges)
    assert torch.allclose(y_on, y_on2, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_capsule_hypergraph_router_torch_compile_matches_eager():
    """``torch.compile(CapsuleHypergraphRouter)`` shares params with eager module."""
    torch.manual_seed(3)
    n, m, k, d_in, d_h, L = 14, 25, 3, 8, 16, 2
    r = CapsuleHypergraphRouter(d_in, d_h, L).eval()
    rc = torch.compile(
        r, mode="reduce-overhead", fullgraph=False, dynamic=True,
    )
    x = torch.randn(n, d_in)
    cycles = torch.randint(0, n, (m, k), dtype=torch.long)
    sigma = torch.where(cycles % 2 == 0, 1.0, -1.0)
    d_v = torch.randint(1, 4, (n,), dtype=torch.float32).clamp_min(1.0)
    dv = d_v.pow(-0.5)
    with torch.no_grad():
        y_e = r(x, cycles, sigma, d_v, dv)
        for _ in range(2):
            _ = rc(x, cycles, sigma, d_v, dv)
        y_c = rc(x, cycles, sigma, d_v, dv)
    assert torch.allclose(y_e, y_c, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_cpml_torch_compile_hypergraph_forward_finite():
    """Full CPML with ``torch_compile_hypergraph`` runs and returns finite scores."""
    n, d_in = 18, 8
    cycles_arr, signs = _toy_cycles(n, n_cycles=24, k=3, seed=19)
    degrees = np.bincount(cycles_arr.ravel(), minlength=n) + 1
    cfg = CPMLConfig(
        d_in=d_in,
        d_layer=8,
        topology="route",
        tier_organization="capsule_soft",
        capsule_soft_router="hypergraph_conv",
        capsule_routing_iterations=1,
        capsule_hg_hidden=16,
        torch_compile_hypergraph=True,
    )
    model = CPML(cfg).eval()
    tier_of = torch.from_numpy(cfg.tier_spec.assign(degrees))
    x = torch.randn(n, d_in)
    cycles = torch.from_numpy(cycles_arr)
    signs_t = torch.from_numpy(signs)
    edges = torch.from_numpy(cycles_arr[:5, :2].copy())
    with torch.no_grad():
        for _ in range(2):
            y = model(x, cycles, signs_t, tier_of, edges)
    assert torch.isfinite(y).all()
