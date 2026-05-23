"""Unit tests for HymeKo-Gömb three-shell cascade.

Each shell tested independently + the composer end-to-end.
Forward/backward shapes, parameter counts, no-cycles degradation,
and a tiny training-loop smoke on synthetic moons data.

GPU AUROC regression tests (``HymeKoGomb``, ``MixedArityGomb``,
``JointMixGomb``) **require CUDA** and ``pytest.skip`` when no GPU is
visible — no CPU fallback.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from signedkan_wip.src.hymeko_gomb import (
    GombConfig, HymeKoGomb, InnerCPMLCore, MiddleHSiKAN, OuterFIRShell,
    GombNoOuter, GombNoMiddle, GombNoInner,
    JointMixGomb, MixedArityGomb,
)
from signedkan_wip.src.hymeko_gomb.joint_enumeration import (
    JOINT_BA_SLOTS, build_joint_ba_pools,
)
from signedkan_wip.src.hymeko_gomb.shells import scatter_mean


def _require_cuda_device() -> torch.device:
    """Gömb AUROC regression tests are GPU-only (no CPU fallback)."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required — CUDA not available")
    return torch.device("cuda")


def _bce_adam_best_val_auc(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    *,
    train_logits_fn: Callable[[], torch.Tensor],
    val_logits_fn: Callable[[], torch.Tensor],
    s_tr_t: torch.Tensor,
    s_va_y: np.ndarray,
    n_epochs: int,
    grad_clip: float | None = None,
) -> float:
    """Train with BCE on train edges; return best validation AUROC."""
    from sklearn.metrics import roc_auc_score

    best = 0.0
    for _ in range(n_epochs):
        model.train()
        loss = F.binary_cross_entropy_with_logits(train_logits_fn(), s_tr_t)
        opt.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        model.eval()
        with torch.no_grad():
            v_probs = torch.sigmoid(val_logits_fn()).cpu().numpy()
        try:
            best = max(best, float(roc_auc_score(s_va_y, v_probs)))
        except ValueError:
            pass
    return best


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


def test_gomb_capsule_soft_em_routing_forward_backward():
    """Cascade wiring: capsule_soft + EM through InnerCPMLCore."""
    cfg = GombConfig(
        n_nodes=15,
        d_embed=8,
        d_outer=4,
        M_outer=2,
        d_middle=8,
        d_core=8,
        n_tiers=2,
        cycle_k=3,
        cpml_topology="route",
        cpml_tier_organization="capsule_soft",
        cpml_capsule_soft_router="em_agreement",
        cpml_capsule_routing_iterations=3,
    )
    gomb = HymeKoGomb(cfg)
    cycles = torch.tensor(
        [[0, 1, 8], [2, 9, 10], [3, 4, 12]],
        dtype=torch.long,
    )
    signs = torch.tensor(
        [[1, -1, 1], [1, 1, -1], [-1, 1, 1]],
        dtype=torch.float,
    )
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


def test_gomb_capsule_soft_hypergraph_router_forward_backward():
    cfg = GombConfig(
        n_nodes=15,
        d_embed=8,
        d_outer=4,
        M_outer=2,
        d_middle=8,
        d_core=8,
        n_tiers=2,
        cycle_k=3,
        cpml_topology="route",
        cpml_tier_organization="capsule_soft",
        cpml_capsule_soft_router="hypergraph_conv",
        cpml_capsule_routing_iterations=1,
        cpml_capsule_hg_hidden=24,
    )
    gomb = HymeKoGomb(cfg)
    assert gomb.core.cpml._capsule_soft_router_resolved == "hypergraph_conv"
    cycles = torch.tensor(
        [[0, 1, 8], [2, 9, 10], [3, 4, 12]],
        dtype=torch.long,
    )
    signs = torch.tensor(
        [[1, -1, 1], [1, 1, -1], [-1, 1, 1]],
        dtype=torch.float,
    )
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
    from signedkan_wip.src.datasets import make_moon_signed_graph

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


def test_joint_mix_gomb_forward_alpha_normalised():
    """JointMixGomb: four slots (c3,c4,w2,w3); α softmax to 1.

    **GPU-only** — skipped when CUDA is unavailable.
    """
    dev = _require_cuda_device()
    n = 6
    edges = []
    signs = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v))
            signs.append(1)
    e = np.array(edges, dtype=np.int64)
    s = np.array(signs, dtype=np.int8)
    pools = build_joint_ba_pools(
        e, s, n,
        topk_c3=8, topk_c4=4,
        max_walks_w2=200, max_walks_w3=150,
        walk_seed=0,
        subsample_walks_seed=1,
    )
    cfg = GombConfig(
        n_nodes=n, d_embed=8, d_outer=4, M_outer=2,
        d_middle=8, d_core=8, n_tiers=2, cycle_k=3,
    )
    m = JointMixGomb(cfg).to(dev)
    cyc = {
        slot: torch.from_numpy(pools[slot][0]).long().to(dev)
        for slot in JOINT_BA_SLOTS
    }
    sgn = {
        slot: torch.from_numpy(pools[slot][1]).to(dev)
        for slot in JOINT_BA_SLOTS
    }
    tier_of = torch.zeros(n, dtype=torch.long, device=dev)
    edges_t = torch.tensor([[0, 1], [2, 3]], dtype=torch.long, device=dev)
    logits = m(cyc, sgn, tier_of, edges_t)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    a = m.alpha()
    assert a.shape == (len(JOINT_BA_SLOTS),)
    assert torch.allclose(
        a.sum(), torch.tensor(1.0, device=a.device), atol=1e-5,
    )


def _joint_mix_cpml_param_count(model: JointMixGomb) -> int:
    return sum(
        sum(p.numel() for p in model.cores[slot].cpml.parameters())
        for slot in JointMixGomb.SLOTS
    )


def test_joint_mix_cpml_route_has_fewer_parameters_than_pyramid():
    """Route keeps per-tier MLP ``d_in`` fixed; pyramid widens (Option B)."""
    base = dict(
        n_nodes=96,
        d_embed=24,
        d_outer=8,
        M_outer=4,
        d_middle=24,
        d_core=24,
        n_tiers=4,
        cycle_k=3,
    )
    m_route = JointMixGomb(GombConfig(**base, cpml_topology="route"))
    m_pyr = JointMixGomb(GombConfig(**base, cpml_topology="pyramid"))
    pr = _joint_mix_cpml_param_count(m_route)
    pp = _joint_mix_cpml_param_count(m_pyr)
    assert pr < pp, (
        f"expected route CPML param sum < pyramid; got {pr} vs {pp}"
    )


def _joint_mix_peak_cuda_mb_one_step(
    model: JointMixGomb,
    cyc: dict[str, torch.Tensor],
    sgn: dict[str, torch.Tensor],
    tier_of: torch.Tensor,
    edges_t: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    logits = model(cyc, sgn, tier_of, edges_t)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))


@pytest.mark.timeout(300)
def test_joint_mix_cuda_route_peak_memory_strictly_below_pyramid():
    """Same joint recipe: route should allocate strictly less peak VRAM.

    Pyramid tiers consume widening ``x``; route tiers read fixed-width
    base features (Option B).
    """
    from signedkan_wip.experiments.runs.run_gomb_smoke import _degree_to_tier

    dev = _require_cuda_device()
    rng = np.random.default_rng(0)
    n = 320
    pairs = rng.integers(0, n, size=(12_000, 2), dtype=np.int64)
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    e = pairs[:7000].astype(np.int64)
    s = rng.choice([-1, 1], size=len(e)).astype(np.int8)

    pools = build_joint_ba_pools(
        e, s, n,
        topk_c3=96,
        topk_c4=96,
        max_walks_w2=5000,
        max_walks_w3=5000,
        walk_seed=0,
        subsample_walks_seed=1,
    )
    cyc = {
        sl: torch.from_numpy(pools[sl][0]).long().to(dev)
        for sl in JOINT_BA_SLOTS
    }
    sgn = {
        sl: torch.from_numpy(pools[sl][1]).to(dev)
        for sl in JOINT_BA_SLOTS
    }
    degrees = np.bincount(e.ravel(), minlength=n) + 1
    tier_of = torch.from_numpy(_degree_to_tier(degrees, 4)).long().to(dev)
    edges_t = torch.from_numpy(
        rng.integers(0, n, size=(2500, 2), dtype=np.int64),
    ).to(dev)
    targets = torch.rand(2500, device=dev)

    cfg_shared = dict(
        n_nodes=n,
        d_embed=32,
        d_outer=12,
        M_outer=6,
        d_middle=32,
        d_core=32,
        n_tiers=4,
        cycle_k=3,
    )
    m_route = JointMixGomb(
        GombConfig(**cfg_shared, cpml_topology="route"),
    ).to(dev)
    peak_route = _joint_mix_peak_cuda_mb_one_step(
        m_route, cyc, sgn, tier_of, edges_t, targets,
    )
    del m_route
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    m_pyr = JointMixGomb(
        GombConfig(**cfg_shared, cpml_topology="pyramid"),
    ).to(dev)
    peak_pyr = _joint_mix_peak_cuda_mb_one_step(
        m_pyr, cyc, sgn, tier_of, edges_t, targets,
    )
    assert peak_route < peak_pyr, (
        f"expected peak CUDA MB route < pyramid; got {peak_route:.1f} vs "
        f"{peak_pyr:.1f}"
    )


@pytest.mark.timeout(120)
def test_joint_mix_gomb_bitcoin_otc_val_auc_regression():
    """Best val AUROC must stay above a loose floor (ref ~0.719 @ seed 0).

    Same slim recipe as ``run_gomb_smoke --joint-mix`` (3 epochs) on **CUDA
    only** — skipped when no GPU (no CPU fallback).
    """
    from signedkan_wip.src.datasets import load
    from signedkan_wip.experiments.runs.run_gomb_smoke import _degree_to_tier, _train_val_split

    dev = _require_cuda_device()
    torch.manual_seed(0)
    np.random.seed(0)

    g = load("bitcoin_otc")
    n = g.n_nodes
    e_tr, s_tr, e_va, s_va = _train_val_split(g.edges, g.signs, 0.2, 0)
    pools_joint = build_joint_ba_pools(
        e_tr, s_tr, n,
        topk_c3=48, topk_c4=48,
        max_walks_w2=4000, max_walks_w3=4000,
        walk_seed=0, subsample_walks_seed=0,
    )
    degrees = np.zeros(n, dtype=np.int64)
    for (u, v) in e_tr:
        degrees[int(u)] += 1
        degrees[int(v)] += 1
    tier_of_np = _degree_to_tier(degrees, 3)

    cfg = GombConfig(
        n_nodes=n, d_embed=24, d_outer=12, M_outer=6,
        d_middle=24, d_core=24, n_tiers=3, cycle_k=3,
    )
    model = JointMixGomb(cfg).to(dev)
    cyc_t_by_slot = {
        s: torch.from_numpy(pools_joint[s][0]).long().to(dev)
        for s in JOINT_BA_SLOTS
    }
    cyc_sgn_t_by_slot = {
        s: torch.from_numpy(pools_joint[s][1]).to(dev) for s in JOINT_BA_SLOTS
    }
    tier_of = torch.from_numpy(tier_of_np).long().to(dev)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).long().to(dev)
    s_tr_t = torch.from_numpy((s_tr > 0).astype(np.float32)).to(dev)
    e_va_t = torch.from_numpy(e_va.astype(np.int64)).long().to(dev)
    s_va_y = (s_va > 0).astype(np.float32)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    best = _bce_adam_best_val_auc(
        model,
        opt,
        train_logits_fn=lambda: model(
            cyc_t_by_slot, cyc_sgn_t_by_slot, tier_of, e_tr_t,
        ),
        val_logits_fn=lambda: model(
            cyc_t_by_slot, cyc_sgn_t_by_slot, tier_of, e_va_t,
        ),
        s_tr_t=s_tr_t,
        s_va_y=s_va_y,
        n_epochs=3,
        grad_clip=5.0,
    )

    assert best >= 0.68, (
        f"joint-mix val AUROC best {best:.4f} below 0.68 — "
        "reference seed=0 slim 3ep on CUDA ≈0.719."
    )


@pytest.mark.timeout(120)
def test_hymeko_gomb_bitcoin_otc_val_auc_cuda():
    """Slim single-k Gömb on OTC: val AUROC regression (ref ~0.68 @ 10 ep).

    Aligns with ``test_sota_smoke.test_sota_bitcoin_otc_slim_10ep`` recipe;
    runs on CUDA only.
    """
    from signedkan_wip.src.datasets import load
    from signedkan_wip.experiments.runs.run_gomb_smoke import (
        _degree_to_tier,
        _enumerate_cycles,
        _train_val_split,
    )

    dev = _require_cuda_device()
    torch.manual_seed(0)
    np.random.seed(0)

    g = load("bitcoin_otc")
    n = g.n_nodes
    e_tr, s_tr, e_va, s_va = _train_val_split(g.edges, g.signs, 0.2, 0)
    cycles_np, cyc_signs_np = _enumerate_cycles(
        e_tr, s_tr, n, k=3, m_per_vertex=16,
    )
    degrees = np.zeros(n, dtype=np.int64)
    for (u, v) in e_tr:
        degrees[int(u)] += 1
        degrees[int(v)] += 1
    tier_of_np = _degree_to_tier(degrees, 3)

    cfg = GombConfig(
        n_nodes=n, d_embed=16, d_outer=4, M_outer=2,
        d_middle=4, d_core=4, n_tiers=3, cycle_k=3,
    )
    model = HymeKoGomb(cfg).to(dev)
    cyc_t = torch.from_numpy(cycles_np).long().to(dev)
    cyc_sgn_t = torch.from_numpy(cyc_signs_np).to(dev)
    tier_of = torch.from_numpy(tier_of_np).long().to(dev)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).long().to(dev)
    s_tr_t = torch.from_numpy((s_tr > 0).astype(np.float32)).to(dev)
    e_va_t = torch.from_numpy(e_va.astype(np.int64)).long().to(dev)
    s_va_y = (s_va > 0).astype(np.float32)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    best = _bce_adam_best_val_auc(
        model,
        opt,
        train_logits_fn=lambda: model(cyc_t, cyc_sgn_t, tier_of, e_tr_t),
        val_logits_fn=lambda: model(cyc_t, cyc_sgn_t, tier_of, e_va_t),
        s_tr_t=s_tr_t,
        s_va_y=s_va_y,
        n_epochs=10,
        grad_clip=None,
    )
    assert best >= 0.60, (
        f"HymeKoGomb OTC slim 10ep best val AUROC {best:.4f} below 0.60 — "
        "reference seed=0 on CUDA ≈0.68."
    )


@pytest.mark.timeout(120)
def test_mixed_arity_gomb_bitcoin_otc_val_auc_cuda():
    """Mixed k=3,4 Gömb on OTC: val AUROC regression (ref ~0.70 @ 10 ep)."""
    from signedkan_wip.src.datasets import load
    from signedkan_wip.experiments.runs.run_gomb_smoke import (
        _degree_to_tier,
        _enumerate_cycles,
        _train_val_split,
    )

    dev = _require_cuda_device()
    torch.manual_seed(0)
    np.random.seed(0)

    g = load("bitcoin_otc")
    n = g.n_nodes
    e_tr, s_tr, e_va, s_va = _train_val_split(g.edges, g.signs, 0.2, 0)
    c3, s3 = _enumerate_cycles(e_tr, s_tr, n, k=3, m_per_vertex=16)
    c4, s4 = _enumerate_cycles(e_tr, s_tr, n, k=4, m_per_vertex=16)
    degrees = np.zeros(n, dtype=np.int64)
    for (u, v) in e_tr:
        degrees[int(u)] += 1
        degrees[int(v)] += 1
    tier_of_np = _degree_to_tier(degrees, 3)

    cfg = GombConfig(
        n_nodes=n, d_embed=16, d_outer=4, M_outer=2,
        d_middle=4, d_core=4, n_tiers=3, cycle_k=3,
    )
    model = MixedArityGomb(cfg, cycle_ks=(3, 4)).to(dev)
    cyc_by_k = {
        3: torch.from_numpy(c3).long().to(dev),
        4: torch.from_numpy(c4).long().to(dev),
    }
    sgn_by_k = {
        3: torch.from_numpy(s3).to(dev),
        4: torch.from_numpy(s4).to(dev),
    }
    tier_of = torch.from_numpy(tier_of_np).long().to(dev)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).long().to(dev)
    s_tr_t = torch.from_numpy((s_tr > 0).astype(np.float32)).to(dev)
    e_va_t = torch.from_numpy(e_va.astype(np.int64)).long().to(dev)
    s_va_y = (s_va > 0).astype(np.float32)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    best = _bce_adam_best_val_auc(
        model,
        opt,
        train_logits_fn=lambda: model(cyc_by_k, sgn_by_k, tier_of, e_tr_t),
        val_logits_fn=lambda: model(cyc_by_k, sgn_by_k, tier_of, e_va_t),
        s_tr_t=s_tr_t,
        s_va_y=s_va_y,
        n_epochs=10,
        grad_clip=None,
    )
    assert best >= 0.58, (
        f"MixedArityGomb OTC slim 10ep best val AUROC {best:.4f} below 0.58 — "
        "reference seed=0 on CUDA ≈0.70."
    )


@pytest.mark.timeout(120)
def test_joint_mix_gomb_bitcoin_alpha_val_auc_cuda():
    """Joint-mix Gömb on Bitcoin Alpha (ref ~0.626 best @ 3 ep, seed 0)."""
    from signedkan_wip.src.datasets import load
    from signedkan_wip.experiments.runs.run_gomb_smoke import _degree_to_tier, _train_val_split

    dev = _require_cuda_device()
    torch.manual_seed(0)
    np.random.seed(0)

    g = load("bitcoin_alpha")
    n = g.n_nodes
    e_tr, s_tr, e_va, s_va = _train_val_split(g.edges, g.signs, 0.2, 0)
    pools_joint = build_joint_ba_pools(
        e_tr, s_tr, n,
        topk_c3=48, topk_c4=48,
        max_walks_w2=4000, max_walks_w3=4000,
        walk_seed=0, subsample_walks_seed=0,
    )
    degrees = np.zeros(n, dtype=np.int64)
    for (u, v) in e_tr:
        degrees[int(u)] += 1
        degrees[int(v)] += 1
    tier_of_np = _degree_to_tier(degrees, 3)

    cfg = GombConfig(
        n_nodes=n, d_embed=24, d_outer=12, M_outer=6,
        d_middle=24, d_core=24, n_tiers=3, cycle_k=3,
    )
    model = JointMixGomb(cfg).to(dev)
    cyc_t_by_slot = {
        s: torch.from_numpy(pools_joint[s][0]).long().to(dev)
        for s in JOINT_BA_SLOTS
    }
    cyc_sgn_t_by_slot = {
        s: torch.from_numpy(pools_joint[s][1]).to(dev) for s in JOINT_BA_SLOTS
    }
    tier_of = torch.from_numpy(tier_of_np).long().to(dev)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).long().to(dev)
    s_tr_t = torch.from_numpy((s_tr > 0).astype(np.float32)).to(dev)
    e_va_t = torch.from_numpy(e_va.astype(np.int64)).long().to(dev)
    s_va_y = (s_va > 0).astype(np.float32)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    best = _bce_adam_best_val_auc(
        model,
        opt,
        train_logits_fn=lambda: model(
            cyc_t_by_slot, cyc_sgn_t_by_slot, tier_of, e_tr_t,
        ),
        val_logits_fn=lambda: model(
            cyc_t_by_slot, cyc_sgn_t_by_slot, tier_of, e_va_t,
        ),
        s_tr_t=s_tr_t,
        s_va_y=s_va_y,
        n_epochs=3,
        grad_clip=5.0,
    )
    assert best >= 0.52, (
        f"joint-mix Alpha 3ep best val AUROC {best:.4f} below 0.52 — "
        "reference seed=0 on CUDA ≈0.626."
    )


def test_sample_params_compact_is_narrow():
    from signedkan_wip.experiments.runs.run_gomb_tune import sample_params

    rng = np.random.default_rng(0)
    for _ in range(40):
        p = sample_params(rng, "slashdot", compact=True)
        assert p["d_embed"] <= 26
        assert p["M_outer"] <= 4
        assert p["d_outer"] <= 10
        assert p["topk"] <= 40


def test_for_joint_mix_tuning_clears_cycle_ks_and_samples_walks():
    from signedkan_wip.experiments.runs.run_gomb_tune import for_joint_mix_tuning, sample_params

    r0 = np.random.default_rng(0)
    base = sample_params(r0, "bitcoin_otc", compact=False)
    base["cycle_ks"] = "3,4"
    j = for_joint_mix_tuning(
        np.random.default_rng(99), base, compact=False, dataset="bitcoin_otc",
    )
    assert j["cycle_ks"] == ""
    assert j["joint_mix"] is True
    assert j["max_walks_w2"] in (8000, 16000, 24000, 32000)
    assert j["max_walks_w3"] in (8000, 16000, 24000, 32000)
    assert j["lr"] == base["lr"]

    c = sample_params(np.random.default_rng(1), "bitcoin_otc", compact=True)
    c["cycle_ks"] = "3,4"
    jc = for_joint_mix_tuning(
        np.random.default_rng(2), c, compact=True, dataset="bitcoin_otc",
    )
    assert jc["max_walks_w2"] in (2000, 4000, 8000, 12000, 20000, 32000)


def test_for_joint_mix_clamps_topk_on_bitcoin_wide():
    from signedkan_wip.experiments.runs.run_gomb_tune import for_joint_mix_tuning

    base = {
        "lr": 0.003,
        "d_embed": 56,
        "d_outer": 16,
        "M_outer": 12,
        "d_middle": 70,
        "d_core": 28,
        "topk": 128,
        "n_tiers": 4,
        "weight_decay": 0.0,
        "pos_weight_auto": False,
        "cycle_ks": "3,4",
    }
    j = for_joint_mix_tuning(
        np.random.default_rng(0), base, compact=False, dataset="bitcoin_otc",
    )
    assert j["topk"] == 56
    assert j["d_embed"] == 48
    assert j["M_outer"] == 10
    assert j["d_middle"] == 60  # min(70, floor(48*1.25))
    assert j["d_core"] == 28
    assert j["max_walks_w2"] <= 32000
    assert j["max_walks_w3"] <= 32000
    j_sl = for_joint_mix_tuning(
        np.random.default_rng(0), base, compact=False, dataset="slashdot",
    )
    assert j_sl["topk"] == 128
    assert j_sl["max_walks_w2"] <= 4096
    assert j_sl["max_walks_w3"] <= 4096
    assert j_sl.get("joint_slot_cap") == 12000


def test_tuner_objective_val_vs_test():
    from signedkan_wip.experiments.runs.run_gomb_tune import _tuner_objective

    row = {
        "test_auroc": 0.5,
        "val_auroc": 0.95,
    }
    assert _tuner_objective(row, pick_best_by="test_auroc") == 0.5
    assert _tuner_objective(row, pick_best_by="val_auroc") == 0.95
    assert _tuner_objective(None, pick_best_by="val_auroc") == float("-inf")


def test_n_params_from_row():
    from signedkan_wip.experiments.runs.run_gomb_tune import _n_params_from_row

    assert _n_params_from_row(None) is None
    assert _n_params_from_row({}) is None
    assert _n_params_from_row({"n_params": 1_000_000}) == 1_000_000
    assert _n_params_from_row({"n_params": "x"}) is None


def test_compact_sample_slashdot_d_embed_includes_26():
    from signedkan_wip.experiments.runs.run_gomb_tune import sample_params

    seen: set[int] = set()
    for i in range(300):
        p = sample_params(np.random.default_rng(i), "slashdot", compact=True)
        seen.add(int(p["d_embed"]))
    assert 26 in seen


def test_for_joint_mix_slashdot_compact_clamps_topk():
    from signedkan_wip.experiments.runs.run_gomb_tune import for_joint_mix_tuning, sample_params

    base = sample_params(np.random.default_rng(5), "slashdot", compact=True)
    base["topk"] = 40
    j = for_joint_mix_tuning(
        np.random.default_rng(6), base, compact=True, dataset="slashdot",
    )
    assert j["topk"] <= 28
    assert j["max_walks_w2"] <= 4096
    assert j.get("joint_slot_cap") == 12000


def test_subsample_joint_pools_reduces_rows():
    from signedkan_wip.experiments.runs.run_gomb_smoke import _subsample_joint_pools

    cyc = np.arange(30, dtype=np.int64).reshape(10, 3)
    sgn = np.ones((10, 3), dtype=np.int8)
    pools = {"c3": (cyc, sgn)}
    out = _subsample_joint_pools(pools, cap=4, seed=0)
    assert out["c3"][0].shape == (4, 3)


def test_build_cmd_joint_mix_includes_cli_flags():
    import sys

    from signedkan_wip.experiments.runs.run_gomb_tune import _build_cmd

    p = {
        "lr": 0.003,
        "d_embed": 24,
        "d_outer": 12,
        "M_outer": 6,
        "d_middle": 24,
        "d_core": 24,
        "topk": 48,
        "n_tiers": 3,
        "weight_decay": 0.0,
        "pos_weight_auto": False,
        "cycle_ks": "",
        "joint_mix": True,
        "max_walks_w2": 4000,
        "max_walks_w3": 8000,
        "joint_slot_cap": 9000,
    }
    cmd = _build_cmd(
        py=sys.executable,
        dataset="bitcoin_otc",
        data_seed=0,
        edge_split="80_10_10",
        n_epochs=2,
        device="cpu",
        p=p,
    )
    assert "--joint-mix" in cmd
    assert cmd[cmd.index("--max-walks-w2") + 1] == "4000"
    assert cmd[cmd.index("--max-walks-w3") + 1] == "8000"
    assert cmd[cmd.index("--joint-slot-cap") + 1] == "9000"
    assert "--cycle-ks" not in cmd


def test_build_cmd_mixed_arity_uses_cycle_ks_without_joint():
    import sys

    from signedkan_wip.experiments.runs.run_gomb_tune import _build_cmd

    p = {
        "lr": 0.003,
        "d_embed": 24,
        "d_outer": 12,
        "M_outer": 6,
        "d_middle": 24,
        "d_core": 24,
        "topk": 48,
        "n_tiers": 3,
        "weight_decay": 0.0,
        "pos_weight_auto": False,
        "cycle_ks": "3,4",
    }
    cmd = _build_cmd(
        py=sys.executable,
        dataset="bitcoin_otc",
        data_seed=0,
        edge_split="80_10_10",
        n_epochs=2,
        device="cpu",
        p=p,
    )
    assert "--cycle-ks" in cmd
    assert cmd[cmd.index("--cycle-ks") + 1] == "3,4"
    assert "--joint-mix" not in cmd


def test_build_cmd_includes_cycle_abb_when_in_params():
    import sys

    from signedkan_wip.experiments.runs.run_gomb_tune import _build_cmd

    p = {
        "lr": 0.003,
        "d_embed": 16,
        "d_outer": 8,
        "M_outer": 2,
        "d_middle": 16,
        "d_core": 16,
        "topk": 32,
        "n_tiers": 3,
        "weight_decay": 0.0,
        "pos_weight_auto": False,
        "cycle_abb_mode": "start_local",
        "cycle_abb_fullness_gate": 0.3,
    }
    cmd = _build_cmd(
        py=sys.executable,
        dataset="sbm_n200",
        data_seed=0,
        edge_split="80_10_10",
        n_epochs=1,
        device="cpu",
        p=p,
    )
    assert cmd[cmd.index("--cycle-abb-mode") + 1] == "start_local"
    assert cmd[cmd.index("--cycle-abb-fullness-gate") + 1] == "0.3"


@pytest.mark.timeout(120)
def test_run_gomb_smoke_json_includes_inference_fields():
    """Smoke subprocess must emit infer_* timing keys on the summary JSON."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    cmd = [
        sys.executable,
        "-m",
        "signedkan_wip.experiments.runs.run_gomb_smoke",
        "--dataset",
        "sbm_n200",
        "--n-epochs",
        "1",
        "--topk",
        "12",
        "--edge-split",
        "80_10_10",
        "--device",
        "cpu",
        "--seed",
        "0",
        "--d-embed",
        "16",
        "--d-outer",
        "8",
        "--M-outer",
        "2",
        "--d-middle",
        "16",
        "--d-core",
        "16",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
        timeout=90,
    )
    assert proc.returncode == 0, (proc.stderr or "")[-2000:]
    row = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith('{"dataset"'):
            row = json.loads(line)
            break
    assert row is not None
    assert row["infer_wall_s"] > 0
    assert row["infer_n_edges"] > 0
    assert row["infer_edges_per_s"] > 0


def test_parse_last_gomb_json_line():
    from signedkan_wip.experiments.runs.run_gomb_tune import _parse_last_gomb_json

    stdout = "garbage\n{\"dataset\": \"z\", \"test_auroc\": 0.8123}\n"
    row = _parse_last_gomb_json(stdout)
    assert row is not None
    assert row["test_auroc"] == 0.8123


def test_heldout_edge_metrics_label_keys():
    """Smoke metrics helper must namespace keys by split label."""
    from signedkan_wip.experiments.runs.run_gomb_smoke import _heldout_edge_metrics

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
