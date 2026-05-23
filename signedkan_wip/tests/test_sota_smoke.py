"""Fast SOTA-grade smoke tests — verify our headline numbers reproduce
in seconds, NOT hours.

These are *fast* protocol-validation runs, not paper-grade benchmarks.
Each test:
  - uses a small Bitcoin OTC / Alpha config (~5 s / ~2 s wall on CUDA)
  - locks in a known-good val_auc range from our measured 5-seed runs
  - acts as a regression gate: if a refactor breaks the model, AUC drops
    visibly and the test fails

Headline reference values (from real measurements 2026-05-11):

    bitcoin_otc, full Gömb (default cfg, 50 ep, seed 0): val_auc ≈ 0.92
    bitcoin_otc, slim Gömb (10 ep, seed 0, cpu):         val_auc ≈ 0.68
    bitcoin_alpha (40 ep, seed 0, slim, cpu, m_per_vertex=32):       val_auc ≈ 0.82

These reflect *non-paper-grade* slim configs chosen so that CI / dev
machines can run them in ≤ 20 s. Paper-grade reproduction lives in the
5-seed scripts under `signedkan_wip/experiments/`.

Run:
    pytest signedkan_wip/tests/test_sota_smoke.py -v
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from signedkan_wip.src.datasets import load
from signedkan_wip.src.hymeko_gomb import GombConfig, HymeKoGomb


# ─── Shared fast train loop (no scaffold duplication) ───────────────


def _enum_cycles(g, k: int = 3, m_per_vertex: int = 16):
    """Tiny CPU cycle enum via the unified strategy entry."""
    import hymeko
    eu = np.ascontiguousarray(g.edges[:, 0], dtype=np.uint32)
    ev = np.ascontiguousarray(g.edges[:, 1], dtype=np.uint32)
    es = np.ascontiguousarray(g.signs,        dtype=np.int8)
    cycles, _ = hymeko.enumerate_cycles_rs(
        eu, ev, es, g.n_nodes, k, m_per_vertex,
        score_kind="fraction_negative", pruner_kind="none",
        filter_kind="none",
    )
    # cycles is uint32 from the Rust side; torch indexers want int64.
    return np.asarray(cycles, dtype=np.int64)


def _train_gomb(
    dataset: str, *, n_epochs: int = 10, seed: int = 0,
    d_embed: int = 16, d_outer: int = 4, M_outer: int = 2,
    d_middle: int = 4, d_core: int = 4, k: int = 3,
    m_per_vertex: int = 16,
) -> tuple[float, float]:
    """Returns (val_auc_best, wall_s)."""
    from sklearn.metrics import roc_auc_score
    import torch.nn.functional as F

    torch.manual_seed(seed); np.random.seed(seed)
    g = load(dataset)
    cycles_np = _enum_cycles(g, k=k, m_per_vertex=m_per_vertex)

    # Derive signs from edge signs (same scheme as the smoke runner).
    sign_of: dict[tuple[int, int], int] = {}
    for (u, v), s in zip(g.edges, g.signs):
        sign_of[(int(u), int(v))] = int(s)
        sign_of[(int(v), int(u))] = int(s)
    cyc_signs_np = np.zeros_like(cycles_np, dtype=np.int8)
    for ci, cycle in enumerate(cycles_np):
        for j in range(k):
            u, v_ = int(cycle[j]), int(cycle[(j + 1) % k])
            cyc_signs_np[ci, j] = sign_of.get((u, v_), 1)

    # Train/val split (val_frac=0.2)
    rng = np.random.default_rng(seed)
    n_e = g.edges.shape[0]
    perm = rng.permutation(n_e); n_val = int(0.2 * n_e)
    e_tr, s_tr = g.edges[perm[n_val:]], g.signs[perm[n_val:]]
    e_va, s_va = g.edges[perm[:n_val]], g.signs[perm[:n_val]]

    # Degree-percentile tier assignment
    degrees = np.zeros(g.n_nodes, dtype=np.int64)
    for (u, v) in e_tr:
        degrees[int(u)] += 1; degrees[int(v)] += 1
    order = np.argsort(degrees, kind="stable")
    ranks = np.empty(g.n_nodes, dtype=np.float64)
    ranks[order] = np.arange(g.n_nodes) / max(1, g.n_nodes - 1)
    tier_of_np = np.where(ranks <= 1/3, 0, np.where(ranks <= 2/3, 1, 2)).astype(np.int64)

    cfg = GombConfig(
        n_nodes=g.n_nodes, d_embed=d_embed, d_outer=d_outer,
        M_outer=M_outer, d_middle=d_middle, d_core=d_core,
        n_tiers=3, cycle_k=k,
    )
    model = HymeKoGomb(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    cyc_t      = torch.from_numpy(cycles_np)
    cyc_sgn_t  = torch.from_numpy(cyc_signs_np)
    tier_of    = torch.from_numpy(tier_of_np)
    e_tr_t     = torch.from_numpy(e_tr.astype(np.int64))
    s_tr_t     = torch.from_numpy((s_tr > 0).astype(np.float32))
    e_va_t     = torch.from_numpy(e_va.astype(np.int64))
    s_va_y     = (s_va > 0).astype(np.float32)

    best = 0.0
    t0 = time.perf_counter()
    for _ in range(n_epochs):
        model.train()
        scores = model(cyc_t, cyc_sgn_t, tier_of, e_tr_t)
        loss = F.binary_cross_entropy_with_logits(scores, s_tr_t)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v_probs = torch.sigmoid(
                model(cyc_t, cyc_sgn_t, tier_of, e_va_t)
            ).cpu().numpy()
        try:
            best = max(best, float(roc_auc_score(s_va_y, v_probs)))
        except ValueError:
            pass
    return best, time.perf_counter() - t0


# ─── Tests ──────────────────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_sota_bitcoin_otc_slim_10ep():
    """Bitcoin OTC slim Gömb @ 10ep on CPU should hit ≥ 0.60 AUC in ≤ 30 s.

    Reference: 5-seed slim run got mean AUC 0.68 (lower bound 0.60 is
    a comfortable regression-gate margin — anything below that means
    something is broken).
    """
    auc, wall = _train_gomb(
        "bitcoin_otc", n_epochs=10, seed=0,
        d_embed=16, d_outer=4, M_outer=2,
        d_middle=4, d_core=4, m_per_vertex=16,
    )
    print(f"\n  bitcoin_otc slim 10ep: AUC={auc:.4f}  wall={wall:.1f}s")
    assert wall < 30, f"too slow: {wall:.1f}s"
    assert auc >= 0.60, (
        f"AUC {auc:.4f} below regression threshold 0.60 — "
        f"5-seed reference: 0.68. Something is broken."
    )


@pytest.mark.timeout(120)
def test_sota_bitcoin_alpha_slim_40ep():
    """Bitcoin Alpha slim Gömb @ 40ep should hit ≥ 0.75 AUC in ≤ 30 s.

    Bitcoin Alpha is smaller (~3.8K nodes, ~24K edges) and easier than
    OTC. With enough cycle samples and epochs the slim CPU config clears
    0.75 comfortably (reference ~0.82 @ 40ep, m_per_vertex=32).
    """
    auc, wall = _train_gomb(
        "bitcoin_alpha", n_epochs=40, seed=0,
        d_embed=16, d_outer=4, M_outer=2,
        d_middle=8, d_core=8, m_per_vertex=32,
    )
    print(f"\n  bitcoin_alpha slim 40ep: AUC={auc:.4f}  wall={wall:.1f}s")
    assert wall < 30, f"too slow: {wall:.1f}s"
    assert auc >= 0.75, (
        f"AUC {auc:.4f} below regression threshold 0.75 — "
        f"BA published baselines hit 0.83-0.90."
    )


@pytest.mark.timeout(240)
def test_sota_bitcoin_otc_default_30ep():
    """Bitcoin OTC default Gömb @ 30ep should hit ≥ 0.85 AUC in ≤ 60 s.

    Reference: the original 5-seed at default config (50ep, cuda)
    got 0.9118 ± 0.0089. Reduced to 30ep on cpu we lose some
    convergence and a lot of speed, but should still clear 0.85.
    """
    auc, wall = _train_gomb(
        "bitcoin_otc", n_epochs=30, seed=0,
        d_embed=32, d_outer=16, M_outer=8,
        d_middle=32, d_core=32, m_per_vertex=64,
    )
    print(f"\n  bitcoin_otc default 30ep: AUC={auc:.4f}  wall={wall:.1f}s")
    assert wall < 60, f"too slow: {wall:.1f}s"
    assert auc >= 0.85, (
        f"AUC {auc:.4f} below regression threshold 0.85 — "
        f"5-seed reference at 50ep was 0.91. We allow some loss "
        f"from 50ep → 30ep."
    )
