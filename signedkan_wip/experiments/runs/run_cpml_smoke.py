"""CPML smoke runner — Bitcoin OTC + simple cycle pool + 10-epoch train.

Validates CPML end-to-end on real signed-graph data with the
feasibility-stub TierAggregator (MLP). If this produces a non-trivial
AUC (> 0.55, i.e. better than random), the topology is alive and we
can promote to the real HSiKAN swap.

Usage:
    python -m signedkan_wip.experiments.runs.run_cpml_smoke \
        --dataset bitcoin_otc --seed 0 --n-epochs 10

This is *not* meant to beat anything — just to confirm:
  1. The CPML model trains without crashing on real data.
  2. Loss decreases.
  3. AUC > 0.55 (a sign the topology learns something).

Promote to the real HSiKAN swap (replacing TierAggregator with
MixedAritySignedKAN) only after this smoke passes.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from signedkan_wip.src.cpml import CPML, CPMLConfig, TierSpec
from signedkan_wip.src.datasets import load


def _build_cycle_pool(
    edges: np.ndarray, signs: np.ndarray, n_vertices: int,
    max_cycles_per_vertex: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Cheap-and-cheerful triangle pool: for each edge (u, v), look at
    the intersection of N(u) and N(v) and pick up to
    max_cycles_per_vertex triangles incident to u.

    Returns cycles (M, 3) and signs (M, 3).  Smoke-quality, not
    publication-quality; the real cycle enumerator runs at the full
    HSiKAN integration step.
    """
    # Build adjacency dict with edge-index lookup.
    adj: dict[int, set[int]] = {v: set() for v in range(n_vertices)}
    sign_of: dict[tuple[int, int], int] = {}
    for ei, ((u, v), s) in enumerate(zip(edges, signs)):
        u, v = int(u), int(v)
        adj[u].add(v); adj[v].add(u)
        sign_of[(u, v)] = int(s)
        sign_of[(v, u)] = int(s)

    cycles: list[tuple[int, int, int]] = []
    cyc_signs: list[tuple[int, int, int]] = []
    seen: set[frozenset] = set()
    for v in range(n_vertices):
        cnt = 0
        for u in adj[v]:
            if u <= v:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w <= u:
                    continue
                key = frozenset({u, v, w})
                if key in seen:
                    continue
                seen.add(key)
                s_uv = sign_of[(u, v)]
                s_vw = sign_of[(v, w)]
                s_uw = sign_of[(u, w)]
                cycles.append((u, v, w))
                cyc_signs.append((s_uv, s_vw, s_uw))
                cnt += 1
                if cnt >= max_cycles_per_vertex:
                    break
            if cnt >= max_cycles_per_vertex:
                break
    cyc_arr = np.array(cycles, dtype=np.int64)
    sgn_arr = np.array(cyc_signs, dtype=np.int8)
    return cyc_arr, sgn_arr


def _train_val_split(
    edges: np.ndarray, signs: np.ndarray, val_frac: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random edge-level train/val split."""
    rng = np.random.default_rng(seed)
    n = edges.shape[0]
    perm = rng.permutation(n)
    n_val = int(val_frac * n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return edges[tr_idx], signs[tr_idx], edges[val_idx], signs[val_idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_otc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--n-tiers", type=int, default=3,
                    choices=[1, 2, 3, 4, 5])
    ap.add_argument("--d-layer", type=int, default=16)
    ap.add_argument("--d-in", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.perf_counter()
    g = load(args.dataset)
    n = g.n_nodes
    print(f"[load] {args.dataset}: |V|={n}, |E|={len(g.edges)}")

    # Train/val split.
    e_tr, s_tr, e_va, s_va = _train_val_split(
        g.edges, g.signs, val_frac=args.val_frac, seed=args.seed,
    )

    # Cycle pool built from TRAIN edges only (no leakage).
    cyc, cyc_sgn = _build_cycle_pool(e_tr, s_tr, n, max_cycles_per_vertex=8)
    print(f"[cycles] {cyc.shape[0]} triangles from train edges "
          f"in {time.perf_counter()-t0:.1f}s")
    if cyc.shape[0] == 0:
        raise SystemExit("smoke failed: no triangles in training graph")

    # Degrees on training graph only.
    degrees = np.zeros(n, dtype=np.int64)
    for (u, v) in e_tr:
        degrees[int(u)] += 1
        degrees[int(v)] += 1

    cuts = tuple(np.linspace(0.0, 1.0, args.n_tiers + 1).tolist())
    cfg = CPMLConfig(
        tier_spec=TierSpec(cuts=cuts),
        d_in=args.d_in, d_layer=args.d_layer,
    )
    model = CPML(cfg).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[model] CPML L={cfg.tier_spec.L} cuts={cuts} "
          f"d_in={cfg.d_in} d_layer={cfg.d_layer} "
          f"final_dim={model.in_dims[-1]} "
          f"n_params={sum(p.numel() for p in model.parameters())}")

    # Tensors.
    device = torch.device(args.device)
    node_features = torch.randn(n, cfg.d_in, device=device) * 0.1
    cyc_t = torch.from_numpy(cyc).to(device)
    cyc_sgn_t = torch.from_numpy(cyc_sgn).to(device)
    tier_of = torch.from_numpy(cfg.tier_spec.assign(degrees)).to(device)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    s_tr_t = torch.from_numpy((s_tr > 0).astype(np.float32)).to(device)
    e_va_t = torch.from_numpy(e_va.astype(np.int64)).to(device)
    s_va_y = (s_va > 0).astype(np.float32)

    print(f"[tiers] sizes: " + ", ".join(
        f"T_{i}={int((tier_of == i).sum())}" for i in range(cfg.tier_spec.L)
    ))

    losses: list[float] = []
    val_aucs: list[float] = []
    for ep in range(args.n_epochs):
        model.train()
        scores = model(node_features, cyc_t, cyc_sgn_t, tier_of, e_tr_t)
        loss = F.binary_cross_entropy_with_logits(scores, s_tr_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))

        # Val AUC.
        model.eval()
        with torch.no_grad():
            val_scores = model(node_features, cyc_t, cyc_sgn_t, tier_of, e_va_t)
            val_probs = torch.sigmoid(val_scores).cpu().numpy()
        try:
            val_auc = roc_auc_score(s_va_y, val_probs)
        except ValueError:
            val_auc = float("nan")
        val_aucs.append(float(val_auc))
        print(f"  ep {ep:02d}  loss={loss.item():.4f}  val_auc={val_auc:.4f}")

    wall = time.perf_counter() - t0
    out = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model": "CPML",
        "L": cfg.tier_spec.L,
        "cuts": list(cuts),
        "d_in": cfg.d_in,
        "d_layer": cfg.d_layer,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "n_train_edges": int(e_tr.shape[0]),
        "n_val_edges": int(e_va.shape[0]),
        "n_cycles": int(cyc.shape[0]),
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "val_auc_start": val_aucs[0],
        "val_auc_end": val_aucs[-1],
        "val_auc_best": max(val_aucs),
        "wall_s": wall,
        "tier_sizes": [int((tier_of == i).sum()) for i in range(cfg.tier_spec.L)],
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
