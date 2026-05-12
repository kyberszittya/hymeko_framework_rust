"""CPML real-data runner — uses the Rust cycle enumerator + learnable
per-vertex embeddings, so the CPML architecture is given a fair shot
at signed-graph link prediction.

Differences from `run_cpml_smoke.py`:
  * Real cycle pool via `hymeko.enumerate_top_k_per_vertex_cycles_signed_*`
    (up to ~100K cycles, not the cheap 3K triangle pool).
  * Learnable per-vertex embedding (not random features) — the model
    can develop a vertex identity.
  * 50 epochs by default (not 10).
  * Wider d_layer (32) to match a small-but-not-tiny capacity budget.
  * Optional walks added to the cycle pool (mixed-arity at CPML
    tier-restricted scale, just like Slashdot SOTA but per-tier).

Compared to `run_final_cell.py --model HSiKAN` (the flat kitchen-sink
recipe), this is the apples-to-apples 5-seed-eligible comparator.

Usage:
    python -m signedkan_wip.src.run_cpml_real \
        --dataset bitcoin_otc --seed 0 --n-epochs 50 \
        --n-tiers 3 --d-layer 32 --topk 64
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import hymeko

from .cpml import CPML, CPMLConfig, TierSpec
from .datasets import load


def _enumerate_cycles(
    edges: np.ndarray, signs: np.ndarray, n: int,
    k: int = 3, m_per_vertex: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the Rust per-vertex top-m enumerator (with ABB OFF) to
    build a rich cycle pool.

    Returns:
        cycles : (M, k) int64
        signs  : (M, k) int8 — boundary-edge signs around each cycle
    """
    eu = np.ascontiguousarray(edges[:, 0], dtype=np.uint32)
    ev = np.ascontiguousarray(edges[:, 1], dtype=np.uint32)
    es = np.ascontiguousarray(signs, dtype=np.int8)
    arr, _scores = hymeko.enumerate_cycles_rs(
        eu, ev, es, n, k, m_per_vertex,
        score_kind="fraction_negative", pruner_kind="none",
        filter_kind="none",
    )
    cycles = np.asarray(arr, dtype=np.int64)
    # Look up edge signs along each cycle's boundary.
    sign_of: dict[tuple[int, int], int] = {}
    for (u, v), s in zip(edges, signs):
        sign_of[(int(u), int(v))] = int(s)
        sign_of[(int(v), int(u))] = int(s)
    cyc_signs = np.zeros_like(cycles, dtype=np.int8)
    for ci, cycle in enumerate(cycles):
        for j in range(k):
            u = int(cycle[j])
            v = int(cycle[(j + 1) % k])
            cyc_signs[ci, j] = sign_of.get((u, v), 1)
    return cycles, cyc_signs


def _train_val_split(
    edges: np.ndarray, signs: np.ndarray, val_frac: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = edges.shape[0]
    perm = rng.permutation(n)
    n_val = int(val_frac * n)
    return (edges[perm[n_val:]], signs[perm[n_val:]],
            edges[perm[:n_val]], signs[perm[:n_val]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_otc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--n-tiers", type=int, default=3,
                    choices=[1, 2, 3, 4, 5])
    ap.add_argument("--d-layer", type=int, default=32)
    ap.add_argument("--d-in", type=int, default=32,
                    help="Per-vertex learnable embedding dim.")
    ap.add_argument("--k", type=int, default=3,
                    help="Cycle length for enumeration (3, 4, ...).")
    ap.add_argument("--topk", type=int, default=64,
                    help="m_per_vertex for cycle enumeration.")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    t0 = time.perf_counter()
    g = load(args.dataset)
    n = g.n_nodes
    print(f"[load] {args.dataset}: |V|={n}, |E|={len(g.edges)}", flush=True)

    # Edge-level train/val split.
    e_tr, s_tr, e_va, s_va = _train_val_split(
        g.edges, g.signs, val_frac=args.val_frac, seed=args.seed,
    )

    # Enumerate cycles on TRAIN graph only (no leakage).
    t_enum = time.perf_counter()
    cycles, cyc_signs = _enumerate_cycles(
        e_tr, s_tr, n, k=args.k, m_per_vertex=args.topk,
    )
    print(f"[cycles] {cycles.shape[0]} k={args.k} cycles "
          f"in {time.perf_counter()-t_enum:.1f}s", flush=True)
    if cycles.shape[0] == 0:
        raise SystemExit("no cycles in train graph; can't train CPML")

    # Degree on training graph.
    degrees = np.zeros(n, dtype=np.int64)
    for (u, v) in e_tr:
        degrees[int(u)] += 1
        degrees[int(v)] += 1

    cuts = tuple(np.linspace(0.0, 1.0, args.n_tiers + 1).tolist())
    cfg = CPMLConfig(
        tier_spec=TierSpec(cuts=cuts),
        d_in=args.d_in, d_layer=args.d_layer,
    )

    # Learnable per-vertex embedding (replaces random features).
    node_embed = torch.nn.Embedding(n, cfg.d_in).to(device)
    torch.nn.init.normal_(node_embed.weight, std=0.1)
    model = CPML(cfg).to(device)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(node_embed.parameters()),
        lr=args.lr,
    )
    n_params = sum(p.numel() for p in model.parameters()) + \
               sum(p.numel() for p in node_embed.parameters())
    print(f"[model] CPML L={cfg.tier_spec.L} cuts={cuts} "
          f"d_in={cfg.d_in} d_layer={cfg.d_layer} "
          f"n_params={n_params}", flush=True)

    cyc_t = torch.from_numpy(cycles).to(device)
    cyc_sgn_t = torch.from_numpy(cyc_signs).to(device)
    tier_of = torch.from_numpy(cfg.tier_spec.assign(degrees)).to(device)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    s_tr_t = torch.from_numpy((s_tr > 0).astype(np.float32)).to(device)
    e_va_t = torch.from_numpy(e_va.astype(np.int64)).to(device)
    s_va_y = (s_va > 0).astype(np.float32)

    sizes = [int((tier_of == i).sum()) for i in range(cfg.tier_spec.L)]
    print(f"[tiers] sizes: {sizes}", flush=True)

    losses: list[float] = []
    val_aucs: list[float] = []
    best_auc = 0.0
    for ep in range(args.n_epochs):
        model.train()
        node_features = node_embed.weight                  # (N, d_in)
        scores = model(node_features, cyc_t, cyc_sgn_t, tier_of, e_tr_t)
        loss = F.binary_cross_entropy_with_logits(scores, s_tr_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(node_embed.parameters()), 5.0,
        )
        opt.step()
        losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            val_scores = model(node_embed.weight, cyc_t, cyc_sgn_t,
                                 tier_of, e_va_t)
            val_probs = torch.sigmoid(val_scores).cpu().numpy()
        try:
            val_auc = roc_auc_score(s_va_y, val_probs)
        except ValueError:
            val_auc = float("nan")
        val_aucs.append(float(val_auc))
        best_auc = max(best_auc, val_auc)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep:02d}  loss={loss.item():.4f}  "
                  f"val_auc={val_auc:.4f}  best={best_auc:.4f}",
                  flush=True)

    wall = time.perf_counter() - t0
    out = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model": "CPML-real",
        "L": cfg.tier_spec.L,
        "cuts": list(cuts),
        "d_in": cfg.d_in,
        "d_layer": cfg.d_layer,
        "k": args.k,
        "topk": args.topk,
        "n_params": int(n_params),
        "n_train_edges": int(e_tr.shape[0]),
        "n_val_edges": int(e_va.shape[0]),
        "n_cycles": int(cycles.shape[0]),
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "val_auc_start": val_aucs[0],
        "val_auc_end": val_aucs[-1],
        "val_auc_best": best_auc,
        "wall_s": wall,
        "tier_sizes": sizes,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
