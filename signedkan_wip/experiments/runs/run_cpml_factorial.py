"""CPML × HSiKAN orthogonal 2x2 factorial on a fixed dataset.

Per-cell measurement:
    cell ∈ { (L, aggregator) : L ∈ {1, 3}, aggregator ∈ {mlp, hsikan} }
Plus a no-structure control per aggregator (cycles=∅).

All cells use IDENTICAL:
    * dataset, seed, 80/20 random val split
    * per-vertex learnable embedding (same init, same dim)
    * training schedule (same lr, same epochs, same optimizer)
    * edge predictor head architecture

Yields a clean factorial decomposition:
    topology effect (L=3 vs L=1) at fixed aggregator
    aggregator effect (HSiKAN vs MLP) at fixed topology
    interaction effect
    no-structure floor (embedding-alone baseline)

Usage:
    python -m signedkan_wip.experiments.runs.run_cpml_factorial \
        --dataset bitcoin_otc --seed 0 --n-epochs 50
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import hymeko

from signedkan_wip.src.core.cpml import CPML, CPMLConfig, TierSpec
from signedkan_wip.src.datasets import load


def _enumerate_cycles(
    edges: np.ndarray, signs: np.ndarray, n: int,
    k: int = 3, m_per_vertex: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    eu = np.ascontiguousarray(edges[:, 0], dtype=np.uint32)
    ev = np.ascontiguousarray(edges[:, 1], dtype=np.uint32)
    es = np.ascontiguousarray(signs, dtype=np.int8)
    arr, _ = hymeko.enumerate_cycles_rs(
        eu, ev, es, n, k, m_per_vertex,
        score_kind="fraction_negative", pruner_kind="none",
        filter_kind="none",
    )
    cycles = np.asarray(arr, dtype=np.int64)
    sign_of: dict[tuple[int, int], int] = {}
    for (u, v), s in zip(edges, signs):
        sign_of[(int(u), int(v))] = int(s)
        sign_of[(int(v), int(u))] = int(s)
    cyc_signs = np.zeros_like(cycles, dtype=np.int8)
    for ci, cycle in enumerate(cycles):
        for j in range(k):
            u, v = int(cycle[j]), int(cycle[(j + 1) % k])
            cyc_signs[ci, j] = sign_of.get((u, v), 1)
    return cycles, cyc_signs


def _train_val_split(edges, signs, val_frac, seed):
    rng = np.random.default_rng(seed)
    n = edges.shape[0]
    perm = rng.permutation(n)
    n_val = int(val_frac * n)
    return (edges[perm[n_val:]], signs[perm[n_val:]],
            edges[perm[:n_val]], signs[perm[:n_val]])


def _run_cell(
    label: str, L: int, aggregator: str, no_cycles: bool,
    n: int, d_in: int, d_layer: int,
    initial_embedding: torch.Tensor,
    cycles_np: np.ndarray, cyc_signs_np: np.ndarray,
    degrees: np.ndarray,
    e_tr: np.ndarray, s_tr: np.ndarray,
    e_va: np.ndarray, s_va: np.ndarray,
    n_epochs: int, lr: float, device: str,
) -> dict:
    torch.manual_seed(0)
    cuts = tuple(np.linspace(0.0, 1.0, L + 1).tolist())
    cfg = CPMLConfig(
        tier_spec=TierSpec(cuts=cuts),
        d_in=d_in, d_layer=d_layer,
        aggregator_kind=aggregator,
        n_nodes=n if aggregator == "hsikan" else None,
        cycle_k=cycles_np.shape[1],
    )
    model = CPML(cfg).to(device)
    # Use the SAME initial embedding across cells (critical for fair
    # comparison); clone so each cell trains its own copy.
    node_embed = torch.nn.Embedding(n, d_in).to(device)
    with torch.no_grad():
        node_embed.weight.copy_(initial_embedding)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(node_embed.parameters()), lr=lr,
    )

    if no_cycles:
        cyc_use = np.zeros((0, cycles_np.shape[1]), dtype=np.int64)
        sgn_use = np.zeros((0, cyc_signs_np.shape[1]), dtype=np.int8)
    else:
        cyc_use = cycles_np
        sgn_use = cyc_signs_np

    cyc_t = torch.from_numpy(cyc_use).to(device)
    cyc_sgn_t = torch.from_numpy(sgn_use).to(device)
    tier_of = torch.from_numpy(cfg.tier_spec.assign(degrees)).to(device)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    s_tr_t = torch.from_numpy((s_tr > 0).astype(np.float32)).to(device)
    e_va_t = torch.from_numpy(e_va.astype(np.int64)).to(device)
    s_va_y = (s_va > 0).astype(np.float32)

    n_params = (
        sum(p.numel() for p in model.parameters())
        + sum(p.numel() for p in node_embed.parameters())
    )
    losses, val_aucs = [], []
    t0 = time.perf_counter()
    for ep in range(n_epochs):
        model.train()
        scores = model(node_embed.weight, cyc_t, cyc_sgn_t, tier_of, e_tr_t)
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
            val_auc = float(roc_auc_score(s_va_y, val_probs))
        except ValueError:
            val_auc = float("nan")
        val_aucs.append(val_auc)
    wall = time.perf_counter() - t0
    return dict(
        cell=label, L=L, aggregator=aggregator, no_cycles=no_cycles,
        n_params=int(n_params), wall_s=wall,
        loss_start=losses[0], loss_end=losses[-1],
        val_auc_start=val_aucs[0], val_auc_end=val_aucs[-1],
        val_auc_best=max(val_aucs),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_otc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--d-layer", type=int, default=32)
    ap.add_argument("--d-in", type=int, default=32)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    g = load(args.dataset)
    n = g.n_nodes
    e_tr, s_tr, e_va, s_va = _train_val_split(
        g.edges, g.signs, args.val_frac, args.seed,
    )
    cycles_np, cyc_signs_np = _enumerate_cycles(
        e_tr, s_tr, n, k=3, m_per_vertex=args.topk,
    )
    degrees = np.zeros(n, dtype=np.int64)
    for (u, v) in e_tr:
        degrees[int(u)] += 1; degrees[int(v)] += 1
    print(f"[setup] {args.dataset} |V|={n} |E_tr|={len(e_tr)} "
          f"|E_va|={len(e_va)} cycles={cycles_np.shape[0]}", flush=True)

    # Shared initial embedding across all cells (deterministic).
    torch.manual_seed(args.seed)
    init_embed = torch.randn(n, args.d_in) * 0.1

    cells = [
        # Topology × Aggregator × Structure factorial.
        # Topology:  L=1 (flat) vs L=3 (CPML)
        # Aggregator: mlp / hsikan / clifford_fir (Clifford-derivative
        #   spine analogue — see cpml.ClifFIRTierAggregator)
        # Structure: cycles vs no-cycles (the embedding-only floor)
        ("flat_mlp",          1, "mlp",          False),
        ("flat_mlp_nocyc",    1, "mlp",          True),
        ("flat_hsikan",       1, "hsikan",       False),
        ("flat_hsikan_nocyc", 1, "hsikan",       True),
        ("flat_cliff",        1, "clifford_fir", False),
        ("flat_cliff_nocyc",  1, "clifford_fir", True),
        ("cpml_mlp",          3, "mlp",          False),
        ("cpml_mlp_nocyc",    3, "mlp",          True),
        ("cpml_hsikan",       3, "hsikan",       False),
        ("cpml_hsikan_nocyc", 3, "hsikan",       True),
        ("cpml_cliff",        3, "clifford_fir", False),
        ("cpml_cliff_nocyc",  3, "clifford_fir", True),
    ]
    results = []
    for (label, L, agg, no_cyc) in cells:
        print(f"[cell] {label} L={L} agg={agg} no_cycles={no_cyc} ...",
              flush=True)
        try:
            r = _run_cell(
                label=label, L=L, aggregator=agg, no_cycles=no_cyc,
                n=n, d_in=args.d_in, d_layer=args.d_layer,
                initial_embedding=init_embed,
                cycles_np=cycles_np, cyc_signs_np=cyc_signs_np,
                degrees=degrees,
                e_tr=e_tr, s_tr=s_tr, e_va=e_va, s_va=s_va,
                n_epochs=args.n_epochs, lr=args.lr, device=str(device),
            )
            print(f"  → AUC best={r['val_auc_best']:.4f} "
                  f"(loss {r['loss_start']:.3f}→{r['loss_end']:.3f}, "
                  f"wall {r['wall_s']:.1f}s, params={r['n_params']})",
                  flush=True)
        except Exception as e:
            print(f"  → FAIL: {type(e).__name__}: {e}", flush=True)
            r = dict(cell=label, error=str(e))
        results.append(r)

    print(json.dumps({
        "dataset": args.dataset, "seed": args.seed,
        "n_epochs": args.n_epochs, "d_layer": args.d_layer,
        "d_in": args.d_in, "topk": args.topk,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
