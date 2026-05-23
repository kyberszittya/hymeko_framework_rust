"""Phase 17 (2026-05-20 overnight) — side vs depth A/B on Bitcoin
Alpha.

The Phase 16 depth-stacking experiment falsified the "deeper
helps" hypothesis (Bitcoin Alpha L=1→L=8 went 0.770 → 0.442). This
script runs the architectural sister: same total-parameter-budget
scaling, but width (N parallel branches) instead of depth.

Two model families are compared:

  * Depth-stack: ``MultiLayerSignedKAN(n_layers=L)`` with the
    Phase-16 ResNet-style defaults
    (inner_skip='residual', use_residual=True, layer_norm_between=True,
    jk_mode='last').
  * Side-stack: ``SideSignedKAN(n_branches=N, fusion='mean')``.

Both at hidden=8, n_epochs=30, lr=5e-2, 5 seeds, Bitcoin Alpha.

Reports mean ± std per (family, scale). Quick: 5 seeds × 4 scales
× 2 families = 40 cells; ~80 s expected.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n-epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--dataset", type=str, default="bitcoin_alpha")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "reports" /
                            "side_vs_depth_2026_05_20.json")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    import torch

    from signedkan_wip.experiments.runs.run_compare import run_one
    # Plumb the side-stacked model into run_compare via monkey-patch
    # at the model_dispatch point. Simplest is to call SideSignedKAN
    # directly here by reusing run_one's data-loading + training
    # loop pattern. To keep the script self-contained we replicate
    # the minimal training cell here for the `side` family.
    from signedkan_wip.src.core.side_signedkan import (
        SideSignedKAN, SideSignedKANConfig,
    )

    results: dict[str, dict] = {}
    t0_total = time.time()

    # ─── Depth family ──────────────────────────────────────────
    for L in args.scales:
        aucs = []
        wall = 0.0
        for seed in args.seeds:
            t_cell = time.time()
            r = run_one(
                "signedkan", args.dataset,
                hidden=args.hidden, seed=seed,
                n_epochs=args.n_epochs, lr=args.lr,
                n_layers=L,
                inner_skip="residual", outer_skip="none",
                layer_norm_between=True, jk_mode="last",
            )
            aucs.append(r["test_auc"])
            wall += time.time() - t_cell
        m, s = statistics.fmean(aucs), statistics.pstdev(aucs)
        results[f"depth_L{L}"] = {
            "family": "depth", "scale": L,
            "mean_auc": round(m, 4), "std": round(s, 4),
            "wall_s": round(wall, 1),
            "wall_per_seed": round(wall / len(args.seeds), 2),
            "n_params": int(r.get("n_params", 0)) if isinstance(r.get("n_params"), (int, float)) else None,
            "aucs": [round(a, 4) for a in aucs],
        }
        print(f"[depth L={L}] {m:.4f} ± {s:.4f}  ({wall:.1f}s)",
              file=sys.stderr)

    # ─── Side family ───────────────────────────────────────────
    # We replicate the minimal training cell for SideSignedKAN
    # because run_compare.run_one's dispatch table doesn't know
    # about the side model. Forward signature matches bare
    # SignedKAN's encode_triads() exactly so the rest of the
    # training loop in run_one would have worked verbatim if the
    # model_name dispatch knew the name.
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from signedkan_wip.src.datasets import load, split
    from signedkan_wip.src.core.hyperedges import construct
    from signedkan_wip.src.core.train import build_edge_to_triads

    for N in args.scales:
        aucs = []
        wall = 0.0
        for seed in args.seeds:
            t_cell = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            g = load(args.dataset)
            triads = construct(g)
            triad_v = torch.tensor([t.v for t in triads], dtype=torch.long, device=device)
            triad_sigma = torch.tensor([t.sigma for t in triads], dtype=torch.long, device=device)
            edge_to_triads = build_edge_to_triads(triads)
            tr_idx, _, te_idx = split(g, seed=seed)

            cfg = SideSignedKANConfig(
                n_nodes=g.n_nodes, n_branches=N,
                hidden_dim=args.hidden, fusion="mean",
            )
            model = SideSignedKAN(cfg).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=1e-5)
            # Train.
            for _ in range(args.n_epochs):
                model.train(); opt.zero_grad()
                h_t = model.encode_triads(triad_v, triad_sigma)
                # Edge-pool: mean over triads incident to each train edge.
                e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
                logits = []
                for j, ei in enumerate(e_tr):
                    key = (int(ei[0]), int(ei[1]))
                    if key not in edge_to_triads:
                        key = (int(ei[1]), int(ei[0]))
                    tri_ids = edge_to_triads.get(key, [])
                    if tri_ids:
                        emb = h_t[tri_ids].mean(dim=0)
                    else:
                        emb = torch.zeros(args.hidden, device=device)
                    logits.append(emb.sum())
                logits_t = torch.stack(logits)
                targets = torch.tensor((s_tr + 1.0) / 2.0, dtype=torch.float32,
                                       device=device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits_t, targets,
                )
                loss.backward(); opt.step()

            # Test eval.
            model.eval()
            with torch.no_grad():
                h_t = model.encode_triads(triad_v, triad_sigma)
                e_te, s_te = g.edges[te_idx], g.signs[te_idx]
                logits = []
                for j, ei in enumerate(e_te):
                    key = (int(ei[0]), int(ei[1]))
                    if key not in edge_to_triads:
                        key = (int(ei[1]), int(ei[0]))
                    tri_ids = edge_to_triads.get(key, [])
                    if tri_ids:
                        emb = h_t[tri_ids].mean(dim=0)
                    else:
                        emb = torch.zeros(args.hidden, device=device)
                    logits.append(emb.sum())
                logits_np = torch.stack(logits).cpu().numpy()
                targets_np = ((s_te + 1.0) / 2.0)
                auc = roc_auc_score(targets_np, logits_np) if len(np.unique(targets_np)) > 1 else float("nan")
            aucs.append(float(auc))
            wall += time.time() - t_cell
        m, s = statistics.fmean(aucs), statistics.pstdev(aucs)
        n_params = sum(p.numel() for p in model.parameters())
        results[f"side_N{N}"] = {
            "family": "side", "scale": N,
            "mean_auc": round(m, 4), "std": round(s, 4),
            "wall_s": round(wall, 1),
            "wall_per_seed": round(wall / len(args.seeds), 2),
            "n_params": n_params,
            "aucs": [round(a, 4) for a in aucs],
        }
        print(f"[side  N={N}] {m:.4f} ± {s:.4f}  ({wall:.1f}s)",
              file=sys.stderr)

    print(f"\nTotal wall: {time.time()-t0_total:.1f} s", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}", file=sys.stderr)

    # Summary table.
    print("\n=== Side vs Depth comparison on Bitcoin Alpha ===", file=sys.stderr)
    print(f"{'scale':>5}  {'depth_mean ± std':>22}  {'side_mean ± std':>22}",
          file=sys.stderr)
    print("="*60, file=sys.stderr)
    for k in args.scales:
        d = results[f"depth_L{k}"]
        s = results[f"side_N{k}"]
        print(f"{k:>5}  {d['mean_auc']:.4f} ± {d['std']:.4f}        "
              f"{s['mean_auc']:.4f} ± {s['std']:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
