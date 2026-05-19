"""SignedKAN — Phase 3.8 ablation.

Isolates the contribution of signed-incidence semantics by comparing:

  - **Full SignedKAN** (Option C): three sub-aggregations (+, −, ~)
    with separate spline pairs per sign.
  - **Incidence-only**: collapse all σ to +1 → single sub-aggregation,
    sign-blind otherwise identical.

If the full version beats the collapsed version on the same config,
the gap quantifies the value of carrying signed semantics through the
spline activations.

Run:
    python3 -m src.run_ablation --dataset bitcoin_alpha
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from signedkan_wip.src.datasets import load, split
from signedkan_wip.src.hyperedges import construct
from signedkan_wip.src.signedkan import SignedKAN, SignedKANConfig
from signedkan_wip.src.train import build_edge_to_triads


SWEEP_DIR = Path("signedkan_wip/experiments/results/phase3_sweep")
ABL_DIR = Path("signedkan_wip/experiments/results/ablation")


def run_one(dataset: str, seed: int, hidden: int, lr: float,
            n_epochs: int, use_minus_branch: bool,
            device: torch.device) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = load(dataset)
    triads = construct(g)
    triad_v = torch.tensor([t.v for t in triads], dtype=torch.long).to(device)
    if use_minus_branch:
        triad_sigma = torch.tensor([t.sigma for t in triads], dtype=torch.long).to(device)
    else:
        # Incidence-only: collapse all signs to +1 (sign-blind aggregation).
        triad_sigma = torch.ones((len(triads), 3), dtype=torch.long).to(device)
    edge_to_triads = build_edge_to_triads(triads)
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    edges_train = g.edges[tr_idx]; signs_train = g.signs[tr_idx]
    edges_test = g.edges[te_idx]; signs_test = g.signs[te_idx]

    # Build sparse incidence matrices (same shape, signs irrelevant).
    def build_M(edges):
        rows, cols, vals = [], [], []
        for ei, e in enumerate(edges):
            tri_ids = edge_to_triads.get(
                (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1]))), [],
            )
            if not tri_ids:
                continue
            w = 1.0 / float(len(tri_ids))
            for t in tri_ids:
                rows.append(ei); cols.append(int(t)); vals.append(w)
        if not rows:
            return torch.zeros((edges.shape[0], len(triads)), device=device)
        idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
        v = torch.tensor(vals, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(idx, v, (edges.shape[0], len(triads))).coalesce()

    M_train = build_M(edges_train)
    M_test = build_M(edges_test)

    cfg = SignedKANConfig(n_nodes=g.n_nodes, hidden_dim=hidden,
                          use_minus_branch=use_minus_branch)
    model = SignedKAN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    target_train = torch.from_numpy((signs_train == 1).astype(np.float32)).to(device)

    t0 = time.time()
    for _ in range(n_epochs):
        model.train()
        triad_emb = model.encode_triads(triad_v, triad_sigma)
        edge_emb = torch.sparse.mm(M_train, triad_emb)
        logits = model.classifier(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target_train)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        triad_emb = model.encode_triads(triad_v, triad_sigma)
        edge_emb_test = torch.sparse.mm(M_test, triad_emb)
        logits_test = model.classifier(edge_emb_test).squeeze(-1).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits_test))
    preds = (probs > 0.5).astype(int)
    y = (signs_test == 1).astype(int)
    from sklearn.metrics import f1_score, roc_auc_score
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    f1_bin = f1_score(y, preds, average="binary", zero_division=0)
    f1_mac = f1_score(y, preds, average="macro", zero_division=0)
    return dict(
        dataset=dataset, seed=seed, hidden=hidden, lr=lr,
        use_minus_branch=use_minus_branch,
        n_params=model.num_parameters(),
        elapsed_s=time.time() - t0,
        test_auc=auc, test_f1_binary=f1_bin, test_f1_macro=f1_mac,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_alpha",
                    choices=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    ABL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[ablation] {args.dataset}, hidden={args.hidden}, lr={args.lr}")

    full_results = []
    inc_results = []
    for seed in args.seeds:
        for use_minus in (True, False):
            r = run_one(args.dataset, seed, args.hidden, args.lr,
                         args.n_epochs, use_minus, device)
            tag = "Full" if use_minus else "IncidenceOnly"
            print(f"  {tag:<14} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  "
                  f"F1bin={r['test_f1_binary']:.4f}  "
                  f"F1mac={r['test_f1_macro']:.4f}  "
                  f"params={r['n_params']:,}")
            (full_results if use_minus else inc_results).append(r)

    out = dict(full=full_results, incidence_only=inc_results)
    out_path = ABL_DIR / f"ablation_{args.dataset}_h{args.hidden}.json"
    out_path.write_text(json.dumps(out, indent=2))

    # Headline.
    print(f"\n=== {args.dataset} ablation (hidden={args.hidden}, lr={args.lr}) ===")
    for label, results in [("Full SignedKAN", full_results),
                           ("Incidence-Only", inc_results)]:
        aucs = [r["test_auc"] for r in results]
        f1ms = [r["test_f1_macro"] for r in results]
        n = len(aucs)
        print(f"  {label:<18}  n={n}  "
              f"AUC = {stats.mean(aucs):.3f} ± {stats.stdev(aucs) if n>1 else 0:.3f}  "
              f"F1_mac = {stats.mean(f1ms):.3f} ± {stats.stdev(f1ms) if n>1 else 0:.3f}")
    # Δ.
    auc_full = stats.mean(r["test_auc"] for r in full_results)
    auc_inc = stats.mean(r["test_auc"] for r in inc_results)
    f1_full = stats.mean(r["test_f1_macro"] for r in full_results)
    f1_inc = stats.mean(r["test_f1_macro"] for r in inc_results)
    print(f"\n  Δ (Full − IncidenceOnly):  "
          f"ΔAUC = {auc_full - auc_inc:+.3f}  "
          f"ΔF1_mac = {f1_full - f1_inc:+.3f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
