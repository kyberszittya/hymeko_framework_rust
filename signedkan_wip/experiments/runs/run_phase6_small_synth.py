"""Phase 6 — small + synthetic dataset panel.

Runs the same architecture panel as Phase 5 on:
  - karate (Zachary, faction-signed): 34 nodes, 78 edges, 86% pos
  - sbm_n200_k4_s0: 200 nodes, ~1.7k edges, ~55% pos
  - sbm_n400_k5_s0: 400 nodes, ~6.3k edges, ~50% pos
  - hier_n240_s0:   240 nodes, ~2.4k edges, ~54% pos (hierarchical
                     SBM designed to favour mixed-arity)

Architectures (strict-Derr protocol, fast iteration):
  - MLP (sign-blind, no graph)
  - GCN (sign-blind, graph)
  - SignedKAN L=1 plain
  - HSiKAN-mixed leanest (k=3 + k=4)
  - SGCN + balance

5 architectures × 4 datasets × 5 seeds (more for tiny) = ~100 runs.
Tiny datasets are fast (<1s/run); SBM-200 fits the same range as
Bitcoin Alpha. Everything fits in ~5 min wall-clock.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from signedkan_wip.src.datasets import load, split
from signedkan_wip.src.baselines.mlp_gcn import MLPEdge, SignBlindGCN, build_unsigned_adj
from .run_compare import run_one
from .run_sgcn_baseline import run_one_sgcn
from .run_phase2_mixed_arity import run_one_mixed


def _eval_simple(model_fn, e_te, s_te, device):
    e_t = torch.from_numpy(e_te.astype(np.int64)).to(device)
    with torch.no_grad():
        logits = model_fn(e_t).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    y = (s_te == 1).astype(int)
    auc = (roc_auc_score(y, probs) if len(np.unique(y)) > 1
           else float("nan"))
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return float(auc), float(f1m)


def run_mlp(dataset: str, seed: int, hidden: int = 32,
             n_epochs: int = 200, lr: float = 5e-3) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load(dataset)
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]
    model = MLPEdge(g.n_nodes, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    target = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    t0 = time.time()
    for _ in range(n_epochs):
        model.train()
        logits = model(e_tr_t)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
    elapsed = time.time() - t0
    model.eval()
    auc, f1m = _eval_simple(lambda e: model(e), e_te, s_te, device)
    return dict(model="mlp", dataset=dataset, hidden=hidden, seed=seed,
                n_params=model.num_parameters(), elapsed_s=elapsed,
                test_auc=auc, test_f1_macro=f1m)


def run_gcn(dataset: str, seed: int, hidden: int = 32,
             n_layers: int = 2, n_epochs: int = 200,
             lr: float = 5e-3) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load(dataset)
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]
    A = build_unsigned_adj(g.edges, g.n_nodes, device)
    model = SignBlindGCN(g.n_nodes, hidden, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    target = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    t0 = time.time()
    for _ in range(n_epochs):
        model.train()
        logits = model(A, e_tr_t)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
    elapsed = time.time() - t0
    model.eval()
    auc, f1m = _eval_simple(lambda e: model(A, e), e_te, s_te, device)
    return dict(model="gcn", dataset=dataset, hidden=hidden, seed=seed,
                n_layers=n_layers, n_params=model.num_parameters(),
                elapsed_s=elapsed, test_auc=auc, test_f1_macro=f1m)


DATASETS = ["karate", "sbm_n200_k4_s0", "sbm_n400_k5_s0", "hier_n240_s0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_epochs", type=int, default=200)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase6_small_synth.json")
    args = ap.parse_args()

    runs = []
    t_total = time.time()

    for dataset in args.datasets:
        for seed in args.seeds:
            # 1. MLP
            r = run_mlp(dataset, seed, hidden=32, n_epochs=args.n_epochs)
            r["arch"] = "mlp_blind"
            print(f"  mlp_blind            {dataset:18s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"{r['elapsed_s']:.1f}s")
            runs.append(r)

            # 2. GCN sign-blind
            r = run_gcn(dataset, seed, hidden=32, n_layers=2,
                         n_epochs=args.n_epochs)
            r["arch"] = "gcn_blind"
            print(f"  gcn_blind            {dataset:18s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"{r['elapsed_s']:.1f}s")
            runs.append(r)

            # 3. SignedKAN L=1 plain (strict-Derr)
            try:
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=args.n_epochs, lr=5e-2,
                             spline_kind="catmull_rom",
                             n_layers=1, grid=5,
                             class_weighted=False, early_stopping=False,
                             weight_decay=0.0)
                r["arch"] = "signedkan_L1"
                print(f"  signedkan_L1         {dataset:18s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)
            except Exception as e:
                print(f"  signedkan_L1 FAILED on {dataset} seed={seed}: {e!r}")

            # 4. HSiKAN-mixed leanest (strict-Derr).
            try:
                r = run_one_mixed(
                    dataset, seed, hidden=16, n_layers=2, grid=3,
                    n_epochs=args.n_epochs,
                    arities=(3, 4), max_k4=30000,
                    only_k3=False,
                    coef_smooth_lam=0.0, participation_lam=0.0,
                    grad_clip=0.0, weight_decay=0.0,
                    early_stopping=False, class_weighted=False,
                )
                r["arch"] = "hsikan_mixed_leanest"
                print(f"  hsikan_mixed         {dataset:18s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                      f"alpha={[round(a,2) for a in r['alpha']]}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)
            except Exception as e:
                print(f"  hsikan_mixed FAILED on {dataset} seed={seed}: {e!r}")

            # 5. SGCN + balance (strict-Derr Derr-faithful).
            r = run_one_sgcn(dataset, seed, hidden=32, n_layers=2,
                              n_epochs=args.n_epochs, lr=5e-3,
                              balance_alpha=0.5,
                              adj_protocol="full_graph",
                              early_stopping=False,
                              class_weighted=False,
                              weight_decay=0.0)
            r["arch"] = "sgcn_balance"
            print(f"  sgcn_balance         {dataset:18s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"{r['elapsed_s']:.1f}s")
            runs.append(r)

    summary = {}
    keys = sorted({(r["arch"], r["dataset"]) for r in runs})
    for arch, dataset in keys:
        cell = [r for r in runs
                 if r["arch"] == arch and r["dataset"] == dataset]
        aucs = [r["test_auc"] for r in cell]
        f1ms = [r["test_f1_macro"] for r in cell]
        elap = [r["elapsed_s"] for r in cell]
        summary[f"{arch}|{dataset}"] = {
            "auc_med":   round(statistics.median(aucs), 4),
            "f1m_med":   round(statistics.median(f1ms), 4),
            "auc_iqr":   round(np.percentile(aucs, 75) - np.percentile(aucs, 25), 4),
            "elapsed_med_s": round(statistics.median(elap), 2),
            "n_seeds":   len(cell),
            "auc_seeds": [round(a, 4) for a in aucs],
            "f1m_seeds": [round(f, 4) for f in f1ms],
        }

    out = {
        "runs": runs,
        "summary": summary,
        "wall_clock_s": round(time.time() - t_total, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({len(runs)} runs in "
          f"{out['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()
