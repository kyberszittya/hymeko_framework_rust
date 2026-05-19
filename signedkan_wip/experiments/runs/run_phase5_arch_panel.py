"""Phase 5 — architecture panel, strict-Derr protocol.

How does HSiKAN sit relative to non-signed-graph-specialised baselines?
SGCN is purpose-built for signed link sign prediction — a head-to-head
loss against it doesn't tell us whether HSiKAN is "good as a KAN" or
"good as a graph network".

Architectures in this panel (all in strict-Derr protocol — no
early stopping, no class-weighted BCE, no weight decay; full-graph
adjacency where applicable; 120 epochs; 3 seeds):

  graph-blind:
    - MLP                : per-node embedding + MLP edge head, no
                           graph propagation at all.

  sign-blind on graph:
    - SignBlindGCN       : 2-layer GCN, mean-aggregate, ignores
                           edge signs.
    - VanillaKAN @ h=32  : sign-blind KAN on edge endpoints. KAN
                           expressiveness, no signed structure.

  signed-aware, KAN family:
    - SignedKAN L=1      : single-layer signed-incidence KAN, no
                           recipe regs.
    - SignedKAN L=3      : 3-layer signed-incidence (no weight share,
                           no skip, no recipe regs).
    - HSiKAN-mixed leanest: best HSiKAN recipe found (lean +
                           highway + share + signed + JK-concat
                           + smooth=0 + R2=0 + clip=0).

  signed-aware, signed-graph-specialised:
    - SGCN no-balance    : SGCN architecture, plain BCE only.
    - SGCN+balance       : full Derr 2018.

Total: 8 architectures × 2 datasets × 3 seeds = 48 runs.
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
    """Evaluate a callable model_fn(edges_t)->logits."""
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
             n_epochs: int = 120, lr: float = 5e-3) -> dict:
    """Strict-Derr protocol: plain BCE, no early stop, no CW, no wd."""
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
             n_layers: int = 2, n_epochs: int = 120,
             lr: float = 5e-3) -> dict:
    """Sign-blind GCN. Strict-Derr protocol."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load(dataset)
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]
    A = build_unsigned_adj(g.edges, g.n_nodes, device)  # full-graph adj

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase5_arch_panel.json")
    args = ap.parse_args()

    runs = []
    t_total = time.time()

    # ── 1. MLP (graph-blind) ────────────────────────────────────────
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_mlp(dataset, seed, hidden=32, n_epochs=args.n_epochs)
            r["arch"] = "mlp_blind"
            print(f"  mlp_blind            {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"params={r['n_params']:>6,}  {r['elapsed_s']:.1f}s")
            runs.append(r)

    # ── 2. SignBlindGCN ─────────────────────────────────────────────
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_gcn(dataset, seed, hidden=32, n_layers=2,
                         n_epochs=args.n_epochs)
            r["arch"] = "gcn_blind"
            print(f"  gcn_blind            {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"params={r['n_params']:>6,}  {r['elapsed_s']:.1f}s")
            runs.append(r)

    # ── 3. VanillaKAN @ h=32, strict-Derr ───────────────────────────
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_one("vanillakan", dataset, hidden=32, seed=seed,
                         n_epochs=args.n_epochs, lr=5e-2,
                         class_weighted=False, early_stopping=False,
                         weight_decay=0.0)
            r["arch"] = "vanillakan"
            print(f"  vanillakan           {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"params={r['n_params']:>6,}  {r['elapsed_s']:.1f}s")
            runs.append(r)

    # ── 4. SignedKAN L=1 plain (strict-Derr, no recipe) ─────────────
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_one("signedkan", dataset, hidden=32, seed=seed,
                         n_epochs=args.n_epochs, lr=5e-2,
                         spline_kind="catmull_rom",
                         n_layers=1, grid=5,
                         class_weighted=False, early_stopping=False,
                         weight_decay=0.0)
            r["arch"] = "signedkan_L1"
            print(f"  signedkan_L1         {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"params={r['n_params']:>6,}  {r['elapsed_s']:.1f}s")
            runs.append(r)

    # ── 5. SignedKAN L=3 plain (no weight-share, no recipe) ─────────
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_one("signedkan", dataset, hidden=32, seed=seed,
                         n_epochs=args.n_epochs, lr=5e-2,
                         spline_kind="catmull_rom",
                         spline_kinds=["catmull_rom"] * 3,
                         n_layers=3, grid=5,
                         class_weighted=False, early_stopping=False,
                         weight_decay=0.0)
            r["arch"] = "signedkan_L3"
            print(f"  signedkan_L3         {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"params={r['n_params']:>6,}  {r['elapsed_s']:.1f}s")
            runs.append(r)

    # ── 6. HSiKAN-mixed leanest, strict-Derr (already in Phase 4) ──
    # We rerun for consistency on this panel.
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_one_mixed(
                dataset, seed, hidden=16, n_layers=2, grid=3,
                n_epochs=args.n_epochs,
                arities=(3, 4), max_k4=30000,
                only_k3=False,
                coef_smooth_lam=0.0, participation_lam=0.0,
                grad_clip=0.0, weight_decay=0.0,
                early_stopping=False, class_weighted=False,
            )
            r["arch"] = "hsikan_mixed_leanest_strict"
            print(f"  hsikan_mixed_leanest {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"alpha={[round(a,2) for a in r['alpha']]}  "
                  f"params={r['n_params']:>6,}  {r['elapsed_s']:.1f}s")
            runs.append(r)

    # ── 7. SGCN no-balance, strict-Derr ─────────────────────────────
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_one_sgcn(dataset, seed, hidden=32, n_layers=2,
                              n_epochs=args.n_epochs, lr=5e-3,
                              balance_alpha=0.0,        # NO balance loss
                              adj_protocol="full_graph",
                              early_stopping=False,
                              class_weighted=False,
                              weight_decay=0.0)
            r["arch"] = "sgcn_nobal_strict"
            print(f"  sgcn_nobal_strict    {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"params={r['n_params']:>6,}  {r['elapsed_s']:.1f}s")
            runs.append(r)

    # ── 8. SGCN+balance, strict-Derr ────────────────────────────────
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_one_sgcn(dataset, seed, hidden=32, n_layers=2,
                              n_epochs=args.n_epochs, lr=5e-3,
                              balance_alpha=0.5,
                              adj_protocol="full_graph",
                              early_stopping=False,
                              class_weighted=False,
                              weight_decay=0.0)
            r["arch"] = "sgcn_bal_strict"
            print(f"  sgcn_bal_strict      {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"params={r['n_params']:>6,}  {r['elapsed_s']:.1f}s")
            runs.append(r)

    # Summary
    summary = {}
    keys = sorted({(r["arch"], r["dataset"]) for r in runs})
    for arch, dataset in keys:
        cell = [r for r in runs
                 if r["arch"] == arch and r["dataset"] == dataset]
        aucs = [r["test_auc"] for r in cell]
        f1ms = [r["test_f1_macro"] for r in cell]
        elap = [r["elapsed_s"] for r in cell]
        params = cell[0]["n_params"] if cell else 0
        summary[f"{arch}|{dataset}"] = {
            "auc_med":   round(statistics.median(aucs), 4),
            "f1m_med":   round(statistics.median(f1ms), 4),
            "elapsed_med_s": round(statistics.median(elap), 2),
            "n_params":  params,
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
