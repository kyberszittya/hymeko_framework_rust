"""Phase 8 — 5-seed Bitcoin Alpha/OTC re-validation including SiGAT-attn.

Closes two open items:
  1. Bitcoin Alpha/OTC 5-seed re-val of the strict-Derr panel
     (current cited Bitcoin numbers are 3-seed).
  2. SiGAT-style attention baseline (in-protocol re-impl, 2-motif
     pos/neg neighbour decomposition with multi-head attention).

Architectures (strict-Derr protocol — plain BCE, no early stop, no
class weighting, no weight decay):
  - MLP (sign-blind, no graph)
  - GCN (sign-blind, graph)
  - SignedKAN L=1 plain
  - HSiKAN-mixed leanest (k=3 + k=4)
  - SGCN + balance
  - SiGAT-attn (NEW — pos/neg motif attention, 1 layer × 4 heads)
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
from signedkan_wip.src.baselines.sigat_model import SiGATAttn, build_neighbour_lists
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
    auc = (roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan"))
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return float(auc), float(f1m)


def run_mlp(dataset, seed, hidden=32, n_epochs=200, lr=5e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load(dataset); tr_idx, _, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]
    model = MLPEdge(g.n_nodes, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    target = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    t0 = time.time()
    for _ in range(n_epochs):
        model.train()
        loss = F.binary_cross_entropy_with_logits(model(e_tr_t), target)
        opt.zero_grad(); loss.backward(); opt.step()
    elapsed = time.time() - t0
    model.eval()
    auc, f1m = _eval_simple(lambda e: model(e), e_te, s_te, device)
    return dict(model="mlp", dataset=dataset, seed=seed,
                n_params=model.num_parameters(), elapsed_s=elapsed,
                test_auc=auc, test_f1_macro=f1m)


def run_gcn(dataset, seed, hidden=32, n_layers=2, n_epochs=200, lr=5e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load(dataset); tr_idx, _, te_idx = split(g, seed=seed)
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
        loss = F.binary_cross_entropy_with_logits(model(A, e_tr_t), target)
        opt.zero_grad(); loss.backward(); opt.step()
    elapsed = time.time() - t0
    model.eval()
    auc, f1m = _eval_simple(lambda e: model(A, e), e_te, s_te, device)
    return dict(model="gcn", dataset=dataset, seed=seed,
                n_layers=n_layers, n_params=model.num_parameters(),
                elapsed_s=elapsed, test_auc=auc, test_f1_macro=f1m)


def run_sigat(dataset, seed, hidden=32, n_heads=4, n_layers=1,
               n_epochs=200, lr=5e-3, adj_protocol="full_graph"):
    """SiGAT-attn in strict-Derr protocol: plain BCE, no early stop,
    no class weighting, no weight decay."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load(dataset); tr_idx, _, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]

    if adj_protocol == "full_graph":
        pos_buckets, neg_buckets = build_neighbour_lists(
            g.edges, g.signs, g.n_nodes)
    else:
        pos_buckets, neg_buckets = build_neighbour_lists(
            e_tr, s_tr, g.n_nodes)

    model = SiGATAttn(g.n_nodes, hidden_dim=hidden,
                      n_heads=n_heads, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    target = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)

    t0 = time.time()
    for _ in range(n_epochs):
        model.train()
        z = model.encode_nodes(pos_buckets, neg_buckets)
        logits = model.edge_logits(z, e_tr_t)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        z = model.encode_nodes(pos_buckets, neg_buckets)
        e_te_t = torch.from_numpy(e_te.astype(np.int64)).to(device)
        logits = model.edge_logits(z, e_te_t).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    y = (s_te == 1).astype(int)
    auc = (roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan"))
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return dict(model="sigat_attn", dataset=dataset, seed=seed,
                hidden=hidden, n_heads=n_heads, n_layers=n_layers,
                n_params=model.num_parameters(), elapsed_s=elapsed,
                test_auc=float(auc), test_f1_macro=float(f1m))


DATASETS = ["bitcoin_alpha", "bitcoin_otc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_epochs", type=int, default=200)
    ap.add_argument("--max_k4", type=int, default=30000)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase8_bitcoin_5seed.json")
    args = ap.parse_args()

    runs = []
    t_total = time.time()

    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_mlp(dataset, seed, hidden=32, n_epochs=args.n_epochs)
            r["arch"] = "mlp_blind"
            print(f"  mlp_blind            {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"{r['elapsed_s']:.1f}s")
            runs.append(r)

            r = run_gcn(dataset, seed, hidden=32, n_layers=2,
                         n_epochs=args.n_epochs)
            r["arch"] = "gcn_blind"
            print(f"  gcn_blind            {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"{r['elapsed_s']:.1f}s")
            runs.append(r)

            try:
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=args.n_epochs, lr=5e-2,
                             spline_kind="catmull_rom",
                             n_layers=1, grid=5,
                             class_weighted=False, early_stopping=False,
                             weight_decay=0.0)
                r["arch"] = "signedkan_L1"
                print(f"  signedkan_L1         {dataset:14s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)
            except Exception as e:
                print(f"  signedkan_L1 FAILED on {dataset} seed={seed}: {e!r}")

            try:
                r = run_one_mixed(
                    dataset, seed, hidden=16, n_layers=2, grid=3,
                    n_epochs=args.n_epochs,
                    arities=(3, 4), max_k4=args.max_k4,
                    only_k3=False,
                    coef_smooth_lam=0.0, participation_lam=0.0,
                    grad_clip=0.0, weight_decay=0.0,
                    early_stopping=False, class_weighted=False,
                )
                r["arch"] = "hsikan_mixed_leanest"
                print(f"  hsikan_mixed         {dataset:14s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)
            except Exception as e:
                print(f"  hsikan_mixed FAILED on {dataset} seed={seed}: {e!r}")

            r = run_one_sgcn(dataset, seed, hidden=32, n_layers=2,
                              n_epochs=args.n_epochs, lr=5e-3,
                              balance_alpha=0.5,
                              adj_protocol="full_graph",
                              early_stopping=False,
                              class_weighted=False,
                              weight_decay=0.0)
            r["arch"] = "sgcn_balance"
            print(f"  sgcn_balance         {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"{r['elapsed_s']:.1f}s")
            runs.append(r)

            try:
                r = run_sigat(dataset, seed, hidden=32, n_heads=4,
                               n_layers=1, n_epochs=args.n_epochs,
                               lr=5e-3, adj_protocol="full_graph")
                r["arch"] = "sigat_attn"
                print(f"  sigat_attn           {dataset:14s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)
            except Exception as e:
                print(f"  sigat_attn FAILED on {dataset} seed={seed}: {e!r}")

    summary = {}
    keys = sorted({(r["arch"], r["dataset"]) for r in runs})
    for arch, dataset in keys:
        cell = [r for r in runs if r["arch"] == arch and r["dataset"] == dataset]
        aucs = [r["test_auc"] for r in cell]
        f1ms = [r["test_f1_macro"] for r in cell]
        elap = [r["elapsed_s"] for r in cell]
        summary[f"{arch}|{dataset}"] = {
            "auc_med":   round(statistics.median(aucs), 4),
            "f1m_med":   round(statistics.median(f1ms), 4),
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
    print(f"\nwrote {out_path} ({len(runs)} runs in {out['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()
