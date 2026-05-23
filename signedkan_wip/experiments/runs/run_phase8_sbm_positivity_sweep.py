"""Phase 8 — SBM positivity sweep (the crossover figure).

Vary the within-community positive-edge probability `pos_in` from
0.50 to 0.95 (roughly: balanced → Bitcoin-like). Fix `pos_out=0.15`
(cross-community structure stays the same so balance-violation
signal is comparable across cells). Holds n_nodes=200,
n_communities=4, p_in=0.20, p_out=0.05, noise=0.05.

Architectures (strict-Derr):
  - MLP, GCN, SignedKAN L=1, HSiKAN-mixed leanest, SGCN+balance.

Sweeps `pos_in ∈ {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95}`,
3 seeds per cell, 5 architectures = 150 runs. SBM-200 runs are
~1-5s each, total ~10 min.

The expected output is a publishable curve: AUC vs pos_in per
architecture, with HSiKAN-mixed leading at low pos_in and SGCN
catching up / overtaking around the Bitcoin-like 0.85+ range.
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

from signedkan_wip.src.datasets import sbm_signed
from signedkan_wip.src.datasets import SignedGraph, split
from signedkan_wip.src.baselines.mlp_gcn import MLPEdge, SignBlindGCN, build_unsigned_adj
from .run_compare import run_one
from .run_sgcn_baseline import run_one_sgcn
from .run_phase2_mixed_arity import run_one_mixed


# Patch load() so the existing runners can be called with our
# parameterised SBM datasets via a string key.
import signedkan_wip.src.datasets as _ds
_orig_load = _ds.load
_dataset_cache: dict[str, SignedGraph] = {}


def _patched_load(name: str) -> SignedGraph:
    if name in _dataset_cache:
        return _dataset_cache[name]
    if name.startswith("sbmsweep_"):
        # sbmsweep_pos85_s0 → pos_in=0.85, seed=0
        parts = name.split("_")
        pos_in_pct = int(parts[1][3:])
        pos_in = pos_in_pct / 100.0
        seed = int(parts[2][1:])
        g, _ = sbm_signed(n_nodes=200, n_communities=4,
                            p_in=0.20, p_out=0.05,
                            pos_in=pos_in, pos_out=0.15,
                            noise=0.05, seed=seed)
        _dataset_cache[name] = g
        return g
    return _orig_load(name)


_ds.load = _patched_load


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
    g = _patched_load(dataset)
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
    return dict(model="mlp", dataset=dataset, seed=seed, elapsed_s=elapsed,
                test_auc=auc, test_f1_macro=f1m)


def run_gcn(dataset, seed, hidden=32, n_layers=2, n_epochs=200, lr=5e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = _patched_load(dataset)
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
    return dict(model="gcn", dataset=dataset, seed=seed, elapsed_s=elapsed,
                test_auc=auc, test_f1_macro=f1m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_ins", nargs="+", type=int,
                    default=[50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=200)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase8_sbm_positivity.json")
    args = ap.parse_args()

    runs = []
    t_total = time.time()
    for pos_in in args.pos_ins:
        for seed in args.seeds:
            ds = f"sbmsweep_pos{pos_in:02d}_s{seed}"
            # Warm load to verify positivity
            g = _patched_load(ds)
            actual_pos = float((g.signs == 1).sum()) / len(g.edges)
            print(f"\n--- pos_in={pos_in/100:.2f} seed={seed}  "
                  f"({len(g.edges)} edges, {100*actual_pos:.1f}% pos) ---")

            for arch_tag, runner in [
                ("mlp_blind",       lambda: run_mlp(ds, seed, n_epochs=args.n_epochs)),
                ("gcn_blind",       lambda: run_gcn(ds, seed, n_layers=2, n_epochs=args.n_epochs)),
                ("signedkan_L1",    lambda: run_one("signedkan", ds, hidden=32, seed=seed,
                                                     n_epochs=args.n_epochs, lr=5e-2,
                                                     spline_kind="catmull_rom", n_layers=1, grid=5,
                                                     class_weighted=False, early_stopping=False,
                                                     weight_decay=0.0)),
                ("hsikan_mixed",    lambda: run_one_mixed(ds, seed, hidden=16, n_layers=2, grid=3,
                                                            n_epochs=args.n_epochs,
                                                            arities=(3, 4), max_k4=30000,
                                                            only_k3=False,
                                                            coef_smooth_lam=0.0,
                                                            participation_lam=0.0,
                                                            grad_clip=0.0, weight_decay=0.0,
                                                            early_stopping=False,
                                                            class_weighted=False)),
                ("sgcn_balance",    lambda: run_one_sgcn(ds, seed, hidden=32, n_layers=2,
                                                          n_epochs=args.n_epochs, lr=5e-3,
                                                          balance_alpha=0.5,
                                                          adj_protocol="full_graph",
                                                          early_stopping=False,
                                                          class_weighted=False,
                                                          weight_decay=0.0)),
            ]:
                try:
                    r = runner()
                    r["arch"] = arch_tag
                    r["pos_in"] = pos_in / 100.0
                    r["actual_pos"] = actual_pos
                    print(f"  {arch_tag:14s}  AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                          f"{r['elapsed_s']:.1f}s")
                    runs.append(r)
                except Exception as e:
                    print(f"  {arch_tag:14s}  FAILED: {e!r}")

    # Summary: per (arch, pos_in), median over seeds.
    summary = {}
    for arch in {r["arch"] for r in runs}:
        for pos_in in args.pos_ins:
            cell = [r for r in runs
                     if r["arch"] == arch
                     and abs(r["pos_in"] - pos_in / 100.0) < 1e-9]
            if not cell:
                continue
            aucs = [r["test_auc"] for r in cell]
            f1ms = [r["test_f1_macro"] for r in cell]
            actual = [r["actual_pos"] for r in cell]
            summary[f"{arch}|pos{pos_in}"] = {
                "auc_med":     round(statistics.median(aucs), 4),
                "f1m_med":     round(statistics.median(f1ms), 4),
                "actual_pos":  round(statistics.median(actual), 3),
                "auc_seeds":   [round(a, 4) for a in aucs],
                "n_seeds":     len(cell),
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
