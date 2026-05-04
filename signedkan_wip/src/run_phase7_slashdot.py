"""Phase 7 — Slashdot panel.

Slashdot Zoo (SNAP): 82,140 nodes, 549,202 edges, 77.4% positive
(intermediate balance — between Bitcoin's 90%+ and SBM's 50%).

This is THE real-data regime test for the regime-dependent claim:
- Bitcoin (90%+ pos): SGCN wins by ~0.07–0.11 AUC.
- SBM (50–55% pos): HSiKAN-mixed wins by +0.17–0.30 AUC.
- Slashdot at 77% sits in between. Where does the ranking land?

Architectures (strict-Derr protocol):
  - MLP (sign-blind, no graph)
  - GCN (sign-blind, graph)
  - SignedKAN L=1 plain
  - HSiKAN-mixed leanest (k=3 + k=4 with max_k3=max_k4=30k)
  - SGCN + balance

3 seeds × 5 architectures × 1 dataset = 15 runs.
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

from .datasets import load, split
from .baselines.mlp_gcn import MLPEdge, SignBlindGCN, build_unsigned_adj
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


def run_mlp(dataset, seed, hidden=32, n_epochs=120, lr=5e-3):
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


def run_gcn(dataset, seed, hidden=32, n_layers=2, n_epochs=120, lr=5e-3):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="slashdot")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--max_k3", type=int, default=30000)
    ap.add_argument("--max_k4", type=int, default=30000)
    ap.add_argument("--enable_k4", action="store_true", default=False,
                    help="Run HSiKAN-mixed (k=3+k=4). Requires the "
                         "Rust k-cycle enumerator (hymeko.enumerate_k_cycles_rs).")
    ap.add_argument("--cycle_batch_size", type=int, default=0,
                    help="If >0, process cycles in mini-batches of this "
                         "size inside encode_edges (option 2 batching). "
                         "Bounds activation memory at O(batch · k · S · d).")
    ap.add_argument("--skip_signedkan_L1", action="store_true",
                    default=True,
                    help="Skip plain SignedKAN-L1 — no triad subsample "
                         "kwarg in run_one path, OOMs on slashdot.")
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase7_slashdot.json")
    args = ap.parse_args()

    runs = []
    t_total = time.time()

    for seed in args.seeds:
        # 1. MLP (no graph, sign-blind)
        r = run_mlp(args.dataset, seed, hidden=32, n_epochs=args.n_epochs)
        r["arch"] = "mlp_blind"
        print(f"  mlp_blind            seed={seed}  AUC={r['test_auc']:.4f} "
              f"F1m={r['test_f1_macro']:.4f}  {r['elapsed_s']:.1f}s")
        runs.append(r)

        # 2. GCN sign-blind
        r = run_gcn(args.dataset, seed, hidden=32, n_layers=2,
                     n_epochs=args.n_epochs)
        r["arch"] = "gcn_blind"
        print(f"  gcn_blind            seed={seed}  AUC={r['test_auc']:.4f} "
              f"F1m={r['test_f1_macro']:.4f}  {r['elapsed_s']:.1f}s")
        runs.append(r)

        # 3. SignedKAN L=1 plain (skipped — no triad subsample in
        #    run_one path; would OOM on slashdot's 580k triads).
        if not args.skip_signedkan_L1:
            try:
                r = run_one("signedkan", args.dataset, hidden=32, seed=seed,
                             n_epochs=args.n_epochs, lr=5e-2,
                             spline_kind="catmull_rom",
                             n_layers=1, grid=5,
                             class_weighted=False, early_stopping=False,
                             weight_decay=0.0)
                r["arch"] = "signedkan_L1"
                print(f"  signedkan_L1         seed={seed}  AUC={r['test_auc']:.4f} "
                      f"F1m={r['test_f1_macro']:.4f}  {r['elapsed_s']:.1f}s")
                runs.append(r)
            except Exception as e:
                print(f"  signedkan_L1 FAILED seed={seed}: {e!r}")

        # 4. HSiKAN — k=3-only leanest, plus optional k=3+k=4 mixed.
        # The Rust k-cycle enumerator (hymeko.enumerate_k_cycles_rs)
        # makes Slashdot k=4 enumeration tractable (~4 min for 55M
        # cycles); ``construct_k`` now subsamples raw cycles before
        # classification so we never materialise the full set.
        try:
            r = run_one_mixed(
                args.dataset, seed, hidden=16, n_layers=2, grid=3,
                n_epochs=args.n_epochs,
                arities=(3,),
                max_k3=args.max_k3,
                only_k3=True,
                coef_smooth_lam=0.0, participation_lam=0.0,
                grad_clip=0.0, weight_decay=0.0,
                early_stopping=False, class_weighted=False,
            )
            r["arch"] = "hsikan_k3_only_leanest"
            print(f"  hsikan_k3_leanest    seed={seed}  AUC={r['test_auc']:.4f} "
                  f"F1m={r['test_f1_macro']:.4f}  "
                  f"{r['elapsed_s']:.1f}s")
            runs.append(r)
        except Exception as e:
            print(f"  hsikan_k3 FAILED seed={seed}: {e!r}")

        if args.enable_k4:
            try:
                r = run_one_mixed(
                    args.dataset, seed, hidden=16, n_layers=2, grid=3,
                    n_epochs=args.n_epochs,
                    arities=(3, 4),
                    max_k3=args.max_k3,
                    max_k4=args.max_k4,
                    coef_smooth_lam=0.0, participation_lam=0.0,
                    grad_clip=0.0, weight_decay=0.0,
                    early_stopping=False, class_weighted=False,
                    cycle_batch_size=(args.cycle_batch_size
                                       if args.cycle_batch_size > 0 else None),
                )
                r["arch"] = "hsikan_k34_mixed"
                tag = (f"_b{args.cycle_batch_size}"
                       if args.cycle_batch_size > 0 else "")
                print(f"  hsikan_k34_mixed{tag}     seed={seed}  "
                      f"AUC={r['test_auc']:.4f} "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"alpha={r['alpha']}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)
            except Exception as e:
                print(f"  hsikan_k34 FAILED seed={seed}: {e!r}")

        # 5. SGCN + balance (strict-Derr)
        r = run_one_sgcn(args.dataset, seed, hidden=32, n_layers=2,
                          n_epochs=args.n_epochs, lr=5e-3,
                          balance_alpha=0.5,
                          adj_protocol="full_graph",
                          early_stopping=False,
                          class_weighted=False,
                          weight_decay=0.0)
        r["arch"] = "sgcn_balance"
        print(f"  sgcn_balance         seed={seed}  AUC={r['test_auc']:.4f} "
              f"F1m={r['test_f1_macro']:.4f}  {r['elapsed_s']:.1f}s")
        runs.append(r)

    summary = {}
    for arch in {r["arch"] for r in runs}:
        cell = [r for r in runs if r["arch"] == arch]
        aucs = [r["test_auc"] for r in cell]
        f1ms = [r["test_f1_macro"] for r in cell]
        elap = [r["elapsed_s"] for r in cell]
        summary[arch] = {
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
