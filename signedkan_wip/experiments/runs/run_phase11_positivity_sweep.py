"""Phase 11 — positivity-ratio sweep on synthetic SBM.

Traces the regime transition: at what edge-positivity does
HSiKAN-mixed lose its architectural advantage over SGCN/SiGAT? Phase 6
established HSiKAN dominates at ~50% positivity (SBM); Phase 8
confirmed SGCN dominates at ~93% positivity (Bitcoin); Phase 7 placed
Slashdot at 77% in the SGCN-favoured regime. This phase fills the
intermediate values directly with controllable synthetic SBMs.

Datasets (sbmsweep_pos<XX>_s<seed>):
  pos_in ∈ {50, 55, 60, 65, 70, 75, 80, 85, 90, 95}
  → 10 positivity points, p_out=0.15 fixed.

Architectures:
  - HSiKAN-mixed leanest k=3+k=4 (mixed-arity story)
  - SGCN+balance
  - SiGAT-attn
  - GCN sign-blind (control)

3 seeds × 10 positivity points × 4 archs = 120 runs.
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
from signedkan_wip.src.baselines.mlp_gcn import SignBlindGCN, build_unsigned_adj
from signedkan_wip.src.baselines.sigat_model import SiGATAttn, build_neighbour_lists
from .run_sgcn_baseline import run_one_sgcn
from .run_phase2_mixed_arity import run_one_mixed


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
    e_te_t = torch.from_numpy(e_te.astype(np.int64)).to(device)
    with torch.no_grad():
        logits = model(A, e_te_t).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    y = (s_te == 1).astype(int)
    auc = (roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan"))
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return dict(model="gcn", dataset=dataset, seed=seed,
                test_auc=float(auc), test_f1_macro=float(f1m),
                elapsed_s=elapsed)


def run_sigat(dataset, seed, hidden=32, n_heads=4, n_epochs=200, lr=5e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load(dataset); tr_idx, _, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]
    pos_b, neg_b = build_neighbour_lists(g.edges, g.signs, g.n_nodes)
    model = SiGATAttn(g.n_nodes, hidden_dim=hidden,
                      n_heads=n_heads, n_layers=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    target = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    t0 = time.time()
    for _ in range(n_epochs):
        model.train()
        z = model.encode_nodes(pos_b, neg_b)
        logits = model.edge_logits(z, e_tr_t)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        z = model.encode_nodes(pos_b, neg_b)
        e_te_t = torch.from_numpy(e_te.astype(np.int64)).to(device)
        logits = model.edge_logits(z, e_te_t).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    y = (s_te == 1).astype(int)
    auc = (roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan"))
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return dict(model="sigat_attn", dataset=dataset, seed=seed,
                test_auc=float(auc), test_f1_macro=float(f1m),
                elapsed_s=elapsed)


POSITIVITIES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=200)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase11_positivity_sweep.json")
    args = ap.parse_args()

    runs = []
    t_total = time.time()

    for pos in POSITIVITIES:
        for seed in args.seeds:
            dataset = f"sbmsweep_pos{pos}_s{seed}"

            # Edge-positivity ratio actually realised
            g = load(dataset)
            frac_pos = float((g.signs == 1).mean())
            n_edges = int(g.edges.shape[0])

            r = run_gcn(dataset, seed, hidden=32, n_layers=2,
                         n_epochs=args.n_epochs)
            r["arch"] = "gcn_blind"; r["pos_in"] = pos; r["frac_pos"] = frac_pos
            r["n_edges"] = n_edges
            print(f"  gcn_blind   pos={pos:>3d} (real {frac_pos:.2f}) seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  {r['elapsed_s']:.1f}s")
            runs.append(r)

            try:
                r = run_one_mixed(
                    dataset, seed, hidden=16, n_layers=2, grid=3,
                    n_epochs=args.n_epochs,
                    arities=(3, 4),
                    max_per_arity={3: 30000, 4: 30000},
                    only_k3=False,
                    coef_smooth_lam=0.0, participation_lam=0.0,
                    grad_clip=0.0, weight_decay=0.0,
                    early_stopping=False, class_weighted=False,
                )
                r["arch"] = "hsikan_k34"; r["pos_in"] = pos; r["frac_pos"] = frac_pos
                r["n_edges"] = n_edges
                print(f"  hsikan_k34  pos={pos:>3d} (real {frac_pos:.2f}) seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  alpha={[round(a,2) for a in r['alpha']]}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)
            except Exception as e:
                print(f"  hsikan_k34 FAILED on {dataset}: {e!r}")

            r = run_one_sgcn(dataset, seed, hidden=32, n_layers=2,
                              n_epochs=args.n_epochs, lr=5e-3,
                              balance_alpha=0.5,
                              adj_protocol="full_graph",
                              early_stopping=False,
                              class_weighted=False,
                              weight_decay=0.0)
            r["arch"] = "sgcn_balance"; r["pos_in"] = pos; r["frac_pos"] = frac_pos
            r["n_edges"] = n_edges
            print(f"  sgcn_bal    pos={pos:>3d} (real {frac_pos:.2f}) seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  {r['elapsed_s']:.1f}s")
            runs.append(r)

            try:
                r = run_sigat(dataset, seed, hidden=32, n_heads=4,
                               n_epochs=args.n_epochs, lr=5e-3)
                r["arch"] = "sigat_attn"; r["pos_in"] = pos; r["frac_pos"] = frac_pos
                r["n_edges"] = n_edges
                print(f"  sigat_attn  pos={pos:>3d} (real {frac_pos:.2f}) seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  {r['elapsed_s']:.1f}s")
                runs.append(r)
            except Exception as e:
                print(f"  sigat_attn FAILED on {dataset}: {e!r}")

    summary = {}
    keys = sorted({(r["arch"], r["pos_in"]) for r in runs})
    for arch, pos in keys:
        cell = [r for r in runs if r["arch"] == arch and r["pos_in"] == pos]
        aucs = [r["test_auc"] for r in cell]
        f1ms = [r["test_f1_macro"] for r in cell]
        frac = cell[0]["frac_pos"]
        summary[f"{arch}|pos{pos}"] = {
            "auc_mean":  round(float(np.mean(aucs)), 4),
            "auc_std":   round(float(np.std(aucs)), 4),
            "f1m_mean":  round(float(np.mean(f1ms)), 4),
            "n_seeds":   len(cell),
            "frac_pos":  round(frac, 4),
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
