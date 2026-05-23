"""SGCN (Derr et al. 2018, our re-implementation) on Bitcoin Alpha + OTC,
in the same train/val/test splits and EC training recipe as our other
recipes — the in-protocol comparison the gap-closing plan flagged as
the open competitive question.

Runs N_seeds × N_datasets cells and reports median AUC / F1m + the
elapsed time per run. Produces the same JSON shape as the other
sweeps so it can drop straight into the comparison table.
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
from signedkan_wip.src.baselines.sgcn_model import SGCN, build_signed_adj, extended_balance_loss


def _evaluate(model, A_pos, A_neg, edges, signs, device):
    model.eval()
    with torch.no_grad():
        z = model.encode_nodes(A_pos, A_neg)
        edges_t = torch.from_numpy(edges.astype(np.int64)).to(device)
        logits = model.edge_logits(z, edges_t).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    y = (signs == 1).astype(int)
    auc = (roc_auc_score(y, probs)
           if len(np.unique(y)) > 1 else float("nan"))
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return float(auc), float(f1m)


def run_one_sgcn(dataset: str, seed: int,
                  hidden: int = 32, n_layers: int = 2,
                  n_epochs: int = 120, lr: float = 5e-3,
                  weight_decay: float = 1e-4,
                  early_stopping: bool = True,
                  val_every: int = 5,
                  class_weighted: bool = True,
                  balance_alpha: float = 0.0,
                  balance_margin: float = 1.0,
                  adj_protocol: str = "train_only") -> dict:
    """``adj_protocol`` ∈ {"train_only", "full_graph"}.
    "full_graph" matches HSiKAN's transductive convention (adjacency
    built from all edges); "train_only" is the strict inductive form
    used by Derr 2018's reference implementation."""
    """One SGCN run with the same EC recipe as our HSiKAN measurements
    (early stop on val AUC, class-weighted BCE, weight decay)."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = load(dataset)
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_va, s_va = g.edges[va_idx], g.signs[va_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]

    if adj_protocol == "full_graph":
        # Transductive: full graph topology + signs available at
        # adjacency construction (matches HSiKAN's pipeline).
        A_pos, A_neg = build_signed_adj(g.edges, g.signs, g.n_nodes, device)
    elif adj_protocol == "train_only":
        # Strict inductive: training edges only.
        A_pos, A_neg = build_signed_adj(e_tr, s_tr, g.n_nodes, device)
    else:
        raise ValueError(f"unknown adj_protocol: {adj_protocol}")

    model = SGCN(n_nodes=g.n_nodes, hidden_dim=hidden,
                  n_layers=n_layers).to(device)
    n_params = model.num_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                            weight_decay=weight_decay)

    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    target_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    s_tr_t = torch.from_numpy(s_tr.astype(np.float32)).to(device)
    if class_weighted:
        n_pos = int((s_tr ==  1).sum())
        n_neg = int((s_tr == -1).sum())
        pos_weight = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                                   device=device)
    else:
        pos_weight = None

    best_val_auc = -1.0
    best_state = None
    best_epoch = -1
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        z = model.encode_nodes(A_pos, A_neg)
        logits = model.edge_logits(z, e_tr_t)
        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, target_tr, pos_weight=pos_weight,
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target_tr)
        if balance_alpha > 0.0:
            d = z.shape[-1] // 2
            z_B, z_U = z[:, :d], z[:, d:]
            l_bal = extended_balance_loss(z_B, z_U, e_tr_t, s_tr_t,
                                            margin=balance_margin)
            loss = loss + balance_alpha * l_bal
        opt.zero_grad(); loss.backward(); opt.step()

        if early_stopping and ((epoch + 1) % val_every == 0
                                or epoch == n_epochs - 1):
            v_auc, _ = _evaluate(model, A_pos, A_neg, e_va, s_va, device)
            if v_auc > best_val_auc:
                best_val_auc = v_auc
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
    elapsed = time.time() - t0

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    test_auc, test_f1m = _evaluate(model, A_pos, A_neg, e_te, s_te, device)
    return dict(
        model="sgcn", dataset=dataset, hidden=hidden, n_layers=n_layers,
        seed=seed, n_epochs=n_epochs, lr=lr,
        weight_decay=weight_decay,
        balance_alpha=balance_alpha, balance_margin=balance_margin,
        adj_protocol=adj_protocol,
        n_params=n_params, elapsed_s=elapsed,
        best_epoch=best_epoch, best_val_auc=best_val_auc,
        test_auc=test_auc, test_f1_macro=test_f1m,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/sgcn_baseline.json")
    args = ap.parse_args()

    runs = []
    t_total = time.time()
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_one_sgcn(dataset, seed,
                              hidden=args.hidden,
                              n_layers=args.n_layers,
                              n_epochs=args.n_epochs,
                              lr=args.lr)
            print(f"  sgcn  {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  "
                  f"F1m={r['test_f1_macro']:.4f}  "
                  f"params={r['n_params']:,}  "
                  f"{r['elapsed_s']:.1f}s")
            runs.append(r)

    summary = {}
    for dataset in args.datasets:
        cell = [r for r in runs if r["dataset"] == dataset]
        aucs = [r["test_auc"] for r in cell]
        f1ms = [r["test_f1_macro"] for r in cell]
        summary[dataset] = {
            "auc_med":   round(statistics.median(aucs), 4),
            "f1m_med":   round(statistics.median(f1ms), 4),
            "auc_seeds": [round(a, 4) for a in aucs],
            "f1m_seeds": [round(f, 4) for f in f1ms],
            "n_params":  cell[0]["n_params"] if cell else 0,
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
