"""Iterative prune + retrain pipeline for SignedKAN.

Pipeline:
  1. Train fresh SignedKAN at the EC recipe for $E_0$ epochs.
  2. Prune at $\\tau_1$, fine-tune $E_r$ epochs holding pruned splines
     at zero (mask applied after each ``opt.step()``).
  3. Prune at $\\tau_2 > \\tau_1$, fine-tune again.
  4. Prune at $\\tau_3 > \\tau_2$, fine-tune again.
  5. Final eval.

Compares against the post-hoc pruned model (no fine-tune) at the
same final threshold.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from signedkan_wip.src.datasets import load, split
from signedkan_wip.src.hyperedges import construct
from signedkan_wip.src.signedkan import SignedKAN, SignedKANConfig
from signedkan_wip.src.train import build_edge_to_triads
from .run_compare import build_edge_incidence
from signedkan_wip.src.iter_prune import PruneMask, count_active_splines


def _eval(model, triad_v, triad_sigma, edges, signs, M, device):
    model.eval()
    with torch.no_grad():
        triad_emb = model.encode_triads(triad_v.to(device),
                                         triad_sigma.to(device))
        edge_emb = torch.sparse.mm(M, triad_emb)
        logits = model.classifier(edge_emb).squeeze(-1).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (signs == 1).astype(int)
    preds = (probs > 0.5).astype(int)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return float(auc), float(f1m)


def train_loop(model, opt, M_train, target_tr, pos_w,
               triad_v_dev, triad_sigma_dev, n_epochs, mask=None):
    for epoch in range(n_epochs):
        model.train()
        triad_emb = model.encode_triads(triad_v_dev, triad_sigma_dev)
        edge_emb = torch.sparse.mm(M_train, triad_emb)
        logits = model.classifier(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits, target_tr, pos_weight=pos_w)
        opt.zero_grad(); loss.backward(); opt.step()
        if mask is not None:
            mask.apply(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-epochs-init", type=int, default=100)
    ap.add_argument("--n-epochs-retrain", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--retrain-lr", type=float, default=2e-2)  # smaller lr for fine-tune
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.5, 1.0, 1.8])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/iter_prune.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"\n=== {dataset}  seed={seed} ===")
            torch.manual_seed(seed); np.random.seed(seed)

            g = load(dataset)
            triads = construct(g)
            triad_v = torch.tensor([t.v for t in triads], dtype=torch.long)
            triad_sigma = torch.tensor([t.sigma for t in triads],
                                        dtype=torch.long)
            edge_to_triads = build_edge_to_triads(triads)
            tr_idx, _, te_idx = split(g, seed=seed)
            e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
            e_te, s_te = g.edges[te_idx], g.signs[te_idx]
            n_triads = triad_v.shape[0]
            M_train = build_edge_incidence(e_tr, edge_to_triads, n_triads, device)
            M_test  = build_edge_incidence(e_te, edge_to_triads, n_triads, device)
            triad_v_dev = triad_v.to(device)
            triad_sigma_dev = triad_sigma.to(device)
            target_tr = torch.from_numpy(
                (s_tr == 1).astype(np.float32)).to(device)
            n_pos = int((s_tr ==  1).sum()); n_neg = int((s_tr == -1).sum())
            pos_w = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                                  device=device)

            cfg = SignedKANConfig(n_nodes=g.n_nodes, hidden_dim=args.hidden,
                                   grid=5, k=3, spline_kind="bspline")
            model = SignedKAN(cfg).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=1e-5)

            t0 = time.time()
            train_loop(model, opt, M_train, target_tr, pos_w,
                        triad_v_dev, triad_sigma_dev, args.n_epochs_init)
            init_t = time.time() - t0
            base_auc, base_f1 = _eval(model, triad_v, triad_sigma, e_te, s_te,
                                       M_test, device)
            active, total = count_active_splines(model)
            print(f"  init   {init_t:.1f}s  AUC={base_auc:.4f}  "
                  f"F1m={base_f1:.4f}  active={active}/{total}")
            stages = [dict(stage="init", n_pruned=0, total=total,
                           auc=base_auc, f1m=base_f1)]

            mask = PruneMask(model)
            for ti, thr in enumerate(args.thresholds, start=1):
                # Update mask from current coefficients, apply, then
                # snapshot post-prune (no retrain).
                mask.update_from_threshold(model, thr)
                mask.apply(model)
                pp_auc, pp_f1 = _eval(model, triad_v, triad_sigma, e_te,
                                       s_te, M_test, device)
                z, t = mask.total_pruned()
                print(f"  prune  τ={thr}  pruned={z}/{t} ({100*z/t:.1f}%) "
                      f"  post-prune AUC={pp_auc:.4f} F1m={pp_f1:.4f}")
                # Retrain holding pruned at zero. New optimizer (so Adam
                # state doesn't carry through pruned coords).
                opt_r = torch.optim.Adam(model.parameters(),
                                          lr=args.retrain_lr,
                                          weight_decay=1e-5)
                train_loop(model, opt_r, M_train, target_tr, pos_w,
                            triad_v_dev, triad_sigma_dev,
                            args.n_epochs_retrain, mask=mask)
                rt_auc, rt_f1 = _eval(model, triad_v, triad_sigma, e_te,
                                       s_te, M_test, device)
                print(f"  retrn  τ={thr}  pruned={z}/{t} ({100*z/t:.1f}%) "
                      f"  retrained AUC={rt_auc:.4f} F1m={rt_f1:.4f}  "
                      f"(Δvinit AUC={rt_auc-base_auc:+.4f})")
                stages.append(dict(stage=f"retrain_t{ti}", threshold=thr,
                                    n_pruned=z, total=t,
                                    auc_post_prune=pp_auc,
                                    f1m_post_prune=pp_f1,
                                    auc_retrained=rt_auc,
                                    f1m_retrained=rt_f1))

            results.append(dict(
                dataset=dataset, seed=seed,
                base_auc=base_auc, base_f1m=base_f1,
                stages=stages,
            ))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
