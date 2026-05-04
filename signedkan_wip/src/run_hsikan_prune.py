"""Highway-SignedKAN pruning + symbolic distillation.

HSiKAN uses weight-sharing across layers — there is one shared
``SignedKANLayer`` applied $L$ times. Pruning that shared layer's
splines therefore reduces compute on every layer call, so sparsity
at fixed depth is even more valuable than for the single-layer
SignedKAN we tested before.

Pipeline per (variant, dataset, seed):
  1. Train HSiKAN-BS / HSiKAN-CR with the canonical recipe.
  2. Sweep prune thresholds on the shared layer's spline coefs.
  3. Re-evaluate test AUC / F1m / loss at each threshold.
  4. Distill surviving splines to symbolic forms.
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

from .datasets import load, split
from .hyperedges import construct
from .highway_signedkan import HighwaySignedKAN, HighwaySignedKANConfig
from .signedkan import build_vertex_triad_incidence
from .train import build_edge_to_triads
from .run_compare import build_edge_incidence
from .entropy_reg import EntropyRegulariser, EntropyRegConfig
from .participation_reg import ParticipationRegulariser, triad_degree
from .prune_distill import (measure_activity, prune_inactive,
                             distill_activation, fit_summary)


def _eval(model, triad_v, triad_sigma, edges, signs, M, M_vt, device):
    model.eval()
    with torch.no_grad():
        triad_emb = model.encode_triads(triad_v, triad_sigma, M_vt)
        edge_emb = torch.sparse.mm(M, triad_emb)
        logits = model.classifier(edge_emb).squeeze(-1)
        target = torch.from_numpy((signs == 1).astype(np.float32)).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, target).item()
        probs = torch.sigmoid(logits).cpu().numpy()
    y = (signs == 1).astype(int)
    preds = (probs > 0.5).astype(int)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return float(auc), float(f1m), float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--prune-thresholds", nargs="+", type=float,
                    default=[0.0, 0.1, 0.3, 0.5, 0.8, 1.2, 1.8])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/hsikan_prune.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for spline_kind in ("bspline", "catmull_rom"):
        tag = "HSiKAN-BS" if spline_kind == "bspline" else "HSiKAN-CR"
        for dataset in args.datasets:
            for seed in args.seeds:
                print(f"\n=== {tag}  {dataset}  seed={seed} ===")
                torch.manual_seed(seed); np.random.seed(seed)

                g = load(dataset)
                triads = construct(g)
                triad_v = torch.tensor([t.v for t in triads],
                                         dtype=torch.long).to(device)
                triad_sigma = torch.tensor([t.sigma for t in triads],
                                              dtype=torch.long).to(device)
                edge_to_triads = build_edge_to_triads(triads)
                tr_idx, va_idx, te_idx = split(g, seed=seed)
                e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
                e_te, s_te = g.edges[te_idx], g.signs[te_idx]
                n_triads = triad_v.shape[0]
                M_train = build_edge_incidence(e_tr, edge_to_triads,
                                                n_triads, device)
                M_test  = build_edge_incidence(e_te, edge_to_triads,
                                                n_triads, device)
                M_vt = build_vertex_triad_incidence(
                    triad_v.cpu().numpy(), g.n_nodes, device, mode="sum")

                cfg = HighwaySignedKANConfig(n_nodes=g.n_nodes,
                                              hidden_dim=32,
                                              spline_kind=spline_kind)
                model = HighwaySignedKAN(cfg).to(device)
                n_pos = int((s_tr ==  1).sum())
                n_neg = int((s_tr == -1).sum())
                pos_w = torch.tensor(float(max(n_neg, 1))
                                       / float(max(n_pos, 1)),
                                      device=device)
                target_tr = torch.from_numpy(
                    (s_tr == 1).astype(np.float32)).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=5e-2,
                                        weight_decay=1e-5)
                ereg = EntropyRegulariser(EntropyRegConfig(
                    lam_0=0.01, lam_a=1.0, lam_b=1.0, eta=5.0,
                    target_entropy=0.5,
                    kl_normalized=True, momentum=0.9))
                part_reg = ParticipationRegulariser(lam=0.05).to(device)
                deg_np = triad_degree(triads, g.n_nodes)
                part_reg.set_degrees(deg_np)

                t0 = time.time()
                for epoch in range(args.n_epochs):
                    model.train()
                    triad_emb = model.encode_triads(triad_v, triad_sigma, M_vt)
                    edge_emb = torch.sparse.mm(M_train, triad_emb)
                    logits = model.classifier(edge_emb).squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(
                        logits, target_tr, pos_weight=pos_w)
                    loss = loss + ereg(model.node_embed.weight)
                    loss = loss + part_reg(model.node_embed.weight)
                    opt.zero_grad(); loss.backward(); opt.step()
                train_t = time.time() - t0
                base_auc, base_f1, base_loss = _eval(
                    model, triad_v, triad_sigma, e_te, s_te, M_test,
                    M_vt, device)
                # Activity on the shared layer only — that is the only
                # spline tensor in the model under weight-sharing.
                inner_act = measure_activity(model.shared_layer.inner).cpu().numpy()
                outer_act = measure_activity(model.shared_layer.outer).cpu().numpy()
                n_splines = inner_act.size + outer_act.size
                print(f"  trained ({train_t:.1f}s)  "
                      f"AUC={base_auc:.4f}  F1m={base_f1:.4f}  "
                      f"loss={base_loss:.4f}  n_splines={n_splines}")

                for thr in args.prune_thresholds:
                    inner_save = model.shared_layer.inner.coef.data.clone()
                    outer_save = model.shared_layer.outer.coef.data.clone()
                    n_in = prune_inactive(model.shared_layer.inner, thr)
                    n_out = prune_inactive(model.shared_layer.outer, thr)
                    p_auc, p_f1, p_loss = _eval(
                        model, triad_v, triad_sigma, e_te, s_te, M_test,
                        M_vt, device)
                    pruned_frac = (n_in + n_out) / max(n_splines, 1)
                    print(f"  τ={thr:.2f}  pruned={n_in+n_out}/{n_splines} "
                          f"({100*pruned_frac:.1f}%)  AUC={p_auc:.4f}  "
                          f"F1m={p_f1:.4f}  loss={p_loss:.4f}")
                    results.append(dict(
                        cfg=tag, dataset=dataset, seed=seed,
                        threshold=thr, base_auc=base_auc, base_f1=base_f1,
                        base_loss=base_loss, pruned_auc=p_auc,
                        pruned_f1=p_f1, pruned_loss=p_loss,
                        n_pruned=n_in + n_out, n_splines=n_splines,
                        pruned_frac=pruned_frac,
                    ))
                    model.shared_layer.inner.coef.data.copy_(inner_save)
                    model.shared_layer.outer.coef.data.copy_(outer_save)

                inner_fits = distill_activation(model.shared_layer.inner)
                outer_fits = distill_activation(model.shared_layer.outer)
                inner_h = fit_summary(inner_fits)
                outer_h = fit_summary(outer_fits)
                print(f"  inner symbolic histogram: {inner_h}")
                print(f"  outer symbolic histogram: {outer_h}")
                results[-1]["inner_symbolic_hist"] = inner_h
                results[-1]["outer_symbolic_hist"] = outer_h

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} rows)")


if __name__ == "__main__":
    main()
