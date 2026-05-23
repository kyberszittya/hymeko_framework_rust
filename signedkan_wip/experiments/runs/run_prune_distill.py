"""Train SignedKAN, then prune + distill its splines into symbolic
forms; report accuracy at each stage.

Pipeline:
  1. Train at the EC recipe (fresh fit each seed, no checkpoint cache).
  2. Measure baseline AUC / macro-F1.
  3. Sweep pruning threshold; report (n_pruned, AUC, F1m).
  4. Distill all surviving splines to symbolic forms.
  5. Print histogram of symbolic forms learnt per layer.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from signedkan_wip.src.datasets import load, split
from signedkan_wip.src.core.hyperedges import construct
from signedkan_wip.src.core.signedkan import SignedKAN, SignedKANConfig
from signedkan_wip.src.core.train import build_edge_to_triads
from .run_compare import build_edge_incidence
from signedkan_wip.src.core.prune_distill import (measure_activity, prune_inactive,
                             distill_activation, fit_summary)


def _evaluate_signedkan(model, triad_v, triad_sigma, edges, signs,
                         M, device):
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


def _time_signedkan_forward(model, triad_v, triad_sigma, M, device,
                              n_warmup=10, n_repeats=30):
    """Median per-call forward latency in ms.

    Mask-based pruning leaves the coef tensor shape unchanged, so this
    measures the genuine inference cost at the given prune threshold.
    Latency stays roughly constant unless the kernel can exploit
    sparsity (it can't, today)."""
    import statistics, time
    tv = triad_v.to(device); ts = triad_sigma.to(device)
    sync = torch.cuda.is_available() and device.type == "cuda"
    def _fwd():
        with torch.no_grad():
            te = model.encode_triads(tv, ts)
            ee = torch.sparse.mm(M, te)
            return model.classifier(ee)
    for _ in range(n_warmup):
        _fwd()
        if sync: torch.cuda.synchronize()
    samples = []
    for _ in range(n_repeats):
        if sync: torch.cuda.synchronize()
        t0 = time.perf_counter()
        _fwd()
        if sync: torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--prune-thresholds", nargs="+", type=float,
                    default=[0.0, 0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5])
    ap.add_argument("--spline-kind", default="bspline",
                    choices=["bspline", "catmull_rom", "kochanek_bartels"])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/prune_distill.json")
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
            tr_idx, va_idx, te_idx = split(g, seed=seed)
            e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
            e_te, s_te = g.edges[te_idx], g.signs[te_idx]
            n_triads = triad_v.shape[0]
            M_train = build_edge_incidence(e_tr, edge_to_triads, n_triads, device)
            M_test  = build_edge_incidence(e_te, edge_to_triads, n_triads, device)

            cfg = SignedKANConfig(n_nodes=g.n_nodes, hidden_dim=args.hidden,
                                   grid=5, k=3, spline_kind=args.spline_kind)
            model = SignedKAN(cfg).to(device)
            triad_v_dev = triad_v.to(device)
            triad_sigma_dev = triad_sigma.to(device)
            target_tr = torch.from_numpy(
                (s_tr == 1).astype(np.float32)).to(device)
            n_pos = int((s_tr ==  1).sum())
            n_neg = int((s_tr == -1).sum())
            pos_w = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                                  device=device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=1e-5)

            t0 = time.time()
            for epoch in range(args.n_epochs):
                model.train()
                triad_emb = model.encode_triads(triad_v_dev, triad_sigma_dev)
                edge_emb = torch.sparse.mm(M_train, triad_emb)
                logits = model.classifier(edge_emb).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(
                    logits, target_tr, pos_weight=pos_w)
                opt.zero_grad(); loss.backward(); opt.step()
            train_time = time.time() - t0

            base_auc, base_f1 = _evaluate_signedkan(
                model, triad_v, triad_sigma, e_te, s_te, M_test, device)
            print(f"  trained ({train_time:.1f}s)  AUC={base_auc:.4f}  "
                  f"F1m={base_f1:.4f}")

            # Per-spline activity histogram (inner / outer).
            inner_act = measure_activity(model.layer.inner).cpu().numpy()
            outer_act = measure_activity(model.layer.outer).cpu().numpy()
            n_splines = inner_act.size + outer_act.size
            print(f"  spline activity (norm): inner mean={inner_act.mean():.3f} "
                  f"std={inner_act.std():.3f} max={inner_act.max():.3f}; "
                  f"outer mean={outer_act.mean():.3f} std={outer_act.std():.3f}")

            # Pruning sweep.
            for thr in args.prune_thresholds:
                # Snapshot coefficients so we can restore.
                inner_save = model.layer.inner.coef.data.clone()
                outer_save = model.layer.outer.coef.data.clone()
                n_in_pruned  = prune_inactive(model.layer.inner, thr)
                n_out_pruned = prune_inactive(model.layer.outer, thr)
                p_auc, p_f1 = _evaluate_signedkan(
                    model, triad_v, triad_sigma, e_te, s_te, M_test, device)
                # Time forward latency at this prune threshold. Mask-
                # based pruning keeps the coef tensor shape constant so
                # the kernel cost is invariant — a flat curve here is
                # the expected baseline for *structural* (channel-
                # removing) pruning to beat.
                lat_ms = _time_signedkan_forward(
                    model, triad_v, triad_sigma, M_test, device)
                pruned_frac = (n_in_pruned + n_out_pruned) / max(n_splines, 1)
                print(f"  thr={thr:.2f}  pruned={n_in_pruned + n_out_pruned}/"
                      f"{n_splines} ({100*pruned_frac:.1f}%)  "
                      f"AUC={p_auc:.4f}  F1m={p_f1:.4f}  "
                      f"lat={lat_ms:.2f}ms")
                results.append(dict(
                    dataset=dataset, seed=seed,
                    threshold=thr, base_auc=base_auc, base_f1=base_f1,
                    pruned_auc=p_auc, pruned_f1=p_f1,
                    n_pruned=n_in_pruned + n_out_pruned,
                    n_splines=n_splines, pruned_frac=pruned_frac,
                    fwd_latency_ms=lat_ms,
                ))
                # Restore for next threshold.
                model.layer.inner.coef.data.copy_(inner_save)
                model.layer.outer.coef.data.copy_(outer_save)

            # Symbolic distillation (no actual replacement of the
            # forward — we only fit forms and report what was learned).
            print("  distilling symbolic forms ...")
            inner_fits = distill_activation(model.layer.inner)
            outer_fits = distill_activation(model.layer.outer)
            inner_hist = fit_summary(inner_fits)
            outer_hist = fit_summary(outer_fits)
            print(f"  inner symbolic histogram: {inner_hist}")
            print(f"  outer symbolic histogram: {outer_hist}")
            results[-1]["inner_symbolic_hist"] = inner_hist
            results[-1]["outer_symbolic_hist"] = outer_hist

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} rows)")


if __name__ == "__main__":
    main()
