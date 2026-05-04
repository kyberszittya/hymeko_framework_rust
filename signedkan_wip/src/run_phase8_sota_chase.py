"""SOTA-chasing run on Bitcoin Alpha (leaky transductive, matches
SGCN/SiGAT published protocols).

Each cell trains an HSiKAN config and records test probabilities. The
runner also ensembles probabilities across cells for the final ensemble
AUC.

Configs cover individual lever ablations + the kitchen-sink combo:
  - baseline:              k=3+k=4, no balance loss
  - + balance_loss(λ=1.0):  k=3+k=4 with balance auxiliary
  - + arity_ensemble:      k=3, k=4, k=34, k=345 each trained, probs avg
  - all combined:          k=345 with balance, ensembled across arities
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from .datasets import load, split
from .run_phase2_mixed_arity import run_one_mixed, _evaluate


def _train_and_get_probs(dataset, seed, arities, max_per, balance_lambda,
                          n_epochs, hidden, grid, feature_edges):
    """Run one HSiKAN training and return (test_probs, test_auc, test_signs)."""
    r = run_one_mixed(
        dataset, seed=seed,
        hidden=hidden, n_layers=2, grid=grid,
        n_epochs=n_epochs,
        arities=arities,
        max_per_arity=max_per,
        coef_smooth_lam=0.0, participation_lam=0.0,
        grad_clip=0.0, weight_decay=0.0,
        early_stopping=False, class_weighted=False,
        lr_schedule="cosine",
        feature_edges=feature_edges,
        m_e_mode="edge_in_cycle",
        balance_lambda=balance_lambda,
    )
    # We need probabilities, not just AUC. Re-run the test forward.
    # Hack: we only have AUC. For ensemble, we re-run the model. The
    # cleanest path is to plumb probs through run_one_mixed; but for
    # speed let's just retrain and grab the output. To avoid this, we
    # could pickle the model — but training is fast enough that we just
    # let each ensemble member train separately and hope it returns
    # consistent probs. Below we use the alternative: don't ensemble
    # via averaging probs across seeds, but average across DIFFERENT
    # arity configs at the same seed.
    return r


def _ensemble_run(dataset, seed, arity_configs, balance_lambda,
                    n_epochs, hidden, grid, feature_edges):
    """Train one HSiKAN per arity config (same seed). Average their test
    probs and compute the ensemble AUC."""
    g = load(dataset)
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    s_te = g.signs[te_idx]
    e_te = g.edges[te_idx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_probs = []
    individual = []
    for arities, max_per in arity_configs:
        # Manual mini-loop: train + extract test probs.
        r = run_one_mixed(
            dataset, seed=seed,
            hidden=hidden, n_layers=2, grid=grid,
            n_epochs=n_epochs,
            arities=arities,
            max_per_arity=max_per,
            coef_smooth_lam=0.0, participation_lam=0.0,
            grad_clip=0.0, weight_decay=0.0,
            early_stopping=False, class_weighted=False,
            lr_schedule="cosine",
            feature_edges=feature_edges,
            m_e_mode="edge_in_cycle",
            balance_lambda=balance_lambda,
        )
        individual.append((arities, r["test_auc"]))
        # Reverse-engineer probabilities: re-train identically and snapshot.
        # ALTERNATIVE (cheap): use the test_auc as a proxy and rank-average.
        # But true ensemble needs probs. For now, just record the AUC and
        # the model's alpha; we'll re-train and snapshot probs for the
        # final ensemble runs only (separate function).

    return individual


def _train_and_snapshot_probs(dataset, seed, arities, max_per, balance_lambda,
                                n_epochs, hidden, grid, feature_edges):
    """Train + return (test_probs, test_signs, test_auc, alpha)."""
    from .hyperedges import construct
    from .n_tuples import construct_k, construct_2
    from .mixed_arity_signedkan import (MixedAritySignedKAN,
                                          MixedAritySignedKANConfig,
                                          subsample_tuples,
                                          build_edge_to_tuples,
                                          build_vertex_to_tuples)
    from .signedkan import (MultiLayerSignedKANConfig,
                             build_vertex_triad_incidence)
    from .run_phase2_mixed_arity import (_build_edge_incidence,
                                           _build_edge_incidence_vertex_adj_scipy)

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load(dataset)
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]

    per_arity_tuples = []
    for k in arities:
        cap = max_per.get(k, 30000)
        if k == 2:
            t_k = construct_2(g)
        elif k == 3:
            t_k = construct(g)
        else:
            t_k = construct_k(g, k=k, max_cycles=cap, seed=seed)
        if cap and len(t_k) > cap:
            t_k = subsample_tuples(t_k, cap, seed=seed)
        per_arity_tuples.append(t_k)

    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=2,
            hidden_dim=hidden, grid=grid, k=3,
            spline_kinds=["catmull_rom"] * 2,
            init_scale=0.05, pool_mode="sum", jk_mode="concat",
            layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none", use_residual=True,
        ),
        arities=arities,
        init_arity_logits=tuple([0.0] * len(arities)),
    )
    model = MixedAritySignedKAN(cfg).to(device)

    per_arity_train = []; per_arity_test = []
    for ai, k in enumerate(arities):
        tuples = per_arity_tuples[ai]
        triad_v_np = np.array([t.v for t in tuples], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in tuples], dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        n_tuples = len(tuples)
        M_vt = build_vertex_triad_incidence(
            triad_v_np, g.n_nodes, device, mode="sum",
        )
        edge_to_tuples = build_edge_to_tuples(tuples)
        M_e_tr = _build_edge_incidence(e_tr, edge_to_tuples, n_tuples, device)
        M_e_te = _build_edge_incidence(e_te, edge_to_tuples, n_tuples, device)
        per_arity_train.append((triad_v, triad_sigma, M_vt, M_e_tr))
        per_arity_test.append((triad_v, triad_sigma, M_vt, M_e_te))

    target_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for _ in range(n_epochs):
        model.train()
        edge_emb = model.encode_edges(per_arity_train)
        logits = model.classifier(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target_tr)
        if balance_lambda > 0.0:
            h = model.node_embed.weight
            u_idx = torch.from_numpy(e_tr[:, 0].astype(np.int64)).to(device)
            v_idx = torch.from_numpy(e_tr[:, 1].astype(np.int64)).to(device)
            cos = F.cosine_similarity(h[u_idx], h[v_idx], dim=-1)
            sign = torch.from_numpy(s_tr.astype(np.float32)).to(device)
            l_balance = (1.0 - sign * cos).mean()
            loss = loss + balance_lambda * l_balance
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    model.eval()
    with torch.no_grad():
        edge_emb_te = model.encode_edges(per_arity_test)
        logits_te = model.classifier(edge_emb_te).squeeze(-1).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_te))
    y = (s_te == 1).astype(int)
    auc = (roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan"))
    alpha = model.alpha().detach().cpu().tolist()
    return probs, y, auc, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_alpha")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--balance_lambda", type=float, default=1.0)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase8_sota_chase.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensemble configs: k=3, k=4, k=34, k=345.
    ensemble_configs = [
        ((3,),       {3: 30000}),
        ((4,),       {4: 30000}),
        ((3, 4),     {3: 30000, 4: 30000}),
        ((3, 4, 5),  {3: 30000, 4: 30000, 5: 30000}),
    ]
    # Single-config baselines (k=34 with vs without balance).
    baseline_configs = [
        ("k34_baseline",     ((3, 4), {3: 30000, 4: 30000}), 0.0),
        ("k34_balance",      ((3, 4), {3: 30000, 4: 30000}), args.balance_lambda),
        ("k345_balance",     ((3, 4, 5), {3: 30000, 4: 30000, 5: 30000}),
                              args.balance_lambda),
    ]

    print(f"Dataset: {args.dataset}, balance_lambda={args.balance_lambda}",
          flush=True)
    rows = []
    t_start = time.time()

    # Step 1: per-config single AUCs (no ensemble).
    for cell_name, (arities, max_per), balance in baseline_configs:
        for seed in args.seeds:
            tag = f"{cell_name:<14s} seed={seed}"
            t0 = time.time()
            probs, y, auc, alpha = _train_and_snapshot_probs(
                args.dataset, seed, arities, max_per, balance,
                args.n_epochs, args.hidden, args.grid, "all",
            )
            r = dict(cell=cell_name, arities=list(arities), seed=seed,
                      balance_lambda=balance, test_auc=float(auc),
                      alpha=alpha, time_s=time.time()-t0)
            rows.append(r)
            with out_path.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[{tag}]  AUC={auc:.4f}  alpha={['%.2f' % x for x in alpha]}  "
                  f"{r['time_s']:.1f}s", flush=True)

    # Step 2: ensemble across arities (with balance lambda fixed).
    print("\n=== Ensemble across arities ===", flush=True)
    for seed in args.seeds:
        all_probs = None; y_ref = None
        per_member_aucs = []
        t0 = time.time()
        for arities, max_per in ensemble_configs:
            probs, y, auc, alpha = _train_and_snapshot_probs(
                args.dataset, seed, arities, max_per,
                args.balance_lambda,
                args.n_epochs, args.hidden, args.grid, "all",
            )
            per_member_aucs.append((arities, auc))
            if all_probs is None:
                all_probs = probs.copy()
                y_ref = y
            else:
                all_probs = all_probs + probs
        all_probs = all_probs / len(ensemble_configs)
        ens_auc = roc_auc_score(y_ref, all_probs)
        r = dict(cell="ensemble_arities", seed=seed,
                  member_aucs=[(list(a), float(auc)) for a, auc in per_member_aucs],
                  test_auc=float(ens_auc), time_s=time.time()-t0)
        rows.append(r)
        with out_path.open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"[ensemble seed={seed}]  ensAUC={ens_auc:.4f}  "
              f"members={[(list(a), round(auc,4)) for a,auc in per_member_aucs]}  "
              f"{r['time_s']:.1f}s", flush=True)

    print(f"\nTotal: {time.time()-t_start:.1f}s   results → {out_path}",
          flush=True)
    print("\n=== Median across seeds ===", flush=True)
    for cell_name in [c[0] for c in baseline_configs] + ["ensemble_arities"]:
        cell_rows = [r for r in rows if r.get("cell") == cell_name]
        if not cell_rows:
            continue
        aucs = [r["test_auc"] for r in cell_rows]
        std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        print(f"{cell_name:<18s}  AUC_med={statistics.median(aucs):.4f}  "
              f"std={std:.4f}", flush=True)


if __name__ == "__main__":
    main()
