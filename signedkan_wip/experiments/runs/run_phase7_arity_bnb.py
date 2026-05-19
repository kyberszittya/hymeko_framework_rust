"""Mixed-arity branch-and-bound subset selection.

Algorithm
---------
1. Train one all-arities HSiKAN at arities=(k1, ..., kN). Cost: 1 train.
2. For each non-empty subset S ⊆ {k1, ..., kN}: mask αₖ to zero
   outside S, re-normalise, run a single test forward pass, record
   AUC. Cost: O(2^N) cheap forwards (~ms each).
3. Rank subsets by mask-AUC. Pick the top-K.
4. Retrain HSiKAN from scratch on each top-K subset to get the honest
   AUC (mask is an approximation; retraining lets αₖ re-tune for the
   smaller set).
5. Compare mask-rank to retrained-rank to validate the bound.

The mask-AUC is the *bound* — an estimate of what AUC any superset of
the all-arities model can achieve when restricted to S without
retraining. Subsets where mask-AUC is far below the all-arities AUC
can be pruned without retraining (the B&B step).

Cost: 1 expensive train + 2^N cheap evaluations + K cheap retrains.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import chain, combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from signedkan_wip.src.datasets import load, split, SignedGraph
from signedkan_wip.src.hyperedges import construct
from signedkan_wip.src.n_tuples import construct_k, construct_2
from signedkan_wip.src.mixed_arity_signedkan import (MixedAritySignedKAN,
                                      MixedAritySignedKANConfig,
                                      subsample_tuples,
                                      build_edge_to_tuples,
                                      build_vertex_to_tuples)
from signedkan_wip.src.signedkan import (MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)
from .run_phase2_mixed_arity import (_build_edge_incidence,
                                       _build_edge_incidence_k2,
                                       _evaluate, run_one_mixed)


def _build_per_arity_inputs(g_features, edges_array, arities,
                              max_per_arity, device, seed):
    """Same construction as run_one_mixed but returns the inputs
    structure directly (so we can rebuild for masked evaluation)."""
    per_arity_inputs = []
    per_arity_tuples = []
    for k in arities:
        cap = max_per_arity.get(k, 30000)
        if k == 2:
            t_k = construct_2(g_features)
        elif k == 3:
            t_k = construct(g_features)
        else:
            t_k = construct_k(g_features, k=k, max_cycles=cap, seed=seed)
        if cap and len(t_k) > cap:
            t_k = subsample_tuples(t_k, cap, seed=seed)
        per_arity_tuples.append(t_k)

    for ai, k in enumerate(arities):
        tuples = per_arity_tuples[ai]
        triad_v_np = np.array([t.v for t in tuples], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in tuples], dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        n_tuples = len(tuples)
        M_vt = build_vertex_triad_incidence(
            triad_v_np, g_features.n_nodes, device, mode="sum",
        )
        if k == 2:
            v2t = build_vertex_to_tuples(tuples)
            self_idx = {}
            for ti, t in enumerate(tuples):
                u, w = int(t.v[0]), int(t.v[1])
                self_idx[(min(u, w), max(u, w))] = ti
            M_e = _build_edge_incidence_k2(
                edges_array, v2t, self_idx, n_tuples, device,
            )
        else:
            edge_to_tuples = build_edge_to_tuples(tuples)
            M_e = _build_edge_incidence(
                edges_array, edge_to_tuples, n_tuples, device,
            )
        per_arity_inputs.append((triad_v, triad_sigma, M_vt, M_e))
    return per_arity_inputs


def _powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sbm_n200_k4_s0")
    ap.add_argument("--arities", nargs="+", type=int,
                    default=[2, 3, 4, 5, 6])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--max_per_arity", type=int, default=10000)
    ap.add_argument("--top_k_retrain", type=int, default=10)
    ap.add_argument("--out_dir", default=
                    "signedkan_wip/experiments/results/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase7_arity_bnb_{args.dataset}.json"

    arities = tuple(args.arities)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}  arities={arities}")

    # === Step 1: train all-arities once ===
    print("\n[1/4] Training all-arities HSiKAN ...")
    t0 = time.time()
    r_all = run_one_mixed(
        args.dataset, seed=args.seed,
        hidden=args.hidden, n_layers=2, grid=args.grid,
        n_epochs=args.n_epochs,
        arities=arities,
        max_per_arity={k: args.max_per_arity for k in arities},
        coef_smooth_lam=0.0, participation_lam=0.0,
        grad_clip=0.0, weight_decay=0.0,
        early_stopping=False, class_weighted=False,
        lr_schedule="cosine",
        feature_edges="all",
    )
    train_time = time.time() - t0
    print(f"  all-arities AUC={r_all['test_auc']:.4f}  "
          f"alpha={['%.3f' % x for x in r_all['alpha']]}  "
          f"in {train_time:.1f}s")

    # === Step 2: rebuild model + inputs for mask evaluation ===
    print("\n[2/4] Setting up model for mask-evaluation ...")
    g = load(args.dataset)
    tr_idx, va_idx, te_idx = split(g, seed=args.seed)
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]
    g_features = g  # feature_edges="all"

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=2,
            hidden_dim=args.hidden, grid=args.grid, k=3,
            spline_kinds=["catmull_rom"] * 2,
            init_scale=0.05,
            pool_mode="sum",
            jk_mode="concat",
            layer_norm_between=True,
            share_weights=True,
            inner_skip="highway",
            outer_skip="none",
            use_residual=True,
        ),
        arities=arities,
        init_arity_logits=tuple([0.0] * len(arities)),
    )
    model = MixedAritySignedKAN(cfg).to(device)
    # Build inputs and re-train to match r_all's state (deterministic
    # given seed). The returned r_all already trained one — we redo it
    # here so we have the live model to mask. (Cheap to retrain on
    # SBM_n200; for larger datasets cache the state_dict.)
    per_arity_test = _build_per_arity_inputs(
        g_features, e_te, arities,
        {k: args.max_per_arity for k in arities},
        device, args.seed,
    )
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    per_arity_train = _build_per_arity_inputs(
        g_features, e_tr, arities,
        {k: args.max_per_arity for k in arities},
        device, args.seed,
    )

    target_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_epochs)
    for _ in range(args.n_epochs):
        model.train()
        edge_emb = model.encode_edges(per_arity_train)
        logits = model.classifier(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target_tr)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    # Validate that the live-trained model matches r_all (close to).
    full_auc, full_f1 = _evaluate(model, per_arity_test, e_te, s_te, device)
    alpha_trained = model.alpha().detach().cpu().numpy()
    print(f"  re-train AUC={full_auc:.4f}  alpha={alpha_trained.tolist()}")

    # === Step 3: mask-evaluate every non-empty subset ===
    print(f"\n[3/4] Mask-evaluating {2**len(arities)-1} non-empty subsets ...")
    rows = []
    t0 = time.time()
    arity_idx = {k: i for i, k in enumerate(arities)}
    for subset in _powerset(arities):
        mask = torch.zeros(len(arities), dtype=torch.float32)
        for k in subset:
            mask[arity_idx[k]] = 1.0
        model.set_arity_mask(mask)
        with torch.no_grad():
            edge_emb = model.encode_edges(per_arity_test)
            logits = model.classifier(edge_emb).squeeze(-1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (s_te == 1).astype(int)
        auc = (roc_auc_score(y, probs)
                if len(np.unique(y)) > 1 else float("nan"))
        rows.append(dict(
            subset=list(subset),
            mask_auc=float(auc),
        ))
    model.set_arity_mask(None)
    print(f"  done in {time.time()-t0:.1f}s")
    rows.sort(key=lambda r: r["mask_auc"], reverse=True)

    print("\n  Top-10 by mask-AUC:")
    for r in rows[:10]:
        print(f"    subset={str(r['subset']):<25s}  mask_auc={r['mask_auc']:.4f}")
    print(f"\n  Bottom-3:")
    for r in rows[-3:]:
        print(f"    subset={str(r['subset']):<25s}  mask_auc={r['mask_auc']:.4f}")

    # === Step 4: retrain top-K subsets from scratch ===
    print(f"\n[4/4] Retraining top-{args.top_k_retrain} subsets ...")
    for r in rows[:args.top_k_retrain]:
        subset = tuple(r["subset"])
        t0 = time.time()
        try:
            r_re = run_one_mixed(
                args.dataset, seed=args.seed,
                hidden=args.hidden, n_layers=2, grid=args.grid,
                n_epochs=args.n_epochs,
                arities=subset,
                max_per_arity={k: args.max_per_arity for k in subset},
                coef_smooth_lam=0.0, participation_lam=0.0,
                grad_clip=0.0, weight_decay=0.0,
                early_stopping=False, class_weighted=False,
                lr_schedule="cosine",
                feature_edges="all",
            )
            r["retrain_auc"] = r_re["test_auc"]
            r["retrain_alpha"] = r_re["alpha"]
            print(f"    subset={list(subset)}  "
                  f"mask={r['mask_auc']:.4f}  retrain={r_re['test_auc']:.4f}  "
                  f"alpha={['%.2f' % x for x in r_re['alpha']]}  "
                  f"{time.time()-t0:.1f}s")
        except Exception as e:
            r["retrain_auc"] = None
            r["retrain_error"] = repr(e)
            print(f"    subset={list(subset)}  RETRAIN FAILED: {e!r}")

    out = {
        "config": dict(
            dataset=args.dataset,
            arities=list(arities),
            seed=args.seed,
            hidden=args.hidden, grid=args.grid,
            n_epochs=args.n_epochs,
            max_per_arity=args.max_per_arity,
        ),
        "all_arities_auc": r_all["test_auc"],
        "all_arities_alpha": r_all["alpha"],
        "retrain_auc": full_auc,
        "trained_alpha": alpha_trained.tolist(),
        "rows": rows,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
