"""Train HSiKAN on k-uniform Davis-1967 hyperedges.

End-to-end driver for the n-tuples extension. Mirrors ``run_compare.run_one``
but consumes ``n_tuples.construct_k(g, k)`` instead of the k=3-specific
``hyperedges.construct(g)``.

Usage::

    python -m signedkan_wip.src.run_ntuples \\
        --datasets bitcoin_alpha --k 4 --seeds 0 1 2

Note on edge↔n-tuple mapping: we use *all-pairs* incidence (every C(k,2)
unordered vertex pair within an n-tuple maps to it), strictly generalising
the existing k=3 code (where all 3 vertex pairs are also cycle edges).
For k=4 this gives 6 pairs/n-tuple; only 4 are cycle-edges. The kept-pair
"presence" semantics matches the triad rule: the n-tuple summarises a
local sign-balance signature that any contained vertex-pair can borrow.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .datasets import load, split
from .n_tuples import construct_k, stats as ntuple_stats
from .signedkan import (MultiLayerSignedKAN, MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)
from .highway_signedkan import HighwaySignedKAN, HighwaySignedKANConfig
from .run_compare import build_edge_incidence


def build_edge_to_ntuples(tuples) -> dict[tuple[int, int], list[int]]:
    """All-pairs incidence: each unordered vertex pair (a, b) within an
    n-tuple's vertex set maps to that n-tuple. Strict superset of the
    k=3 code path."""
    out: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ti, t in enumerate(tuples):
        v = t.v
        k = len(v)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = v[i], v[j]
                out[(min(a, b), max(a, b))].append(ti)
    return dict(out)


def evaluate(model, triad_v, triad_sigma, edges, signs, M, M_vt, device):
    from sklearn.metrics import roc_auc_score, f1_score
    model.eval()
    with torch.no_grad():
        triad_emb = model.encode_triads(triad_v.to(device),
                                         triad_sigma.to(device), M_vt)
        edge_emb = torch.sparse.mm(M, triad_emb)
        logits = model.classifier(edge_emb).squeeze(-1).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int) * 2 - 1
    y = signs.astype(int)
    y01 = (y == 1).astype(int)
    return {
        "auc": float(roc_auc_score(y01, probs)),
        "f1_pos": float(f1_score(y, preds, pos_label=1, zero_division=0)),
        "f1_macro": float(
            f1_score(y, preds, labels=[-1, 1], average="macro", zero_division=0)
        ),
    }


def run_one_k(dataset: str, k: int, seed: int, n_epochs: int = 200,
              lr: float = 5e-2, hidden: int = 32,
              max_tuples: int | None = None) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = load(dataset)
    t0 = time.time()
    tuples = construct_k(g, k)
    enum_s = time.time() - t0
    s = ntuple_stats(tuples)
    print(f"  k={k} construct: {len(tuples):>7d} cycles in {enum_s:.1f}s, "
          f"balanced={s['balanced_frac']:.3f}")
    if max_tuples is not None and len(tuples) > max_tuples:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(tuples), size=max_tuples, replace=False)
        tuples = [tuples[i] for i in keep]
        print(f"  subsampled to {len(tuples)} (max_tuples={max_tuples})")

    triad_v = torch.tensor([t.v for t in tuples], dtype=torch.long)
    triad_sigma = torch.tensor([t.sigma for t in tuples], dtype=torch.long)
    e2t = build_edge_to_ntuples(tuples)

    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_va, s_va = g.edges[va_idx], g.signs[va_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]

    cfg = HighwaySignedKANConfig(
        n_nodes=g.n_nodes, hidden_dim=hidden, n_layers=3, k=3,
        spline_kind="catmull_rom", init_scale=0.05,
    )
    model = HighwaySignedKAN(cfg).to(device)

    M_vt = build_vertex_triad_incidence(triad_v.numpy(), g.n_nodes, device,
                                         mode="sum")
    M_train = build_edge_incidence(e_tr, e2t, len(tuples), device)
    M_val   = build_edge_incidence(e_va, e2t, len(tuples), device)
    M_test  = build_edge_incidence(e_te, e2t, len(tuples), device)

    triad_v_dev = triad_v.to(device)
    triad_sigma_dev = triad_sigma.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_pos = float((s_tr == +1).sum()); n_neg = float((s_tr == -1).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    y_tr01 = torch.tensor((s_tr == +1).astype(np.float32), device=device)
    best_val_auc = -1.0; best_state = None; best_epoch = 0; patience = 0
    train_t0 = time.time()
    for ep in range(n_epochs):
        model.train(); opt.zero_grad()
        h_e = model.encode_triads(triad_v_dev, triad_sigma_dev, M_vt)
        edge_emb_tr = torch.sparse.mm(M_train, h_e)
        logits_tr = model.classifier(edge_emb_tr).squeeze(-1)
        loss = bce(logits_tr, y_tr01)
        loss.backward(); opt.step()
        if (ep + 1) % 5 == 0 or ep == n_epochs - 1:
            v = evaluate(model, triad_v, triad_sigma, e_va, s_va, M_val, M_vt, device)
            if v["auc"] > best_val_auc:
                best_val_auc = v["auc"]; best_epoch = ep + 1
                best_state = {k_: v_.detach().clone()
                              for k_, v_ in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 6:
                    break
    train_s = time.time() - train_t0
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, triad_v, triad_sigma, e_te, s_te, M_test, M_vt, device)
    return {
        "dataset": dataset, "k": k, "seed": seed, "n_tuples": len(tuples),
        "balanced_frac": s["balanced_frac"], "best_epoch": best_epoch,
        "test_auc": test["auc"], "test_f1_pos": test["f1_pos"],
        "test_f1_macro": test["f1_macro"], "elapsed_s": round(train_s, 1),
        "enum_s": round(enum_s, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["bitcoin_alpha"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=200)
    ap.add_argument("--max_tuples", type=int, default=None,
                    help="Cap on n-tuples (random subsample) to fit GPU memory")
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/ntuples.json")
    args = ap.parse_args()

    results = []
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_one_k(dataset, args.k, seed, n_epochs=args.n_epochs,
                           max_tuples=args.max_tuples)
            print(f"  {dataset:14s} k={args.k} seed={seed} "
                  f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                  f"{r['elapsed_s']:.1f}s")
            results.append(r)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
