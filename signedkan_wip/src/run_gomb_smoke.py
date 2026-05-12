"""HymeKo-Gömb smoke runner — real Rust-enumerated cycle pool.

Loads a signed-graph dataset, enumerates cycles via the Rust
top-K per-vertex enumerator on **train** edges only, trains a small
HymeKoGomb for N epochs, logs **validation** ROC-AUC each epoch.

**Edge split (``--edge-split``):**

* ``80_20`` (default): random 80/20 train/val (``--val-frac``), same as
  earlier smoke runs — no held-out test set.
* ``80_10_10``: ``datasets.split`` — same **train/val/test** convention
  as ``run_final_cell.cell_signed_graph`` / ``run_hsikan_sota_gate``.
  After training, prints **val** and **test** AUROC/AP/F1 (threshold 0.5).

**What full HSiKAN / ``cell_signed_graph`` still has (non-exhaustive):**
mixed arities (k=3,4,…), tuple caps / subsampling, optional walks,
``pos_weight`` on BCE, external ``nn.Linear`` classifier head (non-Slashdot),
``MixedAritySignedKAN`` depth/splines/attention/cycle-batch env knobs,
strict no-leakage protocol, entropy regulariser, etc.  Gömb here is a
different architecture; matching headline numbers requires matching
that recipe, not only the edge split.

Usage:
    python -m signedkan_wip.src.run_gomb_smoke \
        --dataset bitcoin_otc --seed 0 --n-epochs 50 --device cpu
    python -m signedkan_wip.src.run_gomb_smoke \
        --dataset bitcoin_alpha --edge-split 80_10_10 --seed 0 --n-epochs 80
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

import hymeko

from .datasets import load, split
from .hymeko_gomb import (
    GombConfig, HymeKoGomb, GombNoOuter, GombNoMiddle, GombNoInner,
    MixedArityGomb,
)

_MODELS = {
    "gomb":      HymeKoGomb,
    "no_outer":  GombNoOuter,
    "no_middle": GombNoMiddle,
    "no_inner":  GombNoInner,
}


def _enumerate_cycles(
    edges: np.ndarray, signs: np.ndarray, n: int,
    k: int = 3, m_per_vertex: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    eu = np.ascontiguousarray(edges[:, 0], dtype=np.uint32)
    ev = np.ascontiguousarray(edges[:, 1], dtype=np.uint32)
    es = np.ascontiguousarray(signs, dtype=np.int8)
    arr, _ = hymeko.enumerate_cycles_rs(
        eu, ev, es, n, k, m_per_vertex,
        score_kind="fraction_negative",
        pruner_kind="none",
        filter_kind="none",
    )
    cycles = np.asarray(arr, dtype=np.int64)
    sign_of: dict[tuple[int, int], int] = {}
    for (u, v), s in zip(edges, signs):
        sign_of[(int(u), int(v))] = int(s)
        sign_of[(int(v), int(u))] = int(s)
    cyc_signs = np.zeros_like(cycles, dtype=np.int8)
    for ci, cycle in enumerate(cycles):
        for j in range(k):
            u, v = int(cycle[j]), int(cycle[(j + 1) % k])
            cyc_signs[ci, j] = sign_of.get((u, v), 1)
    return cycles, cyc_signs


def _train_val_split(edges, signs, val_frac, seed):
    rng = np.random.default_rng(seed)
    n = edges.shape[0]
    perm = rng.permutation(n)
    n_val = int(val_frac * n)
    return (edges[perm[n_val:]], signs[perm[n_val:]],
            edges[perm[:n_val]], signs[perm[:n_val]])


def _param_breakdown(module: nn.Module) -> dict[str, int]:
    """First-level child module parameter counts (rest is unclassified)."""
    by_child: dict[str, int] = {}
    for name, child in module.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n > 0:
            by_child[name] = int(n)
    total = sum(p.numel() for p in module.parameters())
    accounted = sum(by_child.values())
    if accounted < total:
        by_child["_other_direct"] = int(total - accounted)
    return by_child


def _heldout_edge_metrics(
    y_true: np.ndarray, probs: np.ndarray, label: str,
) -> dict[str, float]:
    """Binary edge-sign metrics at 0.5 threshold + AUROC / AP.

    ``label`` is ``'val'`` or ``'test'`` — keys are ``{label}_auroc``, etc.
    """
    y = y_true.astype(np.int32)
    pred = (probs >= 0.5).astype(np.int32)
    out: dict[str, float] = {}
    lk = label
    try:
        out[f"{lk}_auroc"] = float(roc_auc_score(y, probs))
    except ValueError:
        out[f"{lk}_auroc"] = float("nan")
    try:
        out[f"{lk}_average_precision"] = float(average_precision_score(y, probs))
    except ValueError:
        out[f"{lk}_average_precision"] = float("nan")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average=None, labels=[0, 1], zero_division=0,
    )
    out[f"{lk}_precision_neg"] = float(prec[0])
    out[f"{lk}_recall_neg"] = float(rec[0])
    out[f"{lk}_f1_neg"] = float(f1[0])
    out[f"{lk}_precision_pos"] = float(prec[1])
    out[f"{lk}_recall_pos"] = float(rec[1])
    out[f"{lk}_f1_pos"] = float(f1[1])
    out[f"{lk}_f1_macro"] = float(
        f1_score(y, pred, average="macro", zero_division=0),
    )
    return out


def _degree_to_tier(degrees: np.ndarray, n_tiers: int) -> np.ndarray:
    n = degrees.shape[0]
    order = np.argsort(degrees, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n) / max(1, n - 1)
    cuts = np.linspace(0.0, 1.0, n_tiers + 1)
    tiers = np.zeros(n, dtype=np.int64)
    for i in range(n_tiers):
        lo = cuts[i]; hi = cuts[i + 1]
        mask = (ranks >= lo) & (ranks <= hi) if i == 0 else (ranks > lo) & (ranks <= hi)
        tiers[mask] = i
    return tiers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_otc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--d-embed", type=int, default=32)
    ap.add_argument("--d-outer", type=int, default=16)
    ap.add_argument("--M-outer", type=int, default=8)
    ap.add_argument("--d-middle", type=int, default=32)
    ap.add_argument("--d-core", type=int, default=32)
    ap.add_argument("--n-tiers", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW-style L2 (passed to torch.optim.Adam weight_decay).",
    )
    ap.add_argument(
        "--pos-weight-auto",
        action="store_true",
        help="Class-balanced BCE: pos_weight = n_neg/n_pos on **train** edges "
             "(same recipe as run_final_cell for non-Slashdot HSiKAN).",
    )
    ap.add_argument(
        "--edge-split",
        choices=("80_20", "80_10_10"),
        default="80_20",
        help="80_20: train/val via --val-frac (default 0.2). "
             "80_10_10: datasets.split (same convention as run_final_cell / "
             "run_hsikan_sota_gate); reports test AUROC at end.",
    )
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", choices=sorted(_MODELS), default="gomb")
    ap.add_argument(
        "--cycle-ks", default="",
        help="Comma-separated arities for MixedArityGomb (e.g. '3,4' or '4,5'). "
             "If non-empty, overrides --model with MixedArityGomb.",
    )
    args = ap.parse_args()
    cycle_ks: tuple[int, ...] = tuple(
        int(s) for s in args.cycle_ks.split(",") if s.strip()
    )

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    t0 = time.perf_counter()
    g = load(args.dataset)
    n = g.n_nodes
    print(f"[load] {args.dataset}: |V|={n}, |E|={len(g.edges)}", flush=True)

    three_way = args.edge_split == "80_10_10"
    if three_way:
        tr_idx, va_idx, te_idx = split(g, seed=args.seed)
        e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
        e_va, s_va = g.edges[va_idx], g.signs[va_idx]
        e_te, s_te = g.edges[te_idx], g.signs[te_idx]
        print(
            f"[split] 80/10/10 train={len(tr_idx)} val={len(va_idx)} "
            f"test={len(te_idx)} seed={args.seed}",
            flush=True,
        )
    else:
        e_tr, s_tr, e_va, s_va = _train_val_split(
            g.edges, g.signs, args.val_frac, args.seed,
        )
        e_te = None
        s_te = None
        print(
            f"[split] train={e_tr.shape[0]} val={e_va.shape[0]} "
            f"(val_frac={args.val_frac}) seed={args.seed}",
            flush=True,
        )

    mixed = len(cycle_ks) >= 2
    # k-pool enumeration: one (cycles, signs) per arity for mixed,
    # a single pool otherwise.
    if mixed:
        ks_used = cycle_ks
        cycles_by_k_np: dict[int, np.ndarray] = {}
        cyc_signs_by_k_np: dict[int, np.ndarray] = {}
        n_cycles_total = 0
        for k in ks_used:
            cyc_k, sgn_k = _enumerate_cycles(
                e_tr, s_tr, n, k=k, m_per_vertex=args.topk,
            )
            cycles_by_k_np[k] = cyc_k
            cyc_signs_by_k_np[k] = sgn_k
            n_cycles_total += cyc_k.shape[0]
            print(f"[cycles] k={k}: {cyc_k.shape[0]}", flush=True)
        print(f"[cycles] total={n_cycles_total} mixed={ks_used}", flush=True)
    else:
        ks_used = (args.k,)
        cycles_np, cyc_signs_np = _enumerate_cycles(
            e_tr, s_tr, n, k=args.k, m_per_vertex=args.topk,
        )
        n_cycles_total = int(cycles_np.shape[0])
        print(f"[cycles] {n_cycles_total} k={args.k}", flush=True)

    degrees = np.zeros(n, dtype=np.int64)
    for (u, v) in e_tr:
        degrees[int(u)] += 1; degrees[int(v)] += 1
    tier_of_np = _degree_to_tier(degrees, args.n_tiers)

    cfg = GombConfig(
        n_nodes=n, d_embed=args.d_embed,
        d_outer=args.d_outer, M_outer=args.M_outer,
        d_middle=args.d_middle, d_core=args.d_core,
        n_tiers=args.n_tiers, cycle_k=args.k,
    )
    if mixed:
        gomb = MixedArityGomb(cfg, cycle_ks=cycle_ks).to(device)
        model_label = f"mixed_arity_gomb[{','.join(str(k) for k in cycle_ks)}]"
    else:
        gomb = _MODELS[args.model](cfg).to(device)
        model_label = args.model

    if mixed:
        cyc_t_by_k = {
            k: torch.from_numpy(cycles_by_k_np[k]).to(device) for k in ks_used
        }
        cyc_sgn_t_by_k = {
            k: torch.from_numpy(cyc_signs_by_k_np[k]).to(device) for k in ks_used
        }
    else:
        cyc_t = torch.from_numpy(cycles_np).to(device)
        cyc_sgn_t = torch.from_numpy(cyc_signs_np).to(device)
    tier_of = torch.from_numpy(tier_of_np).to(device)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    s_tr_t = torch.from_numpy((s_tr > 0).astype(np.float32)).to(device)
    e_va_t = torch.from_numpy(e_va.astype(np.int64)).to(device)
    s_va_y = (s_va > 0).astype(np.float32)
    if three_way:
        assert e_te is not None and s_te is not None
        e_te_t = torch.from_numpy(e_te.astype(np.int64)).to(device)
        s_te_y = (s_te > 0).astype(np.float32)
    else:
        e_te_t = None
        s_te_y = None

    opt = torch.optim.Adam(
        gomb.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    if args.pos_weight_auto:
        n_pos = float(s_tr_t.sum().item())
        n_neg = float((1.0 - s_tr_t).sum().item())
        pos_w = max(n_neg, 1.0) / max(n_pos, 1.0)
        bce_pw = torch.tensor(pos_w, device=device, dtype=torch.float32)
        print(
            f"[loss] pos_weight_auto  n_pos={int(n_pos)} n_neg={int(n_neg)} "
            f"pos_weight={pos_w:.4f}",
            flush=True,
        )
    else:
        bce_pw = None
    print(
        f"[model] {model_label} d_embed={args.d_embed} M_outer={args.M_outer} "
        f"d_outer={args.d_outer} d_middle={args.d_middle} "
        f"d_core={args.d_core} n_tiers={args.n_tiers} "
        f"n_params={gomb.n_params()}  wd={args.weight_decay}",
        flush=True,
    )

    def _fwd(edges_t):
        if mixed:
            return gomb(cyc_t_by_k, cyc_sgn_t_by_k, tier_of, edges_t)
        return gomb(cyc_t, cyc_sgn_t, tier_of, edges_t)

    losses, val_aucs = [], []
    best = 0.0
    for ep in range(args.n_epochs):
        gomb.train()
        scores = _fwd(e_tr_t)
        loss = F.binary_cross_entropy_with_logits(
            scores, s_tr_t, pos_weight=bce_pw,
        )
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gomb.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))

        gomb.eval()
        with torch.no_grad():
            v_scores = _fwd(e_va_t)
            v_probs = torch.sigmoid(v_scores).cpu().numpy()
        try:
            auc = float(roc_auc_score(s_va_y, v_probs))
        except ValueError:
            auc = float("nan")
        val_aucs.append(auc)
        best = max(best, auc)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep:02d}  loss={loss.item():.4f}  "
                  f"val_auc={auc:.4f}  best={best:.4f}", flush=True)

    gomb.eval()
    with torch.no_grad():
        v_scores_final = _fwd(e_va_t)
        v_probs_final = torch.sigmoid(v_scores_final).cpu().numpy()
        if three_way and e_te_t is not None:
            te_scores_final = _fwd(e_te_t)
            te_probs_final = torch.sigmoid(te_scores_final).cpu().numpy()
        else:
            te_probs_final = None

    metrics_val = _heldout_edge_metrics(s_va_y, v_probs_final, "val")
    if three_way and te_probs_final is not None and s_te_y is not None:
        metrics_test = _heldout_edge_metrics(s_te_y, te_probs_final, "test")
        metrics = {**metrics_val, **metrics_test}
    else:
        metrics_test = None
        metrics = metrics_val
    params_by_module = _param_breakdown(gomb)

    alpha_dump = (
        gomb.alpha().cpu().tolist() if mixed else None  # type: ignore[union-attr]
    )
    summary = {
        "dataset": args.dataset, "seed": args.seed,
        "edge_split": args.edge_split,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pos_weight_auto": bool(args.pos_weight_auto),
        "model": model_label,
        "cycle_ks": list(ks_used) if mixed else None,
        "alpha_k":  alpha_dump,
        "M_outer": args.M_outer, "d_outer": args.d_outer,
        "d_middle": args.d_middle, "d_core": args.d_core,
        "n_tiers": args.n_tiers, "k": args.k, "topk": args.topk,
        "n_params": gomb.n_params(),
        "params_by_module": params_by_module,
        "n_cycles": n_cycles_total,
        "n_train_edges": int(e_tr.shape[0]),
        "n_val_edges": int(e_va.shape[0]),
        "loss_start": losses[0], "loss_end": losses[-1],
        "val_auc_start": val_aucs[0], "val_auc_end": val_aucs[-1],
        "val_auc_best": best,
        "wall_s": time.perf_counter() - t0,
        **metrics,
    }
    if three_way and e_te is not None:
        summary["n_test_edges"] = int(e_te.shape[0])
    if args.edge_split == "80_20":
        summary["val_frac"] = args.val_frac

    if three_way and metrics_test is not None:
        print(
            f"[metrics] val_AUROC={metrics_val['val_auroc']:.4f}  "
            f"val_AP={metrics_val['val_average_precision']:.4f}  "
            f"test_AUROC={metrics_test['test_auroc']:.4f}  "
            f"test_AP={metrics_test['test_average_precision']:.4f}  "
            f"R+_val={metrics_val['val_recall_pos']:.4f}  "
            f"R+_test={metrics_test['test_recall_pos']:.4f}  "
            f"F1_macro_val={metrics_val['val_f1_macro']:.4f}  "
            f"F1_macro_test={metrics_test['test_f1_macro']:.4f}",
            flush=True,
        )
    else:
        print(
            f"[metrics] AUROC={metrics_val['val_auroc']:.4f}  "
            f"AP={metrics_val['val_average_precision']:.4f}  "
            f"R+={metrics_val['val_recall_pos']:.4f}  R-={metrics_val['val_recall_neg']:.4f}  "
            f"P+={metrics_val['val_precision_pos']:.4f}  "
            f"F1_macro={metrics_val['val_f1_macro']:.4f}",
            flush=True,
        )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
