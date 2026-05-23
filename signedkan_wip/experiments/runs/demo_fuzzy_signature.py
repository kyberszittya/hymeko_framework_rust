"""Fuzzy signature demo on Bitcoin Alpha (Optuna SOTA config).

Trains a single HSIKAN at the Optuna config for a few epochs
(quick, not full SOTA — interpretability is independent of
absolute AUC), then extracts and plots three representative
fuzzy signatures:

  - a query where the model predicts positive with high
    confidence,
  - a query where it predicts negative with high confidence,
  - a query near the decision boundary (mixed votes).

Output: ``reports/figures/fuzzy_signature_bitcoin_alpha/`` with
three PNGs and a small JSON summary of each signature.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from signedkan_wip.src.datasets import load, split            # noqa: E402
from signedkan_wip.src.datasets.continuous import load_continuous  # noqa: E402
from signedkan_wip.src.core.hyperedges import construct       # noqa: E402
from signedkan_wip.src.core.n_tuples import construct_2, construct_k  # noqa: E402
from signedkan_wip.src.core.walks import construct_walks            # noqa: E402
from signedkan_wip.src.core.arc_weights import (                  # noqa: E402
    build_edge_weight_lookup, annotate_arc_weights,
)
from signedkan_wip.src.mixed_arity_signedkan import (         # noqa: E402
    MixedAritySignedKAN, MixedAritySignedKANConfig,
    subsample_tuples, build_edge_to_tuples,
)
from signedkan_wip.src.core.signedkan import (                # noqa: E402
    MultiLayerSignedKANConfig, build_vertex_triad_incidence,
)
from signedkan_wip.experiments.runs.run_phase2_mixed_arity import (  # noqa: E402
    _build_edge_incidence,
)
from signedkan_wip.src.interpret import (                     # noqa: E402
    extract_signature, plot_signature,
)


def _arity_kind(arity: int, tuple_spec: str) -> str:
    return "cycle" if tuple_spec.startswith("c") else "walk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_alpha")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--n_epochs", type=int, default=30,
                    help="Light training — demo only.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="reports/figures/fuzzy_signature_bitcoin_alpha")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] device={device} dataset={args.dataset}")

    # --- Load + split ---------------------------------------------------
    g = load(args.dataset)
    tr_idx, va_idx, te_idx = split(g, seed=args.seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]

    # --- Weighted graph for arc-weight extraction ---------------------
    # Bitcoin Alpha's [-10, +10] trust scores carry magnitude info that
    # the binary SignedGraph discards. We load it via load_continuous
    # and build an undirected (u, v) → weight lookup for annotation.
    wg = load_continuous(args.dataset)
    arc_lookup = build_edge_weight_lookup(wg)

    # --- Cycle / walk enumeration (Optuna SOTA mix: c2,c5,w2,w3,w4) ---
    cap = 100000
    arity_specs = [
        ("c2", 2), ("c5", 5), ("w2", 2), ("w3", 3), ("w4", 4),
    ]
    per_arity_tuples: list = []
    for spec, k in arity_specs:
        if spec == "c2":
            t = construct_2(g)
        elif spec.startswith("c"):
            t = construct_k(g, k=int(spec[1:]), max_cycles=cap,
                              seed=args.seed)
        elif spec.startswith("w"):
            # spec 'w3' => arity-3 walk (walk_len = k - 1 = 2).
            walk_len = int(spec[1:]) - 1
            t = construct_walks(g, walk_len=walk_len, max_walks=cap,
                                  seed=args.seed)
        else:
            raise ValueError(f"unknown spec: {spec!r}")
        if len(t) > cap:
            t = subsample_tuples(t, cap, seed=args.seed)
        per_arity_tuples.append(t)
    arities = tuple(k for _spec, k in arity_specs)
    arity_kinds = [_arity_kind(int(k), spec)
                   for spec, k in arity_specs]
    print(f"[demo] arities={arities} kinds={arity_kinds} "
          f"counts={[len(t) for t in per_arity_tuples]}")

    # --- Build per-arity inputs (train + test) -------------------------
    per_arity_train, per_arity_test = [], []
    arity_edge_signs: list = []
    arity_arc_weights: list = []
    for ai, tuples in enumerate(per_arity_tuples):
        # Annotate this arity's tuples with arc weights from the
        # weighted graph. is_walk is True for the walk slots.
        is_walk = arity_kinds[ai] == "walk"
        annotated = annotate_arc_weights(tuples, arc_lookup,
                                            is_walk=is_walk)
        triad_v_np = np.array([t.v for t in tuples], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in tuples], dtype=np.int64)
        arity_edge_signs.append(
            np.array([list(t.edge_signs) for t in tuples],
                      dtype=np.int64)
        )
        arity_arc_weights.append(
            np.array([list(t.arc_weights) for t in annotated],
                      dtype=np.float32)
        )
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        M_vt = build_vertex_triad_incidence(
            triad_v_np, g.n_nodes, device, mode="sum",
        )
        edge_to_tuples = build_edge_to_tuples(tuples, directed=False)
        n_tuples = len(tuples)
        M_e_tr = _build_edge_incidence(e_tr, edge_to_tuples, n_tuples, device)
        M_e_te = _build_edge_incidence(e_te, edge_to_tuples, n_tuples, device)
        per_arity_train.append((triad_v, triad_sigma, M_vt, M_e_tr))
        per_arity_test.append((triad_v, triad_sigma, M_vt, M_e_te))

    # --- Model + light training ----------------------------------------
    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=2, hidden_dim=args.hidden,
            grid=3, k=3, spline_kinds=["catmull_rom"] * 2,
            init_scale=0.05, pool_mode="sum", jk_mode="concat",
            layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none", use_residual=True,
        ),
        arities=arities,
        init_arity_logits=tuple([0.0] * len(arities)),
    )
    model = MixedAritySignedKAN(cfg).to(device)
    print(f"[demo] params={model.num_parameters():,}")

    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    e_te_t = torch.from_numpy(e_te.astype(np.int64)).to(device)
    target_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    target_te = torch.from_numpy((s_te == 1).astype(np.float32)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    t0 = time.time()
    for ep in range(args.n_epochs):
        model.train()
        edge_emb = model.encode_edges(per_arity_train, query_edges=e_tr_t)
        logits = model.classifier(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    train_wall = time.time() - t0
    print(f"[demo] trained {args.n_epochs} epochs in {train_wall:.1f}s")

    # --- Score test set + pick three query rows ------------------------
    model.eval()
    with torch.no_grad():
        edge_emb_te = model.encode_edges(per_arity_test, query_edges=e_te_t)
        logits_te = model.classifier(edge_emb_te).squeeze(-1)
        probs_te = torch.sigmoid(logits_te).cpu().numpy()

    # 1) most-confident positive prediction
    pos_idx = int(np.argmax(probs_te))
    # 2) most-confident negative prediction
    neg_idx = int(np.argmin(probs_te))
    # 3) closest to decision boundary
    mid_idx = int(np.argmin(np.abs(probs_te - 0.5)))

    pick = [
        ("high_positive", pos_idx),
        ("high_negative", neg_idx),
        ("decision_boundary", mid_idx),
    ]
    summary = []
    for label, q in pick:
        sig = extract_signature(
            model, per_arity_test, e_te_t, query_idx=q,
            arity_kinds=arity_kinds,
            arity_edge_signs=arity_edge_signs,
            arity_arc_weights=arity_arc_weights,
        )
        net = sig.net_vote()
        true_sign = int(s_te[q])
        mean_abs_w = sig.mean_abs_arc_weight()
        print(f"[demo] {label:>20s}  q={q} edge={sig.query_edge}  "
              f"p(+)={sig.prob_positive:.3f} (true={'+' if true_sign>0 else '-'})  "
              f"net σ·α={net:+.3f}  n_cycles={len(sig.contributions)}  "
              f"mean |w|={mean_abs_w:.3f}")
        # Plot — returns 3 panels when arc weights are present.
        axes = plot_signature(sig)
        fig = axes[0].figure
        path = out_dir / f"signature_{label}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        # Compact dump
        vote_by = sig.vote_by_arity()
        summary.append({
            "label": label,
            "query_idx": q,
            "query_edge": list(sig.query_edge),
            "true_sign": true_sign,
            "predicted_logit": sig.logit,
            "predicted_prob_positive": sig.prob_positive,
            "net_vote": net,
            "mean_abs_arc_weight": mean_abs_w,
            "n_cycles": len(sig.contributions),
            "vote_by_arity": vote_by,
            "arity_alpha": [float(a) for a in sig.arity_alpha],
            "arity_kinds": sig.arity_kinds,
            "fig_path": str(path),
        })

    (out_dir / "summary.json").write_text(json.dumps(
        {"runs": summary,
         "config": {"dataset": args.dataset, "hidden": args.hidden,
                     "n_epochs": args.n_epochs, "seed": args.seed,
                     "arities": list(arities), "arity_kinds": arity_kinds}},
        indent=2,
    ))
    print(f"[demo] wrote {out_dir}/")


if __name__ == "__main__":
    main()
