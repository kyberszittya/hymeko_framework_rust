"""Demo: Gömb fuzzy signature on Bitcoin Alpha.

Trains a lightweight HymeKoGomb on Bitcoin Alpha for a handful
of epochs (interpretation works independent of absolute AUC),
then extracts three representative signatures: high-confidence
positive, high-confidence negative, and a decision-boundary
query.

Output: ``reports/figures/gomb_signature_bitcoin_alpha/`` with
three PNGs and a summary JSON.

Note: this is an inspection demo, not an SOTA reproduction.
For SOTA Gömb numbers see [[project-gomb-strict-4dataset-2026-05-14]].
"""
from __future__ import annotations

import argparse
import json
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
from signedkan_wip.src.core.arc_weights import (              # noqa: E402
    build_edge_weight_lookup, annotate_arc_weights,
)
from signedkan_wip.src.hymeko_gomb.cascade import (           # noqa: E402
    HymeKoGomb, GombConfig,
)
from signedkan_wip.src.interpret import (                     # noqa: E402
    extract_gomb_signature, plot_gomb_signature,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_alpha")
    ap.add_argument("--n_epochs", type=int, default=120,
                    help="Light training — interpretation demo.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=
                    "reports/figures/gomb_signature_bitcoin_alpha")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] device={device} dataset={args.dataset}")

    # --- Load + split + weighted lookup --------------------------------
    g = load(args.dataset)
    tr_idx, va_idx, te_idx = split(g, seed=args.seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]
    wg = load_continuous(args.dataset)
    arc_lookup = build_edge_weight_lookup(wg)

    # --- Cycle (k=3) enumeration --------------------------------------
    triads = construct(g)
    if not triads:
        raise RuntimeError("no triangles in this graph")
    print(f"[demo] n_triads={len(triads)}")

    # Annotate with arc weights for the signature view.
    triads_with_arc = annotate_arc_weights(triads, arc_lookup,
                                             is_walk=False)

    cycles_np = np.array([t.v for t in triads], dtype=np.int64)
    signs_np = np.array([t.sigma for t in triads], dtype=np.int64)
    edge_signs_np = np.array([list(t.edge_signs) for t in triads],
                              dtype=np.int64)
    arc_weights_np = np.array([list(t.arc_weights)
                                  for t in triads_with_arc],
                                 dtype=np.float32)
    cycles_t = torch.from_numpy(cycles_np).long().to(device)
    signs_t = torch.from_numpy(signs_np).float().to(device)
    tier_of_t = torch.zeros(g.n_nodes, dtype=torch.long, device=device)

    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    e_te_t = torch.from_numpy(e_te.astype(np.int64)).to(device)
    target_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)

    # --- Lightweight Gömb config + training ---------------------------
    cfg = GombConfig(
        n_nodes=g.n_nodes,
        d_embed=8, d_outer=8, M_outer=4,
        d_middle=8,
        d_core=8, n_tiers=2,
        cycle_k=3, middle_grid=5,
    )
    model = HymeKoGomb(cfg).to(device)
    print(f"[demo] params={model.n_params():,}")
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    t0 = time.time()
    for ep in range(args.n_epochs):
        model.train()
        logits = model(cycles_t, signs_t, tier_of_t, e_tr_t).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[demo] trained {args.n_epochs} epochs in "
          f"{time.time() - t0:.1f}s")

    # --- Score test set + pick three queries --------------------------
    model.eval()
    with torch.no_grad():
        logits_te = model(cycles_t, signs_t, tier_of_t,
                            e_te_t).squeeze(-1)
        probs_te = torch.sigmoid(logits_te).cpu().numpy()

    # Filter to queries that actually have triangles incident — without
    # cycles the signature is empty. Per-query incidence count from the
    # cycle vertex sets.
    cycle_vertex_sets = [set(int(x) for x in row) for row in cycles_np]
    incident_count = np.zeros(int(e_te_t.shape[0]), dtype=np.int64)
    for qi in range(int(e_te_t.shape[0])):
        u, v = int(e_te_t[qi, 0]), int(e_te_t[qi, 1])
        incident_count[qi] = sum(
            1 for vs in cycle_vertex_sets if u in vs and v in vs
        )
    eligible = np.where(incident_count >= 3)[0]
    if len(eligible) < 3:
        # Fall back to all queries if the filter is too aggressive.
        eligible = np.arange(int(e_te_t.shape[0]))
    eligible_probs = probs_te[eligible]
    pos_idx = int(eligible[np.argmax(eligible_probs)])
    neg_idx = int(eligible[np.argmin(eligible_probs)])
    mid_idx = int(eligible[np.argmin(np.abs(eligible_probs - 0.5))])
    print(f"[demo] eligible test queries (≥3 incident cycles): "
          f"{len(eligible)} / {int(e_te_t.shape[0])}")
    pick = [
        ("high_positive", pos_idx),
        ("high_negative", neg_idx),
        ("decision_boundary", mid_idx),
    ]

    summary: list[dict] = []
    for label, q in pick:
        sig = extract_gomb_signature(
            model, cycles_t, signs_t, tier_of_t, e_te_t,
            query_idx=q,
            arc_weights=arc_weights_np,
            edge_signs=edge_signs_np,
        )
        dom = sig.shell_dominance()
        consistency = sig.cross_shell_consistency()
        true_sign = int(s_te[q])
        net = sig.net_vote()
        print(
            f"[demo] {label:>20s}  q={q} edge={sig.query_edge}  "
            f"p(+)={sig.prob_positive:.3f} "
            f"(true={'+' if true_sign > 0 else '-'})  "
            f"net σ·|h_mid|={net:+.3f}  "
            f"n_cycles={len(sig.contributions)}  "
            f"dom_outer={dom.get('outer', 0):.3f}  "
            f"dom_middle={dom.get('middle', 0):.3f}  "
            f"r_consistency={consistency:+.2f}"
        )
        axes = plot_gomb_signature(sig)
        fig = axes[0].figure
        path = out_dir / f"gomb_signature_{label}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        summary.append({
            "label": label,
            "query_idx": q,
            "query_edge": list(sig.query_edge),
            "true_sign": true_sign,
            "predicted_logit": sig.logit,
            "predicted_prob_positive": sig.prob_positive,
            "net_vote_middle": net,
            "n_cycles": len(sig.contributions),
            "shell_dominance": dom,
            "cross_shell_consistency": consistency,
            "fig_path": str(path),
        })

    (out_dir / "summary.json").write_text(json.dumps(
        {"runs": summary,
         "config": {
             "dataset": args.dataset,
             "n_epochs": args.n_epochs,
             "seed": args.seed,
             "d_embed": cfg.d_embed,
             "d_outer": cfg.d_outer,
             "M_outer": cfg.M_outer,
             "d_middle": cfg.d_middle,
             "d_core": cfg.d_core,
             "cycle_k": cfg.cycle_k,
         }},
        indent=2,
    ))
    print(f"[demo] wrote {out_dir}/")


if __name__ == "__main__":
    main()
