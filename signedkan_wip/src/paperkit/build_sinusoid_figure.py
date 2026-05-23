"""Train a SignedKAN, pick the 8 highest-activity splines, and plot
the spline output overlaid with its fitted symbolic form.

Produces ``signedkan_wip/paper/figures/sinusoid_distillation.pdf``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..datasets import load, split
from ..core.hyperedges import construct
from ..core.signedkan import SignedKAN, SignedKANConfig
from ..core.train import build_edge_to_triads
from signedkan_wip.experiments.runs.run_compare import build_edge_incidence
from ..core.prune_distill import (measure_activity, sample_spline_activation,
                             fit_symbolic, evaluate_symbolic,
                             SYMBOLIC_LIBRARY)

REPO = Path(__file__).resolve().parents[3]
FIG  = REPO / "signedkan_wip" / "paper" / "figures" / "sinusoid_distillation.pdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_otc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-epochs", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    g = load(args.dataset)
    triads = construct(g)
    triad_v = torch.tensor([t.v for t in triads], dtype=torch.long)
    triad_sigma = torch.tensor([t.sigma for t in triads], dtype=torch.long)
    edge_to_triads = build_edge_to_triads(triads)
    tr_idx, _, _ = split(g, seed=args.seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    n_triads = triad_v.shape[0]
    M_train = build_edge_incidence(e_tr, edge_to_triads, n_triads, device)

    cfg = SignedKANConfig(n_nodes=g.n_nodes, hidden_dim=args.hidden,
                           grid=5, k=3, spline_kind="bspline")
    model = SignedKAN(cfg).to(device)
    triad_v_dev = triad_v.to(device); triad_sigma_dev = triad_sigma.to(device)
    target_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    n_pos = int((s_tr ==  1).sum()); n_neg = int((s_tr == -1).sum())
    pos_w = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                          device=device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-5)

    print(f"training {args.n_epochs} epochs ...")
    for epoch in range(args.n_epochs):
        model.train()
        triad_emb = model.encode_triads(triad_v_dev, triad_sigma_dev)
        edge_emb = torch.sparse.mm(M_train, triad_emb)
        logits = model.classifier(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target_tr,
                                                    pos_weight=pos_w)
        opt.zero_grad(); loss.backward(); opt.step()

    # Collect activity-sorted (module_name, branch, channel) triples
    # across both inner and outer.
    candidates = []  # (activity, module, branch, channel, label)
    for name, mod in [("inner", model.layer.inner),
                       ("outer", model.layer.outer)]:
        act = measure_activity(mod).cpu().numpy()    # (S, C)
        S, C = act.shape
        for s in range(S):
            for c in range(C):
                candidates.append((act[s, c], mod, s, c, name))
    candidates.sort(key=lambda t: -t[0])

    top = candidates[:8]
    print("top-8 high-activity (branch, channel) splines:")
    for act, mod, s, c, name in top:
        print(f"  {name} branch={s} channel={c} ||coef||={act:.3f}")

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5), sharex=True)
    plt.rcParams.update({"font.size": 9})
    for ax, (act, mod, s, c, name) in zip(axes.flat, top):
        x_np, y_np = sample_spline_activation(mod, s, c, n_samples=200)
        fit = fit_symbolic(x_np, y_np)
        ax.plot(x_np, y_np, color="black", lw=1.6, label="learned spline")
        if fit.form != "zero":
            y_fit = evaluate_symbolic(x_np, fit)
            ax.plot(x_np, y_fit, color="tab:red", ls="--", lw=1.4,
                     label=f"{fit.form}, MSE={fit.residual:.1e}")
        ax.set_title(f"{name} σ_branch={('+' if s==0 else '−')} ch{c}",
                      fontsize=9)
        ax.legend(loc="best", fontsize=7, frameon=False)
        ax.set_xlim(-1, 1)
        ax.axhline(0.0, color="gray", lw=0.4)
    fig.suptitle(f"Top-8 high-activity SignedKAN splines after training "
                  f"({args.dataset}, seed={args.seed})  —  ~85% fit as $a\\sin(\\omega x + \\phi) + c$",
                  fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    print(f"\nwrote {FIG}")


if __name__ == "__main__":
    main()
