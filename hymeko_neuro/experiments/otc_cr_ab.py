"""OTC link-sign A/B: the general HSiKAN (one CR body + edge-sign head) on Bitcoin-OTC, Catmull-Rom vs B-spline.

Answers "OTC should use CR": trains the *same* pairwise signed-KAN architecture (``hymeko_neuro.core.SignedGraphHSiKAN``)
with ``activation="cr"`` vs ``"bspline"`` on the same data + split, and reports AUC / macro-F1. Uses the legacy
loader+split (``hymeko_neuro.data.datasets.legacy``) so the split is identical to the published runs; the message-
passing adjacency is built from **train edges only** (no test-label leakage). CPU-friendly, sparse backend.

    python -m hymeko_neuro.experiments.otc_cr_ab --dataset bitcoin_otc --activation cr --epochs 200
    python -m hymeko_neuro.experiments.otc_cr_ab --smoke          # both splines, few epochs, quick check
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from hymeko_neuro.core import SignedGraphHSiKAN, SparseSignedBackend, build_signed_adjacency
from hymeko_neuro.data.datasets.legacy import load, split


def run_one(dataset: str, activation: str, *, seed: int, hidden: int, n_layers: int, skip: str,
            lr: float, epochs: int) -> dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = load(dataset)
    tr, _va, te = split(g, seed=seed)
    a_pos, a_neg = build_signed_adjacency(g.edges[tr], g.signs[tr], g.n_nodes)   # train-only adjacency (no leakage)

    model = SignedGraphHSiKAN(g.n_nodes, a_pos, a_neg, hidden=hidden, n_layers=n_layers,
                              incidence="fixed", activation=activation, skip=skip,
                              backend=SparseSignedBackend())

    tr_edges = torch.tensor(g.edges[tr], dtype=torch.long)
    tr_y = torch.tensor((g.signs[tr] == 1).astype(np.float32)).unsqueeze(-1)
    te_edges = torch.tensor(g.edges[te], dtype=torch.long)
    te_y01 = (g.signs[te] == 1).astype(int)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(tr_edges), tr_y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(te_edges)).squeeze(-1).numpy()
    preds = (probs > 0.5).astype(int)
    auc = float(roc_auc_score(te_y01, probs)) if len(np.unique(te_y01)) > 1 else float("nan")
    return dict(activation=activation, auc=auc,
                f1_macro=float(f1_score(te_y01, preds, average="macro", zero_division=0)),
                f1_binary=float(f1_score(te_y01, preds, average="binary", zero_division=0)),
                wall_s=round(time.time() - t0, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_otc")
    ap.add_argument("--activation", choices=["cr", "bspline", "relu", "tanh"], default=None,
                    help="single run; omit for the cr-vs-bspline A/B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--skip", choices=["none", "residual", "highway"], default="highway")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="few epochs, both splines — a quick end-to-end check")
    a = ap.parse_args()
    epochs = 15 if a.smoke else a.epochs
    kinds = [a.activation] if a.activation else ["cr", "bspline"]
    rows = [run_one(a.dataset, k, seed=a.seed, hidden=a.hidden, n_layers=a.n_layers, skip=a.skip,
                    lr=a.lr, epochs=epochs) for k in kinds]
    print(json.dumps({"dataset": a.dataset, "seed": a.seed, "epochs": epochs, "skip": a.skip,
                      "results": rows}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
