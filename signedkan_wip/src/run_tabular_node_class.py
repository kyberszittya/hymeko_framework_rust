"""HSiKAN tabular node classification (E2 from the tabular plan).

Same k-NN signed graph as `run_tabular_smoke` (P1: class-agreement
signs), but instead of edge-sign prediction, train a per-vertex
softmax head on a 80/20 node split.  Predict the actual class
labels of held-out test nodes.

Compares against:
- LogisticRegression (sklearn) on raw standardised features
- MLPClassifier (1 hidden layer, sklearn) at matched-ish param budget

The encoder is the same MixedAritySignedKAN; we read post-encoder
vertex embeddings via `model._final_h_v` (stashed during forward).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .mixed_arity_signedkan import (
    MixedAritySignedKAN, MixedAritySignedKANConfig, MultiLayerSignedKANConfig,
)
from .run_tabular_smoke import (
    DATASET_LOADERS, build_M_e, build_M_vt, build_per_arity,
)
from .tabular_signed_graph import build_signed_graph_from_tabular


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="iris",
                    choices=list(DATASET_LOADERS.keys()))
    ap.add_argument("--protocol", default="p1",
                    choices=["p1", "p2", "p_unsigned"])
    ap.add_argument("--k_nn", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--n-epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bunch = DATASET_LOADERS[args.dataset]()
    X, y = bunch.data, bunch.target
    n_classes = int(np.max(y) + 1)
    print(f"[{args.dataset}] X={X.shape}, y={y.shape}, "
          f"classes={n_classes}")

    # ── Build signed graph once over the full set (transductive) ────
    g = build_signed_graph_from_tabular(
        X, y=y, k=args.k_nn, protocol=args.protocol,
    )
    print(f"[graph] n_nodes={g.n_nodes}, n_edges={g.edges.shape[0]}")

    # ── Build per-arity tuples (shared across folds — the encoder's
    # cycle enumeration depends on graph topology, not labels).
    arities = (3, 4)
    per_arity_tuples, arities_used = build_per_arity(
        g, arities, max_k=2000, seed=args.seed,
    )

    # M_e is identity-on-vertices for node classification: we don't
    # use the per-edge head.  Build M_vt only.
    per_arity_inputs: list[tuple[torch.Tensor, ...]] = []
    for k_v, triad_v, triad_sigma in per_arity_tuples:
        triad_v_t = torch.from_numpy(triad_v).to(device)
        triad_sigma_t = torch.from_numpy(triad_sigma).to(device)
        M_vt = build_M_vt(triad_v, g.n_nodes, device)
        # Empty M_e — encoder requires this argument; we don't read it.
        # But edge-pool path requires non-empty M_e.  Use a dummy 1×T
        # incidence with ones; node-class head ignores it.
        n_dummy_edges = 1
        rows = np.zeros(triad_v.shape[0], dtype=np.int64)
        cols = np.arange(triad_v.shape[0], dtype=np.int64)
        vals = np.ones(triad_v.shape[0], dtype=np.float32) / max(
            1, triad_v.shape[0]
        )
        idx = torch.tensor(np.stack([rows, cols]),
                            dtype=torch.long, device=device)
        v = torch.tensor(vals, dtype=torch.float32, device=device)
        M_e_dummy = torch.sparse_coo_tensor(
            idx, v, (n_dummy_edges, triad_v.shape[0]),
        ).coalesce()
        per_arity_inputs.append(
            (triad_v_t, triad_sigma_t, M_vt, M_e_dummy)
        )

    # ── 5-fold CV on node split ─────────────────────────────────────
    fold_aucs = []
    fold_accs = []
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(g.n_nodes)
    fold_size = g.n_nodes // args.n_folds

    Xs = StandardScaler().fit_transform(X)
    Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
    lr_accs = []
    mlp_accs = []
    t0 = time.time()
    for fold in range(args.n_folds):
        test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.array(
            [i for i in range(g.n_nodes) if i not in set(test_idx)],
            dtype=np.int64,
        )
        train_idx_t = torch.tensor(train_idx, device=device)
        test_idx_t = torch.tensor(test_idx, device=device)
        y_train = torch.tensor(y[train_idx], dtype=torch.long,
                                device=device)
        y_test = y[test_idx]

        # Build model fresh per fold.
        cfg = MixedAritySignedKANConfig(
            base=MultiLayerSignedKANConfig(
                n_nodes=g.n_nodes, n_layers=2, hidden_dim=args.hidden,
                grid=3, k=3, spline_kinds=["catmull_rom"] * 2,
                init_scale=0.05, pool_mode="sum", jk_mode="concat",
                layer_norm_between=True, share_weights=True,
                inner_skip="highway", outer_skip="none",
                use_residual=True),
            arities=tuple(arities_used),
            init_arity_logits=tuple([0.0] * len(arities_used)),
            vertex_feat_dim=Xs.shape[1],
        )
        model = MixedAritySignedKAN(cfg).to(device)
        head = nn.Linear(args.hidden, n_classes).to(device)
        opt = torch.optim.Adam(
            list(model.parameters()) + list(head.parameters()), lr=5e-3,
        )

        for ep in range(args.n_epochs):
            model.train(); head.train()
            # Trigger forward to populate model._final_h_v.
            _ = model.encode_edges(per_arity_inputs,
                                    vertex_features=Xs_t)
            h_v = model._final_h_v
            logits = head(h_v[train_idx_t])
            loss = F.cross_entropy(logits, y_train)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); head.eval()
        with torch.no_grad():
            _ = model.encode_edges(per_arity_inputs,
                                    vertex_features=Xs_t)
            h_v = model._final_h_v
            logits_test = head(h_v[test_idx_t])
            pred = logits_test.argmax(-1).cpu().numpy()
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        fold_aucs.append(f1)
        fold_accs.append(acc)
        alpha_vec = [float(a) for a in
                      model.alpha().detach().cpu().tolist()]

        # Baselines on raw features.
        X_train, X_test = Xs[train_idx], Xs[test_idx]
        lr = LogisticRegression(max_iter=1000).fit(X_train, y[train_idx])
        lr_accs.append(accuracy_score(y_test, lr.predict(X_test)))
        mlp = MLPClassifier(hidden_layer_sizes=(args.hidden,),
                            max_iter=1000, random_state=args.seed
                            ).fit(X_train, y[train_idx])
        mlp_accs.append(accuracy_score(y_test, mlp.predict(X_test)))
        n_params = sum(p.numel() for p in
                        list(model.parameters()) +
                        list(head.parameters()))

    train_s = time.time() - t0

    out = dict(
        dataset=args.dataset, protocol=args.protocol, k_nn=args.k_nn,
        n_nodes=g.n_nodes, n_edges=int(g.edges.shape[0]),
        n_classes=n_classes, n_folds=args.n_folds,
        hidden=args.hidden, arities=list(arities_used), alpha=alpha_vec,
        hsikan_acc_mean=float(np.mean(fold_accs)),
        hsikan_acc_std=float(np.std(fold_accs)),
        hsikan_f1_mean=float(np.mean(fold_aucs)),
        lr_acc_mean=float(np.mean(lr_accs)),
        mlp_acc_mean=float(np.mean(mlp_accs)),
        n_params=n_params,
        train_s=train_s, n_epochs=args.n_epochs, seed=args.seed,
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
