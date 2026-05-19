"""HSiKAN tabular regression (E3 from the tabular plan).

Per-vertex regression on a tabular-derived signed graph using the P2
(correlation-sign, unsupervised) protocol.  Reads post-encoder
vertex embeddings via `model._final_h_v` and pipes them through a
linear head trained with MSE.

Compares against:
- LinearRegression (sklearn)
- RandomForestRegressor
- GradientBoostingRegressor
- MLPRegressor at matched-ish hidden width

Datasets:
- diabetes (442 samples × 10 features) — sklearn-included
- california housing (20640 × 8) — sklearn fetcher; large, optional
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from signedkan_wip.src.mixed_arity_signedkan import (
    MixedAritySignedKAN, MixedAritySignedKANConfig,
)
# Same 2026-05-11 refactor relocation as in run_tabular_smoke.py.
from signedkan_wip.src.signedkan import MultiLayerSignedKANConfig
from .run_tabular_smoke import build_M_vt, build_per_arity
from signedkan_wip.src.tabular_signed_graph import build_signed_graph_from_tabular


REGRESSION_LOADERS = {
    "diabetes": load_diabetes,
    "california": fetch_california_housing,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="diabetes",
                    choices=list(REGRESSION_LOADERS.keys()))
    ap.add_argument("--protocol", default="p2", choices=["p1", "p2"],
                    help="P2 (correlation-sign) recommended for "
                          "regression — avoids label leak; P1 needs "
                          "labels and is allowed only for ablation.")
    ap.add_argument("--k_nn", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--n-epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-samples-cap", type=int, default=2000,
                    help="Cap dataset size; full California (20K) "
                          "exceeds practical cycle-enum on CPU.")
    ap.add_argument("--target-bins", type=int, default=0,
                    help="If > 0, bin y into N quantile bands and use "
                          "the binned labels for P1 graph construction "
                          "(target-aware signs).  Regression head still "
                          "trains on continuous y.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bunch = REGRESSION_LOADERS[args.dataset]()
    X, y = bunch.data, bunch.target
    if X.shape[0] > args.n_samples_cap:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(X.shape[0], size=args.n_samples_cap, replace=False)
        X, y = X[idx], y[idx]
        print(f"[{args.dataset}] subsampled to n={args.n_samples_cap}")
    print(f"[{args.dataset}] X={X.shape}, y={y.shape}, "
          f"y_range=[{y.min():.2f}, {y.max():.2f}]")

    if args.target_bins > 0:
        # Bin y into quantile bands; use as labels for P1.
        # Forces the protocol to "p1" implicitly.
        bin_edges = np.quantile(
            y, np.linspace(0, 1, args.target_bins + 1)[1:-1]
        )
        y_binned = np.digitize(y, bin_edges)
        print(f"[binning] y → {args.target_bins} bins, edges={bin_edges}")
        g = build_signed_graph_from_tabular(
            X, y=y_binned, k=args.k_nn, protocol="p1",
        )
        graph_protocol_used = f"p1_binned_{args.target_bins}"
    else:
        g = build_signed_graph_from_tabular(
            X, y=None, k=args.k_nn, protocol=args.protocol,
        )
        graph_protocol_used = args.protocol
    print(f"[graph] n_nodes={g.n_nodes}, n_edges={g.edges.shape[0]}, "
          f"pos_frac={(g.signs == 1).mean():.3f}")

    arities = (3, 4)
    per_arity_tuples, arities_used = build_per_arity(
        g, arities, max_k=2000, seed=args.seed,
    )

    per_arity_inputs: list[tuple[torch.Tensor, ...]] = []
    for k_v, triad_v, triad_sigma in per_arity_tuples:
        triad_v_t = torch.from_numpy(triad_v).to(device)
        triad_sigma_t = torch.from_numpy(triad_sigma).to(device)
        M_vt = build_M_vt(triad_v, g.n_nodes, device)
        # Dummy 1×T M_e to satisfy the encoder's interface.
        rows = np.zeros(triad_v.shape[0], dtype=np.int64)
        cols = np.arange(triad_v.shape[0], dtype=np.int64)
        vals = np.ones(triad_v.shape[0], dtype=np.float32) / max(
            1, triad_v.shape[0]
        )
        idx = torch.tensor(np.stack([rows, cols]),
                            dtype=torch.long, device=device)
        v = torch.tensor(vals, dtype=torch.float32, device=device)
        M_e_dummy = torch.sparse_coo_tensor(
            idx, v, (1, triad_v.shape[0]),
        ).coalesce()
        per_arity_inputs.append(
            (triad_v_t, triad_sigma_t, M_vt, M_e_dummy)
        )

    Xs = StandardScaler().fit_transform(X)
    ys = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(-1)
    Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
    ys_t = torch.tensor(ys, dtype=torch.float32, device=device)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(g.n_nodes)
    fold_size = g.n_nodes // args.n_folds

    hsikan_rmses, hsikan_r2s = [], []
    lr_rmses, rf_rmses, gbr_rmses, mlp_rmses = [], [], [], []
    t0 = time.time()
    alpha_vec = []
    for fold in range(args.n_folds):
        test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.array(
            [i for i in range(g.n_nodes) if i not in set(test_idx)],
            dtype=np.int64,
        )
        train_idx_t = torch.tensor(train_idx, device=device)
        test_idx_t = torch.tensor(test_idx, device=device)
        y_train_t = ys_t[train_idx_t]
        y_test_raw = y[test_idx]

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
        head = nn.Linear(args.hidden, 1).to(device)
        opt = torch.optim.Adam(
            list(model.parameters()) + list(head.parameters()), lr=5e-3,
        )

        for ep in range(args.n_epochs):
            model.train(); head.train()
            _ = model.encode_edges(per_arity_inputs,
                                    vertex_features=Xs_t)
            h_v = model._final_h_v
            preds_t = head(h_v[train_idx_t]).squeeze(-1)
            loss = F.mse_loss(preds_t, y_train_t)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); head.eval()
        with torch.no_grad():
            _ = model.encode_edges(per_arity_inputs,
                                    vertex_features=Xs_t)
            h_v = model._final_h_v
            preds_test_std = head(h_v[test_idx_t]).squeeze(-1).cpu().numpy()
        # Un-standardise predictions to compare against raw y.
        y_mean = y.mean(); y_std = y.std() + 1e-12
        preds_test = preds_test_std * y_std + y_mean
        rmse = float(np.sqrt(mean_squared_error(y_test_raw, preds_test)))
        r2 = float(r2_score(y_test_raw, preds_test))
        hsikan_rmses.append(rmse); hsikan_r2s.append(r2)
        alpha_vec = [float(a) for a in
                      model.alpha().detach().cpu().tolist()]

        # Baselines on raw features.
        X_train, X_test = Xs[train_idx], Xs[test_idx]
        y_train_raw = y[train_idx]
        lr = LinearRegression().fit(X_train, y_train_raw)
        lr_rmses.append(float(np.sqrt(
            mean_squared_error(y_test_raw, lr.predict(X_test)))))
        rf = RandomForestRegressor(
            n_estimators=100, random_state=args.seed
        ).fit(X_train, y_train_raw)
        rf_rmses.append(float(np.sqrt(
            mean_squared_error(y_test_raw, rf.predict(X_test)))))
        gbr = GradientBoostingRegressor(
            n_estimators=100, random_state=args.seed
        ).fit(X_train, y_train_raw)
        gbr_rmses.append(float(np.sqrt(
            mean_squared_error(y_test_raw, gbr.predict(X_test)))))
        mlp = MLPRegressor(
            hidden_layer_sizes=(args.hidden,), max_iter=2000,
            random_state=args.seed,
        ).fit(X_train, y_train_raw)
        mlp_rmses.append(float(np.sqrt(
            mean_squared_error(y_test_raw, mlp.predict(X_test)))))
        n_params = sum(p.numel() for p in
                        list(model.parameters()) +
                        list(head.parameters()))

    train_s = time.time() - t0

    out = dict(
        dataset=args.dataset, protocol=graph_protocol_used, k_nn=args.k_nn,
        n_nodes=g.n_nodes, n_edges=int(g.edges.shape[0]),
        n_folds=args.n_folds, hidden=args.hidden,
        arities=list(arities_used), alpha=alpha_vec,
        hsikan_rmse_mean=float(np.mean(hsikan_rmses)),
        hsikan_rmse_std=float(np.std(hsikan_rmses)),
        hsikan_r2_mean=float(np.mean(hsikan_r2s)),
        lr_rmse_mean=float(np.mean(lr_rmses)),
        rf_rmse_mean=float(np.mean(rf_rmses)),
        gbr_rmse_mean=float(np.mean(gbr_rmses)),
        mlp_rmse_mean=float(np.mean(mlp_rmses)),
        n_params=n_params,
        train_s=train_s, n_epochs=args.n_epochs, seed=args.seed,
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
