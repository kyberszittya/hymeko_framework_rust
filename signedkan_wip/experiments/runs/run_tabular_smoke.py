"""HSiKAN tabular smoke test — Iris dataset, edge-sign prediction.

Loads Iris from sklearn, builds a P1 (k-NN + class-agreement) signed
graph, splits edges 80/20, trains a small HSiKAN encoder + per-edge
classifier, reports test AUC.

This is the E1 experiment from
`docs/plans_hsikan_tabular_benchmarks_2026_05_09.md` — the
question is whether HSiKAN trains end-to-end on a tabular-derived
signed graph.

Run:
    python -m signedkan_wip.experiments.runs.run_tabular_smoke
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
from sklearn.metrics import f1_score, roc_auc_score

from signedkan_wip.src.mixed_arity_signedkan import (
    MixedAritySignedKAN, MixedAritySignedKANConfig,
)
# `MultiLayerSignedKANConfig` moved to `signedkan` in the 2026-05-11
# mixed_arity_signedkan refactor (per CLAUDE.md §6.5 #4 split). The
# tabular smoke was written pre-refactor and used the stale re-export.
from signedkan_wip.src.core.signedkan import MultiLayerSignedKANConfig
from signedkan_wip.src.core.n_tuples import construct_2, construct_k
from signedkan_wip.src.tabular_signed_graph import build_signed_graph_from_tabular


DATASET_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def build_per_arity(g, arities, max_k=5000, seed=0):
    """Replicate the per-arity tuple construction from run_final_cell."""
    per_arity = []
    arities_used = []
    for k_v in arities:
        if k_v == 2:
            t_k = construct_2(g)
        else:
            t_k = construct_k(g, k=k_v, max_cycles=max_k, seed=seed)
        if not t_k:
            print(f"  arity {k_v}: empty, skipping")
            continue
        triad_v = np.array([t.v for t in t_k], dtype=np.int64)
        triad_sigma = np.array([t.sigma for t in t_k], dtype=np.int64)
        per_arity.append((k_v, triad_v, triad_sigma))
        arities_used.append(k_v)
    return per_arity, arities_used


def build_M_vt(triad_v: np.ndarray, n_nodes: int, device):
    """Sparse (V, T) sum-mode incidence."""
    T, k = triad_v.shape
    rows = triad_v.reshape(-1)
    cols = np.repeat(np.arange(T, dtype=np.int64), k)
    vals = np.ones_like(rows, dtype=np.float32)
    idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
    val = torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, (n_nodes, T)).coalesce()


def build_M_e(triad_v: np.ndarray, k_v: int, edges_query: np.ndarray,
              n_query_edges: int, device, is_walk=False, k2_self=False):
    """Per-edge sparse incidence: (E, T), value = 1/|N(e)|.
    For non-walk and k_v != 2, exclude k=2 self-tuple from query edge.
    """
    T = triad_v.shape[0]
    k = triad_v.shape[1]
    edge_to_tuples: dict = {}
    edge_to_self: dict = {}
    for ti in range(T):
        cyc = triad_v[ti]
        if is_walk:
            for j in range(k - 1):
                u_, v_ = int(cyc[j]), int(cyc[j + 1])
                key = (min(u_, v_), max(u_, v_))
                edge_to_tuples.setdefault(key, []).append(ti)
        else:
            if k_v == 2:
                key2 = (min(int(cyc[0]), int(cyc[1])),
                         max(int(cyc[0]), int(cyc[1])))
                edge_to_self[key2] = ti
            for j in range(k):
                u_, v_ = int(cyc[j]), int(cyc[(j + 1) % k])
                key = (min(u_, v_), max(u_, v_))
                edge_to_tuples.setdefault(key, []).append(ti)
    rows, cols, vals = [], [], []
    for ei, e in enumerate(edges_query):
        u_, v_ = int(e[0]), int(e[1])
        key = (min(u_, v_), max(u_, v_))
        ids = edge_to_tuples.get(key, [])
        if (not is_walk) and k_v == 2:
            self_t = edge_to_self.get(key)
            if self_t is not None:
                ids = [t for t in ids if t != self_t]
        if not ids:
            continue
        w = 1.0 / float(len(ids))
        for t in ids:
            rows.append(ei); cols.append(int(t)); vals.append(w)
    if not rows:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,)),
            (n_query_edges, T),
        ).to(device)
    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    v = torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, v, (n_query_edges, T)).coalesce()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="iris",
                    choices=list(DATASET_LOADERS.keys()))
    ap.add_argument("--protocol", default="p1", choices=["p1", "p2"])
    ap.add_argument("--k_nn", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load + build signed graph ───────────────────────────────────
    bunch = DATASET_LOADERS[args.dataset]()
    X, y = bunch.data, bunch.target
    print(f"[{args.dataset}] X={X.shape}, y={y.shape}, "
          f"classes={len(np.unique(y))}")
    g = build_signed_graph_from_tabular(
        X, y=y, k=args.k_nn, protocol=args.protocol,
    )
    print(f"[graph] n_nodes={g.n_nodes}, n_edges={g.edges.shape[0]}, "
          f"pos_frac={(g.signs == 1).mean():.3f}")

    # ── Split edges 80/20 train/test ────────────────────────────────
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(g.edges.shape[0])
    n_test = max(20, int(0.2 * g.edges.shape[0]))
    te_idx = perm[:n_test]; tr_idx = perm[n_test:]
    e_tr = g.edges[tr_idx]; e_te = g.edges[te_idx]
    y_tr = torch.tensor((g.signs[tr_idx] == 1).astype(np.float32),
                         device=device)
    y_te = (g.signs[te_idx] == 1).astype(np.float32)

    # The graph passed to cycle enumeration must be the TRAIN graph
    # (M_e_tr / M_e_te are built against full enumeration but query
    # edges differ).
    g_full = g
    arities = (3, 4)
    per_arity_tuples, arities_used = build_per_arity(
        g_full, arities, max_k=2000, seed=args.seed,
    )

    per_arity_tr = []
    per_arity_te = []
    for k_v, triad_v, triad_sigma in per_arity_tuples:
        triad_v_t = torch.from_numpy(triad_v).to(device)
        triad_sigma_t = torch.from_numpy(triad_sigma).to(device)
        M_vt = build_M_vt(triad_v, g.n_nodes, device)
        M_e_tr = build_M_e(triad_v, k_v, e_tr, e_tr.shape[0], device)
        M_e_te = build_M_e(triad_v, k_v, e_te, e_te.shape[0], device)
        per_arity_tr.append((triad_v_t, triad_sigma_t, M_vt, M_e_tr))
        per_arity_te.append((triad_v_t, triad_sigma_t, M_vt, M_e_te))
        print(f"  k={k_v}: T={triad_v.shape[0]}, "
              f"M_e_tr nnz={M_e_tr._nnz()}, M_e_te nnz={M_e_te._nnz()}")

    # ── Build model ─────────────────────────────────────────────────
    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=2, hidden_dim=args.hidden,
            grid=3, k=3, spline_kinds=["catmull_rom"] * 2,
            init_scale=0.05, pool_mode="sum", jk_mode="concat",
            layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none", use_residual=True),
        arities=tuple(arities_used),
        init_arity_logits=tuple([0.0] * len(arities_used)),
    )
    model = MixedAritySignedKAN(cfg).to(device)
    clf = nn.Linear(args.hidden * 2, 1).to(device)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(clf.parameters()), lr=5e-3,
    )
    n_params = sum(p.numel() for p in list(model.parameters()) +
                                       list(clf.parameters()))
    print(f"[model] params={n_params}")

    # ── Train ──────────────────────────────────────────────────────
    t0 = time.time()
    for ep in range(args.n_epochs):
        model.train(); clf.train()
        edge_emb = model.encode_edges(per_arity_tr)
        logits = clf(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:3d} loss={loss.item():.4f}")
    train_s = time.time() - t0

    # ── Eval ───────────────────────────────────────────────────────
    model.eval(); clf.eval()
    with torch.no_grad():
        edge_emb = model.encode_edges(per_arity_te)
        probs = torch.sigmoid(clf(edge_emb).squeeze(-1)).cpu().numpy()
    auc = (roc_auc_score(y_te, probs)
           if len(set(y_te)) > 1 else float("nan"))
    f1 = f1_score(y_te, probs > 0.5, average="macro", zero_division=0)
    alpha = [float(a) for a in model.alpha().detach().cpu().tolist()]

    out = dict(
        dataset=args.dataset, protocol=args.protocol, k_nn=args.k_nn,
        n_nodes=g.n_nodes, n_edges=int(g.edges.shape[0]),
        hidden=args.hidden, arities=list(arities_used), alpha=alpha,
        auc=float(auc), f1m=float(f1), n_params=n_params,
        train_s=train_s, n_epochs=args.n_epochs, seed=args.seed,
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
