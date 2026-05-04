"""Phase 15 — Synthetic VG relation prediction with per-edge features.

End-to-end smoke for the per-edge continuous-feature pathway on a
scene-graph relation-prediction task (the natural use case).

Task: per-edge binary classification — predict relation sign
(spatial-positive vs spatial-negative) given:
  - graph topology (objects + relations)
  - per-vertex features (bbox xyxy)
  - per-edge features (relative position, IoU, sizes, vertical offset)

Synthetic kitchen scenes only have spatial-positive relations (`on`,
`next_to` map to +1 in our SIGN_BY_RELATION). To make this a
meaningful binary classification, we randomly flip a fraction of edges
to -1 ("blocks"/"under" semantic) so the model has both classes to
distinguish — then the per-edge features (vertical offset, IoU, etc.)
become the discriminative input.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .adapters.visual_genome import (
    SIGN_BY_RELATION, edge_features_from_bboxes, synth_dataset,
)
from .datasets import SignedGraph
from .hyperedges import construct
from .mixed_arity_signedkan import (MixedAritySignedKAN,
                                      MixedAritySignedKANConfig,
                                      subsample_tuples,
                                      build_edge_to_tuples)
from .n_tuples import construct_k
from .signedkan import (MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)


def _build_per_arity(g, arity, max_per, device, n_nodes_pad, seed=0):
    if arity == 3:
        t_k = construct(g)
    else:
        t_k = construct_k(g, k=arity, max_cycles=max_per, seed=seed)
    if not t_k:
        return None
    if len(t_k) > max_per:
        t_k = subsample_tuples(t_k, max_per, seed=seed)
    triad_v_np = np.array([t.v for t in t_k], dtype=np.int64)
    triad_sigma_np = np.array([t.sigma for t in t_k], dtype=np.int64)
    triad_v = torch.from_numpy(triad_v_np).to(device)
    triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
    M_vt = build_vertex_triad_incidence(triad_v_np, n_nodes_pad, device, mode="sum")
    edge_to_tuples = build_edge_to_tuples(t_k)
    rows, cols, vals = [], [], []
    for ei, e in enumerate(g.edges):
        key = (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1])))
        ids = edge_to_tuples.get(key, [])
        if not ids: continue
        w = 1.0 / float(len(ids))
        for t in ids:
            rows.append(ei); cols.append(int(t)); vals.append(w)
    if rows:
        idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
        v = torch.tensor(vals, dtype=torch.float32, device=device)
        M_e = torch.sparse_coo_tensor(idx, v, (g.edges.shape[0], len(t_k))).coalesce()
    else:
        M_e = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,)),
            (g.edges.shape[0], len(t_k))).to(device)
    return [(triad_v, triad_sigma, M_vt, M_e)]


def _build_e2v(g, n_nodes, device):
    e_arr = g.edges
    rows = np.concatenate([e_arr[:, 0], e_arr[:, 1]])
    cols = np.concatenate([np.arange(e_arr.shape[0]), np.arange(e_arr.shape[0])])
    vals = np.full(2 * e_arr.shape[0], 0.5, dtype=np.float32)
    idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
    v = torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, v, (n_nodes, e_arr.shape[0])).coalesce()


def _flip_edge_signs(g: SignedGraph, frac: float, rng: random.Random) -> SignedGraph:
    """Randomly flip a fraction of edge signs to produce a balanced
    binary classification task."""
    new_signs = g.signs.copy()
    n = g.signs.shape[0]
    flip_n = int(frac * n)
    if flip_n == 0: return g
    flip_idx = rng.sample(range(n), flip_n)
    for i in flip_idx:
        new_signs[i] = -new_signs[i]
    return SignedGraph(edges=g.edges, signs=new_signs.astype(np.int8),
                         n_nodes=g.n_nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenes", type=int, default=200)
    ap.add_argument("--n_objects", type=int, default=10)
    ap.add_argument("--flip_frac", type=float, default=0.4)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=80)
    args = ap.parse_args()

    print("=== Phase 15 — synthetic VG relation prediction ===", flush=True)
    print(f"  n_scenes={args.n_scenes}  n_objects={args.n_objects}  "
          f"flip_frac={args.flip_frac}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aggregate = {"mlp": [], "hsikan": []}
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = random.Random(seed)

        # Build dataset.
        ds_raw = synth_dataset(n_scenes=args.n_scenes, seed=seed)
        # Filter scenes that have at least 2 edges (need both classes
        # potentially after flipping).
        ds = [(g, vf, sg) for g, vf, sg in ds_raw if g.edges.shape[0] >= 2]
        # Random sign-flip to balance classes.
        ds_flipped = []
        for g, vf, sg in ds:
            g_flipped = _flip_edge_signs(g, args.flip_frac, rng)
            ds_flipped.append((g_flipped, vf, sg))

        # Pool all edges from all scenes into one classification dataset
        # (with per-edge features).
        all_edge_features = []
        all_labels = []
        all_scene_ids = []
        for sid, (g, vf, sg) in enumerate(ds_flipped):
            ef = edge_features_from_bboxes(g, vf)
            for ei in range(g.edges.shape[0]):
                all_edge_features.append(ef[ei])
                all_labels.append(int(g.signs[ei] == 1))
                all_scene_ids.append(sid)
        X = np.stack(all_edge_features)
        y = np.array(all_labels)
        n_edges_total = len(X)
        if seed == args.seeds[0]:
            print(f"  total edges: {n_edges_total}  positive frac: {y.mean():.2f}",
                  flush=True)

        # Train/test split by SCENE (so scenes don't bleed across).
        n_scenes_actual = len(ds_flipped)
        n_train_scenes = int(0.7 * n_scenes_actual)
        train_scene_ids = set(range(n_train_scenes))
        train_mask = np.array([s in train_scene_ids for s in all_scene_ids])
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[~train_mask], y[~train_mask]

        # MLP baseline (per-edge features → relation sign).
        Xt = torch.from_numpy(X_tr.astype(np.float32)).to(device)
        Yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
        Xe = torch.from_numpy(X_te.astype(np.float32)).to(device)
        Ye = torch.from_numpy(y_te.astype(np.float32)).to(device)
        mlp = nn.Sequential(
            nn.Linear(X.shape[1], 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1),
        ).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        for ep in range(args.n_epochs):
            mlp.train()
            logits = mlp(Xt).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, Yt)
            opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            probs = torch.sigmoid(mlp(Xe).squeeze(-1)).cpu().numpy()
        acc = accuracy_score(y_te, probs > 0.5)
        f1 = f1_score(y_te, probs > 0.5, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_te, probs)
        except ValueError:
            auc = float("nan")
        aggregate["mlp"].append((acc, f1, auc))
        print(f"  seed={seed}  MLP: acc={acc:.3f}  f1m={f1:.3f}  auc={auc:.3f}",
              flush=True)

        # HSiKAN per-scene with per-vertex + per-edge features → per-edge classification.
        # Use vertex_adjacency M_e mode in spirit (we want per-edge prediction
        # but the architecture supports it via the existing classifier head).
        #
        # Train: for each scene, forward graph → edge_emb (E, d_jk), classifier (d_jk → 1)
        # Loss: BCE over per-edge predicted sign.
        n_nodes_pad = max(g.n_nodes for g, _, _ in ds_flipped)
        d_v = ds_flipped[0][1].shape[1]   # bbox dim = 4
        d_e = edge_features_from_bboxes(ds_flipped[0][0], ds_flipped[0][1]).shape[1]   # 6
        hidden = 16
        cfg = MixedAritySignedKANConfig(
            base=MultiLayerSignedKANConfig(
                n_nodes=n_nodes_pad, n_layers=2, hidden_dim=hidden, grid=5, k=3,
                spline_kinds=["catmull_rom"]*2, init_scale=0.05, pool_mode="sum",
                jk_mode="concat", layer_norm_between=True, share_weights=True,
                inner_skip="highway", outer_skip="none", use_residual=True),
            arities=(3,), init_arity_logits=(0.0,),
            vertex_feat_dim=d_v, edge_feat_dim=d_e,
        )
        model = MixedAritySignedKAN(cfg).to(device)
        edge_clf = nn.Linear(hidden * 2, 1).to(device)
        opt = torch.optim.Adam(
            list(model.parameters()) + list(edge_clf.parameters()), lr=5e-3)

        # Filter scenes that have ≥1 k=3 cycle so encode_edges works.
        train_scenes = []
        for sid in range(n_train_scenes):
            g, vf, sg = ds_flipped[sid]
            cycles = construct(g)
            if not cycles: continue
            train_scenes.append(sid)
        test_scenes = []
        for sid in range(n_train_scenes, n_scenes_actual):
            g, vf, sg = ds_flipped[sid]
            cycles = construct(g)
            if not cycles: continue
            test_scenes.append(sid)
        if not train_scenes or not test_scenes:
            print(f"  seed={seed}  HSiKAN: no scenes with cycles to train; "
                  f"skipping", flush=True)
            continue

        for ep in range(args.n_epochs):
            model.train(); edge_clf.train()
            random.shuffle(train_scenes)
            loss_total = 0.0
            for sid in train_scenes:
                g, vf, _ = ds_flipped[sid]
                pa = _build_per_arity(g, 3, 1000, device, n_nodes_pad, seed=seed)
                if pa is None: continue
                vf_pad = np.zeros((n_nodes_pad, d_v), dtype=np.float32)
                vf_pad[:vf.shape[0]] = vf
                vf_t = torch.from_numpy(vf_pad).to(device)
                ef_arr = edge_features_from_bboxes(g, vf)
                ef_t = torch.from_numpy(ef_arr).to(device)
                e2v = _build_e2v(g, n_nodes_pad, device)
                edge_emb = model.encode_edges(
                    pa, query_edges=torch.from_numpy(g.edges).long().to(device),
                    vertex_features=vf_t, edge_features=ef_t,
                    edge_to_vertex=e2v)
                logits = edge_clf(edge_emb).squeeze(-1)
                target = torch.from_numpy((g.signs == 1).astype(np.float32)).to(device)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                opt.zero_grad(); loss.backward(); opt.step()
                loss_total += loss.item()
        model.eval(); edge_clf.eval()
        all_probs = []; all_true = []
        with torch.no_grad():
            for sid in test_scenes:
                g, vf, _ = ds_flipped[sid]
                pa = _build_per_arity(g, 3, 1000, device, n_nodes_pad, seed=seed)
                if pa is None: continue
                vf_pad = np.zeros((n_nodes_pad, d_v), dtype=np.float32)
                vf_pad[:vf.shape[0]] = vf
                vf_t = torch.from_numpy(vf_pad).to(device)
                ef_arr = edge_features_from_bboxes(g, vf)
                ef_t = torch.from_numpy(ef_arr).to(device)
                e2v = _build_e2v(g, n_nodes_pad, device)
                edge_emb = model.encode_edges(
                    pa, query_edges=torch.from_numpy(g.edges).long().to(device),
                    vertex_features=vf_t, edge_features=ef_t,
                    edge_to_vertex=e2v)
                probs = torch.sigmoid(edge_clf(edge_emb).squeeze(-1)).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_true.extend((g.signs == 1).astype(int).tolist())
        if all_probs:
            probs_arr = np.array(all_probs); true_arr = np.array(all_true)
            acc_h = accuracy_score(true_arr, probs_arr > 0.5)
            f1_h = f1_score(true_arr, probs_arr > 0.5, average="macro", zero_division=0)
            try:
                auc_h = roc_auc_score(true_arr, probs_arr)
            except ValueError:
                auc_h = float("nan")
            aggregate["hsikan"].append((acc_h, f1_h, auc_h))
            print(f"  seed={seed}  HSiKAN: acc={acc_h:.3f}  f1m={f1_h:.3f}  auc={auc_h:.3f}",
                  flush=True)

    print("\n=== Aggregated (median across seeds) ===", flush=True)
    for cell in ("mlp", "hsikan"):
        results = aggregate[cell]
        if not results: continue
        accs = [r[0] for r in results]
        f1ms = [r[1] for r in results]
        aucs = [r[2] for r in results]
        print(f"  {cell:<8s}  acc={statistics.median(accs):.3f}  "
              f"f1m={statistics.median(f1ms):.3f}  "
              f"auc={statistics.median(aucs):.3f}", flush=True)


if __name__ == "__main__":
    main()
