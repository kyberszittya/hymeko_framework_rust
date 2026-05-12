"""HSiKAN on canonical general (unsigned) graphs without
tabular-derived signs.

Tests the architectural claim that HSiKAN trains end-to-end on
plain unsigned graphs by gracefully degrading to a single-branch
σ-masked aggregator when all edge signs are +1.

Benchmarks
----------
- Karate Club (networkx): 34 nodes, 78 edges, 2 communities
  (Mr. Hi vs Officer).  Canonical community-detection test.
- Florentine Families: 15 nodes, 20 edges, social network
  (binary class: Medici-aligned or not).
- Les Misérables: 77 nodes, 254 edges, character-co-occurrence

For each: SignedGraph with all signs = +1; vertex features =
4-dim graph-structural descriptors (degree, betweenness,
closeness, clustering coefficient).  HSiKAN node classification
with k=3, k=4 cycles.

Compares against:
- Logistic regression on the 4-dim graph features
- Spectral clustering on adjacency (standard community detection)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .datasets import SignedGraph
from .mixed_arity_signedkan import (
    MixedAritySignedKAN, MixedAritySignedKANConfig, MultiLayerSignedKANConfig,
)
from .run_tabular_smoke import build_M_vt, build_per_arity


def graph_features(G: nx.Graph) -> np.ndarray:
    """4-dim per-vertex structural features."""
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G)
    clo = nx.closeness_centrality(G)
    clu = nx.clustering(G)
    n = G.number_of_nodes()
    feats = np.zeros((n, 4), dtype=np.float32)
    nodes = sorted(G.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    for v in nodes:
        i = node_to_idx[v]
        feats[i] = [deg[v], btw[v], clo[v], clu[v]]
    return feats, node_to_idx


def graph_to_signed(G: nx.Graph, node_to_idx: dict) -> SignedGraph:
    """Convert a networkx Graph to a SignedGraph with all signs = +1."""
    edges = []
    for u, v in G.edges():
        ui, vi = node_to_idx[u], node_to_idx[v]
        if ui != vi:
            edges.append((min(ui, vi), max(ui, vi)))
    edges = np.array(sorted(set(edges)), dtype=np.int64)
    signs = np.ones(edges.shape[0], dtype=np.int64)
    return SignedGraph(edges=edges, signs=signs,
                        n_nodes=G.number_of_nodes())


def load_dataset(name: str) -> tuple[nx.Graph, np.ndarray]:
    """Load benchmark + class labels.  Returns (graph, labels)
    where labels are integer class IDs aligned with the sorted
    node order."""
    if name == "karate":
        G = nx.karate_club_graph()
        nodes = sorted(G.nodes())
        labels = np.array([
            0 if G.nodes[v]["club"] == "Mr. Hi" else 1
            for v in nodes
        ], dtype=np.int64)
    elif name == "florentine":
        G = nx.florentine_families_graph()
        nodes = sorted(G.nodes())
        # Medici-aligned families (per Padgett & Ansell 1993).
        medici_aligned = {
            "Medici", "Tornabuoni", "Salviati", "Albizzi",
            "Ridolfi", "Acciaiuoli", "Barbadori", "Ginori",
        }
        labels = np.array([
            0 if v in medici_aligned else 1 for v in nodes
        ], dtype=np.int64)
    elif name == "lesmis":
        G = nx.les_miserables_graph()
        # Use connected-components on a thresholded subgraph as a
        # proxy for "main protagonists vs side characters".
        nodes = sorted(G.nodes())
        # Top-half by degree → class 0, bottom half → class 1.
        deg = dict(G.degree())
        median = np.median(list(deg.values()))
        labels = np.array([
            0 if deg[v] > median else 1 for v in nodes
        ], dtype=np.int64)
    else:
        raise ValueError(f"unknown dataset: {name!r}")
    return G, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="karate",
                    choices=["karate", "florentine", "lesmis"])
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    G, labels = load_dataset(args.dataset)
    n = G.number_of_nodes()
    n_classes = int(labels.max() + 1)
    feats, node_to_idx = graph_features(G)
    feats_s = StandardScaler().fit_transform(feats)
    sg = graph_to_signed(G, node_to_idx)
    print(f"[{args.dataset}] n_nodes={n}, n_edges={sg.edges.shape[0]}, "
          f"classes={n_classes}, label_balance={labels.mean():.3f}")

    arities = (3, 4)
    per_arity_tuples, arities_used = build_per_arity(
        sg, arities, max_k=2000, seed=args.seed,
    )

    per_arity_inputs: list[tuple[torch.Tensor, ...]] = []
    for k_v, triad_v, triad_sigma in per_arity_tuples:
        triad_v_t = torch.from_numpy(triad_v).to(device)
        triad_sigma_t = torch.from_numpy(triad_sigma).to(device)
        M_vt = build_M_vt(triad_v, sg.n_nodes, device)
        # Dummy 1×T M_e (encoder requires it; node-class head ignores)
        rows = np.zeros(triad_v.shape[0], dtype=np.int64)
        cols = np.arange(triad_v.shape[0], dtype=np.int64)
        vals = np.ones(triad_v.shape[0], dtype=np.float32) / max(
            1, triad_v.shape[0]
        )
        idx_t = torch.tensor(np.stack([rows, cols]),
                              dtype=torch.long, device=device)
        v_t = torch.tensor(vals, dtype=torch.float32, device=device)
        M_e_dummy = torch.sparse_coo_tensor(
            idx_t, v_t, (1, triad_v.shape[0]),
        ).coalesce()
        per_arity_inputs.append(
            (triad_v_t, triad_sigma_t, M_vt, M_e_dummy)
        )

    feats_t = torch.tensor(feats_s, dtype=torch.float32, device=device)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    fold_size = max(1, n // args.n_folds)

    hsikan_accs = []
    lr_accs = []
    spec_acc = None
    t0 = time.time()
    for fold in range(args.n_folds):
        test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.array(
            [i for i in range(n) if i not in set(test_idx)],
            dtype=np.int64,
        )
        train_idx_t = torch.tensor(train_idx, device=device)
        test_idx_t = torch.tensor(test_idx, device=device)
        y_train = torch.tensor(labels[train_idx], dtype=torch.long,
                                device=device)
        y_test = labels[test_idx]

        cfg = MixedAritySignedKANConfig(
            base=MultiLayerSignedKANConfig(
                n_nodes=n, n_layers=2, hidden_dim=args.hidden,
                grid=3, k=3, spline_kinds=["catmull_rom"] * 2,
                init_scale=0.05, pool_mode="sum", jk_mode="concat",
                layer_norm_between=True, share_weights=True,
                inner_skip="highway", outer_skip="none",
                use_residual=True),
            arities=tuple(arities_used),
            init_arity_logits=tuple([0.0] * len(arities_used)),
            vertex_feat_dim=4,
        )
        model = MixedAritySignedKAN(cfg).to(device)
        head = nn.Linear(args.hidden, n_classes).to(device)
        opt = torch.optim.Adam(
            list(model.parameters()) + list(head.parameters()), lr=5e-3,
        )
        for ep in range(args.n_epochs):
            model.train(); head.train()
            _ = model.encode_edges(per_arity_inputs,
                                    vertex_features=feats_t)
            h_v = model._final_h_v
            logits = head(h_v[train_idx_t])
            loss = F.cross_entropy(logits, y_train)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); head.eval()
        with torch.no_grad():
            _ = model.encode_edges(per_arity_inputs,
                                    vertex_features=feats_t)
            h_v = model._final_h_v
            pred = head(h_v[test_idx_t]).argmax(-1).cpu().numpy()
        hsikan_accs.append(accuracy_score(y_test, pred))
        # LR baseline.
        lr = LogisticRegression(max_iter=1000).fit(
            feats_s[train_idx], labels[train_idx],
        )
        lr_accs.append(accuracy_score(y_test, lr.predict(feats_s[test_idx])))
        n_params = sum(p.numel() for p in
                        list(model.parameters()) +
                        list(head.parameters()))

    # Spectral clustering baseline (on full graph; report best matching).
    if n_classes == 2:
        adj = nx.adjacency_matrix(G).toarray()
        sc = SpectralClustering(
            n_clusters=2, affinity="precomputed",
            random_state=args.seed,
        ).fit(adj.astype(np.float64) + 1e-9)
        # Both label assignments → take best.
        spec_acc = max(
            accuracy_score(labels, sc.labels_),
            accuracy_score(labels, 1 - sc.labels_),
        )

    train_s = time.time() - t0
    out = dict(
        dataset=args.dataset, n_nodes=n,
        n_edges=int(sg.edges.shape[0]),
        n_classes=n_classes,
        hidden=args.hidden, n_folds=args.n_folds,
        hsikan_acc_mean=float(np.mean(hsikan_accs)),
        hsikan_acc_std=float(np.std(hsikan_accs)),
        lr_acc_mean=float(np.mean(lr_accs)),
        spectral_acc=spec_acc,
        n_params=n_params,
        train_s=train_s, seed=args.seed,
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
