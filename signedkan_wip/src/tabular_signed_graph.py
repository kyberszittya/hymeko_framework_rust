"""Build signed graphs from tabular feature matrices.

Three protocols (see `docs/plans_hsikan_tabular_benchmarks_2026_05_09.md`):

- P1 *(supervised)*: k-NN graph over standardised features; edge sign =
  +1 iff endpoints share a class label, else −1.  Encodes class
  structure into the graph signs.

- P2 *(unsupervised)*: k-NN graph over standardised features; edge sign =
  sgn(cosine of centred features).  Sign encodes feature-similarity
  direction without needing labels.

- P3 *(unsupervised, bipartite)*: bipartite (sample × feature) graph;
  each sample connects to features whose value exceeds a threshold,
  signed by sgn(x_{i,f} − μ_f).  Bipartite means no closed odd cycles.

- P_unsigned: k-NN graph over standardised features; **all signs = +1**.
  Truly unsigned / general-graph test — measures whether HSiKAN's
  signed-cycle bias adds anything over a "graph-KAN" without sign
  structure.  Used to ablate the σ-masked branches.

The output is a `SignedGraph` consumable by `n_tuples.construct_*`
and the rest of the HSiKAN pipeline.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .datasets import SignedGraph


def build_signed_graph_from_tabular(
    X: np.ndarray,
    y: np.ndarray | None = None,
    k: int = 5,
    protocol: str = "p1",
) -> SignedGraph:
    """Construct a signed graph from a tabular matrix X (n, d).

    Parameters
    ----------
    X        : (n, d) feature matrix
    y        : (n,) class labels (required for P1)
    k        : k for k-NN graph
    protocol : "p1" (k-NN + class-sign), "p2" (k-NN + correlation-sign)
    """
    n, d = X.shape
    Xs = StandardScaler().fit_transform(X)

    if protocol == "p_unsigned":
        # k-NN graph with all signs = +1 (unsigned baseline).
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
        _, idx = nbrs.kneighbors(Xs)
        neighbours = idx[:, 1:]
        edge_set: set[tuple[int, int]] = set()
        for i in range(n):
            for j in neighbours[i]:
                u, v = int(min(i, j)), int(max(i, j))
                if u != v:
                    edge_set.add((u, v))
        edges = np.array(sorted(edge_set), dtype=np.int64)
        signs = np.ones(edges.shape[0], dtype=np.int64)
        return SignedGraph(edges=edges, signs=signs, n_nodes=n)

    if protocol in ("p1", "p2"):
        # k-NN graph (excluding self).
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
        _, idx = nbrs.kneighbors(Xs)
        # idx[i, 0] is i itself; idx[i, 1:] are the k neighbours.
        neighbours = idx[:, 1:]
        edge_set: set[tuple[int, int]] = set()
        for i in range(n):
            for j in neighbours[i]:
                u, v = int(min(i, j)), int(max(i, j))
                if u != v:
                    edge_set.add((u, v))
        edges = np.array(sorted(edge_set), dtype=np.int64)
        signs = np.empty(edges.shape[0], dtype=np.int64)
        if protocol == "p1":
            if y is None:
                raise ValueError("P1 requires class labels y")
            for ei, (u, v) in enumerate(edges):
                signs[ei] = 1 if y[u] == y[v] else -1
        else:  # p2
            mu = Xs.mean(axis=0, keepdims=True)
            for ei, (u, v) in enumerate(edges):
                a = Xs[u] - mu[0]
                b = Xs[v] - mu[0]
                cos = float(a @ b) / (np.linalg.norm(a) *
                                       np.linalg.norm(b) + 1e-12)
                signs[ei] = 1 if cos > 0 else -1
        return SignedGraph(edges=edges, signs=signs, n_nodes=n)

    if protocol == "p3":
        # Bipartite: sample → feature.  Vertex IDs: samples 0..n-1,
        # features n..n+d-1.  Edge from sample i to feature f iff
        # |Xs[i, f]| > 0.5; sign = sgn(Xs[i, f]).
        thr = 0.5
        edges_list: list[tuple[int, int]] = []
        signs_list: list[int] = []
        for i in range(n):
            for f in range(d):
                if abs(Xs[i, f]) > thr:
                    edges_list.append((i, n + f))
                    signs_list.append(1 if Xs[i, f] > 0 else -1)
        edges = np.array(edges_list, dtype=np.int64)
        signs = np.array(signs_list, dtype=np.int64)
        return SignedGraph(edges=edges, signs=signs, n_nodes=n + d)

    raise ValueError(f"unknown protocol: {protocol!r}")
