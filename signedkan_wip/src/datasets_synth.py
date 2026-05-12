"""Small synthetic datasets for testing.

Wraps sklearn's `make_moons`, `make_circles`, `make_regression` and
converts each into a SignedGraph by building a k-NN graph in feature
space and signing edges by class agreement (or, for regression,
target-similarity above a threshold).

Used by `tests/test_synth_datasets.py` for CI-friendly, dataset-free
integration tests on the HymeKo pipeline. No external dataset
download required.

Public API:
    make_moon_signed_graph(n_samples, k_neighbors, seed, noise)
    make_circles_signed_graph(n_samples, k_neighbors, seed, noise)
    make_regression_signed_graph(n_samples, k_neighbors, seed)

All return a `SignedGraph` (defined in `signedkan_wip.src.datasets`).
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_circles, make_moons, make_regression
from sklearn.neighbors import NearestNeighbors

from .datasets import SignedGraph


def _knn_signed_graph(
    X: np.ndarray,
    edge_sign: np.ndarray,
    k_neighbors: int = 5,
) -> SignedGraph:
    """Build a k-NN graph from points X, signing edges via the
    pre-computed edge_sign(u, v) callable's array form.

    Args:
        X: (n, d) features (used to determine k-NN structure).
        edge_sign: (n, n) signed adjacency callable's matrix form,
            with edge_sign[u, v] ∈ {-1, +1}.  Sparse usage: only
            entries at k-NN pairs are read.
        k_neighbors: number of neighbours per vertex.

    Returns:
        SignedGraph with (n_nodes=n, edges_arr, signs_arr).
    """
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)  # +1 for self
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    edges = []
    signs = []
    seen: set[tuple[int, int]] = set()
    for u in range(n):
        for j in range(1, k_neighbors + 1):           # skip self at j=0
            v = int(indices[u, j])
            pair = (min(u, v), max(u, v))
            if pair in seen:
                continue
            seen.add(pair)
            s = int(edge_sign[u, v])
            if s == 0:
                continue
            edges.append((u, v))
            signs.append(s)
    return SignedGraph(
        edges=np.array(edges, dtype=np.int64),
        signs=np.array(signs, dtype=np.int8),
        n_nodes=n,
    )


def _class_agreement_signs(y: np.ndarray) -> np.ndarray:
    """edge_sign[u, v] = +1 if y[u] == y[v] else -1, shape (n, n)."""
    n = y.shape[0]
    same = (y[:, None] == y[None, :])
    return np.where(same, 1, -1).astype(np.int8)


def _target_similarity_signs(
    y: np.ndarray, threshold_factor: float = 0.25,
) -> np.ndarray:
    """For regression targets, edge_sign[u, v] = +1 if |y_u - y_v| is
    below a percentile threshold, -1 otherwise.

    threshold_factor: fraction of total target range; e.g. 0.25 means
    pairs within 25% of the range are 'similar'.
    """
    diff = np.abs(y[:, None] - y[None, :])
    threshold = threshold_factor * (y.max() - y.min())
    return np.where(diff <= threshold, 1, -1).astype(np.int8)


def make_moon_signed_graph(
    n_samples: int = 200,
    k_neighbors: int = 5,
    seed: int = 0,
    noise: float = 0.1,
) -> tuple[SignedGraph, np.ndarray, np.ndarray]:
    """Two interleaving half-moons; edge sign by class agreement.

    Returns:
        (graph, X, y) where X is the 2D feature matrix and y is the
        binary class label. graph has |V| = n_samples and ~k_neighbors
        * n_samples / 2 unique edges.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    edge_sign = _class_agreement_signs(y)
    return _knn_signed_graph(X, edge_sign, k_neighbors=k_neighbors), X, y


def make_circles_signed_graph(
    n_samples: int = 200,
    k_neighbors: int = 5,
    seed: int = 0,
    noise: float = 0.05,
    factor: float = 0.5,
) -> tuple[SignedGraph, np.ndarray, np.ndarray]:
    """Two concentric circles; edge sign by class agreement.

    factor: inner-to-outer radius ratio.
    """
    X, y = make_circles(
        n_samples=n_samples, noise=noise, factor=factor, random_state=seed,
    )
    edge_sign = _class_agreement_signs(y)
    return _knn_signed_graph(X, edge_sign, k_neighbors=k_neighbors), X, y


def make_regression_signed_graph(
    n_samples: int = 200,
    n_features: int = 10,
    k_neighbors: int = 5,
    seed: int = 0,
    noise: float = 0.5,
    threshold_factor: float = 0.25,
) -> tuple[SignedGraph, np.ndarray, np.ndarray]:
    """Linear regression problem; edge sign by target-similarity.

    threshold_factor: pairs with |y_u - y_v| <= threshold_factor *
    (y.max() - y.min()) are +1, else -1.
    """
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features, noise=noise,
        random_state=seed,
    )
    edge_sign = _target_similarity_signs(y, threshold_factor=threshold_factor)
    return _knn_signed_graph(X, edge_sign, k_neighbors=k_neighbors), X, y


__all__ = [
    "make_moon_signed_graph",
    "make_circles_signed_graph",
    "make_regression_signed_graph",
]
