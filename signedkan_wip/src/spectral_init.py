"""Signed-Laplacian spectral initialisation for HSiKAN node embeddings.

Cartwright-Harary 1956: a signed graph admits a "k-balanced" partition
(vertices partition into ≤k clusters with positive intra-cluster and
negative inter-cluster edges) iff the *signed Laplacian* has a
non-trivial null space of dimension ≥ k - 1.

For nearly-balanced graphs (Slashdot at 77.4 % positive edges sits in
the "intermediate-balance" regime) the small-eigenvalue eigenvectors
of the signed Laplacian recover an *approximate* signed-community
membership per vertex. Using them to initialise the SignedKAN node
embedding gives the layer a head start on community structure rather
than learning it from scratch.

Construction
------------
    A_signed[u, v] = sign(edge u-v)             (if (u,v) is an edge,
                                                   else 0)
    D[v, v]        = Σ_u |A_signed[v, u]|       (unsigned degree)
    L_signed       = D - A_signed                (signed combinatorial
                                                   Laplacian, Kunegis
                                                   et al. 2010)

For a balanced graph, L_signed has a 1-D null space spanned by the
indicator vector of one of the two balanced communities (signed
indicator). For nearly-balanced graphs, the smallest k eigenvalues
encode increasingly fine community partitions.

We return the smallest-d eigenvectors stacked column-wise, scaled so
each column has the same RMS as the random init that would otherwise
be used (cfg.init_scale = 0.05 by default).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh

from .datasets import SignedGraph


def _build_signed_adjacency(g: SignedGraph) -> csr_matrix:
    """Symmetric signed adjacency. A[u, v] = +1 / -1 / 0; A[v, u] = A[u, v]."""
    n = g.n_nodes
    rows = np.concatenate([g.edges[:, 0], g.edges[:, 1]])
    cols = np.concatenate([g.edges[:, 1], g.edges[:, 0]])
    vals = np.concatenate([g.signs, g.signs]).astype(np.float64)
    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # Coalesce duplicate (u, v) pairs by summing then clipping to ±1.
    # Most signed-graph datasets have no duplicates, but be defensive.
    A.sum_duplicates()
    A.data = np.sign(A.data)
    return A


def signed_laplacian(g: SignedGraph) -> csr_matrix:
    """L = D - A_signed where D uses unsigned degrees |A_signed|."""
    A = _build_signed_adjacency(g)
    abs_deg = np.asarray(np.abs(A).sum(axis=1)).ravel()
    D = diags(abs_deg, format="csr")
    return (D - A).tocsr()


def compute_spectral_init(
    g: SignedGraph,
    hidden_dim: int,
    init_scale: float = 0.05,
    drop_first: bool = True,
    tol: float = 1e-3,
    maxiter: int = 200,
    seed: int = 0,
) -> torch.Tensor:
    """Compute (n_nodes, hidden_dim) tensor from the d smallest eigvecs
    of the signed Laplacian.

    Uses **LOBPCG** (Locally Optimal Block Preconditioned Conjugate
    Gradient) rather than ARPACK shift-invert: signed Laplacian
    factorisation at sigma=0 is too expensive on real-graph sizes
    (Slashdot 82k nodes ≫ 1h with ``eigsh(sigma=0)``). LOBPCG only
    needs sparse-matvec and converges in a few hundred iterations
    for the smallest eigvals of a PSD matrix.

    The signed Laplacian L = D - A_sign with D = Σ|A| is provably PSD
    (Kunegis 2010), so smallest eigenvalues are non-negative.

    ``drop_first`` skips the (approximately) trivial null-space
    eigenvector that exists when the unsigned graph is connected
    (analogous to the constant vector for the unsigned Laplacian).

    Output columns are scaled to RMS ≈ ``init_scale`` to match the
    variance of the default ``nn.init.normal_(std=init_scale)``.
    """
    from scipy.sparse.linalg import lobpcg
    L = signed_laplacian(g)
    n = L.shape[0]
    k = hidden_dim + (1 if drop_first else 0)

    rng = np.random.RandomState(seed)
    X0 = rng.randn(n, k).astype(np.float64)
    # Orthonormalise the initial block (LOBPCG requires orthogonal X).
    X0, _ = np.linalg.qr(X0)

    eigvals, eigvecs = lobpcg(
        L, X0,
        largest=False,
        tol=tol,
        maxiter=maxiter,
        verbosityLevel=0,
    )

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if drop_first:
        eigvecs = eigvecs[:, 1:]
    eigvecs = eigvecs[:, :hidden_dim]

    rms = np.sqrt((eigvecs ** 2).mean(axis=0, keepdims=True) + 1e-12)
    eigvecs = eigvecs * (init_scale / rms)
    return torch.from_numpy(eigvecs.astype(np.float32))
