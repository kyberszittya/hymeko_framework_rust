"""Signed-graph Laplacian eigenvectors as a structural prior for
node-embedding initialisation.

For a signed graph $G = (V, E, \\sigma)$ with $\\sigma : E \\to \\{+1, -1\\}$,
construct the *symmetric signed adjacency*
$\\mathbf{A}_s \\in \\mathbb{R}^{|V| \\times |V|}$ with
$\\mathbf{A}_s[u, v] = \\sigma_{uv}$ for $\\{u,v\\} \\in E$, and the
absolute-degree diagonal $\\mathbf{D}_s = \\mathrm{diag}(\\sum_v |\\mathbf{A}_s[u,v]|)$.
The symmetric normalised signed Laplacian is

    L_s = I - D_s^{-1/2} A_s D_s^{-1/2}

whose smallest eigenvalues correspond to "balanced" cluster
structure (Cartwright-Harary balance is a low-frequency mode of
$L_s$). The corresponding eigenvectors are interpretable structural
features: they place vertices in a coordinate frame where balanced
triads sit close together and unbalanced ones spread apart.

Used as an initialisation prior for ``node_embed`` (Phase 4.x):
the first $k$ dimensions are seeded with the top-$k$ smallest
eigenvectors, the remaining $d - k$ dimensions are random
$\\mathcal{N}(0, \\sigma^2)$ noise.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def signed_normalised_laplacian(edges: np.ndarray, signs: np.ndarray,
                                n_nodes: int) -> sp.csr_matrix:
    """Symmetric normalised signed Laplacian, $L_s = I - D_s^{-1/2} A_s D_s^{-1/2}$.

    edges : (E, 2) src, dst
    signs : (E,) +1 / -1
    Returns a sparse $(n, n)$ csr_matrix.
    """
    # Build symmetric signed adjacency.
    rows = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    cols = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    vals = np.concatenate([signs, signs]).astype(np.float64)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    # Coalesce duplicates by summing (rare for canonical Bitcoin / SNAP files).
    A.sum_duplicates()
    # Absolute-degree.
    deg = np.asarray(np.abs(A).sum(axis=1)).reshape(-1)
    deg = np.maximum(deg, 1e-8)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(n_nodes) - D_inv_sqrt @ A @ D_inv_sqrt


def top_k_eigenvectors(L: sp.csr_matrix, k: int) -> np.ndarray:
    """Smallest-magnitude k eigenvectors of $L_s$.

    Returns a $(n, k)$ numpy array, columns sorted by eigenvalue.
    For small graphs (n < 5000) we densify and use ``np.linalg.eigh``;
    for larger graphs we use ``scipy.sparse.linalg.eigsh`` in
    shift-invert mode.
    """
    n = L.shape[0]
    if n < 5000:
        L_dense = L.toarray()
        # Symmetric: real eigenvalues in ascending order.
        vals, vecs = np.linalg.eigh(L_dense)
        return vecs[:, :k]
    else:
        # Shift-invert near sigma=0 to find smallest eigenvalues robustly.
        try:
            vals, vecs = spla.eigsh(L, k=k, sigma=0.0, which="LM")
        except Exception:
            # Fallback: 'SM' direct, slower but no shift-invert.
            vals, vecs = spla.eigsh(L, k=k, which="SM")
        # Sort ascending.
        order = np.argsort(vals)
        return vecs[:, order]


def make_spectral_init(edges: np.ndarray, signs: np.ndarray,
                       n_nodes: int, hidden_dim: int,
                       k: int, noise_scale: float = 0.1) -> np.ndarray:
    """Build a $(n, d)$ initial node-embedding matrix where the first
    $k$ dimensions carry top-$k$ signed-Laplacian eigenvectors and
    the remaining $d - k$ dimensions are $\\mathcal{N}(0, \\sigma^2)$."""
    if k > hidden_dim:
        raise ValueError(f"spectral_k={k} > hidden_dim={hidden_dim}")
    L = signed_normalised_laplacian(edges, signs, n_nodes)
    eigvecs = top_k_eigenvectors(L, k)               # (n, k)
    # Per-column normalise to unit variance — the splines clamp at +-1
    # and want input in that range; the eigenvectors come out at ~1/sqrt(n)
    # scale otherwise.
    eigvecs = eigvecs / (eigvecs.std(axis=0, keepdims=True) + 1e-8)
    eigvecs = eigvecs * noise_scale                  # match init_scale
    init = np.zeros((n_nodes, hidden_dim), dtype=np.float32)
    init[:, :k] = eigvecs
    if hidden_dim > k:
        init[:, k:] = np.random.randn(n_nodes, hidden_dim - k) * noise_scale
    return init
