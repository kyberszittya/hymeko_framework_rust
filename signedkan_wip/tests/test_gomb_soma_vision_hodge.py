"""Tests for HodgeLaplacian (Ricci-Stim phase 3).

Pins the differential-geometric identities:

  * the fundamental identity \\(\\partial_1 \\partial_2 = 0\\) — boundary
    of boundary is zero (pinned exactly);
  * \\(\\Delta_0\\) equals the standard graph Laplacian \\(D - A\\);
  * Hodge Laplacians are symmetric positive-semidefinite;
  * \\(\\Delta_0\\) eigenspectrum has a 0-eigenvector for each
    connected component (Betti-0 count);
  * harmonic / gradient / curl reconstruction of an edge 1-form on
    a small triangle (Hodge decomposition);
  * canonical orientation: \\(\\partial_1 (u, v)\\) = \\([v] - [u]\\) for u < v.
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma.vision import (
    HodgeLaplacian,
    HodgeOperators,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _triangle_edges_and_face() -> tuple[torch.Tensor, torch.Tensor]:
    edges = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.long)
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.long)
    return edges, triangles


def _path_P3_edges() -> torch.Tensor:
    """Path on 3 vertices: 0 -- 1 -- 2 (no triangle)."""
    return torch.tensor([[0, 1], [1, 2]], dtype=torch.long)


def _square_edges() -> torch.Tensor:
    return torch.tensor(
        [[0, 1], [1, 2], [2, 3], [0, 3]], dtype=torch.long,
    )


def _Kn_edges(n: int) -> torch.Tensor:
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append((i, j))
    return torch.tensor(rows, dtype=torch.long)


# ---------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------


def test_partial_1_canonical_orientation():
    """For canonical edge (u, v) with u < v:
    \\(\\partial_1 e = +[v] - [u]\\).
    Single edge (0, 1): row 0 has -1, row 1 has +1, col 0."""
    h = HodgeLaplacian()
    out = h(torch.tensor([[0, 1]], dtype=torch.long), n_vertices=2)
    dense = out.boundary_1.to_dense()
    expected = torch.tensor([[-1.0], [+1.0]])
    assert torch.equal(dense, expected)


def test_partial_2_triangle_alternating_signs():
    """Triangle [0,1,2]: \\(\\partial_2 = +[1,2] - [0,2] + [0,1]\\)."""
    edges, triangles = _triangle_edges_and_face()
    h = HodgeLaplacian()
    out = h(edges, n_vertices=3, triangles=triangles)
    dense = out.boundary_2.to_dense()
    # edges (canonical, sorted by lex): (0,1)=0, (0,2)=1, (1,2)=2
    # Expected column for triangle 0:
    #   (0,1): +1,  (0,2): -1,  (1,2): +1
    expected = torch.tensor([[+1.0], [-1.0], [+1.0]])
    assert torch.equal(dense, expected)


def test_partial_partial_is_zero_on_triangle():
    """The fundamental identity: \\(\\partial_1 \\partial_2 = 0\\). Pinned exactly."""
    edges, triangles = _triangle_edges_and_face()
    h = HodgeLaplacian()
    out = h(edges, n_vertices=3, triangles=triangles)
    composition = torch.sparse.mm(out.boundary_1, out.boundary_2).to_dense()
    expected = torch.zeros(3, 1)
    assert torch.equal(composition, expected), (
        f"\\partial_1 \\partial_2 != 0; got {composition.tolist()}"
    )


def test_partial_partial_is_zero_on_K4():
    """Same fundamental identity, but on K_4 (4 vertices, 6 edges, 4
    triangles). Pinned exactly."""
    edges = _Kn_edges(4)
    triangles = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long,
    )
    h = HodgeLaplacian()
    out = h(edges, n_vertices=4, triangles=triangles)
    composition = torch.sparse.mm(out.boundary_1, out.boundary_2).to_dense()
    expected = torch.zeros(4, 4)
    assert torch.equal(composition, expected)


def test_self_loops_dropped():
    """Self-loop (u, u) entries silently ignored."""
    edges = torch.tensor([[0, 1], [1, 1], [1, 2]], dtype=torch.long)
    h = HodgeLaplacian()
    out = h(edges, n_vertices=3)
    # Only edges (0,1) and (1,2) survive.
    assert out.n_edges == 2


def test_degenerate_triangles_dropped():
    """Triangles with a repeated vertex are silently ignored."""
    edges = torch.tensor([[0, 1]], dtype=torch.long)
    triangles = torch.tensor(
        [[0, 0, 1]], dtype=torch.long,  # degenerate (0 == 0)
    )
    h = HodgeLaplacian()
    out = h(edges, n_vertices=2, triangles=triangles)
    assert out.n_triangles == 0


# ---------------------------------------------------------------------
# Laplacian tests
# ---------------------------------------------------------------------


def test_laplacian_0_equals_D_minus_A_on_triangle():
    """\\(\\Delta_0\\) on K_3: each vertex has deg 2, two off-diagonal -1s
    per row. Expected $\\Delta_0 = 2 I - A$ where A is all-1 off-diag."""
    edges, _ = _triangle_edges_and_face()
    h = HodgeLaplacian()
    out = h(edges, n_vertices=3)
    L = out.laplacian_0.to_dense()
    expected = torch.tensor([
        [+2.0, -1.0, -1.0],
        [-1.0, +2.0, -1.0],
        [-1.0, -1.0, +2.0],
    ])
    assert torch.equal(L, expected)


def test_laplacian_0_equals_D_minus_A_on_path():
    """Path P_3: deg = [1, 2, 1], A has 1s only at adjacencies."""
    h = HodgeLaplacian()
    out = h(_path_P3_edges(), n_vertices=3)
    L = out.laplacian_0.to_dense()
    expected = torch.tensor([
        [+1.0, -1.0,  0.0],
        [-1.0, +2.0, -1.0],
        [ 0.0, -1.0, +1.0],
    ])
    assert torch.equal(L, expected)


def test_laplacian_0_eigenvalue_zero_for_each_component():
    """For a disconnected graph with c components, \\(\\Delta_0\\) has c
    zero eigenvalues."""
    # Two triangles, disconnected: vertices {0,1,2} and {3,4,5}.
    edges = torch.tensor(
        [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]],
        dtype=torch.long,
    )
    h = HodgeLaplacian()
    out = h(edges, n_vertices=6)
    L = out.laplacian_0.to_dense()
    eig = torch.linalg.eigvalsh(L)
    n_zero = int((eig.abs() < 1e-6).sum().item())
    assert n_zero == 2, f"expected 2 zero eigenvalues, got {n_zero}"


def test_laplacian_0_symmetric():
    h = HodgeLaplacian()
    out = h(_square_edges(), n_vertices=4)
    L = out.laplacian_0.to_dense()
    assert torch.equal(L, L.t()), "L_0 must be symmetric"


def test_laplacian_0_positive_semidefinite():
    h = HodgeLaplacian()
    out = h(_square_edges(), n_vertices=4)
    L = out.laplacian_0.to_dense()
    eig = torch.linalg.eigvalsh(L)
    assert (eig >= -1e-6).all(), (
        f"L_0 must be PSD; got eigenvalues {eig.tolist()}"
    )


def test_laplacian_1_includes_triangle_term_when_triangles_present():
    """\\(\\Delta_1 = \\partial_1^T \\partial_1 + \\partial_2 \\partial_2^T\\) when there
    are triangles. With NO triangles, only the first term."""
    edges = _Kn_edges(3)  # K_3 edges
    h = HodgeLaplacian()
    # Without triangles
    out_no_tri = h(edges, n_vertices=3)
    # With triangle
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.long)
    out_tri = h(edges, n_vertices=3, triangles=triangles)
    L1_no = out_no_tri.laplacian_1.to_dense()
    L1_with = out_tri.laplacian_1.to_dense()
    diff = L1_with - L1_no
    # diff must equal \\(\\partial_2 \\partial_2^T\\); rank-1 contribution
    # of a single triangle.
    b2 = out_tri.boundary_2.to_dense()
    expected_diff = b2 @ b2.t()
    assert torch.equal(diff, expected_diff)


def test_hodge_decomposition_reconstruction_on_triangle():
    """For an edge 1-form \\(\\omega\\), the Hodge decomposition states
    \\(\\omega = h + \\partial_1^* \\alpha + \\partial_2 \\beta\\) with h harmonic
    (in ker \\(\\Delta_1\\)). We verify reconstruction: if we project
    \\(\\omega\\) onto image(\\(\\partial_1^*\\)) ⊕ image(\\(\\partial_2\\)) ⊕ ker(\\(\\Delta_1\\)),
    we recover \\(\\omega\\)."""
    edges, triangles = _triangle_edges_and_face()
    h = HodgeLaplacian()
    out = h(edges, n_vertices=3, triangles=triangles)

    # Random 1-form (one value per edge).
    torch.manual_seed(0)
    omega = torch.randn(3)

    b1 = out.boundary_1.to_dense()
    b2 = out.boundary_2.to_dense()

    # Project onto image of \\partial_1^T (rank = rank(b1)).
    # SVD with full_matrices=False returns U of width min(m,n); we
    # need to slice to columns with non-zero singular values to get
    # the proper image-of-b1^T projector.
    tol = 1e-5
    U_grad, S_grad, _ = torch.linalg.svd(b1.t(), full_matrices=False)
    rank_grad = int((S_grad > tol).sum().item())
    U_grad = U_grad[:, :rank_grad]
    P_grad = U_grad @ U_grad.t()
    # Project onto image of \\partial_2 (col-space of b2).
    U_curl, S_curl, _ = torch.linalg.svd(b2, full_matrices=False)
    rank_curl = int((S_curl > tol).sum().item())
    U_curl = U_curl[:, :rank_curl]
    P_curl = U_curl @ U_curl.t()
    # Harmonic part: orthogonal complement of grad ⊕ curl.
    I = torch.eye(3)
    P_harm = I - P_grad - P_curl

    omega_grad = P_grad @ omega
    omega_curl = P_curl @ omega
    omega_harm = P_harm @ omega
    reconstructed = omega_grad + omega_curl + omega_harm
    assert torch.allclose(reconstructed, omega, atol=1e-5), (
        f"Hodge decomposition reconstruction failed; max diff = "
        f"{(reconstructed - omega).abs().max().item():.2e}"
    )
    # Harmonic part should be in kernel of \\Delta_1.
    L1 = out.laplacian_1.to_dense()
    residual = L1 @ omega_harm
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


# ---------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------


def test_output_is_HodgeOperators():
    h = HodgeLaplacian()
    edges, triangles = _triangle_edges_and_face()
    out = h(edges, n_vertices=3, triangles=triangles)
    assert isinstance(out, HodgeOperators)
    for attr in ("boundary_1", "boundary_2",
                 "laplacian_0", "laplacian_1", "laplacian_2"):
        t = getattr(out, attr)
        assert t.is_sparse, f"{attr} should be sparse"


def test_no_learnable_parameters():
    h = HodgeLaplacian()
    assert sum(p.numel() for p in h.parameters()) == 0


def test_rejects_wrong_edge_shape():
    h = HodgeLaplacian()
    with pytest.raises(ValueError, match="edges must have shape"):
        h(torch.zeros((5,), dtype=torch.long), n_vertices=3)


def test_rejects_wrong_triangle_shape():
    h = HodgeLaplacian()
    with pytest.raises(ValueError, match="triangles must have shape"):
        h(torch.zeros((1, 2), dtype=torch.long), n_vertices=3,
          triangles=torch.zeros((1, 2), dtype=torch.long))


def test_empty_input():
    h = HodgeLaplacian()
    out = h(torch.zeros((0, 2), dtype=torch.long), n_vertices=4)
    assert out.n_edges == 0
    assert out.boundary_1.shape == (4, 0)


# ---------------------------------------------------------------------
# Vectorised boundary_2 + edges_already_canonical flag — regression
# pin for the 2026-05-16 Hodge optimisation pass.
# ---------------------------------------------------------------------


def _random_complex(n_v: int = 12, p_edge: float = 0.45,
                    p_face: float = 0.6, seed: int = 0
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """A deterministic 'random' simplicial complex: pick edges by
    Bernoulli; promote each 3-clique to a triangle with probability
    p_face."""
    gen = torch.Generator().manual_seed(seed)
    edges = []
    for u in range(n_v):
        for v in range(u + 1, n_v):
            if torch.rand((), generator=gen).item() < p_edge:
                edges.append([u, v])
    e_t = torch.tensor(edges, dtype=torch.long) if edges else torch.zeros(
        (0, 2), dtype=torch.long
    )
    adj = {i: set() for i in range(n_v)}
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    tris = []
    for a in range(n_v):
        for b in sorted(adj[a]):
            if b <= a:
                continue
            for c in sorted(adj[a] & adj[b]):
                if c <= b:
                    continue
                if torch.rand((), generator=gen).item() < p_face:
                    tris.append([a, b, c])
    t_t = torch.tensor(tris, dtype=torch.long) if tris else torch.zeros(
        (0, 3), dtype=torch.long
    )
    return e_t, t_t


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_hodge_canonical_flag_invariance(seed: int) -> None:
    """Setting ``edges_already_canonical=True`` on pre-canonicalised
    edges must yield byte-identical Hodge operators.

    Verifies that the optimisation hint added 2026-05-16 does not
    silently change the linear operators in any
    \\(\\Delta_k\\)."""
    h = HodgeLaplacian()
    edges, tris = _random_complex(n_v=12, p_edge=0.45, p_face=0.6, seed=seed)
    # Force canonical: sort rows + unique. _random_complex already
    # produces canonical edges, but make it explicit.
    if edges.shape[0]:
        edges, _ = torch.sort(edges, dim=1)
        edges = torch.unique(edges, dim=0)

    out_default = h(edges, n_vertices=12, triangles=tris)
    out_flag = h(edges, n_vertices=12, triangles=tris,
                 edges_already_canonical=True)

    for attr in ("boundary_1", "boundary_2",
                 "laplacian_0", "laplacian_1", "laplacian_2"):
        a = getattr(out_default, attr).to_dense()
        b = getattr(out_flag, attr).to_dense()
        assert a.shape == b.shape, (
            f"{attr} shape mismatch under the canonical flag: "
            f"{tuple(a.shape)} vs {tuple(b.shape)}"
        )
        diff = (a - b).abs().max().item()
        assert diff < 1e-12, (
            f"{attr} value mismatch under canonical flag at seed={seed}: "
            f"max abs diff = {diff}"
        )


def test_hodge_boundary_2_vectorised_drops_missing_edges() -> None:
    """The vectorised boundary_2 must silently drop boundary entries
    pointing at edges not in the canonical edge set, matching the
    original Python-loop behaviour."""
    h = HodgeLaplacian()
    # Triangle (0,1,2) but edge (0,2) is missing.
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.long)
    out = h(edges, n_vertices=3, triangles=triangles)
    b2 = out.boundary_2.to_dense()
    assert b2.shape == (2, 1)
    # boundary should have entries only for present edges: +[1,2] = +1, +[0,1] = +1.
    # Edge 0 is [0,1] (idx 0), edge 1 is [1,2] (idx 1).
    # +[1,2] at e_idx 1 → b2[1,0] = +1; +[0,1] at e_idx 0 → b2[0,0] = +1;
    # -[0,2] missing → no entry.
    assert b2[1, 0].item() == pytest.approx(1.0)
    assert b2[0, 0].item() == pytest.approx(1.0)


def test_hodge_boundary_2_vectorised_partial_d_squared_zero() -> None:
    """\\(\\partial_1 \\partial_2 = 0\\) must hold under the
    vectorised boundary_2 path."""
    h = HodgeLaplacian()
    edges, tris = _random_complex(n_v=10, p_edge=0.5, p_face=0.8, seed=7)
    if edges.shape[0]:
        edges, _ = torch.sort(edges, dim=1)
        edges = torch.unique(edges, dim=0)
    out = h(edges, n_vertices=10, triangles=tris,
            edges_already_canonical=True)
    if out.n_triangles == 0:
        pytest.skip("random complex had no triangles at this seed")
    composed = torch.sparse.mm(out.boundary_1, out.boundary_2).to_dense()
    assert composed.abs().max().item() < 1e-10, (
        "boundary-of-boundary is not zero under vectorised path"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
