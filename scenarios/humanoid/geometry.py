r"""Differential geometry of the mechanical metric M(q) — curvature for conditioning + entropy diagnostics.

The kinetic-energy matrix ``M(q)`` is a Riemannian metric on configuration space (geometric mechanics): free
motion is geodesic, and the metric's curvature governs geodesic (de)focusing — the conditioning of trajectory
optimisation and the dynamical sensitivity. From a symbolic metric ``g = M(q)`` this computes

    Γ^k_ij            Levi-Civita connection
    R^l_ijk, Ric_ij   Riemann / Ricci
    R (scalar), K     scalar / Gauss curvature

plus the **Bakry–Émery Ricci** ``Ric(g) + ∇²V`` of an energy-shaped closed loop, whose smallest eigenvalue
lower-bounds the entropy-relaxation (log-Sobolev) rate — the bridge that ties the shaping gain to both the
basin geometry and the thermodynamic mixing rate. A numeric variant differentiates ANY metric callable
(e.g. MuJoCo ``mj_fullM``) so the full humanoid's scalar curvature is available where the closed form is not.

# Preconditions: ``g`` symmetric positive-definite over the domain of interest; ``coords`` the matching symbols.
# Postconditions: ``scalar_curvature == 2·gauss_curvature`` in 2-D; a flat metric has zero curvature.
"""

from __future__ import annotations

import numpy as np
import sympy as sp


class RiemannianMetric:
    """A symbolic Riemannian metric ``g(q)`` with its Levi-Civita curvature invariants (cached)."""

    def __init__(self, g: sp.Matrix, coords: "list[sp.Symbol]") -> None:
        n = g.shape[0]
        if g.shape != (n, n):
            raise ValueError(f"metric must be square, got {g.shape}")
        if len(coords) != n:
            raise ValueError(f"coords length {len(coords)} must match metric dimension {n}")
        self.g, self.q, self.n = g, list(coords), n
        self._ginv: sp.Matrix | None = None
        self._gamma: "list | None" = None

    @property
    def inverse(self) -> sp.Matrix:
        if self._ginv is None:
            self._ginv = self.g.inv()
        return self._ginv

    def christoffel(self) -> list:
        r"""Γ^k_ij = ½ g^{kl}(∂_i g_jl + ∂_j g_il − ∂_l g_ij), as nested lists ``G[k][i][j]``."""
        if self._gamma is not None:
            return self._gamma
        n, g, gi, q = self.n, self.g, self.inverse, self.q
        gamma = [[[sp.Rational(1, 2) * sum(
            gi[k, s] * (sp.diff(g[j, s], q[i]) + sp.diff(g[i, s], q[j]) - sp.diff(g[i, j], q[s]))
            for s in range(n)) for j in range(n)] for i in range(n)] for k in range(n)]
        self._gamma = gamma
        return gamma

    def riemann_up(self, up: int, i: int, j: int, k: int) -> sp.Expr:
        r"""R^up_{ijk} = ∂_j Γ^up_ik − ∂_k Γ^up_ij + Σ_m (Γ^up_jm Γ^m_ik − Γ^up_km Γ^m_ij) (``up`` = raised index)."""
        gamma, q, n = self.christoffel(), self.q, self.n
        return (sp.diff(gamma[up][i][k], q[j]) - sp.diff(gamma[up][i][j], q[k])
                + sum(gamma[up][j][m] * gamma[m][i][k] - gamma[up][k][m] * gamma[m][i][j] for m in range(n)))

    def ricci(self) -> sp.Matrix:
        r"""Ric_ik = Σ_l R^l_{ilk}."""
        n = self.n
        return sp.Matrix(n, n, lambda i, k: sp.simplify(sum(self.riemann_up(s, i, s, k) for s in range(n))))

    def scalar_curvature(self, simplify: bool = True) -> sp.Expr:
        r"""R = g^{ik} Ric_ik. In 2-D this equals ``2·gauss_curvature``."""
        ric, gi, n = self.ricci(), self.inverse, self.n
        r = sum(gi[i, k] * ric[i, k] for i in range(n) for k in range(n))
        return sp.simplify(r) if simplify else r

    def gauss_curvature(self, simplify: bool = True) -> sp.Expr:
        r"""Gauss curvature ``K = R_{0101}/det g`` (2-D only). # Preconditions: ``n == 2``."""
        if self.n != 2:
            raise ValueError("gauss_curvature is defined for 2-D metrics; use scalar_curvature")
        r0101 = sum(self.g[0, s] * self.riemann_up(s, 1, 0, 1) for s in range(2))
        k = r0101 / self.g.det()
        return sp.trigsimp(sp.simplify(k)) if simplify else k

    def covariant_hessian(self, f: sp.Expr) -> sp.Matrix:
        r"""∇²f_ij = ∂_i∂_j f − Γ^k_ij ∂_k f (the connection-corrected Hessian)."""
        gamma, q, n = self.christoffel(), self.q, self.n
        return sp.Matrix(n, n, lambda i, j: sp.diff(f, q[i], q[j])
                         - sum(gamma[k][i][j] * sp.diff(f, q[k]) for k in range(n)))

    def bakry_emery(self, potential: sp.Expr) -> sp.Matrix:
        r"""∞-Bakry–Émery Ricci ``Ric(g) + ∇²V``; its min eigenvalue lower-bounds the entropy-relaxation rate."""
        return sp.simplify(self.ricci() + self.covariant_hessian(potential))


def _christoffel_numeric(metric_fn, q: np.ndarray, eps: float) -> "tuple[np.ndarray, np.ndarray]":
    """Γ^k_ij and g^{-1} at ``q`` by central differences of the metric callable. Returns (Gamma[k,i,j], ginv)."""
    n = q.size
    g = np.asarray(metric_fn(q), dtype=float)
    gi = np.linalg.inv(g)
    dg = np.zeros((n, n, n))                                  # dg[a,i,j] = ∂_a g_ij
    for a in range(n):
        e = np.zeros(n)
        e[a] = eps
        dg[a] = (np.asarray(metric_fn(q + e), float) - np.asarray(metric_fn(q - e), float)) / (2 * eps)
    gamma = np.zeros((n, n, n))                               # gamma[k,i,j]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                gamma[k, i, j] = 0.5 * sum(gi[k, s] * (dg[i, j, s] + dg[j, i, s] - dg[s, i, j]) for s in range(n))
    return gamma, gi


def scalar_curvature_numeric(metric_fn, q0, eps: float = 1e-4) -> float:
    r"""Scalar curvature of a metric given as a callable ``q -> (n×n) array``, via nested central differences.

    Enables curvature of the full humanoid metric (e.g. MuJoCo ``mj_fullM``) where the closed form is
    intractable. # Preconditions: ``metric_fn`` returns an SPD matrix; ``eps`` small vs the metric's scale.
    """
    q0 = np.asarray(q0, dtype=float)
    n = q0.size
    gamma0, gi = _christoffel_numeric(metric_fn, q0, eps)
    dgamma = np.zeros((n, n, n, n))                           # dgamma[a,k,i,j] = ∂_a Γ^k_ij
    for a in range(n):
        e = np.zeros(n)
        e[a] = eps
        gp, _ = _christoffel_numeric(metric_fn, q0 + e, eps)
        gm, _ = _christoffel_numeric(metric_fn, q0 - e, eps)
        dgamma[a] = (gp - gm) / (2 * eps)
    scalar = 0.0
    for i in range(n):
        for k in range(n):
            ric_ik = 0.0
            for s in range(n):                               # Ric_ik = Σ_s R^s_{isk}
                ric_ik += (dgamma[s, s, i, k] - dgamma[k, s, i, s]
                           + sum(gamma0[s, s, m] * gamma0[m, i, k] - gamma0[s, k, m] * gamma0[m, i, s]
                                 for m in range(n)))
            scalar += gi[i, k] * ric_ik
    return float(scalar)
