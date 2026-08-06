r"""M2-limit-cycle — certifying the running gait's basin via the Poincaré map (the L-coupled regime).

At soft regulation the fall genuinely depends on `L`, but the attractor is a **limit cycle** (the `z` bounce is
periodic and even `L`, `pitch` settle to a small periodic orbit), so a point-Lyapunov `V` collapses (M2⁺). The
standard fix: take a **Poincaré section** at gait phase 0 — the limit cycle becomes a **fixed point** ``x*`` of
the one-stride map ``P``, and a quadratic ``V(x)=(e)ᵀP(e)`` with ``e = (L,pitch)−(L*,pitch*)`` certifies it by the
stride-to-stride decrease ``V(P(x)) ≤ V(x)``. This reduces limit-cycle stability to point stability.

Verification comes in two forms: an **exact LMI** on the numerically-linearized stride map ``DP`` at ``x*``
(``Q=DPᵀP DP−P ⪯ 0``, plus its spectral radius < 1), and an **empirical** rollout-consistency check (section
states inside ``{V≤c}`` converge to the gait over several strides without falling).

# Preconditions: a stabilising (soft-regulated) ``CentroidalConfig`` whose nominal gait converges.
# Postconditions: ``V(x*)=0``, ``V ⪰ 0``; the certified set is a one-stride capturable inner approximation.
"""

from __future__ import annotations

import numpy as np

from scenarios.humanoid.centroidal import CentroidalConfig, centroidal_step


def soft_running_config(**overrides) -> CentroidalConfig:
    """A regulation-softened config where the fall depends on ``L`` (the certifiable limit-cycle regime)."""
    base = {"pitch_gain": 2.5, "l_damp": 4.0, "torque_bias": 3.0}
    base.update(overrides)
    return CentroidalConfig(**base)


def _steps_per_stride(cfg: CentroidalConfig) -> int:
    return int(round(cfg.cycle / cfg.dt))


def gait_fixed_point(cfg: CentroidalConfig, warmup_strides: int = 200) -> np.ndarray:
    """Iterate the Poincaré (stride) map from rest to its fixed point; return the section state ``x*`` (phase 0).

    Iterating ``stride_map`` (each call is exactly one phase-aligned stride) converges the ``(L, pitch)``
    transverse coordinates geometrically (the map contracts), giving an exact fixed point for the certificate.
    """
    state = np.array([[cfg.z0, 0.0, 0.0, 0.0]])
    for _ in range(warmup_strides):
        state, _ = stride_map(state, cfg)
    return state[0]


def stride_map(x: np.ndarray, cfg: CentroidalConfig) -> "tuple[np.ndarray, np.ndarray]":
    """One full stride forward from the section for a batch; return (next section states, fell-during-stride mask)."""
    state = np.atleast_2d(x).astype(float).copy()
    fell = np.zeros(len(state), dtype=bool)
    for i in range(_steps_per_stride(cfg)):
        state = centroidal_step(state, i * cfg.dt, cfg)
        fell |= np.abs(state[:, 3]) > cfg.fall_pitch
    return state, fell


class PoincareLyapunovCertificate:
    """Quadratic Lyapunov certificate on the Poincaré section (limit cycle ↦ fixed point of the stride map)."""

    def __init__(self, cfg: CentroidalConfig, eps: float = 1e-3) -> None:
        self.cfg, self.eps = cfg, eps
        self.xstar = gait_fixed_point(cfg)                       # the gait fixed point on the section
        self._chol = np.eye(2)                                   # over the (L, pitch) transverse deviation

    def _err(self, x: np.ndarray) -> np.ndarray:
        return np.atleast_2d(x)[:, 2:4] - self.xstar[2:4]        # (L−L*, pitch−pitch*)

    def matrix(self) -> np.ndarray:
        return self._chol @ self._chol.T + self.eps * np.eye(2)

    def value(self, x: np.ndarray) -> np.ndarray:
        e = self._err(x)
        return np.einsum("ni,ij,nj->n", e, self.matrix(), e)

    def _section_grid(self, n: int = 25, offset: float = 0.0) -> np.ndarray:
        """Section samples ``(z*, ż*, L, pitch)`` over the (L, pitch) box centred on ``x*``."""
        ls = self.xstar[2] + np.linspace(-self.cfg.l_max, self.cfg.l_max, n) + offset * (2 * self.cfg.l_max / n)
        ps = (self.xstar[3] + np.linspace(-self.cfg.pitch_max, self.cfg.pitch_max, n)
              + offset * (2 * self.cfg.pitch_max / n))
        ll, pp = np.meshgrid(ls, ps)
        z = np.full(ll.size, self.xstar[0])
        zd = np.full(ll.size, self.xstar[1])
        return np.stack([z, zd, ll.ravel(), pp.ravel()], axis=1)

    def fit(self, n: int = 25, iters: int = 2000, lr: float = 1e-3, tol: float = 1e-6) -> "PoincareLyapunovCertificate":
        """Shape ``V`` to decrease stride-to-stride on the non-falling section: ``min Σ relu(V(P x) − V(x))``."""
        x = self._section_grid(n)
        xp, fell = stride_map(x, self.cfg)
        e, ep = self._err(x)[~fell], self._err(xp)[~fell]
        for _ in range(iters):
            p = self.matrix()
            active = (np.einsum("ni,ij,nj->n", ep, p, ep) - np.einsum("ni,ij,nj->n", e, p, e)) > tol
            if not active.any():
                break
            ei, epi = e[active], ep[active]
            self._chol = self._chol - lr * np.tril(2.0 * (epi.T @ epi - ei.T @ ei) @ self._chol / len(x))
        return self

    def certified_level(self, x: np.ndarray, tol: float = 1e-6) -> float:
        """Largest ``c`` s.t. ``{V≤c}`` neither increases ``V`` (stride-to-stride) nor falls."""
        v = self.value(x)
        xp, fell = stride_map(x, self.cfg)
        bad = fell | (self.value(xp) - v > tol)
        return float(v[bad].min()) if bad.any() else float(v.max())

    def _stride_jacobian(self, eps: float = 1e-4) -> np.ndarray:
        """DP over (L, pitch) at ``x*`` by central differences (the Poincaré linearisation)."""
        cols = []
        for j in (2, 3):
            xp, xm = self.xstar.copy(), self.xstar.copy()
            xp[j] += eps
            xm[j] -= eps
            sp, _ = stride_map(xp[None, :], self.cfg)
            sm, _ = stride_map(xm[None, :], self.cfg)
            cols.append((sp[0, 2:4] - sm[0, 2:4]) / (2 * eps))
        return np.stack(cols, axis=1)

    def formal_verify(self) -> dict:
        """Exact LMI on the linearised stride map: ``Q = DPᵀ P DP − P ⪯ 0`` and spectral radius of ``DP`` < 1."""
        dp = self._stride_jacobian()
        q = dp.T @ self.matrix() @ dp - self.matrix()
        max_eig = float(np.linalg.eigvalsh(q).max())
        rho = float(np.max(np.abs(np.linalg.eigvals(dp))))
        return {"decreasing": bool(max_eig <= 1e-9), "max_eig_Q": max_eig,
                "spectral_radius_DP": rho, "gait_stable": bool(rho < 1.0)}

    def verify(self, n_strides: int = 8) -> dict:
        """Empirical: section states inside ``{V≤c}`` converge to the gait over ``n_strides`` without falling."""
        level = self.certified_level(self._section_grid())
        x = self._section_grid(offset=0.5)                       # held-out section grid
        inside = self.value(x) <= level
        state = x.copy()
        fell = np.zeros(len(x), dtype=bool)
        for _ in range(n_strides):
            state, f = stride_map(state, self.cfg)
            fell |= f
        recover = (self.value(state) < 0.25 * level) & ~fell     # settled near the gait fixed point
        inter = int(np.sum(inside & recover))
        union = int(np.sum(inside | recover))
        return {"certified_level": level, "n_certified": int(inside.sum()),
                "fall_violation_rate": float(np.mean(fell[inside])) if inside.any() else 0.0,
                "iou_vs_recoverable": float(inter / union) if union else 1.0, "n": int(len(x))}
