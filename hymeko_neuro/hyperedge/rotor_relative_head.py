"""Rotor-relative edge head — projects the *full* relative rotation to a logit.

The leakage-free rotor line predicts an edge sign from the two endpoints' node
embeddings. The bilinear head ``γ h_u^T W h_v`` is, on rotor coordinates, a
learned projection of the relative rotation ``R_u^T R_v`` — but it only ever sees
``e = R v0`` (the rotor applied to one reference per block), and a rotation about
the ``v0`` axis fixes ``v0``. That whole DoF is discarded before the head, so
bilinear projects a degraded relative rotation.

This head recovers it: from the endpoint rotors ``q_u, q_v`` (exposed by
``CayleyRotorEmbedding.rotors``) it forms the relative unit quaternion
``r = conj(q_u) ⊗ q_v`` per block — the *full* relative rotation (the "relative
phase" in the complex special case) — and projects the stacked ``(4 · n_blocks)``
relative-rotor coordinates to a scalar. Added to the bilinear logit; a dead-start
LayerScale ``γ`` (like ``BilinearHead``) lets the optimiser switch it on only if
the recovered DoF helps.

Plan: docs/plans/2026-06-17-rotor-relative-projection/.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hymeko_neuro.graph.embeddings.cayley_rotor import quat_conjugate, quat_mul


class RotorRelativeHead(nn.Module):
    """Project the relative rotor ``conj(q_u) ⊗ q_v`` of an edge's endpoints.

    # Preconditions
    ``n_blocks >= 1``; ``forward`` receives ``q_u, q_v`` of shape
    ``(E, n_blocks, 4)`` (unit quaternions, e.g. from ``CayleyRotorEmbedding``).
    # Postconditions
    Returns ``(E,)`` logits; ``γ`` starts at ``gamma_init`` so the term is
    effectively off at init. Parameter count is ``4·n_blocks + 2``, independent of
    graph size.
    # Invariants
    Pure function of the two endpoint rotors — no graph state, no side effects.
    """

    def __init__(self, n_blocks: int, gamma_init: float = 1e-3) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}")
        self.n_blocks = int(n_blocks)
        self.proj = nn.Linear(4 * n_blocks, 1)
        with torch.no_grad():
            self.proj.weight.mul_(1.0 / (4 * n_blocks) ** 0.5)
            self.proj.bias.zero_()
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, q_u: torch.Tensor, q_v: torch.Tensor) -> torch.Tensor:
        """``q_u, q_v``: ``(E, n_blocks, 4)`` endpoint rotors → ``(E,)`` logit."""
        if q_u.shape != q_v.shape or q_u.shape[-1] != 4:
            raise ValueError(
                f"expected matching (E, n_blocks, 4) rotors; got "
                f"{tuple(q_u.shape)}, {tuple(q_v.shape)}")
        r = quat_mul(quat_conjugate(q_u), q_v)            # (E, n_blocks, 4)
        flat = r.reshape(r.shape[0], -1)                  # (E, 4*n_blocks)
        return self.gamma * self.proj(flat).squeeze(-1)


class RotorGeodesicHead(nn.Module):
    """RotatE/geodesic head: score by the *alignment* of the endpoint rotors.

    Per block, the relative rotor's scalar part ``(conj(q_u)⊗q_v)₀ = ⟨q_u,q_v⟩``
    is ``cos(θ/2)`` of the relative rotation — a smooth geodesic alignment on S³
    (1 = same rotation, 0 = orthogonal). A learned linear map over the per-block
    cosines gives the logit. This is the *distance/metric* sibling of
    ``RotorRelativeHead`` (which projects the full 4-vector): here only the
    rotation magnitude/alignment channel is read — the RotatE idea that the
    relation is a rotation and the score is how aligned the endpoints are after it.

    # Preconditions ``q_u, q_v: (E, n_blocks, 4)`` unit quaternions.
    # Postconditions returns ``(E,)``; dead-start ``gamma``; invariant to a global
    rotation of both endpoints (reads only the relative rotation).
    """

    def __init__(self, n_blocks: int, gamma_init: float = 1e-3) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}")
        self.n_blocks = int(n_blocks)
        self.proj = nn.Linear(n_blocks, 1)
        with torch.no_grad():
            self.proj.weight.mul_(1.0 / n_blocks ** 0.5)
            self.proj.bias.zero_()
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, q_u: torch.Tensor, q_v: torch.Tensor) -> torch.Tensor:
        if q_u.shape != q_v.shape or q_u.shape[-1] != 4:
            raise ValueError(
                f"expected matching (E, n_blocks, 4) rotors; got "
                f"{tuple(q_u.shape)}, {tuple(q_v.shape)}")
        cos_half = quat_mul(quat_conjugate(q_u), q_v)[..., 0]   # (E, n_blocks)
        return self.gamma * self.proj(cos_half).squeeze(-1)
