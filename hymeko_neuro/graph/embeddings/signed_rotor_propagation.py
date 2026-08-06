"""Signed slerp/nlerp propagation of rotors over a signed graph.

The leakage-free rotor line is bottlenecked by a degree-only node input (root
cause, 2026-06-17): every node's rotor is a function of its own 6-dim degree
features, so equal-degree nodes are indistinguishable. This module gives each node
a *neighbourhood-aware* rotor by propagating rotors over the signed graph — and it
does so the way computer graphics interpolates rotations/normals: by **nlerp**
(normalise-after-weighted-mean), the cheap slerp approximation that keeps the
result on the unit sphere instead of shrinking toward the centroid (the
"interpolate normals, then renormalise" rule). The aggregation is **signed** by
the edge sign, so it is the manifold analogue of SGCN's signed message passing —
but on S³ (quaternions), not in Euclidean feature space.

One round, per node :math:`i`, block :math:`b`:

    q̃_i = w_self · q_i + (1/deg_i) Σ_{j∼i} s_ij · align(q_j, q_i)
    q_i' = q̃_i / ‖q̃_i‖                                   (back onto S³, nlerp)

with ``align(q_j, q_i) = sign(⟨q_j, q_i⟩) · q_j`` — the double-cover hemisphere
fix (``q`` and ``−q`` are the same rotation; neighbours must share ``q_i``'s
hemisphere before averaging). ``s_ij ∈ {+1, −1}`` is the edge sign.

Leakage-safe: built from TRAIN edges/signs only; permuting the signs
(``--shuffle-train-signs``) changes the aggregate, so the leakage gate must fall
to chance.

Plan: docs/plans/2026-06-17-signed-rotor-slerp-propagation/.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SignedRotorPropagation(nn.Module):
    """Signed nlerp aggregation of per-node rotors on S³.

    The self-retention ``w_self`` controls over-smoothing: a fixed scalar is
    dataset-specific (sw=4 lifts both Bitcoin graphs; sw=1 collapses alpha). With
    ``learnable=True`` it becomes a **per-block** retention
    ``α_b = retention_floor + exp(θ_b)`` (``θ_b`` a parameter), so each block keeps
    as much of its own rotor as it needs and the schedule is learned, not tuned.

    Why ``exp`` (log-space), not ``softplus``: the final ``nlerp`` normalisation
    divides out overall scale, so only the *ratio* α : neighbour-mean matters — a
    multiplicative quantity, naturally learned as ``log(self_weight)``. This is
    exactly a per-block sigmoid **residual gate** ``g_b = σ(θ_b)`` mixing self and
    neighbours: after the normalise, ``(1−g)·nbr + g·q`` reduces to
    ``normalise(nbr + (g/(1−g))·q)`` with ``g/(1−g) = exp(θ)``. ``θ_b`` is
    initialised so ``α_b == self_weight`` at construction — the learnable module
    reproduces the fixed-``self_weight`` behaviour bit-for-bit before training (it
    can only improve), and ``retention_floor`` bounds α away from the documented
    sw→0 over-smoothing collapse.

    # Preconditions
    ``forward(q, edges, signs)``: ``q`` is ``(N, n_blocks, 4)`` unit quaternions
    (``n_blocks`` must equal the constructor's ``n_blocks`` when ``learnable``);
    ``edges`` is ``(E, 2)`` long (undirected — both directions are used);
    ``signs`` is ``(E,)`` in ``{+1, -1}``; all on one device.
    # Postconditions
    Returns ``(N, n_blocks, 4)`` unit quaternions (renormalised each round).
    Isolated nodes (no incident train edge) are returned unchanged (up to the
    self-weight normalisation). Fixed mode is a pure function of
    ``(q, edges, signs)``; learnable mode additionally depends on ``θ_b``.
    """

    def __init__(self, rounds: int = 1, self_weight: float = 1.0,
                 *, learnable: bool = False, n_blocks: int | None = None,
                 retention_floor: float = 0.0, eps: float = 1e-8) -> None:
        super().__init__()
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1; got {rounds}")
        if self_weight < 0:
            raise ValueError(f"self_weight must be >= 0; got {self_weight}")
        if retention_floor < 0:
            raise ValueError(f"retention_floor must be >= 0; got {retention_floor}")
        self.rounds = int(rounds)
        self.self_weight = float(self_weight)
        self.learnable = bool(learnable)
        self.retention_floor = float(retention_floor)
        self.eps = float(eps)
        if self.learnable:
            if n_blocks is None or n_blocks < 1:
                raise ValueError(
                    f"learnable retention requires n_blocks >= 1; got {n_blocks}")
            if self.self_weight <= self.retention_floor:
                raise ValueError(
                    "learnable retention needs self_weight > retention_floor "
                    f"(got {self.self_weight} <= {self.retention_floor})")
            self.n_blocks: int | None = int(n_blocks)
            # Init so exp(θ) + floor == self_weight ⇒ parity with fixed sw.
            theta0 = math.log(self.self_weight - self.retention_floor)
            self.log_self_weight = nn.Parameter(torch.full((self.n_blocks,), theta0))
        else:
            self.n_blocks = None

    def self_retention(self) -> torch.Tensor | float:
        """Effective ``w_self``: a per-block ``(n_blocks,)`` tensor when learnable,
        else the fixed scalar. ``α_b = retention_floor + exp(θ_b) >= floor``."""
        if self.learnable:
            return self.log_self_weight.exp() + self.retention_floor
        return self.self_weight

    def forward(self, q: torch.Tensor, edges: torch.Tensor,
                signs: torch.Tensor) -> torch.Tensor:
        if q.dim() != 3 or q.shape[-1] != 4:
            raise ValueError(f"expected q (N, n_blocks, 4); got {tuple(q.shape)}")
        if self.learnable and q.shape[1] != self.n_blocks:
            raise ValueError(
                f"learnable retention built for n_blocks={self.n_blocks}; "
                f"got q with {q.shape[1]} blocks")
        n_nodes = q.shape[0]
        w_self = self.self_retention()
        if isinstance(w_self, torch.Tensor):
            w_self = w_self.view(1, -1, 1)                # broadcast over (N, B, 4)
        # Symmetric directed incidence: each undirected edge feeds both endpoints.
        src = torch.cat([edges[:, 0], edges[:, 1]])
        dst = torch.cat([edges[:, 1], edges[:, 0]])
        sgn = torch.cat([signs, signs]).to(q.dtype)
        deg = torch.zeros(n_nodes, dtype=q.dtype, device=q.device)
        deg.index_add_(0, dst, torch.ones_like(sgn))
        deg = deg.clamp_min(1.0).view(-1, 1, 1)
        for _ in range(self.rounds):
            q = self._propagate(q, src, dst, sgn, deg, w_self)
        return q

    def _propagate(self, q: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                   sgn: torch.Tensor, deg: torch.Tensor,
                   w_self: torch.Tensor | float) -> torch.Tensor:
        q_src = q[src]                                    # (E2, B, 4)
        q_dst = q[dst]                                    # (E2, B, 4)
        # Hemisphere fix: flip q_src into q_dst's hemisphere before averaging.
        dot = (q_src * q_dst).sum(dim=-1, keepdim=True)   # (E2, B, 1)
        aligned = torch.where(dot < 0, -q_src, q_src)
        contrib = sgn.view(-1, 1, 1) * aligned            # signed (E2, B, 4)
        acc = torch.zeros_like(q)
        acc.index_add_(0, dst, contrib)
        acc = acc / deg + w_self * q                      # neighbour-mean + self
        return acc / acc.norm(dim=-1, keepdim=True).clamp_min(self.eps)  # nlerp → S³
