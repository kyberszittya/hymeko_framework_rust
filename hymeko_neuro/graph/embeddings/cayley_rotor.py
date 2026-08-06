"""Cayley-rotor embedding — a Clifford/quaternion rotor feature on the sphere.

Motivation (plan: ``docs/plans/2026-06-16-soma-structural-highway/``). The
signed-link models here are *transductive and featureless*: a node's feature is a
learned ``nn.Embedding(n_nodes, d)`` row keyed by node ID. That table is the
memorisation channel behind the leakage-audit collapse, and it cannot transfer to
a setting where items have no persistent ID (per-image detector anchors). This
module is the *inductive*, param-light replacement.

Each ``d=3``-block of the embedding is produced by rotating a learned reference
vector ``v0`` by a unit quaternion (a 3-D rotor, the even subalgebra Cl(0,2)+).
The rotor is the **Cayley map** of unconstrained Rodrigues parameters
``b in R^3``::

    q = [1; b] / ||[1; b]||              (unit quaternion, by construction)
    e = q v0 q^{-1}                       (rotor sandwich -> sphere of radius |v0|)

Because *any* ``b`` yields a unit rotor, plain SGD on ``b`` is optimisation on the
rotor manifold (a Cayley parameterisation) — no re-orthogonalisation, no manifold
solver. Reuses the quaternion-block idea already present in ``ac_hsikan``; the
algebra spec is ``hymeko_clifford`` (Rust, Phase-1, not PyO3-exposed, hence the
PyTorch reimplementation for autograd).

Known limit (denoted): the Cayley map cannot represent a 180-degree rotation
(``|b| -> inf``); fine for a learned embedding, documented for honesty.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Rotor algebra (quaternion = Cl(0,2)+); autograd-friendly, no graph state.
# ---------------------------------------------------------------------


def cayley_to_unit_quat(b: torch.Tensor) -> torch.Tensor:
    """Cayley map: Rodrigues params ``b`` to a unit quaternion ``[w, x, y, z]``.

    Preconditions: ``b.shape[-1] == 3``.
    Postconditions: returns ``q`` with ``q.shape == b.shape[:-1] + (4,)`` and
    ``||q|| == 1`` along the last axis (to floating tolerance), with ``w >= 0``.
    """
    if b.shape[-1] != 3:
        raise ValueError(f"bivector last dim must be 3; got {tuple(b.shape)}")
    w = torch.ones(b.shape[:-1] + (1,), dtype=b.dtype, device=b.device)
    raw = torch.cat([w, b], dim=-1)                      # [1, b]
    return raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` by unit quaternions ``q`` (the sandwich product).

    Uses the cross-product form ``v + 2 w (u x v) + 2 (u x (u x v))`` with
    ``q = [w, u]`` — no explicit conjugate, autograd-clean.

    Preconditions: ``q.shape[-1] == 4``, ``v.shape[-1] == 3``, broadcastable
    leading dims. Postconditions: output shape ``broadcast(q[...], v[...]) + (3,)``;
    if ``q`` is unit then ``||rotate(q, v)|| == ||v||``.
    """
    if q.shape[-1] != 4 or v.shape[-1] != 3:
        raise ValueError(f"expected q[...,4] and v[...,3]; got {tuple(q.shape)}, {tuple(v.shape)}")
    w = q[..., :1]
    u = q[..., 1:]
    uxv = torch.cross(u, v.expand_as(u), dim=-1)
    return v + 2.0 * w * uxv + 2.0 * torch.cross(u, uxv, dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate ``[w, -x, -y, -z]`` (the inverse of a *unit* rotor).

    Preconditions: ``q.shape[-1] == 4``. Postconditions: same shape; for unit
    ``q``, ``quat_mul(quat_conjugate(q), q) == [1,0,0,0]`` to fp tolerance."""
    if q.shape[-1] != 4:
        raise ValueError(f"expected q[...,4]; got {tuple(q.shape)}")
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a ⊗ b`` (broadcast over leading dims; last dim 4).

    Autograd-clean (the canonical quaternion product for the rotor line; the
    fused variant in ``ac_hsikan/.../pool_scatter`` is a private Triton kernel for
    that pool's hot path, deliberately not reused here).

    Preconditions: ``a.shape[-1] == b.shape[-1] == 4``, broadcastable leading
    dims. Postconditions: shape ``broadcast(a, b)``; associative; ``[1,0,0,0]``
    is the identity."""
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError(f"expected [...,4] quaternions; got {tuple(a.shape)}, {tuple(b.shape)}")
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ], dim=-1)


# ---------------------------------------------------------------------
# Embedding module
# ---------------------------------------------------------------------


class CayleyRotorEmbedding(nn.Module):
    """A rotor-on-the-sphere embedding, ``embedding_dim = 3 * n_blocks``.

    Exactly one of ``n_items`` (transductive: a learned bivector per item) or
    ``in_features`` (inductive: bivector = a shared linear over input features)
    must be given. The inductive mode has no per-item table, so its parameter
    count is independent of the item count — the leakage-free, reusable path.

    Preconditions: ``n_blocks >= 1``; exactly one of ``n_items``/``in_features``.
    Postconditions: ``forward`` returns ``(batch, 3 * n_blocks)`` with each
    3-block on a sphere of radius ``||v0_block||``.
    """

    def __init__(
        self,
        n_blocks: int = 8,
        *,
        n_items: int | None = None,
        in_features: int | None = None,
        n_refs: int = 1,
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}")
        if n_refs < 1:
            raise ValueError(f"n_refs must be >= 1; got {n_refs}")
        if (n_items is None) == (in_features is None):
            raise ValueError(
                "pass exactly one of n_items (transductive) or in_features (inductive)"
            )
        self.n_blocks = n_blocks
        self.n_refs = n_refs
        self.embedding_dim = 3 * n_blocks * n_refs
        # Learned reference frame(s), n_refs 3-vectors per block (init non-
        # degenerate). n_refs >= 2 makes the full rotation observable from e:
        # one ref leaves the about-ref DoF invisible (R v0 = v0 about v0).
        self.reference = nn.Parameter(torch.randn(n_blocks, n_refs, 3) * 0.1 + 0.5)
        self.inductive = in_features is not None
        if self.inductive:
            # Shared map x -> bivectors; O(in_features * d) params, item-count free.
            self.proj = nn.Linear(in_features, 3 * n_blocks)
            self.bivectors = None
        else:
            # Transductive: one bivector per item (init 0 -> identity rotor).
            self.bivectors = nn.Parameter(torch.zeros(n_items, n_blocks, 3))
            self.proj = None

    def _bivectors(self, x: torch.Tensor) -> torch.Tensor:
        """Rodrigues params ``b`` (batch, n_blocks, 3) from indices/features."""
        if self.inductive:
            return self.proj(x).reshape(x.shape[0], self.n_blocks, 3)
        if x.dtype not in (torch.long, torch.int64, torch.int32):
            raise ValueError("transductive mode expects integer item indices")
        return self.bivectors[x]

    def rotors(self, x: torch.Tensor) -> torch.Tensor:
        """The unit-quaternion rotors per item/block, ``(batch, n_blocks, 4)``.

        Exposes the rotor *before* the lossy ``R v0`` projection, so a relative-
        rotation head can use the full SO(3) DoF the flattened embedding discards.
        Postconditions: unit-norm along the last axis (to fp tolerance)."""
        return cayley_to_unit_quat(self._bivectors(x))

    def embed_rotors(self, q: torch.Tensor) -> torch.Tensor:
        """Rotate the learned reference frame(s) by per-block rotors ``q``
        ``(batch, n_blocks, 4)`` → ``(batch, embedding_dim)``. The second half of
        ``forward``, exposed so a propagation step (e.g. signed slerp/nlerp
        aggregation) can transform ``q`` on the manifold *before* embedding.

        # Preconditions ``q.shape[-2:] == (n_blocks, 4)`` (unit rotors)."""
        if tuple(q.shape[-2:]) != (self.n_blocks, 4):
            raise ValueError(f"expected q[..., {self.n_blocks}, 4]; got {tuple(q.shape)}")
        batch = q.shape[0]
        qe = q.unsqueeze(2).expand(batch, self.n_blocks, self.n_refs, 4)
        v0 = self.reference.unsqueeze(0).expand(batch, -1, -1, -1)
        e = quat_rotate(qe, v0)                           # (batch, n_blocks, n_refs, 3)
        return e.reshape(batch, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is item indices ``(batch,)`` (transductive) or features
        ``(batch, in_features)`` (inductive)."""
        return self.embed_rotors(self.rotors(x))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
