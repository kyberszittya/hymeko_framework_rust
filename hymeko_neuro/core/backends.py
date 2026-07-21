"""Aggregation backends (Strategy): how a signed-KAN layer aggregates neighbour messages.

The one *deep* axis that separated the two HSiKAN implementations. ``DenseBatchedBackend`` is the
inductive/batched path (the RL line): ``A`` is a dense ``(N, N)`` signed adjacency and features are a rollout
batch ``(B, N, d)``, so a single ``einsum`` aggregates over the batch. The sparse-scatter and Triton backends
(the transductive/vision path) register here in phase 3; they implement the same trait, so the core layer is
backend-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AggregationBackend(ABC):
    """Aggregate signed neighbour messages. ``aggregate(a_pos, a_neg, h) -> (agg_pos, agg_neg)`` where each
    output has the same shape as ``h`` and ``agg_s[..., i] = Σ_j A_s[i, j]·h[..., j]`` (receiver ``i`` reads
    from neighbour ``j``)."""

    @abstractmethod
    def aggregate(self, a_pos: torch.Tensor, a_neg: torch.Tensor,
                  h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class DenseBatchedBackend(AggregationBackend):
    """Dense signed adjacency ``(N, N)`` × batched features ``(B, N, d)`` via ``einsum`` — the inductive RL
    path. Correct for the tiny kinematic graphs (``N <= ~20``), where a dense ``A`` batches over the rollout
    while a sparse ``mm`` cannot.

    # Preconditions ``a_pos``/``a_neg`` are ``(N, N)``; ``h`` is ``(B, N, d)``.
    # Postconditions returns two ``(B, N, d)`` tensors.
    """

    def aggregate(self, a_pos: torch.Tensor, a_neg: torch.Tensor,
                  h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h.dim() != 3:
            raise ValueError(f"DenseBatchedBackend expects h as (B, N, d); got {tuple(h.shape)}")
        agg_pos = torch.einsum("nm,bmd->bnd", a_pos, h)
        agg_neg = torch.einsum("nm,bmd->bnd", a_neg, h)
        return agg_pos, agg_neg


class SparseSignedBackend(AggregationBackend):
    """Sparse signed adjacency `(N, N)` × features `(B, N, d)` via `torch.sparse.mm` — the transductive /
    large-graph path (e.g. the signed-graph link-sign datasets, where a dense `(N, N)` is infeasible).

    The dense `(N, N)` would be `N² ≈ 10⁷+` entries on the OTC-class graphs; the sparse mm keeps it `O(|E|·d)`.
    `B` is typically 1 (one transductive graph); each batch row is a separate sparse mm. Produces results
    numerically equal to :class:`DenseBatchedBackend` on the same graph (a parity test pins this).

    # Preconditions `a_pos`/`a_neg` are sparse (or sparse-convertible) `(N, N)`; `h` is `(B, N, d)`.
    # Postconditions returns two `(B, N, d)` tensors.
    """

    def aggregate(self, a_pos: torch.Tensor, a_neg: torch.Tensor,
                  h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h.dim() != 3:
            raise ValueError(f"SparseSignedBackend expects h as (B, N, d); got {tuple(h.shape)}")
        ap = a_pos.to_sparse() if not a_pos.is_sparse else a_pos
        an = a_neg.to_sparse() if not a_neg.is_sparse else a_neg
        agg_pos = torch.stack([torch.sparse.mm(ap, h[b]) for b in range(h.shape[0])])
        agg_neg = torch.stack([torch.sparse.mm(an, h[b]) for b in range(h.shape[0])])
        return agg_pos, agg_neg
