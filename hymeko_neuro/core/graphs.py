"""Signed-graph utilities for the engine — build a signed incidence from an edge list.

Reusable across signed-graph consumers (link-sign datasets, the OTC harness, …): turns ``(edges, signs)`` into the
``(A⁺, A⁻)`` sparse signed adjacency the :class:`~hymeko_neuro.core.backbone.SignedKANBackbone` consumes. Kept here (the
engine) rather than re-implemented per experiment.
"""
from __future__ import annotations

from typing import Any

import torch


def build_signed_adjacency(edges: Any, signs: Any, n_nodes: int, *, symmetric: bool = True,
                           row_normalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """``(edges (E,2), signs (E,) ∈ {+1,-1}) -> (a_pos, a_neg)`` sparse ``(n_nodes, n_nodes)``.

    ``a[i, j]`` = receiver ``i`` aggregates neighbour ``j``. Only the *given* edges populate the adjacency — pass
    train edges only for a leakage-free link-sign setup. ``symmetric`` adds both directions (undirected signed
    graph); ``row_normalize`` divides each row by its degree (so each row sums to 1 within its sign).

    # Preconditions ``edges`` is ``(E, 2)`` int; ``signs`` is ``(E,)`` in ``{+1, -1}``; ``n_nodes`` > max id.
    # Postconditions two coalesced sparse ``(n_nodes, n_nodes)`` tensors.
    """
    e = torch.as_tensor(edges, dtype=torch.long)
    s = torch.as_tensor(signs)
    if e.dim() != 2 or e.shape[1] != 2:
        raise ValueError(f"edges must be (E, 2); got {tuple(e.shape)}")

    def build(mask: torch.Tensor) -> torch.Tensor:
        sub = e[mask]
        if symmetric:
            i = torch.cat([sub[:, 0], sub[:, 1]])
            j = torch.cat([sub[:, 1], sub[:, 0]])
        else:
            i, j = sub[:, 0], sub[:, 1]
        idx = torch.stack([i, j])
        a = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1]), (n_nodes, n_nodes)).coalesce()
        if not row_normalize:
            return a
        deg = torch.sparse.sum(a, dim=1).to_dense().clamp(min=1.0)
        rows = a.indices()[0]
        return torch.sparse_coo_tensor(a.indices(), a.values() / deg[rows], (n_nodes, n_nodes)).coalesce()

    return build(s == 1), build(s == -1)
