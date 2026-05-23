"""Arity-agnostic n-tuple balance loss (Davis 1967 weak balance).

Generalises ``TriadLoss`` to k-uniform signed hyperedges, and ingests
mixed-arity batches via flat (edge × tuple_id) gather tensors so a
single forward pass handles k = 3, 4, 5, ... in the same call.

For each n-tuple ``t`` with cycle ``(v_0, v_1, …, v_{k-1})`` and
cycle-edge signs ``σ_0, …, σ_{k-1}``:

    s(t) = (1/k) · Σ_i  σ_i · cos(h_{v_i}, h_{v_{i+1 mod k}})
    β(t) = Π_i σ_i        ∈ {+1, −1}     (Davis weak balance)
    L_t  = relu(margin − β(t) · s(t))

For ``k = 3`` this is mathematically equivalent (modulo a 1/3
normalisation for k-invariance) to the existing ``TriadLoss`` —
the existing recipe sums three pair-cosines, this version means
them, so margin scale needs a 1/k correction. We absorb that
by letting the alpha kwarg in ``run_one`` re-tune.

The same loss is the HSiKAN-side analog of SGCN's extended
structural balance loss (Derr 2018 §3.3): both encode
Cartwright–Harary balance theory in the embedding-similarity
prior, but on the architecture's native scale (signed n-tuples
for HSiKAN, B/U dual embeddings for SGCN).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class NTupleBalanceLossConfig:
    margin: float = 0.5
    alpha:  float = 1.0
    eps:    float = 1e-8


def build_ntuple_balance_tensors(
        tuples,                       # list[SignedNTuple] or list[SignedTriad]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Pre-compute the flat tensors used by ``NTupleBalanceLoss``.

    Accepts either ``SignedNTuple`` (any arity, has ``edge_signs``) or
    ``SignedTriad`` (k=3, has ``edge_signs``); both expose ``v`` and
    ``edge_signs`` in cycle order.

    Returns
    -------
    pair_idx     : (E_total, 2)  int64 — vertex pairs over ALL cycle edges
    pair_sign    : (E_total,)    float — ±1 per cycle edge
    tuple_id     : (E_total,)    int64 — tuple index each edge belongs to
    beta         : (T,)          float — ±1 balance indicator per tuple
    arity        : (T,)          float — k for each tuple, for s(t) mean
    """
    pair_pairs: list[tuple[int, int]] = []
    pair_signs: list[float] = []
    tuple_ids:  list[int] = []
    betas:      list[float] = []
    arities:    list[float] = []
    for ti, t in enumerate(tuples):
        v = t.v
        edge_signs = t.edge_signs
        k = len(v)
        for i in range(k):
            j = (i + 1) % k
            pair_pairs.append((int(v[i]), int(v[j])))
            pair_signs.append(float(edge_signs[i]))
            tuple_ids.append(ti)
        prod = 1
        for s in edge_signs:
            prod *= int(s)
        betas.append(float(prod))
        arities.append(float(k))
    pair_idx_t  = torch.tensor(pair_pairs,  dtype=torch.long)
    pair_sign_t = torch.tensor(pair_signs,  dtype=torch.float32)
    tuple_id_t  = torch.tensor(tuple_ids,   dtype=torch.long)
    beta_t      = torch.tensor(betas,       dtype=torch.float32)
    arity_t     = torch.tensor(arities,     dtype=torch.float32)
    return pair_idx_t, pair_sign_t, tuple_id_t, beta_t, arity_t


class NTupleBalanceLoss(nn.Module):
    """Arity-agnostic Cartwright-Harary balance loss for HSiKAN."""

    def __init__(self, cfg: NTupleBalanceLossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, h: torch.Tensor,
                pair_idx: torch.Tensor,
                pair_sign: torch.Tensor,
                tuple_id: torch.Tensor,
                beta: torch.Tensor,
                arity: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h         : (V, d) node embeddings
        pair_idx  : (E_total, 2)  int64
        pair_sign : (E_total,)    float ±1
        tuple_id  : (E_total,)    int64 — which tuple each edge belongs to
        beta      : (T,)          float ±1
        arity     : (T,)          float — k (number of cycle edges)
        """
        eps = self.cfg.eps
        h_i = h[pair_idx[:, 0]]                                # (E_total, d)
        h_j = h[pair_idx[:, 1]]
        h_i_n = h_i / (h_i.norm(dim=-1, keepdim=True) + eps)
        h_j_n = h_j / (h_j.norm(dim=-1, keepdim=True) + eps)
        cos_pair = (h_i_n * h_j_n).sum(dim=-1)                 # (E_total,)
        weighted = pair_sign * cos_pair                         # (E_total,)
        # Sum per tuple, then mean over arity for k-invariance.
        s = torch.zeros(beta.shape[0], device=h.device, dtype=weighted.dtype)
        s = s.scatter_add_(0, tuple_id, weighted)               # (T,)
        s = s / arity.clamp_min(1.0)
        return torch.relu(self.cfg.margin - beta * s).mean()
