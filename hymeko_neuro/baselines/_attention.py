"""Vectorized segment (CSR) attention over variable-size neighbour sets.

Shared by ``MotifAttention`` (sigat) and ``SignedAttention`` (sgt) — both
previously rebuilt per-length index tensors and looped over by-length groups on
*every* forward (the Python-side cost that drove the Phase-B grid to ~80 GPU-h).
Here the flattening is done **once per run** (cached on the caller) and the
attention is a handful of gather + segment-reduce ops with ``O(E)`` memory, no
global max-degree padding (which would OOM on Epinions).

Plan: ``docs/plans/2026-06-14-vectorize-signed-attention/``.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def build_csr(
    buckets: list[list[int]],
    device: torch.device,
    signs: list[list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Flatten variable-size neighbour buckets into CSR incidence tensors.

    # Preconditions
    - ``buckets[v]`` lists the neighbour node ids of node ``v``; if ``signs`` is
      given, ``signs[v]`` is the parallel ±1 list (same lengths).

    # Postconditions
    - returns ``(seg, nbr, sgn)`` each of length ``E = sum_v len(buckets[v])``
      (``sgn`` is ``None`` when ``signs`` is ``None``); ``seg[k]`` is the source
      node of incidence ``k``, ``nbr[k]`` its neighbour. Empty buckets contribute
      no incidences (their node rows stay zero downstream).
    """
    n = len(buckets)
    lengths = np.fromiter((len(b) for b in buckets), dtype=np.int64, count=n)
    seg_np = np.repeat(np.arange(n, dtype=np.int64), lengths)
    if lengths.sum() > 0:
        nbr_np = np.concatenate([np.asarray(b, dtype=np.int64) for b in buckets if b])
    else:
        nbr_np = np.empty(0, dtype=np.int64)
    seg = torch.from_numpy(seg_np).to(device)
    nbr = torch.from_numpy(nbr_np).to(device)
    sgn = None
    if signs is not None:
        if lengths.sum() > 0:
            sgn_np = np.concatenate([np.asarray(s, dtype=np.float32) for s in signs if s])
        else:
            sgn_np = np.empty(0, dtype=np.float32)
        sgn = torch.from_numpy(sgn_np).to(device)
    return seg, nbr, sgn


def segment_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    seg: torch.Tensor,
    nbr: torch.Tensor,
    n_nodes: int,
    bias_per_inc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-node multi-head attention over each node's incident neighbour set.

    # Preconditions
    - ``Q, K, V`` are ``(n_nodes, n_heads, head_dim)``.
    - ``seg, nbr`` are ``(E,)`` long; ``bias_per_inc`` (optional) is ``(E, n_heads)``
      additive score bias.

    # Postconditions
    - returns ``(n_nodes, n_heads, head_dim)``; a node with no incidences yields a
      zero row (matching the previous by-length-loop implementation).
    """
    n_heads, head_dim = Q.shape[1], Q.shape[2]
    scale = 1.0 / math.sqrt(head_dim)

    if seg.numel() == 0:
        return Q.new_zeros((n_nodes, n_heads, head_dim))

    q_i = Q[seg]                                   # (E, H, D)
    k_j = K[nbr]
    v_j = V[nbr]
    scores = (q_i * k_j).sum(dim=-1) * scale       # (E, H)
    if bias_per_inc is not None:
        scores = scores + bias_per_inc

    seg_h = seg.unsqueeze(-1).expand(-1, n_heads)  # (E, H)
    # Segment-max (per node, per head) for numerical stability.
    maxes = scores.new_full((n_nodes, n_heads), float("-inf"))
    maxes = maxes.scatter_reduce(0, seg_h, scores, reduce="amax", include_self=True)
    exp = (scores - maxes[seg]).exp()              # (E, H)
    denom = exp.new_zeros((n_nodes, n_heads)).index_add(0, seg, exp)
    attn = exp / denom[seg].clamp_min(1e-20)       # (E, H)

    weighted = v_j * attn.unsqueeze(-1)            # (E, H, D)
    out = weighted.new_zeros((n_nodes, n_heads, head_dim))
    return out.index_add(0, seg, weighted)
