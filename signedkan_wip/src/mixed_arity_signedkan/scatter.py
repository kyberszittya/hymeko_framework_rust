"""Auto-split from mixed_arity_signedkan.py 2026-05-11 (CLAUDE.md §6.5 #4).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _scatter_topk_mask(scores: torch.Tensor, index: torch.Tensor,
                         K: int) -> torch.Tensor:
    """Per-row top-K mask over sparse (scores, index) entries.

    For each unique value of ``index``, returns ``True`` for the K
    highest-scoring entries with that index, ``False`` otherwise.
    Empty rows produce no True entries.

    Vectorised via a single sort: order entries lexicographically by
    (row, -score), then within each row the first K (by sort order)
    are the top-K.  Cost: O(nnz log nnz).
    """
    if K is None or K <= 0:
        return torch.ones_like(scores, dtype=torch.bool)
    nnz = scores.shape[0]
    if nnz == 0:
        return torch.zeros(0, dtype=torch.bool, device=scores.device)
    # Sort key: row * BIG - score so smaller key = (lower row) or
    # (same row, higher score).  Avoid float collision with row by
    # using a large multiplier; subtract score so descending within
    # row.
    n_rows_inferred = int(index.max().item()) + 1
    big = 1e9
    sort_key = index.to(torch.float64) * big - scores.to(torch.float64)
    order = torch.argsort(sort_key)
    sorted_rows = index[order]
    pos = torch.arange(nnz, device=scores.device)
    is_new_row = torch.cat([
        torch.tensor([True], device=scores.device),
        sorted_rows[1:] != sorted_rows[:-1],
    ])
    # row_start_at_pos = pos at every "first slot of a new row", else 0.
    row_start_at_pos = torch.where(
        is_new_row, pos, torch.zeros_like(pos),
    )
    row_start = torch.cummax(row_start_at_pos, dim=0).values
    rank_in_row = pos - row_start
    keep_sorted = rank_in_row < K
    mask = torch.zeros(nnz, dtype=torch.bool, device=scores.device)
    mask[order] = keep_sorted
    return mask


def _attn_softmax_dispatch(scores: torch.Tensor, index: torch.Tensor,
                             n_rows: int) -> torch.Tensor:
    """Dispatcher: dense or top-K row-wise softmax based on the
    ``HSIKAN_SPARSE_ATTN_K`` env var.  Default ``0`` → dense softmax
    (existing behavior).  Set to a positive int K → only the K
    highest-scoring entries per row contribute to softmax; the rest
    get exactly zero attention weight.
    """
    from ..runtime_config import get_runtime
    K = get_runtime().training.sparse_attn_k
    if K > 0:
        return _scatter_topk_softmax(scores, index, n_rows, K)
    return _scatter_softmax(scores, index, n_rows)


def _scatter_topk_softmax(scores: torch.Tensor, index: torch.Tensor,
                            n_rows: int, K: int) -> torch.Tensor:
    """Row-wise softmax over the top-K scores per row, zero elsewhere.

    Identical to ``_scatter_softmax`` when ``K`` is None / non-positive
    or when every row has fewer than K entries.  Otherwise, per row,
    the K highest-scoring entries get the standard softmax
    distribution and the remaining entries get exactly zero
    attention weight.

    This is the sparse-attention path: dense attention over the full
    cycle pool is too diffuse on dense graphs (Epinions); top-K gives
    the model a hard inductive bias toward a small subset of cycles
    per query.
    """
    if K is None or K <= 0:
        return _scatter_softmax(scores, index, n_rows)
    mask = _scatter_topk_mask(scores, index, K)
    neg_inf = scores.new_full(scores.shape, float("-inf"))
    masked_scores = torch.where(mask, scores, neg_inf)
    return _scatter_softmax(masked_scores, index, n_rows)


def _scatter_softmax(scores: torch.Tensor, index: torch.Tensor,
                       n_rows: int) -> torch.Tensor:
    """Row-wise softmax over a sparse representation.

    scores : (nnz,) raw attention scores
    index  : (nnz,) row index for each score
    n_rows : int total number of rows

    Returns: (nnz,) softmax values, where for each row, the values at
    positions in that row sum to 1. Handles empty rows by leaving them
    untouched (no entries).
    """
    # Per-row max for numerical stability.
    max_per_row = torch.full(
        (n_rows,), float("-inf"), device=scores.device, dtype=scores.dtype,
    )
    max_per_row.scatter_reduce_(
        0, index, scores, reduce="amax", include_self=True,
    )
    # Replace -inf (empty rows) with 0 so subtraction doesn't propagate NaN.
    max_per_row = max_per_row.masked_fill(
        max_per_row == float("-inf"), 0.0,
    )
    shifted = scores - max_per_row[index]
    exp_scores = shifted.exp()
    sum_per_row = torch.zeros(
        n_rows, device=scores.device, dtype=scores.dtype,
    )
    sum_per_row.scatter_add_(0, index, exp_scores)
    return exp_scores / (sum_per_row[index] + 1e-12)


