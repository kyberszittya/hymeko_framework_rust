"""Auto-split from mixed_arity_signedkan.py 2026-05-11 (CLAUDE.md §6.5 #4).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .scatter import _attn_softmax_dispatch

class _AttentionM_e(nn.Module):
    """Replaces uniform 1/|N(query)| pooling in M_e with learned softmax
    attention over cycles. Uses dot-product attention between query-edge
    embedding and per-cycle embedding.

    Parameters: 2 Linear layers (W_q, W_k) projecting to ``d_attn`` dim.

    Init strategy: W_q, W_k initialised to very small values so softmax
    starts approximately uniform (same as 1/|N(query)| baseline). Random
    Kaiming init at training start gives extreme attention concentration
    that wrecks early training; near-zero init lets attention learn
    deviations from uniform incrementally.
    """
    def __init__(self, d_query: int, d_cycle: int, d_attn: int = 32,
                 n_heads: int = 1):
        super().__init__()
        if d_attn % n_heads != 0:
            raise ValueError(
                f"_AttentionM_e: d_attn ({d_attn}) must be divisible by "
                f"n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.d_head = d_attn // n_heads
        self.W_q = nn.Linear(d_query, d_attn, bias=False)
        self.W_k = nn.Linear(d_cycle, d_attn, bias=False)
        # Near-uniform init: scale weights by 1e-2 so initial scores ≈ 0
        # → softmax over rows ≈ 1/|N(row)|.
        with torch.no_grad():
            self.W_q.weight.mul_(0.01)
            self.W_k.weight.mul_(0.01)
        self.scale = self.d_head ** -0.5

    def forward(self, h_query: torch.Tensor, h_cycle: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """
        h_query: (E, d_query) query-edge embeddings
        h_cycle: (T, d_cycle) per-cycle embeddings
        indices: (2, nnz) — indices[0]=query row, indices[1]=cycle col

        Returns: (nnz,) softmax-normalised attention weights to use as
        the values of the sparse M_e tensor. With n_heads>1 the per-head
        softmax weights are averaged before returning, so downstream
        scatter sees a single (nnz,) weight tensor.
        """
        rows = indices[0]
        cols = indices[1]
        q_proj = self.W_q(h_query).view(-1, self.n_heads, self.d_head)
        k_proj = self.W_k(h_cycle).view(-1, self.n_heads, self.d_head)
        # Per-head scores: (nnz, n_heads)
        scores = (q_proj[rows] * k_proj[cols]).sum(dim=-1) * self.scale
        if self.n_heads == 1:
            return _attn_softmax_dispatch(scores.squeeze(-1), rows,
                                            h_query.shape[0])
        # Per-head softmax, then average across heads.
        attn_heads = torch.stack(
            [
                _attn_softmax_dispatch(scores[:, h], rows,
                                        h_query.shape[0])
                for h in range(self.n_heads)
            ],
            dim=-1,
        )
        return attn_heads.mean(dim=-1)


class _QuaternionAttentionM_e(nn.Module):
    """Quaternion-valued attention head over (query_edge, cycle) pairs.

    Same I/O contract as :class:`_AttentionM_e` (returns softmax
    weights over the sparse `M_e` non-zeros), but the score function
    treats the per-pair `d_attn` projection as `d_attn / 4`
    independent quaternions and uses

        score = Σ_q  real(q_i ⊗ k_i)
              = Σ_q  (q_a·k_a − q_b·k_b − q_c·k_c − q_d·k_d)

    The negative sign on the (i, j, k) components is the
    distinguishing feature: in standard scalar attention, every
    embedding dimension contributes positively to the score, so
    "agreement" and "anti-agreement" both pull attention up. With
    Hamilton-product real-part scoring, (i, j, k) components
    *subtract* — geometrically, anti-aligned imaginary parts reduce
    the score even when the magnitudes are large. For signed graphs
    where the (i, j, k) axes can carry sign / phase information, this
    asymmetry is what we want.

    Implementation note: the layout is (E, n_quaternions, 4) where
    the last axis is the (real, i, j, k) ordering. Init scale 0.01
    keeps initial scores near zero so softmax starts ≈ uniform —
    same warm-start as :class:`_AttentionM_e`.
    """
    def __init__(self, d_query: int, d_cycle: int, d_attn: int = 32,
                 n_heads: int = 1):
        super().__init__()
        if d_attn % 4 != 0:
            raise ValueError(
                f"_QuaternionAttentionM_e requires d_attn % 4 == 0, "
                f"got d_attn={d_attn}",
            )
        self.d_attn = d_attn
        self.n_quat = d_attn // 4
        if self.n_quat % n_heads != 0:
            raise ValueError(
                f"_QuaternionAttentionM_e: n_quat ({self.n_quat}) must be "
                f"divisible by n_heads ({n_heads}); pick d_attn = 4 · n_heads · k"
            )
        self.n_heads = n_heads
        self.n_quat_per_head = self.n_quat // n_heads
        self.W_q = nn.Linear(d_query, d_attn, bias=False)
        self.W_k = nn.Linear(d_cycle, d_attn, bias=False)
        with torch.no_grad():
            self.W_q.weight.mul_(0.01)
            self.W_k.weight.mul_(0.01)
        self.scale = self.n_quat_per_head ** -0.5

    def forward(self, h_query: torch.Tensor, h_cycle: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        rows = indices[0]
        cols = indices[1]
        # Project to (E, n_heads, n_quat_per_head, 4) and (T, ...).
        q = self.W_q(h_query).view(
            -1, self.n_heads, self.n_quat_per_head, 4,
        )
        k = self.W_k(h_cycle).view(
            -1, self.n_heads, self.n_quat_per_head, 4,
        )
        qg = q[rows]                    # (nnz, n_heads, n_quat_per_head, 4)
        kg = k[cols]
        # Hamilton-product real component per head, summed over the
        # head's quaternion blocks: (nnz, n_heads).
        scores = (
            qg[..., 0] * kg[..., 0]
            - qg[..., 1] * kg[..., 1]
            - qg[..., 2] * kg[..., 2]
            - qg[..., 3] * kg[..., 3]
        ).sum(dim=-1) * self.scale
        if self.n_heads == 1:
            return _attn_softmax_dispatch(scores.squeeze(-1), rows,
                                            h_query.shape[0])
        attn_heads = torch.stack(
            [
                _attn_softmax_dispatch(scores[:, h], rows,
                                        h_query.shape[0])
                for h in range(self.n_heads)
            ],
            dim=-1,
        )
        return attn_heads.mean(dim=-1)


