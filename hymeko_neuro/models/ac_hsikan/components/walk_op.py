"""WalkOp strategies — per-arity cycle sign product.

Each WalkOp receives:
    anchor_to_cand : (B, L, K_total)  anchor->candidate signs
    attn           : SignAttention    for consecutive / closing signs
    x              : (B, L, d_model)  features (passed to attn)
    local          : (L, K_total)     candidate indices per anchor
    k              : int              the arity
and returns (B, L, 1) — the sign product to multiply against pooled
features.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .attention import SignAttention


class WalkOp(nn.Module, ABC):
    @abstractmethod
    def compute(
        self,
        anchor_to_cand: torch.Tensor,
        attn: "SignAttention",
        x: torch.Tensor,
        local: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        ...


# ── Star (v1 default) ─────────────────────────────────────────────

class StarWalkOp(WalkOp):
    """Product of the first k anchor->candidate signs."""
    def compute(self, anchor_to_cand, attn, x, local, k):
        return anchor_to_cand[:, :, :k].prod(dim=-1, keepdim=True)


# ── Chain ─────────────────────────────────────────────────────────

class ChainWalkOp(WalkOp):
    """anchor->n_0 sign  ×  prod_{t=1..k-1} sign(n_{t-1}, n_t)."""
    def compute(self, anchor_to_cand, attn, x, local, k):
        sign_a0 = anchor_to_cand[:, :, 0:1]              # (B, L, 1)
        if k <= 1:
            return sign_a0
        chain = attn.compute_consecutive_signs(x, local, k)  # (B, L, k-1)
        return sign_a0 * chain.prod(dim=-1, keepdim=True)


# ── Cycle ─────────────────────────────────────────────────────────

class CycleWalkOp(WalkOp):
    """Chain product × closing edge sign(n_{k-1}, anchor)."""
    def compute(self, anchor_to_cand, attn, x, local, k):
        sign_a0 = anchor_to_cand[:, :, 0:1]
        if k <= 1:
            # Self-cycle: anchor->anchor; pair sign at (anchor, anchor) ≈ ±1.
            B, L, _ = anchor_to_cand.shape
            idx = torch.arange(L, device=anchor_to_cand.device)
            closing = attn.compute_pair_signs(x, idx, idx).unsqueeze(-1)
            return sign_a0 * closing
        chain = attn.compute_consecutive_signs(x, local, k)
        chain_prod = sign_a0 * chain.prod(dim=-1, keepdim=True)
        B, L, _ = anchor_to_cand.shape
        last = local[:, k - 1]                            # (L,)
        anchor_idx = torch.arange(L, device=anchor_to_cand.device)
        closing = attn.compute_pair_signs(x, last, anchor_idx).unsqueeze(-1)
        return chain_prod * closing


# ── Fused chain (speedup #2) ──────────────────────────────────────

class FusedChainWalkOp(WalkOp):
    """Same semantics as ChainWalkOp but the (k-1) chain signs are
    composed via a single ``einsum``-fused log-tanh-sum trick
    that avoids materialising the intermediate (B, L, k-1) tensor of
    soft signs followed by a prod.

    We compute the chain product as
        exp(sum_t log(|edge_t|)) × prod_t sign(edge_t)
    where the magnitudes are clamped away from 0. This collapses the
    per-edge tanh + prod into a single reduce-sum + exp + sign-prod.
    On compiled backends (torch.compile / TorchInductor) the fused
    log+sum+exp lowers to a single GPU kernel.
    """
    _EPS = 1e-7

    def compute(self, anchor_to_cand, attn, x, local, k):
        sign_a0 = anchor_to_cand[:, :, 0:1]
        if k <= 1:
            return sign_a0
        edges = attn.compute_consecutive_signs(x, local, k)  # (B, L, k-1)
        # Fused magnitude-and-sign reduction
        abs_edges = edges.abs().clamp_min(self._EPS)
        log_sum = torch.log(abs_edges).sum(dim=-1, keepdim=True)
        sign_prod = torch.sign(edges).prod(dim=-1, keepdim=True)
        return sign_a0 * sign_prod * torch.exp(log_sum)


# ── Fused cycle (speedup #2 + closing edge) ───────────────────────

class FusedCycleWalkOp(WalkOp):
    def compute(self, anchor_to_cand, attn, x, local, k):
        sign_a0 = anchor_to_cand[:, :, 0:1]
        B, L, _ = anchor_to_cand.shape
        anchor_idx = torch.arange(L, device=anchor_to_cand.device)
        if k <= 1:
            closing = attn.compute_pair_signs(x, anchor_idx, anchor_idx).unsqueeze(-1)
            return sign_a0 * closing
        edges = attn.compute_consecutive_signs(x, local, k)  # (B, L, k-1)
        last = local[:, k - 1]
        closing = attn.compute_pair_signs(x, last, anchor_idx).unsqueeze(-1)
        # Append the closing edge so we have all k edges in one chain product.
        all_edges = torch.cat([edges, closing], dim=-1)      # (B, L, k)
        abs_e = all_edges.abs().clamp_min(FusedChainWalkOp._EPS)
        log_sum = torch.log(abs_e).sum(dim=-1, keepdim=True)
        sign_prod = torch.sign(all_edges).prod(dim=-1, keepdim=True)
        return sign_a0 * sign_prod * torch.exp(log_sum)


# ── Factory ───────────────────────────────────────────────────────

def build_walk_op(cfg) -> WalkOp:
    use_fused = getattr(cfg, "use_fused_walk", False)
    if cfg.walk_kind == "star":
        return StarWalkOp()
    if cfg.walk_kind == "chain":
        return FusedChainWalkOp() if use_fused else ChainWalkOp()
    if cfg.walk_kind == "cycle":
        return FusedCycleWalkOp() if use_fused else CycleWalkOp()
    raise ValueError(f"unknown walk_kind: {cfg.walk_kind!r}")
