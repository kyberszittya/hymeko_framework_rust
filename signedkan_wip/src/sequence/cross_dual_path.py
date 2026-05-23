"""Cross-attention dual-path block for the Sequential HSiKAN
encoder-decoder.

v2 (2026-05-17): multi-channel support via ``n_channels`` kwarg.
At ``n_channels=1`` the path is byte-identical to v1 (output tap
init to scalar identity makes the final geometric product
identity).

For each decoder query q (channels c) and encoder key k:

    score_{t, s} = Σ_c <q_t[c], k_s[c]> / sqrt(4 C)
    a_{t, s}     = softmax_s(score_{t, s})
    ctx_t[c]     = Σ_s a_{t, s} k_s[c]            (per-channel context)
    y_sig_t[c]   = q_t[c] ⊗ ctx_t[c]              (per-channel geo-prod)
    y_info_t[c]  = (sum over +mask k_s[c]) - (sum over -mask k_s[c])
    g_t          = sigmoid(router features)        (one gate per token)
    y_t          = g_t y_sig + (1-g_t) y_info,    output tap applied per channel

Plan: docs/plans/2026-05-17-sequence-multichannel-v2/.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clifford import CL2_DIM, geometric_product
from .dual_router import _local_spectrum


def _mv_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cl(2,0) inner product = Euclidean inner product on the 4-component
    vector. Sum over the trailing CL2_DIM axis (no channel axis here)."""
    return (a * b).sum(dim=-1)


class CrossDualPathBlock(nn.Module):
    """Cross-attention dual-path block.

    Parameters
    ----------
    n_channels : int, default 1
        Multivector channel count. At ``n_channels=1`` the block is
        byte-identical (numerically) to v1 with the output tap at
        scalar identity init.

    Input shapes (q is the query stream, k is the key stream):
        n_channels = 1:
            q : (B, L_q, 4)        decoder query multivector stream
            k : (B, L_k, 4)        encoder context multivector stream
        n_channels > 1:
            q : (B, L_q, C, 4)
            k : (B, L_k, C, 4)
        sigma_q : (B, L_q)         decoder sign stream
        sigma_k : (B, L_k)         encoder sign stream

    Output shape:
        y matches q.
    """

    def __init__(self, n_channels: int = 1) -> None:
        super().__init__()
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1; got {n_channels}")
        self.n_channels = int(n_channels)
        n_features = 2 + 4  # ||q_t|| (channel-mean), |σ_q_t|, local spec (4 bins)
        self.router = nn.Linear(n_features, 1)
        nn.init.normal_(self.router.weight, std=0.1)
        nn.init.zeros_(self.router.bias)
        # Per-channel output projection multivector tap.
        out_tap = torch.zeros(self.n_channels, CL2_DIM)
        out_tap[:, 0] = 1.0  # scalar-1 init = identity per channel
        self.out_tap = nn.Parameter(out_tap)

    def forward(
        self,
        q: torch.Tensor, k: torch.Tensor,
        sigma_q: torch.Tensor, sigma_k: torch.Tensor,
    ) -> torch.Tensor:
        squeezed = False
        if q.dim() == 3:
            if self.n_channels != 1:
                raise ValueError(
                    f"q is (B, L_q, 4) but n_channels={self.n_channels}"
                )
            q = q.unsqueeze(2)  # (B, L_q, 1, 4)
            k = k.unsqueeze(2)  # (B, L_k, 1, 4)
            squeezed = True
        if q.shape[-1] != CL2_DIM or k.shape[-1] != CL2_DIM:
            raise ValueError("cross-attn requires Cl(2,0) trailing dim 4")
        if q.dim() != 4 or k.dim() != 4:
            raise ValueError(
                f"expected (B, L, C, 4); got q={tuple(q.shape)}, k={tuple(k.shape)}"
            )
        if q.shape[0] != k.shape[0]:
            raise ValueError("batch dims must match")
        if q.shape[2] != self.n_channels or k.shape[2] != self.n_channels:
            raise ValueError(
                f"channel dim mismatch: q={q.shape[2]}, k={k.shape[2]}, "
                f"layer expects {self.n_channels}"
            )
        B, L_q, C, _ = q.shape
        _, L_k, _, _ = k.shape

        # ─── Signal branch: scaled-dot multivector cross-attn ─────
        # Scores: sum across channels and multivector dims.
        scores = torch.einsum("bqcd,bkcd->bqk", q, k) / ((CL2_DIM * C) ** 0.5)
        attn = F.softmax(scores, dim=-1)                  # (B, L_q, L_k)
        # Per-channel weighted multivector context.
        ctx = torch.einsum("bqk,bkcd->bqcd", attn, k)     # (B, L_q, C, 4)
        # Per-channel geometric product q_t[c] ⊗ ctx_t[c].
        y_sig = geometric_product(q, ctx)                  # (B, L_q, C, 4)

        # ─── Info branch: σ-masked cross-pool ─────────────────────
        sq = sigma_q.unsqueeze(-1)                         # (B, L_q, 1)
        sk = sigma_k.unsqueeze(1)                          # (B, 1, L_k)
        joint = sq * sk                                    # (B, L_q, L_k)
        pos_mask = (joint > 0.5).float()
        neg_mask = (joint < -0.5).float()
        cnt_pos = pos_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        cnt_neg = neg_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        # Pool encoder keys across encoder positions; keep channel/multivector dims.
        # Reshape masks to broadcast: (B, L_q, L_k) → (B, L_q, L_k, 1, 1)
        a_pos = torch.einsum("bqk,bkcd->bqcd", pos_mask, k) / cnt_pos.unsqueeze(-1)
        a_neg = torch.einsum("bqk,bkcd->bqcd", neg_mask, k) / cnt_neg.unsqueeze(-1)
        y_info = a_pos - a_neg                             # (B, L_q, C, 4)

        # ─── Router ──────────────────────────────────────────────
        # Channel-mean of multivector norm for the router feature.
        q_norm_per_chan = torch.linalg.norm(q, dim=-1)     # (B, L_q, C)
        q_norm = q_norm_per_chan.mean(dim=-1)              # (B, L_q)
        sigma_mag = sigma_q.abs()
        local = _local_spectrum(q_norm, n_bins=4)
        feats = torch.cat([
            q_norm.unsqueeze(-1),
            sigma_mag.unsqueeze(-1),
            local,
        ], dim=-1)                                          # (B, L_q, 6)
        g = torch.sigmoid(self.router(feats).squeeze(-1))  # (B, L_q)
        g_e = g.unsqueeze(-1).unsqueeze(-1)                # (B, L_q, 1, 1)
        y_mix = g_e * y_sig + (1.0 - g_e) * y_info         # (B, L_q, C, 4)
        # Per-channel output projection.
        tap_e = self.out_tap.view(1, 1, C, CL2_DIM).expand_as(y_mix)
        y_out = geometric_product(tap_e, y_mix)            # (B, L_q, C, 4)

        if squeezed:
            y_out = y_out.squeeze(2)                       # (B, L_q, 4)
        return y_out
