"""Aggregators for the upgraded RicciStim backbone.

The original backbone combined its three Bochner branches by a bare sum and fed
the result straight to the head — no learned mixing, no residual/highway, no
explicit cross-scale fusion. That (falsified) configuration is the weakest
aggregator. These three small modules port what the signed-link Gömb side
already uses, behind opt-in flags so the bare-sum baseline stays reproducible:

  * LearnedBranchMixer — learned-αₖ softmax-weighted branch combine
    (mirrors the signed-link `arity_mixer`);
  * HighwayGate        — gated residual with the encoder features as the skip,
    so the net can fall back to the (working) patch encoder;
  * CrossScalePyramid  — top-down parent→child fusion over the anchor tree,
    fusing scales that today interact only through graph edges.

Each module preserves the (n_anchors, d) shape. No new dependency.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedBranchMixer(nn.Module):
    """Softmax-weighted sum of `n_branches` (n, d) tensors with a learned α.

    Preconditions: `branches` is a list of `n_branches` tensors, each (n, d).
    Postconditions: returns (n, d) = Σ_k softmax(α)_k · branches[k]. With α=0
    (default init) this is the *uniform mean*, not the bare sum — a strictly
    better-conditioned starting point that the optimiser can reshape.
    """

    def __init__(self, n_branches: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(n_branches))

    def weights(self) -> torch.Tensor:
        return F.softmax(self.alpha, dim=0)

    def forward(self, branches: list[torch.Tensor]) -> torch.Tensor:
        if len(branches) != self.alpha.shape[0]:
            raise ValueError(
                f"expected {self.alpha.shape[0]} branches, got {len(branches)}"
            )
        w = F.softmax(self.alpha, dim=0)
        out = branches[0] * w[0]
        for k in range(1, len(branches)):
            out = out + branches[k] * w[k]
        return out


class HighwayGate(nn.Module):
    """Highway-gated residual: y = T ⊙ H(m) + (1 − T) ⊙ s.

    `m` is the mixed branch output, `s` the encoder skip; T = σ(W_T[m; s]) is a
    per-channel gate, H a linear transform of `m`. The gate bias is initialised
    negative so training starts close to carrying the skip (stable), then opens
    the hypergraph path only where it helps.
    Preconditions: `m`, `skip` both (n, d). Postconditions: (n, d).
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.transform = nn.Linear(d, d)
        self.gate = nn.Linear(2 * d, d)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, m: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(self.gate(torch.cat([m, skip], dim=-1)))
        return t * self.transform(m) + (1.0 - t) * skip


class CrossScalePyramid(nn.Module):
    """Top-down parent→child fusion over the anchor tree.

    Processes anchors coarse→fine (ascending scale); each anchor at scale L gets
    a learned projection of its (already-updated) parent's feature added:
        f[i] ← f[i] + W_up(f[parent[i]]).
    Implemented out-of-place per level (`index_add`) so it is autograd-safe.
    A scale-0-only tree (all parent_indices = −1) is a no-op.
    Preconditions: feat (n, d); parent_indices, scales are LongTensor (n,).
    Postconditions: (n, d).
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.up = nn.Linear(d, d)

    def forward(
        self, feat: torch.Tensor, parent_indices: torch.Tensor, scales: torch.Tensor,
    ) -> torch.Tensor:
        n = feat.shape[0]
        if n == 0:
            return feat
        out = feat
        max_scale = int(scales.max().item())
        for level in range(1, max_scale + 1):
            mask = scales == level
            if not bool(mask.any()):
                continue
            idx = mask.nonzero(as_tuple=False).squeeze(1)        # children at this scale
            pidx = parent_indices.index_select(0, idx).clamp(min=0)
            parent_feat = out.index_select(0, pidx)              # already-updated parents
            out = out.index_add(0, idx, self.up(parent_feat))    # out-of-place, autograd-safe
        return out
