"""Phase 3 baseline: sign-blind vanilla KAN.

Operates on aggregated edge features, no sign-awareness in the
representation. Used as the contrast against SignedKAN. The
expectation is that vanilla KAN matches or exceeds SignedKAN on AUC
(positive class is dominant) but underperforms on macro-F1
(class-balanced score where sign asymmetry matters).

This is a STUB: the full implementation reuses
`src.splines.BSplineActivation` and replaces the inner+outer
sign-conditioned splines with a single spline pair that aggregates
all triad members regardless of sign. Drops in tomorrow morning,
Phase 3.5.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..splines import BSplineActivation


class VanillaKAN(nn.Module):
    """KAN with a single inner+outer spline pair, no sign conditioning."""
    def __init__(self, n_nodes: int, hidden_dim: int = 32,
                 grid: int = 5, k: int = 3):
        super().__init__()
        self.node_embed = nn.Embedding(n_nodes, hidden_dim)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.inner = BSplineActivation(hidden_dim, grid, k)
        self.outer = BSplineActivation(hidden_dim, grid, k)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def encode_triads(self, triad_v: torch.Tensor,
                      triad_sigma: torch.Tensor,
                      return_h_v: bool = False):
        """Sign-blind: ignore triad_sigma entirely."""
        x = self.node_embed.weight
        T = triad_v.shape[0]
        d = self.hidden_dim
        h_v = x[triad_v]                                 # (T, 3, d)
        inner = self.inner(h_v.reshape(-1, d)).view(T, 3, d)
        agg = inner.mean(dim=1)                          # mean across triad
        out = self.outer(agg)
        return (out, x) if return_h_v else out

    def predict_edge_sign(self, triad_emb: torch.Tensor,
                          edge_to_triads: list[list[int]]) -> torch.Tensor:
        d = self.hidden_dim
        out = []
        for tri_ids in edge_to_triads:
            emb = triad_emb[tri_ids].mean(dim=0) if tri_ids \
                  else triad_emb.new_zeros(d)
            out.append(self.classifier(emb).squeeze(-1))
        return torch.stack(out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
