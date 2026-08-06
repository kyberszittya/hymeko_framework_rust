"""SignedKAN — hypergraph tuple loss (Phase 4.x).

Generalises the SGCN edge-triplet loss to operate on signed
*hyperedges* (triads). For each triad $t$ with vertex set $V_t$ and
pairwise edge signs $\\sigma_{ij}$ inherited from the original
signed graph, define a signed coherence score:

    s(t) = sum_{(i, j) in pairs(V_t)} sigma_{ij} * (h_i · h_j) / (||h_i|| ||h_j||)

and a Cartwright-Harary balance indicator
$\\beta_t = \\sigma_{12} \\sigma_{23} \\sigma_{13} \\in \\{+1, -1\\}$.
The triad-level margin loss is

    L_triad = (1 / |T|) * sum_t  relu(margin - beta_t * s(t))

Balanced triads ($\\beta_t = +1$) want $s(t) > m$; unbalanced
($\\beta_t = -1$) want $s(t) < -m$. The structural prior of
Cartwright-Harary is encoded in the loss, where SGCN keeps it ---
not just in the architecture's sign-conditioned splines.

Used as `total_loss = L_BCE + alpha * L_triad` with alpha swept;
pure-contrastive ablation runs alpha=inf (BCE term suppressed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TriadLossConfig:
    margin: float = 0.5         # m in relu(m - beta * s)
    alpha: float  = 1.0         # weight in total loss
    eps: float    = 1e-8        # numerical floor on ||h||


def build_triad_pairs(triads_pyobj) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute the pair-level tensors used by ``TriadLoss``.

    For each triad return three pairs (i, j, sigma_ij) for the three
    edges {(v0,v1), (v1,v2), (v0,v2)} together with the balance
    indicator beta_t = product of edge signs.

    Returns
    -------
    pair_idx   : (T, 3, 2)  int64 — vertex pairs per triad
    pair_sign  : (T, 3)     float — +/-1 per pair
    beta       : (T,)       float — +/-1 per triad
    """
    T = len(triads_pyobj)
    pair_idx  = np.empty((T, 3, 2), dtype=np.int64)
    pair_sign = np.empty((T, 3),    dtype=np.float32)
    beta      = np.empty((T,),      dtype=np.float32)
    for t, tri in enumerate(triads_pyobj):
        v0, v1, v2 = tri.v
        s01, s12, s02 = tri.edge_signs        # ordering matches SignedTriad
        pair_idx[t, 0] = (v0, v1); pair_sign[t, 0] = float(s01)
        pair_idx[t, 1] = (v1, v2); pair_sign[t, 1] = float(s12)
        pair_idx[t, 2] = (v0, v2); pair_sign[t, 2] = float(s02)
        beta[t] = float(s01 * s12 * s02)      # +1 if balanced
    return (torch.from_numpy(pair_idx),
            torch.from_numpy(pair_sign),
            torch.from_numpy(beta))


class TriadLoss(nn.Module):
    """Hypergraph tuple loss on signed triads. See module docstring."""
    def __init__(self, cfg: TriadLossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, h: torch.Tensor,
                pair_idx: torch.Tensor,
                pair_sign: torch.Tensor,
                beta: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h         : (V, d) node embeddings
        pair_idx  : (T, 3, 2)  vertex-pair indices per triad
        pair_sign : (T, 3)     +/-1 sign per pair
        beta      : (T,)       +/-1 balance indicator per triad

        Returns
        -------
        scalar loss = mean over triads of relu(margin - beta * s(t))
        """
        # Gather embeddings: (T, 3, 2, d)
        h_pairs = h[pair_idx]
        h_i = h_pairs[..., 0, :]                              # (T, 3, d)
        h_j = h_pairs[..., 1, :]                              # (T, 3, d)
        # L2 normalise (numerically safe).
        eps = self.cfg.eps
        h_i_n = h_i / (h_i.norm(dim=-1, keepdim=True) + eps)
        h_j_n = h_j / (h_j.norm(dim=-1, keepdim=True) + eps)
        # Per-pair signed cosine.
        cos_pair = (h_i_n * h_j_n).sum(dim=-1)                # (T, 3)
        # Coherence score s(t): sum over pairs, weighted by sign.
        s = (pair_sign * cos_pair).sum(dim=-1)                # (T,)
        # Margin loss.
        return torch.relu(self.cfg.margin - beta * s).mean()
