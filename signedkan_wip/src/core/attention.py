"""Signed triad attention + entropy-guided attention regulariser.

Replaces the current mean-pool aggregation
$\\mathbf{h}_e = \\frac{1}{|T_e|} \\sum_{t \\in T_e} \\mathbf{h}_t$
(implemented as the sparse mat-mul ``M_train @ triad_emb``) with a
learned signed-weight attention:

    score(e, t) = tanh( a^T [h_u; h_v; h_t] )
    h_e         = sum_{t in T_e} score(e, t) * h_t

The score is a *tanh*, not a softmax: balanced triads can vote
for the prediction with positive weight, unbalanced ones against
with negative weight, and the magnitude is what the model learns.
This generalises GAT to signed hyperedges.

To prevent attention collapse onto one triad per edge we add an
entropy regulariser on the per-edge attention distribution:

    p_{e, t}     = |score(e, t)| / sum_{t'} |score(e, t')|
    H(alpha_e)   = -sum_t p_{e, t} log p_{e, t}
    L_attn-ent   = -lam_ae * mean_e H(alpha_e)

Maximising the per-edge entropy keeps attention spread and stays
within the project's entropy-feedback family.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AttentionConfig:
    use_attention: bool = False
    entropy_lam: float = 0.0           # weight of the attention-entropy term
    score_init_scale: float = 0.01     # init scale of the score Linear
    eps: float = 1e-8


def build_attention_pairs(edges_array: np.ndarray,
                          edge_to_triads: dict) -> tuple[torch.Tensor,
                                                          torch.Tensor]:
    """Build flat (pair_edge_idx, pair_triad_idx) arrays.

    For each edge $e$ and each triad $t \\in T_e$ incident to $e$,
    emit one pair. Empty-incidence edges are skipped (they have no
    attention contribution; the model uses a zero embedding for them).

    Returns
    -------
    pair_edge_idx  : (n_pairs,) long tensor — edge index per pair
    pair_triad_idx : (n_pairs,) long tensor — triad index per pair
    """
    rows, cols = [], []
    for ei, e in enumerate(edges_array):
        key = (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1])))
        for t in edge_to_triads.get(key, []):
            rows.append(ei)
            cols.append(int(t))
    return (torch.tensor(rows, dtype=torch.long),
            torch.tensor(cols, dtype=torch.long))


class SignedTriadAttention(nn.Module):
    """tanh-attention over edge-triad incidences.

    The triad-embedding feature dimension $d_t$ may differ from the
    vertex-embedding feature dimension $d_v$ (e.g., when the upstream
    model uses JK-concat across $L$ layers, $d_t = L \\cdot d_v$).
    """

    def __init__(self, d: int, d_t: int | None = None,
                 init_scale: float = 0.01):
        super().__init__()
        d_t = d if d_t is None else d_t
        # score : (B, 2*d_v + d_t) -> (B, 1).
        self.attn = nn.Linear(2 * d + d_t, 1)
        with torch.no_grad():
            self.attn.weight.mul_(init_scale)
            self.attn.bias.zero_()

    def forward(self, h_v: torch.Tensor,
                triad_emb: torch.Tensor,
                edge_endpoints: torch.Tensor,    # (n_edges, 2)
                pair_edge_idx: torch.Tensor,     # (n_pairs,)
                pair_triad_idx: torch.Tensor,    # (n_pairs,)
                n_edges: int):
        """Returns (edge_emb, raw_scores).

        edge_emb   : (n_edges, d_t)  — same feature dim as triad_emb
        raw_scores : (n_pairs,)      — signed attention scores per incidence
        """
        d_t = triad_emb.shape[1]
        # Endpoint embeddings per pair.
        h_u = h_v[edge_endpoints[pair_edge_idx, 0]]      # (n_pairs, d_v)
        h_w = h_v[edge_endpoints[pair_edge_idx, 1]]      # (n_pairs, d_v)
        h_t = triad_emb[pair_triad_idx]                  # (n_pairs, d_t)
        cat = torch.cat([h_u, h_w, h_t], dim=-1)         # (n_pairs, 2d_v + d_t)
        scores = torch.tanh(self.attn(cat).squeeze(-1))  # (n_pairs,)
        # Aggregate per edge: edge_emb[e] = sum_pair scores * h_t.
        weighted = scores.unsqueeze(-1) * h_t            # (n_pairs, d_t)
        out = torch.zeros(n_edges, d_t,
                          device=h_v.device, dtype=h_v.dtype)
        out.index_add_(0, pair_edge_idx, weighted)
        return out, scores


def attention_entropy_loss(scores: torch.Tensor,
                           pair_edge_idx: torch.Tensor,
                           n_edges: int,
                           eps: float = 1e-8) -> torch.Tensor:
    """Per-edge attention entropy (treating ``|score|`` as a
    distribution over the edge's incident triads), averaged over edges.

    Returns a scalar tensor; the regulariser pre-multiplies by
    $-\\lambda_{ae}$ so that *minimising* the loss *maximises* per-edge
    entropy (preventing one-triad collapse).
    """
    abs_s = scores.abs() + eps
    sum_per_edge = torch.zeros(n_edges, device=scores.device,
                                dtype=scores.dtype)
    sum_per_edge.index_add_(0, pair_edge_idx, abs_s)
    sum_per_edge = sum_per_edge.clamp_min(eps)
    p = abs_s / sum_per_edge[pair_edge_idx]
    ent_contrib = -p * torch.log(p.clamp_min(eps))
    H_per_edge = torch.zeros(n_edges, device=scores.device,
                              dtype=scores.dtype)
    H_per_edge.index_add_(0, pair_edge_idx, ent_contrib)
    # Edges with no triads contribute 0 — we average over edges that have
    # at least one pair; for simplicity we take a plain mean which under-
    # weights empty edges. That's fine for the regulariser purpose.
    return H_per_edge.mean()
