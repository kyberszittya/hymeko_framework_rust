"""Gömb — the residual stream on the unit hypersphere S^{d-1}.

Token embeddings are projected to the sphere; residual updates retract back onto
it (a normalised-transformer regime). This is the Gömb half of the architecture;
the Gömb-strict no-leakage audit (a practice, not code) is carried over as the
evaluation discipline at Phase 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def l2_normalize(x: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Project the last axis onto the unit sphere.

    # Preconditions ``x`` has ``shape[-1] >= 1``.
    # Postconditions ``||out||_2 == 1`` along the last axis (to fp tolerance), except
    where the input norm is below ``eps`` (then it is left near-zero, not NaN).
    """
    out: torch.Tensor = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return out


def spherical_residual(h: torch.Tensor, delta: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Spherical residual update: add the sub-layer output, retract to the sphere.

    # Preconditions ``h`` and ``delta`` broadcast-compatible; both on S^{d-1} scale.
    # Postconditions output is on the unit sphere along the last axis.
    """
    return l2_normalize(h + delta, eps=eps)


class SphereEmbedding(nn.Module):
    """Token embedding whose rows are read off the unit hypersphere.

    # Preconditions ``vocab_size, d_model >= 1``; ``forward`` ids are ``long`` in ``[0, vocab)``.
    # Postconditions ``forward(ids: (B,T)) -> (B,T,d_model)`` with unit-norm rows.
    """

    def __init__(self, vocab_size: int, d_model: int, *, normalize: bool = True) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, d_model)
        self.normalize = normalize

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.table(ids)
        return l2_normalize(emb) if self.normalize else emb
