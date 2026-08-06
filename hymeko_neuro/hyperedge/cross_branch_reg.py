"""Cross-branch information regulariser for SignedKAN.

Encourages the per-sign-branch spline coefficient tensors to encode
*distinct* functions, not redundant copies. Adds the term

    L_cross = (1 / |pairs|) * sum_{i<j} | cos(coef^{(i)}_{c,:}, coef^{(j)}_{c,:}) |

averaged over channels c, summed over the inner and outer
batched-spline modules. Combined with the existing
``EntropyRegulariser`` (spectral entropy + KL feedback) this gives a
two-axis regularisation: *spectral* concentration on the node
embeddings and *functional* diversity across the sign branches.

Stays in the entropy-feedback family (no new optimisation tricks):
the KL-feedback schedule from the existing entropy regulariser
modulates how strongly all entropy-side terms fire per step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CrossBranchRegConfig:
    lam: float = 0.01           # weight in total loss
    eps: float = 1e-8


def _branch_cosine(coef: torch.Tensor, eps: float) -> torch.Tensor:
    """Mean pairwise |cosine| between branches of a coef tensor.

    coef : (S, C, N)  per-(branch, channel) coefficient vector
    Returns scalar in [0, 1].
    """
    S, C, N = coef.shape
    if S < 2:
        return coef.new_zeros(())
    # L2-normalise per (S, C, :) row.
    norms = coef.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    coef_n = coef / norms                              # (S, C, N)
    # Pairwise dot products.
    sims = []
    for i in range(S):
        for j in range(i + 1, S):
            cos = (coef_n[i] * coef_n[j]).sum(dim=-1).abs().mean()
            sims.append(cos)
    return torch.stack(sims).mean()


class CrossBranchRegulariser(nn.Module):
    """Cross-branch coefficient cosine penalty. Apply to every batched
    spline activation module in the model; sum the contributions."""

    def __init__(self, cfg: CrossBranchRegConfig):
        super().__init__()
        self.cfg = cfg
        self.last_value: float = float("nan")

    def forward(self, model) -> torch.Tensor:
        """Sum the cross-branch cosine terms across every batched spline
        in the model. Walks the model's submodules; picks up any module
        with a ``coef`` parameter of rank 3 (S, C, N)."""
        terms = []
        for m in model.modules():
            if hasattr(m, "coef") and isinstance(m.coef, nn.Parameter) \
                    and m.coef.dim() == 3:
                terms.append(_branch_cosine(m.coef, self.cfg.eps))
        if not terms:
            return next(model.parameters()).new_zeros(())
        out = torch.stack(terms).mean()
        self.last_value = float(out.detach().item())
        return self.cfg.lam * out
