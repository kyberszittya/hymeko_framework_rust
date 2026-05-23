"""Iterative pruning + retraining utilities.

After training a SignedKAN, run:
  1. Prune at threshold $\\tau_1$
  2. Fine-tune for $E_r$ epochs holding the pruned splines at zero
     (mask is applied after each ``opt.step()``)
  3. Prune at threshold $\\tau_2 > \\tau_1$
  4. Fine-tune again
  5. (...repeat)

The retrain phase lets the surviving splines specialise to cover the
function previously absorbed by the pruned ones, often pushing AUC
above the post-hoc pruned model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .prune_distill import measure_activity


class PruneMask:
    """Boolean masks (per batched-spline coef tensor) that stay at zero
    once a coefficient vector is pruned."""

    def __init__(self, model: nn.Module):
        self.masks: dict[str, torch.Tensor] = {}
        for name, m in model.named_modules():
            if hasattr(m, "coef") and isinstance(m.coef, nn.Parameter) \
                    and m.coef.dim() == 3:
                self.masks[name] = torch.ones_like(m.coef)

    def update_from_threshold(self, model: nn.Module,
                              threshold: float) -> int:
        """Compute a fresh mask: zero out any (branch, channel) whose
        coefficient vector has L2-norm < threshold. Returns the
        cumulative number of zeroed (branch, channel) entries."""
        n_zero = 0
        for name, m in model.named_modules():
            if name not in self.masks:
                continue
            norms = m.coef.detach().pow(2).sum(dim=-1).sqrt()  # (S, C)
            keep = (norms >= threshold).to(m.coef.dtype)        # (S, C)
            mask = keep.unsqueeze(-1).expand_as(m.coef)
            self.masks[name] = mask
            n_zero += int((keep == 0).sum().item())
        return n_zero

    def apply(self, model: nn.Module) -> None:
        """Force pruned coefficients to zero. Call after each
        optimiser step."""
        for name, m in model.named_modules():
            if name in self.masks:
                with torch.no_grad():
                    m.coef.mul_(self.masks[name])

    def total_pruned(self) -> tuple[int, int]:
        """Returns (zeroed, total) at the (branch, channel) level."""
        z = 0; t = 0
        for mask in self.masks.values():
            # mask is (S, C, N) but constant along N; collapse first.
            keep_2d = mask[..., 0]
            t += keep_2d.numel()
            z += int((keep_2d == 0).sum().item())
        return z, t


def count_active_splines(model: nn.Module) -> tuple[int, int]:
    """For a model with batched-spline activations, count
    (active, total) (branch, channel) splines (active = nonzero
    coefficient vector L2-norm)."""
    active = 0; total = 0
    for m in model.modules():
        if hasattr(m, "coef") and isinstance(m.coef, nn.Parameter) \
                and m.coef.dim() == 3:
            norms = m.coef.detach().pow(2).sum(dim=-1).sqrt()
            active += int((norms > 1e-12).sum().item())
            total  += int(norms.numel())
    return active, total
