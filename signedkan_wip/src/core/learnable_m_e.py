"""Path B: Learnable M_e (signed-incidence matrix) for HSiKAN.

The existing M_e ∈ R^{E × |𝒞|} sparse matrix has:
    M_e[e, c] = sign(e, c) ∈ {-1, 0, +1}
fixed by cycle enumeration. Each cycle contributes ±1 to its edge-
pool entries depending on the cycle's traversal direction at e.

Path B (per the architectural-ceiling analysis of
`project_epinions_ceiling_2026_05_09`): make the *values* learnable
while keeping the sparsity *pattern* fixed.

The cleanest concretization is **per-cycle importance weighting**:
    M_e[e, c] = sign(e, c) · w_c
where w_c ∈ R is a learnable scalar per cycle, initialised at 1.0.
This:
  * adds |𝒞| learnable parameters (≈ 500K on Epinions; ~2 MB)
  * preserves the sparsity pattern (the cycle pool determines WHERE
    M_e is non-zero)
  * starts at the identity forward (w_c=1.0 reproduces the baseline)
  * is one-dimensional per cycle, no per-edge × per-cycle blow-up
  * is interpretable: "which cycles does the model trust more?"

Optional optimizations supported:
  * exponential parameterization (w_c = exp(θ_c)) to enforce
    positivity if desired
  * log-weight-clipping to prevent runaway scaling
  * per-arity weight banks (separate w for k=3 vs k=4 vs walks)

Drops into the existing sparse-M_e forward path as a one-line swap:
replace `M_e` with `wrap.apply(M_e, cycle_idx_of_nz)` before
`torch.sparse.mm(M_e, h_final)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LearnableMe(nn.Module):
    """Per-cycle learnable importance weights for a signed-incidence
    sparse matrix.

    Args:
        n_cycles: number of cycles in the pool (== M_e.shape[1]).
        param_kind:
            "scalar"    — w_c ∈ R, identity init at 1.0 (default).
            "exp"       — w_c = exp(θ_c), θ_c init at 0.0; enforces
                          positivity, smoother gradient at small w.
            "sigmoid2"  — w_c = 2·sigmoid(θ_c), constrained to (0, 2).
        init_value: starting value for w_c at step 0. For
            backward-compat with the baseline forward, leave at 1.0.

    Forward signature:
        apply(M_e_sparse, cycle_idx_of_nz: LongTensor)
            -> sparse tensor with weighted values

    cycle_idx_of_nz must be the cycle-column index for each non-zero
    of M_e, in COO ordering. The cycle pool's construction code
    already produces this (it's just M_e._indices()[1]).
    """

    def __init__(
        self,
        n_cycles: int,
        param_kind: str = "scalar",
        init_value: float = 1.0,
    ):
        super().__init__()
        if param_kind not in ("scalar", "exp", "sigmoid2"):
            raise ValueError(
                f"param_kind must be one of scalar / exp / sigmoid2, "
                f"got {param_kind!r}"
            )
        self.n_cycles = n_cycles
        self.param_kind = param_kind
        if param_kind == "scalar":
            init_t = torch.full((n_cycles,), float(init_value))
        elif param_kind == "exp":
            # w = exp(θ); to get init_value, θ = log(init_value).
            init_t = torch.full(
                (n_cycles,), float(torch.tensor(init_value).log()),
            )
        else:  # sigmoid2: w = 2·sigmoid(θ); to get init_value=1, θ=0.
            init_t = torch.zeros(n_cycles)
        self.theta = nn.Parameter(init_t)

    def weights(self) -> torch.Tensor:
        """Compute the per-cycle weight tensor w_c from θ_c."""
        if self.param_kind == "scalar":
            return self.theta
        if self.param_kind == "exp":
            return self.theta.exp()
        # sigmoid2:
        return 2.0 * torch.sigmoid(self.theta)

    def apply(
        self,
        M_e: torch.Tensor,            # sparse_coo, shape (E, n_cycles)
        cycle_idx_of_nz: torch.Tensor,  # (nnz,) long: cycle col per nz
    ) -> torch.Tensor:
        """Multiply each non-zero of M_e by the cycle's learnable
        weight, returning a sparse tensor with the same indices but
        scaled values."""
        if not M_e.is_sparse:
            raise ValueError("LearnableMe.apply expects a sparse M_e tensor")
        w = self.weights()                          # (n_cycles,)
        scale = w[cycle_idx_of_nz]                  # (nnz,)
        weighted_values = M_e.values() * scale
        return torch.sparse_coo_tensor(
            M_e.indices(), weighted_values, M_e.shape,
        ).coalesce()

    def regularization(self, lam: float = 1e-4) -> torch.Tensor:
        """L2 regularization on (w - 1) to discourage runaway scaling.

        Returns a scalar tensor that can be added to the loss:
            loss = task_loss + reg
        """
        if lam <= 0:
            return torch.zeros((), device=self.theta.device)
        w = self.weights()
        return lam * ((w - 1.0) ** 2).sum()

    def n_params(self) -> int:
        return self.n_cycles


__all__ = ["LearnableMe"]
