"""Bochner-coupled HypergraphConv (GömbSoma-Ricci-Stim phase 4).

The Bochner-Weitzenböck identity on a Riemannian manifold decomposes
the Hodge Laplacian on 1-forms as

    \\Delta_1 \\omega = \\nabla^* \\nabla \\omega + \\mathrm{Ric}(\\omega^\\sharp)^\\flat

— a flat-connection term plus a curvature correction. On a discrete
signed simplicial complex we promote this to an architectural
identity: each message-passing step is the sum of three terms,

    msg(c) = msg_inner(c)
           + alpha * hodge_proj( mean over v in c of ( Delta_0 x )_v )
           + beta  * primitive_curvature(c) * ricci_proj( mean over v in c of x_v )

where msg_inner is the inner HypergraphConv subclass's flat message,
the Hodge term applies discrete heat-flow smoothing at the vertex
level then aggregates to the primitive, and the Ricci term scales a
projected mean by per-primitive curvature kappa.

Critical contract
-----------------
With alpha = beta = 0, the wrapper's forward output is BIT-IDENTICAL
to the inner HypergraphConv subclass's forward output. Pinned by
unit test; this guarantees Phase 4 is purely additive.

State protocol
--------------
The wrapper exposes the standard HypergraphConv.forward signature
(x, primitives, primitive_signs, M_v) — so it is a drop-in. Pass
the Hodge Laplacian and per-primitive curvatures via the
`prepare()` method before calling forward:

    layer = BochnerHypergraphConv(inner_walk_conv, alpha=0.1, beta=0.05)
    layer.prepare(hodge_laplacian=Delta_0, primitive_curvatures=kappa_per_walk)
    y = layer(x, walks, walk_signs, M_v)

This pattern (state set just before forward) is the same one used
by FiLM conditioning and attention-mask preparation.

Plan: docs/plans/2026-05-14-gomb-soma-ricci-stim/.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from signedkan_wip.src.hymeko_gomb.soma.hg_conv import HypergraphConv


class BochnerHypergraphConv(HypergraphConv):
    """Bochner-Weitzenböck-coupled HypergraphConv wrapper.

    Parameters
    ----------
    inner : HypergraphConv
        Any concrete HypergraphConv subclass (WalkConvLayer,
        PolygonConvLayer, TriangleConvLayer, ...). The wrapper does
        NOT modify the inner — it composes with it.
    alpha : float, default 0.0
        Initial mixing coefficient for the Hodge-smoothing term. Made
        a learnable Parameter so training can drive it.
    beta : float, default 0.0
        Initial mixing coefficient for the Ricci-curvature correction.
    learnable_mixing : bool, default True
        If True, alpha and beta are nn.Parameters; if False, buffers.

    Preconditions
    -------------
    * `inner.config` is the shared config; the wrapper uses the same
      in_features, out_features, k_arity, use_sign_branching.
    * Before forward, the caller may invoke `prepare()` to set the
      Hodge Laplacian (sparse n_v × n_v) and per-primitive curvature
      tensor (n_prim,).

    Postconditions
    --------------
    * With alpha == beta == 0, forward returns bit-identical output
      to `inner.forward(x, primitives, signs, M_v)`.
    * Output shape is (n_nodes, out_features), same as inner.
    """

    def __init__(
        self,
        inner: HypergraphConv,
        alpha: float = 0.0,
        beta: float = 0.0,
        learnable_mixing: bool = True,
    ) -> None:
        # Re-use the inner layer's config for super().__init__.
        super().__init__(inner.config)
        self.inner = inner
        if learnable_mixing:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))
            self.register_buffer("beta", torch.tensor(float(beta)))
        # Sign-branched projections for each correction term: in → out.
        # Default Linear init; gradients flow through alpha / beta gating.
        self.hodge_proj = nn.Linear(inner.in_features, inner.out_features)
        self.ricci_proj = nn.Linear(inner.in_features, inner.out_features)
        # Geometric-context state, set by prepare().
        self._hodge_laplacian: Optional[torch.Tensor] = None
        self._primitive_curvatures: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------
    # State
    # -----------------------------------------------------------------

    def prepare(
        self,
        hodge_laplacian: Optional[torch.Tensor] = None,
        primitive_curvatures: Optional[torch.Tensor] = None,
    ) -> None:
        """Set the geometric context for the next forward pass.

        Parameters
        ----------
        hodge_laplacian : sparse Tensor[n_vertices, n_vertices] or None
            The Hodge Laplacian Δ_0. If None, the Hodge term is
            effectively skipped (its contribution is zero, regardless
            of alpha).
        primitive_curvatures : Tensor[n_primitives] or None
            Per-primitive curvature scalar (e.g., mean Forman κ over
            the primitive's constituent edges). If None, the Ricci
            term is effectively skipped.
        """
        self._hodge_laplacian = hodge_laplacian
        self._primitive_curvatures = primitive_curvatures

    # -----------------------------------------------------------------
    # HypergraphConv hook
    # -----------------------------------------------------------------

    def _forward_messages(
        self,
        x: torch.Tensor,
        primitives: torch.Tensor,
        primitive_signs: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Flat-connection term: the inner layer's message.
        msg = self.inner._forward_messages(x, primitives, primitive_signs)

        # 2. Hodge-smoothing term: alpha * hodge_proj( mean( Delta_0 x )_c ).
        if self._hodge_laplacian is not None:
            x_smooth = torch.sparse.mm(self._hodge_laplacian, x)
            # Gather smoothed features per primitive vertex, mean over k.
            gathered = x_smooth[primitives].mean(dim=1)   # (n_prim, in)
            msg = msg + self.alpha * self.hodge_proj(gathered)

        # 3. Ricci-correction term: beta * kappa(c) * ricci_proj( mean(x)_c ).
        if self._primitive_curvatures is not None:
            x_mean = x[primitives].mean(dim=1)            # (n_prim, in)
            kappa = self._primitive_curvatures.float().unsqueeze(-1)  # (n_prim, 1)
            msg = msg + self.beta * kappa * self.ricci_proj(x_mean)

        return msg

    def _aggregate(
        self,
        messages: torch.Tensor,
        M_v: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to inner's aggregator so any inner-specific
        aggregation (e.g., normalisation, routing) is preserved."""
        return self.inner._aggregate(messages, M_v)

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha.item():.4f}, "
            f"beta={self.beta.item():.4f}, "
            f"inner={type(self.inner).__name__}"
        )
