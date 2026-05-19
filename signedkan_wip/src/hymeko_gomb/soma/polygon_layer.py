"""PolygonConvLayer — closed-cycle hypergraph convolution for GömbSoma phase 3-G.

A polygon of arity ``k_arity`` is a closed cycle of ``k_arity``
vertices connected by ``k_arity`` edges. Unlike a walk, a polygon
has no canonical starting vertex: the cyclic sequence
$(v_0, v_1, \\ldots, v_{k-1})$ represents the same polygon as any
cyclic shift $(v_s, v_{s+1}, \\ldots, v_{s-1})$, and (for undirected
graphs) as the reversal $(v_{k-1}, \\ldots, v_0)$. Therefore the
message function must be invariant under cyclic shifts and
reflections of the vertex order.

Position-aware weights (the WalkConv mechanism) collapse to a
position-agnostic projection under cyclic symmetry — see notes in
``signedkan_wip/src/hymeko_gomb/soma/polygon_layer.py`` below. The
honest position-agnostic message is:

    msg(c) = ψ_{π(c)} \\Bigl( W^{π(c)} \\, \\frac{1}{k} \\sum_{i=0}^{k-1} x_{v_i} + b^{π(c)} \\Bigr)

where ``π(c) ∈ {-1, +1}`` is the cycle's σ-product and W^±, b^±
are sign-branched. This is automatically cyclic- and
reflection-invariant by construction.

For TriangleConvLayer (phase 4), the Cartwright–Harary balance gate
adds back discriminative power by routing balanced triads through a
distinct sub-bank that exploits the k=3 case's strong structural
constraints. For PolygonConvLayer (this phase), we keep it simple.

Recommended arity: ``k_arity >= 4``. The layer accepts ``k_arity == 3``
too, but for triangles the upcoming TriangleConvLayer is preferred.

Aggregation: default sum-pool via ``M_v`` (inherited from
``HypergraphConv._aggregate``).

Parameter count
---------------
Per layer: ``2 × (in_features × out_features) + 2 × out_features``,
independent of ``k_arity``. Significantly lighter than WalkConvLayer
because position-aware weights aren't useful under cyclic symmetry.

At default in_features = out_features = 16: 528 parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from signedkan_wip.src.hymeko_gomb.soma.hg_conv import (
    HypergraphConv,
    HypergraphConvConfig,
)


class PolygonConvLayer(HypergraphConv):
    """Closed-cycle (polygon) hypergraph convolution, sign-branched,
    cyclic-and-reflection-invariant.

    Preconditions
    -------------
    * ``config.k_arity >= 3`` — a polygon has at least three vertices.

    Postconditions
    --------------
    * The message function is invariant under cyclic shifts and
      reflections of each polygon's vertex order.
    * Permutation-equivariant over the global vertex set (inherited
      from HypergraphConv).
    """

    def __init__(self, config: HypergraphConvConfig) -> None:
        if config.k_arity < 3:
            raise ValueError(
                f"PolygonConvLayer requires k_arity >= 3 "
                f"(a polygon has at least three vertices), "
                f"got {config.k_arity}"
            )
        super().__init__(config)
        n_branches = 2 if self.use_sign_branching else 1
        # Sign-branched projection: W shape (n_branches, in, out).
        self.W = nn.Parameter(
            torch.empty(n_branches, self.in_features, self.out_features)
        )
        if self.config.bias:
            self.bias_p = nn.Parameter(
                torch.zeros(n_branches, self.out_features)
            )
        else:
            self.register_parameter("bias_p", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        fan = self.in_features
        bound = (6.0 / fan) ** 0.5
        with torch.no_grad():
            self.W.uniform_(-bound, bound)
            if self.bias_p is not None:
                self.bias_p.zero_()

    # -----------------------------------------------------------------
    # HypergraphConv hook
    # -----------------------------------------------------------------

    def _forward_messages(
        self,
        x: torch.Tensor,
        primitives: torch.Tensor,
        primitive_signs: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-polygon messages.

        Parameters
        ----------
        x : Tensor[n_nodes, in_features]
        primitives : Tensor[n_polygons, k_arity]
        primitive_signs : Tensor[n_polygons]

        Returns
        -------
        Tensor[n_polygons, out_features]
        """
        # Cycle-mean of vertex features: (n_polygons, in_features)
        # This is the canonical cyclic-and-reflection-invariant
        # first-moment feature.
        gathered = x[primitives]                  # (n_polygons, k_arity, in)
        mean_x = gathered.mean(dim=1)             # (n_polygons, in)

        if self.use_sign_branching:
            branch_idx = (primitive_signs < 0).to(torch.long)
        else:
            branch_idx = torch.zeros(
                primitives.shape[0], dtype=torch.long, device=primitives.device
            )

        # Per-polygon W of shape (n_polygons, in, out).
        W_per = self.W[branch_idx]
        # Project: (n_polygons, in) @ (n_polygons, in, out) → (n_polygons, out)
        msg = torch.einsum("ni,nij->nj", mean_x, W_per)

        if self.bias_p is not None:
            msg = msg + self.bias_p[branch_idx]

        return F.gelu(msg)
