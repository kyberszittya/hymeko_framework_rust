"""WalkConvLayer — open-walk hypergraph convolution for GömbSoma phase 2.

A walk of arity ``k_arity`` is an ordered tuple of ``k_arity`` vertices
connected by ``k_arity - 1`` edges; e.g., a walk with ``k_arity = 3``
traverses ``v_0 -e_0- v_1 -e_1- v_2``. The walk is OPEN (no closure
edge between the endpoints); polygons handle the closed case in
phase 3.

The sign of a walk is the σ-product of its ``k_arity - 1`` constituent
edges. Positive walks (π = +1) and negative walks (π = -1) route
through independent learnable banks; this is the WalkConv-level
realisation of the Cartwright-Harary branching idea.

Architecture
------------
For a walk c = (v_0, v_1, ..., v_{k-1}) with sign π(c) ∈ {-1, +1}:

    msg(c) = ψ_{π(c)} \\Bigl( ∑_{i=0}^{k-1} W^{π(c)}_i @ x_{v_i} + b^{π(c)} \\Bigr)

where:
* ``W^±_i ∈ ℝ^{in × out}`` are position-aware sign-branched weights;
* ``b^± ∈ ℝ^out`` are sign-branched biases;
* ``ψ_±`` is a sign-branched activation (default: GELU).

Position-aware weights make walks directed: a walk and its reversal
in general produce different messages. This is the correct behaviour
for the sensorimotor stack — walks represent time-ordered sensor
sequences, not unordered edge sets.

Aggregation: default sum-pool via ``M_v`` (inherited from
``HypergraphConv._aggregate``).

Parameter count
---------------
Per layer: 2 × k_arity × (in_features × out_features) + 2 × k_arity × out_features
+ 0 (no additional embedding).

At default k_arity = 3, in_features = 16, out_features = 16, this is
1 632 parameters — light enough to stack many layers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from signedkan_wip.src.hymeko_gomb.soma.hg_conv import (
    HypergraphConv,
    HypergraphConvConfig,
)


class WalkConvLayer(HypergraphConv):
    """Open-walk hypergraph convolution, sign-branched, position-aware.

    Preconditions (in addition to HypergraphConv's)
    -----------------------------------------------
    * ``config.k_arity >= 2`` — a walk needs at least two vertices.
    * ``config.use_sign_branching = True`` is the canonical mode; if
      False, the layer falls back to a single shared bank (no
      sign-routing). Forbidden if you want the Cartwright-Harary
      branching behaviour.

    Postconditions
    --------------
    * The map is position-aware: reversing a walk's vertex order
      generally changes the message (walks are directed).
    """

    def __init__(self, config: HypergraphConvConfig) -> None:
        super().__init__(config)
        # Position-aware sign-branched weights.
        # W has shape (2, k_arity, in, out); the leading axis selects
        # the sign branch (0 = positive, 1 = negative).
        n_branches = 2 if self.use_sign_branching else 1
        self.W = nn.Parameter(
            torch.empty(n_branches, self.k_arity,
                         self.in_features, self.out_features)
        )
        if self.config.bias:
            self.bias_p = nn.Parameter(
                torch.zeros(n_branches, self.out_features)
            )
        else:
            self.register_parameter("bias_p", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Position-aware Xavier-uniform per branch and per position.
        fan = self.in_features * self.k_arity
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
        """Build per-walk messages.

        Parameters
        ----------
        x : Tensor[n_nodes, in_features]
        primitives : Tensor[n_walks, k_arity]
            Long-int tensor of vertex indices, one row per walk.
        primitive_signs : Tensor[n_walks]
            Int tensor of ±1.

        Returns
        -------
        Tensor[n_walks, out_features]
        """
        # Gather position-wise vertex features: (n_walks, k_arity, in)
        gathered = x[primitives]

        if self.use_sign_branching:
            # Branch index: 0 for σ=+1, 1 for σ=-1.
            branch_idx = (primitive_signs < 0).to(torch.long)
        else:
            branch_idx = torch.zeros(
                primitives.shape[0], dtype=torch.long, device=primitives.device
            )

        # We want, per walk c with branch b(c) and positions i = 0..k-1:
        #     msg(c) = ψ( Σ_i W[b(c), i] @ x_{v_i^c} + bias[b(c)] )
        # Vectorise via per-branch matmul:
        #     contrib[c, i, :] = x_{v_i^c} @ W[b(c), i]
        # Then sum over i.

        # Gather the per-walk weight tensor: (n_walks, k_arity, in, out).
        # Index W (n_branches, k_arity, in, out) at axis 0 by branch_idx.
        W_per_walk = self.W[branch_idx]  # (n_walks, k_arity, in, out)
        # contrib[c, i] = x_{v_i^c} @ W_per_walk[c, i]
        # einsum: 'nki,nkij->nkj'
        contrib = torch.einsum("nki,nkij->nkj", gathered, W_per_walk)
        msg = contrib.sum(dim=1)  # (n_walks, out)

        if self.bias_p is not None:
            msg = msg + self.bias_p[branch_idx]

        # Sign-branched activation: GELU on each branch independently.
        # (Identical activation here; the per-branch separation is
        # already in W and bias_p. Distinct ψ_± would be a phase-3
        # extension.)
        return F.gelu(msg)
