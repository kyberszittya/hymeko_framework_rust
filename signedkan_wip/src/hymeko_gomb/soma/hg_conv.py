"""HypergraphConv — abstract message-passing primitive for GömbSoma.

This ABC is the contract that every GömbSoma layer (WalkConv,
PolygonConv, TriangleConv, AbstractionConv) implements. It fixes:

  - The forward-signature shape.
  - Permutation equivariance over the vertex set (the layer is a
    function of the signed-hypergraph isomorphism class, not the
    vertex labelling).
  - Sparse-aware aggregation (no dense |V|×|P| materialisation).
  - Sign-branching support (positive vs frustrated primitives route
    through distinct sub-banks; concrete layers wire this in).

The ABC itself is not directly instantiable; subclasses must provide
``_forward_messages`` and may override ``_aggregate``. The ``forward``
method is sealed: it validates preconditions, calls ``_forward_messages``,
calls ``_aggregate``, and checks postconditions.

Per CLAUDE.md §8 (Design by Contract): public preconditions / post-
conditions are documented and asserted in debug builds.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class HypergraphConvConfig:
    """Shared configuration object for all HypergraphConv subclasses.

    The fields here are the union of what any layer might need; each
    subclass picks the ones it uses. This avoids the per-layer-kwargs
    blow-up that we explicitly forbid (CLAUDE.md §6.5 anti-pattern #1).

    Attributes
    ----------
    in_features : int
        Vertex-feature dimension on input.
    out_features : int
        Vertex-feature dimension on output.
    k_arity : int
        Primitive arity. WalkConv uses walk length, PolygonConv uses
        cycle length, TriangleConv pins to 3. AbstractionConv ignores
        this field (clique size is variable).
    use_sign_branching : bool
        If True, positive (π = +1) and negative (π = −1) primitives
        route through independent sub-banks. Required by all layers
        that consume signed primitives.
    bias : bool
        Include a learnable bias on the output projection.
    """

    in_features: int
    out_features: int
    k_arity: int
    use_sign_branching: bool = True
    bias: bool = True
    extra: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        """Check the config is internally consistent. Raises ValueError
        on misuse — see Preconditions in HypergraphConv.forward."""
        if self.in_features <= 0:
            raise ValueError(
                f"in_features must be positive, got {self.in_features}"
            )
        if self.out_features <= 0:
            raise ValueError(
                f"out_features must be positive, got {self.out_features}"
            )
        if self.k_arity < 2:
            raise ValueError(
                f"k_arity must be >= 2 (a 1-tuple is just a vertex), "
                f"got {self.k_arity}"
            )


class HypergraphConv(nn.Module, abc.ABC):
    """Permutation-equivariant message-passing layer on a signed hypergraph.

    All GömbSoma layers subclass this. The base class provides the
    ``forward`` orchestration; subclasses implement
    ``_forward_messages`` (the layer-specific message function) and
    optionally override ``_aggregate`` (defaults to sum-pool).

    Forward signature
    -----------------
    x : Tensor[n_nodes, in_features]
        Vertex features.
    primitives : Tensor[n_prim, k_arity]
        Each row lists the vertex indices participating in one
        primitive (walk, cycle, triangle, ...). Vertex indices must
        be in [0, n_nodes).
    primitive_signs : Tensor[n_prim]
        Sign of each primitive (typically the σ-product over its
        edges). Values must be in {-1, +1}.
    M_v : torch.sparse.Tensor[n_nodes, n_prim]
        Vertex-to-primitive incidence. M_v[v, p] != 0 iff vertex v
        participates in primitive p. Used by the aggregation step.

    Returns
    -------
    Tensor[n_nodes, out_features]
        Updated vertex features.

    Preconditions
    -------------
    * ``primitives.shape[1] == self.config.k_arity``
    * ``primitive_signs.shape[0] == primitives.shape[0]``
    * Every entry of ``primitive_signs`` is in {-1, +1}.
    * ``M_v`` is a sparse tensor of shape (x.shape[0], primitives.shape[0]).
    * ``x.shape[1] == self.config.in_features``.

    Postconditions
    --------------
    * Output shape is ``(x.shape[0], self.config.out_features)``.
    * The map is permutation-equivariant: for any vertex permutation
      π, ``forward(π(x), π(primitives), signs, π(M_v)) == π(forward(...))``.
    * No dense |V|×|P| tensor is materialised on the forward path.

    Invariants
    ----------
    * The output features depend only on the isomorphism class of the
      input signed hypergraph plus the layer's learnable parameters.
    """

    def __init__(self, config: HypergraphConvConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.in_features = config.in_features
        self.out_features = config.out_features
        self.k_arity = config.k_arity
        self.use_sign_branching = config.use_sign_branching

    # -----------------------------------------------------------------
    # Subclass hooks
    # -----------------------------------------------------------------

    @abc.abstractmethod
    def _forward_messages(
        self,
        x: torch.Tensor,
        primitives: torch.Tensor,
        primitive_signs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-primitive messages.

        Parameters
        ----------
        x : Tensor[n_nodes, in_features]
        primitives : Tensor[n_prim, k_arity]
        primitive_signs : Tensor[n_prim]

        Returns
        -------
        Tensor[n_prim, out_features]
            One message per primitive. The aggregation step then
            distributes these back to vertices via ``M_v``.
        """
        raise NotImplementedError

    def _aggregate(
        self,
        messages: torch.Tensor,
        M_v: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate per-primitive messages back to vertices.

        Default: sparse-matmul sum-pool ``M_v @ messages``. Subclasses
        may override (e.g., to add normalisation or routing).

        Parameters
        ----------
        messages : Tensor[n_prim, out_features]
        M_v : torch.sparse.Tensor[n_nodes, n_prim]

        Returns
        -------
        Tensor[n_nodes, out_features]
        """
        return torch.sparse.mm(M_v, messages)

    # -----------------------------------------------------------------
    # Sealed forward path
    # -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        primitives: torch.Tensor,
        primitive_signs: torch.Tensor,
        M_v: torch.Tensor,
    ) -> torch.Tensor:
        self._check_preconditions(x, primitives, primitive_signs, M_v)
        messages = self._forward_messages(x, primitives, primitive_signs)
        y = self._aggregate(messages, M_v)
        self._check_postconditions(x, y)
        return y

    # -----------------------------------------------------------------
    # Contract checks
    # -----------------------------------------------------------------

    def _check_preconditions(
        self,
        x: torch.Tensor,
        primitives: torch.Tensor,
        primitive_signs: torch.Tensor,
        M_v: torch.Tensor,
    ) -> None:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"x has shape {tuple(x.shape)}, expected "
                f"(n_nodes, {self.in_features})"
            )
        if primitives.ndim != 2 or primitives.shape[1] != self.k_arity:
            raise ValueError(
                f"primitives has shape {tuple(primitives.shape)}, "
                f"expected (n_prim, {self.k_arity})"
            )
        if primitive_signs.ndim != 1 or (
            primitive_signs.shape[0] != primitives.shape[0]
        ):
            raise ValueError(
                f"primitive_signs has shape {tuple(primitive_signs.shape)}, "
                f"expected ({primitives.shape[0]},)"
            )
        # M_v can be coalesced or not; check shape only.
        if M_v.shape != (x.shape[0], primitives.shape[0]):
            raise ValueError(
                f"M_v has shape {tuple(M_v.shape)}, expected "
                f"({x.shape[0]}, {primitives.shape[0]})"
            )
        # Sign validity: must be in {-1, +1}. Skip if empty.
        if primitive_signs.numel() > 0:
            uniq = torch.unique(primitive_signs.to(torch.int64))
            allowed = torch.tensor([-1, 1], device=uniq.device)
            extra = torch.tensor(
                [v.item() for v in uniq if v.item() not in (-1, 1)],
                device=uniq.device,
            )
            if extra.numel() > 0:
                raise ValueError(
                    f"primitive_signs must be in {{-1, +1}}; "
                    f"found extra values {extra.tolist()}"
                )

    def _check_postconditions(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        if y.shape != (x.shape[0], self.out_features):
            raise ValueError(
                f"output has shape {tuple(y.shape)}, expected "
                f"({x.shape[0]}, {self.out_features})"
            )

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"k_arity={self.k_arity}, "
            f"use_sign_branching={self.use_sign_branching}"
        )
