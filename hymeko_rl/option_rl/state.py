"""Task-independent STRUCTURED STATE for the assimilation runtime (ARCHITECTURAL_ASSIMILATION_V1, track A1).

O2 established that a flat obs → single θ map cannot represent multimodal option structure. The runtime input must therefore
be an explicit STRUCTURED state (entities + relations + attributes + phase/monitor + geometry + contact + task metadata),
not a bare vector — so encoders can condition on geometry/contact/phase and the multimodal proposal can route modes.

This module is DOMAIN-FREE (imports nothing from coin/CIP/pick-place). The first anchor is a hypergraph — but taken by
DUCK TYPE (any object exposing ``node_features()`` + optional ``star_expansion()``/``n_vertices``), so `option_rl` never
imports `agents.hypergraph_state`. `FlatStateView` is the backward-compat path: it exposes a `StructuredState` as the flat
vector the current single-θ models expect, so a task migrates one head at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class StructuredState:
    """The structured runtime state consumed by the representation/temporal encoders and the multimodal proposal.

    # Invariants ``node_features`` is 2-D ``(N, F)``; ``edges`` are index tuples into ``[0,N)``; ``edge_signs`` (if given)
    aligns with ``edges``. All arrays are float32/int. Purely data — no task logic."""

    node_features: np.ndarray                         # (N, F) per-entity continuous attributes
    edges: list[tuple] = field(default_factory=list)  # relations / hyperedges (tuples of vertex indices)
    edge_signs: "np.ndarray | None" = None            # signed incidence per edge (optional)
    phase: int = 0                                    # controller phase id
    monitor: dict = field(default_factory=dict)       # gate/monitor state (e.g. contact streaks, containment)
    geometry: "np.ndarray | None" = None              # object geometry descriptors (shape dims, aspect, orientation)
    contact: "np.ndarray | None" = None               # contact structure (active edges / normals / moment-arm)
    metadata: dict = field(default_factory=dict)      # task metadata (shape family, embodiment, target relative geometry)

    def __post_init__(self):
        self.node_features = np.asarray(self.node_features, np.float32)
        if self.node_features.ndim != 2:
            raise ValueError(f"node_features must be 2-D (N,F); got shape {self.node_features.shape}")
        n = self.node_features.shape[0]
        for e in self.edges:
            if any((not isinstance(i, (int, np.integer)) or i < 0 or i >= n) for i in e):
                raise ValueError(f"edge {e} indexes outside [0,{n})")

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    def flat(self) -> np.ndarray:
        """FlatStateView payload: flattened node features ⊕ geometry ⊕ phase — the vector old single-θ models expect."""
        parts = [self.node_features.reshape(-1)]
        if self.geometry is not None:
            parts.append(np.asarray(self.geometry, np.float32).reshape(-1))
        parts.append(np.asarray([float(self.phase)], np.float32))
        return np.concatenate(parts).astype(np.float32)

    @classmethod
    def from_hypergraph(cls, hg: Any, *, phase: int = 0, monitor: "dict | None" = None,
                        geometry: "np.ndarray | None" = None, contact: "np.ndarray | None" = None,
                        metadata: "dict | None" = None) -> "StructuredState":
        """Build from any hypergraph-like object (duck-typed: ``node_features()`` required; ``edges``/``star_expansion``
        optional). Keeps `option_rl` free of `agents.hypergraph_state` — the anchor is structural, not an import."""
        nf = np.asarray(hg.node_features(), np.float32)
        if nf.ndim == 1:
            nf = nf.reshape(1, -1)
        edges: list[tuple] = []
        if hasattr(hg, "edges"):
            try:
                edges = [tuple(int(i) for i in e) for e in hg.edges]
            except TypeError:
                edges = []
        return cls(nf, edges, None, phase, dict(monitor or {}), geometry, contact, dict(metadata or {}))


@runtime_checkable
class StructuredStateAdapter(Protocol):
    """A task supplies this: raw env observation/state → `StructuredState`. The runtime never sees the raw obs directly."""

    def structured(self, raw: Any) -> StructuredState: ...


@dataclass
class FlatStateView:
    """Backward-compat adapter: expose a `StructuredState` (or a task's `StructuredStateAdapter` output) as the flat vector
    the current single-θ proposals/critics consume — so old models keep working while a task migrates head by head."""

    adapter: "StructuredStateAdapter | None" = None

    def view(self, raw: Any) -> np.ndarray:
        s = raw if isinstance(raw, StructuredState) else (self.adapter.structured(raw) if self.adapter else None)
        if s is None:
            raise ValueError("FlatStateView needs a StructuredState or an adapter to produce one")
        return s.flat()
