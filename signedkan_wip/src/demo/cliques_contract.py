"""Nodelets — the cliques-to-super-vertex contraction operator.

V1 of the multi-scale operator from the belief-planning bridge plan
(`docs/plans/2026-05-14-gomb-belief-planning-bridge/plan.md`).
"Derivative of a clique = nodelet": each detected balanced clique
collapses into a single super-vertex, and the contracted graph is
again a signed graph that Gömb can operate on.

==============================================================
Mathematical definition
==============================================================

Let `H = (V, E, σ)` be a signed graph with σ: E → {±1}. Let
`C = {C_1, ..., C_K}` be a collection of vertex-disjoint balanced
cliques in `H` (balance per the triangle-product check; see
``cliques._clique_balance_indicator``).

The **contraction operator** `D` maps `H` to a coarser signed graph
`H' = (V', E', σ')` defined as:

  V' = C ∪ { {v} : v ∈ V \\ ⋃ C_i }     (cliques + uncovered singletons)

  φ: V → V'   (the surjective coarsening map)
     φ(v) = C_i if v ∈ C_i, else {v}

  E' = { (φ(u), φ(w)) : (u, w) ∈ E ∧ φ(u) ≠ φ(w) }

  σ' : E' → {±1}     (boundary-sign aggregation)
     σ'(s, t) = sign( ∑_{(u, w) ∈ E : φ(u)=s, φ(w)=t} σ(u, w) )
              with ties broken to +1.

Internal edges (those with `φ(u) = φ(w)`) are absorbed into the
super-vertex and **discarded** from `E'`. This is the "loses local
detail in exchange for global shape" property — internal balance
becomes invisible at the coarser scale, only the boundary signs
matter.

==============================================================
Disambiguation: clique overlap
==============================================================

For the math above to be well-defined, the input cliques must be
vertex-disjoint. The cliques *returned* by Bron-Kerbosch / greedy
/ spectral detectors are typically maximal but **may overlap**.

We resolve overlap by **greedy size-descending selection**: process
detected cliques in order of size (largest first); accept a clique
iff its remaining-uncovered members number ≥ ``min_clique_size``;
mark those members as covered; skip subsequent cliques that no
longer have enough fresh members.

This is one of several valid choices. Alternatives:
  - random tie-break (less deterministic, used in some
    renormalisation literature)
  - eigenvector-weighted (use Gömb's vertex embeddings as a score)

V1 ships size-descending; V2 (after Gömb integration lands) may
switch to a learned ordering.

==============================================================
Properties
==============================================================

1. **Vertex count is non-increasing:** |V'| ≤ |V|.
   Equality iff no balanced cliques of size ≥ ``min_clique_size``
   exist in the input.

2. **Idempotence after fixed-point:** the operator is idempotent
   eventually — if `D(H) = H` (no further coarsening possible),
   subsequent applications return `H` unchanged. Concretely:
   `D^∞(H) = D^k(H)` for some finite k ≤ log₂ |V|.

3. **Edge preservation modulo internal absorption:** every edge in
   `E` either appears in `E'` (after the φ-quotient) or is absorbed
   into a super-vertex. **No edge is lost without being explicitly
   absorbed.**

4. **Determinism:** for fixed input, fixed clique-detector, fixed
   `min_clique_size`, the output is fully deterministic.

5. **Sign aggregation respects majority:** if all boundary signs
   between `(s, t)` are +, then σ'(s, t) = +. Mixed-sign boundaries
   resolve to the majority; tied resolves to +.

==============================================================
Iteration: building the hierarchy
==============================================================

``multiscale_hierarchy(bundle, max_levels=k)`` returns a list
`[H, H', H'', ..., H^(k)]` of length up to `k+1`. Iteration stops
early when either:

  - max_levels reached, or
  - `D(H^(i)) = H^(i)` (fixed point reached, no further coarsening),
  - the result has ≤ 1 vertex (degenerate).

==============================================================
What "nodelet" carries beyond the vertex index
==============================================================

A nodelet `s ∈ V'` is conceptually `(C_i, n_members, centroid_pos)`.
The contraction operator preserves:

  - `n_members(s)`  — size of the underlying vertex set
  - `centroid(s)`   — spatial centroid for visualisation
  - `members(s)`    — the original vertex indices (for hierarchical
                       expansion back down)

These are used by downstream tasks (the belief-planning bridge's
plan-then-dispatch flow), not by Gömb's cycle pool directly.
Gömb sees `H'` as a flat signed graph; the metadata is for the
caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..datasets import SignedGraph
from .cliques import (
    Clique, RobotNetworkBundle, _clique_balance_indicator,
    enumerate_balanced_cliques,
)


@dataclass
class ContractedBundle:
    """One level of the multi-scale hierarchy.

    Has the same shape as ``RobotNetworkBundle`` (so the same
    ``cliques_plotting`` code can render it), plus provenance fields
    pointing back to the previous level.
    """

    graph: SignedGraph
    positions: np.ndarray                # (n_supernodes, 2) centroid
    seed: int                            # carried for determinism
    comm_range: float                    # carried (no longer meaningful at H')
    noise_prob: float                    # carried
    area_size: float                     # carried
    name: str = "contracted"

    # Provenance back to the previous level.
    parent_n_vertices: int = 0           # |V| of the input
    super_members: list[tuple[int, ...]] = field(default_factory=list)
    super_clique_indices: list[int] = field(default_factory=list)
    # ↑ for each new super-vertex i, ``super_clique_indices[i]`` is the
    # original detector-clique index it came from, or -1 for singletons.
    phi: np.ndarray | None = None        # (parent_n_vertices,) int

    @property
    def n_robots(self) -> int:
        return self.graph.n_nodes

    @property
    def n_edges(self) -> int:
        return self.graph.edges.shape[0]

    @property
    def n_negative_edges(self) -> int:
        return int((self.graph.signs == -1).sum())

    @property
    def n_positive_edges(self) -> int:
        return int((self.graph.signs == 1).sum())

    @property
    def compression_ratio(self) -> float:
        """|V'| / |V|, the fraction of vertices that survive contraction."""
        return self.n_robots / self.parent_n_vertices if self.parent_n_vertices else 1.0

    @property
    def n_singletons(self) -> int:
        return sum(1 for ci in self.super_clique_indices if ci == -1)


def _select_disjoint_cliques(
    cliques: list[Clique], min_clique_size: int
) -> list[Clique]:
    """Greedy size-descending selection of vertex-disjoint cliques.

    Maximal cliques from Bron-Kerbosch may overlap (a vertex can be
    in two maximal cliques). For the contraction operator to be
    well-defined we need vertex-disjoint cliques. Greedy on size
    favours large super-vertices, which is what the multi-scale
    story wants (more aggressive coarsening per level).
    """
    sorted_cliques = sorted(cliques, key=lambda c: (-c.size, c.members))
    covered: set[int] = set()
    out: list[Clique] = []
    for c in sorted_cliques:
        fresh = [v for v in c.members if v not in covered]
        if len(fresh) < min_clique_size:
            continue
        out.append(c)
        for v in fresh:
            covered.add(v)
    return out


def contract_balanced_cliques(
    bundle: RobotNetworkBundle,
    min_clique_size: int = 3,
    max_clique_size: int = 8,
    sign_aggregation: Literal["majority"] = "majority",
    tie_breaker: Literal["positive", "negative"] = "positive",
    detector_limit: int = 50,
    name_suffix: str = "_D",
) -> ContractedBundle:
    """Apply the contraction operator `D` once.

    Pipeline:
      1. detect balanced cliques in the bundle (Bron-Kerbosch + balance check)
      2. greedy-disjoint selection
      3. build φ: original vertex → super-vertex index
      4. quotient edges; aggregate boundary signs by majority
      5. compute centroid positions for visualisation

    Returns the contracted bundle. Provenance fields preserve the
    inverse map (``phi``) so callers can dispatch decisions made at
    the coarse level back down to the fine level.
    """
    if sign_aggregation != "majority":
        raise NotImplementedError(
            f"sign_aggregation={sign_aggregation!r}: only 'majority' supported in V1"
        )

    # 1. Detect balanced cliques.
    cliques = enumerate_balanced_cliques(
        bundle, min_size=min_clique_size, max_size=max_clique_size,
        limit=detector_limit,
    )

    # 2. Greedy disjoint selection.
    chosen = _select_disjoint_cliques(cliques, min_clique_size)

    # 3. Build the φ map.
    n = bundle.n_robots
    phi = np.full(n, fill_value=-1, dtype=np.int64)
    super_members: list[tuple[int, ...]] = []
    super_clique_indices: list[int] = []
    for clique_idx, c in enumerate(chosen):
        # Store only the FRESH members — those not already covered by
        # a previously-accepted larger clique. The disjoint-selection
        # helper may return overlapping cliques (vertices in earlier
        # ones win); we record only the actually-claimed members.
        fresh: list[int] = []
        new_super_idx = len(super_members)
        for v in c.members:
            if phi[v] != -1:
                continue  # already mapped by an earlier larger clique
            phi[v] = new_super_idx
            fresh.append(int(v))
        if not fresh:
            # Degenerate: every member already covered. Should not
            # happen given _select_disjoint_cliques's min_size gate,
            # but be defensive.
            continue
        super_members.append(tuple(sorted(fresh)))
        super_clique_indices.append(clique_idx)
    # 3b. Singletons for uncovered vertices.
    for v in range(n):
        if phi[v] == -1:
            phi[v] = len(super_members)
            super_members.append((int(v),))
            super_clique_indices.append(-1)

    n_super = len(super_members)

    # 4. Quotient edges, aggregate signs.
    # accumulator: (i, j) with i < j  →  running sum of σ
    sign_sum: dict[tuple[int, int], int] = {}
    for (u, v), s in zip(bundle.graph.edges, bundle.graph.signs):
        u_int, v_int = int(u), int(v)
        i, j = int(phi[u_int]), int(phi[v_int])
        if i == j:
            continue  # internal edge, absorbed
        key = (i, j) if i < j else (j, i)
        sign_sum[key] = sign_sum.get(key, 0) + int(s)

    edges_out: list[tuple[int, int]] = []
    signs_out: list[int] = []
    for key, total in sign_sum.items():
        edges_out.append(key)
        if total > 0:
            signs_out.append(+1)
        elif total < 0:
            signs_out.append(-1)
        else:
            signs_out.append(+1 if tie_breaker == "positive" else -1)

    edges_arr = (np.array(edges_out, dtype=np.int64)
                  if edges_out else np.zeros((0, 2), dtype=np.int64))
    signs_arr = (np.array(signs_out, dtype=np.int8)
                  if signs_out else np.zeros((0,), dtype=np.int8))
    g_out = SignedGraph(edges=edges_arr, signs=signs_arr,
                          n_nodes=n_super)

    # 5. Centroid positions per super-vertex.
    new_pos = np.zeros((n_super, 2), dtype=np.float32)
    for si, members in enumerate(super_members):
        new_pos[si] = bundle.positions[list(members)].mean(axis=0)

    return ContractedBundle(
        graph=g_out,
        positions=new_pos,
        seed=bundle.seed,
        comm_range=bundle.comm_range,
        noise_prob=bundle.noise_prob,
        area_size=bundle.area_size,
        name=getattr(bundle, "name", "graph") + name_suffix,
        parent_n_vertices=n,
        super_members=super_members,
        super_clique_indices=super_clique_indices,
        phi=phi,
    )


def multiscale_hierarchy(
    bundle: RobotNetworkBundle,
    max_levels: int = 5,
    min_clique_size: int = 3,
    max_clique_size: int = 8,
    detector_limit: int = 50,
) -> list[RobotNetworkBundle | ContractedBundle]:
    """Build [H, H', H'', ..., H^(k)] up to ``max_levels``.

    Stops early if:
      - the result has ≤ 1 vertex,
      - the result is structurally identical to its parent
        (|V'| = |V| with same edge sign distribution).

    The first element is the original bundle; subsequent elements
    are ``ContractedBundle`` levels.
    """
    levels: list[RobotNetworkBundle | ContractedBundle] = [bundle]
    cur = bundle
    for _level in range(max_levels):
        contracted = contract_balanced_cliques(
            cur, min_clique_size=min_clique_size,
            max_clique_size=max_clique_size,
            detector_limit=detector_limit,
            name_suffix=f"_D{_level + 1}",
        )
        if contracted.n_robots <= 1:
            levels.append(contracted)
            break
        if contracted.n_robots == cur.n_robots:
            # No contraction happened at this level — fixed point.
            break
        levels.append(contracted)
        cur = contracted
    return levels


__all__ = [
    "ContractedBundle",
    "contract_balanced_cliques",
    "multiscale_hierarchy",
]
