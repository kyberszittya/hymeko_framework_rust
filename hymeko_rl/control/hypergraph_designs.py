"""Combinatorial-design hypergraphs as controller structures (Kato, 2026-06-27).

Extends the topology zoo from ad-hoc *graph* families (2-uniform) to principled *hypergraph* designs with provable
combinatorial structure — and a control-architecture reading for each:

- **Steiner triple system** ``S(2,3,n)`` — every pair of base nodes shares exactly one hyperedge. Control reading:
  a *balanced, minimal-redundancy* all-pairs coupling (every pair of subsystems coordinated exactly once).
- **k-uniform** — every hyperedge couples exactly ``k`` base nodes. The ``k`` axis is the *order* of multi-way
  coupling (pairwise vs three-body vs …) — the higher-order regime the graph zoo cannot reach; ties to the
  aggregation-semantics idea (the combination algebra over a hyperedge).
- **Sunflower** — a shared core ``Y`` plus pairwise-disjoint petals. Control reading: a *shared coordinator +
  independent modules* — a hierarchical / leader–follower / shared-resource architecture.

Each design is a set of *blocks* (member tuples) over ``n`` base vertices; it is instantiated as a signed
:class:`HypergraphState` via the existing **star-expansion bridge** (``with_task_hyperedges`` — one hub per block,
member→hub +1, hub→member −1), so the existing controllers (``SignedKANBackbone``, structured LQR/MPC) consume it
unchanged. The benchmark sees true k>2 hyperedges through their star-expanded signed adjacency.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.agents.hypergraph_state import HypergraphState


def design_to_state(n: int, blocks: "list[tuple[int, ...]]", *, tag: str) -> HypergraphState:
    """Star-expand a design (``n`` base vertices + ``blocks``) into a signed :class:`HypergraphState`: one hub per
    block, signed member↔hub incidence. # Preconditions every member ``< n``; each block non-empty.
    # Postconditions ``n_vertices == n + len(blocks)`` (base + one hub per block)."""
    if any(not block for block in blocks):
        raise ValueError("a design block is empty")
    if any(not 0 <= m < n for block in blocks for m in block):
        raise ValueError("a design block references an out-of-range vertex")
    base = HypergraphState(tuple(f"n{i}" for i in range(n)),
                           np.zeros((0, 2), np.int64), np.zeros((0,), np.int64), topo_hash=f"design:{tag}")
    return base.with_task_hyperedges(
        new_entities=[], hyperedges=[(f"{tag}_b{i}", tuple(block)) for i, block in enumerate(blocks)])


def steiner_blocks(n: int) -> "list[tuple[int, ...]]":
    """The triples of ``S(2,3,n)`` (subsets of ``range(n)``). # Preconditions ``n ∈ {7, 9}``."""
    if n == 7:                                                 # Fano plane: 7 points, 7 lines
        return [(0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 0), (5, 6, 1), (6, 0, 2)]
    if n == 9:                                                 # affine plane AG(2,3): point (x,y) -> 3x+y
        blocks = [tuple(3 * x + ((m * x + b) % 3) for x in range(3)) for m in range(3) for b in range(3)]
        return blocks + [(3 * c, 3 * c + 1, 3 * c + 2) for c in range(3)]   # vertical lines (slope ∞)
    raise ValueError(f"STS implemented for n in {{7, 9}}; got {n} (exists iff n≡1,3 mod 6)")


def steiner_triple_system(n: int) -> HypergraphState:
    """``S(2,3,n)`` — every pair on exactly one triple (``n ∈ {7, 9}``)."""
    return design_to_state(n, steiner_blocks(n), tag=f"sts{n}")


def sunflower_cover(n: int, *, core_size: int) -> "list[tuple[int, ...]]":
    """Sunflower blocks that *partition* ``range(n)``: a shared core ``0..core_size-1`` plus the remaining
    vertices split into disjoint petals (one block per petal, core ∪ petal). For augmenting an existing body set
    (a shared coordinator over all bodies + independent limbs). # Preconditions ``0 < core_size < n``."""
    if not 0 < core_size < n:
        raise ValueError(f"core_size must be in (0, {n}); got {core_size}")
    core = tuple(range(core_size))
    rest = list(range(core_size, n))
    n_petals = max(2, round((n - core_size) ** 0.5))
    petals = [rest[i::n_petals] for i in range(n_petals)]
    return [core + tuple(p) for p in petals if p]


def k_uniform_blocks(n: int, k: int, *, n_blocks: int, seed: int = 0) -> "list[tuple[int, ...]]":
    """``n_blocks`` distinct ``k``-subsets of ``range(n)`` (the blocks of a random k-uniform hypergraph).
    # Preconditions ``1 <= k <= n``; ``n_blocks <= C(n,k)``."""
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    while len(seen) < n_blocks:
        seen.add(tuple(sorted(int(v) for v in rng.choice(n, size=k, replace=False))))
    return sorted(seen)


def k_uniform_random(n: int, k: int, *, n_blocks: int, seed: int = 0) -> HypergraphState:
    """A random ``k``-uniform hypergraph: ``n_blocks`` distinct ``k``-subsets of ``n`` vertices."""
    return design_to_state(n, k_uniform_blocks(n, k, n_blocks=n_blocks, seed=seed), tag=f"kunif{k}")


def sunflower(core_size: int, petal_size: int, n_petals: int) -> HypergraphState:
    """A sunflower: ``n_petals`` blocks sharing a common ``core`` of ``core_size`` vertices, each with
    ``petal_size`` private (pairwise-disjoint) petal vertices. # Postconditions
    ``n = core_size + n_petals·petal_size`` base vertices; every two blocks intersect exactly in the core."""
    core = tuple(range(core_size))
    blocks, nxt = [], core_size
    for _ in range(n_petals):
        petal = tuple(range(nxt, nxt + petal_size))
        nxt += petal_size
        blocks.append(core + petal)
    return design_to_state(nxt, blocks, tag="sunflower")


def closed_neighbourhood_blocks(graph: HypergraphState) -> "list[tuple[int, ...]]":
    """Each vertex's closed neighbourhood ``{v} ∪ N(v)`` as a block — the natural hyperedge lift of a graph
    (``k``-uniform with ``k = d+1`` when the graph is ``d``-regular; well-defined for any graph, incl.\ the
    triangle-free Petersen, where a clique/triangle lift would yield nothing). One block per vertex."""
    n = graph.n_vertices
    nb: "dict[int, set[int]]" = {i: {i} for i in range(n)}
    for a, b in graph.edges.tolist():
        nb[int(a)].add(int(b))
        nb[int(b)].add(int(a))
    return [tuple(sorted(nb[i])) for i in range(n)]


def graph_to_kuniform(graph: HypergraphState, *, tag: str) -> HypergraphState:
    """Lift a graph to its closed-neighbourhood HYPERGRAPH: each vertex's ``{v} ∪ N(v)`` becomes a hyperedge
    (star-expanded into a hub). The hub-mediated walks (member → hub → member) enrich the ``B^L`` walk basis in a
    way a 2-uniform graph cannot --- the hypergraph version of the graph. ``n_vertices == 2n`` (points + one hub
    per vertex)."""
    return design_to_state(graph.n_vertices, closed_neighbourhood_blocks(graph), tag=tag)


# Registry of design families (each a thunk over its natural parameters at n≈9 base vertices).
DESIGNS = {
    "steiner9": lambda: steiner_triple_system(9),
    "fano7": lambda: steiner_triple_system(7),
    "kuniform3": lambda: k_uniform_random(9, 3, n_blocks=8, seed=0),
    "kuniform4": lambda: k_uniform_random(9, 4, n_blocks=8, seed=0),
    "sunflower": lambda: sunflower(core_size=2, petal_size=2, n_petals=3),
}
