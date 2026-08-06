"""Topology zoo — generate typical hypergraph families as :class:`HypergraphState`s of matched size N.

Phase 1 of Kato's isomorphic-controllers program (``docs/plans/2026-06-27-isomorphic-controllers-from-hypergraphs``):
a controller's interconnection structure IS a hypergraph; to ask *which topology controls which plant best* we
first need a family of topologies over the same N vertices. Each builder returns a signed
:class:`~hymeko_rl.agents.hypergraph_state.HypergraphState` (random ±1 per undirected edge, both directions) so the
signed message-passing controller (``SignedKANBackbone``) has a genuine signed structure to reason over.

Families: chain, ring, star, balanced binary tree, 2-D grid (N a perfect square), small-world (ring + random
shortcuts), random ``G(n,p)``, complete. Reuses :func:`hymeko_rl.experiments.structural_probe.build_chain_graph` for the
chain (no duplicate).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from hymeko_rl.agents.hypergraph_state import HypergraphState
from hymeko_rl.experiments.structural_probe import build_chain_graph


def _signed_graph(n: int, undirected: "list[tuple[int, int]]", *, seed: int, tag: str) -> HypergraphState:
    """A signed :class:`HypergraphState` from ``undirected`` edges: each gets a random ±1 sign, expanded to
    both directions. # Preconditions every endpoint ``< n``. # Postconditions ``2·len(undirected)`` arcs."""
    rng = np.random.default_rng(seed)
    edges: list[tuple[int, int]] = []
    signs: list[int] = []
    for a, b in undirected:
        s = int(rng.choice([-1, 1]))
        edges += [(a, b), (b, a)]
        signs += [s, s]
    if not edges:
        raise ValueError(f"topology {tag!r} has no edges")
    return HypergraphState(tuple(f"n{i}" for i in range(n)), np.asarray(edges, np.int64),
                           np.asarray(signs, np.int64), topo_hash=f"{tag}{n}:{seed}")


def chain(n: int, *, seed: int = 0) -> HypergraphState:
    """A line ``0—1—…—(N−1)`` (reuses the structural-probe chain)."""
    return build_chain_graph(n, seed=seed)


def ring(n: int, *, seed: int = 0) -> HypergraphState:
    """A cycle ``0—1—…—(N−1)—0``."""
    return _signed_graph(n, [(i, (i + 1) % n) for i in range(n)], seed=seed, tag="ring")


def star(n: int, *, seed: int = 0) -> HypergraphState:
    """A hub (node 0) connected to all others."""
    return _signed_graph(n, [(0, i) for i in range(1, n)], seed=seed, tag="star")


def tree(n: int, *, seed: int = 0) -> HypergraphState:
    """A balanced binary tree (parent ``i`` → children ``2i+1, 2i+2``)."""
    return _signed_graph(n, [(i, c) for i in range(n) for c in (2 * i + 1, 2 * i + 2) if c < n],
                         seed=seed, tag="tree")


def grid(n: int, *, seed: int = 0) -> HypergraphState:
    """A ``√N × √N`` 4-neighbour grid. # Preconditions ``N`` is a perfect square."""
    r = int(round(n ** 0.5))
    if r * r != n:
        raise ValueError(f"grid needs N a perfect square; got {n}")
    e: list[tuple[int, int]] = []
    for i in range(r):
        for j in range(r):
            v = i * r + j
            if j + 1 < r:
                e.append((v, v + 1))
            if i + 1 < r:
                e.append((v, v + r))
    return _signed_graph(n, e, seed=seed, tag="grid")


def small_world(n: int, *, seed: int = 0, shortcuts: int = 3) -> HypergraphState:
    """A ring with ``shortcuts`` random long-range edges (Watts–Strogatz-style)."""
    rng = np.random.default_rng(seed + 101)
    e = [(i, (i + 1) % n) for i in range(n)]
    for _ in range(shortcuts):
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b and (a, b) not in e and (b, a) not in e:
            e.append((a, b))
    return _signed_graph(n, e, seed=seed, tag="smallworld")


def random_gnp(n: int, *, seed: int = 0, p: float = 0.3) -> HypergraphState:
    """``G(n, p)`` — each pair connected with probability ``p`` (with a spanning chain so it stays connected)."""
    rng = np.random.default_rng(seed + 202)
    e = [(i, i + 1) for i in range(n - 1)]                  # spanning chain → connected, no isolated nodes
    for i in range(n):
        for j in range(i + 2, n):
            if rng.random() < p:
                e.append((i, j))
    return _signed_graph(n, e, seed=seed, tag="random")


def complete(n: int, *, seed: int = 0) -> HypergraphState:
    """All pairs connected (the densest topology)."""
    return _signed_graph(n, [(i, j) for i in range(n) for j in range(i + 1, n)], seed=seed, tag="complete")


# ── extremal / high-symmetry topologies (Phase 4: the interesting spectral & combinatorial invariants) ───────
_PETERSEN_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),    # outer 5-cycle
                   (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),    # inner pentagram
                   (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]    # spokes


def petersen(n: int = 10, *, seed: int = 0) -> HypergraphState:
    """The Petersen graph (natural N=10; ``n`` ignored): 3-regular, girth 5, strongly-regular (10,3,0,1), an
    expander (spectrum 3, 1^5, -2^4). The no-clique, uniform-coupling, well-conditioned topology."""
    return _signed_graph(10, _PETERSEN_EDGES, seed=seed, tag="petersen")


def kneser(n: int = 5, *, seed: int = 0, k: int = 2) -> HypergraphState:
    """Kneser graph K(n, k): vertices = k-subsets of [n], edges between DISJOINT subsets (``n`` is the set size,
    so the graph has C(n,k) vertices). K(5,2) IS the Petersen graph."""
    from itertools import combinations
    verts = list(combinations(range(n), k))
    idx = {v: i for i, v in enumerate(verts)}
    edges = [(idx[a], idx[b]) for i, a in enumerate(verts) for b in verts[i + 1:] if not set(a) & set(b)]
    return _signed_graph(len(verts), edges, seed=seed, tag=f"kneser{n}_{k}")


def _mycielskian(n: int, edges: "list[tuple[int, int]]") -> "tuple[int, list[tuple[int, int]]]":
    """Mycielskian μ(G): originals ``0..n-1``, shadows ``n..2n-1``, apex ``2n``. Shadow ``i`` ~ neighbours of
    ``i``; apex ~ all shadows. Returns ``(2n+1, edges)`` — raises χ by 1, preserves triangle-freeness."""
    adj: "dict[int, set[int]]" = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    out = list(edges)
    apex = 2 * n
    for i in range(n):
        out += [(n + i, j) for j in adj[i]]                  # shadow i ~ original neighbours of i
        out.append((apex, n + i))                            # apex ~ shadow i
    return 2 * n + 1, out


def grotzsch(n: int = 11, *, seed: int = 0) -> HypergraphState:
    """The Grötzsch graph = Mycielskian of C5 (natural N=11; ``n`` ignored): triangle-free, chromatic number 4 —
    the chromatic lift μ(G) with an apex hub."""
    nn, edges = _mycielskian(5, [(i, (i + 1) % 5) for i in range(5)])
    return _signed_graph(nn, edges, seed=seed, tag="grotzsch")


def expander(n: int = 12, *, seed: int = 0, d: int = 3) -> HypergraphState:
    """A random ``d``-regular graph — a near-Ramanujan expander (near-maximal spectral gap) with high
    probability. ``n`` is bumped by 1 if ``n*d`` is odd (a ``d``-regular graph needs ``n*d`` even)."""
    import networkx as nx
    if (n * d) % 2:
        n += 1
    g = nx.random_regular_graph(d, n, seed=seed)
    return _signed_graph(n, [(int(a), int(b)) for a, b in g.edges()], seed=seed, tag="expander")


# The zoo (Strategy registry) — name → builder. All accept (n, *, seed); fixed-N graphs ignore n (natural size).
TOPOLOGIES: "dict[str, Callable[..., HypergraphState]]" = {
    "chain": chain, "ring": ring, "star": star, "tree": tree, "grid": grid,
    "small_world": small_world, "random": random_gnp, "complete": complete,
    "petersen": petersen, "kneser": kneser, "grotzsch": grotzsch, "expander": expander,
}


# ── hypergraph lifts of the (regular) graph families — the closed-neighbourhood lift ({v}∪N(v) per vertex) ────
def _hyper_lift(builder: "Callable[..., HypergraphState]", tag: str) -> "Callable[..., HypergraphState]":
    """Wrap a graph builder so it returns the closed-neighbourhood HYPERGRAPH (points + one hub per vertex) —
    the hypergraph version of that graph, with hub-mediated walks the 2-uniform graph lacks."""
    from hymeko_rl.control.hypergraph_designs import graph_to_kuniform

    def build(n: int = 9, *, seed: int = 0) -> HypergraphState:
        return graph_to_kuniform(builder(n, seed=seed), tag=tag)
    return build


# The hypergraph version of each (regular) graph family — richer B^L walks via hub-mediated transport.
HYPER_TOPOLOGIES: "dict[str, Callable[..., HypergraphState]]" = {
    "petersen_h": _hyper_lift(petersen, "petersen_h"),
    "ring_h": _hyper_lift(ring, "ring_h"),
    "expander_h": _hyper_lift(expander, "expander_h"),
    "complete_h": _hyper_lift(complete, "complete_h"),
}


def permuted(hg: HypergraphState, perm: "np.ndarray") -> HypergraphState:
    """Relabel ``hg``'s vertices by ``perm`` (perm[i] = new label of old vertex i) — an isomorphic copy, for the
    permutation-invariance guard. # Preconditions ``perm`` is a permutation of ``range(n_vertices)``."""
    n = hg.n_vertices
    if sorted(int(p) for p in perm) != list(range(n)):
        raise ValueError("perm must be a permutation of range(n_vertices)")
    relabel = np.asarray(perm, dtype=np.int64)
    new_edges = relabel[hg.edges]                          # map both endpoints
    new_labels = tuple(hg.vertex_labels[int(np.where(relabel == i)[0][0])] for i in range(n))
    return HypergraphState(new_labels, new_edges, hg.signs.copy(), topo_hash=hg.topo_hash + ":perm")
