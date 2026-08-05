r"""Hypergraph zoo — canonical hypergraph constructions as verifiable benchmarks for the HyMeKo / Nagare line.

A HyMeKo model IS a (signed-incidence) hypergraph, so a catalogue of canonical hypergraphs gives principled
benchmarks for the signed / holonomy / cycle-consistency work, for constraint/matroid structure ("which
constraints together cause a dependency?"), and for hypergraph spectral / neural methods. Each generator returns a
:class:`Hypergraph` whose defining combinatorial properties are checkable (and are pinned by the tests).

Families (the ones flagged most relevant to HyMeKo–Nagare, plus the standard companions):
  - Fano plane ``PG(2,2)`` and finite projective planes ``PG(2,q)`` — minimal / scalable regular linear designs;
  - Steiner triple systems ``S(2,3,v)`` — global order from a local covering rule;
  - complete uniform ``K_n^{(k)}`` and Kneser ``KG_r(n,k)`` hypergraphs;
  - loose and tight cycles — feedback / temporal structure;
  - sunflowers (Δ-systems) — a shared core with disjoint petals;
  - random ``H^{(k)}(n,p)`` — phase transitions / robustness;
  - matroid circuit hypergraphs (graphic, Fano matroid) — minimal dependencies / constraints;
  - simplicial complexes (a simplex boundary) — the bridge to algebraic topology.

# Preconditions: generator arguments in their stated domains (``q`` prime; ``v ≡ 1,3 mod 6`` for an STS; …).
# Postconditions: the returned hypergraph satisfies the family's defining property (verifiable via the methods).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

Edge = frozenset


@dataclass(frozen=True)
class Hypergraph:
    """A hypergraph as a vertex count + a tuple of hyperedges (each a frozenset of vertex indices)."""

    n_vertices: int
    edges: "tuple[frozenset, ...]"

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def incidence_matrix(self) -> np.ndarray:
        """(n_vertices × n_edges) 0/1 incidence matrix ``B`` with ``B[v,e] = 1`` iff ``v ∈ edge_e``."""
        b = np.zeros((self.n_vertices, self.n_edges), dtype=int)
        for e, edge in enumerate(self.edges):
            for v in edge:
                b[v, e] = 1
        return b

    def edge_sizes(self) -> "list[int]":
        return [len(e) for e in self.edges]

    def is_uniform(self, k: "int | None" = None) -> bool:
        sizes = set(self.edge_sizes())
        return len(sizes) == 1 and (k is None or next(iter(sizes)) == k)

    def degree_sequence(self) -> np.ndarray:
        return self.incidence_matrix().sum(axis=1)

    def is_regular(self, r: "int | None" = None) -> bool:
        degs = set(self.degree_sequence().tolist())
        return len(degs) == 1 and (r is None or next(iter(degs)) == r)

    def is_linear(self) -> bool:
        """Linear: every two distinct hyperedges meet in at most one vertex."""
        return all(len(a & b) <= 1 for a, b in combinations(self.edges, 2))

    def pair_coverage(self) -> "dict[frozenset, int]":
        """How many hyperedges contain each vertex pair (a ``2-(v,k,λ)`` design has this constant = λ)."""
        counts: "dict[frozenset, int]" = {}
        for edge in self.edges:
            for pair in combinations(sorted(edge), 2):
                key = frozenset(pair)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def is_2_design(self) -> "tuple[bool, int]":
        """Is every vertex pair in exactly the same number λ of edges? Returns (holds, λ)."""
        counts = self.pair_coverage()
        vals = set(counts.values())
        full = len(counts) == self.n_vertices * (self.n_vertices - 1) // 2
        return (full and len(vals) == 1), (next(iter(vals)) if len(vals) == 1 else -1)

    def dual(self) -> "Hypergraph":
        """The dual hypergraph: vertices ↔ edges (edge e of the dual = the set of original edges through vertex e)."""
        rows = [frozenset(e for e, edge in enumerate(self.edges) if v in edge) for v in range(self.n_vertices)]
        return Hypergraph(self.n_edges, tuple(rows))

    def is_self_dual(self) -> bool:
        """Self-dual up to relabelling: the incidence matrix equals its transpose under some row/col permutation."""
        b = self.incidence_matrix()
        if b.shape[0] != b.shape[1]:
            return False
        target = np.sort(np.sort(b, axis=0), axis=1)
        dual = np.sort(np.sort(b.T, axis=0), axis=1)
        return bool(np.array_equal(target, dual))


# ---- projective / affine planes -------------------------------------------------------------------------

def projective_plane(q: int) -> Hypergraph:
    r"""The finite projective plane ``PG(2,q)`` for prime ``q`` — ``q²+q+1`` points and lines, ``(q+1)``-uniform, linear.

    Points = normalised nonzero vectors of ``GF(q)³`` (leading nonzero coordinate 1); lines carry the same
    coordinates, with point ``p`` on line ``L`` iff ``L·p ≡ 0 (mod q)``. ``q = 2`` is the Fano plane.
    """
    if q < 2 or any(q % d == 0 for d in range(2, q)):
        raise ValueError(f"q must be prime for this construction, got {q}")
    reps = []
    for vec in np.ndindex(q, q, q):
        if vec == (0, 0, 0):
            continue
        lead = next(x for x in vec if x != 0)
        inv = pow(int(lead), q - 2, q)                      # Fermat inverse in GF(q)
        reps.append(tuple((x * inv) % q for x in vec))
    points = sorted(set(reps))
    index = {p: i for i, p in enumerate(points)}
    lines = []
    for coeff in points:                                    # lines share the point coordinates (duality)
        line = frozenset(index[p] for p in points if sum(a * b for a, b in zip(coeff, p)) % q == 0)
        lines.append(line)
    return Hypergraph(len(points), tuple(lines))


def fano_plane() -> Hypergraph:
    """The Fano plane ``PG(2,2)`` = ``S(2,3,7)``: 7 points, 7 lines, 3-uniform, 3-regular, self-dual, ``2-(7,3,1)``."""
    return projective_plane(2)


# ---- Steiner triple systems -----------------------------------------------------------------------------

def steiner_triple_system(v: int, seed: int = 0) -> Hypergraph:
    r"""A Steiner triple system ``S(2,3,v)`` (``v ≡ 1 or 3 mod 6``): every vertex pair is in exactly one triple.

    ``v = 7`` is the Fano plane and ``v = 9`` the affine plane ``AG(2,3)``; other admissible ``v`` use a
    randomised greedy triple cover with restarts (deterministic under ``seed``), verified to cover every pair
    exactly once. # Postconditions: ``is_2_design() == (True, 1)`` and ``is_uniform(3)``.
    """
    import random
    if v % 6 not in (1, 3):
        raise ValueError(f"an S(2,3,v) exists only for v ≡ 1 or 3 (mod 6), got v={v}")
    if v == 7:
        return fano_plane()
    if v == 9:
        return affine_plane(3)
    total = v * (v - 1) // 2
    rng = random.Random(seed)
    for _ in range(2000):                                   # most-constrained-first + random tie-break + restarts
        used, triples = set(), []
        remaining = {frozenset(p) for p in combinations(range(v), 2)}
        stuck = False
        while remaining:
            best, best_cands = None, None
            for pair in remaining:
                a, b = tuple(pair)
                cands = [c for c in range(v) if c not in (a, b)
                         and frozenset((a, c)) not in used and frozenset((b, c)) not in used]
                if best_cands is None or len(cands) < len(best_cands):
                    best, best_cands = (a, b), cands
                    if len(cands) <= 1:
                        break                               # can't do better than a forced/dead pair
            if not best_cands:
                stuck = True
                break
            a, b = best
            c = rng.choice(best_cands)
            triples.append(frozenset((a, b, c)))
            new = {frozenset((a, b)), frozenset((a, c)), frozenset((b, c))}
            used |= new
            remaining -= new
        if not stuck and len(used) == total:
            return Hypergraph(v, tuple(triples))
    raise ValueError(f"greedy STS did not close for v={v}; a designed (Bose/Skolem) construction is needed")


def affine_plane(q: int) -> Hypergraph:
    r"""The affine plane ``AG(2,q)`` for prime ``q`` — ``q²`` points, ``q²+q`` lines, ``q``-uniform, ``S(2,q,q²)``."""
    if q < 2 or any(q % d == 0 for d in range(2, q)):
        raise ValueError(f"q must be prime, got {q}")
    points = [(x, y) for x in range(q) for y in range(q)]
    index = {p: i for i, p in enumerate(points)}
    lines = []
    for m in range(q):                                      # non-vertical lines y = m x + c
        for c in range(q):
            lines.append(frozenset(index[(x, (m * x + c) % q)] for x in range(q)))
    for x0 in range(q):                                     # vertical lines x = x0
        lines.append(frozenset(index[(x0, y)] for y in range(q)))
    return Hypergraph(len(points), tuple(lines))


# ---- complete / Kneser ----------------------------------------------------------------------------------

def complete_uniform(n: int, k: int) -> Hypergraph:
    r"""``K_n^{(k)}``: every ``k``-subset of ``n`` vertices is a hyperedge (``K_n^{(2)} = K_n``)."""
    return Hypergraph(n, tuple(frozenset(c) for c in combinations(range(n), k)))


def kneser(n: int, k: int, r: int = 2) -> Hypergraph:
    r"""``KG_r(n,k)``: vertices = ``k``-subsets of ``[n]``; a hyperedge = ``r`` pairwise-disjoint such subsets."""
    subsets = list(combinations(range(n), k))
    index = {frozenset(s): i for i, s in enumerate(subsets)}
    edges = []
    for combo in combinations(subsets, r):
        flat = [x for s in combo for x in s]
        if len(set(flat)) == len(flat):                     # pairwise disjoint
            edges.append(frozenset(index[frozenset(s)] for s in combo))
    return Hypergraph(len(subsets), tuple(edges))


# ---- cycles / paths -------------------------------------------------------------------------------------

def loose_cycle(k: int, length: int) -> Hypergraph:
    r"""A loose (linear) ``k``-uniform cycle of ``length`` edges: consecutive edges share exactly one vertex."""
    if length < 2:
        raise ValueError("length must be ≥ 2")
    step = k - 1
    n = length * step
    edges = [frozenset((i * step + j) % n for j in range(k)) for i in range(length)]
    return Hypergraph(n, tuple(edges))


def tight_cycle(k: int, n: int) -> Hypergraph:
    r"""A tight ``k``-uniform cycle on ``n`` vertices: edge ``i`` = ``{i, i+1, …, i+k−1} (mod n)``."""
    if n < k + 1:
        raise ValueError("need n ≥ k+1 for a tight cycle")
    edges = [frozenset((i + j) % n for j in range(k)) for i in range(n)]
    return Hypergraph(n, tuple(edges))


# ---- sunflower / random / matroid / simplicial ----------------------------------------------------------

def sunflower(core_size: int, n_petals: int, petal_size: int) -> Hypergraph:
    r"""A sunflower (Δ-system): every two edges meet in the same ``core`` of ``core_size`` vertices; disjoint petals."""
    core = list(range(core_size))
    edges, nxt = [], core_size
    for _ in range(n_petals):
        petal = list(range(nxt, nxt + petal_size))
        nxt += petal_size
        edges.append(frozenset(core + petal))
    return Hypergraph(nxt, tuple(edges))


def random_uniform(n: int, k: int, p: float, seed: int = 0) -> Hypergraph:
    r"""The Erdős–Rényi hypergraph ``H^{(k)}(n,p)``: each ``k``-subset is an edge independently with probability ``p``."""
    rng = np.random.RandomState(seed)
    edges = [frozenset(c) for c in combinations(range(n), k) if rng.random() < p]
    return Hypergraph(n, tuple(edges))


def graphic_matroid_circuits(edges: "list[tuple[int, int]]") -> Hypergraph:
    r"""Circuit hypergraph of a graphic matroid: hyperedges = the (edge sets of) simple cycles of the input graph.

    Vertices of the hypergraph are the graph's EDGES (indices into ``edges``); each hyperedge is a minimal
    dependent set = a simple cycle — the "which constraints together are dependent?" view.
    """
    m = len(edges)
    circuits = []
    for size in range(3, min(m, 10) + 1):                   # a circuit = a simple cycle: 2-regular + connected
        for subset in combinations(range(m), size):
            deg: "dict[int, int]" = {}
            adj: "dict[int, list[int]]" = {}
            for ei in subset:
                u, v = edges[ei]
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
                adj.setdefault(u, []).append(v)
                adj.setdefault(v, []).append(u)
            if len(deg) != size or any(d != 2 for d in deg.values()):
                continue                                    # not 2-regular with |V| = |E| ⇒ not a single cycle
            seen, stack = set(), [next(iter(deg))]           # connectivity ⇒ exactly one cycle
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(adj[x])
            if len(seen) == size:
                circuits.append(frozenset(subset))
    return Hypergraph(m, tuple(circuits))


def simplex_boundary(n: int) -> Hypergraph:
    r"""The boundary complex of the ``(n−1)``-simplex: hyperedges = all ``(n−1)``-subsets (the facets) of ``n`` vertices.

    A downward-closed reading (a simplicial complex) is the union of all subsets of the facets; here we return the
    top faces, the natural generators, so ``dual`` / ``incidence`` expose the combinatorial boundary structure.
    """
    return Hypergraph(n, tuple(frozenset(c) for c in combinations(range(n), n - 1)))
