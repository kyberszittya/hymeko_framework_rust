"""Stage 0 — degree/sign-preserving scramble of a signed graph's incidence (the H2 causal ablation).

The structural-leverage hypothesis (``reports/2026-07-10-hsikan-gomb-soma-structural-leverage-hypothesis-prep.md``)
needs a control that **destroys the task-relevant structure while preserving the easy marginal statistics**, so a
HSiKAN advantage that survives it cannot be attributed to structure. This module is that control.

The operation is a **signed double-edge swap** (the standard degree-preserving rewiring, hand-rolled — no networkx
dependency, §1): within each sign class independently, repeatedly replace two undirected edges ``(a,b),(c,d)`` by
``(a,d),(c,b)`` when that introduces no self-loop or duplicate. This is applied to the *undirected* signed graph
(the structural-probe / symmetric form) and re-expanded to the two-arc-per-edge representation
:class:`~hymeko_rl.agents.hypergraph_state.HypergraphState` uses.

**Preserved (exactly):** node count; positive-edge count; negative-edge count; the per-node *signed degree
sequence* (each node keeps its number of ``+`` and ``-`` incident edges); the symmetric two-arc encoding.
**Destroyed:** *which* endpoints carry each signed relation — so cycles, the frustrated triangle, and the ``B²x``
reachability pattern the ``structural`` target depends on are randomised. Easy marginals (degree, sign balance,
node count) are held fixed; higher-order signed structure is not.

Directed/antisymmetric graphs (e.g. the kinematic ``from_mjcf`` hypergraph, where a joint's two arcs carry
*opposite* signs) are rejected — they need a directed-swap variant, deferred until a structural rung requires it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hymeko_rl.agents.hypergraph_state import HypergraphState

# An undirected signed edge: (u, v, sign) with u < v and sign in {+1, -1}.
_UEdge = tuple[int, int, int]


def _undirected_signed_edges(hg: HypergraphState) -> list[_UEdge]:
    """Collapse the two-arc-per-edge encoding to undirected signed edges, asserting symmetry.

    # Preconditions the graph is a *symmetric* signed graph: for every arc ``(a,b,s)`` the mirror ``(b,a,s)``
      is present with the **same** sign, and no unordered pair carries two different signs (no self-loops).
    # Postconditions returns one ``(min,max,sign)`` per undirected edge, deduplicated and sorted.
    # Raises ``ValueError`` if the graph is not a symmetric signed graph.
    """
    arcs = {(int(a), int(b), int(s)) for (a, b), s in zip(hg.edges, hg.signs)}
    sign_of: dict[tuple[int, int], int] = {}
    for a, b, s in arcs:
        if a == b:
            raise ValueError(f"self-loop arc ({a},{b}) — not a simple signed graph")
        if (b, a, s) not in arcs:
            raise ValueError(
                "scramble requires a symmetric signed graph (missing mirror arc); directed/antisymmetric "
                "kinematic graphs need the directed-swap variant, not yet implemented")
        key = (min(a, b), max(a, b))
        if sign_of.setdefault(key, s) != s:
            raise ValueError(f"edge {key} carries two signs — not a well-signed graph")
    return sorted((u, v, sg) for (u, v), sg in sign_of.items())


def _double_edge_swap(edges: list[tuple[int, int]], rng: np.random.Generator, n_swaps: int,
                      occupied: set[frozenset]) -> int:
    """Degree-preserving double-edge swap **in place** on a single sign class; returns #accepted swaps.

    Picks two distinct edges ``(a,b),(c,d)``, rewires to ``(a,d),(c,b)`` iff no self-loop and neither target
    pair is already used **by any sign** — ``occupied`` (shared across sign classes) tracks every used unordered
    pair, so the result is a *simple signed graph*: no pair carries two edges (in particular no ``+``/``−``
    parallel edge). Each node's degree within this sign class is invariant under the swap.

    # Invariants ``len(edges)`` and every node's per-sign degree are unchanged; ``occupied`` stays consistent
      with ``edges`` (old pairs removed, new pairs added); the global edge set stays simple.
    """
    if len(edges) < 2:
        return 0
    accepted = 0
    for _ in range(n_swaps):
        i, j = rng.integers(0, len(edges), size=2)
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if rng.random() < 0.5:                       # randomise which endpoints pair up
            c, d = d, c
        if len({a, b, c, d}) < 4:                    # shared endpoint → would self-loop or collapse
            continue
        new_i, new_j = frozenset((a, d)), frozenset((c, b))
        if new_i in occupied or new_j in occupied or new_i == new_j:
            continue
        occupied.discard(frozenset(edges[i]))
        occupied.discard(frozenset(edges[j]))
        edges[i] = (a, d)
        edges[j] = (c, b)
        occupied.add(new_i)
        occupied.add(new_j)
        accepted += 1
    return accepted


@dataclass(frozen=True)
class ScrambleStats:
    """What the scramble preserved vs destroyed (for the report and the tests)."""

    n_vertices: int
    n_pos_edges: int
    n_neg_edges: int
    signed_degree_preserved: bool
    n_edges_changed: int
    frac_incidence_changed: float
    accepted_swaps: int


def signed_degree_sequences(hg: HypergraphState) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Sorted per-node ``(positive-degree, negative-degree)`` sequences — the invariant the swap preserves.

    # Postconditions two length-``n_vertices`` tuples, each sorted ascending (order-independent comparison)."""
    n = hg.n_vertices
    pos = [0] * n
    neg = [0] * n
    for u, v, s in _undirected_signed_edges(hg):
        tgt = pos if s > 0 else neg
        tgt[u] += 1
        tgt[v] += 1
    return tuple(sorted(pos)), tuple(sorted(neg))


def _to_hypergraph(vertex_labels: tuple[str, ...], uedges: list[_UEdge], topo_hash: str) -> HypergraphState:
    """Re-expand undirected signed edges to the two-arc-per-edge :class:`HypergraphState` encoding."""
    edges: list[tuple[int, int]] = []
    signs: list[int] = []
    for u, v, s in uedges:
        edges += [(u, v), (v, u)]
        signs += [s, s]
    return HypergraphState(vertex_labels=vertex_labels,
                           edges=np.asarray(edges, np.int64), signs=np.asarray(signs, np.int64),
                           topo_hash=topo_hash)


def scramble_signed_incidence(hg: HypergraphState, *, seed: int, swaps_per_edge: int = 20,
                              ) -> HypergraphState:
    """Return a degree/sign-preserving scramble of ``hg`` (a new symmetric :class:`HypergraphState`).

    Signed double-edge swaps within each sign class randomise which endpoints carry each relation while holding
    node count, per-sign edge counts, and the per-node signed degree sequence fixed. Deterministic in ``seed``.

    # Preconditions ``hg`` is a symmetric signed graph (see :func:`_undirected_signed_edges`); ``swaps_per_edge >= 0``.
    # Postconditions the result has identical ``vertex_labels`` and signed degree sequences; ``topo_hash`` is
      suffixed ``":scramble:{seed}"``; the same ``seed`` yields the same scramble.
    """
    if swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be >= 0")
    rng = np.random.default_rng(seed)
    uedges = _undirected_signed_edges(hg)
    occupied = {frozenset((u, v)) for u, v, _s in uedges}   # shared across sign classes → no parallel signed edge
    by_sign: dict[int, list[tuple[int, int]]] = {+1: [], -1: []}
    for u, v, s in uedges:
        by_sign[s].append((u, v))
    for s in (+1, -1):                               # swap each sign class (per-sign degree preserved; shared occupancy)
        _double_edge_swap(by_sign[s], rng, swaps_per_edge * max(1, len(by_sign[s])), occupied)
    scrambled = sorted([(min(u, v), max(u, v), s) for s in (+1, -1) for (u, v) in by_sign[s]])
    return _to_hypergraph(hg.vertex_labels, scrambled, hg.topo_hash + f":scramble:{seed}")


def directed_signed_degree_sequences(hg: HypergraphState,
                                     ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Sorted per-node ``(out+, in+, out-, in-)`` degree sequences — the invariant the *directed* swap preserves.

    For a directed signed graph (e.g. a DirectLiNGAM causal hypergraph, cause→effect arcs), the meaningful
    invariant is per-node out/in degree *per sign*, not the undirected degree. # Postconditions four length-``N``
    tuples, each sorted ascending."""
    n = hg.n_vertices
    out_p, in_p, out_n, in_n = ([0] * n for _ in range(4))
    for (src, dst), s in zip(hg.edges, hg.signs):
        (out_p if s > 0 else out_n)[int(src)] += 1
        (in_p if s > 0 else in_n)[int(dst)] += 1
    return tuple(sorted(out_p)), tuple(sorted(in_p)), tuple(sorted(out_n)), tuple(sorted(in_n))


def _directed_swap(arcs: list[tuple[int, int]], rng: np.random.Generator, n_swaps: int,
                   occupied: set[tuple[int, int]]) -> int:
    """Directed degree-preserving double-edge swap **in place** on one sign class; returns #accepted swaps.

    Picks two arcs ``a→b, c→d`` and rewires to ``a→d, c→b`` iff no self-loop and neither target arc already
    exists (``occupied`` is the ordered-pair set across *all* signs → the result stays a simple directed signed
    graph). Each node's out- and in-degree within this sign class is invariant.
    """
    if len(arcs) < 2:
        return 0
    accepted = 0
    for _ in range(n_swaps):
        i, j = rng.integers(0, len(arcs), size=2)
        if i == j:
            continue
        a, b = arcs[i]
        c, d = arcs[j]
        if a == d or c == b:                         # would create a self-loop
            continue
        new_i, new_j = (a, d), (c, b)
        if new_i in occupied or new_j in occupied or new_i == new_j:
            continue
        occupied.discard(arcs[i])
        occupied.discard(arcs[j])
        arcs[i] = new_i
        arcs[j] = new_j
        occupied.add(new_i)
        occupied.add(new_j)
        accepted += 1
    return accepted


def scramble_directed_signed_incidence(hg: HypergraphState, *, seed: int, swaps_per_edge: int = 20,
                                       ) -> HypergraphState:
    """Directed degree/sign-preserving scramble of a **directed** signed graph (e.g. a causal hypergraph).

    Unlike :func:`scramble_signed_incidence` (symmetric), this keeps arcs directed and preserves each node's
    per-sign **out- and in-degree** while randomising which cause connects to which effect — the H2 causal
    control for the LiNGAM-SH → HSiKAN mechanism model. Deterministic in ``seed``.

    # Preconditions ``hg`` is a simple directed signed graph (no self-loops, no duplicate arc of the same sign);
      ``swaps_per_edge >= 0``.
    # Postconditions identical ``vertex_labels``; identical ``(out±, in±)`` degree sequences; ``topo_hash``
      suffixed ``":dscramble:{seed}"``.
    """
    if swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be >= 0")
    rng = np.random.default_rng(seed)
    arcs_by_sign: dict[int, list[tuple[int, int]]] = {+1: [], -1: []}
    occupied: set[tuple[int, int]] = set()
    for (src, dst), s in zip(hg.edges, hg.signs):
        a = (int(src), int(dst))
        if a[0] == a[1]:
            raise ValueError(f"self-loop arc {a} — not a simple directed graph")
        if a in occupied:
            raise ValueError(f"duplicate arc {a} — not a simple directed graph")
        occupied.add(a)
        arcs_by_sign[int(np.sign(s)) or 1].append(a)
    for s in (+1, -1):
        _directed_swap(arcs_by_sign[s], rng, swaps_per_edge * max(1, len(arcs_by_sign[s])), occupied)
    edges: list[tuple[int, int]] = []
    signs: list[int] = []
    for s in (+1, -1):
        for a in arcs_by_sign[s]:
            edges.append(a)
            signs.append(s)
    return HypergraphState(vertex_labels=hg.vertex_labels,
                           edges=np.asarray(edges, np.int64), signs=np.asarray(signs, np.int64),
                           topo_hash=hg.topo_hash + f":dscramble:{seed}")


def _weighted_directed_swap(arcs: list[list[float]], rng: np.random.Generator, n_swaps: int,
                            occupied: set[tuple[int, int]]) -> int:
    """Weight-carrying directed double-edge swap **in place** on one sign class; returns #accepted swaps.

    Each arc is ``[src, dst, weight]``. Swapping ``a→b, c→d`` to ``a→d, c→b`` carries each arc's weight with its
    **source** (``a`` keeps its outgoing weight, now to ``d``), so per-node out-degree, in-degree, and the weight
    multiset are all preserved while the cause→effect wiring is randomised. ``occupied`` (ordered pairs, all signs)
    keeps the result a simple directed graph.
    """
    if len(arcs) < 2:
        return 0
    accepted = 0
    for _ in range(n_swaps):
        i, j = rng.integers(0, len(arcs), size=2)
        if i == j:
            continue
        a, b, _wi = arcs[i]
        c, d, _wj = arcs[j]
        ai, bi, ci, di = int(a), int(b), int(c), int(d)
        if ai == di or ci == bi:                     # would create a self-loop
            continue
        new_i, new_j = (ai, di), (ci, bi)
        if new_i in occupied or new_j in occupied or new_i == new_j:
            continue
        occupied.discard((ai, bi))
        occupied.discard((ci, di))
        arcs[i][1] = di                              # a → d (weight arcs[i][2] unchanged)
        arcs[j][1] = bi                              # c → b (weight arcs[j][2] unchanged)
        occupied.add(new_i)
        occupied.add(new_j)
        accepted += 1
    return accepted


def scramble_signed_operator(a_pos: np.ndarray, a_neg: np.ndarray, *, seed: int, swaps_per_edge: int = 20,
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Degree/sign/weight-preserving directed scramble of a signed causal operator ``(A⁺, A⁻)``.

    The matrix convention is ``A[dst, src] = A[effect, cause]`` (HSiKAN's receiver-from-sender, matching LiNGAM's
    ``B[effect, cause]``). Within each sign matrix, arcs ``cause → effect`` are rewired by a weight-carrying directed
    double-edge swap: per-node out-/in-degree **and** the weight multiset are preserved; which cause drives which
    effect is randomised. Deterministic in ``seed``. This is the H2 decider for the operator harness.

    # Preconditions ``a_pos, a_neg`` are square ``(n, n)`` with the same shape, nonnegative, disjoint support
      (``a_pos * a_neg == 0``), zero diagonal; ``swaps_per_edge >= 0``.
    # Postconditions same shape; same per-sign nonzero count; same nonzero-value multiset per sign; zero diagonal.
    """
    if a_pos.shape != a_neg.shape or a_pos.ndim != 2 or a_pos.shape[0] != a_pos.shape[1]:
        raise ValueError("a_pos and a_neg must be square matrices of equal shape")
    if swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be >= 0")
    rng = np.random.default_rng(seed)
    occupied: set[tuple[int, int]] = set()
    arcs_by_sign: dict[int, list[list[float]]] = {+1: [], -1: []}
    for mat, sign in ((a_pos, +1), (a_neg, -1)):
        rows, cols = np.nonzero(mat)
        for i, j in zip(rows, cols):                 # i = effect (dst), j = cause (src)
            if int(i) == int(j):
                raise ValueError("operator has a nonzero diagonal (self-loop) — zero it first")
            arc = (int(j), int(i))
            if arc in occupied:
                raise ValueError(f"duplicate arc {arc} across signs — supports must be disjoint")
            occupied.add(arc)
            arcs_by_sign[sign].append([float(j), float(i), float(mat[i, j])])
    for sign in (+1, -1):
        _weighted_directed_swap(arcs_by_sign[sign], rng, swaps_per_edge * max(1, len(arcs_by_sign[sign])), occupied)
    out_pos = np.zeros_like(a_pos)
    out_neg = np.zeros_like(a_neg)
    for sign, out in ((+1, out_pos), (-1, out_neg)):
        for src, dst, w in arcs_by_sign[sign]:
            out[int(dst), int(src)] = w              # A[effect, cause] = weight
    return out_pos, out_neg


def scramble_stats(original: HypergraphState, scrambled: HypergraphState) -> ScrambleStats:
    """Preserved/destroyed diagnostics comparing an original graph to its scramble (drives the §5 report table)."""
    o_edges = {(u, v, s) for u, v, s in _undirected_signed_edges(original)}
    s_edges = {(u, v, s) for u, v, s in _undirected_signed_edges(scrambled)}
    o_pos, o_neg = signed_degree_sequences(original)
    s_pos, s_neg = signed_degree_sequences(scrambled)
    changed = len(o_edges ^ s_edges) // 2            # symmetric-difference edges, halved (each change = -1 old +1 new)
    n_pos = sum(1 for _u, _v, s in o_edges if s > 0)
    n_neg = len(o_edges) - n_pos
    return ScrambleStats(
        n_vertices=original.n_vertices, n_pos_edges=n_pos, n_neg_edges=n_neg,
        signed_degree_preserved=(o_pos == s_pos and o_neg == s_neg),
        n_edges_changed=changed,
        frac_incidence_changed=round(changed / max(1, len(o_edges)), 4),
        accepted_swaps=changed)
