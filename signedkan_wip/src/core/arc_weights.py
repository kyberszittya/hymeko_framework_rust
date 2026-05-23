"""Arc-weight tooling for the ``inner_skip="cr_highway"`` HSIKAN mode.

The Rust cycle / walk enumerators in ``hymeko`` know about signs but
NOT weights — they emit the canonical (vertex, sign) tuple list. This
module post-processes the resulting :class:`SignedNTuple` lists,
reading per-edge arc weights from a :class:`WeightedSignedGraph` and
attaching them to each tuple's ``arc_weights`` field.

For training, the per-edge arc weights are converted to a per-vertex
``(T, k)`` tensor matching the layer's existing ``triad_v / triad_sigma``
shape. The per-vertex value is the mean of the two incident-edge
weights for cycles (cyclic), or the single / mean of incident edges
for walks.

Public surface
--------------

- ``build_edge_weight_lookup(wg)``: ``WeightedSignedGraph`` → undirected
  ``(min(u,v), max(u,v)) → float`` dict.
- ``annotate_arc_weights(tuples, lookup, *, is_walk=False)``: returns a
  new list of :class:`SignedNTuple` with the ``arc_weights`` field
  populated.
- ``per_vertex_arc_weights_array(tuples, *, is_walk=False)``: ``(T, k)``
  ``np.ndarray`` of per-vertex arc weights from a list of annotated
  tuples.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .n_tuples import SignedNTuple


def build_edge_weight_lookup(wg) -> dict[tuple[int, int], float]:
    """Undirected ``(min(u,v), max(u,v)) → float`` lookup from a
    :class:`WeightedSignedGraph`.

    Weights are expected to already be in $[-1, +1]$ (the
    ``load_continuous`` loader normalises Bitcoin's $[-10, +10]$
    ratings; other datasets are intrinsically $\\pm 1$).
    """
    out: dict[tuple[int, int], float] = {}
    edges = wg.edges
    weights = wg.weights
    for ei in range(len(edges)):
        u, v = int(edges[ei, 0]), int(edges[ei, 1])
        out[(min(u, v), max(u, v))] = float(weights[ei])
    return out


def annotate_arc_weights(tuples: Iterable[SignedNTuple],
                          lookup: dict[tuple[int, int], float],
                          *,
                          is_walk: bool = False) -> list[SignedNTuple]:
    """Return a new list of tuples with ``arc_weights`` populated
    from ``lookup``.

    ``is_walk``: if True, edges are
    ``(v_0, v_1), (v_1, v_2), ..., (v_{L-1}, v_L)`` (open walk).
    Otherwise edges are
    ``(v_0, v_1), (v_1, v_2), ..., (v_{k-1}, v_0)`` (closed cycle).
    """
    out: list[SignedNTuple] = []
    for t in tuples:
        verts = t.v
        if is_walk:
            edges = [(verts[i], verts[i + 1])
                     for i in range(len(verts) - 1)]
        else:
            k = len(verts)
            edges = [(verts[i], verts[(i + 1) % k])
                     for i in range(k)]
        ws: list[float] = []
        for u, v in edges:
            key = (min(int(u), int(v)), max(int(u), int(v)))
            # Missing edges shouldn't happen by construction (the cycle
            # / walk was enumerated from this graph), but default to 0
            # rather than KeyError to stay robust during development.
            ws.append(lookup.get(key, 0.0))
        # ``arity`` is on SignedNTuple but not on SignedTriad (the
        # legacy k=3 wrapper from hyperedges.py). Derive it from
        # len(v) so this helper works for both shapes.
        arity = getattr(t, "arity", len(verts))
        out.append(SignedNTuple(
            v=t.v, sigma=t.sigma, edge_signs=t.edge_signs,
            balanced=t.balanced, arity=arity,
            arc_weights=tuple(ws),
        ))
    return out


def per_vertex_arc_weights_array(tuples: list[SignedNTuple],
                                   *,
                                   is_walk: bool = False) -> np.ndarray:
    """Stack per-vertex arc weights into a ``(T, k)`` ``np.ndarray``
    suitable for handing to ``SignedKANLayer(..., arc_weights=...)``.

    Per-vertex value at position $i$:

    - cycle: $\\tfrac{1}{2}(w_{i-1 \\bmod k} + w_i)$ (the two incident
      edges within the cycle).
    - walk endpoints ($i=0$ or $i=L$): the single incident edge.
    - walk interior: $\\tfrac{1}{2}(w_{i-1} + w_i)$.

    Returns a zero array of shape ``(len(tuples), k)`` if any tuple
    has ``arc_weights=None`` (caller probably forgot to annotate).
    """
    if not tuples:
        return np.zeros((0, 0), dtype=np.float32)
    k = len(tuples[0].v)
    n = len(tuples)
    out = np.zeros((n, k), dtype=np.float32)
    for ti, t in enumerate(tuples):
        if t.arc_weights is None:
            continue
        ws = t.arc_weights
        if is_walk:
            L = len(ws)
            # L = k - 1 edges, k = L + 1 vertices
            for i in range(L + 1):
                if i == 0:
                    out[ti, i] = ws[0]
                elif i == L:
                    out[ti, i] = ws[L - 1]
                else:
                    out[ti, i] = 0.5 * (ws[i - 1] + ws[i])
        else:
            for i in range(k):
                out[ti, i] = 0.5 * (ws[(i - 1) % k] + ws[i])
    return out
