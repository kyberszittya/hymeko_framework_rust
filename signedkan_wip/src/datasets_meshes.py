"""Synthetic polyhedral-mesh signed-graph benchmarks.

Designed as a clean test of the geometric hypothesis behind
HSiKAN: cycles encode 2-cells (faces) of polyhedral surfaces in a
way walks cannot.  Walk-HSiKAN should LOSE to cycle-HSiKAN here,
because the signal lives in face balance, not in path-following.

Construction:
  * Take a polyhedron (cube, icosahedron, dodecahedron, ...)
  * Embed it in $\\mathbb{R}^3$
  * Sign each edge by its dominant axis-direction:
        $+1$ if the edge's largest |delta| is in $\\{+x, +y, +z\\}$
        $-1$ if the edge's largest |delta| is in $\\{-x, -y, -z\\}$
    This is a deliberately face-coupled signing — every face has a
    fixed sign-product determined by its orientation in 3D.
  * Stack N copies (different rotations / seeds) into a single
    disconnected signed graph for batch evaluation.
"""
from __future__ import annotations

import math
import numpy as np

from .datasets import SignedGraph


# ─── reference polyhedra (vertex coordinates + edge list) ─────────────


def _cube():
    # 8 vertices at $\\pm 1$, 12 edges along axis directions.
    verts = np.array([(x, y, z) for x in (-1, 1)
                                  for y in (-1, 1)
                                  for z in (-1, 1)], dtype=np.float64)
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            d = verts[i] - verts[j]
            if int((d != 0).sum()) == 1:   # axis-aligned only
                edges.append((i, j))
    return verts, edges


def _icosahedron():
    # 12 vertices on the unit sphere, 30 edges (each connects nearest
    # neighbours at distance $2 / \\sqrt{1 + \\varphi^2}$ for golden ratio).
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    raw = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            raw.append((0, s1, s2 * phi))
            raw.append((s1, s2 * phi, 0))
            raw.append((s2 * phi, 0, s1))
    verts = np.array(raw, dtype=np.float64)
    # Edge if distance ≈ 2 (the icosahedron's edge length when verts
    # use ±1, ±phi coords).
    target = 2.0
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = float(np.linalg.norm(verts[i] - verts[j]))
            if abs(d - target) < 1e-3:
                edges.append((i, j))
    return verts, edges


def _octahedron():
    verts = np.array([
        ( 1,  0,  0), (-1,  0,  0),
        ( 0,  1,  0), ( 0, -1,  0),
        ( 0,  0,  1), ( 0,  0, -1),
    ], dtype=np.float64)
    edges = []
    for i in range(6):
        for j in range(i + 1, 6):
            d = float(np.linalg.norm(verts[i] - verts[j]))
            if abs(d - math.sqrt(2.0)) < 1e-3:
                edges.append((i, j))
    return verts, edges


def _tetrahedron():
    # 4 vertices, 6 edges (every pair). Every face is a triangle.
    verts = np.array([
        ( 1,  1,  1), (-1, -1,  1),
        (-1,  1, -1), ( 1, -1, -1),
    ], dtype=np.float64)
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    return verts, edges


_POLYHEDRA = {
    "tetrahedron":  _tetrahedron,
    "cube":         _cube,
    "octahedron":   _octahedron,
    "icosahedron":  _icosahedron,
}


# ─── signing rule ─────────────────────────────────────────────────────


def _sign_edge(v_i: np.ndarray, v_j: np.ndarray) -> int:
    """Sign the edge by the dominant component of (v_j - v_i).
    Positive iff the largest-magnitude axis-component is positive."""
    d = v_j - v_i
    abs_d = np.abs(d)
    axis = int(np.argmax(abs_d))
    return +1 if d[axis] > 0 else -1


# ─── single-mesh + multi-mesh builders ────────────────────────────────


def build_polyhedron(name: str, seed: int = 0,
                       rotate: bool = True) -> SignedGraph:
    """Build a single signed graph for a named polyhedron.

    With ``rotate=True``, applies a random rotation so the
    axis-aligned signing rule produces a non-trivial sign pattern
    (otherwise edges of e.g. a cube collapse onto exactly 6 sign
    classes).
    """
    if name not in _POLYHEDRA:
        raise ValueError(f"unknown polyhedron: {name}")
    verts, edges = _POLYHEDRA[name]()
    if rotate:
        rng = np.random.default_rng(seed)
        # Uniform random rotation via QR decomposition of a Gaussian
        # matrix (Diaconis-Shahshahani).
        A = rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(A)
        Q = Q * np.sign(np.diag(R))
        verts = verts @ Q.T

    n_edges = len(edges)
    edges_np = np.array(edges, dtype=np.int64)
    signs_np = np.array(
        [_sign_edge(verts[i], verts[j]) for (i, j) in edges],
        dtype=np.int64,
    )
    return SignedGraph(
        edges=edges_np, signs=signs_np, n_nodes=verts.shape[0],
    )


def build_polyhedral_mesh(name: str, n_copies: int = 50,
                           seed: int = 0) -> SignedGraph:
    """Stack ``n_copies`` independently rotated copies of the named
    polyhedron into a single disconnected signed graph.

    Each copy contributes its full vertex / edge set, vertex IDs
    re-numbered to the global tally.  This is the format HSiKAN
    consumes — one big signed graph, with the disconnectedness
    just meaning enumerated cycles never span copies.
    """
    rng = np.random.default_rng(seed)
    all_edges: list[tuple[int, int]] = []
    all_signs: list[int] = []
    n_total = 0
    for k in range(n_copies):
        sub = build_polyhedron(name, seed=int(rng.integers(0, 1 << 30)))
        for (u, v), s in zip(sub.edges.tolist(), sub.signs.tolist()):
            all_edges.append((u + n_total, v + n_total))
            all_signs.append(int(s))
        n_total += sub.n_nodes
    return SignedGraph(
        edges=np.array(all_edges, dtype=np.int64),
        signs=np.array(all_signs, dtype=np.int64),
        n_nodes=n_total,
    )


# ─── single-instance multi-polyhedron benchmark ───────────────────────


def build_mixed_polytope_dataset(seed: int = 0,
                                   n_per_kind: int = 25
                                   ) -> SignedGraph:
    """A single signed graph built from an even mix of all four
    polyhedra in `_POLYHEDRA`.  Lets one benchmark stress-test the
    arity-mixing readout — the αₖ should reveal which face-arity
    dominates the resulting signal.

    Structurally:
      tetrahedron:  4 vertices, 6 edges, 4 triangular faces
      cube:         8 vertices, 12 edges, 6 quadrilateral faces
      octahedron:   6 vertices, 12 edges, 8 triangular faces
      icosahedron: 12 vertices, 30 edges, 20 triangular faces
    Total per copy: 30 vertices, 60 edges, lots of cycles at $k{=}3$
    and $k{=}4$.
    """
    rng = np.random.default_rng(seed)
    all_edges: list[tuple[int, int]] = []
    all_signs: list[int] = []
    n_total = 0
    for kind in ("tetrahedron", "cube", "octahedron", "icosahedron"):
        for _ in range(n_per_kind):
            sub = build_polyhedron(
                kind, seed=int(rng.integers(0, 1 << 30)))
            for (u, v), s in zip(sub.edges.tolist(), sub.signs.tolist()):
                all_edges.append((u + n_total, v + n_total))
                all_signs.append(int(s))
            n_total += sub.n_nodes
    return SignedGraph(
        edges=np.array(all_edges, dtype=np.int64),
        signs=np.array(all_signs, dtype=np.int64),
        n_nodes=n_total,
    )


# ─── CLI smoke ────────────────────────────────────────────────────────


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cube",
                    choices=list(_POLYHEDRA) + ["mesh", "mixed"])
    ap.add_argument("--n-copies", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.name == "mesh":
        g = build_polyhedral_mesh("cube", args.n_copies, args.seed)
    elif args.name == "mixed":
        g = build_mixed_polytope_dataset(args.seed,
                                            n_per_kind=args.n_copies // 4)
    else:
        g = build_polyhedron(args.name, args.seed)

    s = g.stats()
    print(f"  {args.name}: {s}")


if __name__ == "__main__":
    main()
