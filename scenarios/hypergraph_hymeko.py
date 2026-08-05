r"""Bridge the hypergraph zoo to the native HyMeKo tensor path — emit as HyMeKo, round-trip through the engine.

Every :class:`~scenarios.hypergraph_zoo.Hypergraph` is emitted as a HyMeKo source (nodes + ``@``-hyperedges with
signed incidence), parsed back by the native ``PyHypergraphEngine`` into a ``PyHypergraphIR``, and compiled to the
**star-expansion sparse incidence tensor** (``PySparseMatrix2D``) — so the zoo feeds the doctoral canonical sparse
tensor and the Rust cycle engine (``enumerate_cycles_rs``, the signed-cycle / holonomy path). The round-trip is
verified: the star tensor's ``nnz`` equals the hypergraph's total incidence, and the IR's edge count is preserved.

# Preconditions: the built ``hymeko`` pyo3 extension is importable. # Postconditions: ``round_trip`` returns a
#   verification dict whose ``incidence_preserved`` / ``edges_preserved`` are true for a faithful emit.
"""

from __future__ import annotations

from scenarios.hypergraph_zoo import Hypergraph


def _prop(obj: object, name: str):
    """Read a pyo3 attribute that may be a property or a zero-arg method."""
    attr = getattr(obj, name)
    return attr() if callable(attr) else attr


def to_hymeko(hg: Hypergraph, name: str = "zoo_hg", signs: "list[list[int]] | None" = None) -> str:
    r"""Emit the hypergraph as HyMeKo source: a ``node`` per vertex and an ``@``-hyperedge per edge (signed incidence).

    ``signs[j][i]`` (optional) gives the incidence sign (+1/−1) of the ``i``-th vertex of edge ``j`` — the signed
    hypergraph the Nagare line works over; the default is all ``+`` (unsigned incidence).
    """
    lines = [f"{name}_description {{", "}", f"{name} {{", "    node {}", "    hyperedge {}"]
    lines += [f"    v{v}: + <isa> node {{}}" for v in range(hg.n_vertices)]
    for j, edge in enumerate(hg.edges):
        verts = sorted(edge)
        sgn = signs[j] if signs is not None else [1] * len(verts)
        inc = ", ".join(f"{'+' if s >= 0 else '-'} v{v}" for v, s in zip(verts, sgn))
        lines.append(f"    @e{j}: + <isa> hyperedge {{ ({inc}); }}")
    lines.append("}")
    return "\n".join(lines)


def parse_ir(src: str):
    """Parse HyMeKo source with the native engine → (engine, IR)."""
    import hymeko
    engine = hymeko.PyHypergraphEngine()
    return engine, engine.parse_dsl(src)


def star_incidence(hg: Hypergraph, name: str = "zoo_hg"):
    """Emit → parse → compile the star-expansion sparse incidence tensor. Returns (sparse_matrix, IR)."""
    engine, ir = parse_ir(to_hymeko(hg, name))
    return engine.compile_star_expansion(ir), ir


def round_trip(hg: Hypergraph) -> dict:
    """Emit + parse + compile, and verify the native incidence tensor faithfully captures the hypergraph."""
    sparse, ir = star_incidence(hg)
    total_incidence = sum(len(e) for e in hg.edges)
    nnz = _prop(sparse, "nnz")
    edge_count = _prop(ir, "edge_count")
    return {"ir_edge_count": edge_count, "star_nnz": nnz, "incidence_total": total_incidence,
            "shape": _prop(sparse, "shape"), "canonical_hash": _prop(ir, "canonical_hash"),
            "incidence_preserved": nnz == total_incidence, "edges_preserved": edge_count == hg.n_edges}


def signed_graph_cycles(edges: "list[tuple[int, int, int]]", n_nodes: int, k_len: int,
                        m_per_vertex: int = 32):
    r"""Run the native signed-cycle enumerator over a directed signed graph (``(u, v, sign)`` edges).

    Feeds the zoo's graph families (e.g. a matroid's underlying graph, or a cycle family) to
    ``hymeko.enumerate_cycles_rs`` — the Rust top-``m`` signed-cycle / holonomy engine. Returns its cycle list.
    """
    import hymeko
    u = [int(a) for a, _b, _s in edges]
    v = [int(b) for _a, b, _s in edges]
    sgn = [int(s) for _a, _b, s in edges]
    return hymeko.enumerate_cycles_rs(u, v, sgn, int(n_nodes), int(k_len), int(m_per_vertex))
