"""Zoo → HyMeKo → native tensor round-trip — the incidence survives, the sparse tensor + cycle engine engage.

Emitting a zoo hypergraph as HyMeKo, parsing it with the native engine, and compiling the star-expansion sparse
incidence tensor preserves the hypergraph (nnz = total incidence, edge count unchanged); the signed-cycle engine
runs over the zoo's graph families. Skipped where the built ``hymeko`` extension is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hymeko")                                # the built pyo3 extension

from scenarios.hypergraph_hymeko import (  # noqa: E402
    parse_ir,
    round_trip,
    signed_graph_cycles,
    to_hymeko,
)
from scenarios.hypergraph_zoo import (  # noqa: E402
    fano_plane,
    loose_cycle,
    projective_plane,
    steiner_triple_system,
)


def test_round_trip_preserves_incidence() -> None:
    """The star-expansion sparse tensor's nnz equals the total incidence, and the edge count is preserved."""
    for hg in (fano_plane(), projective_plane(3), loose_cycle(3, 5), steiner_triple_system(13)):
        report = round_trip(hg)
        assert report["incidence_preserved"] and report["edges_preserved"]
        assert report["star_nnz"] == sum(len(e) for e in hg.edges)
        assert report["shape"][0] == hg.n_edges                          # tensor first dim = number of hyperedges


def test_emitted_source_parses_and_hash_is_stable() -> None:
    fano = fano_plane()
    parse_ir(to_hymeko(fano))                                            # native parse succeeds (valid HyMeKo)
    assert round_trip(fano)["canonical_hash"] == round_trip(fano)["canonical_hash"]


def test_signed_incidence_is_emitted() -> None:
    """A signed hypergraph emits ``− v`` for negative incidence (the Nagare signed-incidence form)."""
    hg = loose_cycle(3, 2)
    src = to_hymeko(hg, signs=[[1, -1, 1] for _ in hg.edges])
    assert "- v" in src and "+ v" in src


def test_native_signed_cycle_engine_runs_on_zoo_graphs() -> None:
    """The Rust signed-cycle enumerator engages on the zoo's graph families (a triangle and a hexagon)."""
    triangle = [(0, 1, 1), (1, 2, 1), (2, 0, 1)]
    hexagon = [(i, (i + 1) % 6, 1) for i in range(6)]
    assert len(signed_graph_cycles(triangle, 3, 3)) >= 1
    assert len(signed_graph_cycles(hexagon, 6, 6)) >= 1
