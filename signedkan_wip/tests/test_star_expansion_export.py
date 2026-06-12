"""Tests for the star-expansion viewer exporter + the canonical_hash getter.

The exporter (``demo_web/export_star_expansion.py``) is loaded by file path
(``demo_web`` is a static-asset dir, not an installed package). Engine-backed
tests skip when the built ``hymeko`` module is unavailable; run them with the
venv interpreter::

    .venv/Scripts/python.exe -m pytest -p no:randomly \
        signedkan_wip/tests/test_star_expansion_export.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FANO = REPO_ROOT / "data" / "typical_graphs" / "fano_graph.hymeko"


def _load_exporter():
    path = REPO_ROOT / "demo_web" / "export_star_expansion.py"
    spec = importlib.util.spec_from_file_location("export_star_expansion", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod  # py3.12 @dataclass needs the module registered
    spec.loader.exec_module(mod)
    return mod


ex = _load_exporter()

try:
    import hymeko  # noqa: F401

    _HAS_ENGINE = True
except ImportError:
    _HAS_ENGINE = False


# ── fallback (dependency-free) ─────────────────────────────────────────────
def test_literal_fallback_counts_fano() -> None:
    d = ex.export_via_literal(FANO)
    # Fano plane: 7 vertices, 7 arity-3 lines. Σ|e| = 21; Σ C(3,2) = 21.
    assert d.n_vertices == 7
    assert len(d.hyperedges) == 7
    assert all(e.arity == 3 for e in d.hyperedges)
    assert d.star["incidence_nnz"] == 21
    assert d.clique["edge_count"] == 21
    assert d.canonical_hash == ex.HASH_UNAVAILABLE
    assert d.backend == "literal"


def test_incidence_coo_matches_members() -> None:
    d = ex.export_via_literal(FANO)
    coo = d.star["coo"]
    assert len(coo["k"]) == len(coo["j"]) == d.star["incidence_nnz"]
    # Reconstruct members from the COO; must equal the hyperedge member lists.
    rebuilt: dict[int, list[int]] = {}
    for k, j in zip(coo["k"], coo["j"]):
        rebuilt.setdefault(k, []).append(j)
    for ei, e in enumerate(d.hyperedges):
        assert sorted(rebuilt[ei]) == sorted(e.members)


def test_write_outputs_json_and_js(tmp_path) -> None:
    d = ex.export_via_literal(FANO)
    out = tmp_path / "star_expansion_data.json"
    written = ex.write_outputs(d, out)
    assert {p.name for p in written} == {
        "star_expansion_data.json", "star_expansion_data.js",
    }
    import json

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema"] == ex.SCHEMA_VERSION
    assert loaded["star"]["incidence_nnz"] == 21
    js = out.with_suffix(".js").read_text(encoding="utf-8")
    assert js.startswith("window.SXV_DATA = ")


# ── engine path ────────────────────────────────────────────────────────────
@pytest.mark.skipif(not _HAS_ENGINE, reason="built hymeko module unavailable")
def test_engine_matches_fallback_counts() -> None:
    eng = ex.export_via_engine(FANO)
    fb = ex.export_via_literal(FANO)
    assert eng.n_vertices == fb.n_vertices
    assert eng.star["incidence_nnz"] == fb.star["incidence_nnz"]
    assert eng.clique["edge_count"] == fb.clique["edge_count"]
    assert eng.canonical_hash.startswith("blake3:")
    assert eng.canonical_hash != ex.HASH_UNAVAILABLE


@pytest.mark.skipif(not _HAS_ENGINE, reason="built hymeko module unavailable")
def test_canonical_hash_invariant_to_declaration_order() -> None:
    # Empirically (probed 2026-06-10) the canonical hash is invariant to the
    # *declaration order* of nodes and edges and is deterministic; it does
    # change on a structural edit. (It is NOT invariant to relabeling or to
    # within-edge member order — those carry signed-arc meaning; the full
    # isomorphism framing is Demo 2's concern.)
    base = """G{}
g{
  n0 {} n1 {} n2 {} n3 {}
  @e0{ (~n0, ~n1); }
  @e1{ (~n1, ~n2); }
  @e2{ (~n2, ~n3); }
}
"""
    edge_reorder = """G{}
g{
  n0 {} n1 {} n2 {} n3 {}
  @e2{ (~n2, ~n3); }
  @e0{ (~n0, ~n1); }
  @e1{ (~n1, ~n2); }
}
"""
    node_reorder = """G{}
g{
  n3 {} n2 {} n1 {} n0 {}
  @e0{ (~n0, ~n1); }
  @e1{ (~n1, ~n2); }
  @e2{ (~n2, ~n3); }
}
"""
    changed = base.replace("(~n2, ~n3)", "(~n0, ~n3)")

    def h(src: str) -> str:
        return hymeko.PyHypergraphEngine().parse_dsl(src).canonical_hash

    h_base = h(base)
    assert h_base.startswith("blake3:") and len(h_base) == len("blake3:") + 64
    assert h_base == h(base), "hash must be deterministic"
    assert h_base == h(edge_reorder), "hash must be invariant to edge-decl order"
    assert h_base == h(node_reorder), "hash must be invariant to node-decl order"
    assert h_base != h(changed), "a structural change must change the hash"
