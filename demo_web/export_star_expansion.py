"""Export the star-expansion fingerprint the 3D viewer consumes.

Mirrors ``export_kinematic_data.py``: one JSON blob (+ a ``.js`` companion for
``file://`` double-click) describing a hypergraph's vertices, hyperedges
(member lists + signs), the star incidence COO, and the engine's canonical
hash. The browser page derives star/clique geometry in JS; the **counts and
the canonical hash come from the engine** so they match the repo verbatim.

Two runtime paths (Strategy), picked automatically:

* **engine** — if ``hymeko`` imports (the built PyO3 module), parse the
  ``.hymeko`` source → IR, read the snapshot (members + signs), the canonical
  hash, and cross-check counts against ``compile_star_expansion`` /
  ``compile_clique_expansion``.
* **literal fallback** — otherwise a dependency-free reader for the
  ``@edge{ (~a, ~b, ~c); }`` hyperedge-literal grammar (fano/tetra-style
  fixtures). Same count semantics; the canonical hash is marked unavailable.

Counts (the load-bearing, exact claim, O(|E|·d) star vs O(|E|·d²) clique):
  star.incidence_nnz = Σ|e|        clique.edge_count = Σ C(|e|,2)

The 3D layout is force-directed in-browser for legibility, NOT geometric
ground truth — say so when presenting.

Run:
    python demo_web/export_star_expansion.py \
        --src data/typical_graphs/fano_graph.hymeko \
        --out demo_web/star_expansion_data.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
HASH_UNAVAILABLE = "blake3:<unavailable-without-engine>"


@dataclass
class Hyperedge:
    label: str
    members: list[int]
    sign: int
    arity: int


@dataclass
class StarExpansionData:
    """The viewer's data contract (schema 1). ``star.coo`` holds the incidence
    edges (hub-per-hyperedge ↔ member vertex) as parallel ``k`` (edge) / ``j``
    (vertex) lists — exactly what the JS hub layout draws."""

    source: str
    canonical_hash: str
    n_vertices: int
    vertex_labels: list[str]
    hyperedges: list[Hyperedge]
    star: dict[str, Any]
    clique: dict[str, int]
    schema: int = SCHEMA_VERSION
    backend: str = "engine"

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # dataclass order → put schema first for readability.
        return {"schema": d.pop("schema"), **d}


def _edge_sign(member_signs: list[int]) -> int:
    """Per-hyperedge sign = product of non-zero member signs (unsigned → +1).
    Mirrors the cycle sign-product used by the balance demo."""
    prod = 1
    for s in member_signs:
        if s != 0:
            prod *= 1 if s > 0 else -1
    return prod


def _assemble(
    source: str,
    canonical_hash: str,
    vertex_labels: list[str],
    raw_edges: list[tuple[str, list[int], list[int]]],
    backend: str,
) -> StarExpansionData:
    """Build the data contract from per-edge (label, member-vertex-indices,
    member-signs). Computes the exact star/clique counts and the incidence COO.

    Preconditions: member indices are valid into ``vertex_labels``.
    Postconditions: ``star.incidence_nnz == Σ|e|`` and
    ``clique.edge_count == Σ C(|e|,2)``.
    """
    hyperedges: list[Hyperedge] = []
    coo_k: list[int] = []
    coo_j: list[int] = []
    clique_edges = 0
    for ei, (label, members, signs) in enumerate(raw_edges):
        hyperedges.append(Hyperedge(
            label=label, members=list(members),
            sign=_edge_sign(signs), arity=len(members),
        ))
        for v in members:
            coo_k.append(ei)
            coo_j.append(v)
        clique_edges += math.comb(len(members), 2)

    incidence_nnz = len(coo_k)  # Σ|e|
    return StarExpansionData(
        source=source,
        canonical_hash=canonical_hash,
        n_vertices=len(vertex_labels),
        vertex_labels=vertex_labels,
        hyperedges=hyperedges,
        star={"incidence_nnz": incidence_nnz, "coo": {"k": coo_k, "j": coo_j}},
        clique={"edge_count": clique_edges},
        backend=backend,
    )


# ── engine path ──────────────────────────────────────────────────────────
def export_via_engine(src: Path) -> StarExpansionData:
    """Engine-backed export. Raises ImportError if ``hymeko`` is unavailable."""
    import hymeko  # noqa: F401  (presence probed by the caller)

    eng = hymeko.PyHypergraphEngine()
    ir = eng.load_file(str(src))
    snap = json.loads(ir.snapshot_json())

    # Collect the vertices actually referenced by hyperedges (drops the
    # container/root decl that node_count includes). Stable order = target id.
    ref: dict[int, str] = {}
    for e in snap["edges"]:
        for arc in e["arcs"]:
            ref[arc["target_id"]] = arc["target_name"]
    sorted_ids = sorted(ref)
    id2idx = {tid: i for i, tid in enumerate(sorted_ids)}
    vertex_labels = [ref[tid] for tid in sorted_ids]

    raw_edges: list[tuple[str, list[int], list[int]]] = []
    for e in snap["edges"]:
        members = [id2idx[a["target_id"]] for a in e["arcs"]]
        signs = [int(a["sign"]) for a in e["arcs"]]
        raw_edges.append((e["name"], members, signs))

    data = _assemble(
        source=str(src.relative_to(REPO_ROOT)) if src.is_absolute()
        else str(src),
        canonical_hash=ir.canonical_hash,
        vertex_labels=vertex_labels,
        raw_edges=raw_edges,
        backend="engine",
    )
    _cross_check_engine(eng, ir, data)
    return data


def _cross_check_engine(eng: Any, ir: Any, data: StarExpansionData) -> None:
    """Assert the JS-derived counts agree with the engine's COO (no drift).

    Star raw COO nnz is 2·Σ|e| when edges are unsigned (sign 0 pushes both
    hub→node and node→hub); clique raw nnz is 2·(undirected edges). We check
    against those factors so the displayed numbers are engine-sourced.
    """
    star_nnz = eng.compile_star_expansion(ir).nnz
    clique_nnz = eng.compile_clique_expansion(ir).nnz
    inc = data.star["incidence_nnz"]
    assert star_nnz in (inc, 2 * inc), (
        f"star incidence mismatch: Σ|e|={inc} but engine COO nnz={star_nnz}"
    )
    assert clique_nnz in (data.clique["edge_count"], 2 * data.clique["edge_count"]), (
        f"clique mismatch: ΣC(|e|,2)={data.clique['edge_count']} vs "
        f"engine nnz={clique_nnz}"
    )


# ── literal fallback ───────────────────────────────────────────────────────
_EDGE_RE = re.compile(r"@\s*\w+\s*\{\s*\(([^)]*)\)\s*;?\s*\}", re.S)
_MEMBER_RE = re.compile(r"([+\-~]?)\s*([A-Za-z_]\w*)")
_SIGN = {"+": 1, "-": -1, "~": 0, "": 0}


def export_via_literal(src: Path) -> StarExpansionData:
    """Dependency-free reader for the ``@e{ (~a, ~b, ~c); }`` hyperedge grammar.

    Scope: hyperedge literals with bare vertex names (fano/tetra-style
    fixtures). Robot/meta sources need the engine. Canonical hash is marked
    unavailable (only the engine computes the Blake3 WL fingerprint).
    """
    text = src.read_text(encoding="utf-8")
    blocks = _EDGE_RE.findall(text)
    if not blocks:
        raise ValueError(
            f"no '@edge{{ (...); }}' hyperedge literals found in {src}; "
            f"this source needs the engine path (import hymeko)."
        )
    label_to_idx: dict[str, int] = {}
    vertex_labels: list[str] = []
    raw_edges: list[tuple[str, list[int], list[int]]] = []
    for ei, body in enumerate(blocks):
        members: list[int] = []
        signs: list[int] = []
        for sign_tok, name in _MEMBER_RE.findall(body):
            if name not in label_to_idx:
                label_to_idx[name] = len(vertex_labels)
                vertex_labels.append(name)
            members.append(label_to_idx[name])
            signs.append(_SIGN.get(sign_tok, 0))
        raw_edges.append((f"e{ei}", members, signs))

    rel = str(src.relative_to(REPO_ROOT)) if src.is_absolute() else str(src)
    return _assemble(
        source=rel, canonical_hash=HASH_UNAVAILABLE,
        vertex_labels=vertex_labels, raw_edges=raw_edges, backend="literal",
    )


def export(src: Path) -> StarExpansionData:
    """Engine path if ``hymeko`` is importable, else the literal fallback."""
    try:
        import hymeko  # noqa: F401
    except ImportError:
        return export_via_literal(src)
    return export_via_engine(src)


def write_outputs(data: StarExpansionData, out: Path) -> list[Path]:
    """Write ``<out>.json`` and the ``<out>.js`` (``window.SXV_DATA``) companion
    for ``file://`` double-click, mirroring ``kinematic_data.{json,js}``."""
    blob = json.dumps(data.to_json_dict(), indent=2)
    out.write_text(blob + "\n", encoding="utf-8")
    js_path = out.with_suffix(".js")
    js_path.write_text(f"window.SXV_DATA = {blob};\n", encoding="utf-8")
    return [out, js_path]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", default="data/typical_graphs/fano_graph.hymeko")
    p.add_argument("--out", default="demo_web/star_expansion_data.json")
    args = p.parse_args(argv)

    src = Path(args.src)
    if not src.is_absolute():
        src = REPO_ROOT / src
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out

    data = export(src)
    written = write_outputs(data, out)
    print(f"[star-export] backend={data.backend} source={data.source}")
    print(f"[star-export] {data.n_vertices} vertices, {len(data.hyperedges)} "
          f"hyperedges; star.incidence_nnz={data.star['incidence_nnz']} "
          f"clique.edge_count={data.clique['edge_count']}")
    print(f"[star-export] canonical_hash={data.canonical_hash}")
    for w in written:
        print(f"[star-export] wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
