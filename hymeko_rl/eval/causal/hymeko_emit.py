"""Declare a discovered causal DAG as a ``.hymeko`` signed hypergraph and cross-view verify it.

This closes the CIP-scenario loop (Kato-LiNGAM joint #1, ``project-kato-lingam-cip-hymeko``): a DirectLiNGAM
result is a *signed weighted DAG*; declaring it in the canonical HyMeKo IR lets the framework's own engine
re-derive the star/clique tensor views and a Blake3 canonical fingerprint, so **the causal model the agent uses
is provably the one a human would audit** — the declared edge/sign structure must survive the IR round-trip.

Encoding (a signed 2-member hyperedge per causal edge):

* one vertex per LiNGAM variable;
* one hyperedge ``@c{k}{ (<cause>, <effect>); }`` per non-zero adjacency entry, **arc order = direction**
  (first member is the cause, second the effect);
* the cause arc is ``+``; the effect arc is ``+`` for a positive weight, ``-`` for a negative one, so the
  engine's per-edge sign (product of arc signs) equals ``sign(weight)`` — the sign survives the round-trip.

Cross-view verify (``cross_view_verify``) writes the source, loads it through the engine (``import hymeko``), and
asserts the *declared* signed-edge set equals the *engine-reparsed* one and that the star incidence preserves
every member-incidence (``star_nnz ∈ {Σ|e|, 2Σ|e|}``, the invariant ``demo_web/export_star_expansion.py`` checks).
``clique_nnz`` is reported but **not** gated: it is sign-sensitive (a signed ``(+,-)`` edge contracts differently
than an unsigned one), so it is an engine-internal count, not part of the round-trip guarantee. If the native
engine is unavailable, it falls back to a dependency-free reparse of the emitted grammar (star/edge counts only;
canonical hash reported unavailable).

Doctrine unchanged: the DAG is a **proposal**; controlled ablations decide. This module verifies *representation*
consistency (declared ≡ tensor ≡ hash), not causal truth.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .lingam import LingamResult

# A signed edge as it survives the round-trip: (cause, effect, sign) with sign in {-1, +1}.
SignedEdge = tuple[str, str, int]

_IDENT = re.compile(r"^[A-Za-z_]\w*$")
_HASH_UNAVAILABLE = "blake3:<unavailable-without-engine>"


@dataclass(frozen=True)
class CausalHypergraph:
    """A causal DAG staged for declaration as a signed ``.hymeko`` hypergraph.

    # Invariants
    * ``variables`` are unique valid identifiers; every edge references two *distinct* known variables;
      no self-loops (DirectLiNGAM is acyclic by construction, so a self-loop signals a caller bug).
    """

    name: str
    variables: list[str]
    edges: list[tuple[str, str, float]]  # (cause, effect, weight); weight sign is what is declared

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("CausalHypergraph needs at least one variable")
        if not _IDENT.match(self.name):
            raise ValueError(f"graph name {self.name!r} is not a valid identifier")
        seen: set[str] = set()
        for v in self.variables:
            if not _IDENT.match(v):
                raise ValueError(f"variable {v!r} is not a valid .hymeko identifier")
            if v in seen:
                raise ValueError(f"duplicate variable {v!r}")
            seen.add(v)
        for cause, effect, _w in self.edges:
            if cause not in seen or effect not in seen:
                raise ValueError(f"edge ({cause!r}->{effect!r}) references an unknown variable")
            if cause == effect:
                raise ValueError(f"self-loop on {cause!r}; a LiNGAM DAG is acyclic")

    @classmethod
    def from_lingam(cls, result: "LingamResult", name: str, *, min_abs: float = 0.0) -> "CausalHypergraph":
        """Build from a :class:`LingamResult` — one edge per non-zero adjacency entry (``B[effect, cause]``)."""
        b = result.adjacency
        edges = [(result.names[j], result.names[i], float(b[i, j]))
                 for i in range(len(result.names)) for j in range(len(result.names))
                 if abs(b[i, j]) > min_abs]
        return cls(name=name, variables=list(result.names), edges=edges)

    def declared_signed_edges(self) -> set[SignedEdge]:
        """The signed-edge support ``{(cause, effect, sign(weight))}`` this graph declares."""
        return {(c, e, int(np.sign(w))) for c, e, w in self.edges if w != 0.0}


def to_hymeko_source(cg: CausalHypergraph) -> str:
    """Emit the ``.hymeko`` source for ``cg`` (fano-style grammar: typed instance, vertex decls, signed edges).

    # Postconditions the string parses under both the native engine and the literal fallback; the per-edge sign
      equals ``sign(weight)`` (cause arc ``+``, effect arc ``+``/``-``).
    """
    lines = [f"{cg.name}{{}}", cg.name.lower(), "{"]
    lines += [f"    {v} {{}}" for v in cg.variables]
    lines.append("")
    for k, (cause, effect, w) in enumerate(cg.edges):
        effect_sign = "+" if w >= 0.0 else "-"      # cause arc always +, effect arc carries the weight's sign
        lines.append(f"    @c{k}{{ (+{cause}, {effect_sign}{effect}); }}")
    lines.append("}")
    return "\n".join(lines) + "\n"


@dataclass
class CrossViewReport:
    """Result of re-deriving the declared DAG through the HyMeKo IR and comparing the views."""

    agree: bool
    backend: str                      # "engine" | "literal"
    canonical_hash: str
    n_edges_declared: int
    n_edges_engine: int
    star_nnz: int
    clique_nnz: int
    sum_arities: int                  # Σ|e|
    edges_match: bool
    counts_match: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agree": self.agree, "backend": self.backend, "canonical_hash": self.canonical_hash,
            "n_edges_declared": self.n_edges_declared, "n_edges_engine": self.n_edges_engine,
            "star_nnz": self.star_nnz, "clique_nnz": self.clique_nnz, "sum_arities": self.sum_arities,
            "edges_match": self.edges_match, "counts_match": self.counts_match, "notes": list(self.notes),
        }


_ARC_SIGN = {"+": 1, "-": -1, "~": 0, "": 0}
_EDGE_RE = re.compile(r"@\s*\w+\s*\{\s*\(([^)]*)\)\s*;?\s*\}", re.S)
_MEMBER_RE = re.compile(r"([+\-~]?)\s*([A-Za-z_]\w*)")


def _engine_signed_edges(snapshot: dict[str, Any]) -> tuple[set[SignedEdge], int]:
    """Reconstruct ``{(cause, effect, sign)}`` from an engine snapshot (arc[0]=cause, arc[1]=effect)."""
    edges: set[SignedEdge] = set()
    for e in snapshot["edges"]:
        arcs = e["arcs"]
        if len(arcs) != 2:
            raise ValueError(f"edge {e.get('name')!r} has arity {len(arcs)}; a causal edge is binary")
        sign = int(np.sign(int(arcs[0]["sign"]) * int(arcs[1]["sign"])))
        edges.add((arcs[0]["target_name"], arcs[1]["target_name"], sign))
    return edges, len(snapshot["edges"])


def _literal_signed_edges(source: str) -> tuple[set[SignedEdge], int, int]:
    """Dependency-free reparse of the emitted grammar → (signed edges, edge count, Σ|e|). Fallback only."""
    edges: set[SignedEdge] = set()
    sum_arities = 0
    blocks = _EDGE_RE.findall(source)
    for body in blocks:
        members = _MEMBER_RE.findall(body)
        if len(members) != 2:
            raise ValueError(f"literal edge {body!r} is not binary")
        (s0, cause), (s1, effect) = members
        sign = int(np.sign(_ARC_SIGN.get(s0, 0) * _ARC_SIGN.get(s1, 0)))
        edges.add((cause, effect, sign))
        sum_arities += 2
    return edges, len(blocks), sum_arities


def cross_view_verify(cg: CausalHypergraph, out_path: str | Path) -> CrossViewReport:
    """Write ``cg`` as ``.hymeko`` and verify the declared DAG re-derives identically through the HyMeKo IR.

    # Preconditions ``out_path``'s parent exists and is writable.
    # Postconditions the ``.hymeko`` file exists at ``out_path``; the returned report's ``agree`` is ``True`` iff
      the engine-reparsed signed-edge set equals the declared one *and* the star/clique count invariants hold.
    """
    out = Path(out_path)
    source = to_hymeko_source(cg)
    out.write_text(source, encoding="utf-8")

    declared = cg.declared_signed_edges()
    sum_arities = 2 * len(cg.edges)              # every causal edge has exactly two members
    clique_edges = len(cg.edges)                 # ΣC(2,2) = one clique edge per binary hyperedge
    notes: list[str] = []

    try:
        import hymeko  # type: ignore[import-untyped]  # native PyO3 engine (no stubs); built via maturin
    except ImportError:
        engine_edges, n_engine, star_lit = _literal_signed_edges(source)
        edges_match = engine_edges == declared
        counts_match = star_lit == sum_arities
        notes.append("native engine unavailable — literal fallback (no canonical hash, count invariants only)")
        return CrossViewReport(
            agree=edges_match and counts_match, backend="literal", canonical_hash=_HASH_UNAVAILABLE,
            n_edges_declared=len(cg.edges), n_edges_engine=n_engine, star_nnz=star_lit, clique_nnz=clique_edges,
            sum_arities=sum_arities, edges_match=edges_match, counts_match=counts_match, notes=notes)

    eng = hymeko.PyHypergraphEngine()
    ir = eng.load_file(str(out))
    snapshot = json.loads(ir.snapshot_json())
    engine_edges, n_engine = _engine_signed_edges(snapshot)
    star_nnz = int(eng.compile_star_expansion(ir).nnz)
    clique_nnz = int(eng.compile_clique_expansion(ir).nnz)

    # The rigorous, sign-robust cross-view claim: every declared member-incidence is present in the star tensor
    # (star_nnz == Σ|e|, or 2Σ|e| for the unsigned convention) and the edge count is preserved. clique_nnz is
    # sign-sensitive (a signed (+,-) edge contracts differently than an unsigned one), so it is reported, not gated.
    edges_match = engine_edges == declared
    counts_match = star_nnz in (sum_arities, 2 * sum_arities) and n_engine == len(cg.edges)
    if not edges_match:
        notes.append(f"declared\\engine={sorted(declared - engine_edges)}; engine\\declared={sorted(engine_edges - declared)}")
    return CrossViewReport(
        agree=edges_match and counts_match, backend="engine", canonical_hash=ir.canonical_hash,
        n_edges_declared=len(cg.edges), n_edges_engine=n_engine, star_nnz=star_nnz, clique_nnz=clique_nnz,
        sum_arities=sum_arities, edges_match=edges_match, counts_match=counts_match, notes=notes)
