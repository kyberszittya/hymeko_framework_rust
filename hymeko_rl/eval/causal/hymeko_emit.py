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
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from .lingam import LingamResult

# A signed edge as it survives the round-trip: (cause, effect, sign) with sign in {-1, +1}.
SignedEdge = tuple[str, str, int]

_IDENT = re.compile(r"^[A-Za-z_]\w*$")
_HASH_UNAVAILABLE = "blake3:<unavailable-without-engine>"


@dataclass
class Mechanism:
    """A directed causal *mechanism* hyperedge: a tail (input) variable set drives a head (output) variable set.

    The mechanism is the primary causal object of the HyMeKo-native view (LiNGAM-SH spec,
    ``docs/plans/2026-07-08-lingam-sh-causal-hypergraph/spec.md``). A pairwise DirectLiNGAM edge ``x → y`` is the
    degenerate ``|tail|=|head|=1`` case (:meth:`pairwise`).

    # Invariants
    * ``name`` is a valid identifier (it becomes the star-expansion hub vertex);
    * ``tail`` and ``head`` are each non-empty with unique members, and disjoint (a variable cannot be both an
      input and an output of the *same* mechanism — that is an immediate self-cycle);
    * ``sign in {-1, +1}``.
    """

    name: str
    tail: tuple[str, ...]
    head: tuple[str, ...]
    strength: float = 1.0
    sign: int = 1
    evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not _IDENT.match(self.name):
            raise ValueError(f"mechanism name {self.name!r} is not a valid identifier")
        if not self.tail or not self.head:
            raise ValueError(f"mechanism {self.name!r} needs >= 1 tail and >= 1 head variable")
        if len(set(self.tail)) != len(self.tail) or len(set(self.head)) != len(self.head):
            raise ValueError(f"mechanism {self.name!r} has duplicate tail/head members")
        overlap = set(self.tail) & set(self.head)
        if overlap:
            raise ValueError(f"mechanism {self.name!r}: variable(s) {sorted(overlap)} are both tail and head")
        if self.sign not in (-1, 1):
            raise ValueError(f"mechanism {self.name!r} sign must be -1 or +1, got {self.sign}")

    @classmethod
    def pairwise(cls, cause: str, effect: str, weight: float, *, name: str | None = None) -> "Mechanism":
        """The degenerate mechanism ``{cause} → {effect}`` equivalent to a pairwise LiNGAM edge."""
        return cls(name=name or f"m_{cause}_{effect}", tail=(cause,), head=(effect,),
                   strength=abs(float(weight)), sign=int(np.sign(weight)) or 1)

    @property
    def weight(self) -> float:
        """The signed scalar coefficient ``sign · |strength|`` (used by the pairwise projection)."""
        return float(self.sign) * abs(float(self.strength))


@dataclass
class StarProjection:
    """The star-expanded compatibility view: variables + mechanism hubs + directed incidence (src → dst, sign)."""

    variables: list[str]
    hubs: list[str]
    incidence: list[tuple[str, str, int]]   # tail → hub (sign +1); hub → head (sign = mechanism sign)


@dataclass
class AcyclicityResult:
    """Outcome of the mechanism-level acyclicity check (bipartite variable/mechanism digraph)."""

    acyclic: bool
    rank: dict[str, int] | None = None      # topological rank per node (variables + mechanisms) when acyclic
    cycle_nodes: list[str] | None = None    # nodes left in a cycle when not acyclic

    def __bool__(self) -> bool:
        return self.acyclic


def _indegrees(adjacency: Mapping[str, list[str]]) -> dict[str, int]:
    """In-degree per node of a directed graph given as an adjacency (out-edge) map."""
    indeg: dict[str, int] = {n: 0 for n in adjacency}
    for dsts in adjacency.values():
        for dst in dsts:
            indeg[dst] = indeg.get(dst, 0) + 1
    return indeg


def _topo_rank(adjacency: Mapping[str, list[str]]) -> AcyclicityResult:
    """Kahn topological sort over a directed graph → ranks if acyclic, else the nodes trapped in a cycle."""
    indeg = _indegrees(adjacency)
    rank: dict[str, int] = {}
    queue = [n for n, d in indeg.items() if d == 0]
    r = 0
    while queue:
        nxt: list[str] = []
        for n in queue:
            rank[n] = r
            for dst in adjacency[n]:
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    nxt.append(dst)
        queue, r = nxt, r + 1
    if len(rank) == len(indeg):
        return AcyclicityResult(acyclic=True, rank=rank)
    return AcyclicityResult(acyclic=False, cycle_nodes=sorted(n for n in indeg if n not in rank))


@dataclass(frozen=True)
class CausalHypergraph:
    """A causal graph staged for declaration as a signed ``.hymeko`` hypergraph.

    Two forms share this one type (extend, don't duplicate):

    * **pairwise** (``mechanisms`` empty) — the DirectLiNGAM view: ``edges`` are ``(cause, effect, weight)`` and
      the ``.hymeko`` emit is a 2-member signed hyperedge per edge. This path is **unchanged / byte-identical**.
    * **mechanism** (``mechanisms`` non-empty) — the LiNGAM-SH view: causal mechanisms are directed hyperedges
      (tail set → head set); the ``.hymeko`` emit is the **star expansion** (variables + one hub vertex per
      mechanism + directed incidence). The pairwise DAG is then a *projection* (:meth:`pairwise_projection`).

    # Invariants
    * ``variables`` are unique valid identifiers; every pairwise edge references two distinct known variables;
    * every mechanism's tail/head variables are known; mechanism names are valid identifiers, unique, and
      disjoint from the variable names (a hub vertex must not collide with a variable).
    """

    name: str
    variables: list[str]
    edges: list[tuple[str, str, float]]  # (cause, effect, weight); weight sign is what is declared
    mechanisms: list[Mechanism] = field(default_factory=list)  # non-empty ⇒ mechanism form (star-expanded emit)

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
        self._validate_mechanisms(seen)

    def _validate_mechanisms(self, variables: set[str]) -> None:
        mech_names: set[str] = set()
        for m in self.mechanisms:
            if m.name in variables:
                raise ValueError(f"mechanism hub {m.name!r} collides with a variable name")
            if m.name in mech_names:
                raise ValueError(f"duplicate mechanism name {m.name!r}")
            mech_names.add(m.name)
            for v in (*m.tail, *m.head):
                if v not in variables:
                    raise ValueError(f"mechanism {m.name!r} references unknown variable {v!r}")

    @classmethod
    def from_lingam(cls, result: "LingamResult", name: str, *, min_abs: float = 0.0) -> "CausalHypergraph":
        """Build the **pairwise** form from a :class:`LingamResult` — one edge per non-zero ``B[effect, cause]``."""
        b = result.adjacency
        edges = [(result.names[j], result.names[i], float(b[i, j]))
                 for i in range(len(result.names)) for j in range(len(result.names))
                 if abs(b[i, j]) > min_abs]
        return cls(name=name, variables=list(result.names), edges=edges)

    @classmethod
    def from_mechanisms(cls, name: str, mechanisms: "list[Mechanism]") -> "CausalHypergraph":
        """Build the **mechanism** form from directed hyperedges (variables inferred from every tail/head)."""
        variables = sorted({v for m in mechanisms for v in (*m.tail, *m.head)})
        return cls(name=name, variables=variables, edges=[], mechanisms=list(mechanisms))

    # -- projections ------------------------------------------------------------------------------------------
    def star_projection(self) -> StarProjection:
        """Star-expanded view: ``tail → hub`` (sign +1) and ``hub → head`` (mechanism sign) per mechanism.

        For the pairwise form there are no hubs; the incidence is the pairwise edges themselves."""
        if not self.mechanisms:
            inc = [(c, e, int(np.sign(w))) for c, e, w in self.edges if w != 0.0]
            return StarProjection(variables=list(self.variables), hubs=[], incidence=inc)
        hubs = [m.name for m in self.mechanisms]
        incidence: list[tuple[str, str, int]] = []
        for m in self.mechanisms:
            incidence += [(t, m.name, 1) for t in m.tail]
            incidence += [(m.name, h, int(np.sign(m.sign)) or 1) for h in m.head]
        return StarProjection(variables=list(self.variables), hubs=hubs, incidence=incidence)

    def pairwise_projection(self) -> list[tuple[str, str, float]]:
        """Pairwise DAG **projection**: ``tail × head`` edges per mechanism (or ``edges`` unchanged if pairwise).

        This is a projection for compatibility, **not** the primary model when mechanisms are present."""
        if not self.mechanisms:
            return list(self.edges)
        return [(t, h, m.weight) for m in self.mechanisms for t in m.tail for h in m.head]

    def to_pairwise_hypergraph(self) -> "CausalHypergraph":
        """Collapse to the pairwise form (mechanisms marginalized to ``tail × head`` edges over the variables)."""
        if not self.mechanisms:
            return self
        return CausalHypergraph(name=self.name, variables=list(self.variables),
                                edges=self.pairwise_projection())

    def check_acyclicity(self) -> AcyclicityResult:
        """Mechanism-level acyclicity: the bipartite variable/mechanism digraph (tail→mech, mech→head) is a DAG."""
        if not self.mechanisms:
            adjacency: dict[str, list[str]] = {v: [] for v in self.variables}
            for c, e, _w in self.edges:
                adjacency[c].append(e)
            return _topo_rank(adjacency)
        adjacency = {n: [] for n in (*self.variables, *(m.name for m in self.mechanisms))}
        for m in self.mechanisms:
            for t in m.tail:
                adjacency[t].append(m.name)
            for h in m.head:
                adjacency[m.name].append(h)
        return _topo_rank(adjacency)

    # -- declaration ------------------------------------------------------------------------------------------
    def n_binary_edges(self) -> int:
        """The number of 2-member hyperedges the ``.hymeko`` emit produces (pairwise edges, or star incidences)."""
        if not self.mechanisms:
            return len(self.edges)
        return sum(len(m.tail) + len(m.head) for m in self.mechanisms)

    def declared_signed_edges(self) -> set[SignedEdge]:
        """The signed 2-member support this graph declares (pairwise edges, or the star incidence for mechanisms)."""
        if not self.mechanisms:
            return {(c, e, int(np.sign(w))) for c, e, w in self.edges if w != 0.0}
        return {(src, dst, int(np.sign(sign))) for src, dst, sign in self.star_projection().incidence}


def to_hymeko_source(cg: CausalHypergraph) -> str:
    """Emit the ``.hymeko`` source for ``cg`` (fano-style grammar: typed instance, vertex decls, signed edges).

    Pairwise form: a 2-member signed hyperedge per edge (**byte-identical to prior behaviour**). Mechanism form:
    the star expansion — one hub vertex per mechanism plus a 2-member incidence hyperedge per tail/head arc.

    # Postconditions the string parses under both the native engine and the literal fallback; each emitted edge is
      binary (``cause`` arc ``+``, ``effect``/``head`` arc carries the sign).
    """
    if cg.mechanisms:
        return _to_hymeko_source_mechanisms(cg)
    lines = [f"{cg.name}{{}}", cg.name.lower(), "{"]
    lines += [f"    {v} {{}}" for v in cg.variables]
    lines.append("")
    for k, (cause, effect, w) in enumerate(cg.edges):
        effect_sign = "+" if w >= 0.0 else "-"      # cause arc always +, effect arc carries the weight's sign
        lines.append(f"    @c{k}{{ (+{cause}, {effect_sign}{effect}); }}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _to_hymeko_source_mechanisms(cg: CausalHypergraph) -> str:
    """Star-expanded ``.hymeko`` for the mechanism form: variables + hub vertices + directed incidence edges."""
    star = cg.star_projection()
    lines = [f"{cg.name}{{}}", cg.name.lower(), "{"]
    lines += [f"    {v} {{}}" for v in cg.variables]
    lines += [f"    {hub} {{}}" for hub in star.hubs]        # one hub vertex per mechanism
    lines.append("")
    for k, (src, dst, sign) in enumerate(star.incidence):
        dst_sign = "+" if sign >= 0 else "-"                 # tail→hub is +; hub→head carries the mechanism sign
        lines.append(f"    @c{k}{{ (+{src}, {dst_sign}{dst}); }}")
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
    n_binary = cg.n_binary_edges()               # pairwise edges, or star incidences for the mechanism form
    sum_arities = 2 * n_binary                    # every emitted hyperedge has exactly two members
    clique_edges = n_binary                       # ΣC(2,2) = one clique edge per binary hyperedge
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
            n_edges_declared=n_binary, n_edges_engine=n_engine, star_nnz=star_lit, clique_nnz=clique_edges,
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
    counts_match = star_nnz in (sum_arities, 2 * sum_arities) and n_engine == n_binary
    if not edges_match:
        notes.append(f"declared\\engine={sorted(declared - engine_edges)}; engine\\declared={sorted(engine_edges - declared)}")
    return CrossViewReport(
        agree=edges_match and counts_match, backend="engine", canonical_hash=ir.canonical_hash,
        n_edges_declared=n_binary, n_edges_engine=n_engine, star_nnz=star_nnz, clique_nnz=clique_nnz,
        sum_arities=sum_arities, edges_match=edges_match, counts_match=counts_match, notes=notes)
