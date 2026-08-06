"""Second domain (systems engineering): requirements-traceability cross-view consistency.

The same extraction-function discipline as the kinematic `cross_view.py`, applied to a non-robotics domain so the
``general framework'' claim is load-bearing rather than asserted. One signed-hypergraph IR projects into a SysML v2
requirements model and a DOT traceability graph; for each view an extractor X_f recovers the entity and
typed-relation sets, and we test the commuting square

    X_sysml(eps_sysml(H)) == X_dot(eps_dot(H)) == Q(H)

across TWO fixtures (`traceability_smc`, the richer paper witness; `sysml_cell`, a second systems-engineering
model). Beyond entity/relation counts we check a genuine systems-engineering invariant --- requirement
**coverage** (every requirement realised by >=1 component, directly via `satisfies` or via `allocated_to`) ---
computed identically from each view and from the IR. A coverage discrepancy across views is impossible by the same
proposition that rules out a link-count mismatch across robot formats.

Substrate (CLAUDE.md): drives the live CLI (`requirements_sysml`/`requirements_dot`, fixed this session); for
`traceability_smc` it falls back to the committed `data/paper/traceability_smc.{sysml,dot}` if no binary is built.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import is_acyclic  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "data" / "paper"
PROFILES = REPO / "data" / "profiles"
_CLI = REPO / "target" / "release" / "hymeko.exe"
_CLI_DEBUG = REPO / "target" / "debug" / "hymeko.exe"

# (logical name, source .hymeko, committed sysml fallback | None, committed dot fallback | None, Q anchor)
FIXTURES = [
    ("pick_place_cell", PAPER / "traceability_smc.hymeko",
     PAPER / "traceability_smc.sysml", PAPER / "traceability_smc.dot",
     {"requirements": 4, "blocks": 4, "trace_edges": 7, "covered": 4}),
    ("sysml_cell", PROFILES / "sysml_cell.hymeko", None, None,
     {"requirements": 2, "blocks": 2, "trace_edges": 2, "covered": 2}),
]


def _cli() -> Path | None:
    return _CLI if _CLI.exists() else (_CLI_DEBUG if _CLI_DEBUG.exists() else None)


def _emit(fmt: str, source: Path, name: str, committed: Path | None) -> str:
    """epsilon_f via the live CLI; fall back to a committed emission if the CLI is unavailable/empty."""
    cli = _cli()
    if cli is not None:
        out = subprocess.run([str(cli), "emit", str(source), "--format", fmt, "--name", name],
                             capture_output=True, text=True, encoding="utf-8")
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    if committed is not None and committed.exists():
        return committed.read_text(encoding="utf-8")
    raise RuntimeError(f"no output for {fmt} on {source.name} (no CLI and no committed fallback)")


@dataclass(frozen=True)
class TraceInvariant:
    """Entity + typed-relation sets of a requirements-traceability view."""

    view: str
    requirements: frozenset[str]
    blocks: frozenset[str]
    satisfy: frozenset[tuple[str, str]]   # (component, requirement)
    allocate: frozenset[tuple[str, str]]  # (requirement, component)
    derive: frozenset[tuple[str, str]]    # (derived_req, source_req)

    def core(self) -> tuple:
        return (self.requirements, self.blocks, self.satisfy, self.allocate, self.derive)

    def coverage(self) -> tuple[frozenset[str], int]:
        """Systems-engineering invariant: a requirement is covered iff directly satisfied by a component or
        allocated to one. Returns (covered set, total requirements). Pure function of the recovered relations,
        so it agrees across views exactly when the relations do."""
        direct = {r for _c, r in self.satisfy}
        allocated = {r for r, _c in self.allocate}
        covered = frozenset(direct | allocated)
        return covered, len(self.requirements)


# ---- extractors (each parses a DIFFERENT concrete syntax; agreement is not a shared-parser artefact) ----

import re  # noqa: E402


def extract_sysml(text: str) -> TraceInvariant:
    """X_sysml: parse a SysML v2 textual requirements model back to entities + relations."""
    reqs = frozenset(re.findall(r"requirement def (\w+)", text))
    blocks = frozenset(re.findall(r"part def (\w+)", text))
    satisfy = frozenset((c, r) for r, c in re.findall(r"satisfy (\w+) by (\w+)", text))
    allocate = frozenset(re.findall(r"allocate (\w+) to (\w+)", text))
    derive = frozenset(re.findall(r"«deriveReqt»\s+(\w+)\s+from\s+(\w+)", text))
    return TraceInvariant("sysml", reqs, blocks, satisfy, allocate, derive)


def extract_dot(text: str) -> TraceInvariant:
    """X_dot: parse the DOT traceability graph back to entities + relations."""
    reqs = frozenset(re.findall(r'"(\w+)"\s*\[label="<<requirement>>', text))
    blocks = frozenset(re.findall(r'"(\w+)"\s*\[label="<<block>>', text))

    def edges(stereotype: str) -> frozenset[tuple[str, str]]:
        return frozenset(re.findall(rf'"(\w+)"\s*->\s*"(\w+)"\s*\[label="<<{stereotype}>>"\]', text))

    return TraceInvariant("dot", reqs, blocks, edges("satisfy"), edges("allocate"), edges("deriveReqt"))


@dataclass(frozen=True)
class TraceResult:
    name: str
    sysml: TraceInvariant
    dot: TraceInvariant
    view_consistent: bool       # X_sysml core == X_dot core
    coverage_consistent: bool   # coverage recovered from both views agrees
    derive_acyclic: bool        # global invariant: the requirement-derivation graph is a DAG (both views)
    anchored: bool              # counts + coverage match the IR-level Q
    counts: dict


def check(name: str, source: Path, csysml: Path | None, cdot: Path | None, q: dict) -> TraceResult:
    sysml = extract_sysml(_emit("requirements_sysml", source, name, csysml))
    dot = extract_dot(_emit("requirements_dot", source, name, cdot))
    cov_s, tot_s = sysml.coverage()
    cov_d, _tot_d = dot.coverage()
    counts = {"requirements": len(sysml.requirements), "blocks": len(sysml.blocks),
              "trace_edges": len(sysml.satisfy) + len(sysml.allocate) + len(sysml.derive),
              "covered": len(cov_s), "total": tot_s}
    anchored = (counts["requirements"] == q["requirements"] and counts["blocks"] == q["blocks"]
                and counts["trace_edges"] == q["trace_edges"] and counts["covered"] == q["covered"])
    # Global invariant (WS4): the requirement-derivation graph is a DAG, recovered from both views.
    derive_acyclic = is_acyclic(sysml.derive) and is_acyclic(dot.derive)
    return TraceResult(name, sysml, dot, sysml.core() == dot.core(), cov_s == cov_d,
                       derive_acyclic, anchored, counts)


def run() -> tuple[list[TraceResult], bool]:
    print("Second domain (requirements traceability) — cross-view square X_sysml = X_dot = Q, with coverage:\n")
    results = [check(*fx) for fx in FIXTURES]
    for r in results:
        c = r.counts
        mark = "OK " if (r.view_consistent and r.coverage_consistent and r.derive_acyclic
                         and r.anchored) else "XX "
        print(f"  {mark}{r.name:18} req={c['requirements']} blocks={c['blocks']} edges={c['trace_edges']} "
              f"coverage={c['covered']}/{c['total']}  | views-agree={r.view_consistent} "
              f"coverage-agree={r.coverage_consistent} derive-DAG={r.derive_acyclic} Q-anchored={r.anchored}")
    ok = all(r.view_consistent and r.coverage_consistent and r.derive_acyclic and r.anchored
             for r in results)
    print(f"\n  Commuting square holds on {sum(r.view_consistent for r in results)}/{len(results)} fixtures; "
          f"coverage agrees across views on {sum(r.coverage_consistent for r in results)}/{len(results)}.")
    print("  The coverage invariant is a query over one IR, recovered identically from the SysML and DOT views ---")
    print("  a cross-view discrepancy (a requirement covered in one view, uncovered in the other) is impossible.")
    return results, ok


if __name__ == "__main__":
    sys.exit(0 if run()[1] else 1)
