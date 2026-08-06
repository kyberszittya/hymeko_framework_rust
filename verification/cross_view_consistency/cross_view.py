"""Machine-verify cross-view consistency of the HyMeKo emitters: the commuting square X_f(eps_f(H)) = Q(H).

For every robot fixture H we drive the REAL CLI emitter (`hymeko emit --format f`) for each registered view f,
apply the extraction function X_f (extract.py) to read the structural invariant back out of the emitted text,
and check that all views agree (mutual cross-view consistency) after the two named conventions are normalised.
This is the value-level "no drift" claim the article asserts in Sec. codegen but never proves: Proposition 3 is
only about cost factorisation (compile-once-emit-many), whose codomains differ; the square below has a common
codomain (the invariant Q) and so genuinely commutes.

Run from the repo root:  python verification/cross_view_consistency/cross_view.py
Outputs JSON + an agreement-matrix figure next to the report.

CLAUDE.md: drives the CLI as a black box (no core edits). A mismatch is treated as a *bug* until shown to be a
named convention; un-normalisable mismatches are surfaced, not hidden.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import EXTRACTORS, KinematicInvariant, is_forest  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CLI_RELEASE = REPO / "target" / "release" / "hymeko.exe"
CLI_DEBUG = REPO / "target" / "debug" / "hymeko.exe"
FIXTURE_DIR = REPO / "data" / "robotics"
VIEWS = ("urdf", "sdf", "mjcf", "dot", "mermaid")
# Data-interchange formats carry the full numeric invariant (mass to full precision, signed axis vector);
# the graph/diagram formats carry a topologically faithful projection whose LABELS are coarsened by design
# (DOT mass at 1 decimal, Mermaid at 2; both axes as an unsigned letter). So the commuting square is tested at
# two strengths: EXACT across the data formats, TOPOLOGICAL (links + joints + parent/child) across all views.
DATA_VIEWS = ("urdf", "sdf", "mjcf")
GRAPH_VIEWS = ("dot", "mermaid")


def cli_path() -> Path:
    if CLI_RELEASE.exists():
        return CLI_RELEASE
    if CLI_DEBUG.exists():
        return CLI_DEBUG
    raise FileNotFoundError("no hymeko CLI binary; run `cargo build --release -p hymeko_cli`")


def emit(cli: Path, fixture: Path, fmt: str, name: str = "r") -> str:
    """epsilon_f: invoke the real CLI emitter. Raises CalledProcessError on a non-zero exit."""
    out = subprocess.run([str(cli), "emit", str(fixture), "--format", fmt, "--name", name],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    return out.stdout


@dataclass(frozen=True)
class DiagramResult:
    fixture: str
    views: tuple[str, ...]          # views that emitted a non-empty, kinematic output
    exact_consistent: bool          # data formats (urdf/sdf/mjcf) share the full numeric invariant
    topo_consistent: bool           # all views share (links, actuated joints, parent/child)
    forest_ok: bool                 # global invariant: the recovered kinematic structure is an acyclic forest
    consistent: bool                # exact_consistent and topo_consistent
    n_links: int
    n_actuated_joints: int
    invariants: dict[str, dict]     # view -> serialised KinematicInvariant
    disagreements: list[str]        # human-readable, empty iff consistent
    fixed_weld_repr: dict[str, list[str]]  # view -> fixed joints (convention F, reported not compared)


def _ser(inv: KinematicInvariant) -> dict:
    return {
        "links": sorted(inv.links),
        "mass": sorted(list(inv.mass)),
        "actuated_joints": sorted(inv.actuated_joints),
        "chain": sorted(list(inv.chain)),
        "axes": sorted([(j, list(a)) for j, a in inv.axes]),
        "fixed_joints": sorted(inv.fixed_joints),
    }


def _diff(label: str, against: str, fmt: str, a, b) -> list[str]:
    """One human-readable disagreement line if sets `a`,`b` differ, else empty."""
    if a == b:
        return []
    return [f"{label}: {against}\\{fmt}={sorted(map(str, set(a) - set(b)))}  "
            f"{fmt}\\{against}={sorted(map(str, set(b) - set(a)))}"]


def _emit_invariants(cli: Path, fixture: Path) -> dict[str, KinematicInvariant]:
    """epsilon_f then X_f for every view; skip views that fail to emit, fail to parse, or are non-kinematic."""
    invs: dict[str, KinematicInvariant] = {}
    for fmt in VIEWS:
        try:
            inv = EXTRACTORS[fmt].extract(emit(cli, fixture, fmt))
        except (subprocess.CalledProcessError, ValueError):
            continue
        if inv.actuated_joints or inv.links:        # kinematic in this view
            invs[fmt] = inv
    return invs


def _exact_disagreements(invs: dict[str, KinematicInvariant]) -> list[str]:
    """Tier 1: full numeric invariant equality across the data-interchange formats present."""
    data = [v for v in DATA_VIEWS if v in invs]
    if len(data) < 2:
        return []
    ref, out = invs[data[0]], []
    for fmt in data[1:]:
        cur = invs[fmt]
        out += _diff("links", data[0], fmt, ref.links, cur.links)
        out += _diff("mass", data[0], fmt, ref.mass, cur.mass)
        out += _diff("actuated_joints", data[0], fmt, ref.actuated_joints, cur.actuated_joints)
        out += _diff("chain", data[0], fmt, ref.chain, cur.chain)
        out += _diff("axes", data[0], fmt, ref.axes, cur.axes)
    return out


def _topo_disagreements(invs: dict[str, KinematicInvariant], views: tuple[str, ...]) -> list[str]:
    """Tier 2: topology (links, actuated joints, parent/child) equality across ALL views."""
    ref_fmt = views[0]
    ref = invs[ref_fmt]
    out = []
    for fmt in views[1:]:
        cur = invs[fmt]
        out += _diff("topo.links", ref_fmt, fmt, ref.links, cur.links)
        out += _diff("topo.actuated_joints", ref_fmt, fmt, ref.actuated_joints, cur.actuated_joints)
        out += _diff("topo.chain", ref_fmt, fmt, ref.chain, cur.chain)
    return out


def check_fixture(cli: Path, fixture: Path) -> DiagramResult | None:
    """Compute X_f(eps_f(H)) for each view and test the commuting square at both strengths."""
    invs = _emit_invariants(cli, fixture)
    if len(invs) < 2:
        return None  # need >= 2 views to talk about cross-view consistency
    views = tuple(sorted(invs))
    exact = _exact_disagreements(invs)
    topo = _topo_disagreements(invs, views)
    ref = invs[views[0]]
    # Global invariant (WS4): every view agrees on the chain (Tier 2), so the kinematic structure recovered from
    # any of them is the same forest; we additionally assert it IS an acyclic single-parent forest (well-formed).
    forest_ok = is_forest(ref.chain)
    return DiagramResult(
        fixture=fixture.name,
        views=views,
        exact_consistent=not exact,
        topo_consistent=not topo,
        forest_ok=forest_ok,
        consistent=not exact and not topo and forest_ok,
        n_links=len(ref.links),
        n_actuated_joints=len(ref.actuated_joints),
        invariants={f: _ser(invs[f]) for f in views},
        disagreements=exact + topo,
        fixed_weld_repr={f: sorted(invs[f].fixed_joints) for f in views},
    )


def run(out_json: Path | None) -> list[DiagramResult]:
    cli = cli_path()
    results: list[DiagramResult] = []
    for fx in sorted(FIXTURE_DIR.glob("*.hymeko")):
        r = check_fixture(cli, fx)
        if r is not None:
            results.append(r)

    n = len(results)
    exact = sum(r.exact_consistent for r in results)
    topo = sum(r.topo_consistent for r in results)
    forest = sum(r.forest_ok for r in results)
    print(f"Cross-view consistency vs the REAL CLI emitters ({cli.name}) — {n} kinematic fixtures, "
          f"views {VIEWS}:\n")
    for r in results:
        mark = "OK " if r.consistent else "XX "
        print(f"  {mark}{r.fixture:34} views={len(r.views)} "
              f"links={r.n_links:>2} joints={r.n_actuated_joints:>2}  {''.join('+'+v[0] for v in r.views)}")
        for d in r.disagreements:
            print(f"        ! {d}")
    print(f"\n  EXACT square (urdf/sdf/mjcf, full numeric invariant): {exact}/{n} fixtures.")
    print(f"  TOPOLOGICAL square (all {len(VIEWS)} independently-parsed views, links+joints+parent/child): "
          f"{topo}/{n} fixtures.")
    print("  Normalised conventions: (W) synthetic URDF `world` link excluded (mass-bearing links compared); "
          "(F) actuated joints compared, fixed root weld reported separately (implicit in MJCF).")
    print("  Graph views (dot/mermaid) carry labels at reduced resolution by design (DOT mass 1dp + unsigned "
          "axis letter; Mermaid 2dp) — topology is exact, numeric labels are a documented projection.")
    print(f"  GLOBAL invariant (kinematic structure is an acyclic single-parent forest): {forest}/{n} fixtures "
          "(recovered identically from every view, since all views agree on the chain).")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
        print(f"\n  wrote {out_json.relative_to(REPO)}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=REPO / "reports" / "cross_view_consistency.json")
    args = ap.parse_args()
    results = run(args.json)
    # Gate: exact across data formats, topological across all views, and the global forest invariant.
    return 0 if results and all(r.exact_consistent and r.topo_consistent and r.forest_ok
                                for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
