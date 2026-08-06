"""Tests for the cross-view-consistency verification (CLAUDE.md 3: unit / integration / performance).

Run from the repo root:  pytest -p no:randomly verification/cross_view_consistency/tests/
Unit tests are CLI-free (they feed hand-written emitted text). Integration and performance tests drive the real
CLI and skip cleanly if no binary is built.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
REPO = HERE.parents[3]
sys.path.insert(0, str(PKG))

import cross_view as cv  # noqa: E402
from extract import (  # noqa: E402
    DotExtractor, MermaidExtractor, MjcfExtractor, SdfExtractor, UrdfExtractor, _round_mass, _axis_from_vector,
    is_acyclic, is_forest,
)
import trace_witness  # noqa: E402
import commute_z3  # noqa: E402
import storage_regime  # noqa: E402

# --- Minimal hand-written emitted fixtures: a 2-link arm, base_link -(j0, axis Z)-> link_0, plus a world weld. ---
URDF = """<?xml version="1.0"?>
<robot name="t">
  <link name="world"/>
  <link name="base_link"><inertial><mass value="8"/></inertial></link>
  <link name="link_0"><inertial><mass value="3.0"/></inertial></link>
  <joint name="j_fix" type="fixed"><parent link="world"/><child link="base_link"/></joint>
  <joint name="j0" type="revolute"><parent link="base_link"/><child link="link_0"/><axis xyz="0 0 1"/></joint>
</robot>"""

SDF = """<?xml version="1.0"?>
<sdf version="1.7"><model name="t">
  <link name="base_link"><inertial><mass>8</mass></inertial></link>
  <link name="link_0"><inertial><mass>3</mass></inertial></link>
  <joint name="j_fix" type="fixed"><parent>world</parent><child>base_link</child></joint>
  <joint name="j0" type="revolute"><parent>base_link</parent><child>link_0</child>
    <axis><xyz>0 0 1</xyz></axis></joint>
</model></sdf>"""

MJCF = """<mujoco model="t"><worldbody>
  <body name="base_link"><inertial mass="8"/>
    <body name="link_0"><inertial mass="3"/><joint name="j0" type="hinge" axis="0 0 1"/></body>
  </body>
</worldbody></mujoco>"""

DOT = '''digraph "t" {
  "base_link" [label="base_link\\n8.0 kg"];
  "link_0" [label="link_0\\n3.0 kg"];
  "world" -> "base_link" [label="j_fix", style=dashed];
  "base_link" -> "link_0" [label="j0\\n(Z)", style=bold];
}'''

MERMAID = '''flowchart TD
    base_link["<b>base_link</b><br/>8.00 kg"]:::link
    link_0["<b>link_0</b><br/>3.00 kg"]:::link
    world(["world"]):::root
    world -.->|"j_fix (fixed)"| base_link
    base_link -->|"j0 (rev, Z)"| link_0'''


# ----------------------------- unit: helpers (normal / boundary / failure) -----------------------------

def test_round_mass_strips_unit_and_rounds():
    assert _round_mass("8.00 kg") == 8.0
    assert _round_mass("3") == 3.0
    assert _round_mass(0.0200004) == 0.02


def test_axis_from_vector_normalises():
    assert _axis_from_vector("0 0 1") == (0, 0, 1)
    assert _axis_from_vector("0 0 -1") == (0, 0, -1)


def test_axis_from_vector_rejects_wrong_arity():
    with pytest.raises(ValueError):
        _axis_from_vector("0 1")


# ----------------------------- unit: global structural invariants (WS4) -----------------------------

def test_is_acyclic_dag_vs_cycle():
    assert is_acyclic({("a", "b"), ("b", "c"), ("a", "c")})        # DAG
    assert not is_acyclic({("a", "b"), ("b", "c"), ("c", "a")})    # cycle
    assert is_acyclic(set())                                        # empty graph is acyclic


def test_is_forest_tree_vs_violations():
    tree = frozenset({("j0", "base", "l0"), ("j1", "l0", "l1"), ("j2", "l0", "l2")})
    assert is_forest(tree)                                          # acyclic, each child one parent
    assert not is_forest(frozenset({("j0", "a", "b"), ("j1", "b", "a")}))   # cycle
    assert not is_forest(frozenset({("j0", "a", "c"), ("j1", "b", "c")}))   # c has two parents


# ----------------------------- unit: each extractor recovers the same invariant -----------------------------

@pytest.mark.parametrize("ext,text", [
    (UrdfExtractor(), URDF), (SdfExtractor(), SDF), (MjcfExtractor(), MJCF),
])
def test_data_extractors_recover_full_invariant(ext, text):
    inv = ext.extract(text)
    assert inv.links == frozenset({"base_link", "link_0"})        # convention W: no synthetic `world`
    assert inv.mass == frozenset({("base_link", 8.0), ("link_0", 3.0)})
    assert inv.actuated_joints == frozenset({"j0"})               # convention F: fixed weld excluded
    assert inv.chain == frozenset({("j0", "base_link", "link_0")})
    assert inv.axes == frozenset({("j0", (0, 0, 1))})


@pytest.mark.parametrize("ext,text", [(DotExtractor(), DOT), (MermaidExtractor(), MERMAID)])
def test_graph_extractors_recover_topology(ext, text):
    inv = ext.extract(text)
    assert inv.links == frozenset({"base_link", "link_0"})
    assert inv.actuated_joints == frozenset({"j0"})
    assert inv.chain == frozenset({("j0", "base_link", "link_0")})


def test_data_and_graph_views_commute():
    """The square: every view's core agrees on the shared topology (the whole point)."""
    invs = [UrdfExtractor().extract(URDF), SdfExtractor().extract(SDF), MjcfExtractor().extract(MJCF),
            DotExtractor().extract(DOT), MermaidExtractor().extract(MERMAID)]
    topo = {(i.links, i.actuated_joints, i.chain) for i in invs}
    assert len(topo) == 1, "views disagree on topology"
    data = {i.core() for i in invs[:3]}
    assert len(data) == 1, "data formats disagree on the full numeric invariant"


def test_extractor_boundary_empty_model():
    inv = UrdfExtractor().extract('<robot name="empty"/>')
    assert inv.links == frozenset() and inv.actuated_joints == frozenset()


def test_extractor_failure_on_malformed_xml():
    with pytest.raises(ValueError):
        UrdfExtractor().extract("<robot><not closed")


def test_extractor_failure_on_empty_text():
    with pytest.raises(ValueError):
        SdfExtractor().extract("   ")


# ----------------------------- integration: real CLI, full corpus -----------------------------

def _have_cli() -> bool:
    return cv.CLI_RELEASE.exists() or cv.CLI_DEBUG.exists()


@pytest.mark.skipif(not _have_cli(), reason="no hymeko CLI binary built")
def test_integration_cross_view_square_holds():
    results = cv.run(out_json=None)
    assert results, "no kinematic fixtures found"
    bad = [r.fixture for r in results
           if not (r.exact_consistent and r.topo_consistent and r.forest_ok)]
    assert not bad, f"cross-view square / forest invariant failed on: {bad}"


@pytest.mark.skipif(not _have_cli(), reason="no hymeko CLI binary built (sysml_cell has no committed fallback)")
def test_non_robotics_trace_domain():
    """WS1: the requirements-traceability cross-view square + coverage invariant, over both fixtures."""
    results, ok = trace_witness.run()
    assert ok, "requirements-traceability domain failed view/coverage/Q consistency"
    assert len(results) == 2
    for r in results:
        assert r.view_consistent and r.coverage_consistent and r.anchored, r.name
        assert r.derive_acyclic, f"{r.name}: requirement-derivation graph not a DAG"


@pytest.mark.skipif(not _have_cli(), reason="no hymeko CLI binary built")
def test_drift_prevention_demo():
    """WS2: a semantic edit is benign under single-source emission (0 drift) and divergent under multi-file
    maintenance (>=1 drift)."""
    import drift_demo
    hymeko_drift, pairwise_drift, ok = drift_demo.run()
    assert hymeko_drift == 0, f"single-source emission drifted ({hymeko_drift})"
    assert pairwise_drift >= 1, "multi-file baseline did not drift (demo would be a strawman)"
    assert ok


@pytest.mark.skipif(not _have_cli(), reason="no hymeko CLI binary built")
@pytest.mark.parametrize("fmt,marker", [("requirements_sysml", "requirement def"),
                                        ("requirements_dot", "<<requirement>>")])
def test_regression_template_only_emitters_route_to_template_path(fmt, marker):
    """Regression for the hymeko_cli dispatch fix: template-only transforms (emit()->None but a declared
    template_dir) must reach render_from_templates instead of erroring 'model extraction failed'. Before the
    fix this returned empty/non-zero; now it emits the full model."""
    import subprocess
    cli = cv.cli_path()
    fixture = REPO / "data" / "paper" / "traceability_smc.hymeko"
    out = subprocess.run([str(cli), "emit", str(fixture), "--format", fmt, "--name", "pick_place_cell"],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, f"{fmt} emit failed: {out.stderr[:200]}"
    assert marker in out.stdout, f"{fmt} output missing {marker!r}"


# ----------------------------- logical + symbolic proofs -----------------------------

def test_z3_cross_view_proof():
    assert commute_z3.theorem_positive(), "Z3 failed to prove shared-query => agreement"
    assert commute_z3.theorem_negative(), "Z3 failed to witness drift for an untethered view"


def test_storage_regime_robotics_is_small_constant():
    rows = storage_regime.regime_table()
    assert abs(rows[0][2] - 2.0) < 1e-9          # robotics (d_bar=2) -> rho=2.0, a constant (not ->1)
    assert rows[-1][2] < 1.02                     # very-high-arity -> rho ~ 1


# ----------------------------- performance: wall + RSS budget (asserted, not printed) -----------------------------

@pytest.mark.skipif(not _have_cli(), reason="no hymeko CLI binary built")
def test_performance_single_fixture_extraction_budget():
    import psutil

    cli = cv.cli_path()
    fixture = REPO / "data" / "robotics" / "fanuc_lrmate.hymeko"
    # warm-up
    cv.check_fixture(cli, fixture)
    samples = []
    for _ in range(7):
        t0 = time.perf_counter()
        cv.check_fixture(cli, fixture)
        samples.append(time.perf_counter() - t0)
    median = statistics.median(samples)
    iqr = statistics.quantiles(samples, n=4)[2] - statistics.quantiles(samples, n=4)[0]
    worst = max(samples)
    rss_mb = psutil.Process().memory_info().rss / 1e6
    print(f"\n  extraction wall: median={median*1e3:.0f}ms IQR={iqr*1e3:.0f}ms worst={worst*1e3:.0f}ms; "
          f"RSS={rss_mb:.0f}MB")
    assert median < 1.0, f"median {median*1e3:.0f}ms over 1000ms budget"   # 5 subprocess emits + parse
    assert rss_mb < 512, f"RSS {rss_mb:.0f}MB over 512MB budget"
