"""Phase 10 (2026-05-19): end-to-end multi-objective P-graph
pipeline tests.

Drives `hymeko_pgraph_dump` against the new
`data/hsikan/sweep_msg_multicost.hymeko` fixture under three
weight regimes; asserts the new DTO fields surface correctly and
that different weight vectors pick different architectures.

The Rust core (`solve_with_options` weighted-sum cost) was
already unit-tested in
`hymeko_pgraph/tests/multi_objective.rs`; this Python suite pins
the **JSON DTO + dump-binary CLI** layer the Python sweep drivers
consume.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "data" / "hsikan" / "sweep_msg_multicost.hymeko"


def _find_dump() -> Path:
    rel = REPO_ROOT / "target" / "release" / "hymeko_pgraph_dump"
    dbg = REPO_ROOT / "target" / "debug" / "hymeko_pgraph_dump"
    if rel.exists():
        return rel
    if dbg.exists():
        return dbg
    pytest.skip("hymeko_pgraph_dump not built")


def _dump(weights: str | None = None) -> dict:
    cmd = [str(_find_dump()), str(FIXTURE), "--algorithm", "abb"]
    if weights:
        cmd.extend(["--weights", weights])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, f"dump rc={proc.returncode}\nstderr={proc.stderr}"
    return json.loads(proc.stdout)


def test_cost_dimensions_alphabetised_in_dto():
    """Phase 10 DTO field: `cost_dimensions` is alphabetised in
    the dump JSON, matching the lowering contract."""
    j = _dump()
    assert j["cost_dimensions"] == ["gpu_cost", "quality_drop", "time_cost"]


def test_scalar_path_has_none_cost_weights_echo():
    """No `--weights` supplied → DTO echoes `cost_weights_echo =
    None`; ABB still runs (scalar cost fallback). The
    `abb_cost_breakdown` field surfaces the per-dimension
    contribution of whatever the scalar-cost ABB picked."""
    j = _dump()
    assert j["cost_weights_echo"] is None
    assert j["abb_cost_breakdown"] is not None
    assert len(j["abb_cost_breakdown"]) == 3


def test_three_weight_regimes_pick_distinct_architectures():
    """The headline Phase 10 empirical claim: three different
    weight vectors over the same fixture pick three different
    ABB-selected unit sets. Confirms the Pareto-aware NAS
    behaviour at the JSON/CLI boundary."""
    gpu_only = _dump("1,0,0")["abb"]["units"]
    quality_only = _dump("0,1,0")["abb"]["units"]
    balanced_quality_bias = _dump("1,5,1")["abb"]["units"]

    # gpu-only and time-only collapse to the same cheap path on
    # this fixture (the cost-minimum architecture is also the
    # time-minimum); but gpu-only vs quality-only must differ.
    assert set(gpu_only) != set(quality_only), (
        f"gpu-only and quality-only should pick different units;\n"
        f"gpu_only={gpu_only}\nquality_only={quality_only}"
    )

    # Balanced is a Pareto pick: keep cheap cycle/model but
    # invest in long training to reduce quality_drop.
    assert "train_long" in balanced_quality_bias
    assert "cycle_topk_m4" in balanced_quality_bias
    assert balanced_quality_bias != quality_only
    assert balanced_quality_bias != gpu_only


def test_cost_weights_echo_round_trips():
    """The exact `cost_weights` vector passed to the binary is
    surfaced back in `cost_weights_echo`."""
    j = _dump("0.5,1.0,0.25")
    assert j["cost_weights_echo"] == [0.5, 1.0, 0.25]


def test_abb_cost_breakdown_sums_match_per_dim_unit_costs():
    """`abb_cost_breakdown` is a list of (dim_name, sum_of_per_unit_costs)
    over the ABB-selected units. The sum within each dimension
    should match the raw cost vectors in the fixture."""
    # Under (0, 1, 0) ABB picks m64 + h16 + long, whose
    # quality_drop entries are 0 + 10 + 0 = 10. Pin that.
    j = _dump("0,1,0")
    breakdown = dict(j["abb_cost_breakdown"])
    assert breakdown["quality_drop"] == pytest.approx(10.0)
    # m64 + h16 + long gpu costs: 160 + 60 + 10 = 230
    assert breakdown["gpu_cost"] == pytest.approx(230.0)
    # m64 + h16 + long time costs: 160 + 60 + 120 = 340
    assert breakdown["time_cost"] == pytest.approx(340.0)


def test_dot_cost_matches_breakdown_weighted_sum():
    """`abb.cost` is the weighted dot product
    `weights · cost_vector` of the selection. Confirms the
    breakdown rounds-trip the dot product."""
    j = _dump("2,3,5")
    breakdown = dict(j["abb_cost_breakdown"])
    expected = (
        2.0 * breakdown["gpu_cost"]
        + 3.0 * breakdown["quality_drop"]
        + 5.0 * breakdown["time_cost"]
    )
    assert j["abb"]["cost"] == pytest.approx(expected, rel=1e-9)


def test_axiom_certificate_independent_of_weights():
    """Sanity: the Friedler certificate (canonical + extension)
    is independent of `cost_weights`. The same `.hymeko` schema
    passes/fails the axiom check regardless of which dot product
    the engine uses."""
    j_default = _dump()
    j_weighted = _dump("3,1,7")
    assert j_default["canonical_full"]["status"] == j_weighted["canonical_full"]["status"]
    assert j_default["extension_full"]["status"] == j_weighted["extension_full"]["status"]
