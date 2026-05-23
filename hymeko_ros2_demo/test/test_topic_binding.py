"""Unit tests for the topic_binding IR walker + aggregation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from hymeko_ros2_demo.topic_binding import (
    Hyperedge,
    aggregate_grasp_stability,
    extract_hyperedges,
    find_context_block,
)

SCEN_PATH = Path(__file__).resolve().parents[1] / \
    "hymeko_ros2_demo" / "scenarios" / "hymeko_robot.hymeko"


@pytest.fixture(scope="module")
def ir():
    import hymeko
    return hymeko.parse_hymeko_rs(SCEN_PATH.read_text())


def test_scenario_file_exists():
    assert SCEN_PATH.exists(), \
        f"scenario file missing at {SCEN_PATH}"


def test_grasping_context_block_found(ir):
    blk = find_context_block(ir, "grasping_context")
    assert blk is not None
    assert blk.get("name") == "grasping_context"


def test_grasping_context_has_six_hyperedges(ir):
    blk = find_context_block(ir, "grasping_context")
    edges = extract_hyperedges(blk)
    assert len(edges) == 6, \
        f"expected 6 hyperedges (paper eq.\\ grasp_hyperedges); got {len(edges)}"


def test_grasping_edges_have_paper_names(ir):
    blk = find_context_block(ir, "grasping_context")
    names = {e.name for e in extract_hyperedges(blk)}
    expected = {
        "derive_tool", "derive_payload", "loading_state",
        "grasp_config", "load_force", "grasp_stability",
    }
    assert names == expected, f"mismatch: got {names}; expected {expected}"


def test_grasp_stability_has_correct_signed_incidence(ir):
    blk = find_context_block(ir, "grasping_context")
    edges = {e.name: e for e in extract_hyperedges(blk)}
    e = edges["grasp_stability"]
    assert set(e.inputs) == {"force_vector", "grip_force"}
    assert set(e.outputs) == {"stability_margin"}


def test_maintenance_and_safety_contexts_present(ir):
    """Tier 2 hook: the file should already declare the contexts
    even though Tier 1 only binds grasping."""
    for ctx in ("maintenance_context", "safety_context"):
        blk = find_context_block(ir, ctx)
        assert blk is not None, f"{ctx} missing"
        assert len(extract_hyperedges(blk)) >= 2, \
            f"{ctx} should declare >=2 hyperedges (paper §6 figure)"


def test_unknown_context_returns_none(ir):
    assert find_context_block(ir, "no_such_context") is None


def test_aggregate_grasp_stability_decreasing_in_gap():
    """1/(1+|F_l - F_g|) is 1 at zero gap and decreases monotonically."""
    a = aggregate_grasp_stability(
        {"force_vector": 5.0, "grip_force": 5.0})
    b = aggregate_grasp_stability(
        {"force_vector": 5.0, "grip_force": 4.0})
    c = aggregate_grasp_stability(
        {"force_vector": 5.0, "grip_force": 0.0})
    assert a["stability_margin"] == pytest.approx(1.0)
    assert a["stability_margin"] > b["stability_margin"] > c["stability_margin"]


def test_aggregate_grasp_stability_handles_missing_inputs():
    """Missing inputs default to 0; output stays finite."""
    out = aggregate_grasp_stability({})
    assert "stability_margin" in out
    assert 0.0 < out["stability_margin"] <= 1.0
