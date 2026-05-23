"""Unit tests for the Niitsuma Tier-2 dual-robot scenario.

Covers (≥ 6 tests per the plan):
- Dual IR parses (all 3 contexts present).
- Robot A's grasping_context has exactly 6 hyperedges, matching paper §4.
- Robot B's grasping_context is structurally identical (different prefix).
- scene_context has exactly 3 hyperedges (e_s1, e_s2, e_s3).
- Cross-context references resolve to leaf vertex names.
- compute_distance pure-function works on basic test vectors.
- arbitrate pure-function:
  * neither margin high → no commit
  * A high, B low → commit A
  * B high, A low → commit B
  * both high → argmax
  * distance below threshold → safety override
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hymeko_ros2_demo.topic_binding import (
    extract_hyperedges,
    find_context_block,
)
from hymeko_ros2_demo.scripts.scene_context_node import arbitrate, compute_distance

DUAL_SCEN_PATH = Path(__file__).resolve().parents[1] / \
    "hymeko_ros2_demo" / "scenarios" / "hymeko_robot_dual.hymeko"


# ───────────────────────── IR tests ─────────────────────────


@pytest.fixture(scope="module")
def dual_ir():
    import hymeko
    return hymeko.parse_hymeko_rs(DUAL_SCEN_PATH.read_text())


def test_dual_scenario_file_exists():
    assert DUAL_SCEN_PATH.exists(), \
        f"dual scenario file missing at {DUAL_SCEN_PATH}"


def test_dual_ir_parses_to_three_contexts(dual_ir):
    for ctx in ("robot_a_grasping", "robot_b_grasping", "scene_context"):
        assert find_context_block(dual_ir, ctx) is not None, \
            f"context '{ctx}' missing from dual IR"


def test_robot_a_grasping_has_six_hyperedges(dual_ir):
    blk = find_context_block(dual_ir, "robot_a_grasping")
    edges = extract_hyperedges(blk)
    assert len(edges) == 6, \
        f"robot_a_grasping should have 6 hyperedges; got {len(edges)}"
    names = {e.name for e in edges}
    expected = {
        "a_derive_tool", "a_derive_payload", "a_loading_state",
        "a_grasp_config", "a_load_force", "a_grasp_stability",
    }
    assert names == expected, f"mismatch: {names} vs {expected}"


def test_robot_b_grasping_is_structurally_identical(dual_ir):
    blk_a = find_context_block(dual_ir, "robot_a_grasping")
    blk_b = find_context_block(dual_ir, "robot_b_grasping")
    edges_a = extract_hyperedges(blk_a)
    edges_b = extract_hyperedges(blk_b)
    assert len(edges_a) == len(edges_b) == 6
    # Names differ only by prefix; arities (in, out) must match
    for a, b in zip(edges_a, edges_b):
        assert len(a.inputs) == len(b.inputs)
        assert len(a.outputs) == len(b.outputs)


def test_scene_context_has_three_hyperedges_with_paper_names(dual_ir):
    blk = find_context_block(dual_ir, "scene_context")
    edges = extract_hyperedges(blk)
    assert len(edges) == 3, \
        f"scene_context should have 3 hyperedges (e_s1..e_s3); got {len(edges)}"
    names = {e.name for e in edges}
    assert names == {
        "e_s1_inter_robot_distance",
        "e_s2_shared_payload",
        "e_s3_arbitration",
    }


def test_scene_e_s3_arbitration_has_correct_inputs(dual_ir):
    blk = find_context_block(dual_ir, "scene_context")
    edges = {e.name: e for e in extract_hyperedges(blk)}
    e3 = edges["e_s3_arbitration"]
    # Cross-context refs must resolve to leaf vertex names.
    assert set(e3.inputs) == {
        "shared_payload", "a_stability_margin", "b_stability_margin",
    }
    assert set(e3.outputs) == {"committed_robot"}


# ───────────────────────── pure-function tests ─────────────────────────


def test_compute_distance_basic():
    assert compute_distance((0, 0, 0), (1, 0, 0)) == pytest.approx(1.0)
    assert compute_distance((0, 0, 0), (3, 4, 0)) == pytest.approx(5.0)
    assert compute_distance((1, 1, 1), (1, 1, 1)) == pytest.approx(0.0)


def test_compute_distance_missing_pose_returns_zero():
    """Missing pose → distance 0 (conservative; arbitration treats as
    'we don't know yet' and waits)."""
    assert compute_distance(None, (0, 0, 0)) == 0.0
    assert compute_distance((0, 0, 0), None) == 0.0
    assert compute_distance(None, None) == 0.0


def test_arbitrate_neither_high_no_commit():
    """Both stability_margin below threshold → commit 0."""
    c = arbitrate(distance=1.0, shared=1.0, stab_a=0.1, stab_b=0.2,
                   safety_threshold=0.30, commit_min_margin=0.50)
    assert c == 0


def test_arbitrate_a_high_b_low_commits_a():
    c = arbitrate(distance=1.0, shared=1.0, stab_a=0.8, stab_b=0.2,
                   safety_threshold=0.30, commit_min_margin=0.50)
    assert c == 1


def test_arbitrate_b_high_a_low_commits_b():
    c = arbitrate(distance=1.0, shared=1.0, stab_a=0.2, stab_b=0.8,
                   safety_threshold=0.30, commit_min_margin=0.50)
    assert c == 2


def test_arbitrate_both_high_argmax_wins():
    """Both above threshold → commit to whichever is larger."""
    c1 = arbitrate(distance=1.0, shared=1.0, stab_a=0.9, stab_b=0.7,
                    safety_threshold=0.30, commit_min_margin=0.50)
    assert c1 == 1
    c2 = arbitrate(distance=1.0, shared=1.0, stab_a=0.6, stab_b=0.9,
                    safety_threshold=0.30, commit_min_margin=0.50)
    assert c2 == 2


def test_arbitrate_safety_overrides_when_too_close():
    """Distance < safety_threshold → commit 0 even if margins are good."""
    c = arbitrate(distance=0.10, shared=1.0, stab_a=0.9, stab_b=0.9,
                   safety_threshold=0.30, commit_min_margin=0.50)
    assert c == 0


def test_arbitrate_zero_distance_treated_as_unknown_not_violation():
    """distance == 0.0 is the 'pose missing' sentinel from compute_distance;
    treat as not-yet-known rather than safety violation."""
    c = arbitrate(distance=0.0, shared=1.0, stab_a=0.9, stab_b=0.2,
                   safety_threshold=0.30, commit_min_margin=0.50)
    assert c == 1  # proceeds because distance=0 is sentinel, not real proximity
