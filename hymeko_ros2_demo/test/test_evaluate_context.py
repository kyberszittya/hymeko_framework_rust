"""Unit tests for the pure per-cycle evaluation core (`evaluate_context`).

This is the control-cycle computation extracted from
``GraspingContextNode._tick`` on 2026-06-10 so it can be tested and
benchmarked without rclpy. These tests pin its contract; they import
only ``topic_binding`` (pure Python, no ROS).
"""
from __future__ import annotations

import pytest

from hymeko_ros2_demo.topic_binding import (
    Hyperedge,
    default_edge_aggregate,
    evaluate_context,
)

# The grasping context's 6 edges (scenarios/hymeko_robot.hymeko).
_GRASP = [
    Hyperedge("derive_tool", ("active_tool",), ("tool_params",)),
    Hyperedge("derive_payload", ("active_payload",), ("payload_params",)),
    Hyperedge("loading_state", ("tool_params", "payload_params"), ("loaded_state",)),
    Hyperedge("grasp_config", ("mode_parallel", "payload_params"), ("configuration",)),
    Hyperedge("load_force", ("configuration", "robot_pose"), ("force_vector",)),
    Hyperedge("grasp_stability", ("force_vector", "grip_force"), ("stability_margin",)),
]
_INPUTS = {"robot_pose": 0.3, "active_tool": 1.0, "active_payload": 3.0,
           "mode_parallel": 0.0, "grip_force": 5.001}


def test_determinism():
    a = evaluate_context(_GRASP, dict(_INPUTS))
    b = evaluate_context(_GRASP, dict(_INPUTS))
    assert a == b


def test_external_inputs_not_overwritten():
    v = dict(_INPUTS)
    evaluate_context(_GRASP, v)
    for k, val in _INPUTS.items():
        assert v[k] == val, f"external input {k} was clobbered"


def test_topological_propagation():
    # e1 produces tool_params; e3 consumes it. After one pass all six
    # outputs must be present (each edge ran and wrote its output).
    v = dict(_INPUTS)
    evaluate_context(_GRASP, v)
    for produced in ("tool_params", "payload_params", "loaded_state",
                     "configuration", "force_vector", "stability_margin"):
        assert produced in v


def test_outputs_in_unit_range():
    v = evaluate_context(_GRASP, dict(_INPUTS))
    for k in ("loaded_state", "configuration", "force_vector", "stability_margin"):
        assert 0.0 <= v[k] <= 1.0


def test_stability_edge_uses_special_form():
    # grasp_stability must route through aggregate_grasp_stability, not
    # the clamped mean — equal F_l and normalised F_g give margin ~1.
    edge = _GRASP[-1]
    out = default_edge_aggregate(edge, {"force_vector": 0.5, "grip_force": 5.0})
    # F_g_norm = 0.5, F_l = 0.5 -> diff 0 -> margin 1.0
    assert out == pytest.approx(1.0, abs=1e-6)


def test_default_aggregate_is_clamped_mean():
    edge = Hyperedge("loading_state", ("a", "b"), ("c",))
    assert default_edge_aggregate(edge, {"a": 0.2, "b": 0.4}) == pytest.approx(0.3)
    assert default_edge_aggregate(edge, {"a": 5.0, "b": 7.0}) == 1.0  # clamp
    assert default_edge_aggregate(edge, {}) == 0.0  # empty -> 0


def test_empty_edges_leaves_state_untouched():
    v = dict(_INPUTS)
    evaluate_context([], v)
    assert v == _INPUTS
