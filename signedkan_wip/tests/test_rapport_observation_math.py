"""Tests for the pure-Python pose → observation math (Stage G)."""
from __future__ import annotations

import math

import pytest


def test_distance_close_fires_below_threshold():
    from signedkan_wip.src.rapport_ros2.observation_math import (
        Pose2D, derive_pairwise_observations,
    )
    a = Pose2D(0.0, 0.0, 0.0)
    b = Pose2D(1.0, 0.0, 0.0)
    events = derive_pairwise_observations(
        "alice", a, "bob", b,
        distance_close=1.5, gaze_cosine=0.8,
    )
    kinds = [e.kind for e in events]
    assert "distance_close" in kinds


def test_distance_close_does_not_fire_above_threshold():
    from signedkan_wip.src.rapport_ros2.observation_math import (
        Pose2D, derive_pairwise_observations,
    )
    a = Pose2D(0.0, 0.0, 0.0)
    b = Pose2D(3.0, 0.0, 0.0)
    events = derive_pairwise_observations(
        "alice", a, "bob", b,
        distance_close=1.5, gaze_cosine=0.8,
    )
    kinds = [e.kind for e in events]
    assert "distance_close" not in kinds


def test_gaze_at_fires_when_heading_aligned():
    from signedkan_wip.src.rapport_ros2.observation_math import (
        Pose2D, derive_pairwise_observations,
    )
    # alice at origin, heading along +x; bob at (2, 0) — alice looks at bob.
    a = Pose2D(0.0, 0.0, 0.0)
    b = Pose2D(2.0, 0.0, 0.0)
    events = derive_pairwise_observations(
        "alice", a, "bob", b,
        distance_close=0.5, gaze_cosine=0.8,  # dist > 0.5 so no distance event
    )
    kinds = [e.kind for e in events]
    assert "gaze_at" in kinds
    assert "distance_close" not in kinds


def test_gaze_at_does_not_fire_when_heading_misaligned():
    from signedkan_wip.src.rapport_ros2.observation_math import (
        Pose2D, derive_pairwise_observations,
    )
    # alice looks along +x; bob is at (0, 2) — perpendicular.
    a = Pose2D(0.0, 0.0, 0.0)
    b = Pose2D(0.0, 2.0, 0.0)
    events = derive_pairwise_observations(
        "alice", a, "bob", b,
        distance_close=0.5, gaze_cosine=0.8,
    )
    assert "gaze_at" not in [e.kind for e in events]


def test_heading_cosine_extreme_cases():
    from signedkan_wip.src.rapport_ros2.observation_math import (
        Pose2D, heading_cosine_to_target,
    )
    a = Pose2D(0.0, 0.0, 0.0)
    # b directly in front (alice looks +x, b at +x)
    assert heading_cosine_to_target(a, Pose2D(1.0, 0.0, 0.0)) == pytest.approx(1.0)
    # b directly behind
    assert heading_cosine_to_target(a, Pose2D(-1.0, 0.0, 0.0)) == pytest.approx(-1.0)
    # b at 90° (cosine 0)
    assert abs(heading_cosine_to_target(a, Pose2D(0.0, 1.0, 0.0))) < 1e-9
    # b coincident → 0 (no defined direction)
    assert heading_cosine_to_target(a, Pose2D(0.0, 0.0, 0.0)) == 0.0


def test_quaternion_to_yaw_identity():
    from signedkan_wip.src.rapport_ros2.observation_math import quaternion_to_yaw
    # Identity quaternion = 0 yaw
    assert quaternion_to_yaw(0, 0, 0, 1) == pytest.approx(0.0)
    # 90° about Z: q = (0, 0, sin(pi/4), cos(pi/4))
    s = math.sin(math.pi / 4); c = math.cos(math.pi / 4)
    assert quaternion_to_yaw(0, 0, s, c) == pytest.approx(math.pi / 2, abs=1e-7)


def test_withdrawal_fires_when_distance_increases_fast():
    from signedkan_wip.src.rapport_ros2.observation_math import (
        Pose2D, derive_withdrawal,
    )
    src_past = Pose2D(0.0, 0.0, 0.0, stamp_s=0.0)
    dst_past = Pose2D(1.0, 0.0, 0.0, stamp_s=0.0)
    src_now = Pose2D(0.0, 0.0, 0.0, stamp_s=1.0)
    dst_now = Pose2D(2.0, 0.0, 0.0, stamp_s=1.0)
    # Distance went 1 → 2 in 1 s; rate = 1 m/s > 0.3 m/s threshold
    events = derive_withdrawal(
        "alice", src_now, src_past, "bob", dst_now, dst_past,
        rate_threshold=0.3,
    )
    assert len(events) == 1
    assert events[0].kind == "withdrawal"


def test_withdrawal_does_not_fire_when_stable():
    from signedkan_wip.src.rapport_ros2.observation_math import (
        Pose2D, derive_withdrawal,
    )
    src_past = Pose2D(0.0, 0.0, 0.0, stamp_s=0.0)
    dst_past = Pose2D(1.0, 0.0, 0.0, stamp_s=0.0)
    src_now = Pose2D(0.0, 0.0, 0.0, stamp_s=1.0)
    dst_now = Pose2D(1.05, 0.0, 0.0, stamp_s=1.0)
    events = derive_withdrawal(
        "alice", src_now, src_past, "bob", dst_now, dst_past,
        rate_threshold=0.3,
    )
    assert events == []


def test_derive_all_observations_three_agents():
    from signedkan_wip.src.rapport_ros2.observation_math import (
        Pose2D, derive_all_observations,
    )
    poses = {
        "alice": Pose2D(0.0, 0.0, 0.0, stamp_s=1.0),
        "bob":   Pose2D(1.0, 0.0, math.pi, stamp_s=1.0),    # facing alice
        "r1":    Pose2D(0.0, -1.0, math.pi/2, stamp_s=1.0), # facing alice/bob midpoint
    }
    events = derive_all_observations(
        poses, poses_past=None,
        distance_close=1.5, gaze_cosine=0.8, withdrawal_rate=0.3,
    )
    # Alice and bob are 1.0 apart → distance_close on both directions.
    # alice's yaw 0 → looks at bob (cos +1)
    # bob's yaw π → looks at alice (cos +1)
    # r1's yaw π/2 → looks at +y, alice is at +y, so gaze_at(r1, alice) fires.
    kinds_pairs = [(e.kind, e.src, e.dst) for e in events]
    assert ("distance_close", "alice", "bob") in kinds_pairs
    assert ("distance_close", "bob", "alice") in kinds_pairs
    assert ("gaze_at", "alice", "bob") in kinds_pairs
    assert ("gaze_at", "bob", "alice") in kinds_pairs
    assert ("gaze_at", "r1", "alice") in kinds_pairs


def test_thresholds_from_hymeko_file():
    """The HyMeKo file's observation_threshold blocks must surface
    in the Coalition.thresholds dict with the expected values."""
    from pathlib import Path
    from signedkan_wip.src.rapport.coalition import load_coalition
    repo_root = Path(__file__).resolve().parents[2]
    c = load_coalition(repo_root / "data" / "coalitions" / "triad_hri.hymeko")
    assert c.threshold("distance_close") == 1.5
    assert c.threshold("gaze_at") == 0.8
    assert c.threshold("withdrawal") == 0.3
    # gz_bindings populated for the three agents.
    assert set(c.gz_bindings.keys()) == {"alice", "bob", "r1"}
    assert c.gz_bindings["r1"].cmd_vel_topic == "/cmd_vel"
    assert c.gz_bindings["alice"].cmd_vel_topic is None
