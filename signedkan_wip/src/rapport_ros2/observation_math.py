"""Pure-Python pose → observation-event computation.

Stage G's GzObserver node wraps this module with rclpy plumbing.
Keeping the math here lets us unit-test the derivation without
spinning up a ROS 2 process or a GZ simulator.

Conventions:

* A pose is a tuple ``(x, y, yaw)`` in the world frame. We collapse
  to 2D because the rapport observations only care about ground-
  plane geometry; the z-coordinate is ignored.

* Observation events have the shape ``(kind, src, dst)``: a
  directional channel from agent ``src`` to agent ``dst``.

Thresholds are passed in from the HyMeKo-declared
``observation_threshold`` blocks (see
``data/coalitions/triad_hri.hymeko``); the module itself has no
hard-coded constants.

Plan: docs/plans/2026-05-18-gz-rapport-demo/.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    """A 2D world-frame pose: position (x, y) + heading yaw."""
    x: float
    y: float
    yaw: float
    stamp_s: float = 0.0


@dataclass(frozen=True)
class ObservationEvent:
    """One-shot derived observation."""
    kind: str
    src: str
    dst: str
    stamp_s: float = 0.0


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Standard ROS quaternion → yaw (Z-axis rotation, radians).

    The ZYX-yaw extraction with edge-case safety: handles the
    canonical `geometry_msgs/msg/Quaternion` ordering.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def distance_2d(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def heading_cosine_to_target(a: Pose2D, b: Pose2D) -> float:
    """Cosine of the angle between a's forward direction and (b-a).

    Returns 1.0 if a is looking directly at b, -1.0 if directly away.
    Returns 0.0 if a and b are coincident (no defined direction).
    """
    dx = b.x - a.x
    dy = b.y - a.y
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return 0.0
    heading_x = math.cos(a.yaw)
    heading_y = math.sin(a.yaw)
    return heading_x * (dx / dist) + heading_y * (dy / dist)


def derive_pairwise_observations(
    src_name: str, src_pose: Pose2D,
    dst_name: str, dst_pose: Pose2D,
    *,
    distance_close: float,
    gaze_cosine: float,
) -> list[ObservationEvent]:
    """Emit observation events for a single directional (src, dst) pair.

    Symmetric channels (distance_close) are emitted on each call; the
    caller decides whether to de-duplicate.

    Args:
        src_name, src_pose: name + 2D pose of the source agent.
        dst_name, dst_pose: name + 2D pose of the destination agent.
        distance_close: threshold for the distance_close channel
            (in metres).
        gaze_cosine: threshold for the gaze_at channel (cosine of the
            angle between source's heading and the direction to dst).

    Returns:
        list of ObservationEvent (possibly empty).
    """
    events: list[ObservationEvent] = []
    dist = distance_2d(src_pose, dst_pose)
    stamp = max(src_pose.stamp_s, dst_pose.stamp_s)
    if dist < distance_close:
        events.append(ObservationEvent(
            kind="distance_close", src=src_name, dst=dst_name,
            stamp_s=stamp,
        ))
    cos_angle = heading_cosine_to_target(src_pose, dst_pose)
    if cos_angle > gaze_cosine:
        events.append(ObservationEvent(
            kind="gaze_at", src=src_name, dst=dst_name,
            stamp_s=stamp,
        ))
    return events


def derive_withdrawal(
    src_name: str, src_now: Pose2D, src_past: Pose2D,
    dst_name: str, dst_now: Pose2D, dst_past: Pose2D,
    *, rate_threshold: float,
) -> list[ObservationEvent]:
    """Detect that the pair's distance is increasing fast enough
    to count as `withdrawal`.

    Args:
        src/dst_now/past: pose snapshots at the latest frame and at
            an earlier reference frame.
        rate_threshold: distance-increase rate in m/s above which the
            event fires.
    """
    dt = src_now.stamp_s - src_past.stamp_s
    if dt < 1e-3:
        return []
    d_now = distance_2d(src_now, dst_now)
    d_past = distance_2d(src_past, dst_past)
    rate = (d_now - d_past) / dt
    if rate > rate_threshold:
        return [ObservationEvent(
            kind="withdrawal", src=src_name, dst=dst_name,
            stamp_s=src_now.stamp_s,
        )]
    return []


def derive_all_observations(
    poses_now: dict[str, Pose2D],
    poses_past: dict[str, Pose2D] | None,
    *,
    distance_close: float,
    gaze_cosine: float,
    withdrawal_rate: float,
) -> list[ObservationEvent]:
    """Compute observations across every ordered pair of agents.

    Args:
        poses_now: {agent_name: Pose2D} latest frame.
        poses_past: {agent_name: Pose2D} reference frame for
            withdrawal detection. If None, withdrawal events are
            skipped.
        distance_close / gaze_cosine / withdrawal_rate: thresholds
            (from HyMeKo).
    """
    names = sorted(poses_now.keys())
    out: list[ObservationEvent] = []
    for s in names:
        for d in names:
            if s == d:
                continue
            out.extend(derive_pairwise_observations(
                s, poses_now[s], d, poses_now[d],
                distance_close=distance_close,
                gaze_cosine=gaze_cosine,
            ))
            if poses_past is not None and s in poses_past and d in poses_past:
                out.extend(derive_withdrawal(
                    s, poses_now[s], poses_past[s],
                    d, poses_now[d], poses_past[d],
                    rate_threshold=withdrawal_rate,
                ))
    return out
