"""GzObserver — ROS 2 node bridging GZ pose topics to rapport Observations.

Subscribes to each coalition agent's pose topic (declared in the
HyMeKo coalition file's `gz_binding` blocks), maintains a recent-pose
buffer per agent, and emits derived observation events at a fixed
publish rate to ``/rapport/observations`` (std_msgs/String, JSON-
encoded).

The math is in
``signedkan_wip.src.rapport_ros2.observation_math``; this node is
the ROS 2 plumbing. Tests for the math live in
``test_rapport_observation_math.py`` and can run without a ROS 2
process; the node itself is exercised by the Day 4 integration
launch test.

Run (single command, after sourcing ROS 2 Kilted + the
.venv-rapport-ros2 venv):
    python -m signedkan_wip.src.rapport_ros2.gz_observer_node \\
        --coalition data/coalitions/triad_hri.hymeko

Plan: docs/plans/2026-05-18-gz-rapport-demo/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

# rclpy + geometry_msgs are loaded lazily so this file is import-
# safe outside a ROS 2 environment (the math module's tests don't
# need them).
def _import_ros() -> tuple[Any, Any, Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Pose
        from std_msgs.msg import String
        return rclpy, Node, Pose, String
    except ImportError as e:
        raise RuntimeError(
            "rclpy / geometry_msgs not importable. Source ROS 2 Kilted "
            "and the .venv-rapport-ros2 venv before running this node. "
            f"(error: {e})"
        ) from e


def make_observer_node(coalition_path: str | Path,
                       publish_rate_hz: float = 5.0,
                       history_seconds: float = 1.0) -> Any:
    """Construct an `rclpy.Node` instance that subscribes to the
    coalition's pose topics and emits observation events."""
    rclpy, Node, Pose, String = _import_ros()
    from .observation_math import (
        Pose2D, derive_all_observations, quaternion_to_yaw,
    )
    from ..rapport.coalition import load_coalition

    coalition = load_coalition(Path(coalition_path))
    if not coalition.gz_bindings:
        raise ValueError(
            f"coalition {coalition.name!r} has no gz_bindings; "
            f"cannot observe physical poses"
        )
    distance_close = coalition.threshold("distance_close", default=1.5)
    gaze_cosine = coalition.threshold("gaze_at", default=0.8)
    withdrawal_rate = coalition.threshold("withdrawal", default=0.3)

    class GzObserverNode(Node):
        def __init__(self) -> None:
            super().__init__("gz_observer")
            self._poses_now: dict[str, Pose2D] = {}
            # History buffer: agent → deque[Pose2D] up to history_seconds.
            self._history: dict[str, deque[Pose2D]] = {
                name: deque(maxlen=int(publish_rate_hz * history_seconds + 4))
                for name in coalition.agents
            }
            # Subscriptions, one per agent.
            for agent_name, binding in coalition.gz_bindings.items():
                self.create_subscription(
                    Pose, binding.pose_topic,
                    self._make_pose_cb(agent_name), 10,
                )
                self.get_logger().info(
                    f"subscribed: {binding.pose_topic} → agent {agent_name}"
                )
            # Multiplex affect channel (scripted-conflict injection).
            self._affect_sub = self.create_subscription(
                String, "/rapport/scripted_affect",
                self._on_affect, 10,
            )
            self._obs_pub = self.create_publisher(
                String, "/rapport/observations", 10,
            )
            self._timer = self.create_timer(
                1.0 / publish_rate_hz, self._publish_observations,
            )
            self.get_logger().info(
                f"thresholds: distance_close={distance_close} "
                f"gaze_at={gaze_cosine} withdrawal={withdrawal_rate}"
            )

        def _make_pose_cb(self, name: str):
            def _cb(msg) -> None:
                yaw = quaternion_to_yaw(
                    msg.orientation.x, msg.orientation.y,
                    msg.orientation.z, msg.orientation.w,
                )
                t = self.get_clock().now().nanoseconds / 1e9
                p = Pose2D(x=msg.position.x, y=msg.position.y,
                            yaw=yaw, stamp_s=t)
                self._poses_now[name] = p
                self._history[name].append(p)
            return _cb

        def _on_affect(self, msg) -> None:
            # Affect events are pass-through: scripted-conflict events
            # arrive as JSON on /rapport/scripted_affect and we
            # re-emit them on /rapport/observations as-is.
            self._obs_pub.publish(msg)

        def _publish_observations(self) -> None:
            if len(self._poses_now) < len(coalition.agents):
                return
            poses_past = {
                name: dq[0]
                for name, dq in self._history.items()
                if len(dq) >= 2
            } or None
            events = derive_all_observations(
                self._poses_now, poses_past,
                distance_close=distance_close,
                gaze_cosine=gaze_cosine,
                withdrawal_rate=withdrawal_rate,
            )
            for ev in events:
                msg = String()
                msg.data = json.dumps({
                    "kind": ev.kind,
                    "src": ev.src,
                    "dst": ev.dst,
                    "stamp_s": ev.stamp_s,
                })
                self._obs_pub.publish(msg)

    return GzObserverNode()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coalition",
        default="data/coalitions/triad_hri.hymeko",
        help="HyMeKo coalition spec (declares agents + gz topics + thresholds).",
    )
    ap.add_argument(
        "--publish-rate-hz", type=float, default=5.0,
        help="Observation emission rate.",
    )
    args = ap.parse_args(argv)

    rclpy, _, _, _ = _import_ros()
    rclpy.init()
    node = make_observer_node(args.coalition, args.publish_rate_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
