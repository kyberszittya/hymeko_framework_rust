"""GzRobotControllerNode — translate policy action symbols to robot motion.

Subscribes:
  /rapport/policy_action  (std_msgs/String)    — symbolic action name
  /model/{alice,bob}/pose (geometry_msgs/Pose) — to plan mediation target
  /model/r1/pose          (geometry_msgs/Pose) — robot's own pose for control

Publishes:
  /cmd_vel                 (geometry_msgs/Twist) — r1 differential drive

Symbolic actions, per ``triad_hri.hymeko``'s policy blocks:
  * ``signal_alignment`` — turn head toward the human whose dyadic edge
    is most imbalanced (v1: just turn r1 in place toward alice).
  * ``mediation_offer`` — navigate to the midpoint between alice and bob.
  * ``withdraw`` — back away from the alice-bob centroid.

Each action triggers a short ``_action_duration_s`` motion command;
overlapping actions are queued, with the most recent action taking
priority (rapport repair shouldn't be over-suppressed by stale
intents).

Plan: docs/plans/2026-05-18-gz-rapport-demo/.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


def _import_ros() -> tuple[Any, ...]:
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Pose, Twist
        from std_msgs.msg import String
        return rclpy, Node, Pose, Twist, String
    except ImportError as e:
        raise RuntimeError(
            "rclpy / geometry_msgs / std_msgs not importable. "
            "Source ROS 2 Kilted + activate .venv-rapport-ros2. "
            f"(error: {e})"
        ) from e


def make_controller_node(coalition_path: str | Path,
                         action_duration_s: float = 2.0,
                         linear_speed: float = 0.3,
                         angular_speed: float = 0.8) -> Any:
    """Construct the rclpy.Node controlling r1 in response to policy actions."""
    rclpy, Node, Pose, Twist, String = _import_ros()
    from ..rapport.coalition import load_coalition
    from .observation_math import Pose2D, quaternion_to_yaw

    coalition = load_coalition(Path(coalition_path))
    if "r1" not in coalition.gz_bindings:
        raise ValueError(
            f"coalition {coalition.name!r} has no gz_binding for r1 — "
            f"controller cannot drive any robot"
        )
    r1_binding = coalition.gz_bindings["r1"]
    if r1_binding.cmd_vel_topic is None:
        raise ValueError(
            f"r1 gz_binding has no cmd_vel_topic; cannot drive"
        )

    class GzRobotControllerNode(Node):
        def __init__(self) -> None:
            super().__init__("gz_robot_controller")
            self._poses: dict[str, Pose2D] = {}
            for agent_name, binding in coalition.gz_bindings.items():
                self.create_subscription(
                    Pose, binding.pose_topic,
                    self._make_pose_cb(agent_name), 10,
                )
            self.create_subscription(
                String, "/rapport/policy_action",
                self._on_action, 10,
            )
            self._cmd_pub = self.create_publisher(
                Twist, r1_binding.cmd_vel_topic, 10,
            )
            # 10 Hz control loop.
            self._timer = self.create_timer(0.1, self._control_step)
            # Action state machine.
            self._active_action: str | None = None
            self._action_start_t: float | None = None
            self.get_logger().info(
                f"controller publishing /cmd_vel for r1 (topic="
                f"{r1_binding.cmd_vel_topic})"
            )

        def _make_pose_cb(self, name: str):
            def _cb(msg) -> None:
                yaw = quaternion_to_yaw(
                    msg.orientation.x, msg.orientation.y,
                    msg.orientation.z, msg.orientation.w,
                )
                t = self.get_clock().now().nanoseconds / 1e9
                self._poses[name] = Pose2D(
                    x=msg.position.x, y=msg.position.y,
                    yaw=yaw, stamp_s=t,
                )
            return _cb

        def _on_action(self, msg) -> None:
            self._active_action = msg.data
            self._action_start_t = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(f"action received: {msg.data}")

        def _control_step(self) -> None:
            now_t = self.get_clock().now().nanoseconds / 1e9
            cmd = Twist()
            if (self._active_action is None
                    or self._action_start_t is None
                    or now_t - self._action_start_t > action_duration_s
                    or "r1" not in self._poses):
                # Idle: zero velocity.
                self._active_action = None
                self._cmd_pub.publish(cmd)
                return
            r1 = self._poses["r1"]
            alice = self._poses.get("alice")
            bob = self._poses.get("bob")
            target: tuple[float, float] | None = None
            if self._active_action == "signal_alignment":
                # Turn head/body to face alice if she exists.
                if alice is not None:
                    target = (alice.x, alice.y)
            elif self._active_action == "mediation_offer":
                # Navigate to midpoint of alice and bob.
                if alice is not None and bob is not None:
                    target = ((alice.x + bob.x) / 2.0,
                              (alice.y + bob.y) / 2.0)
            elif self._active_action == "withdraw":
                # Move away from alice-bob centroid.
                if alice is not None and bob is not None:
                    cx, cy = (alice.x + bob.x) / 2.0, (alice.y + bob.y) / 2.0
                    dx, dy = r1.x - cx, r1.y - cy
                    dist = math.hypot(dx, dy)
                    if dist > 1e-6:
                        target = (r1.x + dx / dist * 2.0,
                                  r1.y + dy / dist * 2.0)

            if target is None:
                self._cmd_pub.publish(cmd)
                return

            # Simple proportional controller: align heading first, then
            # drive forward if heading error is small.
            tx, ty = target
            dx, dy = tx - r1.x, ty - r1.y
            dist = math.hypot(dx, dy)
            desired_yaw = math.atan2(dy, dx)
            heading_err = math.atan2(
                math.sin(desired_yaw - r1.yaw),
                math.cos(desired_yaw - r1.yaw),
            )
            cmd.angular.z = max(-angular_speed,
                                  min(angular_speed, 1.5 * heading_err))
            # Drive forward (or backward for withdraw) only when roughly
            # facing the target.
            if abs(heading_err) < 0.35 and dist > 0.05:
                cmd.linear.x = min(linear_speed, dist * 0.5)
            self._cmd_pub.publish(cmd)

    return GzRobotControllerNode()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coalition",
        default="data/coalitions/triad_hri.hymeko",
    )
    args = ap.parse_args(argv)
    rclpy, *_ = _import_ros()
    rclpy.init()
    node = make_controller_node(args.coalition)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
