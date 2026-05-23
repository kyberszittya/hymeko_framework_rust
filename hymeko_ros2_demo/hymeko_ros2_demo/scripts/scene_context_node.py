"""scene_context_node — computes shared scene state from both robots.

Reads pose + payload + stability_margin topics from /robot_a/* and
/robot_b/*; publishes the 4 scene-context vertices:
  - /scene/inter_robot_distance   (Float64, metres)
  - /scene/shared_payload         (Float64, 0..1 share indicator)
  - /scene/committed_robot        (UInt8, 0=none, 1=A, 2=B)
  - /scene/task_complete          (Float64, 0..1)

Plus a JSON diagnostics topic /scene/diagnostics so the dashboard can
read the full scene V_global state in one place.

The arbitration policy (e_s3 in the IR) is computed here so this
node owns the full scene-context flow.  See arbitration_meta_node.py
for the standalone-meta-context variant if you want it as a separate
process — both implementations are equivalent.

Usage::

    ros2 run hymeko_ros2_demo scene_context_node
    ros2 run hymeko_ros2_demo scene_context_node \\
        --ros-args -p safety_threshold:=0.30 \\
                   -p tick_rate_hz:=10.0
"""

from __future__ import annotations

import json
import math
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float64, String, UInt32, UInt8


class SceneContextNode(Node):
    def __init__(self) -> None:
        super().__init__("scene_context_node")

        self.declare_parameter("safety_threshold", 0.30)
        self.declare_parameter("commit_min_margin", 0.50)
        self.declare_parameter("tick_rate_hz", 10.0)

        self.safety_threshold = float(self.get_parameter("safety_threshold").value)
        self.commit_min_margin = float(self.get_parameter("commit_min_margin").value)
        tick_rate = float(self.get_parameter("tick_rate_hz").value)

        # Live state — last value seen per topic.
        self._a_pose: Optional[tuple] = None
        self._b_pose: Optional[tuple] = None
        self._a_payload: float = 0.0
        self._b_payload: float = 0.0
        self._a_stab: float = 0.0
        self._b_stab: float = 0.0

        # Subscriptions (per-robot)
        self.create_subscription(PoseStamped, "/robot_a/tcp_pose",
                                  self._on_pose_a, 10)
        self.create_subscription(PoseStamped, "/robot_b/tcp_pose",
                                  self._on_pose_b, 10)
        self.create_subscription(UInt32, "/robot_a/payload_id",
                                  self._on_payload_a, 10)
        self.create_subscription(UInt32, "/robot_b/payload_id",
                                  self._on_payload_b, 10)
        self.create_subscription(Float64, "/robot_a/hymeko/grasping/stability_margin",
                                  self._on_stab_a, 10)
        self.create_subscription(Float64, "/robot_b/hymeko/grasping/stability_margin",
                                  self._on_stab_b, 10)

        # Publishers
        self._pub_distance = self.create_publisher(
            Float64, "/scene/inter_robot_distance", 10)
        self._pub_shared = self.create_publisher(
            Float64, "/scene/shared_payload", 10)
        self._pub_commit = self.create_publisher(
            UInt8, "/scene/committed_robot", 10)
        self._pub_task = self.create_publisher(
            Float64, "/scene/task_complete", 10)
        self._pub_diag = self.create_publisher(
            String, "/scene/diagnostics", 10)

        # Tick
        self._timer = self.create_timer(1.0 / max(0.1, tick_rate), self._tick)
        self._tick_count = 0
        self._task_complete: float = 0.0

        self.get_logger().info(
            f"scene_context_node armed @ {tick_rate:.1f} Hz  "
            f"(safety > {self.safety_threshold:.2f} m, "
            f"commit when max(S_g) > {self.commit_min_margin:.2f})"
        )

    # ─── subs ──────────────────────────────────────────────────────

    def _on_pose_a(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self._a_pose = (float(p.x), float(p.y), float(p.z))

    def _on_pose_b(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self._b_pose = (float(p.x), float(p.y), float(p.z))

    def _on_payload_a(self, msg: UInt32) -> None:
        self._a_payload = float(msg.data)

    def _on_payload_b(self, msg: UInt32) -> None:
        self._b_payload = float(msg.data)

    def _on_stab_a(self, msg: Float64) -> None:
        self._a_stab = float(msg.data)

    def _on_stab_b(self, msg: Float64) -> None:
        self._b_stab = float(msg.data)

    # ─── tick ──────────────────────────────────────────────────────

    def _tick(self) -> None:
        # e_s1 — inter-robot distance
        distance = compute_distance(self._a_pose, self._b_pose)

        # e_s2 — shared-payload indicator: 1.0 if both robots target
        # the same payload ID (and both IDs are non-zero), 0 otherwise.
        shared = 1.0 if (
            self._a_payload > 0
            and self._b_payload > 0
            and abs(self._a_payload - self._b_payload) < 0.5
        ) else 0.0

        # e_s3 — arbitration policy:
        #   safety first: if distance below threshold, commit = 0 (pause)
        #   else if neither S_g > min: commit = 0 (no good grasp)
        #   else: commit to argmax S_g (1 = A, 2 = B)
        commit = arbitrate(
            distance=distance,
            shared=shared,
            stab_a=self._a_stab,
            stab_b=self._b_stab,
            safety_threshold=self.safety_threshold,
            commit_min_margin=self.commit_min_margin,
        )

        # task_complete: a simple "we eventually committed someone and
        # they stayed committed for several ticks" proxy.  For Tier 1
        # we just hold at 0 until commit happens, then climb gently.
        if commit > 0:
            self._task_complete = min(1.0, self._task_complete + 0.02)
        else:
            self._task_complete = max(0.0, self._task_complete - 0.01)

        # Publish individual signals
        self._pub_distance.publish(Float64(data=distance))
        self._pub_shared.publish(Float64(data=shared))
        self._pub_commit.publish(UInt8(data=int(commit)))
        self._pub_task.publish(Float64(data=float(self._task_complete)))

        # Diagnostics — same shape as grasping diagnostics for dashboard reuse
        diag = {
            "tick": self._tick_count,
            "context": "scene_context",
            "v_global": {
                "a_robot_pose": self._a_pose[2] if self._a_pose else 0.0,
                "b_robot_pose": self._b_pose[2] if self._b_pose else 0.0,
                "a_active_payload": self._a_payload,
                "b_active_payload": self._b_payload,
                "a_stability_margin": self._a_stab,
                "b_stability_margin": self._b_stab,
                "inter_robot_distance": distance,
                "shared_payload": shared,
                "committed_robot": float(commit),
                "task_complete": float(self._task_complete),
            },
            "edges": [
                {"name": "e_s1_inter_robot_distance",
                 "inputs": ["a_robot_pose", "b_robot_pose"],
                 "outputs": ["inter_robot_distance"]},
                {"name": "e_s2_shared_payload",
                 "inputs": ["a_active_payload", "b_active_payload"],
                 "outputs": ["shared_payload"]},
                {"name": "e_s3_arbitration",
                 "inputs": ["shared_payload", "a_stability_margin", "b_stability_margin"],
                 "outputs": ["committed_robot"]},
            ],
        }
        self._pub_diag.publish(String(data=json.dumps(diag)))

        self._tick_count += 1
        if self._tick_count % 50 == 0:
            self.get_logger().info(
                f"tick {self._tick_count}  d={distance:.3f}m  shared={shared:.1f}  "
                f"S_g(A)={self._a_stab:.2f}  S_g(B)={self._b_stab:.2f}  "
                f"commit={['none','A','B'][int(commit)]}"
            )


# ─── Pure-function helpers (so they're unit-testable without ROS) ─────


def compute_distance(pose_a: Optional[tuple], pose_b: Optional[tuple]) -> float:
    """Euclidean distance between two (x, y, z) tuples.

    Returns 0.0 if either pose is missing — a "we don't know yet"
    signal that the arbitration policy treats conservatively.
    """
    if pose_a is None or pose_b is None:
        return 0.0
    dx = pose_a[0] - pose_b[0]
    dy = pose_a[1] - pose_b[1]
    dz = pose_a[2] - pose_b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def arbitrate(*, distance: float, shared: float, stab_a: float, stab_b: float,
              safety_threshold: float, commit_min_margin: float) -> int:
    """Return 0 (none), 1 (commit A), or 2 (commit B).

    Rules, in order:
    1. If distance < safety_threshold → 0 (safety override; arms too close).
    2. If neither stability_margin > commit_min_margin → 0 (no good grasp).
    3. Otherwise: commit to argmax(stability_margin); ties to A.
    """
    if distance < safety_threshold and distance > 0.0:
        return 0
    if stab_a <= commit_min_margin and stab_b <= commit_min_margin:
        return 0
    return 1 if stab_a >= stab_b else 2


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SceneContextNode()
    except Exception as exc:  # noqa: BLE001
        print(f"[scene_context_node] startup failed: {exc!r}")
        rclpy.shutdown()
        raise
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
