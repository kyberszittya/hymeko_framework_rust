"""pick_and_place_moveit — drive the UR5e via MoveIt2's MoveGroup action.

This is the *closed-loop* variant of ``pick_and_place.py``.  Instead of
sending raw ``JointTrajectory`` goals to the controller, this node
calls the MoveIt2 ``/move_action`` action with joint-space goals.  MoveIt
then:

1. Plans a collision-free, smoothed trajectory from the current state to
   each joint target,
2. Executes that trajectory through the same
   ``scaled_joint_trajectory_controller`` that the direct path uses.

The *closed-loop demo bit*: between each waypoint, the node waits until
the live ``/hymeko/grasping/stability_margin`` signal (the article's
``S_g``) crosses the configurable ``--stability-gate`` threshold.  This
makes the planning layer visibly *consume* the contextual flow's
output — which is the architectural story the MDPI Technologies article
implies but does not implement.

Usage::

    ros2 run hymeko_ros2_demo pick_and_place_moveit
    ros2 run hymeko_ros2_demo pick_and_place_moveit \\
        --ros-args -p stability_gate:=0.10 -p use_sim_time:=true

Requires the MoveIt2 ``move_group`` node to be running (started by
``ur_sim_moveit.launch.py`` automatically).
"""

from __future__ import annotations

import math
import time
from typing import Callable, List, Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float64

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)


JOINT_NAMES: List[str] = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Reuse the dramatic 8-pose sequence from the direct path so the two
# motion backends produce a visually equivalent demo.
POSES: List[Tuple[float, float, float, float, float, float]] = [
    (0.0,    -1.57,    0.0,   -1.57,    0.0,    0.0),
    (2.0,    -0.40,   -2.00,  -1.20,    1.57,   1.57),
    (2.0,    -2.50,    2.20,  -2.20,    3.14,   3.14),
    (-2.0,   -0.40,   -2.00,  -1.20,   -1.57,  -1.57),
    (-2.0,   -2.50,    2.20,  -2.20,   -3.14,  -3.14),
    (0.0,    -3.10,    0.10,  -1.57,    0.0,    0.0),
    (0.0,    -1.57,   -1.20,  -1.50,    1.57,   0.0),
    (0.0,    -1.57,    0.0,   -1.57,    0.0,    0.0),
]

MOVE_ACTION = "/move_action"
PLANNING_GROUP = "ur_manipulator"


class PickAndPlaceMoveIt(Node):
    def __init__(self) -> None:
        super().__init__("hymeko_pick_and_place_moveit")

        # Parameters
        self.declare_parameter("planning_time_s", 5.0)
        self.declare_parameter("vel_scale", 0.5)
        self.declare_parameter("acc_scale", 0.5)
        self.declare_parameter("stability_gate", 0.05)
        self.declare_parameter("gate_timeout_s", 12.0)
        self.declare_parameter("loop", True)

        self.planning_time = float(self.get_parameter("planning_time_s").value or 5.0)
        self.vel_scale = float(self.get_parameter("vel_scale").value or 0.5)
        self.acc_scale = float(self.get_parameter("acc_scale").value or 0.5)
        self.stability_gate = float(self.get_parameter("stability_gate").value or 0.0)
        self.gate_timeout = float(self.get_parameter("gate_timeout_s").value or 12.0)
        self.loop = bool(self.get_parameter("loop").value)

        # Action client
        self._client = ActionClient(self, MoveGroup, MOVE_ACTION)
        self.get_logger().info(f"waiting for MoveIt action server: {MOVE_ACTION}")
        if not self._client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError(
                f"MoveIt action server '{MOVE_ACTION}' not available after 30s. "
                "Ensure ur_sim_moveit.launch.py is running (or pass "
                "motion_backend:=direct to skip MoveIt)."
            )
        self.get_logger().info("MoveIt action server ready")

        # Closed-loop input: live stability margin (the article's S_g).
        self._last_stability: float = 0.0
        self.create_subscription(
            Float64, "/hymeko/grasping/stability_margin",
            lambda msg: setattr(self, "_last_stability", float(msg.data)),
            10,
        )

    # ─── motion ────────────────────────────────────────────────────

    def wait_for_gate(self) -> bool:
        """Wait until ``stability_margin`` crosses the gate, or timeout."""
        if self.stability_gate <= 0:
            return True
        deadline = time.time() + self.gate_timeout
        self.get_logger().info(
            f"  gate: waiting for stability_margin > "
            f"{self.stability_gate:.3f} (current={self._last_stability:.3f})"
        )
        while rclpy.ok() and self._last_stability <= self.stability_gate:
            if time.time() > deadline:
                self.get_logger().warn(
                    f"  gate: timeout after {self.gate_timeout:.0f}s — "
                    f"proceeding with stability_margin={self._last_stability:.3f}"
                )
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(
            f"  gate: passed (stability_margin={self._last_stability:.3f})"
        )
        return True

    def plan_and_execute(self, joint_target: Tuple[float, ...]) -> bool:
        """Send a single MoveGroup goal with joint constraints."""
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.num_planning_attempts = 5
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.vel_scale
        req.max_acceleration_scaling_factor = self.acc_scale

        # One Constraints set with 6 JointConstraints (one per joint).
        c = Constraints()
        for j, pos in zip(JOINT_NAMES, joint_target):
            jc = JointConstraint()
            jc.joint_name = j
            jc.position = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)

        opts = PlanningOptions()
        opts.plan_only = False
        opts.look_around = False
        opts.replan = True
        opts.replan_attempts = 3
        opts.planning_scene_diff.is_diff = True
        opts.planning_scene_diff.robot_state.is_diff = True

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = opts

        target_str = ", ".join(f"{p:+.2f}" for p in joint_target)
        self.get_logger().info(f"  plan_and_execute → [{target_str}]")
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().error("    goal rejected")
            return False
        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_future, timeout_sec=30.0)
        result = res_future.result()
        if result is None:
            self.get_logger().warn("    no result returned within 30s")
            return False
        ec = result.result.error_code.val
        if ec == 1:  # SUCCESS
            ptime = float(result.result.planning_time)
            self.get_logger().info(f"    ok  (planning_time={ptime:.2f}s)")
            return True
        self.get_logger().warn(f"    MoveIt error_code={ec}")
        return False

    def run_loop(self) -> None:
        cycle = 0
        while rclpy.ok():
            cycle += 1
            self.get_logger().info(f"=== cycle {cycle} ===")
            for i, pose in enumerate(POSES):
                if not rclpy.ok():
                    break
                self.get_logger().info(f"waypoint {i + 1}/{len(POSES)}")
                # Optional closed-loop gate from /hymeko/grasping/stability_margin.
                self.wait_for_gate()
                ok = self.plan_and_execute(pose)
                if not ok:
                    self.get_logger().warn(
                        f"  waypoint {i + 1} failed; backing off 2s"
                    )
                    time.sleep(2.0)
            if not self.loop:
                self.get_logger().info("loop=false → exiting after one cycle")
                break


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PickAndPlaceMoveIt()
    except Exception as exc:  # noqa: BLE001 — fail loudly at startup
        print(f"[pick_and_place_moveit] startup failed: {exc!r}")
        rclpy.shutdown()
        raise
    try:
        node.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
