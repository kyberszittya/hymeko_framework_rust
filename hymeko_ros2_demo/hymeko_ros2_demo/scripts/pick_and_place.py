"""pick_and_place — drive the UR5e through a canned motion loop.

Sends ``JointTrajectory`` waypoints to the
``/scaled_joint_trajectory_controller/follow_joint_trajectory``
action server.  Loops through a small set of poses so the demo
video shows the robot actually moving.

This is deliberately simpler than a MoveIt2-planned pick-and-place
(no IK, no scene awareness): for the Tier-1 reviewer demo we just
need the arm to visibly do something while the grasping_context_node
ticks against the synthetic upstream-perception inputs.  The
demo's purpose is to show the HyMeKo contextual flow integrated
with a moving robot, not to show a real grasp.

Usage::

    ros2 run hymeko_ros2_demo pick_and_place
"""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


JOINT_NAMES: List[str] = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Sequence of poses (rad).  Each row is a 6-tuple of joint angles.
# Designed for *visible* motion in Gazebo: full ~270° base sweeps,
# big shoulder dips, full wrist rolls.  No gripper concerns — the
# arm just demonstrates a dramatic sweep covering its workspace.
#
# Joint order:
#   shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
#
# UR5e effective limits (rad): ±π for shoulder_lift/elbow, ±2π elsewhere.
POSES: List[Tuple[float, float, float, float, float, float]] = [
    # 1. home — upright stowed
    (0.0,           -1.57,    0.0,    -1.57,    0.0,    0.0),
    # 2. big swing left + elbow extended — full reach
    (2.0,           -0.40,   -2.00,   -1.20,    1.57,   1.57),
    # 3. low dive — shoulder dropped, elbow folded high
    (2.0,           -2.50,    2.20,   -2.20,    3.14,   3.14),
    # 4. swing across to right (full ~225° base rotation)
    (-2.0,          -0.40,   -2.00,   -1.20,   -1.57,  -1.57),
    # 5. mirror dive on the right
    (-2.0,          -2.50,    2.20,   -2.20,   -3.14,  -3.14),
    # 6. upward stretch — arm vertical
    (0.0,           -3.10,    0.10,   -1.57,    0.0,    0.0),
    # 7. horizontal sweep forward
    (0.0,           -1.57,   -1.20,   -1.50,    1.57,   0.0),
    # 8. back to home
    (0.0,           -1.57,    0.0,    -1.57,    0.0,    0.0),
]

# Seconds to reach each waypoint (cumulative duration along the trajectory).
# Shorter = faster, more dramatic.  At 2.0 s a full 8-waypoint loop is 16 s.
WAYPOINT_DT_S = 2.0


class PickAndPlace(Node):
    def __init__(self) -> None:
        super().__init__("hymeko_pick_and_place")
        self.action_name = (
            "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )
        self._client = ActionClient(self, FollowJointTrajectory, self.action_name)
        self.get_logger().info(f"waiting for action server: {self.action_name}")
        self._client.wait_for_server()
        self.get_logger().info("action server ready")

    def run_once(self) -> bool:
        """Send the full POSES sequence as one trajectory + await result."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self._build_trajectory()
        self.get_logger().info(
            f"sending trajectory with {len(POSES)} waypoints "
            f"({len(POSES) * WAYPOINT_DT_S:.0f} s total)"
        )
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().error("goal rejected")
            return False
        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        result = res_future.result()
        ec = result.result.error_code if result else None
        if ec == 0:
            self.get_logger().info("trajectory complete (error_code=0)")
            return True
        self.get_logger().warn(
            f"trajectory finished with error_code={ec}"
        )
        return False

    def _build_trajectory(self) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        for i, pose in enumerate(POSES):
            point = JointTrajectoryPoint()
            point.positions = [float(p) for p in pose]
            t = (i + 1) * WAYPOINT_DT_S
            sec = int(t)
            nsec = int((t - sec) * 1e9)
            point.time_from_start = Duration(sec=sec, nanosec=nsec)
            traj.points.append(point)
        return traj


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlace()
    try:
        # Loop the motion so the demo keeps moving for the screen recording.
        while rclpy.ok():
            ok = node.run_once()
            if not ok:
                # On failure, back off briefly and retry rather than
                # spinning a tight loop.
                time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
