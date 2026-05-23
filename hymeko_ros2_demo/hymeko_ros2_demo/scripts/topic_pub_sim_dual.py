"""topic_pub_sim_dual — synthetic upstream-perception publisher for
both robots in the Tier-2 dual scenario.

Publishes /robot_a/* and /robot_b/* on the topics that the per-robot
grasping_context_node binds against.  Two robots oscillate within a
shared workspace; their positions are *deliberately* designed to
periodically converge close enough to trip the scene-context safety
threshold (default 0.30 m).  This makes the closed-loop arbitration
demo interesting in <30 s without any external interaction.

Sequence (per 8 s cycle):
- Robots start at x=-0.6 and x=+0.6 m respectively (1.2 m apart).
- Their (y, z) coordinates oscillate at different periods, phasing
  in and out of "close" relative positions every ~4 s.
- One full cycle covers: well-separated → close-approach (safety
  fires) → both-grasping-same-payload (arbitration fires) → reset.
"""

from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.node import Node
from std_msgs.msg import UInt32


class TopicPubSimDual(Node):
    def __init__(self) -> None:
        super().__init__("topic_pub_sim_dual")
        # Robot A publishers
        self.pub_pose_a    = self.create_publisher(PoseStamped,    "/robot_a/tcp_pose",   10)
        self.pub_tool_a    = self.create_publisher(UInt32,         "/robot_a/tool_id",    10)
        self.pub_payload_a = self.create_publisher(UInt32,         "/robot_a/payload_id", 10)
        self.pub_mode_a    = self.create_publisher(UInt32,         "/robot_a/grasp_mode", 10)
        self.pub_wrench_a  = self.create_publisher(WrenchStamped,  "/robot_a/wrench",     10)
        # Robot B publishers
        self.pub_pose_b    = self.create_publisher(PoseStamped,    "/robot_b/tcp_pose",   10)
        self.pub_tool_b    = self.create_publisher(UInt32,         "/robot_b/tool_id",    10)
        self.pub_payload_b = self.create_publisher(UInt32,         "/robot_b/payload_id", 10)
        self.pub_mode_b    = self.create_publisher(UInt32,         "/robot_b/grasp_mode", 10)
        self.pub_wrench_b  = self.create_publisher(WrenchStamped,  "/robot_b/wrench",     10)

        self._timer = self.create_timer(0.1, self._tick)
        self._t0 = time.time()
        self.get_logger().info(
            "topic_pub_sim_dual armed; robot A at x=-0.6, robot B at x=+0.6"
        )

    def _tick(self) -> None:
        t = time.time() - self._t0

        # Robot A and B oscillate y/z at different freqs so the distance
        # cycles between ~0.2 m (safety violation) and ~1.4 m (clear).
        # Period: 8 s.
        stamp = self.get_clock().now().to_msg()

        # Pose for A
        pa = PoseStamped()
        pa.header.stamp = stamp
        pa.header.frame_id = "world"
        pa.pose.position.x = -0.6 + 0.3 * math.sin(2.0 * math.pi * 0.125 * t)
        pa.pose.position.y =  0.0 + 0.4 * math.sin(2.0 * math.pi * 0.08  * t)
        pa.pose.position.z =  0.3 + 0.2 * math.sin(2.0 * math.pi * 0.4   * t)
        pa.pose.orientation.w = 1.0
        self.pub_pose_a.publish(pa)

        # Pose for B (phase-shifted to cause periodic convergence)
        pb = PoseStamped()
        pb.header.stamp = stamp
        pb.header.frame_id = "world"
        pb.pose.position.x =  0.6 - 0.3 * math.sin(2.0 * math.pi * 0.125 * t)
        pb.pose.position.y =  0.0 - 0.4 * math.sin(2.0 * math.pi * 0.08  * t)
        pb.pose.position.z =  0.3 + 0.2 * math.cos(2.0 * math.pi * 0.4   * t)
        pb.pose.orientation.w = 1.0
        self.pub_pose_b.publish(pb)

        # Tool / payload / mode — both robots cycle through choices.
        # Periodically they target the same payload (triggers shared_payload=1).
        a_tool    = int((t // 2.0) % 3) + 1
        a_payload = int((t // 1.5) % 4) + 1
        a_mode    = int((t // 3.0) % 3)
        b_tool    = int((t // 1.7) % 3) + 1
        # Robot B's payload occasionally matches A (every ~6 s cycle)
        b_payload = a_payload if (int(t) % 6 < 3) else int((t // 2.0) % 4) + 1
        b_mode    = int((t // 2.5) % 3)

        self.pub_tool_a.publish(UInt32(data=a_tool))
        self.pub_payload_a.publish(UInt32(data=a_payload))
        self.pub_mode_a.publish(UInt32(data=a_mode))
        self.pub_tool_b.publish(UInt32(data=b_tool))
        self.pub_payload_b.publish(UInt32(data=b_payload))
        self.pub_mode_b.publish(UInt32(data=b_mode))

        # Wrench — different frequencies so A and B produce different S_g
        # values, exercising the arbitration argmax.
        wa = WrenchStamped()
        wa.header.stamp = stamp
        wa.header.frame_id = "tool0"
        wa.wrench.force.z = 5.0 + 3.5 * math.sin(2.0 * math.pi * 0.5  * t)
        self.pub_wrench_a.publish(wa)

        wb = WrenchStamped()
        wb.header.stamp = stamp
        wb.header.frame_id = "tool0"
        wb.wrench.force.z = 5.0 + 3.5 * math.cos(2.0 * math.pi * 0.35 * t)
        self.pub_wrench_b.publish(wb)


def main(args=None):
    rclpy.init(args=args)
    node = TopicPubSimDual()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
