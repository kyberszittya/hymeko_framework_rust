"""topic_pub_sim — synthetic input publisher for node-only smoke runs.

Publishes plausible values on the 5 input topics of the grasping
context at 10 Hz so the ``grasping_context_node`` has data to process
without needing the UR + Gazebo stack.  Used by the
``grasping_context_only.launch.py`` smoke launch and by the unit tests.
"""

from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.node import Node
from std_msgs.msg import UInt32


class TopicPubSim(Node):
    def __init__(self) -> None:
        super().__init__("topic_pub_sim")
        self.pub_pose = self.create_publisher(PoseStamped, "/tcp_pose", 10)
        self.pub_tool = self.create_publisher(UInt32, "/tool_id", 10)
        self.pub_payload = self.create_publisher(UInt32, "/payload_id", 10)
        self.pub_mode = self.create_publisher(UInt32, "/grasp_mode", 10)
        self.pub_wrench = self.create_publisher(WrenchStamped, "/wrench", 10)
        self._timer = self.create_timer(0.1, self._tick)
        self._t0 = time.time()

    def _tick(self) -> None:
        t = time.time() - self._t0
        # Pose: a faster-oscillating Z (height).  Was 0.1 Hz (10 s period).
        # Now 0.4 Hz (2.5 s period) so the contextual flow shows obvious
        # motion in a short demo window.
        p = PoseStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = "base_link"
        p.pose.position.z = 0.3 + 0.2 * math.sin(2.0 * math.pi * 0.4 * t)
        p.pose.orientation.w = 1.0
        self.pub_pose.publish(p)

        # IDs (rotated faster — 2 s, 1.5 s, 3 s periods).
        self.pub_tool.publish(UInt32(data=int((t // 2.0) % 3) + 1))
        self.pub_payload.publish(UInt32(data=int((t // 1.5) % 4) + 1))
        self.pub_mode.publish(UInt32(data=int((t // 3.0) % 3)))

        # Wrench: 0.5 Hz grip-force oscillation (2 s period).  Range
        # 1.5 – 8.5 N.  Combined with the faster pose stream above, the
        # downstream stability_margin gauge cycles its full 0.3–1.0 range
        # every ~2 s.
        w = WrenchStamped()
        w.header.stamp = p.header.stamp
        w.header.frame_id = "tool0"
        w.wrench.force.z = 5.0 + 3.5 * math.sin(2.0 * math.pi * 0.5 * t)
        self.pub_wrench.publish(w)


def main(args=None):
    rclpy.init(args=args)
    node = TopicPubSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
