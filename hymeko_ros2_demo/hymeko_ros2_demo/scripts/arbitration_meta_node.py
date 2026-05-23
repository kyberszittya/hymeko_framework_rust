"""arbitration_meta_node — standalone variant of the §6 meta-context.

The scene_context_node already computes the arbitration commit signal
inline (e_s3).  This standalone node is provided for the case where you
want the meta-context as a *separate process* — closer to the paper's
§6 description that the arbitration meta-context is structurally
distinct from the scene context.

The two implementations are equivalent.  Use whichever fits the
deployment shape.

Subscribes:
- /scene/inter_robot_distance       (Float64)
- /robot_a/hymeko/grasping/stability_margin   (Float64)
- /robot_b/hymeko/grasping/stability_margin   (Float64)

Publishes:
- /scene/arbitration_commit         (UInt8, 0=none, 1=A, 2=B)

Usage::

    ros2 run hymeko_ros2_demo arbitration_meta_node
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, UInt8

# Import the pure-function policy so both nodes use the same rule.
from hymeko_ros2_demo.scripts.scene_context_node import arbitrate


class ArbitrationMetaNode(Node):
    def __init__(self) -> None:
        super().__init__("arbitration_meta_node")

        self.declare_parameter("safety_threshold", 0.30)
        self.declare_parameter("commit_min_margin", 0.50)
        self.declare_parameter("tick_rate_hz", 10.0)

        self.safety_threshold = float(self.get_parameter("safety_threshold").value)
        self.commit_min_margin = float(self.get_parameter("commit_min_margin").value)
        tick_rate = float(self.get_parameter("tick_rate_hz").value)

        self._distance: float = 0.0
        self._stab_a: float = 0.0
        self._stab_b: float = 0.0

        self.create_subscription(Float64, "/scene/inter_robot_distance",
                                  lambda m: setattr(self, "_distance", float(m.data)),
                                  10)
        self.create_subscription(Float64, "/robot_a/hymeko/grasping/stability_margin",
                                  lambda m: setattr(self, "_stab_a", float(m.data)),
                                  10)
        self.create_subscription(Float64, "/robot_b/hymeko/grasping/stability_margin",
                                  lambda m: setattr(self, "_stab_b", float(m.data)),
                                  10)
        self._pub = self.create_publisher(UInt8, "/scene/arbitration_commit", 10)

        self._timer = self.create_timer(1.0 / max(0.1, tick_rate), self._tick)
        self._tick_count = 0

        self.get_logger().info(
            f"arbitration_meta_node armed @ {tick_rate:.1f} Hz "
            f"(safety > {self.safety_threshold:.2f} m, "
            f"commit when max(S_g) > {self.commit_min_margin:.2f})"
        )

    def _tick(self) -> None:
        commit = arbitrate(
            distance=self._distance,
            shared=1.0,  # standalone variant: always treat as shared
            stab_a=self._stab_a,
            stab_b=self._stab_b,
            safety_threshold=self.safety_threshold,
            commit_min_margin=self.commit_min_margin,
        )
        self._pub.publish(UInt8(data=int(commit)))
        self._tick_count += 1
        if self._tick_count % 50 == 0:
            self.get_logger().info(
                f"tick {self._tick_count}  d={self._distance:.3f}m  "
                f"S_g(A)={self._stab_a:.2f}  S_g(B)={self._stab_b:.2f}  "
                f"commit={['none','A','B'][int(commit)]}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ArbitrationMetaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
