"""RapportPipelineNode — runs the σ-cycle balance pipeline as a ROS 2 node.

Subscribes:
  /rapport/observations      (std_msgs/String JSON: {kind, src, dst, stamp_s})

Publishes:
  /rapport/sigma             (std_msgs/Float64MultiArray)
      Layout: [t_frame, sigma_for_cycle_0, sigma_for_cycle_1, ...].
      Cycle names available on /rapport/cycle_names (latched String).
  /rapport/weights           (std_msgs/Float64MultiArray)
      Layout: [t_frame, w_relation_0, w_relation_1, ...].
      Relation names available on /rapport/relation_names (latched String).
  /rapport/policy_action     (std_msgs/String)
      Emitted whenever the policy engine fires an action.
  /rapport/cycle_names       (std_msgs/String, latched JSON list)
  /rapport/relation_names    (std_msgs/String, latched JSON list)

The CoalitionEstimator, sigma_cycle, and PolicyEngine modules are
the same code as the Tk demo (already 21/21 tests passing). This
node is the ROS 2 plumbing; no math is duplicated.

Plan: docs/plans/2026-05-18-gz-rapport-demo/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _import_ros() -> tuple[Any, ...]:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
        )
        from std_msgs.msg import Float64MultiArray, String
        return rclpy, Node, QoSProfile, QoSDurabilityPolicy, \
               QoSReliabilityPolicy, Float64MultiArray, String
    except ImportError as e:
        raise RuntimeError(
            "rclpy / std_msgs not importable. Source ROS 2 Kilted "
            "and the .venv-rapport-ros2 venv before running this node. "
            f"(error: {e})"
        ) from e


def make_pipeline_node(coalition_path: str | Path) -> Any:
    """Construct an rclpy.Node that runs the σ-cycle balance pipeline."""
    rclpy, Node, QoSProfile, QoSDurabilityPolicy, \
        QoSReliabilityPolicy, Float64MultiArray, String = _import_ros()
    from ..rapport.coalition import load_coalition
    from ..rapport.coherence import sigma_cycle
    from ..rapport.estimator import CoalitionEstimator, Observation
    from ..rapport.policy import PolicyEngine

    coalition = load_coalition(Path(coalition_path))
    if not coalition.cycles:
        raise ValueError(f"coalition {coalition.name!r} has no cycles")

    latched_qos = QoSProfile(
        depth=1,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        reliability=QoSReliabilityPolicy.RELIABLE,
    )

    class RapportPipelineNode(Node):
        def __init__(self) -> None:
            super().__init__("rapport_pipeline")
            self._estimator = CoalitionEstimator(coalition, alpha=0.2)
            self._policy = PolicyEngine(coalition, cooldown_frames=15)
            self._frame = 0

            # Subscribers
            self.create_subscription(
                String, "/rapport/observations",
                self._on_observation, 50,
            )

            # Publishers
            self._sigma_pub = self.create_publisher(
                Float64MultiArray, "/rapport/sigma", 10,
            )
            self._weights_pub = self.create_publisher(
                Float64MultiArray, "/rapport/weights", 10,
            )
            self._action_pub = self.create_publisher(
                String, "/rapport/policy_action", 10,
            )
            self._cycle_names_pub = self.create_publisher(
                String, "/rapport/cycle_names", latched_qos,
            )
            self._relation_names_pub = self.create_publisher(
                String, "/rapport/relation_names", latched_qos,
            )

            # Latched metadata (the layout of the Float64MultiArray
            # publications).
            self._cycle_names = list(coalition.cycles.keys())
            self._relation_names = list(coalition.relations.keys())
            cn = String(); cn.data = json.dumps(self._cycle_names)
            rn = String(); rn.data = json.dumps(self._relation_names)
            self._cycle_names_pub.publish(cn)
            self._relation_names_pub.publish(rn)

            self._pending_observations: list[Observation] = []
            # 5 Hz tick → consume the buffer, step the pipeline.
            self._timer = self.create_timer(0.20, self._tick)
            self.get_logger().info(
                f"pipeline ready: {len(coalition.relations)} relations, "
                f"{len(coalition.cycles)} cycles, "
                f"{len(coalition.policies)} policies"
            )

        def _on_observation(self, msg) -> None:
            try:
                d = json.loads(msg.data)
                self._pending_observations.append(Observation(
                    t=self._frame,
                    kind=str(d.get("kind", "")),
                    src=str(d.get("src", "")),
                    dst=str(d.get("dst", "")),
                ))
            except Exception as e:
                self.get_logger().warn(f"bad observation: {e}")

        def _tick(self) -> None:
            obs_batch = self._pending_observations
            self._pending_observations = []
            weights = self._estimator.step(obs_batch)
            sigmas = {
                cn: sigma_cycle(weights, coalition.cycles[cn])
                for cn in self._cycle_names
            }
            # Publish σ vector.
            sigma_msg = Float64MultiArray()
            sigma_msg.data = [float(self._frame)] + [
                sigmas[cn] for cn in self._cycle_names
            ]
            self._sigma_pub.publish(sigma_msg)
            # Publish weights vector.
            w_msg = Float64MultiArray()
            w_msg.data = [float(self._frame)] + [
                weights[rn] for rn in self._relation_names
            ]
            self._weights_pub.publish(w_msg)
            # Step policy.
            out = self._policy.step(self._frame, sigmas)
            for action in out.actions:
                action_msg = String()
                action_msg.data = action
                self._action_pub.publish(action_msg)
                self.get_logger().info(
                    f"policy fired @t={self._frame}: action={action}"
                )
            self._frame += 1

    return RapportPipelineNode()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coalition",
        default="data/coalitions/triad_hri.hymeko",
    )
    args = ap.parse_args(argv)

    rclpy, *_ = _import_ros()
    rclpy.init()
    node = make_pipeline_node(args.coalition)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
