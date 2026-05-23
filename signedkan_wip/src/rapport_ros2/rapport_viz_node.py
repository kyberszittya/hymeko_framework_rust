"""RapportVizNode — publish RViz markers for the triadic graph + σ HUD.

Subscribes:
  /rapport/weights        (std_msgs/Float64MultiArray, per-edge weights)
  /rapport/sigma          (std_msgs/Float64MultiArray, per-cycle σ values)
  /rapport/relation_names (std_msgs/String, latched JSON list)
  /rapport/cycle_names    (std_msgs/String, latched JSON list)
  /model/{alice,bob,r1}/pose  (geometry_msgs/Pose, agent positions)

Publishes:
  /rapport/markers  (visualization_msgs/MarkerArray)

Markers rendered:
  * Per-agent agent labels at agent positions.
  * Per-relation arrow + colour-by-sign (green / red) + opacity-by-|w|.
  * One TextViewFacing marker at the room's centroid showing
    σ(triad) = … in large bold.

Plan: docs/plans/2026-05-18-gz-rapport-demo/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _import_ros() -> tuple[Any, ...]:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy,
        )
        from geometry_msgs.msg import Point, Pose
        from std_msgs.msg import ColorRGBA, Float64MultiArray, String
        from visualization_msgs.msg import Marker, MarkerArray
        return (rclpy, Node, QoSProfile, QoSDurabilityPolicy,
                QoSReliabilityPolicy, Point, Pose, ColorRGBA,
                Float64MultiArray, String, Marker, MarkerArray)
    except ImportError as e:
        raise RuntimeError(
            "ROS 2 message types not importable. Source ROS 2 Kilted "
            "+ activate .venv-rapport-ros2 first. "
            f"(error: {e})"
        ) from e


def make_viz_node(coalition_path: str | Path,
                  publish_rate_hz: float = 10.0) -> Any:
    (rclpy, Node, QoSProfile, QoSDurabilityPolicy,
     QoSReliabilityPolicy, Point, Pose, ColorRGBA,
     Float64MultiArray, String, Marker, MarkerArray) = _import_ros()
    from ..rapport.coalition import load_coalition
    coalition = load_coalition(Path(coalition_path))

    latched_qos = QoSProfile(
        depth=1,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        reliability=QoSReliabilityPolicy.RELIABLE,
    )

    class RapportVizNode(Node):
        def __init__(self) -> None:
            super().__init__("rapport_viz")
            self._poses: dict[str, tuple[float, float, float]] = {}
            self._weights: dict[str, float] = {
                r.name: 0.0 for r in coalition.relations.values()
            }
            self._sigma: dict[str, float] = {
                c.name: 0.0 for c in coalition.cycles.values()
            }
            self._relation_names: list[str] = list(coalition.relations.keys())
            self._cycle_names: list[str] = list(coalition.cycles.keys())

            # Subscribe to bridged pose topics for each agent.
            for name, binding in coalition.gz_bindings.items():
                self.create_subscription(
                    Pose, binding.pose_topic,
                    self._make_pose_cb(name), 10,
                )

            # Subscribe to pipeline outputs.
            self.create_subscription(
                Float64MultiArray, "/rapport/weights",
                self._on_weights, 10,
            )
            self.create_subscription(
                Float64MultiArray, "/rapport/sigma",
                self._on_sigma, 10,
            )
            self.create_subscription(
                String, "/rapport/relation_names",
                self._on_relation_names, latched_qos,
            )
            self.create_subscription(
                String, "/rapport/cycle_names",
                self._on_cycle_names, latched_qos,
            )

            self._marker_pub = self.create_publisher(
                MarkerArray, "/rapport/markers", 10,
            )
            self._timer = self.create_timer(
                1.0 / publish_rate_hz, self._publish_markers,
            )
            self.get_logger().info(
                f"viz publishing markers for {len(coalition.agents)} agents, "
                f"{len(coalition.relations)} relations, "
                f"{len(coalition.cycles)} cycles"
            )

        def _make_pose_cb(self, name: str):
            def _cb(msg) -> None:
                self._poses[name] = (
                    float(msg.position.x), float(msg.position.y),
                    float(msg.position.z),
                )
            return _cb

        def _on_weights(self, msg) -> None:
            if len(msg.data) < 1 + len(self._relation_names):
                return
            for i, name in enumerate(self._relation_names):
                self._weights[name] = float(msg.data[i + 1])

        def _on_sigma(self, msg) -> None:
            if len(msg.data) < 1 + len(self._cycle_names):
                return
            for i, name in enumerate(self._cycle_names):
                self._sigma[name] = float(msg.data[i + 1])

        def _on_relation_names(self, msg) -> None:
            try:
                self._relation_names = list(json.loads(msg.data))
            except Exception:
                pass

        def _on_cycle_names(self, msg) -> None:
            try:
                self._cycle_names = list(json.loads(msg.data))
            except Exception:
                pass

        def _make_marker(self, ns: str, mid: int, mtype: int) -> Any:
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = ns
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
            m.frame_locked = False
            return m

        def _publish_markers(self) -> None:
            if not self._poses:
                return
            arr = MarkerArray()

            # ─── Agent labels at agent positions ────────────────────
            for i, (agent_name, (x, y, z)) in enumerate(self._poses.items()):
                m = self._make_marker("agents", i, Marker.TEXT_VIEW_FACING)
                m.pose.position.x = x
                m.pose.position.y = y
                m.pose.position.z = z + 1.95
                m.scale.z = 0.20
                m.text = agent_name
                m.color = ColorRGBA(r=0.10, g=0.10, b=0.10, a=1.0)
                arr.markers.append(m)

            # ─── Relation edges (signed, opacity-weighted) ─────────
            for j, rel in enumerate(coalition.relations.values()):
                if rel.src not in self._poses or rel.dst not in self._poses:
                    continue
                w = self._weights.get(rel.name, 0.0)
                m = self._make_marker("relations", j, Marker.LINE_STRIP)
                m.scale.x = 0.06  # line width
                src = self._poses[rel.src]
                dst = self._poses[rel.dst]
                # Lift the line so it floats above the floor.
                lift = 1.0
                m.points.append(Point(x=src[0], y=src[1], z=src[2] + lift))
                m.points.append(Point(x=dst[0], y=dst[1], z=dst[2] + lift))
                if w >= 0:
                    m.color = ColorRGBA(
                        r=0.18, g=0.62, b=0.32,
                        a=max(0.15, min(1.0, abs(w))),
                    )
                else:
                    m.color = ColorRGBA(
                        r=0.85, g=0.17, b=0.21,
                        a=max(0.15, min(1.0, abs(w))),
                    )
                arr.markers.append(m)
                # Weight label at edge midpoint.
                mx, my, mz = (
                    (src[0] + dst[0]) / 2.0,
                    (src[1] + dst[1]) / 2.0,
                    (src[2] + dst[2]) / 2.0 + lift + 0.10,
                )
                lbl = self._make_marker("relation_labels", j,
                                          Marker.TEXT_VIEW_FACING)
                lbl.pose.position.x = mx
                lbl.pose.position.y = my
                lbl.pose.position.z = mz
                lbl.scale.z = 0.12
                lbl.text = f"{w:+.2f}"
                lbl.color = (
                    ColorRGBA(r=0.13, g=0.40, b=0.20, a=1.0) if w >= 0
                    else ColorRGBA(r=0.55, g=0.10, b=0.13, a=1.0)
                )
                arr.markers.append(lbl)

            # ─── σ HUD: text at the room centroid ──────────────────
            if self._sigma:
                centroid_x = sum(p[0] for p in self._poses.values()) / len(self._poses)
                centroid_y = sum(p[1] for p in self._poses.values()) / len(self._poses)
                hud = self._make_marker("sigma_hud", 0,
                                          Marker.TEXT_VIEW_FACING)
                hud.pose.position.x = centroid_x
                hud.pose.position.y = centroid_y
                hud.pose.position.z = 2.6
                hud.scale.z = 0.30
                # Show all cycle σ values; in the triad demo there's
                # exactly one, but the design generalises.
                lines = []
                for cn, sv in self._sigma.items():
                    lines.append(f"σ({cn}) = {sv:+.3f}")
                hud.text = "\n".join(lines)
                # Colour by the FIRST cycle's σ (the visit demo has only one).
                first_sigma = next(iter(self._sigma.values()))
                if first_sigma > 0:
                    hud.color = ColorRGBA(r=0.18, g=0.62, b=0.32, a=1.0)
                else:
                    hud.color = ColorRGBA(r=0.85, g=0.17, b=0.21, a=1.0)
                arr.markers.append(hud)

            self._marker_pub.publish(arr)

    return RapportVizNode()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coalition",
        default="data/coalitions/triad_hri.hymeko",
    )
    args = ap.parse_args(argv)
    rclpy, *_ = _import_ros()
    rclpy.init()
    node = make_viz_node(args.coalition)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
