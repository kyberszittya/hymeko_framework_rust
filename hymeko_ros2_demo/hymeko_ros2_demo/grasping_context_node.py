"""grasping_context_node — the Tier-1 bridge.

Loads the HyMeKo IR from a ``.hymeko`` scenario file, subscribes to
the ROS topics that bind the grasping context's input vertices,
evaluates the 6 signed hyperedges at a fixed rate, and publishes the
aggregated outputs (stability margin, configuration, contact force)
back to ROS.

Usage::

    ros2 run hymeko_ros2_demo grasping_context_node \\
        --ros-args -p scenario_file:=/path/to/hymeko_robot.hymeko \\
                   -p topic_mapping_file:=/path/to/topic_mapping.yaml \\
                   -p tick_rate_hz:=10.0
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# hymeko is the PyO3 wheel (cp312 install via pip --user, see README).
try:
    import hymeko
except ImportError as exc:
    raise ImportError(
        "hymeko Python wheel not importable. Install with: "
        "pip install --user "
        "target/wheels/hymeko-0.1.0-cp312-cp312-linux_x86_64.whl"
    ) from exc

from hymeko_ros2_demo.topic_binding import (
    BindingConfig,
    Hyperedge,
    aggregate_grasp_stability,
    extract_hyperedges,
    find_context_block,
    load_yaml_config,
)


def _import_msg_class(msg_type: str):
    """Resolve 'std_msgs/msg/Float64' → the actual Python class."""
    pkg, _, cls = msg_type.replace(".", "/").rpartition("/")
    if not pkg:
        raise ValueError(f"bad msg_type: {msg_type}")
    # 'std_msgs/msg/Float64' → 'std_msgs.msg', 'Float64'
    parts = pkg.replace("/", ".").split(".")
    module_name = ".".join(parts)
    module = importlib.import_module(module_name)
    return getattr(module, cls)


def _extract_field(msg: Any, field: Optional[str]) -> float:
    """Read a single scalar from a ROS message via a dot-path."""
    if field is None:
        field = "data"
    cur: Any = msg
    for piece in field.split("."):
        cur = getattr(cur, piece)
    try:
        return float(cur)
    except (TypeError, ValueError):
        return float("nan")


class GraspingContextNode(Node):
    """The Tier-1 grasping-context bridge node."""

    def __init__(self) -> None:
        super().__init__("grasping_context_node")

        # Parameters.
        self.declare_parameter("scenario_file", "")
        self.declare_parameter("topic_mapping_file", "")
        self.declare_parameter("tick_rate_hz", 10.0)

        scen_path = Path(
            self.get_parameter("scenario_file").get_parameter_value().string_value
        )
        map_path = Path(
            self.get_parameter("topic_mapping_file").get_parameter_value().string_value
        )
        tick_rate = float(
            self.get_parameter("tick_rate_hz").get_parameter_value().double_value
        )

        if not scen_path.exists():
            raise FileNotFoundError(
                f"scenario_file does not exist: {scen_path!s} "
                "(pass --ros-args -p scenario_file:=...)"
            )
        if not map_path.exists():
            raise FileNotFoundError(
                f"topic_mapping_file does not exist: {map_path!s}"
            )

        # Load the .hymeko file → IR → context hyperedges.
        src = scen_path.read_text()
        self.ir = hymeko.parse_hymeko_rs(src)
        self.config: BindingConfig = load_yaml_config(map_path)
        ctx_block = find_context_block(self.ir, self.config.context)
        if ctx_block is None:
            raise RuntimeError(
                f"context '{self.config.context}' not in {scen_path.name}"
            )
        self.edges: List[Hyperedge] = extract_hyperedges(ctx_block)
        if not self.edges:
            raise RuntimeError(
                f"context '{self.config.context}' has 0 hyperedges"
            )

        self.get_logger().info(
            f"loaded {len(self.edges)} hyperedges from "
            f"{scen_path.name}::{self.config.context}"
        )
        for e in self.edges:
            self.get_logger().info(
                f"  {e.name}: + {list(e.inputs)} -> - {list(e.outputs)}"
            )

        # V_global state — last value seen on each input topic.
        self._v_global: Dict[str, float] = {}

        # Subscriptions for each input vertex.
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self._subs = []
        for tmap in self.config.inputs:
            try:
                msg_cls = _import_msg_class(tmap.msg_type)
            except (ImportError, ValueError, AttributeError) as exc:
                self.get_logger().warn(
                    f"skipping input {tmap.vertex} ({tmap.topic}): "
                    f"could not import {tmap.msg_type!r}: {exc!r}"
                )
                continue
            sub = self.create_subscription(
                msg_cls, tmap.topic,
                self._make_input_callback(tmap.vertex, tmap.field),
                qos,
            )
            self._subs.append(sub)
            self.get_logger().info(
                f"  sub  {tmap.vertex:20s} <- {tmap.topic} "
                f"({tmap.msg_type} . {tmap.field or 'data'})"
            )

        # Publishers for each output vertex.
        self._pubs: Dict[str, Any] = {}
        for tmap in self.config.outputs:
            try:
                msg_cls = _import_msg_class(tmap.msg_type)
            except (ImportError, ValueError, AttributeError) as exc:
                self.get_logger().warn(
                    f"skipping output {tmap.vertex}: {exc!r}"
                )
                continue
            self._pubs[tmap.vertex] = (
                self.create_publisher(msg_cls, tmap.topic, 10),
                tmap.field,
            )
            self.get_logger().info(
                f"  pub  {tmap.vertex:20s} -> {tmap.topic}"
            )

        # Diagnostics publisher: full V_global state as a JSON-encoded
        # std_msgs/String. Lets the dashboard subscribe to one topic
        # instead of reconstructing V_global from the raw input topics.
        from std_msgs.msg import String as _StringMsg
        self._diag_pub = self.create_publisher(
            _StringMsg, "/hymeko/grasping/diagnostics", 10,
        )
        # Edge metadata cached for the dashboard (vertex names per edge).
        self._edge_meta = [
            {"name": e.name, "inputs": list(e.inputs), "outputs": list(e.outputs)}
            for e in self.edges
        ]

        # Tick timer.
        period_s = 1.0 / max(0.1, tick_rate)
        self._timer = self.create_timer(period_s, self._tick)
        self._tick_count = 0
        self.get_logger().info(
            f"grasping_context_node armed @ {tick_rate:.1f} Hz "
            f"(context={self.config.context}, edges={len(self.edges)})"
        )

    def _make_input_callback(self, vertex: str, field: Optional[str]):
        def _cb(msg):
            self._v_global[vertex] = _extract_field(msg, field)
        return _cb

    def _tick(self):
        """Evaluate the hyperedges and publish the outputs."""
        # Topological evaluation: each edge's outputs become available
        # for subsequent edges in the same tick.  The file ordering
        # already encodes a valid topological sort for grasping_context.
        for edge in self.edges:
            bound: Dict[str, float] = {
                inp: float(self._v_global.get(inp, 0.0)) for inp in edge.inputs
            }
            out_value = self._aggregate(edge, bound)
            # Write outputs back into V_global so downstream edges see them.
            for out_name in edge.outputs:
                self._v_global[out_name] = out_value

        # Publish whatever the config asked for.
        for vertex, (pub, field) in self._pubs.items():
            value = float(self._v_global.get(vertex, 0.0))
            try:
                msg = self._build_output_msg(pub.msg_type, field, value)
                pub.publish(msg)
            except Exception as exc:  # noqa: BLE001 — surface at boundary
                self.get_logger().warn(
                    f"failed to publish {vertex}: {exc!r}"
                )

        # Diagnostics: full V_global + edge metadata as JSON.
        try:
            import json as _json
            from std_msgs.msg import String as _StringMsg
            payload = {
                "tick": self._tick_count,
                "context": self.config.context,
                "v_global": {k: float(v) for k, v in self._v_global.items()},
                "edges": self._edge_meta,
            }
            diag = _StringMsg()
            diag.data = _json.dumps(payload)
            self._diag_pub.publish(diag)
        except Exception as exc:  # noqa: BLE001 — diagnostics must never crash the node
            self.get_logger().warn(f"diagnostics publish failed: {exc!r}")

        self._tick_count += 1
        # Compact per-tick stability line so you can SEE the gauge moving
        # in the terminal without opening the GUI.  Emits at 2 Hz when
        # ticking at 10 Hz.
        if self._tick_count % 5 == 0:
            f_l = self._v_global.get("force_vector", 0.0)
            f_g = self._v_global.get("grip_force", 0.0)
            s_g = self._v_global.get("stability_margin", 0.0)
            c_g = self._v_global.get("configuration", 0.0)
            self.get_logger().info(
                f"tick {self._tick_count:4d}  "
                f"F_l={f_l:.3f}  F_g={f_g:.2f}N  C_g={c_g:.2f}  "
                f"S_g={s_g:.3f}"
            )
        if self._tick_count % 100 == 0:
            self.get_logger().info(
                f"tick {self._tick_count}: V_global = "
                f"{ {k: round(v, 3) for k, v in self._v_global.items()} }"
            )

    @staticmethod
    def _build_output_msg(msg_cls: type, field: Optional[str], value: float):
        msg = msg_cls()
        target_field = field or "data"
        cur = msg
        parts = target_field.split(".")
        for piece in parts[:-1]:
            cur = getattr(cur, piece)
        setattr(cur, parts[-1], float(value))
        return msg

    def _aggregate(self, edge: Hyperedge, inputs: Dict[str, float]) -> float:
        """Placeholder per-edge aggregation function.

        Special-cases the stability edge (uses the practitioner-style
        1/(1+|F_l - F_g|) formula); all other edges use the
        clamped-mean default.  See ``topic_binding.aggregate_*`` for
        the rationale (the paper does not pin closed forms).
        """

        if "stability" in edge.name or edge.name == "grasp_stability":
            out = aggregate_grasp_stability(inputs)
            if "stability_margin" in out:
                return out["stability_margin"]
        # Default: mean clipped to [0, 1].
        if not inputs:
            return 0.0
        vals = [float(v) for v in inputs.values()]
        m = sum(vals) / len(vals)
        return max(0.0, min(1.0, m))


def main(args=None):
    rclpy.init(args=args)
    try:
        node = GraspingContextNode()
    except Exception as exc:  # noqa: BLE001 — fail loudly at startup
        print(f"[grasping_context_node] startup failed: {exc!r}", file=sys.stderr)
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
