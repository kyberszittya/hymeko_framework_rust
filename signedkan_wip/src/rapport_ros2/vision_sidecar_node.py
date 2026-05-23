"""VisionSidecar — r1's onboard-camera agent detector.

Stage G' CV-2: subscribes to r1's RGB camera feed, runs a fast
HSV-segmented blob detector on each frame, and publishes the
detected agent bounding boxes (image space). For the visit demo
this gives the visual "r1 has eyes" narrative — Mihoko's team can
see, in RViz, what r1 actually perceives.

The detector targets the capsule body colour declared in
``data/models/triad_human/model.sdf`` (RGB 0.42, 0.62, 0.83 →
HSV roughly hue 210). Detection is identity-agnostic in v1: two
human-coloured blobs are reported by their image-space bboxes.
Identity assignment (alice vs bob) is handled downstream by
spatial back-projection against the bridged pose topics if/when
the vision-to-rapport-observation channel is wired up
(Stage G' CV-3 stretch).

Subscribes:
  /r1/camera/image       (sensor_msgs/Image, bridged from gz)

Publishes:
  /vision/detections     (std_msgs/String, JSON list of bboxes)
  /vision/annotated      (sensor_msgs/Image, image + bbox overlay)

Plan: docs/plans/2026-05-18-gz-rapport-demo/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _import_ros() -> tuple[Any, ...]:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        return rclpy, Node, Image, String
    except ImportError as e:
        raise RuntimeError(
            "rclpy / sensor_msgs / std_msgs not importable. "
            "Source ROS 2 Kilted + activate .venv-rapport-ros2. "
            f"(error: {e})"
        ) from e


def rgb_image_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel RGB → HSV (H in [0, 360), S, V in [0, 1]).

    Vectorised; uses the standard hexagonal-coords formula. Avoids
    introducing cv2/opencv as a dependency for a one-off use.
    """
    r = rgb[..., 0].astype(np.float32) / 255.0
    g = rgb[..., 1].astype(np.float32) / 255.0
    b = rgb[..., 2].astype(np.float32) / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    h = np.zeros_like(mx)
    # Hue computation per dominant channel
    mask_r = (mx == r) & (delta > 0)
    mask_g = (mx == g) & (delta > 0)
    mask_b = (mx == b) & (delta > 0)
    h = np.where(mask_r, ((g - b) / np.maximum(delta, 1e-9)) % 6.0, h)
    h = np.where(mask_g, (b - r) / np.maximum(delta, 1e-9) + 2.0, h)
    h = np.where(mask_b, (r - g) / np.maximum(delta, 1e-9) + 4.0, h)
    h = (h * 60.0) % 360.0
    s = np.where(mx > 0, delta / np.maximum(mx, 1e-9), 0.0)
    v = mx
    return np.stack([h, s, v], axis=-1)


def detect_human_blobs(
    rgb: np.ndarray,
    *,
    hue_target: float = 210.0,
    hue_tolerance: float = 25.0,
    sat_min: float = 0.20,
    val_min: float = 0.20,
    min_pixels: int = 40,
) -> list[tuple[int, int, int, int]]:
    """Find connected components of human-coloured pixels.

    Args:
        rgb: (H, W, 3) uint8.
        hue_target: target hue (deg). The triad_human capsule blue
            sits around hue 210.
        hue_tolerance: half-width of the accepted hue range.
        sat_min / val_min: minimum saturation and value, used to
            reject white walls / dark shadow / floor pixels.
        min_pixels: minimum CC size before we count it as a detection.

    Returns:
        list of (x0, y0, x1, y1) bounding boxes in image space.
    """
    hsv = rgb_image_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_diff = np.minimum(
        np.abs(h - hue_target),
        360.0 - np.abs(h - hue_target),
    )
    mask = (hue_diff < hue_tolerance) & (s > sat_min) & (v > val_min)
    return _label_components_to_bboxes(mask, min_pixels=min_pixels)


def _label_components_to_bboxes(
    mask: np.ndarray, *, min_pixels: int = 40,
) -> list[tuple[int, int, int, int]]:
    """Tiny connected-components labeller via flood-fill iteration.

    Avoids depending on scipy.ndimage / opencv for this minimal
    use case. Returns axis-aligned bounding boxes for components
    with at least `min_pixels` pixels.
    """
    # Two-pass approach using a stack-based flood-fill.
    visited = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    out: list[tuple[int, int, int, int]] = []
    ys, xs = np.where(mask & ~visited)
    seen = 0
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        cc_y0, cc_y1 = y0, y0
        cc_x0, cc_x1 = x0, x0
        size = 0
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= H or x < 0 or x >= W:
                continue
            if visited[y, x] or not mask[y, x]:
                continue
            visited[y, x] = True
            size += 1
            if y < cc_y0: cc_y0 = y
            if y > cc_y1: cc_y1 = y
            if x < cc_x0: cc_x0 = x
            if x > cc_x1: cc_x1 = x
            stack.append((y + 1, x)); stack.append((y - 1, x))
            stack.append((y, x + 1)); stack.append((y, x - 1))
        if size >= min_pixels:
            out.append((int(cc_x0), int(cc_y0), int(cc_x1), int(cc_y1)))
        seen += 1
        if seen > 20:
            break  # Sanity cap; real demo has at most 2 humans.
    return out


def make_vision_sidecar_node(coalition_path: str | Path) -> Any:
    rclpy, Node, Image, String = _import_ros()
    from ..rapport.coalition import load_coalition
    coalition = load_coalition(Path(coalition_path))
    # Find the camera-bearing agent.
    cam_topic = None
    cam_agent = None
    for name, binding in coalition.gz_bindings.items():
        if binding.camera_topic is not None:
            cam_topic = binding.camera_topic
            cam_agent = name
            break
    if cam_topic is None:
        raise ValueError(
            f"no agent in coalition {coalition.name!r} has a camera_topic"
        )

    # Stage H / Stage D-3 dispatch: read vision_config from the
    # HyMeKo coalition file. Default to the HSV blob detector if no
    # vision_config block is declared (backward compat).
    detector_kind = "hsv_blob"
    detector_ckpt = ""
    score_threshold = 0.3
    if coalition.vision_configs:
        cfg = next(iter(coalition.vision_configs.values()))
        detector_kind = cfg.detector_kind
        detector_ckpt = cfg.checkpoint
        score_threshold = cfg.score_threshold
    voc_detector_instance = None
    if detector_kind != "hsv_blob":
        if not detector_ckpt:
            raise ValueError(
                f"vision_config detector_kind={detector_kind!r} requires "
                f"a non-empty checkpoint path"
            )
        from .voc_detector import VocPersonDetector
        voc_detector_instance = VocPersonDetector(
            ckpt_path=detector_ckpt,
            score_threshold=score_threshold,
            device="cpu",
        )

    class VisionSidecarNode(Node):
        def __init__(self) -> None:
            super().__init__("vision_sidecar")
            self.create_subscription(Image, cam_topic, self._on_image, 10)
            self._det_pub = self.create_publisher(
                String, "/vision/detections", 10,
            )
            self._annotated_pub = self.create_publisher(
                Image, "/vision/annotated", 10,
            )
            self._frame_idx = 0
            self.get_logger().info(
                f"vision sidecar subscribed to {cam_topic} "
                f"(camera owner = {cam_agent!r}, detector = {detector_kind!r})"
            )
            if voc_detector_instance is not None:
                self.get_logger().info(
                    f"VocPersonDetector loaded: {voc_detector_instance.info}"
                )

        def _on_image(self, msg) -> None:
            # sensor_msgs/Image: data is bytes, encoding tells us layout.
            if msg.encoding not in ("rgb8", "bgr8"):
                self.get_logger().warn(
                    f"unexpected encoding {msg.encoding!r}; expected rgb8/bgr8"
                )
                return
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = buf.reshape(msg.height, msg.width, 3)
            if msg.encoding == "bgr8":
                img = img[..., ::-1]
            self._frame_idx += 1

            # Dispatch on detector kind.
            if voc_detector_instance is not None:
                voc_dets = voc_detector_instance.detect(img)
                bboxes_for_overlay = [
                    (d.x0, d.y0, d.x1, d.y1) for d in voc_dets
                ]
                detection_dicts = [
                    {"x0": d.x0, "y0": d.y0, "x1": d.x1, "y1": d.y1,
                     "agent_kind": d.agent_kind, "score": d.score}
                    for d in voc_dets
                ]
            else:
                bboxes_for_overlay = detect_human_blobs(img)
                detection_dicts = [
                    {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                     "agent_kind": "human"}
                    for (x0, y0, x1, y1) in bboxes_for_overlay
                ]

            payload = {
                "frame": self._frame_idx,
                "stamp_s": (msg.header.stamp.sec
                             + msg.header.stamp.nanosec / 1e9),
                "detections": detection_dicts,
            }
            det_msg = String()
            det_msg.data = json.dumps(payload)
            self._det_pub.publish(det_msg)
            # Annotated image (red bboxes overlayed).
            annotated = img.copy()
            for (x0, y0, x1, y1) in bboxes_for_overlay:
                # Clamp coordinates to image bounds.
                x0 = max(0, min(msg.width - 1, x0))
                y0 = max(0, min(msg.height - 1, y0))
                x1 = max(0, min(msg.width - 1, x1))
                y1 = max(0, min(msg.height - 1, y1))
                annotated[y0:y0 + 2, x0:x1] = (220, 50, 50)
                annotated[y1 - 2:y1, x0:x1] = (220, 50, 50)
                annotated[y0:y1, x0:x0 + 2] = (220, 50, 50)
                annotated[y0:y1, x1 - 2:x1] = (220, 50, 50)
            ann_msg = Image()
            ann_msg.header = msg.header
            ann_msg.height = msg.height
            ann_msg.width = msg.width
            ann_msg.encoding = "rgb8"
            ann_msg.step = msg.width * 3
            ann_msg.data = annotated.tobytes()
            self._annotated_pub.publish(ann_msg)

    return VisionSidecarNode()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coalition",
        default="data/coalitions/triad_hri.hymeko",
    )
    args = ap.parse_args(argv)
    rclpy, *_ = _import_ros()
    rclpy.init()
    node = make_vision_sidecar_node(args.coalition)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
