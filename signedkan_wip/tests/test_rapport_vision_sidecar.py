"""Unit tests for the vision sidecar's pure-Python detection math."""
from __future__ import annotations

import numpy as np


def test_rgb_to_hsv_pure_red():
    from signedkan_wip.src.rapport_ros2.vision_sidecar_node import rgb_image_to_hsv
    img = np.array([[[255, 0, 0]]], dtype=np.uint8)
    hsv = rgb_image_to_hsv(img)
    h, s, v = hsv[0, 0]
    assert h == 0.0
    assert s == 1.0
    assert v == 1.0


def test_rgb_to_hsv_pure_blue():
    from signedkan_wip.src.rapport_ros2.vision_sidecar_node import rgb_image_to_hsv
    img = np.array([[[0, 0, 255]]], dtype=np.uint8)
    hsv = rgb_image_to_hsv(img)
    h, s, v = hsv[0, 0]
    assert abs(h - 240.0) < 1e-3
    assert s == 1.0
    assert v == 1.0


def test_detect_human_blobs_finds_blue_patch():
    """A 240x320 image with a single blue patch should yield 1 bbox."""
    from signedkan_wip.src.rapport_ros2.vision_sidecar_node import detect_human_blobs
    img = np.full((240, 320, 3), 250, dtype=np.uint8)   # white-ish bg
    # Paint a blue capsule-coloured rectangle (HSV hue ~210).
    img[80:160, 130:170] = (107, 158, 212)  # ~ alice/bob body colour
    bboxes = detect_human_blobs(img)
    assert len(bboxes) == 1
    x0, y0, x1, y1 = bboxes[0]
    assert 125 <= x0 <= 135
    assert 75 <= y0 <= 85
    assert 165 <= x1 <= 175
    assert 155 <= y1 <= 165


def test_detect_human_blobs_finds_two_separated_patches():
    from signedkan_wip.src.rapport_ros2.vision_sidecar_node import detect_human_blobs
    img = np.full((240, 320, 3), 250, dtype=np.uint8)
    img[80:160, 60:100] = (107, 158, 212)   # alice
    img[80:160, 220:260] = (107, 158, 212)  # bob
    bboxes = detect_human_blobs(img)
    assert len(bboxes) == 2


def test_detect_human_blobs_ignores_non_blue_objects():
    from signedkan_wip.src.rapport_ros2.vision_sidecar_node import detect_human_blobs
    img = np.full((240, 320, 3), 250, dtype=np.uint8)
    img[80:160, 60:100] = (220, 60, 60)    # red — should be rejected
    img[80:160, 220:260] = (60, 220, 60)   # green — should be rejected
    bboxes = detect_human_blobs(img)
    assert bboxes == []


def test_detect_human_blobs_rejects_too_small():
    from signedkan_wip.src.rapport_ros2.vision_sidecar_node import detect_human_blobs
    img = np.full((240, 320, 3), 250, dtype=np.uint8)
    img[100:103, 100:103] = (107, 158, 212)  # only 9 pixels — below min
    bboxes = detect_human_blobs(img, min_pixels=40)
    assert bboxes == []


def test_camera_topic_threaded_through_hymeko():
    """The new camera_topic field in gz_binding must surface in the
    Coalition.gz_bindings dict after parsing the .hymeko file."""
    from pathlib import Path
    from signedkan_wip.src.rapport.coalition import load_coalition
    repo_root = Path(__file__).resolve().parents[2]
    c = load_coalition(repo_root / "data" / "coalitions" / "triad_hri.hymeko")
    r1 = c.gz_bindings["r1"]
    assert r1.camera_topic == "/r1/camera/image"
    # alice/bob shouldn't have cameras.
    assert c.gz_bindings["alice"].camera_topic is None
    assert c.gz_bindings["bob"].camera_topic is None
