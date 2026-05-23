"""Tests for VocPersonDetector (Stage H / D-3 inference wrapper)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_stage_h_ckpt() -> Path | None:
    """Find a Stage H person checkpoint on disk if one exists."""
    candidates = sorted(REPO_ROOT.glob(
        "signedkan_wip/experiments/results/stage_h_voc_person_*/checkpoints/*.pt"
    ))
    return candidates[0] if candidates else None


def test_vision_config_threaded_through_hymeko():
    """vision_config block in triad_hri.hymeko surfaces in Coalition.

    The current production triad uses ``detector_kind = "voc_person"``
    (Stage H single-class person detector). The test pins this
    state — if the .hymeko is reverted to a placeholder backend,
    update both the file and this assertion together.
    """
    from signedkan_wip.src.rapport.coalition import load_coalition
    c = load_coalition(REPO_ROOT / "data" / "coalitions" / "triad_hri.hymeko")
    assert len(c.vision_configs) == 1
    cfg = next(iter(c.vision_configs.values()))
    assert cfg.detector_kind == "voc_person"
    # The checkpoint path is populated; just sanity-check non-empty.
    assert cfg.checkpoint != ""
    assert 0.0 <= cfg.score_threshold <= 1.0


def test_voc_detector_loads_from_stage_h_ckpt():
    """Load a Stage H checkpoint and run inference on a synthetic image."""
    ckpt = _find_stage_h_ckpt()
    if ckpt is None:
        pytest.skip("no Stage H checkpoint found yet")
    from signedkan_wip.src.rapport_ros2.voc_detector import VocPersonDetector
    detector = VocPersonDetector(ckpt, score_threshold=0.0, device="cpu")
    info = detector.info
    assert info["n_classes"] in (1, 20)
    if info["n_classes"] == 1:
        assert detector.class_names == ("person",)
    # Run inference on a small synthetic frame.
    rgb = (np.random.default_rng(0).integers(0, 256, size=(240, 320, 3))
           ).astype(np.uint8)
    dets = detector.detect(rgb)
    assert isinstance(dets, list)
    for d in dets:
        assert 0 <= d.x0 <= d.x1 <= 320
        assert 0 <= d.y0 <= d.y1 <= 240
        assert 0.0 <= d.score <= 1.0


def test_voc_detector_rejects_bad_input():
    """detect() must validate dtype and shape."""
    ckpt = _find_stage_h_ckpt()
    if ckpt is None:
        pytest.skip("no Stage H checkpoint found yet")
    from signedkan_wip.src.rapport_ros2.voc_detector import VocPersonDetector
    detector = VocPersonDetector(ckpt, device="cpu")
    with pytest.raises(ValueError, match="uint8"):
        detector.detect(np.zeros((240, 320, 3), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        detector.detect(np.zeros((240, 320), dtype=np.uint8))


def test_voc_detector_score_threshold_filter():
    """A very high score threshold should filter out all detections."""
    ckpt = _find_stage_h_ckpt()
    if ckpt is None:
        pytest.skip("no Stage H checkpoint found yet")
    from signedkan_wip.src.rapport_ros2.voc_detector import VocPersonDetector
    detector = VocPersonDetector(ckpt, score_threshold=0.99, device="cpu")
    rgb = (np.random.default_rng(0).integers(0, 256, size=(240, 320, 3))
           ).astype(np.uint8)
    dets = detector.detect(rgb)
    assert dets == []


def test_voc_detector_class_names_match_checkpoint():
    """Class-name table is consistent with n_classes from the ckpt."""
    ckpt = _find_stage_h_ckpt()
    if ckpt is None:
        pytest.skip("no Stage H checkpoint found yet")
    from signedkan_wip.src.rapport_ros2.voc_detector import VocPersonDetector
    detector = VocPersonDetector(ckpt, device="cpu")
    assert len(detector.class_names) == detector.n_classes


def test_voc_detector_swap_in_hymeko_path(tmp_path):
    """End-to-end: a .hymeko file declaring vision_config "voc_person"
    must be parseable; the detector_kind / checkpoint / score_threshold
    fields all surface in the Coalition.
    """
    from signedkan_wip.src.rapport.coalition import load_coalition
    # Read the meta_hri schema (it's required by the @import).
    meta = REPO_ROOT / "data" / "coalitions" / "meta_hri.hymeko"
    (tmp_path / "meta_hri.hymeko").write_text(meta.read_text())
    test_file = tmp_path / "voc_person_coalition.hymeko"
    test_file.write_text("""
test_description {
    @"meta_hri.hymeko";
    using hri_meta as hri;
}

test_coalition: hri {
    alice: hri.human {}
    bob: hri.human {}
    r1: hri.robot {}

    r_ab: hri.interpersonal { from alice; to bob; sign 1; magnitude 1.0; }
    r_ar: hri.hri_relation  { from alice; to r1;  sign 1; magnitude 1.0; }
    r_br: hri.hri_relation  { from bob;   to r1;  sign 1; magnitude 1.0; }

    vision_r1: hri.vision_config {
        detector_kind   "voc_person";
        checkpoint      "/path/to/some.pt";
        score_threshold 0.42;
    }
}
""")
    c = load_coalition(test_file)
    cfg = c.vision_configs["vision_r1"]
    assert cfg.detector_kind == "voc_person"
    assert cfg.checkpoint == "/path/to/some.pt"
    assert cfg.score_threshold == 0.42
