"""Tests for the kinematic family classifier (v0.5).

The classifier is a real torch model; full training takes ~30 s per
arity. These tests:

  - cover the cheap, pure-data paths (rule-based fallback, format
    round-trip) without invoking the model,
  - exercise prediction end-to-end on the pretrained checkpoints
    *only if those checkpoints are present on disk*. CI without
    pretrained checkpoints skips the heavy tests cleanly rather
    than retraining (which would take minutes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from signedkan_wip.src.demo.kinematic import (
    KinematicBundle, find_urdf_by_id, load_urdf_bundle,
    load_urdf_registry,
)
from signedkan_wip.src.demo.kinematic_classifier import (
    CLASSIFIER_FORMAT_VERSION, ClassificationResult, FAMILY_NAMES,
    PRETRAINED_DIR, load_classifier, predict_family, save_classifier,
)
from signedkan_wip.src.datasets import SignedGraph


def _empty_bundle() -> KinematicBundle:
    """An empty-graph bundle that exercises the rule-based path."""
    g = SignedGraph(edges=np.zeros((0, 2), dtype=np.int64),
                     signs=np.zeros((0,), dtype=np.int8), n_nodes=0)
    return KinematicBundle(
        name="empty", urdf_path=Path("/dev/null"),
        graph=g, link_names=[], joints=[],
        cycle_counts={3: 0, 4: 0, 5: 0, 6: 0},
    )


def test_family_names_are_canonical():
    assert FAMILY_NAMES == ["four_bar", "stewart", "delta_3rrr", "serial"]


def test_rule_based_path_for_empty_bundle():
    """No cycles → rule-based 'serial' prediction with 1.0 confidence."""
    b = _empty_bundle()
    r = predict_family(b)
    assert isinstance(r, ClassificationResult)
    assert r.predicted_family == "serial"
    assert r.confidence == pytest.approx(1.0)
    assert r.rule_based is True
    assert r.arity_used is None
    assert r.probs.shape == (4,)
    assert r.probs.sum() == pytest.approx(1.0)


def test_rule_based_path_for_serial_chain():
    """A real open-chain URDF must classify as 'serial' via the rule."""
    entries = load_urdf_registry()
    chain = load_urdf_bundle(
        find_urdf_by_id(entries, "chain_10").path, name="chain_10")
    r = predict_family(chain)
    assert r.predicted_family == "serial"
    assert r.rule_based is True
    assert r.arity_used is None


def test_save_load_roundtrip(tmp_path):
    """Without retraining: a hand-built small GraphLevelHSiKAN must
    serialise + deserialise + match its state_dict."""
    from signedkan_wip.experiments.runs.run_phase11_kinematic_tasks import GraphLevelHSiKAN
    model = GraphLevelHSiKAN(n_nodes_max=8, arity=4, hidden=8, n_classes=4)
    ckpt = tmp_path / "k4_test.pt"
    save_classifier(ckpt, model, arity=4, n_nodes_max=8, train_acc=0.95,
                       hidden=8)
    assert ckpt.is_file()
    loaded, arity, n_max = load_classifier(ckpt)
    assert arity == 4
    assert n_max == 8
    # State dict must match.
    sd1 = model.state_dict()
    sd2 = loaded.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k].cpu(), sd2[k].cpu())


def test_load_rejects_wrong_kind(tmp_path):
    """A torch.save of an arbitrary dict must be rejected by the loader."""
    bad = tmp_path / "bad.pt"
    torch.save({"kind": "something_else", "arity": 4}, bad)
    with pytest.raises(ValueError, match="not a kinematic_family_classifier"):
        load_classifier(bad)


def test_load_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_classifier(tmp_path / "nope.pt")


def test_classifier_format_version_pinned():
    """If we ever bump the format, the loader must understand both
    versions or this test fails noisily."""
    assert CLASSIFIER_FORMAT_VERSION == 1


# ---- Heavy tests: only if pretrained checkpoints exist on disk ----

PRETRAINED_PRESENT = (
    (PRETRAINED_DIR / "family_classifier_k4.pt").is_file()
    and (PRETRAINED_DIR / "family_classifier_k6.pt").is_file()
)


@pytest.mark.skipif(not PRETRAINED_PRESENT,
                     reason="pretrained classifiers not present "
                            "(run: python -m signedkan_wip.src.demo.kinematic_classifier)")
def test_predict_family_on_four_bar():
    entries = load_urdf_registry()
    b = load_urdf_bundle(find_urdf_by_id(entries, "four_bar").path,
                           name="four_bar")
    r = predict_family(b)
    assert r.predicted_family == "four_bar"
    assert r.arity_used == 4
    assert r.rule_based is False
    assert r.confidence > 0.9


@pytest.mark.skipif(not PRETRAINED_PRESENT,
                     reason="pretrained classifiers not present")
def test_predict_family_on_stewart():
    entries = load_urdf_registry()
    b = load_urdf_bundle(find_urdf_by_id(entries, "stewart").path,
                           name="stewart")
    r = predict_family(b)
    assert r.predicted_family == "stewart"
    assert r.arity_used == 6
    assert r.rule_based is False
    assert r.confidence > 0.9


@pytest.mark.skipif(not PRETRAINED_PRESENT,
                     reason="pretrained classifiers not present")
def test_predict_family_discriminates_stewart_from_delta():
    """The most interesting bit — both have k=6 cycles, the classifier
    must tell them apart by cycle count / topology."""
    entries = load_urdf_registry()
    stewart = load_urdf_bundle(find_urdf_by_id(entries, "stewart").path,
                                 name="stewart")
    delta = load_urdf_bundle(find_urdf_by_id(entries, "delta_3rrr").path,
                               name="delta_3rrr")
    rs = predict_family(stewart)
    rd = predict_family(delta)
    assert rs.predicted_family == "stewart"
    assert rd.predicted_family == "delta_3rrr"
    # They must end up in different argmax classes.
    assert rs.predicted_label != rd.predicted_label
