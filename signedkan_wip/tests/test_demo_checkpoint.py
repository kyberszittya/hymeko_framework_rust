"""Round-trip tests for the demo checkpoint format.

Doesn't require a real `MixedAritySignedKAN` — uses a toy `nn.Module`
to verify state_dict + cfg + meta + optional inference_bundle +
optional classifier_module all serialize and reload correctly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn as nn

from signedkan_wip.src.demo.checkpoint import (
    CheckpointMeta, InferenceBundle, FORMAT_VERSION,
    load_checkpoint, save_checkpoint,
)


# A toy model that can stand in for the round-trip test.

@dataclass
class _ToyCfg:
    hidden: int = 4
    n_classes: int = 2


class _Toy(nn.Module):
    def __init__(self, cfg: _ToyCfg):
        super().__init__()
        self.cfg = cfg
        self.lin = nn.Linear(cfg.hidden, cfg.n_classes)

    def forward(self, x):
        return self.lin(x)


def _model_class_qualified() -> str:
    return f"{_Toy.__module__}.{_Toy.__name__}"


def test_save_and_load_minimum(tmp_path):
    cfg = _ToyCfg(hidden=8, n_classes=3)
    model = _Toy(cfg)
    meta = CheckpointMeta(
        dataset="bitcoin_alpha", n_nodes=5881, seed=0, n_epochs=80,
        test_auc=0.997, test_f1=0.92, n_params=10,
        tuple_specs=[["cycle", 2, None], ["walk", 3, 2]],
    )
    out = tmp_path / "toy.pt"
    save_checkpoint(out, model, cfg, _model_class_qualified(), meta)
    assert out.exists()

    model2, cfg2, meta2, bundle2, clf2 = load_checkpoint(out)
    assert isinstance(model2, _Toy)
    assert cfg2.hidden == 8
    assert cfg2.n_classes == 3
    assert meta2.dataset == "bitcoin_alpha"
    assert meta2.test_auc == pytest.approx(0.997)
    assert meta2.tuple_specs == [["cycle", 2, None], ["walk", 3, 2]]
    assert bundle2 is None
    assert clf2 is None

    # State_dicts should match exactly.
    sd1 = model.state_dict()
    sd2 = model2.state_dict()
    for k in sd1:
        assert torch.equal(sd1[k].cpu(), sd2[k].cpu())


def test_save_and_load_with_inference_bundle_and_classifier(tmp_path):
    cfg = _ToyCfg(hidden=4, n_classes=2)
    model = _Toy(cfg)
    classifier = nn.Linear(8, 1)  # external clf
    meta = CheckpointMeta(
        dataset="bitcoin_otc", n_nodes=5881, seed=2, n_epochs=80,
        tuple_specs=[["cycle", 2, None], ["cycle", 5, None],
                     ["walk", 3, 2], ["walk", 4, 3], ["walk", 5, 4]],
    )
    bundle = InferenceBundle(
        per_arity_te=[("opaque", "tuple", "structure", 42)],
        query_edges=np.array([[0, 1], [2, 3]], dtype=np.int64),
        true_signs=np.array([1, -1], dtype=np.int64),
    )
    out = tmp_path / "with_bundle.pt"
    save_checkpoint(out, model, cfg, _model_class_qualified(), meta,
                     inference_bundle=bundle, classifier_module=classifier)

    _, _, _, b2, c2 = load_checkpoint(out)
    assert b2 is not None
    assert b2.per_arity_te == [("opaque", "tuple", "structure", 42)]
    np.testing.assert_array_equal(b2.query_edges, bundle.query_edges)
    np.testing.assert_array_equal(b2.true_signs, bundle.true_signs)

    assert c2 is not None
    # Classifier state_dict round-trips.
    for k in classifier.state_dict():
        assert torch.equal(classifier.state_dict()[k].cpu(),
                            c2.state_dict()[k].cpu())


def test_load_rejects_non_demo_file(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"some": "other_format"}, bad)
    with pytest.raises(ValueError, match="not a demo checkpoint"):
        load_checkpoint(bad)


def test_load_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


def test_format_version_recorded(tmp_path):
    cfg = _ToyCfg()
    model = _Toy(cfg)
    meta = CheckpointMeta(dataset="slashdot", n_nodes=82140)
    out = tmp_path / "v.pt"
    save_checkpoint(out, model, cfg, _model_class_qualified(), meta)
    payload = torch.load(out, weights_only=False)
    assert payload["format_version"] == FORMAT_VERSION


def test_meta_dataclass_roundtrip(tmp_path):
    """train_args and notes (nested dicts) survive asdict/reconstruction."""
    cfg = _ToyCfg()
    model = _Toy(cfg)
    meta = CheckpointMeta(
        dataset="epinions", n_nodes=131828,
        train_args={"hidden": 8, "max_k4": 100000, "model_name": "HSiKAN"},
        notes={"git_sha": "0c55fa8", "git_dirty": "1"},
    )
    out = tmp_path / "meta.pt"
    save_checkpoint(out, model, cfg, _model_class_qualified(), meta)
    _, _, meta2, _, _ = load_checkpoint(out)
    assert meta2.train_args == {"hidden": 8, "max_k4": 100000,
                                  "model_name": "HSiKAN"}
    assert meta2.notes == {"git_sha": "0c55fa8", "git_dirty": "1"}
