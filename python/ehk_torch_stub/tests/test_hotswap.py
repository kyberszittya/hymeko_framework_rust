"""Tests for weight transfer + proposal loading (step 5)."""
from __future__ import annotations

import json

import torch
import torch.nn as nn

from ehk_torch_stub import (
    HypergraphConv,
    SplitProposal,
    TransferReport,
    load_proposal,
    reinfer_structure_and_rebuild,
    transfer_compatible_weights,
)


# ─── Weight transfer ─────────────────────────────────────────────────


def _linear_model(d_in: int, d_out: int, *, seed: int = 0) -> nn.Module:
    """Tiny fixture: one Linear layer with a deterministic init."""
    torch.manual_seed(seed)
    m = nn.Linear(d_in, d_out)
    return m


def test_identity_rebuild_transfers_all_weights():
    old = _linear_model(3, 5, seed=0)
    new = _linear_model(3, 5, seed=999)  # different fresh init

    # Before transfer the two disagree on every weight tensor.
    for key in old.state_dict():
        assert not torch.allclose(old.state_dict()[key], new.state_dict()[key]), (
            f"seeds should produce distinct initial {key}"
        )

    report = transfer_compatible_weights(old, new)

    assert set(report.transferred) == set(old.state_dict().keys())
    assert report.shape_mismatch == []
    assert report.fresh_in_new == []
    assert report.dropped_from_old == []
    # After transfer the two agree on every weight tensor.
    for key in old.state_dict():
        assert torch.equal(old.state_dict()[key], new.state_dict()[key])


def test_shape_mismatch_skipped_and_reported():
    old = _linear_model(3, 5, seed=0)
    new = _linear_model(3, 7, seed=999)  # d_out changed → shape mismatch

    pre_transfer = {k: v.clone() for k, v in new.state_dict().items()}
    report = transfer_compatible_weights(old, new)

    # weight + bias both have different shapes — both flagged.
    mismatch_keys = {key for key, _, _ in report.shape_mismatch}
    assert mismatch_keys == {"weight", "bias"}
    assert report.transferred == []
    # New model's parameters remain at their fresh init (unchanged).
    for key, pre in pre_transfer.items():
        assert torch.equal(pre, new.state_dict()[key]), (
            f"new.{key} must not have been overwritten on shape mismatch"
        )


def test_layer_added_in_new_is_fresh():
    # Old: only layer_0. New: layer_0 + layer_1.
    class OldModel(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.layer_0 = nn.Linear(3, 5)

    class NewModel(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(999)
            self.layer_0 = nn.Linear(3, 5)
            self.layer_1 = nn.Linear(5, 2)

    old, new = OldModel(), NewModel()
    layer_1_pre = {k: v.clone() for k, v in new.layer_1.state_dict().items()}

    report = transfer_compatible_weights(old, new)

    assert {"layer_0.weight", "layer_0.bias"}.issubset(set(report.transferred))
    assert set(report.fresh_in_new) == {"layer_1.weight", "layer_1.bias"}
    # layer_1 must still be at its fresh init.
    for k, v in layer_1_pre.items():
        assert torch.equal(v, new.layer_1.state_dict()[k])


def test_layer_removed_from_old_is_dropped():
    # Old: layer_0 + layer_1. New: only layer_0.
    class OldModel(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.layer_0 = nn.Linear(3, 5)
            self.layer_1 = nn.Linear(5, 2)

    class NewModel(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(999)
            self.layer_0 = nn.Linear(3, 5)

    old, new = OldModel(), NewModel()
    report = transfer_compatible_weights(old, new)

    assert {"layer_0.weight", "layer_0.bias"}.issubset(set(report.transferred))
    assert set(report.dropped_from_old) == {"layer_1.weight", "layer_1.bias"}


def test_hypergraph_conv_transfers_through_nested_linear():
    # The stub HypergraphConv wraps a nn.Linear — the nested
    # parameter keys ("linear.weight", "linear.bias") should transfer.
    torch.manual_seed(0)
    old = HypergraphConv(d_in=3, d_out=5)
    torch.manual_seed(999)
    new = HypergraphConv(d_in=3, d_out=5)

    report = transfer_compatible_weights(old, new)
    assert set(report.transferred) == {"linear.weight", "linear.bias"}
    assert torch.equal(old.linear.weight, new.linear.weight)
    assert torch.equal(old.linear.bias,   new.linear.bias)


# ─── reinfer_structure_and_rebuild ──────────────────────────────────


def test_reinfer_factory_returns_new_model_with_transferred_weights():
    torch.manual_seed(0)
    old = nn.Linear(3, 5)

    def factory():
        torch.manual_seed(999)
        return nn.Linear(3, 5)

    new, report = reinfer_structure_and_rebuild(old, factory)
    assert new is not old, "factory must return a fresh instance"
    assert isinstance(report, TransferReport)
    assert torch.equal(old.weight, new.weight)
    assert torch.equal(old.bias, new.bias)


def test_reinfer_with_proposal_annotates_report():
    torch.manual_seed(0)
    old = nn.Linear(3, 5)

    def factory():
        return nn.Linear(3, 5)

    proposal = SplitProposal(
        target_scope="simple_net",
        cluster_a=("x", "h"),
        cluster_b=("y", "layer_1"),
        edge_assignments=(("flow_0", "A"), ("flow_1", "cross")),
        n_cross_edges=1,
        inertia=0.6667,
    )
    _, report = reinfer_structure_and_rebuild(old, factory, proposal=proposal)
    assert getattr(report, "proposal_scope") == "simple_net"
    assert getattr(report, "proposal_n_cross_edges") == 1


# ─── Proposal JSON loader ────────────────────────────────────────────


SAMPLE_JSON = """{
  "target_scope": "simple_net",
  "cluster_a": ["x", "h", "layer_0"],
  "cluster_b": ["y", "layer_1"],
  "edge_assignments": [
    {"edge": "flow_0", "cluster": "A"},
    {"edge": "flow_1", "cluster": "cross"}
  ],
  "n_cross_edges": 1,
  "inertia": 0.6666666666666667
}
"""


def test_load_proposal_from_json_string():
    proposal = load_proposal(SAMPLE_JSON)
    assert proposal.target_scope == "simple_net"
    assert proposal.cluster_a == ("x", "h", "layer_0")
    assert proposal.cluster_b == ("y", "layer_1")
    assert proposal.n_cross_edges == 1
    assert 0.666 < proposal.inertia < 0.667


def test_load_proposal_from_dict():
    data = json.loads(SAMPLE_JSON)
    proposal = load_proposal(data)
    assert proposal.target_scope == "simple_net"


def test_load_proposal_from_path(tmp_path):
    path = tmp_path / "proposal.json"
    path.write_text(SAMPLE_JSON, encoding="utf-8")
    # Accept both str and pathlib.Path.
    for arg in (str(path), path):
        proposal = load_proposal(arg)
        assert proposal.target_scope == "simple_net"


def test_proposal_cluster_lookups():
    proposal = load_proposal(SAMPLE_JSON)
    assert proposal.vertex_cluster("x") == "A"
    assert proposal.vertex_cluster("y") == "B"
    assert proposal.vertex_cluster("missing") is None
    assert proposal.edge_cluster("flow_0") == "A"
    assert proposal.edge_cluster("flow_1") == "cross"
    assert proposal.edge_cluster("missing") is None
    assert proposal.cross_edges == ("flow_1",)
    assert proposal.cluster_a_edges == ("flow_0",)


def test_load_proposal_missing_key_raises():
    bad = json.loads(SAMPLE_JSON)
    del bad["inertia"]
    try:
        load_proposal(bad)
    except ValueError as e:
        assert "inertia" in str(e)
    else:
        raise AssertionError("expected ValueError for missing key")
