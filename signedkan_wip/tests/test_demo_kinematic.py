"""Tests for the kinematic demo loader + registry.

Covers: registry parse + schema, URDF → bundle round-trip on a real
in-repo URDF, topology-signature heuristic on chain/tree fixtures,
graceful degradation when the registry file is missing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from signedkan_wip.src.demo.kinematic import (
    DEFAULT_URDF_REGISTRY, KinematicBundle, URDFEntry,
    find_urdf_by_id, load_urdf_bundle, load_urdf_registry,
    topology_signature, urdf_dropdown_choices,
)


def test_default_registry_loads():
    entries = load_urdf_registry()
    assert len(entries) >= 1, "kinematic_models.yaml is empty / unparseable"
    ids = [e.id for e in entries]
    assert len(set(ids)) == len(ids), f"duplicate URDF ids: {ids}"


def test_every_entry_resolves_or_marked_missing():
    """A URDF that ships with the repo must resolve. The catalogue is
    only allowed to point at non-existent files for stretch entries
    (drchubo, WAM in .hymeko-only form), and even those should fail
    visibly rather than silently."""
    for e in load_urdf_registry():
        assert isinstance(e.path, Path)
        # Every entry id from the v0 catalogue is shipped — fail if any
        # of them goes missing in a later edit.
        assert e.available, f"{e.id} resolves to {e.path} but file is missing"


def test_known_entries_present():
    ids = {e.id for e in load_urdf_registry()}
    for required in ("moveo", "chain_5", "chain_10", "humanoid_f0",
                     "tree_10", "mini_arm",
                     "four_bar", "stewart", "delta_3rrr",
                     "serial_4", "serial_7"):
        assert required in ids, f"missing required URDF entry: {required}"


def test_parallel_mechanisms_produce_expected_cycle_signatures():
    """The headline demo cases — these signatures are what makes the
    cycle-arity bar chart light up. Pin them so future fixture edits
    don't quietly break the Stewart-spike-at-k=6 story."""
    entries = load_urdf_registry()

    four = load_urdf_bundle(
        find_urdf_by_id(entries, "four_bar").path, name="four_bar")
    assert four.cycle_counts.get(4) == 1, (
        f"4-bar should have exactly 1 cycle at k=4, got {four.cycle_counts}"
    )
    assert four.cycle_counts.get(3, 0) == 0
    assert four.cycle_counts.get(6, 0) == 0
    assert topology_signature(four) == "4-bar / planar parallel"

    stewart = load_urdf_bundle(
        find_urdf_by_id(entries, "stewart").path, name="stewart")
    assert stewart.cycle_counts.get(6, 0) >= 6, (
        f"Stewart platform should have many k=6 cycles, got {stewart.cycle_counts}"
    )
    assert topology_signature(stewart) == "Stewart / delta / spatial parallel"

    delta = load_urdf_bundle(
        find_urdf_by_id(entries, "delta_3rrr").path, name="delta_3rrr")
    assert delta.cycle_counts.get(6, 0) >= 1
    assert topology_signature(delta) == "Stewart / delta / spatial parallel"


def test_load_urdf_bundle_on_mini_arm():
    """mini_arm.urdf — the trivial 2-link fixture — should round-trip
    cleanly: 2 links, 1 revolute joint, no cycles."""
    entries = load_urdf_registry()
    entry = find_urdf_by_id(entries, "mini_arm")
    assert entry is not None
    b = load_urdf_bundle(entry.path, name=entry.id)
    assert isinstance(b, KinematicBundle)
    assert b.n_links == 2
    assert b.n_joints == 1
    assert b.n_revolute == 1
    assert b.n_prismatic == 0
    assert all(v == 0 for v in b.cycle_counts.values())
    assert b.is_open_chain
    # Balance summary should be well-formed.
    s = b.balance_summary()
    assert s["n_edges"] == 1 and s["n_pos"] == 1 and s["n_neg"] == 0


def test_topology_signature_distinguishes_chain_from_tree():
    """A serial chain has max-degree 2 (→ 'open chain'); a tree URDF
    has some node with degree ≥ 3 (→ 'tree'). Without cycles they
    must not collapse to the same label."""
    entries = load_urdf_registry()
    chain = load_urdf_bundle(
        find_urdf_by_id(entries, "chain_10").path, name="chain_10")
    tree = load_urdf_bundle(
        find_urdf_by_id(entries, "tree_10").path, name="tree_10")
    assert topology_signature(chain) == "open chain"
    assert topology_signature(tree) == "tree"


def test_topology_signature_handles_empty_graph():
    """Degenerate edge case: an empty SignedGraph should not crash
    the signature heuristic."""
    import numpy as np
    from signedkan_wip.src.datasets import SignedGraph
    g = SignedGraph(edges=np.zeros((0, 2), dtype=np.int64),
                     signs=np.zeros((0,), dtype=np.int8),
                     n_nodes=0)
    b = KinematicBundle(
        name="empty", urdf_path=Path("/dev/null"),
        graph=g, link_names=[], joints=[],
        cycle_counts={3: 0, 4: 0, 5: 0, 6: 0},
    )
    # Empty graph collapses to "open chain" per the heuristic (no cycles,
    # max degree 0). Acceptable; just don't crash.
    assert topology_signature(b) == "open chain"


def test_dropdown_choices_partition_by_availability(tmp_path):
    fake = tmp_path / "real.urdf"
    fake.write_text("<robot/>")
    e_avail = URDFEntry(id="zz", label="Z real", path=fake)
    e_miss = URDFEntry(id="aa", label="A missing", path=tmp_path / "nope.urdf")
    pairs = urdf_dropdown_choices([e_miss, e_avail])
    labels = [p[0] for p in pairs]
    assert labels == ["Z real", "[MISSING] A missing"]


def test_find_urdf_by_id_returns_entry_or_none():
    entries = load_urdf_registry()
    assert find_urdf_by_id(entries, "moveo") is not None
    assert find_urdf_by_id(entries, "does_not_exist") is None


def test_missing_registry_returns_empty(tmp_path):
    entries = load_urdf_registry(tmp_path / "nope.yaml")
    assert entries == []


def test_default_registry_path_is_packaged():
    assert DEFAULT_URDF_REGISTRY.is_file(), (
        f"default URDF registry not packaged: {DEFAULT_URDF_REGISTRY}"
    )
