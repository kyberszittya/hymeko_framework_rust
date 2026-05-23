"""Tests for the dual-FANUC BT.CPP emitter.

Pins the BT.XML output structure against handover_task.hymeko —
catches IR-parser drift, emitter-mapping bugs, and accidental
schema changes in BehaviorTree.CPP attrs.

Run:
    pytest signedkan_wip/tests/test_emit_handover_bt.py
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from signedkan_wip.scripts.emit_handover_bt import BTEmitter
import hymeko


HANDOVER = Path(__file__).resolve().parents[2] / (
    "data/robotics/sim/dual_fanuc/handover_task.hymeko"
)


@pytest.fixture(scope="module")
def emitted_tree():
    src = HANDOVER.read_text()
    ast = hymeko.parse_hymeko_rs(src)
    body = ast["items"][0]["body"]
    emitter = BTEmitter(body)
    return emitter.emit(tree_id="HandoverTask"), emitter


def test_tree_has_root_with_btcpp_format(emitted_tree):
    tree, _ = emitted_tree
    root = tree.getroot()
    assert root.tag == "root"
    assert root.attrib.get("BTCPP_format") == "4"


def test_tree_has_one_behavior_tree_with_id(emitted_tree):
    tree, _ = emitted_tree
    bts = tree.getroot().findall("BehaviorTree")
    assert len(bts) == 1
    assert bts[0].attrib.get("ID") == "HandoverTask"


def test_top_composite_is_sequence(emitted_tree):
    tree, _ = emitted_tree
    bt = tree.getroot().find("BehaviorTree")
    seq = bt.find("Sequence")
    assert seq is not None
    assert seq.attrib.get("name") == "pick_handover_place"


def test_sequence_has_seven_children(emitted_tree):
    """The pick-handover-place sequence in handover_task.hymeko declares
    seven phases: move_left_to_input, grip_left_close, phase2_parallel,
    do_handover, move_right_to_output, grip_right_open, phase5_parallel.
    """
    tree, _ = emitted_tree
    seq = tree.getroot().find("BehaviorTree").find("Sequence")
    children = list(seq)
    assert len(children) == 7, [c.tag for c in children]


def test_actions_have_actor_attr(emitted_tree):
    """Every <Action> emitted from an a/* hyperedge must carry the actor
    (the first +-signed reference) as an XML attribute."""
    tree, _ = emitted_tree
    for action in tree.getroot().iter("Action"):
        if action.attrib.get("ID") == "TODO":
            continue
        assert "actor" in action.attrib, ET.tostring(action, encoding="unicode")


def test_parallel_uses_btcpp4_attrs(emitted_tree):
    tree, _ = emitted_tree
    parallels = list(tree.getroot().iter("Parallel"))
    assert len(parallels) == 2  # phase2_parallel + phase5_parallel
    for p in parallels:
        assert "success_count" in p.attrib
        assert "failure_count" in p.attrib
        # Two-child parallels with policy="all": success_count=2, failure_count=1.
        assert p.attrib["success_count"] == "2"
        assert p.attrib["failure_count"] == "1"


def test_handover_expands_to_three_step_subtree(emitted_tree):
    """coord/handover expands to:
       Sequence:
         <Action ID="GripClose" actor=to_gripper object=part />
         <SyncPoint />
         <Action ID="GripOpen" actor=from_gripper />
    """
    tree, _ = emitted_tree
    # Find the Sequence named "do_handover" (the expansion target).
    do_handover = None
    for seq in tree.getroot().iter("Sequence"):
        if seq.attrib.get("name") == "do_handover":
            do_handover = seq
            break
    assert do_handover is not None
    children = list(do_handover)
    assert len(children) == 3
    assert children[0].tag == "Action"
    assert children[0].attrib["ID"] == "GripClose"
    assert children[0].attrib["actor"] == "fanuc_right"
    assert children[0].attrib["object"] == "part"
    assert children[1].tag == "SyncPoint"
    assert children[2].tag == "Action"
    assert children[2].attrib["ID"] == "GripOpen"
    assert children[2].attrib["actor"] == "fanuc_left"


def test_no_unsupported_ir_types(emitted_tree):
    """The handover_task.hymeko fixture should hit only the MVP-supported
    IR types (a/move_to, a/grip_*, c/sequence, c/parallel, c/entry,
    coord/handover). If new unsupported kinds appear, this test fails
    loudly so the emitter is extended before the regression lands."""
    _, emitter = emitted_tree
    # cond/postcondition is in the file but is currently *unused* — it
    # references do_handover but is not a child of the entry composite,
    # so the walker never visits it. If walking ever covers conditions,
    # update this test to expect those.
    assert emitter.unsupported == [], (
        f"unexpected unsupported IR types: {emitter.unsupported}"
    )


def test_agents_comment_present(emitted_tree):
    tree, _ = emitted_tree
    bt = tree.getroot().find("BehaviorTree")
    # ET stores Comments as a special element with .tag == ET.Comment;
    # iter through children of <BehaviorTree>.
    comment_text = None
    for child in bt:
        if not isinstance(child.tag, str):  # comment
            comment_text = child.text
            break
    assert comment_text is not None
    assert "fanuc_left" in comment_text
    assert "fanuc_right" in comment_text


def test_all_six_move_to_actions_present(emitted_tree):
    """The fixture declares six a/move_to edges:
       move_left_to_input, move_left_to_handover, move_right_to_handover,
       move_right_to_output, move_left_home, move_right_home.
    """
    tree, _ = emitted_tree
    move_to_names = sorted(
        a.attrib["name"] for a in tree.getroot().iter("Action")
        if a.attrib.get("ID") == "MoveTo"
    )
    assert move_to_names == sorted([
        "move_left_to_input",
        "move_left_to_handover",
        "move_right_to_handover",
        "move_right_to_output",
        "move_left_home",
        "move_right_home",
    ])
