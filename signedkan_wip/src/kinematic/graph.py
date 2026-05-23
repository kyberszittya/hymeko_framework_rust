"""Kinematic graph adapter — URDF → SignedGraph for HSiKAN.

A robot kinematic structure naturally maps to a signed graph:

    Vertices   = links (rigid bodies)
    Edges      = movable joints (revolute / prismatic / continuous /
                  planar / floating; ``fixed`` joints encode rigid
                  unions and are skipped by default)
    Sign       = joint type binary:
                    +1  rotational (revolute, continuous, planar*)
                    -1  translational (prismatic)
                 (mixed-DOF joints like floating get ±1 by convention)

Cycles in the resulting graph correspond to closed kinematic loops:
parallel manipulators, four-bar linkages, Stewart platforms, delta
robots all produce k=4–6 cycles. Davis weak balance ("parity of
negative-rotation joints around the loop") becomes a structural
constraint on loop closure.

Usage
-----

>>> from signedkan_wip.src.kinematic import urdf_to_signed_graph
>>> g, link_names, joint_names = urdf_to_signed_graph("mini_arm.urdf")
>>> g.stats()
{'n_nodes': 4, 'n_edges': 3, 'n_pos': 3, 'n_neg': 0, 'pos_frac': 1.0}

Then run the existing HSiKAN cycle enumerator + αₖ-mixing on top of
``g`` to discover whether the model autonomously selects k=4 cycles
for parallel mechanisms or k=3 for serial ones.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..datasets import SignedGraph


# Joint type → sign convention.
# Mixed-DOF joints (floating, planar) default to +1; override if the
# downstream task wants a different binary.
JOINT_SIGN: dict[str, int] = {
    "revolute": +1,
    "continuous": +1,
    "planar": +1,
    "floating": +1,
    "prismatic": -1,
    # "fixed" intentionally omitted — handled separately.
}


@dataclass
class KinematicJoint:
    name: str
    parent_link: str
    child_link: str
    joint_type: str
    sign: int   # ±1 per JOINT_SIGN
    axis: tuple[float, float, float] | None = None


def parse_urdf(path: str | Path) -> tuple[list[str], list[KinematicJoint]]:
    """Parse a URDF file. Returns (link_names, joints).

    ``joints`` excludes ``fixed`` joints by default — those represent
    rigid unions and don't add structural mobility.
    """
    tree = ET.parse(str(path))
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"expected <robot> root, got <{root.tag}>")

    # Collect links.
    link_names: list[str] = []
    for link_el in root.findall("link"):
        name = link_el.get("name")
        if name:
            link_names.append(name)

    # Collect joints (movable only).
    joints: list[KinematicJoint] = []
    for j_el in root.findall("joint"):
        jname = j_el.get("name", "?")
        jtype = j_el.get("type", "fixed").strip().lower()
        if jtype == "fixed":
            continue
        if jtype not in JOINT_SIGN:
            # Unknown joint type: skip with a hint (could also raise).
            continue
        parent_el = j_el.find("parent")
        child_el = j_el.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link")
        child = child_el.get("link")
        if parent is None or child is None:
            continue
        axis_el = j_el.find("axis")
        axis = None
        if axis_el is not None and axis_el.get("xyz"):
            try:
                xyz = [float(x) for x in axis_el.get("xyz").split()]
                axis = (xyz[0], xyz[1], xyz[2]) if len(xyz) == 3 else None
            except ValueError:
                axis = None
        joints.append(KinematicJoint(
            name=jname, parent_link=parent, child_link=child,
            joint_type=jtype, sign=JOINT_SIGN[jtype], axis=axis,
        ))
    return link_names, joints


def urdf_to_signed_graph(
    path: str | Path,
    include_fixed: bool = False,
) -> tuple[SignedGraph, list[str], list[KinematicJoint]]:
    """Convert a URDF file to a SignedGraph for HSiKAN consumption.

    Returns ``(g, link_names, joints)``. ``g.edges[i]`` is
    ``(parent_idx, child_idx)`` for ``joints[i]``; ``g.signs[i]`` is
    ``+1``/``−1`` per the JOINT_SIGN mapping.

    Vertex ordering follows ``link_names``; the resulting indices match
    the order in ``g.edges``.
    """
    link_names, joints = parse_urdf(path)
    if include_fixed:
        # Re-parse to grab fixed joints as +1-sign edges (no Davis-
        # balance meaning for them, but the cycle topology is preserved).
        tree = ET.parse(str(path))
        root = tree.getroot()
        for j_el in root.findall("joint"):
            jtype = j_el.get("type", "fixed").strip().lower()
            if jtype != "fixed":
                continue
            p_el = j_el.find("parent"); c_el = j_el.find("child")
            if p_el is None or c_el is None:
                continue
            p = p_el.get("link"); c = c_el.get("link")
            if p is None or c is None:
                continue
            joints.append(KinematicJoint(
                name=j_el.get("name", "?"), parent_link=p, child_link=c,
                joint_type="fixed", sign=+1, axis=None,
            ))

    name_to_idx = {name: i for i, name in enumerate(link_names)}
    edges = []
    signs = []
    for j in joints:
        if j.parent_link not in name_to_idx or j.child_link not in name_to_idx:
            continue
        u = name_to_idx[j.parent_link]
        v = name_to_idx[j.child_link]
        edges.append((u, v))
        signs.append(j.sign)
    edges_arr = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
    signs_arr = np.array(signs, dtype=np.int8) if signs else np.zeros((0,), dtype=np.int8)
    g = SignedGraph(edges=edges_arr, signs=signs_arr,
                     n_nodes=len(link_names))
    return g, link_names, joints


def kinematic_loop_summary(g: SignedGraph, joint_names: list[KinematicJoint],
                            max_k: int = 6) -> dict:
    """Quick characterisation of a kinematic graph: cycle counts at
    each arity 3..max_k. Tells you if the mechanism is open-chain
    (no cycles) or has parallel/closed-loop structure.

    Useful as a sanity check before throwing HSiKAN at a robot.
    """
    from ..core.n_tuples import _enumerate_cycles_fast
    out = {
        "n_links": g.n_nodes,
        "n_joints": len(joint_names),
        "n_revolute": sum(1 for j in joint_names if j.sign == +1
                           and j.joint_type in ("revolute", "continuous")),
        "n_prismatic": sum(1 for j in joint_names if j.joint_type == "prismatic"),
        "cycles_per_arity": {},
    }
    for k in range(3, max_k + 1):
        try:
            c = _enumerate_cycles_fast(g, k=k, max_cycles=10_000)
            out["cycles_per_arity"][k] = len(c)
        except Exception:
            out["cycles_per_arity"][k] = -1
    return out


if __name__ == "__main__":
    # Smoke test on the repo's mini_arm.urdf.
    g, links, joints = urdf_to_signed_graph(
        Path(__file__).resolve().parent.parent.parent / "mini_arm.urdf",
    )
    print(f"mini_arm: {g.n_nodes} links, {len(joints)} movable joints")
    for j in joints:
        print(f"  {j.name:<14s} {j.parent_link}→{j.child_link}  "
              f"type={j.joint_type:<10s} sign={j.sign:+d}")
    summary = kinematic_loop_summary(g, joints, max_k=6)
    print(f"\nLoop summary: {summary}")
