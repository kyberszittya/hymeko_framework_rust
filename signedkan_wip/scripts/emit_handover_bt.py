"""Emit a BehaviorTree.CPP XML file from a HymeKo task description.

Reads a .hymeko file that uses the `meta_task` vocabulary
(actions, composites, conditions, coordination primitives) and
emits a BehaviorTree.CPP-compatible XML behavior tree.

Status:
  * MVP supporting: a/move_to, a/grip_open, a/grip_close,
    c/sequence, c/parallel, c/entry, coord/handover (expanded
    inline to its 3-step subtree).
  * Pending: a/joint_move, a/wait, a/apply_force, c/fallback,
    c/loop, c/invert, cond/* (treated as comments for now),
    coord/synchronize.

Designed as a stand-alone Python script so the BT.CPP path can
ship without touching the core `hymeko emit` pipeline (which is
template-driven and CORE.YAML-gated). The next-tier evolution is
to fold this logic into `transforms/bt/` as a proper Handlebars
template + queries.hymeko pair — same as `transforms/sdf/` — once
the design is validated end-to-end.

Usage:
    python -m signedkan_wip.scripts.emit_handover_bt \
        data/robotics/sim/dual_fanuc/handover_task.hymeko \
        --out data/robotics/sim/dual_fanuc/handover_task.bt.xml
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import hymeko


# ─── HymeKo IR helpers ───────────────────────────────────────────────


def _bases_signature(item: dict[str, Any]) -> str:
    """Canonical string for an item's base type chain.

    E.g. an action bound by `bases=[a/move_to]` returns "a/move_to".
    Items can have multiple bases; we join with '|' for the
    multi-base case (the task-level handover_task node, which
    declares multiple namespace aliases as its bases).
    """
    parts = ["/".join(r["path"]) for r in item.get("bases") or []]
    return "|".join(parts)


def _arc_refs(item: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract (sign, target_name) pairs from an edge's arc body.

    Edges in the HymeKo IR carry their hyperedge incidence as a
    list of arcs in their body, each with a `refs` list of signed
    references. We collapse each arc's first ref into a flat
    sequence for the BT.CPP emitter (which doesn't model
    hypergraph-style multi-incidence).
    """
    out: list[tuple[str, str]] = []
    for sub in item.get("body") or []:
        if sub.get("kind") != "arc":
            continue
        for r in sub.get("refs") or []:
            out.append((r["sign"], "/".join(r["path"])))
    return out


# ─── BT.CPP XML emission ─────────────────────────────────────────────


class BTEmitter:
    """Walks the HymeKo task IR and produces BehaviorTree.CPP XML.

    The walker is driven by the entry node: starting from the
    unique `c/entry` instance, it follows the entry's body
    reference (the first `+`-signed arc) and recursively expands
    composites and actions.
    """

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.by_name: dict[str, dict[str, Any]] = {
            it["name"]: it for it in items if it.get("name")
        }
        self.unsupported: list[str] = []

    def emit(self, tree_id: str = "HandoverTask") -> ET.ElementTree:
        entry = self._find_entry()
        body_arcs = _arc_refs(entry)
        if not body_arcs:
            raise ValueError(
                f"entry node {entry.get('name')} has no body arc"
            )
        # The first arc is the body composite; remaining are agents.
        body_name = body_arcs[0][1]
        agents = [name for sign, name in body_arcs[1:] if sign == "-"]

        root = ET.Element("root", {"BTCPP_format": "4"})
        bt = ET.SubElement(root, "BehaviorTree", {"ID": tree_id})
        bt.append(self._emit_item(self.by_name[body_name]))

        # Agent declaration as an XML comment for downstream tooling.
        if agents:
            comment = ET.Comment(
                f"agents: {', '.join(agents)}  (multi-arm cell from "
                f"dual_fanuc/world.sdf)"
            )
            bt.insert(0, comment)

        if self.unsupported:
            warn = ET.Comment(
                "unsupported IR types (emitted as <Action ID='TODO'>): "
                + ", ".join(sorted(set(self.unsupported)))
            )
            root.insert(0, warn)

        return ET.ElementTree(root)

    def _find_entry(self) -> dict[str, Any]:
        for item in self.by_name.values():
            if item.get("kind") != "edge":
                continue
            if _bases_signature(item) == "c/entry":
                return item
        raise ValueError(
            "no c/entry node found in task description; "
            "task must declare exactly one @entry-typed composite"
        )

    # Composite dispatch.

    def _emit_item(self, item: dict[str, Any]) -> ET.Element:
        sig = _bases_signature(item)
        # Composites.
        if sig == "c/sequence":
            return self._emit_composite(item, "Sequence")
        if sig == "c/parallel":
            return self._emit_parallel(item)
        if sig == "c/fallback":
            return self._emit_composite(item, "Fallback")
        # Actions.
        if sig == "a/move_to":
            return self._emit_action(item, "MoveTo")
        if sig == "a/grip_open":
            return self._emit_action(item, "GripOpen")
        if sig == "a/grip_close":
            return self._emit_action(item, "GripClose")
        if sig == "a/joint_move":
            return self._emit_action(item, "JointMove")
        if sig == "a/wait":
            return self._emit_action(item, "Wait")
        if sig == "a/noop":
            return ET.Element("AlwaysSuccess")
        # Coordination.
        if sig == "coord/handover":
            return self._emit_handover(item)
        if sig == "coord/synchronize":
            return self._emit_synchronize(item)
        # Unknown.
        self.unsupported.append(sig)
        el = ET.Element("Action", {"ID": "TODO", "raw_kind": sig})
        el.set("name", item.get("name") or "")
        return el

    def _emit_composite(
        self, item: dict[str, Any], xml_tag: str,
    ) -> ET.Element:
        el = ET.Element(xml_tag, {"name": item.get("name") or xml_tag.lower()})
        for sign, target in _arc_refs(item):
            if sign != "+":
                # `-` arcs on a composite are agent annotations, not children.
                continue
            child = self.by_name.get(target)
            if child is None:
                self.unsupported.append(f"unresolved/{target}")
                continue
            el.append(self._emit_item(child))
        return el

    def _emit_parallel(self, item: dict[str, Any]) -> ET.Element:
        # BT.CPP 4 Parallel takes a success_count / failure_count.
        # Map meta_task's policy field:
        #   "all"            → success_count = N (all must succeed)
        #   "any"            → success_count = 1
        #   "any_else_abort" → success_count = 1, failure_count = N
        children_refs = [t for s, t in _arc_refs(item) if s == "+"]
        n = len(children_refs)
        policy = None
        # Field values are stored as raw items in body too.
        for sub in item.get("body") or []:
            if sub.get("kind") == "node" and sub.get("name") == "policy":
                v = sub.get("value")
                if isinstance(v, str):
                    policy = v
        if policy in (None, "all"):
            attrs = {"success_count": str(n), "failure_count": "1"}
        elif policy == "any":
            attrs = {"success_count": "1", "failure_count": str(n)}
        elif policy == "any_else_abort":
            attrs = {"success_count": "1", "failure_count": "1"}
        else:
            attrs = {"success_count": str(n), "failure_count": "1"}
        attrs["name"] = item.get("name") or "parallel"
        el = ET.Element("Parallel", attrs)
        for target in children_refs:
            child = self.by_name.get(target)
            if child is None:
                self.unsupported.append(f"unresolved/{target}")
                continue
            el.append(self._emit_item(child))
        return el

    def _emit_action(
        self, item: dict[str, Any], action_id: str,
    ) -> ET.Element:
        """Map an action edge to a BT.CPP <Action ID="..." actor="..." target="..."/>.

        The hyperedge convention from meta_task.hymeko:
          + actor (always the first +-sign reference)
          - target (the -sign reference, optional for grip_open)
        """
        actor = None
        target = None
        for sign, ref in _arc_refs(item):
            if sign == "+" and actor is None:
                actor = ref
            elif sign == "-" and target is None:
                target = ref
        attrs = {"ID": action_id, "name": item.get("name") or action_id.lower()}
        if actor is not None:
            attrs["actor"] = actor
        if target is not None:
            attrs["target"] = target
        return ET.Element("Action", attrs)

    def _emit_handover(self, item: dict[str, Any]) -> ET.Element:
        """Expand coord/handover to its 3-step subtree per meta_task.hymeko.

        Hyperedge convention:
          (+ from_gripper, - to_gripper, + object)

        Expansion:
          Sequence:
            1. <Action ID="GripClose" actor=to_gripper object=object/>
            2. <SyncPoint name="handover_grasp"/>
            3. <Action ID="GripOpen" actor=from_gripper/>
        """
        refs = _arc_refs(item)
        # First + is from_gripper, second + is the object, single - is to_gripper.
        from_g = None
        to_g = None
        obj = None
        for sign, ref in refs:
            if sign == "+":
                if from_g is None:
                    from_g = ref
                else:
                    obj = ref
            elif sign == "-":
                to_g = ref
        name = item.get("name") or "handover"
        seq = ET.Element("Sequence", {"name": name})
        if to_g and obj:
            close = ET.SubElement(
                seq, "Action",
                {"ID": "GripClose", "actor": to_g, "object": obj,
                 "name": f"{name}_close_right"},
            )
        sync = ET.SubElement(
            seq, "SyncPoint",
            {"name": f"{name}_grasp_sync"},
        )
        if from_g:
            opn = ET.SubElement(
                seq, "Action",
                {"ID": "GripOpen", "actor": from_g,
                 "name": f"{name}_open_left"},
            )
        return seq

    def _emit_synchronize(self, item: dict[str, Any]) -> ET.Element:
        refs = [t for _s, t in _arc_refs(item)]
        attrs = {"name": item.get("name") or "synchronize"}
        if refs:
            attrs["points"] = ", ".join(refs)
        return ET.Element("SyncPoint", attrs)


# ─── ET pretty-printer (Python ≥3.9) ─────────────────────────────────


def _pretty(tree: ET.ElementTree) -> str:
    ET.indent(tree, space="  ")
    head = '<?xml version="1.0" encoding="UTF-8"?>\n'
    body = ET.tostring(tree.getroot(), encoding="unicode")
    return head + body + "\n"


# ─── CLI ─────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "source", type=Path,
        help="Path to a .hymeko task description (e.g. "
             "data/robotics/sim/dual_fanuc/handover_task.hymeko)",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output .bt.xml path. Defaults to <source-stem>.bt.xml "
             "next to the source file.",
    )
    p.add_argument(
        "--tree-id", default="HandoverTask",
        help="BehaviorTree ID attribute. Default: HandoverTask.",
    )
    args = p.parse_args()

    src_text = args.source.read_text()
    ast = hymeko.parse_hymeko_rs(src_text)
    # The task description is the single top-level node; its body
    # holds the actions / composites / conditions.
    top = ast["items"]
    if len(top) != 1 or top[0].get("kind") != "node":
        print(
            f"warning: expected one top-level description node, got "
            f"{len(top)} items of kinds "
            f"{[it.get('kind') for it in top]}", file=sys.stderr,
        )
    task_node = top[0]
    body_items = task_node.get("body") or []

    emitter = BTEmitter(body_items)
    tree = emitter.emit(tree_id=args.tree_id)

    out_path = args.out
    if out_path is None:
        out_path = args.source.with_suffix(".bt.xml")
    out_path.write_text(_pretty(tree))
    print(f"wrote {out_path}")
    if emitter.unsupported:
        print(
            f"  note: {len(emitter.unsupported)} unsupported IR refs "
            f"emitted as TODO: {sorted(set(emitter.unsupported))}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
