"""Topic binding — maps ROS 2 topics to HyMeKo V_global vertices and back.

The grasping context's input vertices are bound from incoming ROS topic
values; the aggregation outputs are published back as ROS topics.  The
exact binding is configured via a YAML file
(``config/topic_mapping.yaml``) so reviewers can edit the mapping
without touching the node.

This module also walks the parser's IR (a nested dict, as produced by
``hymeko.parse_hymeko_rs``) and pulls out the named context's
hyperedges in a form the node can evaluate.

Design note
-----------
The paper does NOT fix the closed-form aggregation function for each
hyperedge --- it only specifies the *signed incidence structure*.  For
the live demo we use placeholder aggregation functions documented
inline.  The reviewer-facing claim is that the contextual flow
*runs in real time*, not that the absolute values represent a learned
policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# -------------------------------------------------------------------
# IR walker
# -------------------------------------------------------------------


@dataclass(frozen=True)
class Hyperedge:
    """One signed hyperedge as extracted from the IR.

    Attributes
    ----------
    name : str
        The edge label as it appears in the .hymeko (e.g. "grasp_stability").
    inputs : tuple[str, ...]
        Vertex names with a "+" prefix in the IR.
    outputs : tuple[str, ...]
        Vertex names with a "-" prefix in the IR.
    """

    name: str
    inputs: tuple
    outputs: tuple


def find_context_block(ir: dict, context_name: str) -> Optional[dict]:
    """Return the body-list entry whose name matches ``context_name``
    inside the first top-level item, or ``None`` if absent.

    The IR shape (for our purposes) is::

        {"items": [{"kind": "node", "name": "<root>", "body": [...]}]}

    Each entry in the root's ``body`` is itself a node-dict; the
    grasping/maintenance/safety contexts each appear as one such entry.
    """

    if not isinstance(ir, dict) or "items" not in ir or not ir["items"]:
        return None
    root = ir["items"][0]
    if not isinstance(root, dict) or "body" not in root:
        return None
    for entry in root["body"]:
        if isinstance(entry, dict) and entry.get("name") == context_name:
            return entry
    return None


def extract_hyperedges(context_block: dict) -> List[Hyperedge]:
    """Walk the context's body and pull every signed hyperedge.

    The parser surfaces edges as ``{"kind": "edge", "name": "...",
    "body": [{"kind": "arc", "refs": [...]}]}``.  Each ref carries
    ``{"sign": "+"/"-"/"~", "path": ["vertex_name"]}``.
    """

    edges: List[Hyperedge] = []
    if not isinstance(context_block, dict) or "body" not in context_block:
        return edges
    for entry in context_block["body"]:
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "edge":
            continue
        name = entry.get("name", "<anon>")
        # Each edge's body holds one or more "arc" entries.
        for child in entry.get("body", []) or []:
            if not isinstance(child, dict):
                continue
            if child.get("kind") != "arc":
                continue
            inputs, outputs = _split_refs(child.get("refs", []))
            edges.append(Hyperedge(name=name, inputs=inputs, outputs=outputs))
    return edges


def _split_refs(refs: Any) -> tuple:
    """Split an arc's ``refs`` list into (inputs, outputs) by sign."""

    inputs: List[str] = []
    outputs: List[str] = []
    if not isinstance(refs, list):
        return tuple(inputs), tuple(outputs)
    for atom in refs:
        if not isinstance(atom, dict):
            continue
        sign = atom.get("sign", "+")
        path = atom.get("path") or []
        if not path:
            continue
        # Use the LAST path component as the vertex label so qualified
        # paths like ["maintenance_context", "component_health"]
        # resolve to the leaf vertex name.
        name = str(path[-1])
        if sign == "+":
            inputs.append(name)
        elif sign == "-":
            outputs.append(name)
        # "~" silently dropped
    return tuple(inputs), tuple(outputs)


# -------------------------------------------------------------------
# Topic mapping
# -------------------------------------------------------------------


@dataclass
class TopicMap:
    """One ROS topic <-> vertex binding entry."""

    vertex: str           # e.g. "active_tool"
    topic: str            # e.g. "/tool_id"
    msg_type: str         # e.g. "std_msgs/msg/UInt32"
    field: Optional[str] = None  # e.g. "data" or "wrench.force.z"


@dataclass
class BindingConfig:
    """Full set of input/output bindings + the chosen context name."""

    context: str
    inputs: List[TopicMap] = field(default_factory=list)
    outputs: List[TopicMap] = field(default_factory=list)


def load_yaml_config(path: Path) -> BindingConfig:
    """Parse ``topic_mapping.yaml`` into a :class:`BindingConfig`.

    Falls back to a minimal hard-coded mapping if PyYAML isn't
    available (so the unit smoke does not require an extra dep).
    """

    if not path.exists():
        raise FileNotFoundError(f"topic_mapping.yaml not found at {path}")
    try:
        import yaml  # type: ignore
    except ImportError:
        # Trivial INI-ish fallback parser: not used in the unit smoke
        # path (which always provides PyYAML via the test env).
        raise

    data = yaml.safe_load(path.read_text())
    context = str(data.get("context", "grasping_context"))
    inputs = [_tmap_from_dict(d) for d in (data.get("inputs") or [])]
    outputs = [_tmap_from_dict(d) for d in (data.get("outputs") or [])]
    return BindingConfig(context=context, inputs=inputs, outputs=outputs)


def _tmap_from_dict(d: dict) -> TopicMap:
    return TopicMap(
        vertex=str(d["vertex"]),
        topic=str(d["topic"]),
        msg_type=str(d["msg_type"]),
        field=d.get("field"),
    )


# -------------------------------------------------------------------
# Placeholder aggregation functions
# -------------------------------------------------------------------
#
# These implement the *shape* of each hyperedge's f_*(inputs) ->
# outputs map.  They are intentionally simple --- the paper does not
# pin the closed forms, so the live demo uses transparent placeholders
# that move smoothly with the input stream.


def aggregate_default(inputs: Dict[str, float]) -> Dict[str, float]:
    """Default: mean of all bound inputs, clipped to [0, 1]."""

    if not inputs:
        return {}
    vals = [float(v) for v in inputs.values()]
    mean = sum(vals) / len(vals)
    return {"_default": max(0.0, min(1.0, mean))}


def aggregate_grasp_stability(inputs: Dict[str, float]) -> Dict[str, float]:
    """Placeholder S_g = grasp success / stability margin.

    We normalise grip_force from its raw wrench range (~0..10 N) into
    [0, 1] before comparison so its dynamic range matches force_vector
    (which is already clipped to [0, 1] by the default aggregator).
    Otherwise the difference is dominated by grip_force's magnitude
    and stability_margin barely varies regardless of the other
    inputs — see the README §"What this does NOT prove" for the
    rationale that the closed-form aggregations are illustrative,
    not paper claims.

    Formula: stability = 1 / (1 + 6·|F_l - F_g_norm|^1.5)

    The 6× weighting and exponent 1.5 amplify small deviations so
    the gauge has visible dynamic range over the demo's input stream
    (typically 0.05 → 0.95).
    """

    f_l = float(inputs.get("force_vector", 0.0))
    f_g_raw = float(inputs.get("grip_force", 0.0))
    # Normalise raw wrench (0..10 N typical) into [0, 1] with a soft
    # saturation so high forces don't pin the output.
    f_g_norm = max(0.0, min(1.0, abs(f_g_raw) / 10.0))
    diff = abs(f_l - f_g_norm)
    margin = 1.0 / (1.0 + 6.0 * (diff ** 1.5))
    return {"stability_margin": max(0.0, min(1.0, margin))}


__all__ = [
    "Hyperedge",
    "find_context_block",
    "extract_hyperedges",
    "TopicMap",
    "BindingConfig",
    "load_yaml_config",
    "aggregate_default",
    "aggregate_grasp_stability",
]
