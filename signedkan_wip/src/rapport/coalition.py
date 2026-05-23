"""Load an HRI coalition description from a .hymeko file.

The file is parsed by HyMeKo's native ``parse_hymeko_rs`` (no
grammar extension required — coalition descriptions reuse the
existing typed-node + body-properties syntax). The loader walks
the resulting IR and produces typed dataclasses:

    Agent
    Relation        (a signed dyadic edge)
    SigmaCycle      (a list of relation names whose σ-product is
                     monitored)
    Policy          (a trigger predicate + symbolic action name)
    Coalition       (the bundle of agents, relations, cycles,
                     policies for a single .hymeko coalition spec)

Plan: docs/plans/2026-05-18-rapport-coherence-demo-nagoya/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ─── Typed dataclasses ──────────────────────────────────────────────


@dataclass
class Agent:
    name: str
    kind: str  # "human" | "robot" (the base type's leaf segment)


@dataclass
class Relation:
    name: str
    kind: str       # "interpersonal" | "hri_relation"
    src: str        # agent name (from)
    dst: str        # agent name (to)
    sign: int       # initial sign in {-1, 0, +1}
    magnitude: float  # initial confidence


@dataclass
class SigmaCycle:
    name: str
    members: list[str]  # relation names in the cycle


@dataclass
class Policy:
    name: str
    condition: str
    action: str


@dataclass
class GzBinding:
    """Stage G — ROS 2 / GZ topic bindings for a single agent.

    ``camera_topic`` (added in Stage G' CV-2) is the gz topic for
    the agent's onboard camera, if any. Used by the vision sidecar
    to subscribe to image frames; bridged via ros_gz_image.
    """
    agent: str
    pose_topic: str
    cmd_vel_topic: str | None = None
    gaze_cmd_topic: str | None = None
    camera_topic: str | None = None


@dataclass
class ObservationThreshold:
    """Stage G — single-channel threshold for the GzObserver node.

    Reads:
        thr_distance: hri.observation_threshold {
            kind "distance_close";
            value 1.5;
        }
    """
    kind: str
    value: float


@dataclass
class VisionConfig:
    """Stage H / Stage D-3 — vision-sidecar detector selection.

    Declares which detector the vision sidecar node uses and the
    confidence threshold. Empty ``checkpoint`` for the "hsv_blob"
    backend; absolute or repo-relative path for the trained variants.
    """
    detector_kind: str
    checkpoint: str
    score_threshold: float = 0.3


@dataclass
class Coalition:
    name: str
    agents: dict[str, Agent] = field(default_factory=dict)
    relations: dict[str, Relation] = field(default_factory=dict)
    cycles: dict[str, SigmaCycle] = field(default_factory=dict)
    policies: dict[str, Policy] = field(default_factory=dict)
    # Stage G additions — optional; coalitions used only by the Tk
    # demo won't have these fields populated.
    gz_bindings: dict[str, GzBinding] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    # Stage H / Stage D-3 addition — optional vision-sidecar config.
    # Empty dict for coalitions that don't have a robot with a camera.
    vision_configs: dict[str, VisionConfig] = field(default_factory=dict)

    def agent_names(self) -> list[str]:
        return list(self.agents.keys())

    def relation_names(self) -> list[str]:
        return list(self.relations.keys())

    def cycle_names(self) -> list[str]:
        return list(self.cycles.keys())

    def policy_names(self) -> list[str]:
        return list(self.policies.keys())

    def threshold(self, kind: str, default: float | None = None) -> float:
        """Look up an observation threshold by kind, with optional default."""
        if kind in self.thresholds:
            return self.thresholds[kind]
        if default is not None:
            return default
        raise KeyError(
            f"no threshold declared for kind={kind!r}; "
            f"have: {list(self.thresholds)}"
        )


# ─── Parser ──────────────────────────────────────────────────────────


# Map the HyMeKo base-type leaf segment (after the namespace alias)
# onto the typed Coalition slot.
_AGENT_KINDS = ("human", "robot")
_RELATION_KINDS = ("interpersonal", "hri_relation")
_CYCLE_KINDS = ("sigma_cycle",)
_POLICY_KINDS = ("policy",)
_GZ_BINDING_KINDS = ("gz_binding",)
_THRESHOLD_KINDS = ("observation_threshold",)
_VISION_CONFIG_KINDS = ("vision_config",)


def _base_leaf(item: dict) -> str | None:
    """The leaf segment of the first base, e.g. ``hri.human`` → ``human``.
    Returns None if the item has no bases.
    """
    bases = item.get("bases") or []
    if not bases:
        return None
    path = bases[0].get("path") or []
    return path[-1] if path else None


def _scalar_field(item: dict, name: str, default: Any = None) -> Any:
    """Lookup a scalar body field by name.

    Returns the field's `value` for scalars, or the resolved single
    ref for `<ref>` values, or ``default`` if absent.
    """
    for sub in item.get("body") or []:
        if sub.get("name") == name:
            v = sub.get("value")
            if isinstance(v, dict) and "ref" in v:
                return v["ref"][0]
            return v
    return default


def _ref_list_field(item: dict, name: str) -> list[str]:
    """Lookup a list-of-refs body field, e.g. ``members [r_ab, r_ar, r_br];``.

    Returns the list of names referenced (empty list if absent).
    """
    for sub in item.get("body") or []:
        if sub.get("name") == name:
            v = sub.get("value")
            if isinstance(v, list):
                out: list[str] = []
                for entry in v:
                    if isinstance(entry, dict) and "ref" in entry:
                        out.append(entry["ref"][0])
                    elif isinstance(entry, str):
                        out.append(entry)
                return out
    return []


def _walk_top_level_body(ir: dict) -> Iterable[dict]:
    """Yield the body items of the *first* concrete top-level decl
    (the coalition itself, not the description block)."""
    items = ir.get("items") or []
    # First item is typically the description block (no bases), the
    # second is the concrete coalition with bases (e.g. `: hri`).
    # Walk both and yield any items that have bases — those are the
    # ones we care about.
    for top in items:
        body = top.get("body") or []
        for sub in body:
            yield sub


def parse_coalition_ir(ir: dict) -> Coalition:
    """Convert a HyMeKo IR (output of ``parse_hymeko_rs``) into a
    typed :class:`Coalition`.

    Raises
    ------
    ValueError
        If the IR is malformed (missing required fields, unresolved
        agent references in a relation, unknown base types).
    """
    # The coalition's name is the *second* top-level item (the first
    # is the description header). If only one item, fall back to it.
    items = ir.get("items") or []
    if not items:
        raise ValueError("empty HyMeKo IR — no top-level items")
    if len(items) >= 2:
        coalition_name = items[1].get("name", "<unnamed>")
    else:
        coalition_name = items[0].get("name", "<unnamed>")

    coalition = Coalition(name=coalition_name)

    for item in _walk_top_level_body(ir):
        leaf = _base_leaf(item)
        if leaf is None:
            continue
        nm = item.get("name") or ""
        if leaf in _AGENT_KINDS:
            coalition.agents[nm] = Agent(name=nm, kind=leaf)
        elif leaf in _RELATION_KINDS:
            src = _scalar_field(item, "from")
            dst = _scalar_field(item, "to")
            sign_v = _scalar_field(item, "sign", default=1.0)
            mag = _scalar_field(item, "magnitude", default=1.0)
            if src is None or dst is None:
                raise ValueError(
                    f"relation {nm!r} missing from/to (got from={src!r}, "
                    f"to={dst!r})"
                )
            try:
                sign_int = int(sign_v)
            except (TypeError, ValueError):
                sign_int = 1 if (sign_v or 0) > 0 else -1
            coalition.relations[nm] = Relation(
                name=nm, kind=leaf,
                src=str(src), dst=str(dst),
                sign=sign_int, magnitude=float(mag),
            )
        elif leaf in _CYCLE_KINDS:
            members = _ref_list_field(item, "members")
            coalition.cycles[nm] = SigmaCycle(name=nm, members=members)
        elif leaf in _POLICY_KINDS:
            cond = _scalar_field(item, "condition")
            act = _scalar_field(item, "action")
            if cond is None or act is None:
                raise ValueError(
                    f"policy {nm!r} missing condition/action (got "
                    f"condition={cond!r}, action={act!r})"
                )
            coalition.policies[nm] = Policy(
                name=nm, condition=str(cond), action=str(act),
            )
        elif leaf in _GZ_BINDING_KINDS:
            agent = _scalar_field(item, "agent")
            pose_topic = _scalar_field(item, "pose_topic")
            cmd_vel_topic = _scalar_field(item, "cmd_vel_topic")
            gaze_cmd_topic = _scalar_field(item, "gaze_cmd_topic")
            camera_topic = _scalar_field(item, "camera_topic")
            if agent is None or pose_topic is None:
                raise ValueError(
                    f"gz_binding {nm!r} missing agent or pose_topic"
                )
            coalition.gz_bindings[str(agent)] = GzBinding(
                agent=str(agent),
                pose_topic=str(pose_topic),
                cmd_vel_topic=(str(cmd_vel_topic)
                               if cmd_vel_topic is not None else None),
                gaze_cmd_topic=(str(gaze_cmd_topic)
                                if gaze_cmd_topic is not None else None),
                camera_topic=(str(camera_topic)
                              if camera_topic is not None else None),
            )
        elif leaf in _THRESHOLD_KINDS:
            kind = _scalar_field(item, "kind")
            value = _scalar_field(item, "value")
            if kind is None or value is None:
                raise ValueError(
                    f"observation_threshold {nm!r} missing kind/value"
                )
            try:
                coalition.thresholds[str(kind)] = float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"observation_threshold {nm!r}: value {value!r} "
                    f"is not a float ({e})"
                ) from e
        elif leaf in _VISION_CONFIG_KINDS:
            detector_kind = _scalar_field(item, "detector_kind")
            checkpoint = _scalar_field(item, "checkpoint", default="")
            score_threshold = _scalar_field(
                item, "score_threshold", default=0.3,
            )
            if detector_kind is None:
                raise ValueError(
                    f"vision_config {nm!r} missing detector_kind"
                )
            try:
                score_t = float(score_threshold)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"vision_config {nm!r}: score_threshold {score_threshold!r} "
                    f"is not a float ({e})"
                ) from e
            coalition.vision_configs[nm] = VisionConfig(
                detector_kind=str(detector_kind),
                checkpoint=str(checkpoint or ""),
                score_threshold=score_t,
            )
        # Other leaves (unknown types) are silently ignored — the
        # loader is forward-compatible with meta-schema additions.

    # Cross-check: every relation's src/dst must name an agent.
    for r in coalition.relations.values():
        if r.src not in coalition.agents:
            raise ValueError(
                f"relation {r.name!r}: src={r.src!r} not an agent in {list(coalition.agents)}"
            )
        if r.dst not in coalition.agents:
            raise ValueError(
                f"relation {r.name!r}: dst={r.dst!r} not an agent in {list(coalition.agents)}"
            )

    # Cross-check: every cycle's member must name a relation.
    for c in coalition.cycles.values():
        for m in c.members:
            if m not in coalition.relations:
                raise ValueError(
                    f"cycle {c.name!r}: member {m!r} not a relation"
                )

    return coalition


def load_coalition(path: Path | str) -> Coalition:
    """Load and parse a coalition .hymeko file.

    Args:
        path: Filesystem path to the .hymeko coalition spec.

    Returns:
        Typed :class:`Coalition` describing the agents, relations,
        cycles, and policies.
    """
    try:
        import hymeko  # the PyO3 wheel
    except ImportError as e:
        raise RuntimeError(
            f"hymeko PyO3 wheel not installed; pip-install "
            f"target/wheels/hymeko-*.whl (got: {e})"
        ) from e
    src = Path(path).read_text(encoding="utf-8")
    ir = hymeko.parse_hymeko_rs(src)
    return parse_coalition_ir(ir)
