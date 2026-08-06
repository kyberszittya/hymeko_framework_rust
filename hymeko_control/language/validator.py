"""HyMeKo control language --- validator.

``validate(raw)`` turns a parsed ``.hymeko.yaml`` dict into a
:class:`~hymeko_control.language.ir.ControlModel`, or raises
:class:`ValidationError` with a precise message. It never returns a partially
built or degenerate model: either every reference resolves, or it raises.

The core entry point takes a **dict** (already parsed), so the core stays
stdlib-only. ``load_yaml`` is an optional convenience that imports ``yaml``
lazily; scenarios without PyYAML can build the dict themselves.
"""

from __future__ import annotations

from typing import Any, Mapping

from .ir import (
    AuthoritySpec,
    CertificateSpec,
    ControlModel,
    Entity,
    IntentSpec,
    Mode,
    MorphologyRelation,
    PhysicsElement,
    Port,
    Transition,
)
from .schema_v0 import (
    REQUIRED_HYBRID_KEYS,
    REQUIRED_SECTIONS,
    REQUIRED_TASK_KEYS,
    SCHEMA_VERSION,
    CertificateKind,
    EntityKind,
    PhysicsRole,
    PortKind,
    RelationKind,
)

_VALID_AUTHORITY_SOURCES = ("observed", "modeled", "assumed")


class ValidationError(ValueError):
    """Raised when a raw scenario spec does not conform to schema v0."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValidationError(msg)


def _enum(value: str, enum_cls: type, section: str) -> Any:
    try:
        return enum_cls(value)
    except ValueError as err:
        allowed = [e.value for e in enum_cls]
        raise ValidationError(
            f"{section}: unknown {enum_cls.__name__} {value!r}; allowed {allowed}"
        ) from err


def _check_sections(raw: Mapping[str, Any]) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in raw]
    _require(not missing, f"missing required section(s): {missing}")
    _require(
        raw["schema_version"] == SCHEMA_VERSION,
        f"schema_version must be {SCHEMA_VERSION!r}, got {raw['schema_version']!r}",
    )
    for key in REQUIRED_HYBRID_KEYS:
        _require(key in raw["hybrid"], f"hybrid: missing {key!r}")
    for key in REQUIRED_TASK_KEYS:
        _require(key in raw["task"], f"task: missing {key!r}")


def _parse_entities(raw: Mapping[str, Any]) -> tuple[Entity, ...]:
    out: list[Entity] = []
    seen: set[str] = set()
    for item in raw["entities"]:
        eid = item["id"]
        _require(eid not in seen, f"entities: duplicate id {eid!r}")
        seen.add(eid)
        out.append(
            Entity(eid, _enum(item["kind"], EntityKind, "entities"),
                   item.get("attributes", {}))
        )
    return tuple(out)


def _parse_ports(raw: Mapping[str, Any], entity_ids: set[str]) -> tuple[Port, ...]:
    out: list[Port] = []
    seen: set[str] = set()
    for item in raw["ports"]:
        pid = item["id"]
        _require(pid not in seen, f"ports: duplicate id {pid!r}")
        seen.add(pid)
        owner = item["owner"]
        _require(owner in entity_ids, f"port {pid!r}: unknown owner {owner!r}")
        out.append(
            Port(pid, _enum(item["kind"], PortKind, "ports"), owner,
                 item.get("attributes", {}))
        )
    return tuple(out)


def _parse_morphology(
    raw: Mapping[str, Any], entity_ids: set[str]
) -> tuple[MorphologyRelation, ...]:
    out: list[MorphologyRelation] = []
    for item in raw["morphology"]:
        src, tgt = item["source"], item["target"]
        _require(src in entity_ids, f"morphology: unknown source {src!r}")
        _require(tgt in entity_ids, f"morphology: unknown target {tgt!r}")
        out.append(
            MorphologyRelation(_enum(item["kind"], RelationKind, "morphology"), src, tgt)
        )
    return tuple(out)


def _parse_physics(raw: Mapping[str, Any]) -> tuple[PhysicsElement, ...]:
    return tuple(
        PhysicsElement(item["name"], _enum(item["role"], PhysicsRole, "physics"),
                       item.get("attributes", {}))
        for item in raw["physics"]
    )


def _parse_hybrid(raw: Mapping[str, Any]) -> tuple[tuple[Mode, ...], tuple[Transition, ...], str]:
    hybrid = raw["hybrid"]
    modes = tuple(
        Mode(m["name"], m.get("invariant"), m.get("reset", {}))
        for m in hybrid["modes"]
    )
    mode_names = {m.name for m in modes}
    _require(len(mode_names) == len(modes), "hybrid: duplicate mode name")
    transitions: list[Transition] = []
    for t in hybrid["transitions"]:
        _require(t["source"] in mode_names, f"transition: unknown source mode {t['source']!r}")
        _require(t["dest"] in mode_names, f"transition: unknown dest mode {t['dest']!r}")
        transitions.append(Transition(t["source"], t["dest"], t["guard"], t.get("event")))
    initial = hybrid["initial_mode"]
    _require(initial in mode_names, f"hybrid: initial_mode {initial!r} is not a declared mode")
    return modes, tuple(transitions), initial


def _parse_task(
    raw: Mapping[str, Any],
) -> tuple[tuple[IntentSpec, ...], tuple[AuthoritySpec, ...], tuple[CertificateSpec, ...]]:
    task = raw["task"]
    intents = tuple(
        IntentSpec(i["name"], float(i["lower"]), float(i["upper"]), i.get("unit"))
        for i in task["intents"]
    )
    _require(len({i.name for i in intents}) == len(intents), "task: duplicate intent name")
    authorities: list[AuthoritySpec] = []
    for a in task["authorities"]:
        _require(
            a["source"] in _VALID_AUTHORITY_SOURCES,
            f"authority {a['name']!r}: source must be one of {_VALID_AUTHORITY_SOURCES}",
        )
        authorities.append(AuthoritySpec(a["name"], a["source"], a["provenance"]))
    certificates = tuple(
        CertificateSpec(c["name"], _enum(c["kind"], CertificateKind, "certificates"),
                        c["predicate"])
        for c in task["certificates"]
    )
    _require(any(c.kind == CertificateKind.SUCCESS for c in certificates),
             "task: at least one success certificate is required")
    return intents, tuple(authorities), certificates


def validate(raw: Mapping[str, Any]) -> ControlModel:
    """Validate a parsed scenario dict and build a :class:`ControlModel`.

    # Preconditions
    ``raw`` is a mapping parsed from a ``.hymeko.yaml`` scenario contract.

    # Postconditions
    Returns a fully-referenced, immutable :class:`ControlModel`. Raises
    :class:`ValidationError` (never a bare ``KeyError``/``ValueError``) on any
    missing section, unknown kind, dangling reference, duplicate id, or
    inverted intent bound.
    """
    try:
        _check_sections(raw)
        entities = _parse_entities(raw)
        entity_ids = {e.eid for e in entities}
        ports = _parse_ports(raw, entity_ids)
        morphology = _parse_morphology(raw, entity_ids)
        physics = _parse_physics(raw)
        modes, transitions, initial = _parse_hybrid(raw)
        intents, authorities, certificates = _parse_task(raw)
    except ValidationError:
        raise
    except (KeyError, ValueError, TypeError) as err:
        raise ValidationError(f"malformed scenario spec: {err}") from err

    return ControlModel(
        name=raw["name"],
        entities=entities,
        morphology=morphology,
        ports=ports,
        physics=physics,
        modes=modes,
        transitions=transitions,
        initial_mode=initial,
        intents=intents,
        authorities=authorities,
        certificates=certificates,
    )


def load_yaml(path: str) -> ControlModel:
    """Convenience: parse a ``.hymeko.yaml`` file then :func:`validate` it.

    Imports ``yaml`` lazily so the core import path stays stdlib-only.
    """
    import yaml  # local import: keeps hymeko_control torch/dep-free at import time

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    _require(isinstance(raw, Mapping), f"{path}: top-level document must be a mapping")
    return validate(raw)
