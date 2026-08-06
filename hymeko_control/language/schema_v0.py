"""HyMeKo control language --- schema v0 (the *vocabulary*, not the IR).

This module fixes the closed sets of kinds the language recognises and the
required top-level sections of a ``.hymeko.yaml`` scenario contract. The IR node
types live in :mod:`hymeko_control.language.ir`; validation lives in
:mod:`hymeko_control.language.validator`.

The vocabulary follows a port-Hamiltonian / bond-graph reading of a physical
system (energy storage, dissipation, interconnection, constraint) so that the
same language describes a gripper, a humanoid and a quadruped.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class EntityKind(str, Enum):
    """The kinds of declarable entities."""

    BODY = "body"
    JOINT = "joint"
    OBJECT = "object"
    TARGET = "target"
    SENSOR = "sensor"
    CONTROLLER = "controller"


class PortKind(str, Enum):
    """Port kinds. Effort/flow are the bond-graph power-conjugate pair."""

    EFFORT = "effort"
    FLOW = "flow"
    ACTUATION = "actuation"
    OBSERVATION = "observation"
    CONTACT = "contact"
    CONSTRAINT = "constraint"


class RelationKind(str, Enum):
    """Morphology relations between entities."""

    KINEMATIC = "kinematic"
    DYNAMIC = "dynamic"
    ACTUATION = "actuation"


class PhysicsRole(str, Enum):
    """Bond-graph role of a physics element."""

    ENERGY_STORAGE = "energy_storage"
    DISSIPATION = "dissipation"
    INTERCONNECTION = "interconnection"
    CONTACT = "contact"
    CONSTRAINT = "constraint"


class CertificateKind(str, Enum):
    """External certificate kinds. Safety dominates success (see certificate.py)."""

    SUCCESS = "success"
    SAFETY = "safety"


SCHEMA_VERSION = "v0"

#: Top-level sections a scenario contract must declare (order-independent).
REQUIRED_SECTIONS: tuple[str, ...] = (
    "schema_version",
    "name",
    "entities",
    "morphology",
    "ports",
    "physics",
    "hybrid",
    "task",
)

#: Sub-keys required inside the ``hybrid`` section.
REQUIRED_HYBRID_KEYS: tuple[str, ...] = ("modes", "transitions", "initial_mode")

#: Sub-keys required inside the ``task`` section.
REQUIRED_TASK_KEYS: tuple[str, ...] = ("intents", "authorities", "certificates")


def schema_descriptor() -> dict[str, Any]:
    """Return a JSON-serialisable description of schema v0.

    # Postconditions
    The result is a plain ``dict`` of strings / lists of strings, suitable for
    ``json.dump`` into ``schema_contract.json``. It enumerates the closed kind
    sets and the required sections so the contract is machine-checkable.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "required_sections": list(REQUIRED_SECTIONS),
        "required_hybrid_keys": list(REQUIRED_HYBRID_KEYS),
        "required_task_keys": list(REQUIRED_TASK_KEYS),
        "entity_kinds": [k.value for k in EntityKind],
        "port_kinds": [k.value for k in PortKind],
        "relation_kinds": [k.value for k in RelationKind],
        "physics_roles": [r.value for r in PhysicsRole],
        "certificate_kinds": [k.value for k in CertificateKind],
    }
