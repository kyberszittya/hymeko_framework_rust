"""The declarative HyMeKo control language (schema v0 + IR + validator)."""

from __future__ import annotations

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
    SCHEMA_VERSION,
    CertificateKind,
    EntityKind,
    PhysicsRole,
    PortKind,
    RelationKind,
    schema_descriptor,
)
from .validator import ValidationError, validate

__all__ = [
    "SCHEMA_VERSION",
    "CertificateKind",
    "EntityKind",
    "PhysicsRole",
    "PortKind",
    "RelationKind",
    "schema_descriptor",
    "AuthoritySpec",
    "CertificateSpec",
    "ControlModel",
    "Entity",
    "IntentSpec",
    "Mode",
    "MorphologyRelation",
    "PhysicsElement",
    "Port",
    "Transition",
    "ValidationError",
    "validate",
]
