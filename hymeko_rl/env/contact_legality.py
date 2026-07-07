"""Declarative contact-quality contract for contact-rich manipulation envs.

The *structural contact contract* — which geoms are the manipulated object, which contacts are the
preferred grasp, and which are a physically-allowed-but-non-preferred arm-body contact — is declared once
as a :class:`ContactLegalitySpec` (built from the compiled model at env init) and consumed by
:func:`classify_contacts`, instead of being scattered as flags, geom-id lists, and hardcoded name checks.

The model is **graded**, not binary (Hajdu 2026-07-06). Arm-body↔coin collision is physically real and
*allowed*; it is simply lower-quality than a fingertip grasp:

* ``OBJECT`` ↔ ``FINGERTIP``  — the preferred, high-quality contact (the fingertips moving the coin).
* ``OBJECT`` ↔ ``ARM_BODY``   — a non-preferred contact (a non-fingertip arm link touching the coin);
  tracked (count / duration / impulse) and penalised by the reward, and used to grade a delivery as
  clean / assisted / exploit. It does **not** by itself void the episode.

:class:`ContactMode` selects the *consequence* of an arm-body contact: ``GRADED`` (default, for training and
normal evaluation) only tracks and lets the reward/metrics grade it; ``STRICT`` (clean-paper validation)
additionally invalidates the delivery and can terminate on the first such contact. Contacts not involving
the object are ignored here — arm self-contact (a crash) is a separate concern handled by the env's metric
loop. v1 (no spec) keeps the abstract fingertip-only prototype where the coin passes through the arm bodies.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import mujoco
import numpy as np


class GeomRole(Enum):
    """The role a geom plays in the contact contract (assigned to each geom id by the declarative spec)."""

    OTHER = "other"              # not part of the contract
    OBJECT = "object"            # the manipulated object (the coin/disk)
    FINGERTIP = "fingertip"      # the preferred contact partner for the object (a fingertip)
    ARM_BODY = "arm_body"        # a non-preferred partner (a non-fingertip arm link)


class ContactMode(Enum):
    """What an arm-body↔object contact *does*, beyond being tracked and reward-penalised.

    ``GRADED`` — physically allowed, non-preferred; never voids the episode (default: training + normal eval).
    ``STRICT`` — additionally invalidates the delivery, and (if the spec asks) terminates on first contact
    (clean-paper validation only).
    """

    GRADED = "graded"
    STRICT = "strict"


def contact_force_magnitude(model: Any, data: Any, i: int) -> float:
    """Magnitude of contact ``i``'s force (normal + friction, contact frame) via ``mj_contactForce`` — the
    available proxy for a contact impulse.

    # Preconditions ``0 <= i < data.ncon``; ``(model, data)`` freshly stepped.
    # Postconditions finite ``>= 0``.
    """
    f6 = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, i, f6)
    return float(np.linalg.norm(f6[:3]))


@dataclass(frozen=True)
class ContactLegalitySpec:
    """The declarative contact-quality contract, built once at env init and inspected/consumed thereafter.

    Geom roles are stored as id sets (assigned by :meth:`from_model`); the left/right split of the fingertip
    contacts carries the side attribution the grasp metrics need. The quality rule is implicit in the role
    sets: object↔fingertip = preferred, object↔arm-body = non-preferred. :attr:`mode` picks the consequence.

    # Invariants ``left_fingertip_geoms``, ``right_fingertip_geoms``, ``arm_body_geoms``, ``object_geoms``
      are pairwise disjoint; ``STRICT`` mode invalidates a delivery on any arm-body contact.
    """

    object_geoms: frozenset[int]
    left_fingertip_geoms: frozenset[int]
    right_fingertip_geoms: frozenset[int]
    arm_body_geoms: frozenset[int]
    mode: ContactMode = ContactMode.GRADED
    strict_terminate: bool = True          # STRICT only: end the episode on the first arm-body contact

    @property
    def fingertip_geoms(self) -> frozenset[int]:
        """All preferred (fingertip) contact partners, both sides."""
        return self.left_fingertip_geoms | self.right_fingertip_geoms

    @property
    def invalidates_on_arm_body(self) -> bool:
        """True iff an arm-body contact voids the delivery (STRICT only)."""
        return self.mode is ContactMode.STRICT

    def role(self, geom: int) -> GeomRole:
        """The declared role of a geom id (for inspection / debugging the contract)."""
        if geom in self.object_geoms:
            return GeomRole.OBJECT
        if geom in self.left_fingertip_geoms or geom in self.right_fingertip_geoms:
            return GeomRole.FINGERTIP
        if geom in self.arm_body_geoms:
            return GeomRole.ARM_BODY
        return GeomRole.OTHER

    @classmethod
    def from_model(cls, model: Any, *, object_geoms: "frozenset[int] | set[int]",
                   arm_bodies_left: frozenset[int], arm_bodies_right: frozenset[int],
                   fingertip_prefix: str = "fingertip", mode: ContactMode = ContactMode.GRADED,
                   strict_terminate: bool = True) -> "ContactLegalitySpec":
        """Assign geom roles **once** from the compiled model: every geom on an arm body is a FINGERTIP (its
        name starts with ``fingertip_prefix``) or an ARM_BODY (any other arm-link geom). This is the single
        place the naming convention lives — the classifier reads roles, never names.

        # Preconditions ``object_geoms`` non-empty; ``arm_bodies_{left,right}`` are the two arm chains.
        # Postconditions at least one fingertip geom exists (else the contract is unsatisfiable — raises);
          every arm-body geom is fingertip xor arm-body.
        """
        left_ft: set[int] = set()
        right_ft: set[int] = set()
        arm_body: set[int] = set()
        for g in range(int(model.ngeom)):
            b = int(model.geom_bodyid[g])
            on_left, on_right = b in arm_bodies_left, b in arm_bodies_right
            if not (on_left or on_right):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith(fingertip_prefix):
                (left_ft if on_left else right_ft).add(g)
            else:
                arm_body.add(g)
        if not (left_ft or right_ft):
            raise ValueError(
                f"ContactLegalitySpec: no fingertip geoms (name prefix {fingertip_prefix!r}) found on the "
                "arm bodies — the contact contract would be unsatisfiable")
        return cls(frozenset(object_geoms), frozenset(left_ft), frozenset(right_ft),
                   frozenset(arm_body), mode, strict_terminate)


@dataclass(frozen=True)
class ContactLegalityState:
    """Per-step result of classifying an env's contacts against a :class:`ContactLegalitySpec`."""

    left_fingertip_contact: bool = False
    right_fingertip_contact: bool = False
    arm_body_contact: bool = False
    arm_body_contact_count: int = 0
    arm_body_contact_impulse: float = 0.0

    @property
    def fingertip_contact(self) -> bool:
        """At least one fingertip touches the object (a preferred contact)."""
        return self.left_fingertip_contact or self.right_fingertip_contact

    @property
    def both_fingertip_contact(self) -> bool:
        """Both fingertips touch the object (the two-finger grasp)."""
        return self.left_fingertip_contact and self.right_fingertip_contact


def classify_contacts(model: Any, data: Any, spec: ContactLegalitySpec) -> ContactLegalityState:
    """Classify every **object** contact this step against the declarative ``spec``.

    object↔fingertip is a preferred grasp (attributed to the side whose fingertip set the partner is in);
    object↔any-other-arm-geom is a non-preferred ARM_BODY contact (counted, with its ``mj_contactForce``
    magnitude accumulated). Non-object contacts are ignored. This function *only classifies* — the
    consequence (grade, penalise, or in STRICT invalidate/terminate) is applied by the env from ``spec.mode``.

    # Preconditions ``(model, data)`` freshly stepped; ``spec`` built for this model.
    # Postconditions ``arm_body_contact`` iff ``arm_body_contact_count > 0``; counts/impulse ``>= 0``.
    """
    left = right = arm_body = False
    count = 0
    impulse = 0.0
    obj = spec.object_geoms
    for i in range(int(data.ncon)):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 in obj or g2 in obj:
            partner = g2 if g1 in obj else g1
            if partner in spec.left_fingertip_geoms:
                left = True
            elif partner in spec.right_fingertip_geoms:
                right = True
            elif partner in spec.arm_body_geoms:
                arm_body = True
                count += 1
                impulse += contact_force_magnitude(model, data, i)
    return ContactLegalityState(left, right, arm_body, count, impulse)
