"""The composed scenario configuration: a contact policy × a kinematic embodiment, addressable by id."""
from __future__ import annotations

from dataclasses import dataclass

from hymeko_rl.coin_delivery.scenarios.contact_policy import ContactPolicy, contact_policy_from_dict
from hymeko_rl.coin_delivery.scenarios.kinematic_variant import (
    KinematicVariant,
    kinematic_variant_from_dict,
)


@dataclass(frozen=True)
class ScenarioConfig:
    """One coin-delivery scenario = an id + a contact-continuity policy + a kinematic embodiment + a description.

    # Preconditions ``scenario_id`` is a non-empty stable id. # Postconditions immutable and value-equal;
      ``from_dict(to_dict())`` reconstructs an identical config (the components are frozen + serializable)."""

    scenario_id: str
    contact_policy: ContactPolicy
    kinematic_variant: KinematicVariant
    description: str

    def to_dict(self) -> dict:
        return {"scenario_id": self.scenario_id, "contact_policy": self.contact_policy.to_dict(),
                "kinematic_variant": self.kinematic_variant.to_dict(), "description": self.description}

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        """Reconstruct from :meth:`to_dict`. # Preconditions ``d`` carries the four keys with registered kinds."""
        return cls(scenario_id=d["scenario_id"],
                   contact_policy=contact_policy_from_dict(d["contact_policy"]),
                   kinematic_variant=kinematic_variant_from_dict(d["kinematic_variant"]),
                   description=d["description"])
