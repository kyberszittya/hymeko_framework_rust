"""Scenario definitions: contact policies, kinematic embodiments, the composed config, and the registry."""
from __future__ import annotations

from hymeko_rl.coin_delivery.scenarios.config import ScenarioConfig
from hymeko_rl.coin_delivery.scenarios.contact_policy import (
    C0Continuous,
    C1Intermittent,
    Contact,
    ContactCertification,
    ContactPolicy,
    ContactSample,
)
from hymeko_rl.coin_delivery.scenarios.kinematic_variant import (
    K0Canonical,
    K1DistalOrientation,
    KinematicVariant,
    distal_dof_probe,
    with_distal_pad_orientation,
)
from hymeko_rl.coin_delivery.scenarios.registry import ScenarioRegistry

__all__ = [
    "ScenarioConfig", "ScenarioRegistry", "ContactPolicy", "C0Continuous", "C1Intermittent",
    "Contact", "ContactSample", "ContactCertification", "KinematicVariant", "K0Canonical",
    "K1DistalOrientation", "with_distal_pad_orientation", "distal_dof_probe",
]
