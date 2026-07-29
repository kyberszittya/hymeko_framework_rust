"""R11.4A target-conditioned delivery + settle teacher.

Replaces the frozen canonical R2 downstream with a per-scenario CEM over the existing
``forward_displacement.rollout_primitive`` (deliver mode), which already aims at the relocatable target via
``direction_to_zone``. The search is driven by K6/distance/safety (the existing ``score``) — energy is a non-invasive
diagnostic observer, never an objective (that is R11.8). Generation/diagnosis only: no BC/RL/refinement.
"""
from __future__ import annotations

from hymeko_rl.coin_delivery.delivery_teacher.phase_energy import (
    PhaseEnergyLedger,
    PhaseEnergyProbe,
    build_ledger,
    energy_certificate,
)
from hymeko_rl.coin_delivery.delivery_teacher.record import (
    SCHEMA_VERSION,
    DeliveryRecord,
    build_delivery_record,
)
from hymeko_rl.coin_delivery.delivery_teacher.solver import (
    PHASE_A_FROZEN,
    PHASE_A_SEARCH,
    DeliveryResult,
    DeliverySearchSpec,
    solve_delivery,
)

__all__ = [
    "PhaseEnergyProbe",
    "PhaseEnergyLedger",
    "build_ledger",
    "energy_certificate",
    "DeliverySearchSpec",
    "DeliveryResult",
    "solve_delivery",
    "PHASE_A_SEARCH",
    "PHASE_A_FROZEN",
    "DeliveryRecord",
    "build_delivery_record",
    "SCHEMA_VERSION",
]
