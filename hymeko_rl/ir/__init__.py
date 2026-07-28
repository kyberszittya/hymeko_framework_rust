"""HyMeKo hybrid-delivery IR (R11.2).

A domain-generic intermediate representation for the exact-zero-home hybrid delivery task: the initial-condition contract
and its certificate, the certificate-filtered initial distribution, the hybrid mode automaton M0..M7 with per-mode
invariants and trace validation, guarded transitions with handoff descriptors, the two-level (measured, not conserved)
energy contract, and rollout provenance + success certificates. No simulator or domain geometry lives here — adapters read
a live state into :class:`RolloutState` and everything downstream is verified over that struct.
"""
from __future__ import annotations

from hymeko_rl.ir.energy import (
    BALANCE_RESIDUAL_RECORDED,
    LEDGER_COMPLETE,
    EnergyTransitionCertificate,
    MeasuredEnergyLedger,
)
from hymeko_rl.ir.hybrid_mode import (
    HybridMode,
    ModeTrace,
    StateInvariant,
    build_mode_trace,
    zero_home_invariant,
)
from hymeko_rl.ir.initial_condition import (
    AdmissibilityResult,
    InitialCondition,
    InitialConditionCertificate,
    InitialConditionViolation,
    InitialDistribution,
    RejectionLedger,
)
from hymeko_rl.ir.provenance import RolloutProvenance, SuccessCertificate
from hymeko_rl.ir.rollout import RolloutState
from hymeko_rl.ir.transition import (
    HandoffDescriptor,
    HybridTransition,
    TransitionGuard,
    TransitionGuardError,
)

__all__ = [
    "RolloutState",
    "InitialCondition",
    "InitialConditionCertificate",
    "InitialConditionViolation",
    "InitialDistribution",
    "AdmissibilityResult",
    "RejectionLedger",
    "HybridMode",
    "StateInvariant",
    "ModeTrace",
    "build_mode_trace",
    "zero_home_invariant",
    "TransitionGuard",
    "TransitionGuardError",
    "HandoffDescriptor",
    "HybridTransition",
    "MeasuredEnergyLedger",
    "EnergyTransitionCertificate",
    "LEDGER_COMPLETE",
    "BALANCE_RESIDUAL_RECORDED",
    "RolloutProvenance",
    "SuccessCertificate",
]
