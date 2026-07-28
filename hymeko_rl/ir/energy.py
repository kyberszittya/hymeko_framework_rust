"""Two-level energy contract for hybrid transitions.

Under MuJoCo contact, damping, actuator governance, and numerical integration, not every energy term is exactly
recoverable, so R11.2 asserts only the **measurement** level. :class:`MeasuredEnergyLedger` records the per-mode/transition
energy budget (robot and object kinetic energy, potential energy, positive/negative actuator work tracked *separately*,
contact impulse, a dissipation proxy, pre/post energy, and a numerical residual). The R11.2
:class:`EnergyTransitionCertificate` asserts ``ENERGY_LEDGER_COMPLETE`` + ``ENERGY_BALANCE_RESIDUAL_RECORDED`` and **never**
a conservation verdict — :meth:`EnergyTransitionCertificate.conservation_verdict` deliberately refuses, because the
*modelled* level (which residual band is acceptable) is calibrated later, at R11.8.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

LEDGER_COMPLETE = "ENERGY_LEDGER_COMPLETE"
BALANCE_RESIDUAL_RECORDED = "ENERGY_BALANCE_RESIDUAL_RECORDED"


@dataclass(frozen=True)
class MeasuredEnergyLedger:
    """Measured energy terms across a mode or a transition. Positive and negative actuator work are separate so they can
    never silently cancel. All values are joules (or J-equivalent proxies); every field must be a finite number."""

    robot_ke: float
    object_ke: float
    potential_energy: float
    w_actuator_pos: float
    w_actuator_neg: float
    contact_impulse: float
    dissipation_proxy: float
    energy_pre: float
    energy_post: float
    numerical_residual: float

    def is_complete(self) -> bool:
        """Postcondition: True iff every recorded term is a finite number (measurement completeness — not conservation)."""
        return all(isinstance(getattr(self, f.name), (int, float))
                   and math.isfinite(float(getattr(self, f.name))) for f in fields(self))

    def balance_residual(self) -> float:
        """The first-law imbalance actually measured for this segment:
        ``E_post - (E_pre + W+ - W- - dissipation - impact)``. Recorded, **not** asserted small at R11.2."""
        accounted = (self.energy_pre + self.w_actuator_pos - self.w_actuator_neg
                     - self.dissipation_proxy - self.contact_impulse)
        return self.energy_post - accounted


@dataclass(frozen=True)
class EnergyTransitionCertificate:
    """The measurement-level certificate for one segment. Carries the verdicts it can honestly assert now; refuses the
    conservation verdict until R11.8 calibrates the acceptable residual band."""

    ledger: MeasuredEnergyLedger

    @property
    def verdicts(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.ledger.is_complete():
            out.append(LEDGER_COMPLETE)
            if math.isfinite(self.ledger.balance_residual()):
                out.append(BALANCE_RESIDUAL_RECORDED)
        return tuple(out)

    def is_measurement_complete(self) -> bool:
        return LEDGER_COMPLETE in self.verdicts and BALANCE_RESIDUAL_RECORDED in self.verdicts

    def conservation_verdict(self) -> str:
        """R11.2 does not judge conservation. The modelled Hamiltonian certificate (acceptable residual band) is
        calibrated at R11.8; asking here is a contract error, so this raises rather than returning a misleading pass."""
        raise NotImplementedError(
            "HAMILTONIAN_CONSERVATION_PASS is calibrated at R11.8 (ModelledHamiltonianCertificate); "
            "R11.2 records the measured ledger + residual only.")
