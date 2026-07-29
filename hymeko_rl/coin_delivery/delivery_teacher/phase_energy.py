"""Non-invasive, phase-marked energy instrumentation for the delivery+settle rollout.

A :class:`PhaseEnergyProbe` is plugged in as the ``rollout_primitive`` ``frame_hook`` — which is contractually
non-behavioural ("physical measurements only — no reward, no pin, no hidden force"). The probe only *reads* the stepped
state after each control step: it accumulates positive/negative actuator work and snapshots robot + coin kinetic energy at
the phase boundaries (after capture, braking onset, target entry, release, settle start, terminal). From those it builds a
:class:`PhaseEnergyLedger` with the derived diagnostics (target-entry coin KE, ``dH`` across phases, target-directed
energy / W+). **Energy is a diagnostic observer here, never an optimization objective** — so a lower target-entry energy
in the successful teachers is discovered, not fitted.

Coin kinetic energy uses a unit-mass proxy (specific KE), consistent with the R11.2 two-level contract; the true masses
enter at R11.8. The ledger asserts only completeness + a recorded residual (no conservation verdict).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import CENTER_TOL
from hymeko_rl.ir.energy import EnergyTransitionCertificate, MeasuredEnergyLedger


def _robot_ke(rl: Any) -> float:
    m, d = rl.inner.model, rl.inner.data
    full = np.zeros((m.nv, m.nv), np.float64)
    mujoco.mj_fullM(m, d, full)  # mujoco>=3 signature (model, data, dst)
    v = np.asarray(d.qvel[:4], np.float64)
    return float(0.5 * v @ full[:4, :4] @ v)


def _coin_ke(rl: Any) -> float:
    """Specific coin kinetic energy 0.5(|v_xy|^2 + w^2) from the true generalized coin velocities (unit-mass proxy)."""
    qv = np.asarray(rl.inner.data.qvel, np.float64)
    trans = float(qv[4:6] @ qv[4:6])
    rot = float(qv[6] ** 2) if qv.shape[0] > 6 else 0.0
    return 0.5 * (trans + rot)


def _dtz(rl: Any) -> float:
    return float(rl.inner.direction_to_zone()[1])


@dataclass
class PhaseSnapshot:
    """Energy state captured at one phase boundary."""

    t: int
    robot_ke: float
    coin_ke: float
    dtz_mm: float

    @property
    def total(self) -> float:
        return self.robot_ke + self.coin_ke


@dataclass
class PhaseEnergyProbe:
    """Read-only per-control-step energy probe. ``ramp``/``release`` are the θ phase boundaries (control steps)."""

    ramp: int
    release: int
    dt: float
    w_pos: float = 0.0
    w_neg: float = 0.0
    peak_coin_ke: float = 0.0
    _entered: bool = False
    _snaps: dict[str, PhaseSnapshot] = field(default_factory=dict)
    _min_dtz_after_entry: float = 1e9
    _max_dtz_after_entry: float = 0.0

    def __call__(self, rl: Any, t: int) -> None:
        d = rl.inner.data
        power = float(np.asarray(d.ctrl[:4], np.float64) @ np.asarray(d.qvel[:4], np.float64))
        if power >= 0.0:
            self.w_pos += power * self.dt
        else:
            self.w_neg += -power * self.dt
        cke = _coin_ke(rl)
        self.peak_coin_ke = max(self.peak_coin_ke, cke)
        dtz_mm = _dtz(rl) * 1000.0
        snap = PhaseSnapshot(t=t, robot_ke=_robot_ke(rl), coin_ke=cke, dtz_mm=dtz_mm)
        if "after_capture" not in self._snaps:
            self._snaps["after_capture"] = snap
        if t == self.ramp:
            self._snaps["braking_onset"] = snap
        if t == self.release:
            self._snaps["release"] = snap
        if (not self._entered) and dtz_mm <= CENTER_TOL * 1000.0:
            self._entered = True
            self._snaps["target_entry"] = snap
        if self._entered:                                    # overshoot / rebound band after first target entry
            self._min_dtz_after_entry = min(self._min_dtz_after_entry, dtz_mm)
            self._max_dtz_after_entry = max(self._max_dtz_after_entry, dtz_mm)
        self._snaps["terminal"] = snap
        if t == self.release + 1:
            self._snaps["settle_start"] = snap


def _s(snaps: dict[str, PhaseSnapshot], key: str) -> Optional[PhaseSnapshot]:
    return snaps.get(key)


@dataclass(frozen=True)
class PhaseEnergyLedger:
    """The delivery+settle energy diagnostic. Boundary energies + derived phase deltas; None where a phase did not occur
    (e.g. target entry never reached). Asserts completeness of the *reached* phases, never conservation."""

    w_actuator_pos: float
    w_actuator_neg: float
    peak_coin_ke: float
    e_after_capture: float
    e_braking_onset: Optional[float]
    e_target_entry: Optional[float]
    e_release: Optional[float]
    e_settle_start: Optional[float]
    e_terminal: float
    t_coin_entry: Optional[float]          # coin KE at target entry (the mechanism variable of interest)
    overshoot_mm: Optional[float]          # max dtz excursion after first entry (rebound out of the zone)
    dwell_min_dtz_mm: Optional[float]
    reached_target: bool

    @property
    def dH_capture_to_entry(self) -> Optional[float]:
        return None if self.e_target_entry is None else self.e_target_entry - self.e_after_capture

    @property
    def dH_entry_to_settle(self) -> Optional[float]:
        if self.e_target_entry is None or self.e_settle_start is None:
            return None
        return self.e_settle_start - self.e_target_entry

    @property
    def target_directed_energy_ratio(self) -> Optional[float]:
        """coin KE delivered at target entry / positive actuator work (a measured efficiency proxy)."""
        if self.t_coin_entry is None or self.w_actuator_pos <= 1e-12:
            return None
        return self.t_coin_entry / self.w_actuator_pos

    def is_complete(self) -> bool:
        """Every *recorded* boundary is a finite number (reached-phase completeness — not conservation)."""
        vals = [self.w_actuator_pos, self.w_actuator_neg, self.peak_coin_ke, self.e_after_capture, self.e_terminal]
        opt = [self.e_braking_onset, self.e_target_entry, self.e_release, self.e_settle_start, self.t_coin_entry]
        return (all(math.isfinite(v) for v in vals)
                and all(v is None or math.isfinite(v) for v in opt))


def build_ledger(probe: PhaseEnergyProbe) -> PhaseEnergyLedger:
    """Assemble the phase energy ledger from a completed probe. Postcondition: reached-phase boundaries are finite."""
    s = probe._snaps
    entry = _s(s, "target_entry")
    ac = _s(s, "after_capture")
    return PhaseEnergyLedger(
        w_actuator_pos=round(probe.w_pos, 6), w_actuator_neg=round(probe.w_neg, 6),
        peak_coin_ke=round(probe.peak_coin_ke, 6),
        e_after_capture=round(ac.total, 6) if ac else 0.0,
        e_braking_onset=_opt(_s(s, "braking_onset")),
        e_target_entry=_opt(entry), e_release=_opt(_s(s, "release")), e_settle_start=_opt(_s(s, "settle_start")),
        e_terminal=round(_s(s, "terminal").total, 6) if _s(s, "terminal") else 0.0,
        t_coin_entry=round(entry.coin_ke, 6) if entry else None,
        overshoot_mm=round(probe._max_dtz_after_entry, 2) if probe._entered else None,
        dwell_min_dtz_mm=round(probe._min_dtz_after_entry, 2) if probe._entered else None,
        reached_target=probe._entered)


def _opt(snap: Optional[PhaseSnapshot]) -> Optional[float]:
    return None if snap is None else round(snap.total, 6)


def as_measured_ledger(led: PhaseEnergyLedger) -> MeasuredEnergyLedger:
    """Project the phase ledger onto the R11.2 :class:`MeasuredEnergyLedger` so the existing two-level energy certificate
    (ENERGY_LEDGER_COMPLETE / ENERGY_BALANCE_RESIDUAL_RECORDED; conservation refused) applies unchanged."""
    return MeasuredEnergyLedger(
        robot_ke=0.0, object_ke=led.peak_coin_ke, potential_energy=0.0, w_actuator_pos=led.w_actuator_pos,
        w_actuator_neg=led.w_actuator_neg, contact_impulse=0.0, dissipation_proxy=led.w_actuator_neg,
        energy_pre=led.e_after_capture, energy_post=led.e_terminal,
        numerical_residual=led.e_terminal - (led.e_after_capture + led.w_actuator_pos - led.w_actuator_neg))


def energy_certificate(led: PhaseEnergyLedger) -> EnergyTransitionCertificate:
    return EnergyTransitionCertificate(as_measured_ledger(led))
