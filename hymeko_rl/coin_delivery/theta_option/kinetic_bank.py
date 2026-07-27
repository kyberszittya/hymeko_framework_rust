"""K1 — the KINETIC feedback-bank primitives (neighbourhood generation), built on the frozen K0 contract.

The K0 contract (`kinetic_contract.py`, tag `coin-r9-learned-s1-kinetic-k0-positive-control`) is frozen; this module adds the
K1 bank primitives WITHOUT touching it. Diversity is produced the only legal way — **branch the frozen KINETIC entry and evolve
a PERTURBED control through real physics** (no state edit, no teleport, no hidden force): a short θ-schedule advance under
per-step Δτ noise from the frozen entry lands the coin in a nearby, physically-reachable transport state. Each candidate is
then admissibility-gated (dual-contact ∧ straddle ∧ motion-contract) and only admissible states enter the bank; rejections are
counted (never silently dropped).

The perturbation axes the campaign brief asks for map onto legal control perturbations: forward magnitude → coin velocity;
squeeze → normal force / clamp; balance → L/R imbalance and slip; per-step Δτ noise → previous-action perturbation; the number
of advance steps → distance-along-transport / relative geometry. Reuses the K0 `roll_until` step kernel (bit-identical to the
frozen `velocity_rollout`), `TransportSnapshot`, and the canonical `kinetic_observe` — no new physics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, _schedule_increment
from hymeko_rl.coin_delivery.theta_option.kinetic_contract import (
    FEATURE_NAMES, TransportSnapshot, kinetic_observe, roll_until)
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG

_FI = {n: i for i, n in enumerate(FEATURE_NAMES)}          # feature-name → index (state-descriptor extraction)


class _ThetaScheduleController:
    """Drives a fixed θ torque-primitive schedule (the teacher's `_schedule_increment`) with optional per-step Δτ noise, so the
    K0 `roll_until` step kernel can advance a snapshot under a perturbed control. The zone reference `e_par` is frozen at the
    first step (exactly as the teacher rollout uses it). # Invariants: emits only slew-admissible Δτ (the noise is clipped to
    ±slew); no teacher/future/state leak — it is a pure feed-forward generator of a legal transport branch."""

    def __init__(self, snap: Any, theta: Any, noise_slew_frac: float, rng: np.random.Generator) -> None:
        self.theta = np.asarray(theta, np.float64)
        self.step = float(snap.stack.tau_rate * snap.stack.control_dt)
        self.noise = float(noise_slew_frac) * self.step
        self.rng = rng
        self._e_par: np.ndarray | None = None

    def reset(self) -> None:
        self._e_par = None

    def dtau_for_step(self, rl: Any, t: int, prev_tau: np.ndarray) -> np.ndarray:
        if self._e_par is None:
            self._e_par = np.asarray(rl.inner.direction_to_zone()[0], np.float64)
        dtau = np.asarray(_schedule_increment(rl, self.theta, t, self._e_par, self.step, _coin_xy(rl)), np.float64)
        if self.noise > 0.0:
            dtau = dtau + self.rng.normal(0.0, self.noise, size=4)
        return np.clip(dtau, -self.step, self.step)

    def before_release(self, t: int) -> bool:
        return True

    @property
    def release_boundary_step(self) -> "int | None":
        return None


@dataclass(frozen=True)
class PerturbSpec:
    """One legal perturbation of the entry neighbourhood: an additive θ offset (from the generation base θ), a number of
    advance steps `k`, and per-step Δτ noise (fraction of slew). # Invariants: physically realized by evolving control through
    physics, never by editing coin/tip state."""

    category: str                                          # "easy" | "medium" | "edge"
    label: str
    dtheta: tuple
    k: int
    noise_slew_frac: float


# Generation base θ (small grip retains contact through the SHORT advance; the RELABEL warm-start is the squeeze≈0
# entry-delivering θ, a separate object). Perturbations vary push (velocity), squeeze (force/clamp), balance (imbalance/slip).
GEN_BASE_THETA = (0.05, 0.24, 0.0, 16.0, 9.0, 3.0)

NEIGHBOURHOOD_SPECS: tuple[PerturbSpec, ...] = (
    # 4 easy — near the teacher trace, stable contact, normal v_par
    PerturbSpec("easy", "trace_k2", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 2, 0.0),
    PerturbSpec("easy", "trace_k3", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 3, 0.0),
    PerturbSpec("easy", "push_soft_k4", (0.0, -0.02, 0.0, 0.0, 0.0, 0.0), 4, 0.0),
    PerturbSpec("easy", "grip_soft_k3", (0.01, 0.0, 0.0, 0.0, 0.0, 0.0), 3, 0.0),
    # 8 medium — varied slip / squeeze-force imbalance / coin velocity / tip-offset / previous-action noise
    PerturbSpec("medium", "vel_hi_k5", (0.0, 0.05, 0.0, 0.0, 0.0, 0.0), 5, 0.02),
    PerturbSpec("medium", "vel_lo_k5", (0.0, -0.05, 0.0, 0.0, 0.0, 0.0), 5, 0.02),
    PerturbSpec("medium", "grip_firm_k4", (0.05, 0.0, 0.0, 0.0, 0.0, 0.0), 4, 0.02),
    PerturbSpec("medium", "imbal_L_k5", (0.02, 0.0, 0.05, 0.0, 0.0, 0.0), 5, 0.03),
    PerturbSpec("medium", "imbal_R_k5", (0.02, 0.0, -0.05, 0.0, 0.0, 0.0), 5, 0.03),
    PerturbSpec("medium", "noise_k6", (0.02, 0.0, 0.0, 0.0, 0.0, 0.0), 6, 0.05),
    PerturbSpec("medium", "slide_k7", (0.0, 0.03, 0.02, 0.0, 0.0, 0.0), 7, 0.03),
    PerturbSpec("medium", "grip_push_k6", (0.03, 0.04, 0.0, 0.0, 0.0, 0.0), 6, 0.02),
    # 4 edge — still admissible, but near contact-loss / low-or-high v_par / asymmetric contact
    PerturbSpec("edge", "slipout_k6", (-0.05, 0.10, 0.0, 0.0, 0.0, 0.0), 6, 0.04),
    PerturbSpec("edge", "clamp_k8", (0.12, -0.08, 0.0, 0.0, 0.0, 0.0), 8, 0.04),
    PerturbSpec("edge", "asym_hi_k7", (0.0, 0.06, 0.10, 0.0, 0.0, 0.0), 7, 0.06),
    PerturbSpec("edge", "stall_k9", (0.08, -0.12, 0.0, 0.0, 0.0, 0.0), 9, 0.05),
)


def _clip_theta(theta: np.ndarray) -> np.ndarray:
    return np.clip(theta, np.asarray(DELIVERY_CFG.lo, np.float64), np.asarray(DELIVERY_CFG.hi, np.float64))


def perturbed_transport_branch(entry_tsnap: TransportSnapshot, spec: PerturbSpec, *,
                               seed: int, base_theta: Any = GEN_BASE_THETA) -> "tuple[TransportSnapshot | None, dict]":
    """Advance the frozen entry snapshot `spec.k` steps under the perturbed generation θ (+ per-step Δτ noise) and capture the
    resulting transport state. Physically legal (real governed physics, branched from the entry — no state edit). # Preconditions:
    `entry_tsnap` the frozen KINETIC entry; `seed` fixes the noise RNG. # Postconditions: returns (admissible TransportSnapshot
    or None, provenance) — None ⇒ the perturbation left the admissible set (counted, not fabricated)."""
    theta = _clip_theta(np.asarray(base_theta, np.float64) + np.asarray(spec.dtheta, np.float64))
    ctrl = _ThetaScheduleController(entry_tsnap, theta, spec.noise_slew_frac, np.random.default_rng(seed))
    cap = roll_until(entry_tsnap, ctrl, DELIVERY_CFG, stop_when=lambda c, rl, t: t >= spec.k)
    tsnap = TransportSnapshot.from_live(cap.rl, entry_tsnap.stack, cap.prev_tau)
    adm = tsnap.admissibility()
    prov = {"category": spec.category, "label": spec.label, "k": spec.k, "noise_slew_frac": spec.noise_slew_frac,
            "gen_theta": [round(float(x), 4) for x in theta], "admissible": adm.admissible,
            "straddle_dot": adm.straddle_dot, "fn_min": adm.fn_min, "qdot_max": adm.qdot_max}
    return (tsnap if adm.admissible else None), prov


def state_descriptor(tsnap: TransportSnapshot) -> dict:
    """The physical descriptors used for the distinct-state-spread audit, read from the canonical `kinetic_observe` (so the
    spread is measured in the exact policy-input space). # Postconditions: pure — reads a fresh branch."""
    feat, _h = kinetic_observe(tsnap.branch(), [])
    fn_l, fn_r = float(feat[_FI["fn_l"]]), float(feat[_FI["fn_r"]])
    return {"dtz_mm": round(float(feat[_FI["dtz"]]) * 1000, 2), "v_par": round(float(feat[_FI["v_par"]]), 4),
            "v_lat": round(float(feat[_FI["v_lat"]]), 4), "spin": round(float(feat[_FI["spin"]]), 4),
            "slip": round(max(abs(float(feat[_FI["slip_l"]])), abs(float(feat[_FI["slip_r"]]))), 5),
            "fn_min": round(min(fn_l, fn_r), 4), "fn_imbalance": round(abs(fn_l - fn_r), 4),
            "relpos_par_l": round(float(feat[_FI["relpos_par_l"]]), 5),
            "relpos_par_r": round(float(feat[_FI["relpos_par_r"]]), 5)}


def generate_neighbourhood(entry_tsnap: TransportSnapshot, *,
                           seed: int = 20260728) -> "tuple[list[dict], list[dict]]":
    """Generate the 4-easy / 8-medium / 4-edge KINETIC-entry neighbourhood by legal perturbed-control branching, keeping only
    admissible states. Deterministic given `seed` (each spec gets a distinct sub-seed). # Postconditions: returns
    (accepted, rejected); every accepted entry carries its TransportSnapshot + descriptor + provenance; rejections are the
    perturbations that left the admissible set (the rejection rate is a K1 metric)."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    for i, spec in enumerate(NEIGHBOURHOOD_SPECS):
        tsnap, prov = perturbed_transport_branch(entry_tsnap, spec, seed=seed + 101 * i)
        if tsnap is None:
            rejected.append(prov)
            continue
        accepted.append({"tsnap": tsnap, "descriptor": state_descriptor(tsnap), "provenance": prov})
    return accepted, rejected
