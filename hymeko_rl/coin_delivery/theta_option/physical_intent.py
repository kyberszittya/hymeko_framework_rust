"""R3 — the cradle-AGNOSTIC physical-intent vocabulary + deterministic teacher extraction.

The R2 basin audit proved the held-out failure is a WRONG FACTORISATION: a direct cradle -> 6-D theta regressor lands 5-8
search-stds from the working basin. R3 replaces the target of prediction with a *physical intent* — the same physical plan
on every cradle — and lets a deterministic authority-aware decoder (`authority_decoder.py`) turn that intent into the
cradle-specific theta. This module defines the 7 intent roles and the deterministic extractor that reads them off a teacher
trajectory (frozen R3 contract `reports/2026-07-27-coin-r3-physical-intent-decoder-contract.md`).

Everything is done in the R1 CANONICAL frame, so the intent is mirror-INVARIANT (a cradle and its mirror extract the same
intent). Six roles map INVERTIBLY to theta through the measured authority (extract and decode are exact inverses on the
teacher -> the D0 round-trip is meaningful, not an arbitrary theta-distance pass). The 7th role, `peak_velocity`, is the
physical peak coin speed (a dynamic outcome, not forward*ramp — s3 reaches 0.45 m/s with a tiny forward push), used by the
decoder as an achievable-intent estimate + weak-link diagnostic, so it does not distort the round-trip.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.theta_option.canonical_frame import canonicalise, r1_grouped_features, to_canonical_theta
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG

# Role order is FROZEN. Units in the contract audit. Bounds are generous PHYSICAL ranges (not held-out-tuned) — the decoder
# additionally clips theta to the frozen box, so these only bound a learned predictor's output range.
INTENT_ROLES = ("forward_drive", "peak_velocity", "lateral", "squeeze", "brake_entry", "braking_demand", "release")
INTENT_LO = np.array([0.0, 0.0, -0.06, 0.0, 0.02, 0.0, 0.05], np.float64)
INTENT_HI = np.array([0.20, 0.80, 0.06, 0.35, 0.50, 2.50, 0.85], np.float64)
_EPS = 1e-6


def slew_of(snap: CradleSnapshot) -> float:
    """The per-step slew magnitude τ̇·dt (the Δτ command bound). # Postconditions: > 0."""
    return float(snap.stack.tau_rate * snap.stack.control_dt)


def authority_scalars(g_canon: "dict[str, np.ndarray]") -> "tuple[float, float, float, float]":
    """(forward_push_reach, brake_opposed_reach, lateral_reach, normal_force_reach) as positive scalars from the CANONICAL
    grouped R1 features (already carry every authority quantity). lateral/normal use the side-pair mean. ε-floored so the
    authority-normalised decode never divides by zero. # Postconditions: all ≥ _EPS."""
    fr = max(float(np.asarray(g_canon["forward_push_reach"]).ravel()[0]), _EPS)
    br = max(float(np.asarray(g_canon["brake_opposed_reach"]).ravel()[0]), _EPS)
    lr = max(float(np.mean(np.asarray(g_canon["lateral_reach_pair"], np.float64))), _EPS)
    nf = max(float(np.mean(np.asarray(g_canon["normal_force_reach_pair"], np.float64))), _EPS)
    return fr, br, lr, nf


@dataclass(frozen=True)
class PhysicalIntent:
    """A cradle-AGNOSTIC physical plan (canonical frame). forward_drive/lateral/squeeze/braking_demand are authority-
    normalised physical demands (m/s or contact v_n); brake_entry/release are horizon fractions; peak_velocity is the target
    peak coin speed (diagnostic/shaping). All mirror-invariant."""

    forward_drive: float
    peak_velocity: float
    lateral: float
    squeeze: float
    brake_entry: float
    braking_demand: float
    release: float

    def to_vector(self) -> np.ndarray:
        return np.array([self.forward_drive, self.peak_velocity, self.lateral, self.squeeze,
                         self.brake_entry, self.braking_demand, self.release], np.float64)

    @staticmethod
    def from_vector(v: np.ndarray) -> "PhysicalIntent":
        v = np.asarray(v, np.float64).ravel()
        return PhysicalIntent(*(float(x) for x in v[:7]))

    def clip(self) -> "PhysicalIntent":
        return PhysicalIntent.from_vector(np.clip(self.to_vector(), INTENT_LO, INTENT_HI))


def extract_teacher_intent(snap: CradleSnapshot, theta_phys: np.ndarray, cfg: Any = DELIVERY_CFG) -> "tuple[PhysicalIntent, dict[str, Any]]":
    """Deterministically read the physical intent off a teacher (or any) physical theta at ``snap``. Canonicalise the state
    + theta (mirror-invariant), authority-normalise the 6 theta-determining roles, and take peak_velocity from the physical
    rollout. # Preconditions: theta_phys length 6, legal. # Postconditions: returns (clipped intent, provenance) with
    decode(extract) ≈ theta on the 6 invertible roles. NO held-out-derived or new features enter."""
    from hymeko_rl.coin_delivery.forward_displacement import rollout_primitive
    theta_phys = np.asarray(theta_phys, np.float64)
    g_canon, sw = canonicalise(r1_grouped_features(snap))
    theta_canon = np.asarray(to_canonical_theta(theta_phys, sw), np.float64)
    slew, H = slew_of(snap), float(cfg.horizon)
    fr, br, lr, nf = authority_scalars(g_canon)
    m = rollout_primitive(snap, tuple(theta_phys), cfg)
    raw = PhysicalIntent(forward_drive=theta_canon[1] * fr / slew, peak_velocity=float(m["peak_coin_speed"]),
                         lateral=theta_canon[2] * lr / slew, squeeze=theta_canon[0] * nf / slew,
                         brake_entry=theta_canon[3] / H, braking_demand=theta_canon[5] * br / slew,
                         release=theta_canon[4] / H)
    intent = raw.clip()
    prov = {"was_swapped": bool(sw), "slew": round(slew, 5), "horizon": H,
            "authority": {"forward_push_reach": round(fr, 5), "brake_opposed_reach": round(br, 5),
                          "lateral_reach": round(lr, 5), "normal_force_reach": round(nf, 5)},
            "theta_phys": [round(float(x), 5) for x in theta_phys],
            "theta_canonical": [round(float(x), 5) for x in theta_canon],
            "intent_raw": [round(float(x), 5) for x in raw.to_vector()],
            "intent_clipped": [round(float(x), 5) for x in intent.to_vector()],
            "clipped_roles": [INTENT_ROLES[i] for i in range(7) if not np.isclose(raw.to_vector()[i], intent.to_vector()[i])],
            "k6_peak_coin_speed": round(float(m["peak_coin_speed"]), 5), "k6_delivered": bool(m["k6_delivered"])}
    return intent, prov
