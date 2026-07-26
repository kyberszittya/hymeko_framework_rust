"""R3 — the deterministic authority-aware decoder: physical intent + canonical geometry + measured authority -> canonical
6-D theta. This is the load-bearing R3 element: NOT another free regressor, but a physical inverse. The same cradle-
agnostic intent decodes to a cradle-SPECIFIC theta by dividing each authority-normalised demand by that cradle's measured
one-step authority (weak-authority cradle -> more torque), so the coupled mechanism (squeeze + forward drive + brake entry
+ brake magnitude + release) is reconstructed rather than memorised.

Object-motion authority (B_coin reachable: forward_push_reach, brake_opposed_reach, lateral_reach) drives the forward /
brake / lateral roles; contact/internal authority (B_τ null-space: normal_force_reach) drives squeeze — the two spaces stay
separate. Everything is in the canonical frame; `from_canonical_theta` returns the physical theta (theta[2] balance sign
flips iff the state was mirror-swapped). The decoder NEVER falls back to a teacher theta; it reports saturation /
infeasibility diagnostics instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.theta_option.canonical_frame import canonicalise, from_canonical_theta, r1_grouped_features
from hymeko_rl.coin_delivery.theta_option.physical_intent import PhysicalIntent, authority_scalars, slew_of
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG, ThetaBox

_WEAK_LINKS = ("push", "squeeze", "brake", "release")


@dataclass
class DecoderResult:
    """The decoded theta + the full deterministic diagnostics (no teacher fallback ever)."""

    canonical_theta: np.ndarray
    physical_theta: np.ndarray
    was_swapped: bool
    authority: "dict[str, float]"
    saturation: "dict[str, bool]"
    achievable_peak_estimate: float
    residual_per_role: "dict[str, float]"
    weak_link: str

    def as_dict(self) -> "dict[str, Any]":
        return {"canonical_theta": [round(float(x), 5) for x in self.canonical_theta],
                "physical_theta": [round(float(x), 5) for x in self.physical_theta],
                "was_swapped": bool(self.was_swapped), "authority": self.authority, "saturation": self.saturation,
                "achievable_peak_estimate": round(float(self.achievable_peak_estimate), 5),
                "residual_per_role": {k: round(float(v), 5) for k, v in self.residual_per_role.items()},
                "weak_link": self.weak_link,
                "any_saturation": bool(any(self.saturation.values()))}


def decode_from_canonical(intent: PhysicalIntent, g_canon: "dict[str, np.ndarray]", sw: bool, slew: float,
                          horizon: float) -> DecoderResult:
    """PURE (physics-free) deterministic core: canonical intent + canonical authority -> canonical theta -> physical theta.
    Split out so mirror/bounds/separation/monotonicity/determinism are testable without a snapshot. # Postconditions:
    physical_theta is a LEGAL 6-D theta; balance sign flips iff ``sw``; NEVER a teacher theta."""
    H = float(horizon)
    fr, br, lr, nf = authority_scalars(g_canon)
    box = ThetaBox()
    # authority-normalised inverse: canonical theta = demand * slew / authority (object: fr/br/lr; internal: nf); timings * H
    raw = np.array([intent.squeeze * slew / nf,          # theta[0] squeeze_mag  (internal)
                    intent.forward_drive * slew / fr,    # theta[1] forward_mag  (object)
                    intent.lateral * slew / lr,          # theta[2] balance      (object, canonical)
                    intent.brake_entry * H,              # theta[3] ramp_steps   (timing)
                    intent.release * H,                  # theta[4] release_step (timing)
                    intent.braking_demand * slew / br],  # theta[5] brake_gain   (object)
                   np.float64)
    theta_canon = np.asarray(box.clip(raw), np.float64)
    names = ("squeeze_mag", "forward_mag", "balance", "ramp_steps", "release_step", "brake_gain")
    saturation = {names[i]: bool(not np.isclose(raw[i], theta_canon[i], atol=1e-6)) for i in range(6)}
    theta_phys = np.asarray(from_canonical_theta(theta_canon, sw), np.float64)
    # peak_velocity is a load-bearing DIAGNOSTIC: a monotone reachable-speed proxy (forward authority x push duration x
    # forward-torque fraction) + the residual vs the requested peak, and the most-stressed role as the weak link.
    hi = np.asarray(box.hi, np.float64)
    achievable = fr * theta_canon[3] * (theta_canon[1] / max(hi[1], 1e-9))
    residual = {"peak_velocity": intent.peak_velocity - achievable,
                "forward_drive": raw[1] - theta_canon[1], "lateral": raw[2] - theta_canon[2],
                "squeeze": raw[0] - theta_canon[0], "braking_demand": raw[5] - theta_canon[5]}
    stress = np.array([raw[1] / hi[1], raw[0] / hi[0], raw[5] / hi[5], raw[4] / hi[4]])   # push, squeeze, brake, release
    weak_link = _WEAK_LINKS[int(np.argmax(stress))]
    return DecoderResult(canonical_theta=theta_canon, physical_theta=theta_phys, was_swapped=bool(sw),
                         authority={"forward_push_reach": round(fr, 5), "brake_opposed_reach": round(br, 5),
                                    "lateral_reach": round(lr, 5), "normal_force_reach": round(nf, 5)},
                         saturation=saturation, achievable_peak_estimate=float(achievable),
                         residual_per_role=residual, weak_link=weak_link)


def decode_intent(intent: PhysicalIntent, snap: CradleSnapshot, cfg: Any = DELIVERY_CFG) -> DecoderResult:
    """Deterministic authority-aware decode from a physical snapshot: canonicalise the R1 features, read the measured
    authority, and run the pure core. # Postconditions: identical output + diagnostics for identical (intent, snap); LEGAL
    physical theta; NEVER a teacher fallback."""
    g_canon, sw = canonicalise(r1_grouped_features(snap))
    return decode_from_canonical(intent, g_canon, sw, slew_of(snap), float(cfg.horizon))
