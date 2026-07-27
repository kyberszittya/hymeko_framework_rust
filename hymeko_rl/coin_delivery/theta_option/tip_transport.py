"""R8 — the SAFE tip-referenced transport scaffold (the base the bounded residual RL sits on).

R7 proved a servo on the *coin* velocity through a soft frictional grip does not regulate (the coin is dragged by the tips
and blows up / reverses). R8 regulates the **directly-actuated joint (tip) velocity** instead: a torque-TARGET joint-velocity
servo with slew anti-windup drives the joints toward a bounded reference that decays to zero at the zone, so the tips — and
the gripped coin following them — stay bounded and stop WITHOUT reversing. The grip is the minimum that retains contact and
decays as the coin settles; the R6 release certificate is a SHIELD that decides when release is safe (it does not carry the
coin to goal). This is the S1 scaffold; a bounded residual actor (S2+) later adds `clip(δu, ±Δ)` on top.

Servo (per control step): `qdot_ref_mag = clip(k_q · v_ref, 0, qdot_max)` toward the zone joint-direction; the torque
increment tracks it, `Δτ = clip(k_v·(qdot_ref − qdot), ±slew)` (slew IS the anti-windup). Reuses the R7 `velocity_ref`, the
frozen force directions, the R6 certificate, and the `velocity_rollout` driver (frozen governed physics + K6). No teacher θ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL
from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot, primary_fingertip_contacts
from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, arm_jac_dir
from hymeko_rl.coin_delivery.mobile_conditioning import _fingertip_geoms, arm_inward_geom
from hymeko_rl.coin_delivery.theta_option.release_certificate import ReleaseCertMonitor, ReleaseCertParams
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_ref

_LEFT_DOF, _RIGHT_DOF = (0, 1), (2, 3)
REGULATE, RELEASE = 0, 1


@dataclass
class TipTransportParams:
    """FROZEN (dev-tuned) tip-referenced servo parameters; the release certificate carries its own tolerances."""

    k_d: float = 4.0                        # 1/s — coin approach-speed reference v_ref = k_d·d_remain
    v_max: float = 0.30                     # m/s — bounded approach speed
    k_q: float = 8.0                        # (rad/s)/(m/s) — coin-speed → joint-speed reference scale
    qdot_max: float = 2.0                   # rad/s — hard cap on the joint-velocity reference (≤ joint_vel_safe)
    k_v: float = 0.6                        # N·m per (rad/s) — joint-velocity servo gain (torque increment)
    settle_speed: float = 0.05              # m/s — below this, in-zone ⇒ decay the grip
    squeeze_min: float = 0.06               # N·m — minimum grip to retain contact
    squeeze_hold: float = 0.14              # N·m — grip during transport
    squeeze_decay: float = 0.8              # per-step multiplicative grip decay in the settle
    cert: ReleaseCertParams = field(default_factory=ReleaseCertParams)


class TipReferencedController:
    """Safe tip-referenced (joint-velocity) transport scaffold. `dtau_for_step(rl, t, prev_tau)` returns a slew-admissible
    Δτ that tracks a bounded, zone-decaying joint-velocity reference; grip decays in the settle; RELEASE from the R6
    certificate. # Postconditions: |Δτ| ≤ slew; joint-velocity reference ≤ qdot_max and → 0 at the zone (bounded,
    non-reversing); never a teacher θ; monotone REGULATE→RELEASE."""

    def __init__(self, snap: CradleSnapshot, params: TipTransportParams = TipTransportParams(),
                 cfg: Any = DELIVERY_CFG) -> None:
        self.snap, self.p, self.cfg = snap, params, cfg
        self.slew = float(snap.stack.tau_rate * snap.stack.control_dt)
        self.reset()

    def reset(self) -> None:
        self.phase = REGULATE
        self.cert = ReleaseCertMonitor(self.p.cert)
        self._sqz = self.p.squeeze_hold
        self.trace: list[dict[str, Any]] = []

    def before_release(self, t: int) -> bool:
        return self.phase < RELEASE

    @property
    def release_boundary_step(self) -> "int | None":
        return self.cert.armed_at

    def _live(self, rl: Any) -> "tuple[np.ndarray, float, float]":
        u, dtz = rl.inner.direction_to_zone()
        e_par = np.asarray(u, np.float64)[:2]
        n = float(np.linalg.norm(e_par))
        e_par = e_par / n if n > 1e-9 else np.array([1.0, 0.0])
        v = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
        return e_par, float(dtz), float(v @ e_par)

    def _directions(self, rl: Any, e_par: np.ndarray, coin: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        gl, gr = _fingertip_geoms(rl.inner.model)
        fwd = arm_jac_dir(rl, gl, _LEFT_DOF, e_par) + arm_jac_dir(rl, gr, _RIGHT_DOF, e_par)   # joint dir moving tips → zone
        fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
        inward = arm_inward_geom(rl, gl, _LEFT_DOF, coin) + arm_inward_geom(rl, gr, _RIGHT_DOF, coin)
        sqz = inward / (np.linalg.norm(inward) + 1e-12)
        return fwd, sqz

    def dtau_for_step(self, rl: Any, t: int, prev_tau: np.ndarray) -> np.ndarray:
        e_par, dtz, v_par = self._live(rl)
        coin = _coin_xy(rl)
        con = primary_fingertip_contacts(rl)
        both = con["left"] is not None and con["right"] is not None
        contact_risk = (not both) or (min(float(con["left"]["fn"]) if con["left"] else 0.0,
                                          float(con["right"]["fn"]) if con["right"] else 0.0) < 0.05)
        armed, _cdiag = self.cert.update(rl, t)
        if armed:
            self.phase = RELEASE
        fwd_dir, sqz_dir = self._directions(rl, e_par, coin)
        if self.phase == RELEASE:
            dtau = -0.5 * self._sqz * sqz_dir                  # relax grip; the settled coin stays
        else:
            d_remain = max(dtz - CENTER_TOL, 0.0)
            v_ref = velocity_ref(d_remain, self.p.k_d, self.p.v_max)          # coin approach-speed reference → 0 at zone
            qdot_ref_mag = float(np.clip(self.p.k_q * v_ref, 0.0, self.p.qdot_max))  # bounded joint-velocity reference
            qdot_ref = qdot_ref_mag * fwd_dir
            qdot = np.asarray(rl.inner.data.qvel[:4], np.float64)
            dtau = self.p.k_v * (qdot_ref - qdot)              # torque-target joint-velocity servo (slew = anti-windup)
            settling = (dtz < CENTER_TOL) and (abs(v_par) < self.p.settle_speed)
            if settling and not contact_risk:
                self._sqz = max(self.p.squeeze_min, self._sqz * self.p.squeeze_decay)
            elif contact_risk:
                self._sqz = self.p.squeeze_hold
            dtau = dtau + self._sqz * sqz_dir
        dtau = np.clip(np.asarray(dtau, np.float64), -self.slew, self.slew)
        self.trace.append({"t": int(t), "phase": int(self.phase), "dtz_mm": round(dtz * 1000, 2),
                           "v_par": round(v_par, 5), "qdot_max": round(float(np.max(np.abs(rl.inner.data.qvel[:4]))), 4),
                           "sqz": round(self._sqz, 4)})
        return dtau
