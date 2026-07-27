"""R6 — the release certificate (the load-bearing new element of grip-held brake-to-stop).

The pin/preload audit showed `v_coin ≈ 0 while gripped` is NOT a safe release condition: zero coin velocity can hide a large
contact force, stored elastic energy, a non-null fingertip object wrench, or actuator preload — and on release the coin
shoots out. So the controller may leave the observable HELD mode only from a certified rest state, verified over several
consecutive frames:

    C_release = C_zone ∧ C_velocity ∧ C_spin ∧ C_wrench ∧ C_contact_vel ∧ C_qdot ∧ C_squeeze_decayed

All quantities are read from the live rl through the SAME causal channels the K6 monitor and ResponseState already use
(`direction_to_zone`, the planar metrics, `primary_fingertip_contacts`, `measure_contact_velocities`, `qvel`, `qacc`); no
future value, no K6 outcome, no teacher. The "fingertip-only object wrench" is proxied by (a) the coin's residual
acceleration `‖qacc_coin‖` (net wrench / mass — near-zero ⇒ force equilibrium) and (b) the grip-pressure imbalance
`|fn_L − fn_R|` and magnitude `max(fn)` (a decayed, balanced grip stores little energy to eject).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL
from hymeko_rl.coin_delivery.contact_velocity import (
    frozen_contact_frames, measure_contact_velocities, primary_fingertip_contacts)
from hymeko_rl.coin_delivery.forward_displacement import _coin_speed


@dataclass(frozen=True)
class ReleaseCertParams:
    """FROZEN (dev-tuned) release-certificate tolerances. Tighter than K6 where it guards ejection; held-out never tunes."""

    zone_tol: float = CENTER_TOL            # m — coin inside the delivery zone (the K6 zone)
    settle_tol: float = 0.04               # m/s — coin speed below this (tighter than K6's 0.06)
    spin_tol: float = 0.6                  # rad/s — small disk spin
    qacc_tol: float = 8.0                  # m/s² — coin residual accel (net-wrench/mass proxy) ≈ force equilibrium
    fn_max_tol: float = 0.9                # N — grip pressure decayed (little stored elastic energy)
    fn_balance_tol: float = 0.7            # N — |fn_L − fn_R| balanced grip (no net normal push)
    crv_tol: float = 0.05                  # m/s — contact-relative velocity small (not sliding / separating)
    qdot_tol: float = 0.6                  # rad/s — arm joints quiet
    n_frames: int = 3                      # consecutive frames the certificate must hold before RELEASE


def _coin_qacc(rl: Any) -> float:
    """‖coin planar acceleration‖ = net fingertip wrench / mass proxy; near-zero ⇒ the coin is in force equilibrium."""
    d = rl.inner.data
    ix, iy = int(rl.inner._disk_dofx), int(rl.inner._disk_dofy)
    return float(np.hypot(d.qacc[ix], d.qacc[iy])) if iy < d.qacc.shape[0] else 0.0


def _coin_spin(rl: Any) -> float:
    d = rl.inner.data
    idx = int(rl.inner._disk_dofy) + 1
    return float(d.qvel[idx]) if idx < d.qvel.shape[0] else 0.0


def release_certificate(rl: Any, p: ReleaseCertParams = ReleaseCertParams()) -> "tuple[bool, dict[str, Any]]":
    """Evaluate the per-frame release predicate on the live rl. # Postconditions: (ok, per-condition dict); ok ⟺ ALL of
    zone/velocity/spin/wrench/contact-vel/qdot/squeeze-decayed hold; uses only past+present physical signals."""
    _u, dtz = rl.inner.direction_to_zone()
    speed = _coin_speed(rl)
    spin = abs(_coin_spin(rl))
    qacc = _coin_qacc(rl)
    qdot = float(np.max(np.abs(rl.inner.data.qvel[:4])))
    con = primary_fingertip_contacts(rl)
    both = con["left"] is not None and con["right"] is not None
    fn = (float(con["left"]["fn"]) if con["left"] is not None else 0.0,
          float(con["right"]["fn"]) if con["right"] is not None else 0.0)
    crv = 0.0
    try:
        per = measure_contact_velocities(rl, frozen_contact_frames(rl))["per"]
        crv = max(abs(float(per[s]["v_n"])) + abs(float(per[s]["v_t"])) for s in ("left", "right") if per[s] is not None)
    except Exception:                                          # contact identity churn ⇒ treat as unknown (conservative fail)
        crv = 1e9
    cond = {"C_zone": bool(dtz < p.zone_tol), "C_velocity": bool(speed < p.settle_tol), "C_spin": bool(spin < p.spin_tol),
            "C_wrench": bool(qacc < p.qacc_tol and abs(fn[0] - fn[1]) < p.fn_balance_tol),
            "C_contact_vel": bool(crv < p.crv_tol), "C_qdot": bool(qdot < p.qdot_tol),
            "C_squeeze_decayed": bool(max(fn) < p.fn_max_tol and both)}
    ok = all(cond.values())
    diag = {**cond, "dtz_mm": round(float(dtz) * 1000, 2), "speed": round(speed, 4), "spin": round(spin, 4),
            "qacc": round(qacc, 3), "qdot": round(qdot, 4), "fn": [round(x, 3) for x in fn], "crv": round(min(crv, 9.9), 4)}
    return bool(ok), diag


class ReleaseCertMonitor:
    """Latches RELEASE only after the certificate holds for `n_frames` CONSECUTIVE frames (a transient in-zone touch is not a
    certified rest). Monotone: once `armed`, it stays armed. # Postconditions: `armed` ⇒ the certificate held n_frames in a
    row at least once."""

    def __init__(self, params: ReleaseCertParams = ReleaseCertParams()) -> None:
        self.p = params
        self.streak = 0
        self.armed = False
        self.armed_at: int | None = None

    def update(self, rl: Any, t: int) -> "tuple[bool, dict[str, Any]]":
        ok, diag = release_certificate(rl, self.p)
        self.streak = self.streak + 1 if ok else 0
        if self.streak >= self.p.n_frames and not self.armed:
            self.armed, self.armed_at = True, int(t)
        return self.armed, {"t": int(t), "streak": int(self.streak), "armed": self.armed, **diag}
