"""R7 — velocity-regulated held-contact transport + saturating non-reversing stop.

R6 proved the frozen option's per-step map (open-loop accelerating push + opposing-velocity brake) cannot hold a bounded
transport speed nor arrest at zero (blow-up / reversal). R7 replaces ONLY that per-step control law with a velocity servo on
the MEASURED object velocity, inside the same frozen physics (`step_ablation` + `govern_torque` + slew/motion contract + K6)
and reusing the R6 release certificate + the frozen force DIRECTIONS (`arm_jac_dir` toward the zone; `arm_inward_geom` grip).

Servo (target-directed coin velocity; the reference decays to zero at the zone):

    v_ref = clip(k_d · d_remain, 0, v_max) ;   a_cmd = clip(k_v·(v_ref − v∥), −a_stop_max, a_push_max)

with a MANDATORY non-reversing stop — positive motion may go to rest but NEVER to escape. `a_cmd` drives the along-track
force direction (scaled to a slew-admissible Δτ; the closed loop corrects the scale), plus a minimum grip that decays once
the coin is settled; RELEASE only from a certified rest (`release_certificate`). One continuous trajectory; NEVER a teacher θ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot, primary_fingertip_contacts
from hymeko_rl.coin_delivery.forward_displacement import ForwardConfig, _coin_speed, _coin_xy, arm_jac_dir
from hymeko_rl.coin_delivery.mobile_conditioning import _fingertip_geoms, arm_inward_geom
from hymeko_rl.coin_delivery.theta_option.release_certificate import ReleaseCertMonitor, ReleaseCertParams
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.env.motion_contract import govern_torque

_LEFT_DOF, _RIGHT_DOF = (0, 1), (2, 3)
_EPS = 1e-9
REGULATE, RELEASE = 0, 1


# ── V0: pure servo algebra (physics-free, unit-tested) ───────────────────────────────────────────────────────────────
def velocity_ref(d_remain: float, k_d: float, v_max: float) -> float:
    """Target-directed reference speed that DECAYS to zero at the zone. # Postconditions: ∈ [0, v_max]; 0 at d_remain ≤ 0."""
    return float(np.clip(float(k_d) * max(float(d_remain), 0.0), 0.0, float(v_max)))


def accel_cmd(v_ref: float, v_par: float, k_v: float, a_push_max: float, a_stop_max: float) -> float:
    """Servo acceleration command, saturated. # Postconditions: ∈ [−a_stop_max, a_push_max]."""
    return float(np.clip(float(k_v) * (float(v_ref) - float(v_par)), -float(a_stop_max), float(a_push_max)))


def non_reversing_accel(a_cmd: float, v_par: float, dt: float, v_deadband: float = 0.0) -> float:
    """Saturating NON-REVERSING stop: if positive motion would be turned negative next step, clamp so it reaches (at most)
    the deadband, never an escape. # Postconditions: if v_par > 0 then v_par + result·dt ≥ min(v_par, v_deadband) ≥ 0;
    otherwise returns a_cmd unchanged."""
    if float(v_par) > 0.0 and (float(v_par) + float(a_cmd) * float(dt)) < 0.0:
        return -max(float(v_par) - float(v_deadband), 0.0) / max(float(dt), _EPS)
    return float(a_cmd)


@dataclass
class VelocityTransportParams:
    """FROZEN (dev-tuned) R7 servo parameters; the release certificate carries its own tolerances."""

    k_d: float = 4.0                        # 1/s — v_ref = k_d·d_remain (decays to 0 at the zone)
    v_max: float = 0.35                     # m/s — bounded transport speed
    k_v: float = 12.0                       # 1/s — velocity-servo gain
    a_push_max: float = 4.0                 # m/s² — max forward accel command
    a_stop_max: float = 8.0                 # m/s² — max decel command
    v_deadband: float = 0.0                 # m/s — the non-reversing target (0 ⇒ stop exactly, never reverse)
    k_tau: float = 0.06                     # N·m per (m/s²) — a_cmd → Δτ scale along the force direction (servo corrects)
    settle_speed: float = 0.05              # m/s — below this, in-zone ⇒ decay the grip
    squeeze_min: float = 0.06               # N·m — minimum grip to retain contact
    squeeze_hold: float = 0.14              # N·m — grip during transport
    squeeze_decay: float = 0.8              # per-step multiplicative grip decay in the settle
    cert: ReleaseCertParams = field(default_factory=ReleaseCertParams)


class VelocityTransportController:
    """The R7 velocity-servo controller: `dtau_for_step(rl, t, prev_tau)` returns the slew-admissible Δτ. HELD_REGULATE
    (servo transports + non-reversingly stops the coin, grip held) until the coin is settled in-zone, then the grip decays
    and RELEASE latches from the R6 certificate. # Postconditions: |Δτ| ≤ slew; never a teacher θ; grip held until certified
    release; monotone REGULATE→RELEASE."""

    def __init__(self, snap: CradleSnapshot, params: VelocityTransportParams = VelocityTransportParams(),
                 cfg: Any = DELIVERY_CFG) -> None:
        self.snap, self.p, self.cfg = snap, params, cfg
        self.slew = float(snap.stack.tau_rate * snap.stack.control_dt)
        self.dt = float(snap.stack.control_dt)
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
        """Reuse the frozen force directions: forward = analytic tip-Jacobian toward the zone; squeeze = inward grip."""
        gl, gr = _fingertip_geoms(rl.inner.model)
        fwd = arm_jac_dir(rl, gl, _LEFT_DOF, e_par) + arm_jac_dir(rl, gr, _RIGHT_DOF, e_par)
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
        armed, cdiag = self.cert.update(rl, t)
        if armed:
            self.phase = RELEASE
        fwd_dir, sqz_dir = self._directions(rl, e_par, coin)
        if self.phase == RELEASE:
            dtau = -0.5 * self._sqz * sqz_dir                  # relax grip; the settled coin stays
        else:
            d_remain = max(dtz - CENTER_TOL, 0.0)
            v_ref = velocity_ref(d_remain, self.p.k_d, self.p.v_max)
            a = accel_cmd(v_ref, v_par, self.p.k_v, self.p.a_push_max, self.p.a_stop_max)
            a = non_reversing_accel(a, v_par, self.dt, self.p.v_deadband)
            settling = (dtz < CENTER_TOL) and (abs(v_par) < self.p.settle_speed)
            if settling and not contact_risk:                  # SQUEEZE_DECAY: drop grip toward the minimum
                self._sqz = max(self.p.squeeze_min, self._sqz * self.p.squeeze_decay)
            elif contact_risk:
                self._sqz = self.p.squeeze_hold
            dtau = self.p.k_tau * a * fwd_dir + self._sqz * sqz_dir
        dtau = np.clip(np.asarray(dtau, np.float64), -self.slew, self.slew)
        self.trace.append({"t": int(t), "phase": int(self.phase), "dtz_mm": round(dtz * 1000, 2),
                           "v_par": round(v_par, 5), "sqz": round(self._sqz, 4), "cert_streak": cdiag["streak"]})
        return dtau


def _new_acc() -> "dict[str, Any]":
    return {"peak_qdot": 0.0, "peak_coin_speed": 0.0, "terminal_speed": 0.0, "contact_lost_steps": 0,
            "lost_before_release": 0, "straddle_min": 1.0, "min_fn_push": 1e9, "forward_at_release": 0.0,
            "k6_dwell": 0, "k6_max_dwell": 0, "touched": False}


def _accumulate(acc: "dict[str, Any]", rl: Any, t: int, before_rel: bool, rb: "int | None", coin0: np.ndarray,
                e_par: np.ndarray) -> None:
    """The per-step K6 / peak / contact accounting (identical to the frozen rollout), factored out of the loop."""
    d = rl.inner.data
    acc["peak_qdot"] = max(acc["peak_qdot"], float(np.max(np.abs(d.qvel[:4]))))
    speed_t = _coin_speed(rl)
    acc["peak_coin_speed"] = max(acc["peak_coin_speed"], speed_t)
    acc["terminal_speed"] = speed_t
    _u_t, dtz_t = rl.inner.direction_to_zone()
    con = primary_fingertip_contacts(rl)
    both = con["left"] is not None and con["right"] is not None
    acc["touched"] = acc["touched"] or both
    acc["k6_dwell"] = acc["k6_dwell"] + 1 if (dtz_t <= CENTER_TOL and speed_t < SETTLE_VEL) else 0
    acc["k6_max_dwell"] = max(acc["k6_max_dwell"], acc["k6_dwell"])
    if both:
        acc["straddle_min"] = min(acc["straddle_min"], float(con["left"]["n"] @ con["right"]["n"]))
        if before_rel:
            acc["min_fn_push"] = min(acc["min_fn_push"], con["left"]["fn"], con["right"]["fn"])
    else:
        acc["contact_lost_steps"] += 1
        if before_rel:
            acc["lost_before_release"] += 1
    if rb is not None and t == rb:
        acc["forward_at_release"] = float((_coin_xy(rl) - coin0) @ e_par)


def velocity_rollout(snap: CradleSnapshot, controller: Any, cfg: ForwardConfig, *, frame_hook: Any = None) -> dict:
    """Run the R7 velocity-servo controller through the FROZEN physics (governed step + K6 monitor), the controller
    returning Δτ directly. Reuses the exact K6 / peak / contact accounting of the frozen rollout. # Postconditions: a
    rollout_primitive-shaped metrics dict; one continuous rl; no pin/teleport/coin edit."""
    controller.reset()
    rl = snap.branch()
    e_par, dtz_start = rl.inner.direction_to_zone()
    e_par = np.asarray(e_par, np.float64)
    e_cross = np.array([-e_par[1], e_par[0]], np.float64)
    coin0 = _coin_xy(rl)
    prev_tau = snap.prev_tau.copy()
    trace, acc = [], _new_acc()

    def _gcb(_mo: Any, dt2: Any) -> None:
        dt2.ctrl[:4] = govern_torque(dt2.ctrl[:4], dt2.qvel[:4], snap.stack.gov)
    mujoco.set_mjcb_control(_gcb)
    try:
        for t in range(1, cfg.horizon + 1):
            dtau = controller.dtau_for_step(rl, t, prev_tau)
            prev_tau = np.clip(prev_tau + dtau, snap.lo, snap.hi)
            step_ablation(rl, np.asarray(prev_tau, np.float32), "A")
            trace.append(_coin_xy(rl).copy())
            _accumulate(acc, rl, t, controller.before_release(t), controller.release_boundary_step, coin0, e_par)
            if frame_hook is not None:
                frame_hook(rl, t)
    finally:
        mujoco.set_mjcb_control(None)
    disp = _coin_xy(rl) - coin0
    _u_end, dtz_end = rl.inner.direction_to_zone()
    rb = controller.release_boundary_step
    return {"forward": float(disp @ e_par), "cross": float(abs(disp @ e_cross)), "total_disp": float(np.linalg.norm(disp)),
            "dtz_start": float(dtz_start), "dtz_end": float(dtz_end),
            "gap_closed": float((dtz_start - dtz_end) / max(dtz_start, 1e-9)),
            "k6_max_dwell": int(acc["k6_max_dwell"]),
            "k6_delivered": bool(acc["k6_max_dwell"] >= HELD_DWELL and acc["touched"]), "touched": bool(acc["touched"]),
            "forward_at_release": acc["forward_at_release"], "peak_qdot": acc["peak_qdot"],
            "peak_coin_speed": acc["peak_coin_speed"], "terminal_coin_speed": acc["terminal_speed"],
            "contact_lost_steps": acc["contact_lost_steps"], "lost_before_release": acc["lost_before_release"],
            "release_step": int(rb) if rb is not None else cfg.horizon,
            "straddle_min": (None if acc["straddle_min"] == 1.0 else acc["straddle_min"]),
            "min_fn_push": (None if acc["min_fn_push"] == 1e9 else float(acc["min_fn_push"])),
            "coin_trace": [c.tolist() for c in trace]}
