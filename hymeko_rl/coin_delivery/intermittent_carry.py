"""Intermittent-contact coin delivery controller — the option language that FITS the coin's real contact class
(push-and-coast), for the frozen COIN_DYNAMICS_CONTRACT_V4_INTERMITTENT_CONTACT.

The continuous contact-retaining transport (C1) fought the physics: a single tip on a low-mass / low-friction disk cannot
hold continuous contact (the coin squirts away). The intermittent controller instead uses many SHORT, LOW-IMPULSE,
feedback-driven pushes with re-contact — the macro FSM:

    IMPULSE  (acquire → a bounded push toward the zone, then release)
      → COAST   (retract the tip; OBSERVE the coin sliding; do not touch)
      → RE-CONTACT (coin stalled short of the zone → re-acquire → IMPULSE)     [ablation: enable_recontact]
      → BRAKE   (coin near the zone with excess speed → a braking contact)     [ablation: enable_brake]
      → SETTLE  (coin in the zone → low-speed settling → frozen pi_0)          [ablation: enable_settle]

Everything goes through the SHARED, physically-inescapable GovernedArm stack (position targets → pd_governed_torque →
per-sub-step governor + over_hard_brake → saturation). No raw-torque bypass. The C2 ablation progressively enables the
options (impulse-only → +coast → +re-contact → +brake → +settle) to isolate which element bears the transport.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_carry_structured import (
    CENTER_TOL, HELD_DWELL, SETTLE_VEL, _aug, _det, stable_engagement_signals)
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.motion_robust_expert import (
    _acquire_direction, _contact_forces, _contact_jacobian, _jac_solve, _unit)
from hymeko_rl.env.governed_arm import pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque


@dataclass(frozen=True)
class IntermittentConfig:
    """Impulse / coast / re-contact / brake / settle knobs. The C2 ablation toggles the three ``enable_*`` flags; the
    numeric knobs are shared so an arm differs ONLY by which options are on (a clean progressive ablation)."""

    impulse_disp: float = 0.06         # rad/step — bounded push displacement toward the zone during an IMPULSE
    impulse_steps: int = 8             # a SHORT push burst (low impulse), then release — the anti-squirt primitive
    coast_min_speed: float = 0.02      # m/s — below this the coasting coin has stalled ⇒ re-contact / re-impulse
    coast_max_steps: int = 40          # give the coin this long to coast before forcing a decision
    retract_disp: float = 0.08         # rad/step — retract the tip away from the coin during COAST (break contact)
    brake_dist: float = 1.5 * CENTER_TOL   # m — enter BRAKE when the coin is this close to the zone …
    brake_speed: float = 1.5 * SETTLE_VEL  # … and still moving faster than this
    brake_disp: float = 0.05           # rad/step — braking displacement opposing the coin velocity
    release_band: float = 1.5 * SETTLE_VEL
    replan_every: int = 4
    enable_recontact: bool = True      # C2 ablation: re-acquire after a stalled coast (off ⇒ single impulse+coast)
    enable_brake: bool = True          # C2 ablation: velocity-aware braking near the zone
    enable_settle: bool = True         # C2 ablation: low-speed settling in the zone
    probe_mag: float = 2.0
    probe_steps: int = 4


def intermittent_carry(rl, gate, pi0, base, stack, *, horizon: int, cfg: IntermittentConfig | None = None,
                       frame_hook=None):
    """Run the intermittent-contact delivery FSM on the shared V4 stack. Returns the K6/handoff outcome + phase-ladder
    metrics (acquire / transport distance / zone entry / release / settle) + realised motion (contact-conditioned peak
    velocity, impulse, coin speed, episodes, governor/brake/saturation) — the SAME metric keys as motion_robust_carry
    plus delivery metrics, so the C2 harness reads both controllers uniformly. K6/zone are OUTPUTS, never inputs."""
    cfg = cfg or IntermittentConfig()
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4] = stack.armature
    m.dof_damping[:4] = stack.damping
    m.dof_frictionloss[:4] = stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gov = stack.gov
    subs = {"n": 0, "gov": 0, "brake": 0}

    def _gcb(_model, data):
        raw = data.ctrl[:4].copy()
        data.ctrl[:4] = govern_torque(raw, data.qvel[:4], gov)
        subs["n"] += 1
        subs["gov"] += int(not np.allclose(data.ctrl[:4], raw))
        subs["brake"] += int(bool(np.any(np.abs(data.qvel[:4]) > gov.qdot_hard)))
    mujoco.set_mjcb_control(_gcb)

    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    u0, _n0 = rl.inner.direction_to_zone()
    zone_u = np.asarray(u0, np.float32)
    handed, md, touched, prev_tau, J = False, int(rl._strict), rl._touched, None, None
    state, impulse_t, coast_t = "IMPULSE", 0, 0
    phases = {"ACQUIRE": 0, "IMPULSE": 0, "COAST": 0, "RECONTACT": 0, "BRAKE": 0, "SETTLE": 0}
    peak_vel, peak_vel_c, integ_c, peak_fn, peak_ft, impulse = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    peak_coin, ctrl_frames, sat_frames, n_episodes, prev_contact = 0.0, 0, 0, 0, False
    contact_frames, entered_zone, transport_dist, coin_speed = 0, False, 0.0, 0.0
    t = 0
    try:
        for t in range(horizon):
            s = int(rl._strict)
            mpl = rl.inner._planar_metrics
            contact = bool(mpl.left_contact or mpl.right_contact)
            coin_vel = np.asarray(mpl.disk_vel, np.float32)[:2]
            coin_speed = float(np.linalg.norm(coin_vel))
            dtz = rl._dtz()
            if gate.gate != 1.0:
                a, cur = _det(pi0, rl.obs()), "PI0"
            elif handed or s >= 1:
                handed, cur = True, "SETTLE"
                a = _det(base, _aug(rl.obs(), s))
            else:
                if J is None or t % cfg.replan_every == 0:
                    J = _contact_jacobian(rl, cfg)
                delta, cur, state, impulse_t, coast_t = _fsm_delta(
                    rl, J, cfg, zone_u, dtz, contact, coin_vel, coin_speed, state, impulse_t, coast_t)
                q, qd = d.qpos[:4].copy(), d.qvel[:4].copy()
                a = pd_governed_torque(q, qd, q + delta, stack, prev_tau, lo, hi)
                prev_tau = a
                ctrl_frames += 1
                sat_frames += int(bool(np.any((a <= lo + 1e-6) | (a >= hi - 1e-6))))
                phases[cur] = phases.get(cur, 0) + 1
            _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
            gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            md = max(md, int(rl._strict))
            touched = touched or rl._touched
            v = float(np.max(np.abs(d.qvel[:4])))
            peak_vel = max(peak_vel, v)
            now_contact = bool(rl.inner._planar_metrics.left_contact or rl.inner._planar_metrics.right_contact)
            contact_frames += int(now_contact)
            peak_coin = max(peak_coin, float(np.linalg.norm(np.asarray(rl.inner._planar_metrics.disk_vel, np.float32)[:2])))
            if now_contact:
                peak_vel_c = max(peak_vel_c, v)
                integ_c += max(0.0, v / 2.0 - 1.0) * stack.control_dt
                fn, ft = _contact_forces(m, d)
                peak_fn, peak_ft = max(peak_fn, fn), max(peak_ft, ft)
                impulse += fn * stack.control_dt
                if not prev_contact:
                    n_episodes += 1
            prev_contact = now_contact
            disp = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2] - disk0
            transport_dist = max(transport_dist, float(disp @ zone_u))    # signed progress toward the zone
            entered_zone = entered_zone or (rl._dtz() <= CENTER_TOL)
            if frame_hook is not None:
                frame_hook(cur, int(rl._strict))
            if term or trunc:
                break
    finally:
        mujoco.set_mjcb_control(None)
    n = max(1, t + 1)
    nsub = max(1, subs["n"])
    return {"k6": int(md >= HELD_DWELL and touched), "max_dwell": md, "reached_handoff": int(md >= 1),
            "touched": int(touched), "acquired_contact": int(contact_frames > 0), "contact_frac": round(contact_frames / n, 3),
            "entered_zone": int(entered_zone), "transport_dist": round(transport_dist, 4), "phases": phases, "steps": n,
            "contact_frames": contact_frames, "n_contact_episodes": n_episodes, "peak_joint_vel": round(peak_vel, 2),
            "peak_joint_vel_in_contact": round(peak_vel_c, 2), "integrated_overspeed_in_contact": round(integ_c, 3),
            "peak_contact_normal_force": round(peak_fn, 2), "peak_contact_tangential_force": round(peak_ft, 2),
            "contact_normal_impulse": round(impulse, 4), "peak_coin_speed": round(peak_coin, 3),
            "terminal_coin_speed": round(coin_speed, 3), "terminal_joint_vel": round(float(np.max(np.abs(d.qvel[:4]))), 3),
            "governor_active_frac": round(subs["gov"] / nsub, 3), "active_brake_frac": round(subs["brake"] / nsub, 3),
            "torque_saturation_frac": round(sat_frames / max(1, ctrl_frames), 3)}


def _fsm_delta(rl, J, cfg: IntermittentConfig, zone_u, dtz, contact, coin_vel, coin_speed, state, impulse_t, coast_t):
    """One macro step of the intermittent FSM → (Δq position-target, phase label, next state, impulse_t, coast_t).
    Delivery uses only the coin→zone geometry (direction_to_zone), never K6. Ablation flags gate RECONTACT/BRAKE/SETTLE."""
    # SETTLE: in the zone and slow — hold (damping brings the coin to rest); only if settling is enabled
    if cfg.enable_settle and dtz <= CENTER_TOL and coin_speed < cfg.release_band:
        return np.zeros(4, np.float32), "SETTLE", "SETTLE", 0, 0
    # BRAKE: near the zone with excess coin speed — a braking contact opposing the coin velocity
    if cfg.enable_brake and dtz <= cfg.brake_dist and coin_speed > cfg.brake_speed:
        return cfg.brake_disp * _unit(-_jac_solve(J, coin_vel)), "BRAKE", "BRAKE", 0, 0
    if state == "COAST":
        # retract the tip and let the coin slide; decide when the coast is spent
        if coast_t >= cfg.coast_max_steps or coin_speed < cfg.coast_min_speed:
            nxt = "RECONTACT" if cfg.enable_recontact else "IMPULSE"
            return cfg.impulse_disp * _acquire_direction(rl, cfg), ("RECONTACT" if cfg.enable_recontact else "ACQUIRE"), nxt, 0, 0
        return -cfg.retract_disp * _acquire_direction(rl, cfg), "COAST", "COAST", 0, coast_t + 1   # move AWAY from the coin
    if state == "RECONTACT":
        if not contact:
            return cfg.impulse_disp * _acquire_direction(rl, cfg), "RECONTACT", "RECONTACT", 0, 0
        return cfg.impulse_disp * _unit(_jac_solve(J, zone_u)), "IMPULSE", "IMPULSE", 1, 0     # re-contacted → push
    # IMPULSE (default): acquire if not touching, else a short bounded push toward the zone, then COAST
    if not contact:
        return cfg.impulse_disp * _acquire_direction(rl, cfg), "ACQUIRE", "IMPULSE", 0, 0
    if impulse_t < cfg.impulse_steps:
        return cfg.impulse_disp * _unit(_jac_solve(J, zone_u)), "IMPULSE", "IMPULSE", impulse_t + 1, 0
    return -cfg.retract_disp * _acquire_direction(rl, cfg), "COAST", "COAST", 0, 0    # impulse spent → release into COAST
