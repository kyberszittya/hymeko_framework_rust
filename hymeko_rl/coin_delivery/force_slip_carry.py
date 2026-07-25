"""FORCE_SLIP_SEMANTICS_V1 — a contact-level manipulation controller for the frozen RUBBER_TIP_LOW_DRAG_COIN_V2 physics.
The V1→V2 re-run showed the fingertip operates BELOW its friction limit (Ft/Fn ≈ 0.5, tip μ = 2.0) and imparts only
~0.2 m/s coin velocity: the position-following controller pushes the coin kinematically, it never creates a controlled
IMPULSE. This adds the missing contact semantics, as a progressive mechanistic ablation (each stage adds one element to
the one before), on frozen physics + frozen V4 motion contract — NO retraining, NO proposal, NO RL.

  S0  position-following reference (the current controller)
  S1  + bounded NORMAL PRELOAD      — regulate contact compression into a safe band (enough Fn, no 17 N spike)
  S2  + target-directed TANGENTIAL VELOCITY — drive the coin toward a launch velocity DERIVED FROM THE CALIBRATED COAST
        MODEL (v_launch = sqrt(2·μ·g·remaining_distance)), not a hand-tuned push time
  S3  + SLIP-AWARE modulation       — if the tip slips on the coin, raise preload (within band) rather than blindly push
  S4  + controlled IMPULSE + coast  — push until the coin reaches the target-velocity band (or an impulse/safety budget),
        then release and observe the coast
  S5  + PREDICTIVE BRAKING          — when remaining distance ≈ predicted stopping distance, re-contact to brake

Everything goes through the shared governed stack (position targets → pd_governed_torque → per-sub-step governor). The
launch-velocity target is set from the coast model, never from K6. Primary metrics: coin velocity at push end, target-
directed impulse, Ft/Fn, transport distance, stopping-distance prediction error.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_carry_structured import CENTER_TOL, HELD_DWELL, SETTLE_VEL, _aug, _det, stable_engagement_signals
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.motion_robust_expert import _acquire_direction, _contact_forces, _contact_jacobian, _jac_solve, _unit
from hymeko_rl.env.governed_arm import pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque

_G = 9.81


def _tip_world_velocity(m, d, tip_geoms) -> np.ndarray:
    """Mean planar (x,y) world velocity of the fingertip geoms — for the tip↔coin slip measurement (no tip-velocity field
    exists on the planar metrics). ``mj_objectVelocity`` returns a 6-D spatial velocity; [3:5] are the linear x,y."""
    res = np.zeros(6, np.float64)
    v = np.zeros(2, np.float64)
    for g in tip_geoms:
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, int(g), res, 0)
        v += res[3:5]
    return (v / max(1, len(tip_geoms))).astype(np.float32)


@dataclass(frozen=True)
class ForceSlipConfig:
    """Progressive ablation flags (S1–S5) + physically-motivated constants. The numeric knobs are shared so an arm differs
    only by which semantics are on. ``coast_mu`` is the CALIBRATED Coulomb μ_eff (from the coast calibration)."""

    coast_mu: float = 0.15             # calibrated coin↔floor Coulomb μ_eff (drives the launch-velocity target)
    preload_lo: float = 2.0            # N — normal-force band: enough grip …
    preload_hi: float = 8.0            # N — … without a 17 N press spike
    preload_gain: float = 0.03         # rad/step per (relative Fn error) — normal displacement regulation
    accel_gain: float = 0.25           # rad/step — tangential drive while accelerating the coin. MEASURED (2026-07-25):
    #                                    0.05 under-drove (coin 0.335 m/s, arm headroom unused); 0.25 imparts 0.6 m/s +
    #                                    0.155 m transport WITHIN the motion contract (governor caps joint vel at 2.2 ≤
    #                                    3.45) — the single-tip impedance is NOT the wall; the earlier controller was gentle.
    v_target_cap: float = 0.8          # m/s — never command a launch faster than this (safety / not a projectile)
    v_band: float = 0.08               # m/s — coin-velocity target band half-width (stop pushing inside it)
    slip_thresh: float = 0.15          # m/s — tip↔coin tangential slip above which grip is failing
    impulse_budget: float = 3.0        # N·s — max normal impulse per push before forcing a coast (safety)
    brake_margin: float = 0.01         # m — begin predictive braking this much before the predicted stop
    replan_every: int = 4
    enable_preload: bool = True        # S1
    enable_target_velocity: bool = True  # S2
    enable_slip_aware: bool = True     # S3
    enable_impulse_gate: bool = True   # S4
    enable_predictive_brake: bool = True  # S5
    probe_mag: float = 2.0
    probe_steps: int = 4


def _launch_velocity(mu: float, remaining: float, cap: float) -> float:
    """Coast model: to stop AT the target after coasting ``remaining`` m under Coulomb μ, launch at sqrt(2 μ g d)."""
    return float(min(cap, np.sqrt(max(0.0, 2.0 * mu * _G * remaining))))


def force_slip_carry(rl, gate, pi0, base, stack, *, horizon: int, cfg: ForceSlipConfig | None = None, frame_hook=None):
    """Contact-semantic delivery on frozen V2 physics. Returns delivery + the PRIMARY force/slip metrics: coin velocity at
    the end of the controlled push, target-directed normal impulse, Ft/Fn, transport distance, and the stopping-distance
    prediction error (|predicted stop − actual coast|). K6/zone are outputs, never inputs to the control law."""
    cfg = cfg or ForceSlipConfig()
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gov = stack.gov
    tip_geoms = [g for g in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("fingertip_left", "fingertip_right")) if g >= 0]
    subs = {"n": 0, "gov": 0}

    def _gcb(_model, data):
        raw = data.ctrl[:4].copy()
        data.ctrl[:4] = govern_torque(raw, data.qvel[:4], gov)
        subs["n"] += 1
        subs["gov"] += int(not np.allclose(data.ctrl[:4], raw))
    mujoco.set_mjcb_control(_gcb)

    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    u0, _n = rl.inner.direction_to_zone()
    zone_u = np.asarray(u0, np.float32)
    handed, md, touched, J, prev_tau = False, int(rl._strict), rl._touched, None, None
    phase = "ACQUIRE"
    push_impulse, v_launch, coast_start = 0.0, 0.0, None
    peak_vel, peak_fn, peak_ft, transport, coin_push_end_v = 0.0, 0.0, 0.0, 0.0, 0.0
    peak_coin_v, peak_v_along, peak_v_cross = 0.0, 0.0, 0.0        # speed, TARGET-projected, CROSS-track coin velocity
    signed_target_disp = 0.0                                       # final coin displacement along the target (may be < 0)
    stop_pred_err, entered_zone = None, False
    coin_bid = int(m.geom_bodyid[rl.inner._disk_geom])
    coin_mass = float(m.body_mass[coin_bid])
    # PHASE-TRANSITION log — proves whether each semantic stage actually FIRED (a stage that never runs is not a stage
    # that failed). Counts rising edges of each phase + whether the coin ever reached the launch band.
    phase_log = {"launch_reached": 0, "coast_entered": 0, "brake_entered": 0, "settle_entered": 0, "acquire_entered": 0}
    prev_phase = None
    t = 0
    try:
        for t in range(horizon):
            s = int(rl._strict)
            mpl = rl.inner._planar_metrics
            contact = bool(mpl.left_contact or mpl.right_contact)
            coin_v = np.asarray(mpl.disk_vel, np.float32)[:2]
            coin_speed = float(np.linalg.norm(coin_v))
            dtz = rl._dtz()
            v_launch = _launch_velocity(cfg.coast_mu, dtz, cfg.v_target_cap) if cfg.enable_target_velocity else cfg.v_target_cap
            if gate.gate != 1.0:
                a, cur = _det(pi0, rl.obs()), "PI0"
            elif handed or s >= 1:
                handed, cur = True, "SETTLE"
                a = _det(base, _aug(rl.obs(), s))
            else:
                if J is None or t % cfg.replan_every == 0:
                    J = _contact_jacobian(rl, cfg)
                tip_v = _tip_world_velocity(m, d, tip_geoms)
                delta, phase = _semantic_delta(rl, J, cfg, zone_u, dtz, contact, coin_v, coin_speed, v_launch,
                                                push_impulse, phase, tip_v)
                cur = phase
                q, qd = d.qpos[:4].copy(), d.qvel[:4].copy()
                a = pd_governed_torque(q, qd, q + delta, stack, prev_tau, lo, hi)
                prev_tau = a
            _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
            gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            md = max(md, int(rl._strict))
            touched = touched or rl._touched
            peak_vel = max(peak_vel, float(np.max(np.abs(d.qvel[:4]))))
            now_contact = bool(rl.inner._planar_metrics.left_contact or rl.inner._planar_metrics.right_contact)
            if now_contact:
                fn, ft = _contact_forces(m, d)
                peak_fn, peak_ft = max(peak_fn, fn), max(peak_ft, ft)
                push_impulse += fn * stack.control_dt
            csp_now = np.asarray(rl.inner._planar_metrics.disk_vel, np.float32)[:2]
            v_along = float(csp_now @ zone_u)                            # TARGET-projected coin velocity (signed)
            v_cross = float(np.linalg.norm(csp_now - v_along * zone_u))  # CROSS-track coin velocity (drift)
            peak_coin_v = max(peak_coin_v, float(np.linalg.norm(csp_now)))
            peak_v_along, peak_v_cross = max(peak_v_along, v_along), max(peak_v_cross, v_cross)
            if cur != prev_phase:                                        # phase rising-edge log
                key = {"COAST": "coast_entered", "BRAKE": "brake_entered", "SETTLE": "settle_entered",
                       "ACQUIRE": "acquire_entered"}.get(cur)
                if key:
                    phase_log[key] += 1
                prev_phase = cur
            if cfg.enable_target_velocity and v_along >= v_launch - cfg.v_band:
                phase_log["launch_reached"] = 1                          # the coin reached the coast-model launch band
            # capture the coin velocity at the moment the push releases into a coast
            if phase == "COAST" and coast_start is None:
                coast_start = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
                coin_push_end_v = float(np.linalg.norm(csp_now))
            disp = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2] - disk0
            transport = max(transport, float(disp @ zone_u))
            signed_target_disp = float(disp @ zone_u)                    # (last value = final signed displacement)
            entered_zone = entered_zone or (rl._dtz() <= CENTER_TOL)
            if frame_hook is not None:
                frame_hook(cur, int(rl._strict))
            if term or trunc:
                break
    finally:
        mujoco.set_mjcb_control(None)
    if coast_start is not None:                                    # stopping-distance prediction error
        actual_coast = float(np.linalg.norm(np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2] - coast_start))
        predicted_stop = coin_push_end_v ** 2 / (2 * cfg.coast_mu * _G + 1e-9)
        stop_pred_err = round(abs(predicted_stop - actual_coast), 4)
    n = max(1, t + 1)
    nsub = max(1, subs["n"])
    return {"k6": int(md >= HELD_DWELL and touched), "max_dwell": md, "touched": int(touched),
            "entered_zone": int(entered_zone), "transport_dist": round(transport, 4),
            "signed_target_displacement": round(signed_target_disp, 4),
            "coin_push_end_velocity": round(coin_push_end_v, 3), "peak_coin_velocity": round(peak_coin_v, 3),
            "peak_target_velocity": round(peak_v_along, 3), "peak_cross_track_velocity": round(peak_v_cross, 3),
            "target_directed_impulse": round(coin_mass * max(0.0, peak_v_along), 5),
            "lateral_impulse": round(coin_mass * peak_v_cross, 5), "contact_normal_impulse": round(push_impulse, 4),
            "peak_contact_normal_force": round(peak_fn, 2), "peak_contact_tangential_force": round(peak_ft, 2),
            "ftfn": round(peak_ft / (peak_fn + 1e-6), 3), "launch_velocity_target": round(v_launch, 3),
            "stopping_distance_pred_error": stop_pred_err, "peak_joint_vel": round(peak_vel, 2),
            "phase_log": phase_log, "acquired_contact": int(peak_fn > 0),
            "governor_active_frac": round(subs["gov"] / nsub, 3), "steps": n}


def _semantic_delta(rl, J, cfg: ForceSlipConfig, zone_u, dtz, contact, coin_v, coin_speed, v_launch, push_impulse, phase, tip_v):
    """One control step: compose a joint displacement from a bounded NORMAL PRELOAD (toward the coin) and a target-directed
    TANGENTIAL command (toward the zone), gated by the coin-velocity target, slip, impulse budget, and predictive braking."""
    if not contact:
        return 0.10 * _acquire_direction(rl, cfg), "ACQUIRE"
    fn, _ft = _contact_forces(rl.inner.model, rl.inner.data)
    v_along = float(coin_v @ zone_u)
    # S5 predictive braking: begin braking when the remaining distance ≈ the predicted stopping distance
    brake_d = v_along * v_along / (2 * cfg.coast_mu * _G + 1e-9)
    if cfg.enable_predictive_brake and dtz <= brake_d + cfg.brake_margin and v_along > 1.5 * SETTLE_VEL:
        return cfg.accel_gain * _unit(-_jac_solve(J, coin_v)), "BRAKE"
    if dtz <= CENTER_TOL and coin_speed < 1.5 * SETTLE_VEL:
        return np.zeros(4, np.float32), "SETTLE"
    # S1 bounded normal preload: press toward the coin to keep Fn in [lo, hi]
    preload = np.zeros(4, np.float32)
    if cfg.enable_preload:
        acq = _acquire_direction(rl, cfg)                          # unit joint dir into the coin
        if fn < cfg.preload_lo:
            preload = cfg.preload_gain * (cfg.preload_lo - fn) / cfg.preload_lo * acq
        elif fn > cfg.preload_hi:
            preload = -cfg.preload_gain * (fn - cfg.preload_hi) / cfg.preload_hi * acq
    # S4 impulse gate: stop pushing once the coin reaches the target-velocity band (or the impulse budget is spent)
    reached_band = cfg.enable_target_velocity and v_along >= v_launch - cfg.v_band
    spent = cfg.enable_impulse_gate and push_impulse >= cfg.impulse_budget
    if reached_band or spent:
        return preload, "COAST"                                    # release the tangential drive; keep light preload
    # S2 target-directed tangential command toward the zone
    tang = cfg.accel_gain * _unit(_jac_solve(J, zone_u))
    # S3 slip-aware modulation: if the tip slides on the coin, raise preload rather than push harder
    if cfg.enable_slip_aware:
        slip = float(np.linalg.norm(np.asarray(tip_v, np.float32) - coin_v))
        if slip > cfg.slip_thresh and fn < cfg.preload_hi:
            preload = preload + cfg.preload_gain * _acquire_direction(rl, cfg)   # grip harder, within the band
            tang = tang * 0.5                                       # ease the tangential accel while re-gripping
    return preload + tang, "ACCEL"
