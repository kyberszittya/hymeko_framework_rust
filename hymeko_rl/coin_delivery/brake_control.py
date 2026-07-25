"""Micro-test 2 controller — BRAKE-ONLY, with CONTROLLED initial conditions. Launch is removed: the coin is placed at a
known remaining distance from the zone and given a known velocity toward it; the arm starts from a config from which a
braking contact is executable (the fingertip is driven to the zone side as a barrier). We then compare progressive
braking arms and log an EXPLICIT event chain so a stage that never fires is not mistaken for a stage that failed.

Arms (progressive):
  B0 coast_only                — arm retracts; the coin coasts under the calibrated Coulomb drag
  B1 predicted_passive_landing — arm holds the barrier at the coast-model predicted stop; no active braking
  B2 velocity_triggered_recontact — when overshoot is predicted, re-contact the coin (no counter-impulse)
  B3 bounded_counter_impulse   — B2 + a BOUNDED counter-impulse opposing the coin velocity toward v_terminal ≈ 0
  B4 low_speed_terminal_correction — B3 + a low-speed correction to seat the coin at the zone centre
  B5 settle_certificate        — B4 + a settle hold (terminal certificate)

Braking targets a terminal velocity ≈ 0 with a bounded negative impulse (not an unbounded "stop"). Everything runs on the
shared governed stack. The coast model (accurate to ~1 cm) supplies the trigger — not a hand-tuned time.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_carry_structured import CENTER_TOL, HELD_DWELL, SETTLE_VEL, stable_engagement_signals
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.motion_robust_expert import CarryControllerConfig, _contact_jacobian, _jac_solve, _unit
from hymeko_rl.env.governed_arm import pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque

_G = 9.81
_MODES = ("B0_coast", "B1_passive_landing", "B2_recontact", "B3_counter_impulse", "B4_terminal_correction", "B5_settle")


@dataclass(frozen=True)
class BrakeConfig:
    coast_mu: float = 0.179
    brake_margin: float = 0.01         # m — trigger braking this much before the predicted stop
    counter_gain: float = 0.05         # rad/step — bounded counter-impulse displacement magnitude
    counter_cap: float = 0.5           # m/s — target terminal coin velocity ceiling the brake drives toward (≈0)
    terminal_gain: float = 0.03        # rad/step — low-speed terminal correction toward the zone centre
    replan_every: int = 4
    setup_steps: int = 25              # steps to drive the fingertip to the barrier position before releasing the coin


def _zone_xy(rl) -> np.ndarray:
    return np.array([rl.inner._zone_x, rl.inner._zone_y], np.float32)


def _coin_to_zone(rl, zone) -> float:
    """Geometric coin→zone distance (rl._dtz() uses a different reference; for the brake benchmark we need the literal xy
    distance so the stop error is interpretable)."""
    return float(np.linalg.norm(np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2] - zone))


def _tip_xy(rl, tip_geoms) -> np.ndarray:
    d = rl.inner.data
    return np.mean([d.geom_xpos[g][:2] for g in tip_geoms], axis=0).astype(np.float32)


def _tip_to_point_dir(rl, tip_geoms, target_xy, stack, probe=2.0, steps=3) -> np.ndarray:
    """FD joint direction that reduces the fingertip→target distance (drive the tip to a chosen world point)."""
    import copy
    d0 = float(np.linalg.norm(_tip_xy(rl, tip_geoms) - target_xy))
    grad = np.zeros(4, np.float32)
    for j in range(4):
        r2 = copy.deepcopy(rl)
        a = np.zeros(4, np.float32)
        a[j] = probe
        for _ in range(steps):
            step_ablation(r2, a, "A")
        grad[j] = (float(np.linalg.norm(_tip_xy(r2, tip_geoms) - target_xy)) - d0) / (probe * steps)
    g = -grad
    n = float(np.linalg.norm(g))
    return (g / n).astype(np.float32) if n > 1e-6 else np.zeros(4, np.float32)


def place_coin_for_brake(rl, d_remaining: float, v0: float, u: np.ndarray):
    """Place the coin at ``d_remaining`` m from the zone along the approach direction ``u`` and give it velocity ``v0``
    toward the zone. (Position + velocity on the disk's planar-slide DOFs.)"""
    m, d = rl.inner.model, rl.inner.data
    adr = int(rl.inner._disk_x_adr)
    zone = _zone_xy(rl)
    d.qpos[adr:adr + 2] = (zone - d_remaining * u).astype(np.float64)   # d_remaining short of the zone, on the approach line
    d.qvel[adr:adr + 2] = (v0 * u).astype(np.float64)
    mujoco.mj_forward(m, d)


def brake_carry(rl, gate, pi0, base, stack, *, mode: str, d_remaining: float, v0: float, cfg: BrakeConfig | None = None,
                horizon: int = 240):
    """Run one brake arm from controlled initial conditions. Returns stop error, overshoot, the event chain, re-contact +
    counter-impulse, terminal coin speed, in-zone dwell, and motion-contract pass. K6/zone are outputs."""
    cfg = cfg or BrakeConfig()
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gov, ccfg = stack.gov, CarryControllerConfig()
    tip_geoms = [g for g in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("fingertip_left", "fingertip_right")) if g >= 0]
    zone = _zone_xy(rl)
    u0, _dtz0 = rl.inner.direction_to_zone()                       # approach direction (from the natural coin position)
    u0 = np.asarray(u0, np.float32)
    barrier = zone - CENTER_TOL * u0                               # barrier ON the approach line, at the zone edge

    def _gcb(_model, data):
        data.ctrl[:4] = govern_torque(data.ctrl[:4], data.qvel[:4], gov)
    mujoco.set_mjcb_control(_gcb)

    predicted_stop = v0 * v0 / (2 * cfg.coast_mu * _G + 1e-9)
    events = {"coast_entered": 0, "brake_condition_crossed": 0, "recontact_attempted": 0, "recontact_acquired": 0,
              "counter_impulse_delivered": 0, "terminal_phase_entered": 0, "settled": 0}
    active = _MODES.index(mode)                                    # which capabilities are on (progressive)
    prev_tau, J, md, touched = None, None, int(rl._strict), rl._touched
    peak_joint, dwell, brake_frames, counter_impulse = 0.0, 0, 0, 0.0
    u = None
    try:
        # ── setup: drive the fingertip to the barrier (zone side), THEN release the coin with its launch velocity ──
        bdir = np.zeros(4, np.float32)
        if active >= 1:                                           # B1+ position the barrier; B0 leaves the arm at home
            for k in range(cfg.setup_steps):
                if k % cfg.replan_every == 0:                     # cache the FD barrier direction (keeps setup cheap)
                    bdir = _tip_to_point_dir(rl, tip_geoms, barrier, stack)
                q, qd = d.qpos[:4].copy(), d.qvel[:4].copy()
                a = pd_governed_torque(q, qd, q + 0.10 * bdir, stack, prev_tau, lo, hi)
                prev_tau = a
                step_ablation(rl, np.asarray(a, np.float32), "A")
        place_coin_for_brake(rl, d_remaining, v0, u0)
        u = u0
        q_home = d.qpos[:4].copy()
        for t in range(horizon):
            mpl = rl.inner._planar_metrics
            contact = bool(mpl.left_contact or mpl.right_contact)
            coin_v = np.asarray(mpl.disk_vel, np.float32)[:2]
            v_along = float(coin_v @ u)
            coin_speed = float(np.linalg.norm(coin_v))
            dtz = _coin_to_zone(rl, zone)
            overshoot_pred = dtz <= predicted_stop + cfg.brake_margin
            if overshoot_pred and coin_speed > SETTLE_VEL:
                events["brake_condition_crossed"] = 1
            if J is None or t % cfg.replan_every == 0:
                J = _contact_jacobian(rl, ccfg)
            # ── arm action by mode ──
            if active == 0:                                       # B0 coast: home, do not touch
                delta = q_home - d.qpos[:4]
                events["coast_entered"] = 1
            elif active == 1:                                     # B1 passive barrier: hold the tip at the zone
                delta = 0.08 * _tip_to_point_dir(rl, tip_geoms, barrier, stack)
            elif dtz <= CENTER_TOL and coin_speed < 1.5 * SETTLE_VEL and active >= 4:
                events["terminal_phase_entered"] = 1
                delta = cfg.terminal_gain * _unit(_jac_solve(J, (zone - np.asarray(mpl.disk_pos, np.float32)[:2]))) \
                    if active >= 4 else np.zeros(4, np.float32)
                if coin_speed < SETTLE_VEL and dtz <= CENTER_TOL:
                    events["settled"] = 1 if active >= 5 else events["settled"]
                    dwell += 1
            elif overshoot_pred and coin_speed > SETTLE_VEL:      # B2+ predictive re-contact braking
                if not contact:
                    events["recontact_attempted"] = 1
                    delta = 0.10 * _tip_to_point_dir(rl, tip_geoms, np.asarray(mpl.disk_pos, np.float32)[:2], stack)
                else:
                    events["recontact_acquired"] = 1
                    if active >= 3:                               # B3+ bounded counter-impulse toward v_terminal≈0
                        excess = max(0.0, v_along - 0.0)
                        mag = min(cfg.counter_gain, cfg.counter_gain * excess / (cfg.counter_cap + 1e-9))
                        delta = mag * _unit(-_jac_solve(J, coin_v))
                        events["counter_impulse_delivered"] = 1
                        counter_impulse += float(np.linalg.norm(delta))
                        brake_frames += 1
                    else:                                          # B2 re-contact only (barrier, no active brake)
                        delta = np.zeros(4, np.float32)
            else:
                delta = 0.06 * _tip_to_point_dir(rl, tip_geoms, barrier, stack) if active >= 1 else (q_home - d.qpos[:4])
            q, qd = d.qpos[:4].copy(), d.qvel[:4].copy()
            a = pd_governed_torque(q, qd, q + np.asarray(delta, np.float32), stack, prev_tau, lo, hi)
            prev_tau = a
            _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
            gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            md = max(md, int(rl._strict))
            touched = touched or rl._touched
            peak_joint = max(peak_joint, float(np.max(np.abs(d.qvel[:4]))))
            if _coin_to_zone(rl, zone) <= CENTER_TOL and coin_speed < SETTLE_VEL:
                dwell += 1
            if term or trunc:
                break
    finally:
        mujoco.set_mjcb_control(None)
    final_dtz = _coin_to_zone(rl, zone)
    term_speed = float(np.linalg.norm(rl.inner._planar_metrics.disk_vel[:2]))
    return {"mode": mode, "d_remaining": d_remaining, "v0": v0, "predicted_stop_dist": round(predicted_stop, 4),
            "signed_stop_error": round(final_dtz, 4), "overshoot": bool(predicted_stop > d_remaining + CENTER_TOL),
            "entered_zone": int(final_dtz <= CENTER_TOL), "in_zone_dwell": dwell, "terminal_coin_speed": round(term_speed, 3),
            "recontacted": events["recontact_acquired"], "counter_impulse": round(counter_impulse, 4),
            "brake_frames": brake_frames, "peak_joint_vel": round(peak_joint, 2),
            "motion_contract_pass": bool(peak_joint <= 3.45), "events": events, "k6": int(md >= HELD_DWELL and touched)}
