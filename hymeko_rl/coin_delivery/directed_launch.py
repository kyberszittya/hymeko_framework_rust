"""TARGET_DIRECTED_LAUNCH_V1 — aim the (already-proven) contact impulse at the zone. Frozen: V2 material, V4 motion
contract, the B1 passive-barrier brake. The ONLY thing that changes is HOW the tip transfers the impulse.

The launch benchmark showed the impulse magnitude is sufficient but mis-aimed (s1: target-projected 0.478, cross-track
0.546 — pushed roughly diagonally). The fix is a TARGET-FRAME force-direction controller. In each state:
    e_parallel = coin→zone unit,  e_cross = ⟂
    desired coin velocity  = v_launch*(coast model) · e_parallel  (cross ≈ 0)
After contact, solve a directed contact-Jacobian step Δq that drives v_∥ toward v_∥* AND v_⊥ toward 0:
    Δq = argmin ‖ (R·J_coin) Δq − [v_∥* − v_∥ ; −k_⊥ v_⊥] ‖² + λ‖Δq‖²      (R rotates world→target frame)
The governor and every motion limit stay OUTSIDE as hard constraints.

Progressive stages (config flags):
  L0 current tuned launch (push toward the zone, no frame decomposition)
  L1 + target-projected velocity objective (drive v_∥ → v_∥*)
  L2 + cross-track suppression (drive v_⊥ → 0)
  L3 + state-dependent contact-point / push-side selection (probe both tips, pick best target-authority − cross-leak)
  L4 + short receding-horizon directional correction (re-solve every step — already the default cadence)
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_carry_structured import HELD_DWELL, _aug, _det, stable_engagement_signals
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.brake_control import _tip_to_point_dir
from hymeko_rl.coin_delivery.motion_robust_expert import CarryControllerConfig, _acquire_direction, _contact_forces, _contact_jacobian, _unit
from hymeko_rl.env.governed_arm import pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque

_G = 9.81


@dataclass(frozen=True)
class DirectedLaunchConfig:
    coast_mu: float = 0.179
    k_cross: float = 2.0               # cross-track suppression gain (drive v_⊥ → 0 faster than v_∥ → target)
    gain: float = 0.06                 # rad/step — directed displacement magnitude
    v_band: float = 0.08               # m/s — release once v_∥ reaches v_∥* − v_band
    lam: float = 0.08                  # DLS damping on the directed solve
    replan_every: int = 2
    enable_directed: bool = True       # L1: target-frame parallel objective (off ⇒ L0 push-to-zone)
    enable_cross_suppress: bool = True  # L2: cross-track suppression term
    enable_contact_select: bool = False  # L3: acquire the coin's FAR side (opposite the zone) so the push is target-directed
    coin_radius: float = 0.02          # m — the disk radius (far-side contact point = coin − r·e_parallel)
    probe_mag: float = 2.0
    probe_steps: int = 4


def _frame(rl):
    """Target frame at the coin: e_parallel = coin→zone, e_cross = ⟂ (right-hand)."""
    u, _dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(u, np.float32)
    e_cross = np.array([-e_par[1], e_par[0]], np.float32)
    return e_par, e_cross


def _directed_delta(J, e_par, e_cross, coin_v, v_target, cfg: DirectedLaunchConfig):
    """Δq that drives the coin velocity toward v_target·e_parallel with v_cross→0 (directed damped least squares)."""
    v_par = float(coin_v @ e_par)
    v_cross = float(coin_v @ e_cross)
    r = np.stack([e_par, e_cross]).astype(np.float64)             # 2×2 world→frame rotation
    jf = r @ J.astype(np.float64)                                 # 2×4 coin-velocity Jacobian in the target frame
    cross_term = -cfg.k_cross * v_cross if cfg.enable_cross_suppress else 0.0
    desired = np.array([v_target - v_par, cross_term], np.float64)
    dq = jf.T @ np.linalg.solve(jf @ jf.T + cfg.lam ** 2 * np.eye(2), desired)
    return _unit(dq.astype(np.float32)), v_par, v_cross


def directed_launch_carry(rl, gate, pi0, base, stack, *, horizon: int, cfg: DirectedLaunchConfig | None = None,
                          frame_hook=None):
    """Directed launch on the shared governed stack. Returns the launch quality in the TARGET FRAME: peak v_parallel, peak
    v_cross, launch-angle error, target-directed vs lateral impulse, signed target displacement, plus contact + motion."""
    cfg = cfg or DirectedLaunchConfig()
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gov, ccfg = stack.gov, CarryControllerConfig()
    coin_bid = int(m.geom_bodyid[rl.inner._disk_geom])
    coin_mass = float(m.body_mass[coin_bid])
    tip_geoms = [g for g in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("fingertip_left", "fingertip_right")) if g >= 0]

    def _gcb(_model, data):
        data.ctrl[:4] = govern_torque(data.ctrl[:4], data.qvel[:4], gov)
    mujoco.set_mjcb_control(_gcb)

    e_par, e_cross = _frame(rl)
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    _side = "far_side" if cfg.enable_contact_select else None
    handed, md, touched, J, prev_tau = False, int(rl._strict), rl._touched, None, None
    peak_v_par, peak_v_cross, peak_joint, peak_fn, peak_ft = 0.0, 0.0, 0.0, 0.0, 0.0
    released, v_target_last = False, 0.0
    t = 0
    try:
        for t in range(horizon):
            s = int(rl._strict)
            mpl = rl.inner._planar_metrics
            contact = bool(mpl.left_contact or mpl.right_contact)
            coin_v = np.asarray(mpl.disk_vel, np.float32)[:2]
            dtz = rl._dtz()
            v_target = float(min(0.8, np.sqrt(max(0.0, 2.0 * cfg.coast_mu * _G * dtz))))
            v_target_last = v_target
            v_par = float(coin_v @ e_par)
            if gate.gate != 1.0:
                a, cur = _det(pi0, rl.obs()), "PI0"
            elif handed or s >= 1:
                handed, cur, a = True, "SETTLE", _det(base, _aug(rl.obs(), s))
            elif not contact:
                if cfg.enable_contact_select:                     # L3: acquire the FAR SIDE (opposite the zone) so the push
                    coin_xy = np.asarray(mpl.disk_pos, np.float32)[:2]   # through the coin centre is target-directed
                    far_pt = coin_xy - cfg.coin_radius * e_par
                    acq = _tip_to_point_dir(rl, tip_geoms, far_pt, stack)
                else:
                    acq = _acquire_direction(rl, ccfg)
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + 0.10 * acq, stack, prev_tau, lo, hi)
                prev_tau, cur = a, "ACQUIRE"
            elif released or v_par >= v_target - cfg.v_band:      # impulse gate: released into coast once aimed + up to speed
                released, cur = True, "COAST"
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4], stack, prev_tau, lo, hi)
                prev_tau = a
            else:
                if J is None or t % cfg.replan_every == 0:
                    J = _contact_jacobian(rl, ccfg)
                if cfg.enable_directed:                            # L1/L2 directed contact-Jacobian
                    ddir, _vp, _vc = _directed_delta(J, e_par, e_cross, coin_v, v_target, cfg)
                else:                                              # L0 push straight at the zone (no frame decomposition)
                    from hymeko_rl.coin_delivery.motion_robust_expert import _jac_solve
                    ddir = _unit(_jac_solve(J, e_par))
                delta = cfg.gain * ddir
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + delta, stack, prev_tau, lo, hi)
                prev_tau, cur = a, "LAUNCH"
            _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
            gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            md = max(md, int(rl._strict))
            touched = touched or rl._touched
            cvn = np.asarray(rl.inner._planar_metrics.disk_vel, np.float32)[:2]
            peak_v_par = max(peak_v_par, float(cvn @ e_par))
            peak_v_cross = max(peak_v_cross, abs(float(cvn @ e_cross)))
            peak_joint = max(peak_joint, float(np.max(np.abs(d.qvel[:4]))))
            if rl.inner._planar_metrics.left_contact or rl.inner._planar_metrics.right_contact:
                fn, ft = _contact_forces(m, d)
                peak_fn, peak_ft = max(peak_fn, fn), max(peak_ft, ft)
            if frame_hook is not None:
                frame_hook(cur, int(rl._strict))
            if term or trunc:
                break
    finally:
        mujoco.set_mjcb_control(None)
    final_disp = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2] - disk0
    angle_err = float(np.degrees(np.arctan2(peak_v_cross, max(1e-6, peak_v_par))))
    return {"v_target": round(v_target_last, 3), "peak_v_parallel": round(peak_v_par, 3),
            "peak_v_cross": round(peak_v_cross, 3), "cross_ratio": round(peak_v_cross / max(1e-6, peak_v_par), 3),
            "launch_angle_error_deg": round(angle_err, 1),
            "target_directed_impulse": round(coin_mass * max(0.0, peak_v_par), 5),
            "lateral_impulse": round(coin_mass * peak_v_cross, 5),
            "signed_target_displacement": round(float(final_disp @ e_par), 4),
            "cross_displacement": round(float(final_disp @ e_cross), 4), "peak_joint_vel": round(peak_joint, 2),
            "peak_contact_normal_force": round(peak_fn, 2), "peak_contact_tangential_force": round(peak_ft, 2),
            "acquired_contact": int(peak_fn > 0), "contact_side": _side, "k6": int(md >= HELD_DWELL and touched), "steps": t + 1}
