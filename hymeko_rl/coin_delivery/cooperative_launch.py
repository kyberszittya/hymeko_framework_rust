"""Cooperative two-contact launch with a REACHABILITY-GATED, DECOUPLED, synchronized close (the (1) fix after BIMANUAL_V1
found two-contact acquisition unreliable). Two facts drove the design:
  * the 4 DoF split cleanly — DoF 0–1 = LEFT arm, 2–3 = RIGHT arm — so each arm is driven INDEPENDENTLY (a coupled 4-DoF
    gradient let one arm dominate);
  * on part of the panel the RIGHT arm cannot reach the coin at all (workspace limit) ⇒ two-contact is geometrically
    impossible there — a REACHABILITY fact that must be checked, not assumed.

Acquire: each arm closes toward the coin CENTRE with its own DoF at equal gain (a SYNCHRONISED symmetric pinch — if the
light coin drifts toward one tip, that tip contacts sooner, so the close is self-correcting). Once both tips contact, run
the cooperative twist-Jacobian allocation (translate toward the zone, zero spin). A per-state REACHABILITY PROBE reports
whether both arms can individually reach the coin — the relational fact the structural prior would predict.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.bimanual_launch import _coin_twist, _coin_twist_jacobian, _unit2
from hymeko_rl.coin_delivery.coin_carry_structured import HELD_DWELL, _det, stable_engagement_signals
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.env.governed_arm import pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque

_G = 9.81
_LEFT_DOF, _RIGHT_DOF = (0, 1), (2, 3)


@dataclass(frozen=True)
class CooperativeConfig:
    coast_mu: float = 0.179
    coin_radius: float = 0.02
    close_gain: float = 0.10
    launch_gain: float = 0.06
    lam: float = 0.06
    w_omega: float = 3.0
    replan_every: int = 2
    reach_steps: int = 90
    contact_tol: float = 0.03          # m — tip within this of the coin centre counts as reach-capable
    probe_mag: float = 2.0
    probe_steps: int = 3


def _tip_xy(rl, g):
    return rl.inner.data.geom_xpos[g][:2].astype(np.float32)


def _arm_dir(rl, g, dofs, target, cfg):
    """FD joint direction over ONLY this arm's DoF driving its tip to ``target`` (decoupled per-arm control)."""
    c0 = float(np.linalg.norm(_tip_xy(rl, g) - target))
    grad = np.zeros(4, np.float32)
    for j in dofs:
        r2 = copy.deepcopy(rl)
        a = np.zeros(4, np.float32)
        a[j] = cfg.probe_mag
        for _ in range(cfg.probe_steps):
            step_ablation(r2, a, "A")
        grad[j] = (float(np.linalg.norm(_tip_xy(r2, g) - target)) - c0) / (cfg.probe_mag * cfg.probe_steps)
    g2 = -grad
    n = float(np.linalg.norm(g2))
    return (g2 / n).astype(np.float32) if n > 1e-6 else np.zeros(4, np.float32)


def reachability_probe(rl, stack, cfg: CooperativeConfig | None = None):
    """Per-arm reachability: drive EACH arm alone toward the coin and report whether it can contact / how near it gets.
    Delivery-independent — the relational fact (which arm can reach the coin from this state)."""
    cfg = cfg or CooperativeConfig()
    out = {}
    for side, g_name, dofs in (("left", "fingertip_left", _LEFT_DOF), ("right", "fingertip_right", _RIGHT_DOF)):
        r = copy.deepcopy(rl)
        m, d = r.inner.model, r.inner.data
        m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
        lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g_name)

        def _gcb(_mo, dt, _gov=stack.gov):
            dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], _gov)
        mujoco.set_mjcb_control(_gcb)
        prev, touched, min_d = None, 0, 9.0
        try:
            for _ in range(cfg.reach_steps):
                coin = np.asarray(r.inner._planar_metrics.disk_pos, np.float32)[:2]
                min_d = min(min_d, float(np.linalg.norm(_tip_xy(r, g) - coin)))
                v = _tip_xy(r, g) - coin
                tgt = coin + cfg.coin_radius * (v / (np.linalg.norm(v) + 1e-9))
                delta = cfg.close_gain * _arm_dir(r, g, dofs, tgt, cfg)
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + delta, stack, prev, lo, hi)
                prev = a
                step_ablation(r, np.asarray(a, np.float32), "A")
                c = r.inner._planar_metrics.left_contact if side == "left" else r.inner._planar_metrics.right_contact
                touched += int(bool(c))
        finally:
            mujoco.set_mjcb_control(None)
        out[side] = {"reachable": bool(touched > 0), "contact_frames": touched, "min_tip_coin": round(min_d, 4)}
    out["two_contact_reachable"] = bool(out["left"]["reachable"] and out["right"]["reachable"])
    return out


def cooperative_launch_carry(rl, gate, pi0, base, stack, *, horizon: int, cfg: CooperativeConfig | None = None):
    """Decoupled synchronized close (both arms → coin centre) then cooperative twist allocation. Returns launch quality +
    both-tips contact/simultaneity + force-line-miss ω. K6/zone are outputs."""
    cfg = cfg or CooperativeConfig()
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    u, _dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(u, np.float32)
    e_cross = np.array([-e_par[1], e_par[0]], np.float32)

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], stack.gov)
    mujoco.set_mjcb_control(_gcb)
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    peak_vp, peak_vc, peak_om, peak_joint, both_frames, both_seen, prev_tau, J = 0.0, 0.0, 0.0, 0.0, 0, 0, None, None
    md, touched, t = int(rl._strict), rl._touched, 0
    W = np.diag([1.0, 1.0, cfg.w_omega])
    try:
        for t in range(horizon):
            mpl = rl.inner._planar_metrics
            coin = np.asarray(mpl.disk_pos, np.float32)[:2]
            lc, rcn = bool(mpl.left_contact), bool(mpl.right_contact)
            dtz = rl._dtz()
            v_target = float(min(0.8, np.sqrt(max(0.0, 2.0 * cfg.coast_mu * _G * dtz))))
            if gate is not None and gate.gate != 1.0:
                a = _det(pi0, rl.obs())
            elif not (lc and rcn):                                # synchronized decoupled close: BOTH arms → coin centre
                dl = _arm_dir(rl, gl, _LEFT_DOF, coin, cfg)
                dr = _arm_dir(rl, gr, _RIGHT_DOF, coin, cfg)
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + cfg.close_gain * (dl + dr), stack, prev_tau, lo, hi)
                prev_tau = a
            else:                                                 # cooperative twist allocation: translate +e_par, ω→0
                both_frames += 1
                both_seen = 1
                if J is None or t % cfg.replan_every == 0:
                    J = _coin_twist_jacobian(rl, cfg)
                desired = np.array([v_target * e_par[0], v_target * e_par[1], 0.0], np.float64) - _coin_twist(rl)
                dq = J.T @ np.linalg.solve((W @ J) @ J.T + cfg.lam ** 2 * np.eye(3), W @ desired)
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + cfg.launch_gain * _unit2(dq), stack, prev_tau, lo, hi)
                prev_tau = a
            _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
            if gate is not None:
                l2, r2c, coin_s, ltp, rtp = stable_engagement_signals(rl.inner)
                gate.update(l2, r2c, coin_s, ltp, rtp, terminated=bool(term))
            md = max(md, int(rl._strict))
            touched = touched or rl._touched
            cvn = np.asarray(rl.inner._planar_metrics.disk_vel, np.float32)[:2]
            peak_vp = max(peak_vp, float(cvn @ e_par))
            peak_vc = max(peak_vc, abs(float(cvn @ e_cross)))
            peak_om = max(peak_om, abs(float(rl.inner.data.qvel[int(rl.inner._disk_x_adr) + 2])))
            peak_joint = max(peak_joint, float(np.max(np.abs(d.qvel[:4]))))
            if term or trunc:
                break
    finally:
        mujoco.set_mjcb_control(None)
    disp = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2] - disk0
    return {"peak_v_parallel": round(peak_vp, 3), "peak_v_cross": round(peak_vc, 3),
            "cross_ratio": round(peak_vc / max(1e-6, peak_vp), 3), "peak_omega": round(peak_om, 3),
            "both_tips_contact": both_seen, "both_contact_frames": both_frames,
            "signed_target_displacement": round(float(disp @ e_par), 4), "peak_joint_vel": round(peak_joint, 2),
            "k6": int(md >= HELD_DWELL and touched), "steps": t + 1}
