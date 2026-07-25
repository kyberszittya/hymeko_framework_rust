"""BIMANUAL_TARGET_DIRECTED_LAUNCH_V1 — two arms create a controllable RESULTANT on the coin. This is the essential
mechanism of the coin task (not a last resort): with a single contact the push-line is fixed by the accessible contact
point, but with TWO contacts the resultant force direction AND the torque about the coin are partly separately shapeable.

Formulation: the coin is a planar body with twist (v_x, v_y, ω). The 3×4 COIN-TWIST JACOBIAN J_t maps the two arms' 4-DoF
action to the coin twist. Once both tips contact, solve
    Δq = J_t⁺ · ( [v_target·e_parallel ; 0] − twist )
so the coin translates toward the zone with ZERO spin — i.e. F_L + F_R is aimed along the zone axis and τ_c ≈ 0. The two
arms allocate the force cooperatively; the governor + motion limits stay OUTSIDE as hard constraints. Deterministic, no RL.

Ablation flags:
  A1 symmetric fixed acquire  |  A2 state-dependent acquire  |  A3 + force balancing / cross-track suppression
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_carry_structured import HELD_DWELL, _det, stable_engagement_signals
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.motion_robust_expert import _contact_forces
from hymeko_rl.env.governed_arm import pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque

_G = 9.81


@dataclass(frozen=True)
class BimanualConfig:
    coast_mu: float = 0.179
    coin_radius: float = 0.02
    acquire_gain: float = 0.10
    launch_gain: float = 0.06          # rad/step — twist-allocation displacement magnitude
    lam: float = 0.06                  # DLS damping on the twist solve
    w_omega: float = 3.0               # A3: weight the zero-spin row (force-line through COM)
    replan_every: int = 2
    state_dependent: bool = False      # A2: acquire toward each side's nearest coin surface point (vs symmetric ±90°)
    force_balance: bool = True         # A3: include the zero-torque objective + cross suppression
    probe_mag: float = 2.0
    probe_steps: int = 3


def _frame(rl):
    u, _dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(u, np.float32)
    return e_par, np.array([-e_par[1], e_par[0]], np.float32)


def _tip_xy(rl, g):
    return rl.inner.data.geom_xpos[g][:2].astype(np.float32)


def _coin_twist(rl):
    adr = int(rl.inner._disk_x_adr)
    return np.asarray(rl.inner.data.qvel[adr:adr + 3], np.float64)   # (v_x, v_y, ω)


def _coin_twist_jacobian(rl, cfg):
    """3×4 finite-difference Jacobian d(coin twist)/d(arm action) — the cooperative map both arms share. Probes each DoF on
    a COPY and reads the coin's (Δv_x, Δv_y, Δω)."""
    t0 = _coin_twist(rl)
    J = np.zeros((3, 4), np.float64)
    for j in range(4):
        r2 = copy.deepcopy(rl)
        a = np.zeros(4, np.float32)
        a[j] = cfg.probe_mag
        for _ in range(cfg.probe_steps):
            step_ablation(r2, a, "A")
        J[:, j] = (_coin_twist(r2) - t0) / (cfg.probe_mag * cfg.probe_steps)
    return J


def _two_tip_dir(rl, gl, gr, p_l, p_r, probe=2.0, steps=3):
    def cost(r2):
        return float(np.linalg.norm(_tip_xy(r2, gl) - p_l) + np.linalg.norm(_tip_xy(r2, gr) - p_r))
    c0 = cost(rl)
    grad = np.zeros(4, np.float32)
    for j in range(4):
        r2 = copy.deepcopy(rl)
        a = np.zeros(4, np.float32)
        a[j] = probe
        for _ in range(steps):
            step_ablation(r2, a, "A")
        grad[j] = (cost(r2) - c0) / (probe * steps)
    g = -grad
    n = float(np.linalg.norm(g))
    return (g / n).astype(np.float32) if n > 1e-6 else np.zeros(4, np.float32)


def bimanual_launch_carry(rl, gate, pi0, base, stack, *, horizon: int, cfg: BimanualConfig | None = None):
    """Two-arm cooperative launch via the coin-twist Jacobian. Returns v_parallel/v_cross, force-line-miss ω, per-tip
    normal-force imbalance, both-tips contact + simultaneity, signed displacement, and motion. K6/zone are outputs."""
    cfg = cfg or BimanualConfig()
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gov = stack.gov
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    e_par, e_cross = _frame(rl)

    def _gcb(_model, data):
        data.ctrl[:4] = govern_torque(data.ctrl[:4], data.qvel[:4], gov)
    mujoco.set_mjcb_control(_gcb)
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    peak_vp, peak_vc, peak_om, peak_joint = 0.0, 0.0, 0.0, 0.0
    both_frames, max_imbalance, prev_tau, J, both_seen = 0, 0.0, None, None, 0
    md, touched, t = int(rl._strict), rl._touched, 0
    W = np.diag([1.0, 1.0, cfg.w_omega]) if cfg.force_balance else np.diag([1.0, 1.0, 0.0])
    try:
        for t in range(horizon):
            mpl = rl.inner._planar_metrics
            coin_xy = np.asarray(mpl.disk_pos, np.float32)[:2]
            lc, rcn = bool(mpl.left_contact), bool(mpl.right_contact)
            dtz = rl._dtz()
            v_target = float(min(0.8, np.sqrt(max(0.0, 2.0 * cfg.coast_mu * _G * dtz))))
            if gate is not None and gate.gate != 1.0:
                a = _det(pi0, rl.obs())
            elif not (lc and rcn):                                # acquire BOTH sides of the coin
                if cfg.state_dependent:                           # A2: each tip toward its own nearest coin-surface point
                    lt, rt = _tip_xy(rl, gl), _tip_xy(rl, gr)
                    p_l = coin_xy + cfg.coin_radius * _unit2(lt - coin_xy)
                    p_r = coin_xy + cfg.coin_radius * _unit2(rt - coin_xy)
                else:                                             # A1: symmetric ±e_cross about the coin
                    p_l = coin_xy + cfg.coin_radius * e_cross
                    p_r = coin_xy - cfg.coin_radius * e_cross
                delta = cfg.acquire_gain * _two_tip_dir(rl, gl, gr, p_l, p_r)
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + delta, stack, prev_tau, lo, hi)
                prev_tau = a
            else:                                                 # cooperative twist allocation: translate +e_par, ω→0
                both_frames += 1
                both_seen = 1
                if J is None or t % cfg.replan_every == 0:
                    J = _coin_twist_jacobian(rl, cfg)
                twist = _coin_twist(rl)
                desired = np.array([v_target * e_par[0], v_target * e_par[1], 0.0], np.float64) - twist
                jw = W @ J
                dq = J.T @ np.linalg.solve(jw @ J.T + cfg.lam ** 2 * np.eye(3), W @ desired)
                delta = cfg.launch_gain * _unit2(dq)
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + delta, stack, prev_tau, lo, hi)
                prev_tau = a
                fn_l, fn_r = _side_normal(m, d, gl), _side_normal(m, d, gr)
                max_imbalance = max(max_imbalance, abs(fn_l - fn_r))
            _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
            if gate is not None:
                l2, r2c, coin, ltp, rtp = stable_engagement_signals(rl.inner)
                gate.update(l2, r2c, coin, ltp, rtp, terminated=bool(term))
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
            "force_imbalance": round(max_imbalance, 3), "signed_target_displacement": round(float(disp @ e_par), 4),
            "peak_joint_vel": round(peak_joint, 2), "k6": int(md >= HELD_DWELL and touched), "steps": t + 1}


def _unit2(x):
    n = float(np.linalg.norm(x))
    return (np.asarray(x, np.float32) / n) if n > 1e-9 else np.zeros_like(np.asarray(x, np.float32))


def _side_normal(m, d, geom):
    """Peak normal force on contacts involving a specific fingertip geom (per-side force, for the L/R imbalance)."""
    f = np.zeros(6, np.float64)
    peak = 0.0
    for ci in range(d.ncon):
        c = d.contact[ci]
        if geom in (int(c.geom1), int(c.geom2)):
            mujoco.mj_contactForce(m, d, ci, f)
            peak = max(peak, abs(float(f[0])))
    _ = _contact_forces
    return peak
