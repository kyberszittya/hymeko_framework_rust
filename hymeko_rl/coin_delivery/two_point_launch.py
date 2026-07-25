"""TARGET_DIRECTED_LAUNCH_V2 — structural contact-MODE control. V1 showed the load-bearing variable is the contact point /
force-line, and a single reachable point cannot aim the launch from every state. V2 tests whether a TWO-POINT contact can
place the resultant force-line through the coin centre toward the zone. Frozen: V2 material, V4 motion contract, coast
launch target, B1 barrier. Deterministic teacher (no RL).

Force-line mechanism: if both fingertips contact the coin symmetrically about the zone axis on the FAR side and squeeze
toward the centre, the resultant force passes through the COM along e_parallel ⇒ net v_∥ toward the zone, net v_⊥ ≈ 0,
net torque ≈ 0. The direct mechanistic proof is the coin's ANGULAR velocity ω_c: an off-centre (force-line-miss) push
spins the coin; a centre-line push does not.

  L5a symmetric two-point: fixed contact half-angle φ about the far-side axis.
  L5b edge-aware pair select: sweep φ candidates and pick the pair minimising a force-line / cross-track / rotation cost.
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
class TwoPointConfig:
    coast_mu: float = 0.179
    coin_radius: float = 0.02
    phi_deg: float = 35.0              # L5a: contact half-angle about the far-side (−e_parallel) axis
    phi_candidates: tuple = (20.0, 30.0, 40.0, 50.0)   # L5b: edge-aware candidates
    acquire_gain: float = 0.10
    squeeze_gain: float = 0.06         # rad/step — squeeze both tips toward the COM (target-directed launch)
    v_band: float = 0.08
    acquire_steps: int = 90
    edge_aware: bool = False           # L5b: pick φ per state by a short delivery-independent probe
    w_cross: float = 1.0               # L5b cost weights
    w_rot: float = 0.5
    w_par: float = 1.0


def _frame(rl):
    u, _dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(u, np.float32)
    return e_par, np.array([-e_par[1], e_par[0]], np.float32)


def _tip_xy(rl, g):
    return rl.inner.data.geom_xpos[g][:2].astype(np.float32)


def _contact_points(coin_xy, e_par, e_cross, phi_deg, r):
    """Two far-side (−e_parallel) contact points symmetric in e_cross at ±φ — squeezing toward COM from here gives a
    resultant along +e_parallel (toward the zone) with cancelling cross + torque."""
    phi = np.radians(phi_deg)
    p_l = coin_xy + r * (-np.cos(phi) * e_par + np.sin(phi) * e_cross)
    p_r = coin_xy + r * (-np.cos(phi) * e_par - np.sin(phi) * e_cross)
    return p_l.astype(np.float32), p_r.astype(np.float32)


def _two_tip_dir(rl, gl, gr, p_l, p_r, probe=2.0, steps=3):
    """FD joint direction reducing (‖left_tip−p_l‖ + ‖right_tip−p_r‖) — drives BOTH tips to their targets at once."""
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


def _omega(rl):
    """Coin angular velocity (planar rotation DOF) — the force-line-miss proxy."""
    return float(rl.inner.data.qvel[int(rl.inner._disk_x_adr) + 2])


def _select_phi(rl, gl, gr, e_par, e_cross, cfg, pi0, base, stack):
    """L5b: probe each φ candidate (short two-point squeeze on a COPY) and pick the one minimising the force-line/cross/
    rotation cost J = w⊥ v̂⊥² + wθ ω̂_c² + w∥ (v̂∥ − v*)². Delivery-independent."""
    coin0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2]
    best_phi, best_cost = cfg.phi_deg, 1e9
    for phi in cfg.phi_candidates:
        r2 = copy.deepcopy(rl)
        o = _run_two_point(r2, None, pi0, base, stack, cfg, e_par, e_cross, phi, gl, gr, horizon=70, probe_only=True)
        vp, vc, om = o["peak_v_parallel"], o["peak_v_cross"], o["peak_omega"]
        j = cfg.w_cross * vc * vc + cfg.w_rot * om * om + cfg.w_par * (vp - 0.4) ** 2
        if j < best_cost:
            best_phi, best_cost = phi, j
    _ = coin0
    return best_phi


def _run_two_point(rl, gate, pi0, base, stack, cfg, e_par, e_cross, phi, gl, gr, *, horizon, probe_only=False):
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gov = stack.gov

    def _gcb(_model, data):
        data.ctrl[:4] = govern_torque(data.ctrl[:4], data.qvel[:4], gov)
    mujoco.set_mjcb_control(_gcb)
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    peak_vp, peak_vc, peak_om, peak_joint, both_contact, prev_tau = 0.0, 0.0, 0.0, 0.0, 0, None
    md, touched = int(rl._strict), rl._touched
    t = 0
    try:
        for t in range(horizon):
            mpl = rl.inner._planar_metrics
            coin_xy = np.asarray(mpl.disk_pos, np.float32)[:2]
            lc, rcn = bool(mpl.left_contact), bool(mpl.right_contact)
            p_l, p_r = _contact_points(coin_xy, e_par, e_cross, phi, cfg.coin_radius)
            if not probe_only and gate is not None and gate.gate != 1.0:
                a = _det(pi0, rl.obs())
            elif not (lc and rcn):                                # acquire BOTH tips to the far-side points
                delta = cfg.acquire_gain * _two_tip_dir(rl, gl, gr, p_l, p_r)
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + delta, stack, prev_tau, lo, hi)
                prev_tau = a
            else:                                                 # squeeze both tips toward the COM (target-directed launch)
                both_contact = 1
                delta = cfg.squeeze_gain * _two_tip_dir(rl, gl, gr, coin_xy, coin_xy)
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + delta, stack, prev_tau, lo, hi)
                prev_tau = a
            _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
            if not probe_only and gate is not None:
                l2, r2c, coin, lt, rtp = stable_engagement_signals(rl.inner)
                gate.update(l2, r2c, coin, lt, rtp, terminated=bool(term))
            md = max(md, int(rl._strict))
            touched = touched or rl._touched
            cvn = np.asarray(rl.inner._planar_metrics.disk_vel, np.float32)[:2]
            peak_vp = max(peak_vp, float(cvn @ e_par))
            peak_vc = max(peak_vc, abs(float(cvn @ e_cross)))
            peak_om = max(peak_om, abs(_omega(rl)))
            peak_joint = max(peak_joint, float(np.max(np.abs(d.qvel[:4]))))
            if term or trunc:
                break
    finally:
        mujoco.set_mjcb_control(None)
    disp = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2] - disk0
    return {"phi": phi, "peak_v_parallel": round(peak_vp, 3), "peak_v_cross": round(peak_vc, 3),
            "cross_ratio": round(peak_vc / max(1e-6, peak_vp), 3), "peak_omega": round(peak_om, 3),
            "both_tips_contact": both_contact, "signed_target_displacement": round(float(disp @ e_par), 4),
            "peak_joint_vel": round(peak_joint, 2), "k6": int(md >= HELD_DWELL and touched), "steps": t + 1}


def two_point_launch_carry(rl, gate, pi0, base, stack, *, horizon: int, cfg: TwoPointConfig | None = None):
    """Two-point launch. Returns v_parallel/v_cross, force-line-miss proxy (peak coin angular velocity ω_c), both-tips
    contact, signed displacement, chosen φ, and motion. K6/zone are outputs."""
    cfg = cfg or TwoPointConfig()
    m = rl.inner.model
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    e_par, e_cross = _frame(rl)
    phi = _select_phi(rl, gl, gr, e_par, e_cross, cfg, pi0, base, stack) if cfg.edge_aware else cfg.phi_deg
    o = _run_two_point(rl, gate, pi0, base, stack, cfg, e_par, e_cross, phi, gl, gr, horizon=horizon)
    o["chosen_phi"] = phi
    o["contact_mode"] = "edge_aware_two_point" if cfg.edge_aware else "symmetric_two_point"
    _ = _contact_forces  # (available for a direct force-line-miss d_line extension)
    return o
