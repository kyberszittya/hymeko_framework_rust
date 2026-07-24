"""C1 — motion-robust receding-horizon coin expert (NO RL, NO proposal), for the frozen COIN_DYNAMICS_CONTRACT_V2.

The legacy push→brake→release macro was open-loop and impulsive — under realistic motion limits it collapses (G1: expert
0.312). This controller instead follows the task PHYSICS in closed loop: at every control step it reads the coin
position/velocity, distance-to-zone, contact state, and a live contact Jacobian, and picks a phase action —

    acquire contact → contact-retaining LOW-SPEED transport → velocity-aware braking → braking-distance-gated release →
    active settling → (valid handoff → frozen settling pi_0) → K6

The push END is not a fixed time: it brakes/releases when ``remaining distance ≈ predicted braking distance`` and the coin
velocity is inside the release band. It is a physical controller, not a slower-played throw. The goal of C1 is to show the
capability is RECOVERABLE under V2 (beat the 0.312 expert), and to expose which PHASE is the wall (phase-level metrics).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_carry_structured import (
    CENTER_TOL, HELD_DWELL, SETTLE_VEL, _aug, _det, stable_engagement_signals)
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.env.governed_arm import pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque


@dataclass(frozen=True)
class CarryControllerConfig:
    """The high-level controller emits POSITION-TARGET DISPLACEMENTS (rad) — the full torque path is the shared
    GovernedArm stack (PD → rate limit → per-sub-step governor → saturation), which the controller cannot bypass."""

    track_disp: float = 0.06           # rad — per-step joint displacement toward moving the coin to the zone
    acquire_disp: float = 0.10         # rad — per-step joint displacement along the tip-approach direction
    brake_disp: float = 0.05           # rad — per-step joint displacement opposing the coin velocity (active braking)
    brake_k: float = 6.0               # braking-distance ≈ brake_k·v² + brake_lin·|v| (predicted stopping distance)
    brake_lin: float = 0.4
    brake_margin: float = 0.01         # m — enter BRAKE this much before the predicted braking distance
    release_band: float = 1.5 * SETTLE_VEL   # coin speed below which release/settle is admissible
    replan_every: int = 4              # re-estimate the contact Jacobian every N steps (receding horizon; ≥horizon = off)
    enable_braking: bool = True        # C2 ablation: velocity-aware braking + braking-distance-gated release (off ⇒ arm C)
    probe_mag: float = 2.0
    probe_steps: int = 4


def _contact_jacobian(rl, cfg: CarryControllerConfig) -> np.ndarray:
    """Live contact Jacobian J (2×4): d(coin xy)/d(arm action), by a short finite-difference probe on a COPY (no state
    change to ``rl``). The receding-horizon controller re-estimates this so transport/braking track the true coupling."""
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    J = np.zeros((2, 4), np.float32)
    for j in range(4):
        r2 = copy.deepcopy(rl)
        a = np.zeros(4, np.float32)
        a[j] = cfg.probe_mag
        for _ in range(cfg.probe_steps):
            step_ablation(r2, a, "A")
        J[:, j] = (np.asarray(r2.inner._planar_metrics.disk_pos, np.float32)[:2] - disk0) / (cfg.probe_mag * cfg.probe_steps)
    return J


def _jac_solve(J: np.ndarray, v_des: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """Damped-least-squares arm action that produces the desired coin velocity: a = Jᵀ(JJᵀ+λ²I)⁻¹ v."""
    return (J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(2), np.asarray(v_des, np.float64))).astype(np.float32)


def _acquire_direction(rl, cfg: CarryControllerConfig) -> np.ndarray:
    """Unit joint direction that REDUCES the nearer fingertip's distance to the coin — the pre-contact ACQUIRE primitive
    (the contact Jacobian is degenerate before contact, so transport/braking cannot acquire). FD probe of
    d(min tip distance)/d(a) on copies; move down that gradient."""
    m0 = rl.inner._planar_metrics
    d0 = min(m0.left_tip_dist, m0.right_tip_dist)
    grad = np.zeros(4, np.float32)
    for j in range(4):
        r2 = copy.deepcopy(rl)
        a = np.zeros(4, np.float32)
        a[j] = cfg.probe_mag
        for _ in range(cfg.probe_steps):
            step_ablation(r2, a, "A")
        m1 = r2.inner._planar_metrics
        grad[j] = (min(m1.left_tip_dist, m1.right_tip_dist) - d0) / (cfg.probe_mag * cfg.probe_steps)
    d = -grad
    nrm = float(np.linalg.norm(d))
    return (d / nrm).astype(np.float32) if nrm > 1e-6 else np.zeros(4, np.float32)


def _unit(x):
    n = float(np.linalg.norm(x))
    return (np.asarray(x, np.float32) / n) if n > 1e-9 else np.zeros_like(np.asarray(x, np.float32))


def _macro_delta(rl, J, cfg: CarryControllerConfig):
    """The closed-loop POSITION-TARGET DISPLACEMENT (Δq, rad) + phase label, from the live state. Braking distance
    decides the push end, not a clock. Δq goes to the shared PD stack — the controller emits no torque."""
    u, dtz = rl.inner.direction_to_zone()
    m = rl.inner._planar_metrics
    coin_vel = np.asarray(m.disk_vel, np.float32)
    contact = bool(m.left_contact or m.right_contact)
    v_along = float(coin_vel @ np.asarray(u, np.float32))
    brake_d = cfg.brake_k * v_along * v_along + cfg.brake_lin * abs(v_along)
    if dtz <= CENTER_TOL and rl._speed() < cfg.release_band:
        return np.zeros(4, np.float32), "SETTLE"                        # hold position; damping brings the coin to rest
    if cfg.enable_braking and dtz <= brake_d + cfg.brake_margin and v_along > cfg.release_band:
        return cfg.brake_disp * _unit(-_jac_solve(J, coin_vel)), "BRAKE"   # velocity-aware braking (ablation: arm D/E)
    if not contact:                                                     # ACQUIRE: close the fingertip gap
        return cfg.acquire_disp * _acquire_direction(rl, cfg), "ACQUIRE"
    return cfg.track_disp * _unit(_jac_solve(J, np.asarray(u, np.float32))), "TRANSPORT"   # push toward the zone


def motion_robust_carry(rl, gate, pi0, base, stack, *, horizon: int, cfg: CarryControllerConfig | None = None,
                        frame_hook=None):
    """Receding-horizon closed-loop delivery on the SHARED, physically-inescapable V3 stack: this function sets the
    stack's armature/damping/friction on the model AND installs the per-sub-step governor (mjcb_control), then drives the
    macro phases as POSITION TARGETS through the shared ``pd_governed_torque`` path. The frozen settling pi_0 (after a
    valid handoff) is the only non-PD action, and it is still governed per sub-step. Returns the K6/handoff outcome +
    PHASE-LEVEL metrics + the realised motion (peak/integrated overspeed, governor activation). No raw-torque bypass."""
    cfg = cfg or CarryControllerConfig()
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4] = stack.armature
    m.dof_damping[:4] = stack.damping
    m.dof_frictionloss[:4] = stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gov = stack.gov

    def _gcb(_model, data):
        data.ctrl[:4] = govern_torque(data.ctrl[:4], data.qvel[:4], gov)
    mujoco.set_mjcb_control(_gcb)
    handed, md, touched, J, prev_tau = False, int(rl._strict), rl._touched, None, None
    phases = {"ACQUIRE": 0, "TRANSPORT": 0, "BRAKE": 0, "SETTLE": 0}
    contact_frames, entered_zone, peak_vel, integ_over = 0, False, 0.0, 0.0
    t = 0
    try:
        for t in range(horizon):
            s = int(rl._strict)
            if gate.gate != 1.0:
                a = _det(pi0, rl.obs())
                cur = "PI0"
            elif handed or s >= 1:
                handed = True
                a = _det(base, _aug(rl.obs(), s))                       # frozen settling torque (still governed per sub-step)
                cur = "SETTLE"
            else:
                if J is None or t % cfg.replan_every == 0:
                    J = _contact_jacobian(rl, cfg)
                delta, cur = _macro_delta(rl, J, cfg)                    # POSITION-TARGET displacement (rad)
                q, qd = d.qpos[:4].copy(), d.qvel[:4].copy()
                a = pd_governed_torque(q, qd, q + delta, stack, prev_tau, lo, hi)   # SHARED torque path
                prev_tau = a
                phases[cur] = phases.get(cur, 0) + 1
            _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
            gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            md = max(md, int(rl._strict))
            touched = touched or rl._touched
            v = float(np.max(np.abs(d.qvel[:4])))
            peak_vel = max(peak_vel, v)
            integ_over += max(0.0, v / 2.0 - 1.0) * stack.control_dt     # ∫ReLU(|q̇|/q̇_safe−1)dt  (q̇_safe≈2)
            contact_frames += int(bool(rl.inner._planar_metrics.left_contact or rl.inner._planar_metrics.right_contact))
            entered_zone = entered_zone or (rl._dtz() <= CENTER_TOL)
            if frame_hook is not None:
                frame_hook(cur, int(rl._strict))
            if term or trunc:
                break
    finally:
        mujoco.set_mjcb_control(None)
    n = max(1, t + 1)
    return {"k6": int(md >= HELD_DWELL and touched), "max_dwell": md, "reached_handoff": int(md >= 1),
            "touched": int(touched), "contact_frac": round(contact_frames / n, 3), "entered_zone": int(entered_zone),
            "acquired_contact": int(contact_frames > 0), "phases": phases, "steps": n,
            "peak_joint_vel": round(peak_vel, 2), "integrated_overspeed": round(integ_over, 3)}
