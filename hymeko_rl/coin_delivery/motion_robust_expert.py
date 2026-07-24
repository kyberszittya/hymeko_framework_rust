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

import numpy as np

from hymeko_rl.coin_delivery.coin_carry_structured import (
    ACTION_SCALE, CENTER_TOL, HELD_DWELL, SETTLE_VEL, _aug, _det, stable_engagement_signals)
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation


@dataclass(frozen=True)
class CarryControllerConfig:
    """Physical, interpretable gains. Defaults are a starting point; C2 ablates the mechanisms these enable."""

    v_transport: float = 0.10          # m/s — the capped low transport speed of the coin toward the zone
    approach_gain: float = 1.2         # 1/s — desired coin speed ∝ (distance − tol), capped at v_transport
    track_gain: float = 6.0            # arm action gain to track the desired coin velocity (via the contact Jacobian)
    brake_gain: float = 8.0            # arm action gain opposing the coin velocity in the BRAKE phase
    brake_k: float = 6.0               # braking-distance ≈ brake_k·v² + brake_lin·|v| (predicted stopping distance)
    brake_lin: float = 0.4
    brake_margin: float = 0.01         # m — enter BRAKE this much before the predicted braking distance
    release_band: float = 1.5 * SETTLE_VEL   # coin speed below which release/settle is admissible
    acquire_mag: float = 3.0           # push magnitude along the tip-approach direction to (re)acquire contact
    replan_every: int = 4              # re-estimate the contact Jacobian every N steps (receding horizon)
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


def _macro_action(rl, J, cfg: CarryControllerConfig):
    """The closed-loop action + phase label, from the live state. Braking distance decides the push end, not a clock."""
    u, dtz = rl.inner.direction_to_zone()
    m = rl.inner._planar_metrics
    coin_vel = np.asarray(m.disk_vel, np.float32)
    contact = bool(m.left_contact or m.right_contact)
    v_along = float(coin_vel @ np.asarray(u, np.float32))
    brake_d = cfg.brake_k * v_along * v_along + cfg.brake_lin * abs(v_along)
    if dtz <= CENTER_TOL and rl._speed() < cfg.release_band:
        return np.zeros(4, np.float32), "SETTLE"                 # active settling: release, let damping bring it to rest
    if dtz <= brake_d + cfg.brake_margin and v_along > cfg.release_band:
        return _jac_solve(J, -coin_vel * cfg.brake_gain), "BRAKE"   # velocity-aware braking (oppose the coin velocity)
    if not contact:                                             # ACQUIRE: close the fingertip gap (contact J degenerate)
        return cfg.acquire_mag * _acquire_direction(rl, cfg), "ACQUIRE"
    v_des = min(cfg.v_transport, max(0.0, dtz - CENTER_TOL) * cfg.approach_gain) * np.asarray(u, np.float32)
    return _jac_solve(J, (v_des - coin_vel) * cfg.track_gain), "TRANSPORT"   # contact-retaining low-speed transport


def motion_robust_carry(rl, gate, pi0, base, *, horizon: int, cfg: CarryControllerConfig | None = None, frame_hook=None):
    """Receding-horizon closed-loop delivery until a valid handoff (strict≥1), then the FROZEN settling pi_0 — the same
    outer contract as structured_carry_rollout, with the OPEN-LOOP macro replaced by the physical closed-loop controller.
    Returns the K6/handoff outcome + PHASE-LEVEL metrics (which phase reached / failed)."""
    cfg = cfg or CarryControllerConfig()
    handed = False
    md = int(rl._strict)
    touched = rl._touched
    J = None
    phases_seen = {"ACQUIRE": 0, "TRANSPORT": 0, "BRAKE": 0, "SETTLE": 0}
    contact_frames = 0
    entered_zone = False
    for t in range(horizon):
        s = int(rl._strict)
        gate_on = gate.gate == 1.0
        if not gate_on:
            a = _det(pi0, rl.obs())
            cur = "PI0"
        elif handed or s >= 1:
            handed = True
            a = _det(base, _aug(rl.obs(), s))                   # frozen settling after a valid handoff
            cur = "SETTLE"
        else:
            if J is None or t % cfg.replan_every == 0:
                J = _contact_jacobian(rl, cfg)
            a, cur = _macro_action(rl, J, cfg)
            phases_seen[cur] = phases_seen.get(cur, 0) + 1
        a = np.clip(np.asarray(a, np.float32), -ACTION_SCALE, ACTION_SCALE)
        _r, term, trunc = step_ablation(rl, a, "A")
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
        gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        md = max(md, int(rl._strict))
        touched = touched or rl._touched
        contact_frames += int(bool(rl.inner._planar_metrics.left_contact or rl.inner._planar_metrics.right_contact))
        entered_zone = entered_zone or (rl._dtz() <= CENTER_TOL)
        if frame_hook is not None:
            frame_hook(cur, int(rl._strict))
        if term or trunc:
            break
    n = max(1, t + 1)
    return {"k6": int(md >= HELD_DWELL and touched), "max_dwell": md, "reached_handoff": int(md >= 1),
            "touched": int(touched), "contact_frac": round(contact_frames / n, 3), "entered_zone": int(entered_zone),
            "acquired_contact": int(contact_frames > 0), "phases": phases_seen, "steps": n}
