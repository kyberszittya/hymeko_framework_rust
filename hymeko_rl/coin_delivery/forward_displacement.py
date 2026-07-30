"""CONTROLLED_BIMANUAL_FORWARD_COIN_DISPLACEMENT — a short-horizon slew-admissible torque-PRIMITIVE search.

With the Δτ interface established (Route B: usable one-step Δτ authority on all certified cradles), the coin move is
treated as a DYNAMIC coordinated transition to pass through, NOT a quasi-static cradle to prove stable first. From the
exact certified handoff (free coin), a low-dimensional torque primitive drives a bilateral push toward the target zone:

    θ = (squeeze_mag, forward_mag, balance, ramp_steps, release_step)   — 5 params

mapped, per control step, to a slew-admissible increment Δτ_cmd_t (|Δτ_cmd| ≤ τ̇·dt) built from the LIVE geometry:
  * squeeze_dir  = normalize(inward_L + inward_R)         — grip (each tip toward the coin centre; analytic tip Jacobian)
  * forward_dir  = normalize(Jᵀe_par|_L + Jᵀe_par|_R)     — push both tips toward the zone (e_par = direction_to_zone)
  * balance      shifts the forward push between the two arms (steer / null spin)
schedule: ramp the push+grip increment up over ``ramp_steps``, hold (coast), then relax the grip at ``release_step``.
Applied as a = clip(prev_tau + Δτ_cmd, lo, hi) then the per-sub-step governor — the SAME governed stack as Route B.

Objective (external, physical — NOT a reward): maximise target-directed coin displacement with HARD exclusion on a
motion-contract breach, a coin speed blow-up, or net-backward motion. Success predicate
``CONTROLLED_BIMANUAL_FORWARD_COIN_DISPLACEMENT`` = forward ≥ max(5 mm, 5× passive drift) ∧ forward > 5× passive drift
∧ cross-track bounded ∧ no motion breach ∧ no pin/hidden force. Bounded CEM searches θ (deterministic RNG seed).

NO pin, NO teleport, NO hidden force, NO τ_rate change, V2/V4 read-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot, primary_fingertip_contacts
from hymeko_rl.coin_delivery.mobile_conditioning import _fingertip_geoms, arm_inward_geom
from hymeko_rl.env.governed_arm import pd_governed_torque
from hymeko_rl.env.motion_contract import MotionLimits, govern_torque

_LEFT_DOF, _RIGHT_DOF = (0, 1), (2, 3)


@dataclass(frozen=True)
class ForwardConfig:
    """Frozen forward-displacement primitive + search parameters."""

    horizon: int = 16                    # control steps of the dynamic transition (8–16 per the pivot)
    limits: MotionLimits = field(default_factory=MotionLimits)
    deliver: bool = False                # delivery mode: FROZEN K6 certificate (dtz≤CENTER_TOL ∧ speed<SETTLE_VEL, held-dwell)
    # θ bounds: squeeze_mag, forward_mag (N·m incr), balance (N·m), ramp_steps, release_step, brake_gain (N·m per m/s)
    lo: tuple = (0.0, 0.0, -0.10, 1.0, 4.0, 0.0)
    hi: tuple = (0.25, 0.30, 0.10, 12.0, 16.0, 3.0)
    cross_frac_max: float = 1.0          # cross-track ≤ this × forward displacement (predominantly forward)
    min_forward_m: float = 0.005         # absolute forward threshold floor
    drift_mult: float = 5.0              # forward must exceed this × passive drift
    fn_push_floor: float = 0.05          # N — dual contact must carry ≥ this through the push phase (controlled, not flick)
    release_speed_max: float = 1.0       # m/s — terminal coin speed bound (controlled release, no throw-through)
    lost_penalty: float = 0.01           # m per grip-loss step before release (steer CEM toward contact-retained pushes)
    # CEM
    pop: int = 32
    iters: int = 6
    elite: int = 8
    init_std: tuple = (0.08, 0.10, 0.05, 4.0, 4.0, 0.8)
    cem_seed: int = 20260727


def arm_jac_dir(rl, g, dofs, d_world) -> np.ndarray:
    """Analytic joint-space direction (over this arm's dofs) that moves tip geom ``g`` along the world direction
    ``d_world`` (xy): ``normalize(Jₚ,dofsᵀ·d_world)`` from ``mj_jacGeom``. Coin-motion-independent (valid on the free
    coin). # Postconditions: unit-norm 4-vector nonzero only on ``dofs`` (0 if kinematically decoupled)."""
    m, d = rl.inner.model, rl.inner.data
    jacp = np.zeros((3, m.nv), np.float64)
    mujoco.mj_jacGeom(m, d, jacp, None, int(g))
    dw = np.asarray(d_world, np.float64)
    v = np.zeros(4, np.float64)
    for j in dofs:
        v[j] = jacp[0, j] * dw[0] + jacp[1, j] * dw[1]
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-9 else v


def _coin_xy(rl) -> np.ndarray:
    return np.asarray(rl.inner._planar_metrics.disk_pos, np.float64)[:2]


def _coin_speed(rl) -> float:
    return float(np.linalg.norm(np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]))


def _velocity_damp_dir(rl, gl, gr, v_coin, sp) -> np.ndarray:
    """Unit joint torque driving both tips opposite to the coin velocity (decelerate the coin through the contacts)."""
    if sp <= 1e-4:
        return np.zeros(4)
    neg = -v_coin / sp
    d = arm_jac_dir(rl, gl, _LEFT_DOF, neg) + arm_jac_dir(rl, gr, _RIGHT_DOF, neg)
    return d / (np.linalg.norm(d) + 1e-12)


def _schedule_increment(rl, th, t, e_par, step, coin) -> np.ndarray:
    """The slew-admissible Δτ_cmd at control step ``t`` (1-based) for the primitive
    ``th`` = (squeeze, forward, balance, ramp, release, brake_gain[, settle_gain]). Phases (grip maintained until release):
      * PUSH   (t ≤ ramp):     ramped forward push toward the zone + grip + L/R balance
      * BRAKE  (ramp < t ≤ release): VELOCITY-FEEDBACK brake — a decelerating increment ∝ measured coin speed, driving
                               the tips opposite to the coin velocity (through the contacts) + grip. Not a timed impulse.
      * RELEASE/SETTLE(t > release): with the optional 7th param ``settle_gain > 0``, hold a LIGHT grip + continue the
                               velocity-feedback damp to bring the coin to rest in the zone (rebound suppression — the
                               target-entry→settle→certificate transition). ``settle_gain == 0`` (or a 6-vector θ) is the
                               ORIGINAL behaviour: relax the grip and coast. Backward-compatible.
    All slew-admissible (clipped to ±step) and, downstream, motion-governed."""
    sqz, fwd, bal, ramp, rel, brake_gain = th[:6]
    settle_gain = float(th[6]) if len(th) > 6 else 0.0
    gl, gr = _fingertip_geoms(rl.inner.model)
    inward = arm_inward_geom(rl, gl, _LEFT_DOF, coin) + arm_inward_geom(rl, gr, _RIGHT_DOF, coin)
    squeeze_dir = inward / (np.linalg.norm(inward) + 1e-12)
    if t <= ramp:                                         # PUSH
        fwd_l, fwd_r = arm_jac_dir(rl, gl, _LEFT_DOF, e_par), arm_jac_dir(rl, gr, _RIGHT_DOF, e_par)
        forward_dir = (fwd_l + fwd_r)
        forward_dir = forward_dir / (np.linalg.norm(forward_dir) + 1e-12)
        dtau = sqz * squeeze_dir + fwd * min(1.0, t / max(1.0, ramp)) * forward_dir + bal * (fwd_l - fwd_r)
    elif t <= rel:                                        # BRAKE — velocity feedback (decelerate the coin via the grip)
        v_coin = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
        sp = float(np.linalg.norm(v_coin))
        dtau = sqz * squeeze_dir + brake_gain * sp * _velocity_damp_dir(rl, gl, gr, v_coin, sp)
    elif settle_gain > 0.0:                               # SETTLE — light grip + damp to rest (rebound suppression)
        v_coin = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
        sp = float(np.linalg.norm(v_coin))
        dtau = 0.3 * sqz * squeeze_dir + settle_gain * sp * _velocity_damp_dir(rl, gl, gr, v_coin, sp)
    else:                                                 # RELEASE — relax the grip, let the coin coast (original 6-θ)
        dtau = -0.5 * sqz * squeeze_dir
    return np.clip(dtau, -step, step)


def rollout_primitive(snap: CradleSnapshot, theta, cfg: ForwardConfig, *, frame_hook=None) -> dict:
    """Run the primitive Δτ schedule ``horizon`` steps from the free-coin handoff; measure target-directed / cross-track
    coin displacement, spin, contact retention, and the motion contract. ``frame_hook(rl, t)`` (optional) observes the
    STEPPED rl after each control step (rendering) — non-behavioural. # Postconditions: physical measurements only —
    no reward, no pin, no hidden force."""
    rl = snap.branch()
    d = rl.inner.data
    e_par, dtz_start = rl.inner.direction_to_zone()
    e_par = np.asarray(e_par, np.float64)
    e_cross = np.array([-e_par[1], e_par[0]], np.float64)
    coin0 = _coin_xy(rl)
    step = float(snap.stack.tau_rate * snap.stack.control_dt)
    prev_tau = snap.prev_tau.copy()
    trace: list = []

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)
    mujoco.set_mjcb_control(_gcb)
    rel = int(round(float(theta[4])))                     # release step: dual contact must hold up to here
    peak_qdot, peak_coin_speed, contact_lost_steps, lost_before_release = 0.0, 0.0, 0, 0
    straddle_min, min_fn_push = 1.0, 1e9
    forward_at_release, terminal_speed = 0.0, 0.0
    k6_dwell, k6_max_dwell, touched = 0, 0, False       # FROZEN K6 monitor: dtz≤CENTER_TOL ∧ speed<SETTLE_VEL, HELD_DWELL, touched
    try:
        for t in range(1, cfg.horizon + 1):
            coin = _coin_xy(rl)
            dtau = _schedule_increment(rl, theta, t, e_par, step, coin)
            a = np.clip(prev_tau + dtau, snap.lo, snap.hi)
            prev_tau = a
            step_ablation(rl, np.asarray(a, np.float32), "A")
            trace.append(_coin_xy(rl).copy())
            peak_qdot = max(peak_qdot, float(np.max(np.abs(d.qvel[:4]))))
            speed_t = _coin_speed(rl)
            peak_coin_speed = max(peak_coin_speed, speed_t)
            terminal_speed = speed_t
            _u_t, dtz_t = rl.inner.direction_to_zone()
            con = primary_fingertip_contacts(rl)
            both = con["left"] is not None and con["right"] is not None
            touched = touched or both
            k6_dwell = k6_dwell + 1 if (dtz_t <= CENTER_TOL and speed_t < SETTLE_VEL) else 0
            k6_max_dwell = max(k6_max_dwell, k6_dwell)
            if both:
                straddle_min = min(straddle_min, float(con["left"]["n"] @ con["right"]["n"]))
                if t <= rel:
                    min_fn_push = min(min_fn_push, con["left"]["fn"], con["right"]["fn"])
            else:
                contact_lost_steps += 1
                if t <= rel:
                    lost_before_release += 1
            if t == rel:
                forward_at_release = float((_coin_xy(rl) - coin0) @ e_par)
            if frame_hook is not None:
                frame_hook(rl, t)
    finally:
        mujoco.set_mjcb_control(None)
    k6_delivered = bool(k6_max_dwell >= HELD_DWELL and touched)
    disp = _coin_xy(rl) - coin0
    forward = float(disp @ e_par)
    cross = float(abs(disp @ e_cross))
    _u_end, dtz_end = rl.inner.direction_to_zone()
    return {"forward": forward, "cross": cross, "total_disp": float(np.linalg.norm(disp)),
            "dtz_start": float(dtz_start), "dtz_end": float(dtz_end),
            "gap_closed": float((dtz_start - dtz_end) / max(dtz_start, 1e-9)),
            "k6_max_dwell": int(k6_max_dwell), "k6_delivered": k6_delivered, "touched": bool(touched),
            "forward_at_release": forward_at_release, "peak_qdot": peak_qdot, "peak_coin_speed": peak_coin_speed,
            "terminal_coin_speed": terminal_speed, "contact_lost_steps": contact_lost_steps,
            "lost_before_release": lost_before_release, "release_step": rel,
            "straddle_min": (None if straddle_min == 1.0 else straddle_min),
            "min_fn_push": (None if min_fn_push == 1e9 else float(min_fn_push)), "coin_trace": [c.tolist() for c in trace]}


def rotate2d(v: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a 2-vector by ``deg`` degrees (scenario-relative direction correction — never a world-frame rule)."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    v = np.asarray(v, np.float64)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], np.float64)


def run_align_segment(snap: CradleSnapshot, e_par_align: np.ndarray, align_theta: "tuple[float, ...]",
                      cfg: ForwardConfig, n_steps: int) -> "tuple[Any, np.ndarray, dict[str, Any]]":
    """TARGET_RELATIVE_ALIGNMENT_PHASE: run ``n_steps`` of the primitive's PUSH schedule toward ``e_par_align`` (a
    scenario-relative correction direction), grip maintained, on a fresh branch of ``snap``. This is NOT a shorter first
    PUSH toward the zone — it drives the coin toward a *virtual* intermediate direction to set up a favourable
    force-transfer state for the transport phase that follows.

    Reuses the exact same ``_schedule_increment`` PUSH physics (``align_theta`` = (squeeze, forward, balance, ramp, rel,
    brake_gain) with ramp = n_steps, rel > n_steps so the whole segment stays in PUSH — no brake/release).

    # Preconditions: ``snap`` a certified-grasp handoff; ``n_steps >= 1``; ``e_par_align`` a unit 2-vector.
    # Postconditions: returns the LIVE post-align branch ``rl`` (to build the transport snapshot from), the last governed
    #   ``prev_tau``, and read-only ALIGN metrics (velocity-direction cosine vs the TRUE target, lateral speed, grasp
    #   retention, coin displacement). No state edit, no hidden force — every command passes the governed step.
    """
    rl = snap.branch()
    e_par_true, _dtz = rl.inner.direction_to_zone()
    e_par_true = np.asarray(e_par_true, np.float64)
    coin0 = _coin_xy(rl)
    step = float(snap.stack.tau_rate * snap.stack.control_dt)
    prev_tau = snap.prev_tau.copy()

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)

    mujoco.set_mjcb_control(_gcb)
    lost = 0
    try:
        for t in range(1, n_steps + 1):
            coin = _coin_xy(rl)
            dtau = _schedule_increment(rl, align_theta, t, e_par_align, step, coin)
            prev_tau = np.clip(prev_tau + dtau, snap.lo, snap.hi)
            step_ablation(rl, np.asarray(prev_tau, np.float32), "A")
            con = primary_fingertip_contacts(rl)
            if not (con["left"] is not None and con["right"] is not None):
                lost += 1
    finally:
        mujoco.set_mjcb_control(None)
    v_coin = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
    sp = float(np.linalg.norm(v_coin))
    cos_true = float(v_coin @ e_par_true / sp) if sp > 1e-6 else 1.0
    lateral = float(abs(v_coin @ np.array([-e_par_true[1], e_par_true[0]])))
    con = primary_fingertip_contacts(rl)
    held = con["left"] is not None and con["right"] is not None
    metrics = {"cos_true": round(cos_true, 4), "lateral_speed": round(lateral, 4), "grasp_held": bool(held),
               "lost_steps": int(lost), "coin_disp_mm": round(float(np.linalg.norm(_coin_xy(rl) - coin0)) * 1000, 2)}
    return rl, prev_tau, metrics


def passive_drift(snap: CradleSnapshot, cfg: ForwardConfig) -> float:
    """Target-directed coin displacement under the POSITION hold (pd servo at q_hold — the controller trying to keep the
    coin PUT) — the honest 'no active push' baseline. A torque hold (Δτ=0) is NOT stationary: the residual acquisition
    torque itself pushes the coin, which would inflate the baseline; the position hold is the true stay-put reference."""
    rl = snap.branch()
    d = rl.inner.data
    e_par, _dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(e_par, np.float64)
    coin0 = _coin_xy(rl)
    prev_tau = snap.prev_tau.copy()

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)
    mujoco.set_mjcb_control(_gcb)
    try:
        for _t in range(cfg.horizon):
            a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), snap.q_hold, snap.stack, prev_tau, snap.lo, snap.hi)
            prev_tau = a
            step_ablation(rl, np.asarray(a, np.float32), "A")
    finally:
        mujoco.set_mjcb_control(None)
    return abs(float((_coin_xy(rl) - coin0) @ e_par))


def _feasible(m: dict, cfg: ForwardConfig) -> bool:
    """Hard exclusion: motion-contract breach, coin-speed blow-up, or net-backward motion."""
    return bool(m["peak_qdot"] <= cfg.limits.joint_vel_hard and m["peak_coin_speed"] <= cfg.limits.ee_speed_hard
                and m["forward"] > 0.0)


def score(m: dict, cfg: ForwardConfig) -> float:
    """CEM score: −inf if infeasible; else (deliver mode) reward closing the distance-to-zone and settling in-zone;
    (forward mode) forward displacement minus cross-track + grip-loss + over-speed penalties (a CONTROLLED push)."""
    if not _feasible(m, cfg):
        return -np.inf
    if cfg.deliver:                        # push-and-coast: close the distance-to-zone + accumulate the FROZEN K6 held-dwell
        return (-m["dtz_end"] - cfg.lost_penalty * m["lost_before_release"]
                + 0.02 * m["k6_max_dwell"] + (0.10 if m["k6_delivered"] else 0.0))
    return (m["forward"] - 0.5 * max(0.0, m["cross"] - cfg.cross_frac_max * m["forward"])
            - cfg.lost_penalty * m["lost_before_release"]
            - 0.5 * max(0.0, m["terminal_coin_speed"] - cfg.release_speed_max))    # prefer a controlled release


def delivery_success(m: dict, cfg: ForwardConfig) -> bool:
    """The FROZEN external K6 delivery certificate: coin held in-zone (dtz ≤ CENTER_TOL ∧ speed < SETTLE_VEL) for
    HELD_DWELL consecutive steps and touched — with the motion contract held over a single continuous trajectory
    (no re-grip / teleport / coin-state edit). This is the real monitor, NOT a loose proxy."""
    return bool(m["k6_delivered"] and m["peak_qdot"] <= cfg.limits.joint_vel_hard
                and m["peak_coin_speed"] <= cfg.limits.ee_speed_hard)


def success(m: dict, threshold: float, passive: float, cfg: ForwardConfig) -> bool:
    """External CONTROLLED_BIMANUAL_FORWARD_COIN_DISPLACEMENT certificate (physical, not reward): enough target-directed
    displacement over passive drift, predominantly forward, dual contact RETAINED through the push phase with real
    force, a CONTROLLED release (bounded terminal coin speed), and no motion-contract breach."""
    return bool(m["forward"] >= threshold and m["forward"] > cfg.drift_mult * passive
                and m["cross"] <= cfg.cross_frac_max * m["forward"]
                and m["lost_before_release"] == 0 and (m["min_fn_push"] or 0.0) >= cfg.fn_push_floor
                and m["terminal_coin_speed"] <= cfg.release_speed_max
                and m["peak_qdot"] <= cfg.limits.joint_vel_hard and m["peak_coin_speed"] <= cfg.limits.ee_speed_hard)


def _clip_theta(v, cfg: ForwardConfig) -> np.ndarray:
    return np.clip(v, np.asarray(cfg.lo, np.float64), np.asarray(cfg.hi, np.float64))


def cem_search(snap: CradleSnapshot, cfg: ForwardConfig) -> dict:
    """Bounded CEM over the 5-D primitive θ maximising ``score`` (deterministic RNG). Returns the best feasible θ, its
    metrics, the passive-drift baseline, the frozen success threshold, and the search history."""
    rng = np.random.default_rng(cfg.cem_seed)
    passive = passive_drift(snap, cfg)
    threshold = max(cfg.min_forward_m, cfg.drift_mult * passive)
    mean = 0.5 * (np.asarray(cfg.lo, np.float64) + np.asarray(cfg.hi, np.float64))
    std = np.asarray(cfg.init_std, np.float64).copy()
    best = {"score": -np.inf, "theta": None, "metrics": None}
    history = []
    for it in range(cfg.iters):
        pop = _clip_theta(mean + std * rng.standard_normal((cfg.pop, 6)), cfg)
        scored = []
        for v in pop:
            m = rollout_primitive(snap, tuple(v), cfg)
            s = score(m, cfg)
            scored.append((s, v, m))
            if s > best["score"]:
                best = {"score": float(s), "theta": [round(float(x), 4) for x in v], "metrics": m}
        scored.sort(key=lambda z: z[0], reverse=True)
        elite = np.asarray([v for _s, v, _m in scored[:cfg.elite]])
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-6
        n_feas = int(sum(1 for s, _v, _m in scored if np.isfinite(s)))
        history.append({"iter": it, "best_score": round(float(best["score"]), 5), "n_feasible": n_feas,
                        "best_forward": round(float(best["metrics"]["forward"]), 5) if best["metrics"] else None})
    ok = False
    if best["metrics"]:
        ok = (delivery_success(best["metrics"], cfg) if cfg.deliver
              else success(best["metrics"], threshold, passive, cfg))
    return {"passive_drift": round(passive, 6), "threshold": round(threshold, 6),
            "best_theta": best["theta"], "best_metrics": best["metrics"], "success": ok, "history": history}
