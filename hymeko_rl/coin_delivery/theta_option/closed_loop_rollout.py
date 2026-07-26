"""R4 Stage 2 — the continuous closed-loop rollout driver + the budget-8 closed-loop search.

`rollout_primitive`'s `frame_hook` is observe-only and post-step, so it cannot swap θ mid-trajectory. This driver runs ONE
continuous physical trajectory (a single `snap.branch()`, stepped 1…horizon, NO reset / snapshot restore) and at each step
asks a *controller* for the effective θ, then reuses the frozen option VERBATIM — the per-step Δτ map
`_schedule_increment`, the governed step `step_ablation` + the `govern_torque` control callback, and the IDENTICAL
K6 / peak / contact accounting as `rollout_primitive`. It returns a `rollout_primitive`-shaped metrics dict, so
`delivery_success` / `score` are the unchanged judges.

Anti-divergence guarantee: under a `ConstantController` (fixed θ, no correction) this driver is **bit-identical** to
`rollout_primitive(snap, θ)` — every metric incl. `coin_trace` (the Stage-3 golden). That proves the closed-loop rollout
is a strict generalisation preserving the frozen option exactly, and justifies expressing the loop separately (fixed-θ vs
controller-driven are genuinely different control flow, §6.5 #8). `forward_displacement.py` is NOT modified.
"""
from __future__ import annotations

from typing import Any, Protocol

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot, primary_fingertip_contacts
from hymeko_rl.coin_delivery.forward_displacement import (
    ForwardConfig, _coin_speed, _coin_xy, _schedule_increment, delivery_success, score as fd_score)
from hymeko_rl.coin_delivery.theta_option.authority_decoder import decode_from_canonical
from hymeko_rl.coin_delivery.theta_option.closed_loop_intent import (
    ClosedLoopController, CorrectionParams, IntentCorrector, SnapContext)
from hymeko_rl.coin_delivery.theta_option.physical_intent import PhysicalIntent, slew_of
from hymeko_rl.coin_delivery.theta_option.search import SEARCH_STD, ThetaCandidateGenerator
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG, ThetaBox
from hymeko_rl.env.motion_contract import govern_torque


class Controller(Protocol):
    """The step interface `closed_loop_rollout` drives. `theta_for_step` returns the effective θ (phase-monotone) for step
    t; `before_release`/`release_boundary_step` define the contact-retention bookkeeping window (so a constant controller
    reproduces `rollout_primitive`'s rounded-`rel` semantics exactly)."""

    def reset(self) -> None: ...
    def theta_for_step(self, rl: Any, t: int, prev_tau: np.ndarray) -> np.ndarray: ...
    def before_release(self, t: int) -> bool: ...
    @property
    def release_boundary_step(self) -> "int | None": ...


class ConstantController:
    """Returns a FIXED θ every step (no correction). Reproduces `rollout_primitive`'s bookkeeping exactly: the per-step map
    sees the raw float θ; the contact-retention window uses `rel = int(round(θ[4]))`. Used by the bit-identity golden and
    as the open-loop leg of a closed-loop candidate when no correction is desired."""

    def __init__(self, theta: np.ndarray) -> None:
        self.theta = np.asarray(theta, np.float64).copy()
        self.rel = int(round(float(self.theta[4])))

    def reset(self) -> None:
        pass

    def theta_for_step(self, rl: Any, t: int, prev_tau: np.ndarray) -> np.ndarray:
        return self.theta

    def before_release(self, t: int) -> bool:
        return int(t) <= self.rel

    @property
    def release_boundary_step(self) -> "int | None":
        return self.rel


def closed_loop_rollout(snap: CradleSnapshot, controller: Controller, cfg: ForwardConfig, *, frame_hook: Any = None) -> dict:
    """Run one continuous closed-loop trajectory from the free-coin handoff, ``controller`` supplying the effective θ each
    step. Reuses the frozen `_schedule_increment` / governed step / K6 monitor VERBATIM. # Postconditions: a
    `rollout_primitive`-shaped metrics dict (same keys); physical measurements only — no pin/teleport/coin edit; ONE
    continuous `rl` (no reset). Under a ConstantController: bit-identical to `rollout_primitive(snap, θ)`."""
    controller.reset()
    rl = snap.branch()
    d = rl.inner.data
    e_par, dtz_start = rl.inner.direction_to_zone()
    e_par = np.asarray(e_par, np.float64)
    e_cross = np.array([-e_par[1], e_par[0]], np.float64)
    coin0 = _coin_xy(rl)
    step = float(snap.stack.tau_rate * snap.stack.control_dt)
    prev_tau = snap.prev_tau.copy()
    trace: list = []

    def _gcb(_mo: Any, dt: Any) -> None:
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)
    mujoco.set_mjcb_control(_gcb)
    peak_qdot, peak_coin_speed, contact_lost_steps, lost_before_release = 0.0, 0.0, 0, 0
    straddle_min, min_fn_push = 1.0, 1e9
    forward_at_release, terminal_speed = 0.0, 0.0
    k6_dwell, k6_max_dwell, touched = 0, 0, False
    try:
        for t in range(1, cfg.horizon + 1):
            coin = _coin_xy(rl)
            theta_t = controller.theta_for_step(rl, t, prev_tau)       # closed-loop: effective θ (phase-monotone)
            dtau = _schedule_increment(rl, theta_t, t, e_par, step, coin)
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
            before_rel = controller.before_release(t)
            if both:
                straddle_min = min(straddle_min, float(con["left"]["n"] @ con["right"]["n"]))
                if before_rel:
                    min_fn_push = min(min_fn_push, con["left"]["fn"], con["right"]["fn"])
            else:
                contact_lost_steps += 1
                if before_rel:
                    lost_before_release += 1
            if controller.release_boundary_step is not None and t == controller.release_boundary_step:
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
    rel_out = controller.release_boundary_step if controller.release_boundary_step is not None else cfg.horizon
    return {"forward": forward, "cross": cross, "total_disp": float(np.linalg.norm(disp)),
            "dtz_start": float(dtz_start), "dtz_end": float(dtz_end),
            "gap_closed": float((dtz_start - dtz_end) / max(dtz_start, 1e-9)),
            "k6_max_dwell": int(k6_max_dwell), "k6_delivered": k6_delivered, "touched": bool(touched),
            "forward_at_release": forward_at_release, "peak_qdot": peak_qdot, "peak_coin_speed": peak_coin_speed,
            "terminal_coin_speed": terminal_speed, "contact_lost_steps": contact_lost_steps,
            "lost_before_release": lost_before_release, "release_step": int(rel_out),
            "straddle_min": (None if straddle_min == 1.0 else straddle_min),
            "min_fn_push": (None if min_fn_push == 1e9 else float(min_fn_push)), "coin_trace": [c.tolist() for c in trace]}


def decode_initial_theta(snap: CradleSnapshot, intent: PhysicalIntent, cfg: ForwardConfig = DELIVERY_CFG) -> np.ndarray:
    """The initial-segment θ₀ = frozen R3 decode of the initial intent at ``snap`` (physical θ). Physics-free re-decodes
    thereafter go through the controller's cached canonical context."""
    from hymeko_rl.coin_delivery.theta_option.canonical_frame import canonicalise, r1_grouped_features
    g_canon, sw = canonicalise(r1_grouped_features(snap))
    return np.asarray(decode_from_canonical(intent, g_canon, sw, slew_of(snap), float(cfg.horizon)).physical_theta, np.float64)


def closed_loop_search_select(snap: CradleSnapshot, base_intent: PhysicalIntent, corrector: IntentCorrector,
                              params: CorrectionParams, rng: np.random.Generator, *, budget: int,
                              cfg: ForwardConfig = DELIVERY_CFG, std: float = SEARCH_STD) -> "dict[str, Any]":
    """The budget-8 CENTRE-INCLUSIVE search over the INITIAL-segment θ₀ (mirrors `search.fixed_search_select`: same
    generator, same frozen std, same centre-inclusion), where each candidate is a FULL continuous closed-loop trajectory
    (probe → measure → correct → continue). Corrections add NO budget (design freeze §1). Keeps the winning controller's
    correction/response trace for provenance. # Postconditions: exactly max(1,budget) candidates; θ₀ (Bellman-style centre)
    ≠ θ_exec object; selected score ≥ centre score (monotone)."""
    box = ThetaBox()
    sc = SnapContext.build(snap, cfg)                          # authority FD computed ONCE, shared across candidates
    center = box.clip(np.asarray(decode_from_canonical(
        base_intent, sc.g_canon, sc.was_swapped, sc.slew, sc.horizon).physical_theta, np.float64)).astype(np.float64)
    cands = [center]
    if int(budget) > 1:
        gen = ThetaCandidateGenerator(std=std, box=box)
        cands += [np.asarray(c, np.float64) for c in gen.sample(center, int(budget) - 1, rng)]
    best = None
    for i, c in enumerate(cands):
        ctrl = ClosedLoopController(snap, base_intent, c, corrector, params, cfg, snap_ctx=sc)
        m = closed_loop_rollout(snap, ctrl, cfg)
        cand_score = float(fd_score(m, cfg))
        if best is None or cand_score > best["score"]:
            best = {"idx": i, "score": cand_score, "theta_exec": np.asarray(c, np.float64), "metrics": m, "controller": ctrl}
    assert best is not None
    m = best["metrics"]
    return {"theta0": [round(float(x), 5) for x in center], "theta_exec": [round(float(x), 5) for x in best["theta_exec"]],
            "search_displacement_norm": round(float(np.linalg.norm(box.norm(best["theta_exec"]) - box.norm(center))), 5),
            "budget_total": len(cands), "score": best["score"],
            "k6_delivered": bool(m["k6_delivered"]), "k6_max_dwell": int(m["k6_max_dwell"]),
            "delivery_success": bool(delivery_success(m, cfg)), "dtz_start_mm": round(m["dtz_start"] * 1000, 2),
            "dtz_end_mm": round(m["dtz_end"] * 1000, 2), "terminal_coin_speed": round(m["terminal_coin_speed"], 5),
            "peak_qdot": round(m["peak_qdot"], 5), "peak_coin_speed": round(m["peak_coin_speed"], 5),
            "release_step": int(m["release_step"]), "n_corrections": len(best["controller"].corrections),
            "corrections": best["controller"].corrections, "responses": best["controller"].responses,
            "base_intent": [round(float(x), 5) for x in base_intent.to_vector()],
            "final_intent": [round(float(x), 5) for x in best["controller"].cur_intent.to_vector()],
            "was_swapped": bool(best["controller"].was_swapped), "authority": best["controller"].authority}
