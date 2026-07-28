"""R3-B — frontier curriculum + stall-aware champion + kinetic-velocity envelope for the state-dependent residual.

R3-A regressed into a clamp-stall (held contact closer by STOPPING the coin). R3-B keeps the same residual/action-basis and
targets the real skill — *hold contact while the coin keeps kinetic forward progress* — by three focused changes, none of which
is a new model or architecture:

  1. STALL-AWARE CHAMPION (`champion_key3`): safety ≻ clamp=0 ≻ stall=0 ≻ reversal=0 ≻ K6 ≻ close-moving-release ≻ min_dtz ≻
     contact-exit dtz ≻ smoothness. A closer-but-stalled policy can no longer beat a farther-but-cleanly-moving one.
  2. KINETIC-VELOCITY ENVELOPE (`reward3`): a loose, distance-dependent v_par floor (a reward REFERENCE, not a hard guard, read
     off the teacher trace) so the RL cannot win by "slow gripping" — the positive reward is still only RETAINED progress
     (contact ∧ light Fn ∧ v_par>0 ∧ Δdtz), never contact alone.
  3. FRONTIER CURRICULUM: 50 % KINETIC-entry episodes + 50 % episodes started from LEGAL frontier snapshots (48–58 mm, +v_par,
     light contact, no stall) branched from real R2-champion rollouts (env + frozen clone hidden + prev_tau + prev residual —
     NO state edit), putting RL credit where the coin is currently lost.

Reuses the tested `kinetic_rl2` update loop via `train_perstep(collect_override=…, champion_override=…)`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import ACT_DIM, CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import (
    AUG_DIM, KineticTemporalResidualController, deterministic_residual)
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.env.motion_contract import govern_torque
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm

ENVELOPE_W = 6.0                                # weight on the velocity-floor shaping penalty


def envelope_floor(dtz_mm: float) -> float:
    """A loose distance-dependent v_par floor from the teacher trace (~50 mm: 0.18, ~30 mm: 0.12, ~20 mm: 0.08). A reward
    REFERENCE only — the RL is discouraged from letting v_par fall below it (no slow-gripping), never hard-clamped to it."""
    if dtz_mm >= 50.0:
        return 0.18
    if dtz_mm >= 30.0:
        return 0.12 + (dtz_mm - 30.0) / 20.0 * 0.06
    return max(0.08, 0.08 + (dtz_mm - 20.0) / 10.0 * 0.04)


def reward3(kin: list[dict], entry_dtz: float, min_dtz: float, k6: bool, safe: bool,
            w: krl2.Reward2Weights = krl2.Reward2Weights(), envelope_w: float = ENVELOPE_W) -> "tuple[list[float], dict]":
    """`reward2` (retained-progress + separate stall/clamp/reversal/contact-loss penalties) PLUS the kinetic-velocity envelope
    penalty. # Postconditions: reward is still a pure function of physical signals; the positive term rewards only light-contact
    forward progress (never contact duration or raw v_par magnitude)."""
    rewards, decomp = krl2.reward2(kin, entry_dtz, min_dtz, k6, safe, w)
    env_total = 0.0
    for i, s in enumerate(kin):
        if min(s["fn_l"], s["fn_r"]) > 0.05:                          # contact present ⇒ the envelope applies
            pen = envelope_w * max(0.0, envelope_floor(s["dtz_mm"]) - s["v_par"])
            rewards[i] -= pen
            env_total += pen
    decomp["envelope"] = round(env_total, 2)
    return rewards, decomp


def champion_key3(min_dtz: float, exit_dtz: float, k6: bool, safe: bool, released: bool, jerk: float,
                  has_clamp: bool, has_stall: bool, has_reversal: bool) -> tuple:
    """Stall-aware lexicographic champion (higher is better). Cleanliness (no clamp/stall/reversal) ranks ABOVE contact-exit
    dtz, so a closer-but-stalled policy can never beat a cleanly-moving one; min_dtz ranks above exit_dtz."""
    return (int(safe), int(not has_clamp), int(not has_stall), int(not has_reversal), int(k6), int(released),
            -round(min_dtz, 2), -round(exit_dtz, 2), -round(jerk, 4))


# --------------------------------------------------------------------------------------------------------------------
# Frontier snapshots — legal mid-transport restart points branched from a real R2-champion rollout (no state edit)
# --------------------------------------------------------------------------------------------------------------------
# A frontier is only a valid curriculum state if it is an INTERIOR of the KINETIC mode — not merely a frame where contact is
# numerically present. Aliasing at the mode boundary (FRONTIER_RESTART_FAILURE_DUE_TO_HYBRID_GUARD_BOUNDARY_ALIASING) means the
# continuous state alone does not identify the control situation; the discrete mode, guard margin, and restorable causal state
# (clone hidden, previous residual, prev_tau, phase counter) must all be captured, and the restart must be behaviour-invariant.
FN_MARGIN = 0.12                                # N — contact confidence above this = real contact, not numerical residue
N_ENTRY_MARGIN = 2                             # frames since KINETIC entry (past the entry transient)
N_EXIT_MARGIN = 2                              # min restart KINETIC steps (frames before the predicted mode exit)


@dataclass
class FrontierSnapshot:
    """A legal, restart-ADMISSIBLE segment-local KINETIC restart point: the PRE-step env s_t + the frozen clone GRU hidden going
    INTO step t (h_{t-1}) + the applied torque + the previous residual. Duck-types the snapshot interface `velocity_rollout`
    reads. # Invariants: reachable by legal control (branched, never state-edited); a KINETIC-interior state, not a mode-boundary
    frame (verified by the restart-invariance probe)."""

    _rl: Any
    prev_tau: np.ndarray
    stack: Any
    lo: np.ndarray
    hi: np.ndarray
    clone_hidden: Any
    prev_res: np.ndarray
    dtz_mm: float
    v_par: float = 0.0
    guard_margin_fn: float = 0.0
    frames_since_entry: int = 0
    restart_steps: int = 0
    step_index: int = 0                             # the step t in the source rollout (s_t = the PRE-step env captured here)
    q_hold: Any = None

    def branch(self) -> Any:
        return copy.deepcopy(self._rl)

    def start_state(self) -> dict:
        return {"clone_hidden": self.clone_hidden, "prev_res": self.prev_res}


def _restart_kinetic_steps(fs: FrontierSnapshot, model: Any, norm: Any, bounds: ResidualBounds,
                           residual_fn: Callable[[np.ndarray], np.ndarray], cfg: Any = DELIVERY_CFG) -> int:
    """Restart-invariance probe: restart from ``fs`` with the REACHING policy (the champion that produced the frontier) and count
    the KINETIC steps. This measures the distance from the mode exit UNDER THE POLICY that will train there — a mode-boundary-
    aliased frontier yields ~0-1 steps (the guard re-fires almost immediately); a KINETIC-interior one continues several steps.
    (Probing with a zero residual would reject exactly the residual-held states the curriculum wants, since the clone alone
    cannot hold contact past its own ~61 mm limit.)"""
    ctrl = KineticTemporalResidualController(fs, CloneActor(model, norm), residual_fn, bounds, start_kinetic=fs.start_state())
    velocity_rollout(fs, ctrl, cfg)
    return sum(1 for r in ctrl.clone_trace if r["kind"] == "KINETIC_CLONE")


def capture_frontiers(snap: Any, clone: CloneActor, actor: Any, bounds: ResidualBounds, *, dtz_lo: float = 51.0,
                      dtz_hi: float = 58.0, cfg: Any = DELIVERY_CFG) -> "tuple[list[FrontierSnapshot], list[FrontierSnapshot]]":
    """Run the (deterministic) champion and capture healthy-frontier candidates by the HYBRID-INTERIORITY predicate — KINETIC
    mode, +v_par above the envelope floor, light non-degenerate contact (Fn ≥ margin), no stall, no sign reversal, ≥ N_ENTRY_
    MARGIN frames since KINETIC entry — capturing the PRE-step causal state (env s_t + clone hidden h_{t-1} + prev residual +
    prev_tau), then keeping only those whose zero-residual restart is invariant (≥ N_EXIT_MARGIN KINETIC steps). Mirrors the
    frozen `velocity_rollout` step kernel (to expose the mid-loop prev_tau). # Postconditions: returns (healthy, boundary) —
    legal branches only, no state edit; ``boundary`` = admissible-looking but restart-aliased (rejected)."""
    controller = KineticTemporalResidualController(snap, clone, deterministic_residual(actor), bounds)
    controller.reset()
    rl = snap.branch()
    prev_tau = np.asarray(snap.prev_tau, np.float64).copy()
    cand: list[FrontierSnapshot] = []
    prev_vpar = None

    def _gcb(_mo: Any, dt: Any) -> None:
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)
    mujoco.set_mjcb_control(_gcb)
    try:
        for t in range(1, cfg.horizon + 1):
            pre_rl = copy.deepcopy(rl)                                    # s_t (BEFORE the step — matches the logged dtz)
            pre_h = controller.actor.hidden_tensor()                     # h_{t-1} (going INTO step t)
            pre_res = controller._prev_res.copy()
            pre_tau = prev_tau.copy()
            kin_before = controller._kinetic_steps
            dtau = controller.dtau_for_step(rl, t, prev_tau)
            prev_tau = np.clip(prev_tau + dtau, snap.lo, snap.hi)
            step_ablation(rl, np.asarray(prev_tau, np.float32), "A")
            last = controller.clone_trace[-1] if controller.clone_trace else None
            if last is None or last["kind"] != "KINETIC_CLONE":
                continue
            fn, vpar, dtz = min(last["fn_l"], last["fn_r"]), last["v_par"], last["dtz_mm"]
            no_reversal = prev_vpar is None or vpar * prev_vpar >= 0
            if (dtz_lo <= dtz <= dtz_hi and vpar > envelope_floor(dtz) and FN_MARGIN <= fn < krl2.FN_LIGHT
                    and vpar > 0.0 and no_reversal and kin_before >= N_ENTRY_MARGIN):
                cand.append(FrontierSnapshot(pre_rl, pre_tau, snap.stack, snap.lo, snap.hi, pre_h, pre_res,
                                             round(dtz, 2), round(vpar, 4), round(fn, 3), int(kin_before), step_index=t))
            prev_vpar = vpar
    finally:
        mujoco.set_mjcb_control(None)
    healthy, boundary = [], []
    probe_fn = deterministic_residual(actor)                          # restart under the REACHING policy (the champion)
    for fs in cand:
        fs.restart_steps = _restart_kinetic_steps(fs, clone.model, clone.norm, bounds, probe_fn)
        (healthy if fs.restart_steps >= N_EXIT_MARGIN else boundary).append(fs)
    return healthy, boundary


# --------------------------------------------------------------------------------------------------------------------
# Curriculum collection + champion (injected into the tested train_perstep loop)
# --------------------------------------------------------------------------------------------------------------------
def collect_episode3(start: Any, clone: CloneActor, residual_fn: Callable[[np.ndarray], np.ndarray], bounds: ResidualBounds,
                     w: krl2.Reward2Weights, envelope_w: float, *, frontier: bool, cfg: Any = DELIVERY_CFG,
                     make_controller: "Callable | None" = None) -> krl2.Episode:
    """One episode from either the KINETIC entry (full chain) or a legal frontier snapshot (segment-local restart), scored by
    `reward3`. ``make_controller(start, clone, residual_fn, bounds, **kw)`` swaps the controller (default = the R2 temporal
    residual; R3-C injects the action-preserving authority-expansion controller) without duplicating the reward/transition code.
    # Postconditions: per-step transitions terminal at the KINETIC exit; no state edit / teacher."""
    kw = {"start_kinetic": start.start_state()} if frontier else {}
    build = make_controller if make_controller is not None else KineticTemporalResidualController
    controller = build(start, clone, residual_fn, bounds, **kw)
    m = velocity_rollout(start, controller, cfg)
    kin = [r for r in controller.clone_trace if r["kind"] == "KINETIC_CLONE"]
    aug = controller.aug_trace
    entry_dtz = kin[0]["dtz_mm"] if kin else 999.0
    min_dtz = _min_dtz_mm(start, m)
    safe = bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)
    rewards, decomp = reward3(kin, entry_dtz, min_dtz, bool(m["k6_delivered"]), safe, w, envelope_w)
    trans = []
    for t in range(len(aug)):
        s, a = aug[t]
        s2 = aug[t + 1][0] if t + 1 < len(aug) else np.zeros(AUG_DIM, np.float64)
        trans.append((s.astype(np.float32), a.astype(np.float32), float(rewards[t]) / krl2.REWARD_SCALE,
                      s2.astype(np.float32), 1.0 if t == len(aug) - 1 else 0.0))
    return krl2.Episode(trans, min_dtz, decomp["exit_dtz_mm"], bool(m["k6_delivered"]), safe,
                        np.array([r for _s, r in aug], np.float64) if aug else np.zeros((0, ACT_DIM)), decomp)


def make_collect3(entry_snap: Any, frontiers: list[FrontierSnapshot], clone_factory: Callable[[], CloneActor],
                  bounds: ResidualBounds, w: krl2.Reward2Weights, *, frac_frontier: float = 0.5, envelope_w: float = ENVELOPE_W,
                  seed: int = 0, make_controller: "Callable | None" = None) -> Callable:
    """A `collect_override(residual_fn) -> transitions` drawing (1−frac)·KINETIC-entry / frac·frontier episodes (deterministic).
    ``make_controller`` swaps the controller (R3-C authority expansion); default = the R2 temporal residual."""
    rng = np.random.default_rng(seed)

    def collect(residual_fn: Callable[[np.ndarray], np.ndarray]) -> list:
        use_frontier = bool(frontiers) and rng.random() < frac_frontier
        start = frontiers[int(rng.integers(0, len(frontiers)))] if use_frontier else entry_snap
        return collect_episode3(start, clone_factory(), residual_fn, bounds, w, envelope_w, frontier=use_frontier,
                                make_controller=make_controller).transitions
    return collect


def make_dev_eval3(entry_snap: Any, clone_factory: Callable[[], CloneActor], bounds: ResidualBounds,
                   w: krl2.Reward2Weights, *, envelope_w: float = ENVELOPE_W) -> Callable:
    """A `champion_override(actor) -> (key, aux)` that evaluates on the full frozen chain and ranks by the stall-aware champion."""
    def ev(actor: Any) -> "tuple[tuple, dict]":
        ep = collect_episode3(entry_snap, clone_factory(), deterministic_residual(actor), bounds, w, envelope_w, frontier=False)
        jerk = float(np.mean(np.abs(np.diff(ep.residuals, axis=0)))) if len(ep.residuals) > 1 else 0.0
        d = ep.decomp
        key = champion_key3(ep.min_dtz, ep.exit_dtz, ep.k6, ep.safe, bool(d["released"]), jerk,
                            d["clamp"] > 0, d["neg"] > 0, d["reversal"] > 0)
        return key, {"min_dtz": round(ep.min_dtz, 2), "exit_dtz": round(ep.exit_dtz, 2), "k6": ep.k6,
                     "released": bool(d["released"]), "stall_pen": d["neg"], "clamp_pen": d["clamp"], "reversal_pen": d["reversal"]}
    return ev
