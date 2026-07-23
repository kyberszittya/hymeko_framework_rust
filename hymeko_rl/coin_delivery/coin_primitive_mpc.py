"""CONTACT_STABILIZED_PRIMITIVE_MPC_V1 — a low-dimensional (10-D) push→brake→settle primitive on top of the FROZEN pi_0
low-level feedback controller, with an IMMEDIATE contact-preserving execution guard (no training).

Unlike the raw-action H=30 planner (which optimised a 120-D open-loop-ish action sequence and whose lookahead feasibility
did not govern the executed path), this baseline:
  * keeps pi_0 immutable as the per-timestep feedback law (a0_t = pi_0(obs_t));
  * parameterises the improvement as ONE 10-D primitive θ = {delta_push[4], delta_brake[4], brake_distance, settle_speed};
  * runs a closed-loop 3-mode controller (PUSH/BRAKE/SETTLE) whose mode is recomputed from LIVE state every low-level step;
  * before executing any non-pi_0 action, evaluates its IMMEDIATE (guard_horizon-step) effect vs a pi_0 reference and
    line-searches α∈{1,.75,.5,.25,0} toward pi_0, so the executed action never loses contact pi_0 would have kept;
  * optimises θ (not raw actions) by bounded CEM, re-planned every 4 low-level steps (receding horizon on θ).

The SAME controller (mode detection + pi_0 feedback + primitive offset + execution guard) is used in candidate simulation
and in real execution. All simulation uses deterministic snapshot/restore and never alters the real rollout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_planner_repair import FeasibilityConfig, classify_feasibility
from hymeko_rl.coin_delivery.coin_rl_env import HELD_DWELL, SETTLE_VEL

ENTRY_TOL = 0.05
ACTION_SCALE = 4.0
MODES = ("PUSH", "BRAKE", "SETTLE")
ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.0)
THETA_DIM = 10


# ── frozen conservative bounds + θ (un)packing ──
@dataclass(frozen=True)
class PrimitiveBounds:
    lo: tuple = (-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 0.02, 0.02)
    hi: tuple = (1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.12, 0.12)

    @property
    def lo_a(self):
        return np.asarray(self.lo, np.float32)

    @property
    def hi_a(self):
        return np.asarray(self.hi, np.float32)


NEUTRAL_THETA = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.05, SETTLE_VEL], np.float32)   # deltas 0 ⇒ pi_0-equivalent


@dataclass(frozen=True)
class Theta:
    delta_push: np.ndarray
    delta_brake: np.ndarray
    brake_distance: float
    settle_speed: float


def unpack_theta(v) -> Theta:
    v = np.asarray(v, np.float32)
    assert v.shape == (THETA_DIM,), f"theta must be {THETA_DIM}-D"
    return Theta(v[0:4].copy(), v[4:8].copy(), float(v[8]), float(v[9]))


def pack_theta(th: Theta) -> np.ndarray:
    return np.concatenate([th.delta_push, th.delta_brake, [th.brake_distance, th.settle_speed]]).astype(np.float32)


# ── deterministic state restoration (injectable seam so the guard/controller are unit-testable without MuJoCo) ──
def _default_snap(rl):
    from hymeko_rl.env.planar_snapshot import snapshot_planar
    return (snapshot_planar(rl.inner), rl._t, rl._strict, rl._touched)


def _default_restore(rl, s):
    from hymeko_rl.env.planar_snapshot import restore_planar
    restore_planar(rl.inner, s[0]); rl._t, rl._strict, rl._touched = s[1], s[2], s[3]


def _pi0_action(pi0, obs):
    with torch.no_grad():
        return np.clip(pi0.action_mean(torch.as_tensor(np.asarray(obs, np.float32)[None]))[0].numpy(),
                       -ACTION_SCALE, ACTION_SCALE).astype(np.float32)


def _contact(rl) -> bool:
    m = rl.inner._planar_metrics
    return bool(m.left_contact or m.right_contact)


# ── mode detection (LIVE state, never elapsed time) ──
def detect_mode(rl, th: Theta, mode: str, *, proj_steps: int = 3) -> str:
    dtz = rl._dtz(); v = rl._speed(); inside = dtz <= ENTRY_TOL
    if mode == "SETTLE":
        return "BRAKE" if dtz > ENTRY_TOL else "SETTLE"          # coin left the target → re-brake
    if mode == "BRAKE":
        return "SETTLE" if (inside and v <= th.settle_speed) else "BRAKE"
    overshoot = v * proj_steps > dtz                             # projected motion crosses the target
    return "BRAKE" if (dtz <= th.brake_distance or overshoot) else "PUSH"


def mode_offset(mode: str, th: Theta) -> np.ndarray:
    if mode == "PUSH":
        return np.asarray(th.delta_push, np.float32)
    if mode == "BRAKE":
        return np.asarray(th.delta_brake, np.float32)
    return np.zeros(4, np.float32)                               # SETTLE ⇒ pi_0 exactly


# ── immediate contact-preserving execution guard ──
def _guard_probe(rl, actions, pi0, snap, restore):
    """From the exact current state run ``actions`` (None ⇒ pi_0); return (contact_at_end, exited_after_entry).
    Deterministic: snapshots on entry, restores on exit — the real rollout is unchanged."""
    s = snap(rl)
    entered = rl._dtz() <= ENTRY_TOL; exited = False; kept = True     # kept = contact held at EVERY step in the window
    for a in actions:
        act = _pi0_action(pi0, rl.obs()) if a is None else a
        rl.step(act); kept = kept and _contact(rl); dtz = rl._dtz()
        if dtz <= ENTRY_TOL:
            entered = True
        elif entered:
            exited = True
    restore(rl, s)
    return kept, exited


def execution_guard(rl, offset, pi0, *, horizon=2, alphas=ALPHAS, snap=_default_snap, restore=_default_restore):
    """Return (executed_action, alpha, fallback). pi_0 is the safety reference: for the largest α with
    ``pi_0 + α·offset`` NOT causing a required-contact loss pi_0 preserves (nor an illegal exit pi_0 avoids), execute it;
    else fall back to pi_0 exactly (α=0) and flag ``fallback``. Release after certified placement is legal (no gate)."""
    a0 = _pi0_action(pi0, rl.obs())
    if float(np.abs(offset).max()) < 1e-9:
        return a0, 0.0, False                                   # SETTLE / no offset ⇒ pi_0, not an intervention
    if rl._strict >= HELD_DWELL and rl._touched:                # certified stable placement ⇒ release legal
        return np.clip(a0 + offset, -ACTION_SCALE, ACTION_SCALE).astype(np.float32), 1.0, False
    kept_pi0, exit_pi0 = _guard_probe(rl, [None] * horizon, pi0, snap, restore)
    for alpha in alphas:
        if alpha == 0.0:
            break
        cand = np.clip(a0 + alpha * np.asarray(offset, np.float32), -ACTION_SCALE, ACTION_SCALE).astype(np.float32)
        kept_c, exit_c = _guard_probe(rl, [cand] + [None] * (horizon - 1), pi0, snap, restore)
        loses_contact = kept_pi0 and not kept_c
        illegal_exit = exit_c and not exit_pi0
        if not (loses_contact or illegal_exit):
            return cand, float(alpha), False
    return a0, 0.0, True                                        # no nonzero α admissible ⇒ PRIMITIVE_GUARD_FALLBACK


class PrimitiveController:
    """Closed-loop controller: LIVE mode detection + pi_0 feedback + primitive offset + immediate execution guard. The
    SAME instance-logic is used in candidate simulation and in real execution (contract step 5/8)."""

    def __init__(self, mode="PUSH", *, guard_horizon=2, proj_steps=3, snap=_default_snap, restore=_default_restore):
        assert mode in MODES
        self.mode = mode; self.guard_horizon = guard_horizon; self.proj_steps = proj_steps
        self.snap = snap; self.restore = restore

    def step_action(self, rl, th: Theta, pi0):
        self.mode = detect_mode(rl, th, self.mode, proj_steps=self.proj_steps)
        offset = mode_offset(self.mode, th)
        a, alpha, fb = execution_guard(rl, offset, pi0, horizon=self.guard_horizon, snap=self.snap, restore=self.restore)
        return a, {"mode": self.mode, "alpha": float(alpha), "fallback": bool(fb),
                   "offset_nonzero": bool(np.abs(offset).max() > 1e-9)}


# ── candidate simulation of the FULL closed-loop primitive (θ scoring) ──
def _task_fields(contact, dtz, speed, rewards, controlled_entry, entry_speed):
    n = len(dtz); dtz = np.asarray(dtz); speed = np.asarray(speed)
    near = dtz < 0.06
    braking = (float(speed[near].max()) if near.any() else 0.0) - float(speed[-1]) if n else 0.0
    excess = max(0.0, (entry_speed - controlled_entry)) if entry_speed is not None else 0.0
    disc = 0.99 ** np.arange(n)
    return {"min_dtz": float(dtz.min()) if n else 9.0, "braking": round(braking, 4), "excess_entry_speed": round(excess, 4),
            "total_return": round(float((disc * np.asarray(rewards)).sum()), 4) if n else 0.0}


def simulate_primitive(rl, th: Theta, pi0, *, horizon=60, guard_horizon=2, proj_steps=3, start_mode="PUSH",
                       cfg: FeasibilityConfig = FeasibilityConfig(boundary="stable_entry", contact_floor=0.5),
                       snap=_default_snap, restore=_default_restore):
    """Simulate the complete closed-loop primitive from the CURRENT state on a snapshot; classify feasibility + task +
    controller-occupancy fields. Restores the env on exit (never alters the real rollout)."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance
    s = snap(rl); carry = bool(rl._touched)
    ctrl = PrimitiveController(mode=start_mode, guard_horizon=guard_horizon, proj_steps=proj_steps, snap=snap, restore=restore)
    cert = DeliveryCertifier(initial_clearance=_clearance(rl.inner)); cert.robot_touched = carry; cert.delivery_dwell = int(rl._strict)
    contact, dtz, speed, dwell, certified, rewards = [], [], [], [], [], []
    modes = {"PUSH": 0, "BRAKE": 0, "SETTLE": 0}; n_off = n_alpha_pos = n_fb = eff_n = 0; eff = 0.0
    for _t in range(horizon):
        cert.update(_cert_step(rl.inner, rl.cf))
        m = rl.inner._planar_metrics
        contact.append(bool(m.left_contact or m.right_contact)); dtz.append(rl._dtz()); speed.append(rl._speed())
        dwell.append(int(cert.delivery_dwell)); certified.append(bool(cert.delivery_certified))
        if cert.delivery_certified:
            break
        a, diag = ctrl.step_action(rl, th, pi0)
        modes[diag["mode"]] += 1; n_off += int(diag["offset_nonzero"]); n_alpha_pos += int(diag["alpha"] > 0)
        n_fb += int(diag["fallback"]); eff += float(np.abs(a).mean()); eff_n += 1
        _o, r, term, trunc, _ = rl.step(a); rewards.append(float(r))
        if term or trunc or not np.all(np.isfinite(rl.inner.data.qvel)):
            break
    restore(rl, s)
    feas = classify_feasibility(contact, dtz, speed, dwell, certified, cfg=cfg, carry_touched=carry)
    tf = _task_fields(contact, dtz, speed, rewards, cfg.controlled_entry, feas["entry_speed"])
    tot = max(sum(modes.values()), 1)
    return {**feas, **tf, "any_strict": bool(np.any(certified)), "max_dwell": int(max(dwell, default=int(rl._strict))),
            "effort": round(eff / max(eff_n, 1), 4), "mode_occupancy": {k: round(v / tot, 3) for k, v in modes.items()},
            "intervention_rate": round(n_alpha_pos / max(eff_n, 1), 4),
            "fallback_rate": round(n_fb / max(n_off, 1), 4), "n_offset_steps": n_off}


def primitive_key(r):
    """Feasibility-gated lexicographic task rank (step 7): strict → dwell → progress(−min_dtz) → controlled entry →
    braking → minimum primitive intervention → minimum effort."""
    return (int(r["feasible"]), -int(r["n_violations"]), int(r["any_strict"]), int(r["max_dwell"]),
            -float(r["min_dtz"]), -float(r["excess_entry_speed"]), float(r["braking"]),
            -float(r["intervention_rate"]), -float(r["effort"]))


def plan_theta(rl, seed, bounds: PrimitiveBounds, pi0, *, pop=40, iters=6, elite=8, horizon=60, guard_horizon=2,
               proj_steps=3, start_mode="PUSH"):
    """Bounded CEM over the 10-D primitive θ (NOT raw actions). Returns (best_theta[10], best_result, all_fallback)."""
    lo, hi = bounds.lo_a, bounds.hi_a
    mean = NEUTRAL_THETA.copy(); sigma = (hi - lo) * 0.25
    rng = np.random.default_rng(seed); best_key = None; best_theta = mean.copy(); best_res = None
    for _it in range(iters):
        cand = np.clip(rng.normal(mean, sigma, size=(pop, THETA_DIM)), lo, hi).astype(np.float32); cand[0] = mean
        scored = [simulate_primitive(rl, unpack_theta(c), pi0, horizon=horizon, guard_horizon=guard_horizon,
                                     proj_steps=proj_steps, start_mode=start_mode) for c in cand]
        keys = [primitive_key(s) for s in scored]
        order = sorted(range(pop), key=lambda i: keys[i])
        el = np.stack([cand[i] for i in order[-elite:]]); mean, sigma = el.mean(0), el.std(0) + (hi - lo) * 0.02
        bi = order[-1]
        if best_key is None or keys[bi] > best_key:
            best_key, best_theta, best_res = keys[bi], cand[bi].copy(), scored[bi]
    return best_theta, best_res, bool(best_res["fallback_rate"] > 0.95 and best_res["n_offset_steps"] > 0)


class PrimitiveMPCPolicy:
    """Receding-horizon primitive MPC as a ``rollout_from_handoff`` policy: re-optimise θ every ``replan_every`` low-level
    steps; each step recompute the low-level action feedback-wise via the closed-loop controller. Stores θ (10-D), never a
    raw action sequence (contract step 10). Accumulates mode occupancy + guard intervention/fallback + θ samples."""

    def __init__(self, bounds: PrimitiveBounds, pi0, *, plan_seed_base=0, replan_every=4, pop=40, iters=6, elite=8,
                 horizon=60, guard_horizon=2, proj_steps=3):
        self.bounds = bounds; self.pi0 = pi0; self.plan_seed_base = plan_seed_base; self.replan_every = replan_every
        self.pop, self.iters, self.elite, self.sim_horizon = pop, iters, elite, horizon
        self.ctrl = PrimitiveController(mode="PUSH", guard_horizon=guard_horizon, proj_steps=proj_steps)
        self.theta = None; self.since = replan_every
        self.mode_counts = {"PUSH": 0, "BRAKE": 0, "SETTLE": 0}
        self.n = self.n_alpha_pos = self.n_off = self.n_fb = 0
        self.n_plans = self.n_all_fallback = 0; self.thetas: list = []

    def __call__(self, rl, prev_action, step):
        if self.theta is None or self.since >= self.replan_every:
            th, _res, allfb = plan_theta(rl, self.plan_seed_base + step, self.bounds, self.pi0, pop=self.pop,
                                         iters=self.iters, elite=self.elite, horizon=self.sim_horizon,
                                         guard_horizon=self.ctrl.guard_horizon, proj_steps=self.ctrl.proj_steps,
                                         start_mode=self.ctrl.mode)
            self.theta = th; self.since = 0; self.thetas.append(th.tolist()); self.n_plans += 1; self.n_all_fallback += int(allfb)
        a, diag = self.ctrl.step_action(rl, unpack_theta(self.theta), self.pi0)
        self.mode_counts[diag["mode"]] += 1; self.n += 1; self.since += 1
        self.n_off += int(diag["offset_nonzero"]); self.n_alpha_pos += int(diag["alpha"] > 0); self.n_fb += int(diag["fallback"])
        return a

    def stats(self):
        tot = max(self.n, 1); th = np.asarray(self.thetas, np.float32) if self.thetas else np.zeros((1, THETA_DIM), np.float32)
        return {"mode_occupancy": {k: round(v / tot, 3) for k, v in self.mode_counts.items()},
                "guard_intervention_rate": round(self.n_alpha_pos / tot, 4),
                "guard_fallback_rate": round(self.n_fb / max(self.n_off, 1), 4),
                "all_fallback_plan_rate": round(self.n_all_fallback / max(self.n_plans, 1), 4),
                "theta_mean": th.mean(0).round(4).tolist(), "theta_std": th.std(0).round(4).tolist(), "n_plans": self.n_plans}


def first_primitive_stability(rl, bounds: PrimitiveBounds, pi0, *, n_seeds=6, **plan_kw):
    """θ agreement across repeated CEM seeds from the SAME handoff (plan_theta restores internally). Reports normalised
    dispersion of the selected θ (0 = identical)."""
    lo, hi = bounds.lo_a, bounds.hi_a
    thetas = np.stack([plan_theta(rl, sd, bounds, pi0, **plan_kw)[0] for sd in range(n_seeds)])
    return {"theta_std_norm": round(float((thetas.std(0) / (hi - lo)).mean()), 4),
            "theta_mean": thetas.mean(0).round(4).tolist()}
