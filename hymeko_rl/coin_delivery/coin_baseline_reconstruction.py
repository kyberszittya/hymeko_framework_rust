"""COIN_FEEDBACK_BASELINE_RECONSTRUCTION_V1 — one canonical evaluation harness for four gold controllers + the
load-bearing FULL-HORIZON H=30 teacher qualification audit (no training).

Controllers:
  A. recovered frozen E-approach            (neutral → approach/grasp only)                 — approach competence
  B. frozen pi_0                            (from a reconstructed dev late-start handoff)    — deployed late controller
  C. proven H=30 receding-horizon planner   (from the SAME handoff, replan every step, M=1) — the improvement teacher
  D. composed E-approach → grasp → H=30      (neutral → full episode)                        — read from the H=30 pilot

Every controller is graded by the SAME deployed strict-K=6 certificate (``CoinRL4Dof._strict >= 6 ∧ touched``, the
canonical v3 reward env) over the SAME 11 metrics. B and C are the apples-to-apples pair for the teacher qualification.

Phase-conditional contact legality (step 6): robot-attributed contact is REQUIRED only until stable target entry
(``dtz <= center_tol ∧ settled``); contact loss AFTER stable placement is LEGAL (push-and-coast delivery). The
qualification therefore judges required-window contact, not blanket full-horizon persistence.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.coin_late_start import reconstruct_handoff
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals

ENTRY_TOL = 0.05
NEAR_ZONE = 0.06
GAMMA = 0.99
CONTACT_TOL = 0.05                 # allowed drop in required-window contact retention before it counts as a loss
EXIT_TOL = 0.05                    # allowed increase in target-exit rate
CANONICAL_H30 = {"horizon": 30, "pop": 40, "iters": 6, "elite": 8}   # accepted-canonical proven config (SHA 1115ade3…)
ELEVEN_METRICS = ("first_contact", "bilateral_grasp", "required_contact_retention", "first_contact_loss_step",
                  "target_entry", "entry_velocity", "target_exit", "braking", "max_dwell", "strict_success",
                  "total_return")


class RolloutTrace:
    """Streams per-step live-env readings and emits the 11 canonical metrics + phase-conditional contact fields.

    Preconditions: ``record`` is called once per executed env step with the post-step live ``CoinRL4Dof`` and reward.
    Postcondition: :meth:`metrics` returns a dict with all of :data:`ELEVEN_METRICS` + ``lost_required_contact``.
    """

    def __init__(self) -> None:
        self.contact: list[bool] = []; self.bilat: list[bool] = []
        self.dtz: list[float] = []; self.speed: list[float] = []
        self.strict: list[int] = []; self.rewards: list[float] = []
        self.touched_final = False

    def record(self, rl, reward: float) -> None:
        m = rl.inner._planar_metrics
        lc, rc = bool(m.left_contact), bool(m.right_contact)
        self.contact.append(lc or rc); self.bilat.append(lc and rc)
        self.dtz.append(rl._dtz()); self.speed.append(rl._speed())
        self.strict.append(int(rl._strict)); self.rewards.append(float(reward))
        self.touched_final = bool(rl._touched)

    def metrics(self, *, gamma: float = GAMMA) -> dict:
        n = len(self.dtz)
        if n == 0:
            return {k: None for k in ELEVEN_METRICS} | {"lost_required_contact": False, "n_steps": 0}
        contact = np.array(self.contact); bilat = np.array(self.bilat)
        dtz = np.array(self.dtz); speed = np.array(self.speed); strict = np.array(self.strict)
        first_contact = int(np.argmax(contact)) if contact.any() else None
        bilateral_step = int(np.argmax(bilat)) if bilat.any() else None
        entered = dtz <= ENTRY_TOL
        entry_step = int(np.argmax(entered)) if entered.any() else None
        entry_velocity = float(speed[entry_step]) if entry_step is not None else None
        stable = (dtz <= CENTER_TOL) & (speed < SETTLE_VEL)
        stable_entry_step = int(np.argmax(stable)) if stable.any() else None
        exited = bool((dtz[entry_step:] > ENTRY_TOL).any()) if entry_step is not None else False
        req_end = stable_entry_step if stable_entry_step is not None else n     # required-contact window
        required_contact_retention = float(contact[:req_end].mean()) if req_end > 0 else 0.0
        first_loss = None
        if first_contact is not None:
            after = np.where(~contact[first_contact:])[0]
            first_loss = int(first_contact + after[0]) if after.size else None
        lost_required_contact = first_loss is not None and (stable_entry_step is None or first_loss < stable_entry_step)
        near = dtz < NEAR_ZONE
        max_coin_speed_near = float(speed[near].max()) if near.any() else 0.0
        braking = max_coin_speed_near - float(speed[-1])                        # near-zone deceleration (>0 = braked)
        max_dwell = int(strict.max())
        strict_success = bool(max_dwell >= HELD_DWELL and self.touched_final)
        k6 = np.where(strict >= HELD_DWELL)[0]                                  # additive: strict-K6 certification step
        k6_step = int(k6[0]) if k6.size else None
        exit_before_k6 = False                                                 # entered then exited BEFORE K6 (step 2/9)
        if entry_step is not None:
            end = k6_step if k6_step is not None else n
            exit_before_k6 = bool((dtz[entry_step:end] > ENTRY_TOL).any()) if end > entry_step else False
        progress = float(self.dtz[0] - self.dtz[-1])
        disc = gamma ** np.arange(n)
        total_return = float((disc * np.array(self.rewards)).sum())
        return {"first_contact": first_contact is not None, "first_contact_step": first_contact,
                "bilateral_grasp": bilateral_step is not None, "bilateral_step": bilateral_step,
                "required_contact_retention": round(required_contact_retention, 4),
                "first_contact_loss_step": first_loss, "lost_required_contact": bool(lost_required_contact),
                "target_entry": entry_step is not None, "entry_step": entry_step,
                "entry_velocity": round(entry_velocity, 4) if entry_velocity is not None else None,
                "target_exit": exited, "braking": round(braking, 4),
                "max_coin_speed_near": round(max_coin_speed_near, 4),
                "max_dwell": max_dwell, "strict_success": strict_success,
                "progress": round(progress, 4), "total_return": round(total_return, 4),
                "stable_entry_step": stable_entry_step, "k6_step": k6_step, "exit_before_k6": bool(exit_before_k6),
                "n_steps": n}


# ── controllers as policy callables (rl, prev_action, step) → action(4) ──
def pi0_policy(pi0):
    import torch

    def fn(rl, prev_action, step):
        with torch.no_grad():
            return np.clip(pi0.action_mean(torch.as_tensor(rl.obs()[None], dtype=torch.float32))[0].numpy(),
                           -4.0, 4.0).astype(np.float32)
    return fn


def planner_policy(*, plan_seed_base=0, horizon=30, pop=40, iters=6, elite=8):
    """H=30 receding-horizon feedback planner: replan the CEM every step from the live state, execute the first action.
    Reuses ``plan_first_action`` (leaves the env unchanged internally). Carry = deployed rl._touched / rl._strict."""
    from hymeko_rl.coin_delivery.coin_v3_receding_horizon import plan_first_action
    from hymeko_rl.experiments.coin_neutral_start import _clearance

    def fn(rl, prev_action, step):
        a, _res = plan_first_action(rl.inner, rl.cf, _clearance(rl.inner), bool(rl._touched), int(rl._strict),
                                    plan_seed_base + step, horizon=horizon, pop=pop, iters=iters, elite=elite)
        return np.asarray(a, np.float32)
    return fn


def rollout_from_handoff(pi0, ls, policy_fn, *, horizon=60):
    """Reconstruct the dev late-start handoff (live env) and roll ``policy_fn`` for ``horizon`` steps under the deployed
    strict-K=6 grading. Returns the 11-metric dict. Deterministic per (ls, policy)."""
    rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
    tr = RolloutTrace(); pa = np.asarray(rec.base, np.float32)
    for step in range(horizon):
        a = policy_fn(rl, pa, step)
        _o, r, term, trunc, _ = rl.step(a)
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        tr.record(rl, r); pa = a
        if term or trunc:
            break
    return tr.metrics()


# ── step-5 teacher qualification ──
def qualify_teacher(pi0_rows, planner_rows):
    """Full-rollout qualification (step 5, phase-conditional contact per step 6). The planner qualifies only if, in
    aggregate over disjoint dev states, it (1) does not lose required (pre-stable-entry) robot contact pi_0 preserved,
    (2) does not materially increase target exit, and (3) improves ≥1 of {progress, braking, dwell, strict, return}."""
    assert len(pi0_rows) == len(planner_rows) and pi0_rows, "aligned non-empty rows required"
    per = []; new_losses = 0; new_exits = 0
    for b, c in zip(pi0_rows, planner_rows):
        contact_ok = ((not c["lost_required_contact"]) or b["lost_required_contact"]) and \
                     (c["required_contact_retention"] >= b["required_contact_retention"] - CONTACT_TOL)
        if c["lost_required_contact"] and not b["lost_required_contact"]:
            new_losses += 1
        exit_ok = (not c["target_exit"]) or b["target_exit"]
        if c["target_exit"] and not b["target_exit"]:
            new_exits += 1
        improved = (c["max_dwell"] > b["max_dwell"] or int(c["strict_success"]) > int(b["strict_success"])
                    or c["total_return"] > b["total_return"] + 1e-6 or c["braking"] > b["braking"] + 1e-3
                    or c["progress"] > b["progress"] + 1e-3)
        per.append({"contact_ok": bool(contact_ok), "exit_ok": bool(exit_ok), "improved": bool(improved),
                    "d_contact": round(c["required_contact_retention"] - b["required_contact_retention"], 4),
                    "d_dwell": c["max_dwell"] - b["max_dwell"], "d_strict": int(c["strict_success"]) - int(b["strict_success"]),
                    "d_return": round(c["total_return"] - b["total_return"], 4)})
    n = len(per)
    mean = lambda k: round(float(np.mean([r[k] for r in per])), 4)
    rate = lambda rows, k: round(float(np.mean([int(r[k]) for r in rows])), 4)
    agg = {"n": n, "new_required_contact_losses": new_losses, "new_target_exits": new_exits,
           "mean_d_contact_retention": mean("d_contact"), "mean_d_dwell": mean("d_dwell"),
           "mean_d_return": mean("d_return"),
           "planner_strict_rate": rate(planner_rows, "strict_success"), "pi0_strict_rate": rate(pi0_rows, "strict_success"),
           "planner_exit_rate": rate(planner_rows, "target_exit"), "pi0_exit_rate": rate(pi0_rows, "target_exit"),
           "planner_req_contact": round(float(np.mean([r["required_contact_retention"] for r in planner_rows])), 4),
           "pi0_req_contact": round(float(np.mean([r["required_contact_retention"] for r in pi0_rows])), 4),
           "n_improved": int(sum(r["improved"] for r in per))}
    contact_clause = new_losses == 0 and agg["mean_d_contact_retention"] >= -CONTACT_TOL
    exit_clause = (agg["planner_exit_rate"] - agg["pi0_exit_rate"]) <= EXIT_TOL
    improve_clause = (agg["planner_strict_rate"] > agg["pi0_strict_rate"] or agg["mean_d_dwell"] > 0
                      or agg["mean_d_return"] > 0)
    reasons = []
    if not contact_clause:
        reasons.append("LOSES_REQUIRED_CONTACT")
    if not exit_clause:
        reasons.append("INCREASES_TARGET_EXIT")
    if not improve_clause:
        reasons.append("NO_IMPROVEMENT")
    qualified = not reasons
    return {"aggregate": agg, "per_state": per, "clauses": {"contact": contact_clause, "exit": exit_clause, "improve": improve_clause},
            "qualified": qualified,
            "verdict": "H30_TEACHER_QUALIFIED" if qualified else "H30_TEACHER_UNQUALIFIED:" + ",".join(reasons)}
