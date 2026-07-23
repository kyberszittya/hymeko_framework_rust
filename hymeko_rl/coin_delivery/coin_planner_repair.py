"""REPAIR_H30_PLANNER_OBJECTIVE_V1 — phase-conditional feasibility constraints gating a lexicographic TASK objective.

The proven H=30 planner (``coin_v3_receding_horizon``) ranks candidates by ``_lexo`` = strict → dwell → −min_dtz →
−min_speed → −effort, which has NO contact/exit term and is therefore free to slam the coin in (measured
H30_TEACHER_UNQUALIFIED: loses required contact, increases exit). This module leaves that proven scorer byte-untouched
and adds a REPAIRED scoring strategy:

  1. Two phase-conditional feasibility CONSTRAINTS (never task terms):
       * ``premature_required_contact_loss`` — robot-attributed contact retention in the pre-boundary window falls below
         a floor (a *material* abandonment before the frozen physical completion boundary). Boundary ∈ {stable_entry, k6}.
         Contact release AFTER the boundary is LEGAL and never penalised.
       * ``illegal_target_exit`` — the coin entered the zone and later exited BEFORE strict-K6 certification.
  2. A candidate is ``feasible`` iff it commits neither violation. If ≥1 feasible candidate exists an infeasible one is
     never selected. If ALL are infeasible the LEAST-VIOLATING is chosen and ``ALL_CANDIDATES_INFEASIBLE`` is flagged.
  3. Among feasible candidates the LEXICOGRAPHIC TASK objective decides (strict success stays high in the tuple, never
     last, so the repair cannot yield a safe-but-non-delivering planner):
         (feasible, any_strict, max_dwell, −min_dtz, −excess_entry_speed, −effort)
  4. Raw contact duration after successful placement is never rewarded (the in-plan rollout stops at K6).

CEM candidate *distribution* (seed, pop, iters, elite, sigma) is identical to ``plan_first_action`` — only the
candidate SELECTION (this scorer) changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.coin_delivery.coin_v3_receding_horizon import ACT_DIM, CTRL_LIM, _restore, _snapshot

ENTRY_TOL = 0.05


@dataclass(frozen=True)
class FeasibilityConfig:
    """Constraint thresholds. ``boundary`` is the frozen physical completion boundary for required contact.

    # Preconditions: ``boundary`` ∈ {"stable_entry", "k6"}; 0 < contact_floor ≤ 1; tolerances > 0.
    """
    boundary: str = "stable_entry"        # A = "stable_entry"; B = "k6" — frozen by measurement before qualification
    contact_floor: float = 0.75           # min pre-boundary contact retention below which the loss is MATERIAL
    center_tol: float = CENTER_TOL
    settle_vel: float = SETTLE_VEL
    entry_tol: float = ENTRY_TOL
    dwell_req: int = HELD_DWELL
    controlled_entry: float = 0.12        # coin entry speed above this is "excess" (2× settle_vel)


def classify_feasibility(contact, dtz, speed, dwell, certified, *, cfg: FeasibilityConfig, carry_touched: bool) -> dict:
    """Pure per-step feasibility classifier over one candidate's simulated horizon arrays (all length H).

    Preconditions: arrays aligned length ≥ 1; ``contact`` bool; ``dtz``/``speed`` ≥ 0. Postcondition: returns the two
    binary violations + all separately-tracked timing fields (step 2 of the contract). No env access.
    """
    n = len(dtz)
    contact = np.asarray(contact, bool); dtz = np.asarray(dtz, float); speed = np.asarray(speed, float)
    certified = np.asarray(certified, bool)
    entered = dtz <= cfg.entry_tol
    first_entry = int(np.argmax(entered)) if entered.any() else None
    stable = (dtz <= cfg.center_tol) & (speed < cfg.settle_vel)
    stable_entry = int(np.argmax(stable)) if stable.any() else None
    k6 = int(np.argmax(certified)) if certified.any() else None
    boundary_step = stable_entry if cfg.boundary == "stable_entry" else k6

    exit_after_entry = bool((dtz[first_entry:] > cfg.entry_tol).any()) if first_entry is not None else False
    exit_after_stable = bool((dtz[stable_entry:] > cfg.entry_tol).any()) if stable_entry is not None else False
    exit_before_k6 = False
    if first_entry is not None:
        end = k6 if k6 is not None else n
        exit_before_k6 = bool((dtz[first_entry:end] > cfg.entry_tol).any()) if end > first_entry else False
    illegal_target_exit = bool(exit_before_k6)

    # required-contact window = up to the frozen boundary (or the whole horizon if the boundary is not reached in-plan)
    win_end = boundary_step if boundary_step is not None else n
    acquired = bool(carry_touched or contact[:win_end].any() if win_end > 0 else carry_touched)
    if not acquired or win_end <= 0:
        pre_boundary_retention = 1.0; loss_step = None; loss_events = 0
    else:
        acq = int(np.argmax(contact[:win_end])) if contact[:win_end].any() else 0
        window = contact[acq:win_end]
        pre_boundary_retention = float(window.mean()) if window.size else 1.0
        drops = np.where(~window)[0]
        loss_step = int(acq + drops[0]) if drops.size else None
        loss_events = int(np.sum((~window[1:]) & (window[:-1])))          # True→False transitions in-window
        if not contact[acq]:                                             # already released at window start
            loss_events += 1
    premature_required_contact_loss = bool(acquired and pre_boundary_retention < cfg.contact_floor)

    n_violations = int(premature_required_contact_loss) + int(illegal_target_exit)
    return {"premature_required_contact_loss": premature_required_contact_loss,
            "illegal_target_exit": illegal_target_exit, "n_violations": n_violations, "feasible": n_violations == 0,
            "first_entry": first_entry, "first_stable_entry": stable_entry, "k6_step": k6,
            "exit_after_entry": exit_after_entry, "exit_after_stable_entry": exit_after_stable,
            "exit_before_k6": exit_before_k6, "pre_boundary_contact_retention": round(pre_boundary_retention, 4),
            "contact_loss_step": loss_step, "contact_loss_events": loss_events,
            "entry_speed": round(float(speed[first_entry]), 4) if first_entry is not None else None}


def score_candidate(inner, cf, qpos, qvel, actions_h, clearance, carry_touched, carry_dwell, *, cfg: FeasibilityConfig) -> dict:
    """Roll one candidate (H,4) open-loop from the snapshot (mirrors ``_score_horizon``'s rollout), collect per-step
    arrays, and classify feasibility + the lexicographic task fields. Stops at in-plan K6 (no post-placement reward)."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step
    _restore(inner, qpos, qvel)
    cert = DeliveryCertifier(initial_clearance=clearance)
    cert.robot_touched = bool(carry_touched); cert.delivery_dwell = int(carry_dwell)
    contact, dtz, speed, dwell, certified = [], [], [], [], []
    for t in range(len(actions_h)):
        cert.update(_cert_step(inner, cf))
        m = inner._planar_metrics
        contact.append(bool(m.left_contact or m.right_contact)); dtz.append(float(m.disk_to_zone))
        speed.append(float(np.linalg.norm(inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2])))
        dwell.append(int(cert.delivery_dwell)); certified.append(bool(cert.delivery_certified))
        if cert.delivery_certified:
            break
        inner.step(np.asarray(actions_h[t], np.float32))
        if not np.all(np.isfinite(inner.data.qvel)):                     # blow-up = maximally infeasible
            feas = classify_feasibility(contact, dtz, speed, dwell, certified, cfg=cfg, carry_touched=carry_touched)
            return {**feas, "finite": False, "any_strict": False, "max_dwell": int(max(dwell, default=carry_dwell)),
                    "min_dtz": 9.0, "excess_entry_speed": 9.0, "effort": 9.0, "n_violations": feas["n_violations"] + 2,
                    "feasible": False}
    feas = classify_feasibility(contact, dtz, speed, dwell, certified, cfg=cfg, carry_touched=carry_touched)
    entry_speed = feas["entry_speed"]
    excess = max(0.0, entry_speed - cfg.controlled_entry) if entry_speed is not None else 0.0
    return {**feas, "finite": True, "any_strict": bool(np.any(certified)),
            "max_dwell": int(max(dwell, default=carry_dwell)), "min_dtz": float(min(dtz, default=9.0)),
            "excess_entry_speed": round(excess, 4), "effort": round(float(np.mean(np.abs(actions_h))), 4)}


def repaired_key(r: dict):
    """Sort key (ascending; take max). Feasible beats infeasible; among infeasible, fewer violations wins (least-
    violating); then the lexicographic TASK objective with strict-K6 kept high (never last)."""
    return (int(r["feasible"]), -int(r["n_violations"]), int(r["any_strict"]), int(r["max_dwell"]),
            -float(r["min_dtz"]), -float(r["excess_entry_speed"]), -float(r["effort"]))


def select_candidate(results):
    """Return (best_index, best_result, all_candidates_infeasible). Never selects an infeasible candidate when a
    feasible one exists; flags ALL_CANDIDATES_INFEASIBLE when the best is still infeasible (least-violating chosen)."""
    idx = max(range(len(results)), key=lambda i: repaired_key(results[i]))
    return idx, results[idx], not bool(results[idx]["feasible"])


def plan_first_action_repaired(inner, cf, clearance, touched, dwell, seed, *, cfg: FeasibilityConfig,
                               horizon=30, pop=40, iters=6, elite=8, sigma0=0.6):
    """CEM over the H×4 arm sequence with the REPAIRED feasibility-gated selection. Identical candidate distribution to
    ``plan_first_action`` (same seed/pop/iters/elite/sigma, incumbent kept); leaves ``inner`` unchanged.

    Returns (first_action[4], best_result, all_candidates_infeasible)."""
    qpos, qvel = _snapshot(inner)
    dim = horizon * ACT_DIM
    mean = np.zeros(dim, np.float32); sigma = np.full(dim, sigma0, np.float32)
    rng = np.random.default_rng(seed)
    best_key = None; best_seq = mean.copy(); best_res = None
    for _it in range(iters):
        cand = np.clip(rng.normal(mean, sigma, size=(pop, dim)), -CTRL_LIM, CTRL_LIM).astype(np.float32)
        cand[0] = mean
        scored = [score_candidate(inner, cf, qpos, qvel, c.reshape(horizon, ACT_DIM), clearance, touched, dwell, cfg=cfg)
                  for c in cand]
        idx, res, _all_infeasible = select_candidate(scored)
        keys = [repaired_key(s) for s in scored]
        order = sorted(range(pop), key=lambda i: keys[i])                # ascending by the tuple key; elites are the top
        el = np.stack([cand[i] for i in order[-elite:]])
        mean, sigma = el.mean(0), el.std(0) + 0.05
        if best_key is None or repaired_key(res) > best_key:
            best_key, best_seq, best_res = repaired_key(res), cand[idx].copy(), res
    _restore(inner, qpos, qvel)
    first = np.clip(best_seq.reshape(horizon, ACT_DIM)[0], -CTRL_LIM, CTRL_LIM).astype(np.float32)
    return first, best_res, bool(not best_res["feasible"])


# ── execution: repaired planner as a harness policy + first-action stability + step-9 verdict ──
class RepairedPlannerPolicy:
    """Repaired H=30 planner as a ``rollout_from_handoff`` policy callable. Records per-step ALL_CANDIDATES_INFEASIBLE
    so the qualification can report its frequency. Same CEM config as the proven planner; only selection changes."""

    def __init__(self, *, cfg: FeasibilityConfig, plan_seed_base=0, horizon=30, pop=40, iters=6, elite=8):
        self.cfg = cfg; self.plan_seed_base = plan_seed_base
        self.horizon, self.pop, self.iters, self.elite = horizon, pop, iters, elite
        self.infeasible_steps: list[bool] = []

    def __call__(self, rl, prev_action, step):
        from hymeko_rl.experiments.coin_neutral_start import _clearance
        a, _res, all_inf = plan_first_action_repaired(
            rl.inner, rl.cf, _clearance(rl.inner), bool(rl._touched), int(rl._strict), self.plan_seed_base + step,
            cfg=self.cfg, horizon=self.horizon, pop=self.pop, iters=self.iters, elite=self.elite)
        self.infeasible_steps.append(bool(all_inf))
        return np.asarray(a, np.float32)


def _pairwise_cosine(mat):
    x = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    g = x @ x.T; iu = np.triu_indices(len(mat), k=1)
    return float(g[iu].mean()) if len(mat) > 1 else 1.0


def repaired_first_action_stability(rl, cfg: FeasibilityConfig, *, n_seeds=6, horizon=30, pop=40, iters=6, elite=8):
    """Run the repaired planner ``n_seeds`` times (distinct CEM seeds) from the SAME reconstructed handoff; report
    first-action agreement. plan_first_action_repaired restores the env internally, so the state is identical each run."""
    from hymeko_rl.experiments.coin_neutral_start import _clearance
    clearance = _clearance(rl.inner); touched = bool(rl._touched); dwell = int(rl._strict)
    firsts = [plan_first_action_repaired(rl.inner, rl.cf, clearance, touched, dwell, sd, cfg=cfg,
                                         horizon=horizon, pop=pop, iters=iters, elite=elite)[0] for sd in range(n_seeds)]
    fa = np.stack(firsts)
    return {"mean_mag": round(float(np.linalg.norm(fa.mean(0))), 4), "std_abs": round(float(fa.std(0).mean()), 4),
            "pairwise_cosine": round(_pairwise_cosine(fa), 4)}


def final_qualification(pi0_rows, planner_rows, *, contact_tol=0.05, exit_tol=0.05, strict_margin=0.05, dwell_margin=0.5):
    """Step-9 verdict. Emit H30_TEACHER_QUALIFIED only if the REPAIRED planner (1) does not materially reduce required-
    contact retention (and creates no new material contact loss), (2) does not materially increase target exit BEFORE
    K6, and (3) preserves a meaningful strict/dwell advantage over pi_0. Otherwise a named failure mechanism."""
    assert len(pi0_rows) == len(planner_rows) and pi0_rows, "aligned non-empty rows"
    rate = lambda rows, k: float(np.mean([int(r[k]) for r in rows]))
    mean = lambda rows, k: float(np.mean([r[k] for r in rows]))
    new_losses = sum(1 for b, c in zip(pi0_rows, planner_rows)
                     if c["lost_required_contact"] and not b["lost_required_contact"])
    d_contact = mean(planner_rows, "required_contact_retention") - mean(pi0_rows, "required_contact_retention")
    d_exit_k6 = rate(planner_rows, "exit_before_k6") - rate(pi0_rows, "exit_before_k6")
    d_strict = rate(planner_rows, "strict_success") - rate(pi0_rows, "strict_success")
    d_dwell = mean(planner_rows, "max_dwell") - mean(pi0_rows, "max_dwell")
    contact_ok = d_contact >= -contact_tol and new_losses == 0
    exit_ok = d_exit_k6 <= exit_tol
    advantage_ok = d_strict >= strict_margin or d_dwell >= dwell_margin
    reasons = []
    if not contact_ok:
        reasons.append("REPAIRED_LOSES_REQUIRED_CONTACT")
    if not exit_ok:
        reasons.append("REPAIRED_INCREASES_EXIT_BEFORE_K6")
    if not advantage_ok:
        reasons.append("REPAIRED_LOST_STRICT_ADVANTAGE")
    agg = {"mean_d_contact_retention": round(d_contact, 4), "new_required_contact_losses": new_losses,
           "d_exit_before_k6_rate": round(d_exit_k6, 4), "d_strict_rate": round(d_strict, 4), "mean_d_dwell": round(d_dwell, 4),
           "planner_req_contact": round(mean(planner_rows, "required_contact_retention"), 4),
           "pi0_req_contact": round(mean(pi0_rows, "required_contact_retention"), 4),
           "planner_strict_rate": round(rate(planner_rows, "strict_success"), 4), "pi0_strict_rate": round(rate(pi0_rows, "strict_success"), 4),
           "planner_exit_before_k6": round(rate(planner_rows, "exit_before_k6"), 4), "pi0_exit_before_k6": round(rate(pi0_rows, "exit_before_k6"), 4)}
    return {"aggregate": agg, "clauses": {"contact": contact_ok, "exit_before_k6": exit_ok, "advantage": advantage_ok},
            "qualified": not reasons,
            "verdict": "H30_TEACHER_QUALIFIED" if not reasons else "H30_TEACHER_UNQUALIFIED:" + ",".join(reasons)}
