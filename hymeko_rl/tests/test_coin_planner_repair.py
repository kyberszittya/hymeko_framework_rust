"""REPAIR_H30_PLANNER_OBJECTIVE_V1 deterministic ordering + feasibility tests (pure; no env). Proves the five laws the
repaired scorer must satisfy (contract step 7) on the pure classifier and the feasibility-gated sort key."""
import numpy as np

from hymeko_rl.coin_delivery.coin_planner_repair import (
    FeasibilityConfig,
    classify_feasibility,
    repaired_key,
    select_candidate,
)


def _res(*, feasible=True, n_violations=0, any_strict=False, max_dwell=0, min_dtz=0.1, excess=0.0, effort=0.1):
    return {"feasible": feasible, "n_violations": n_violations, "any_strict": any_strict, "max_dwell": max_dwell,
            "min_dtz": min_dtz, "excess_entry_speed": excess, "effort": effort}


# ── step-7 law 1: an early contact-breaking strict candidate loses to a safe strict candidate ──
def test_early_contact_break_loses_to_safe_strict():
    early = _res(feasible=False, n_violations=1, any_strict=True, max_dwell=6)   # delivers but broke contact early
    safe = _res(feasible=True, any_strict=True, max_dwell=6)                     # delivers and kept contact
    idx, best, all_inf = select_candidate([early, safe])
    assert idx == 1 and not all_inf and repaired_key(safe) > repaired_key(early)


# ── step-7 law 2: a fast enter-and-exit candidate loses to a stable-entry candidate ──
def test_fast_exit_loses_to_stable_entry():
    fast_exit = _res(feasible=False, n_violations=1, any_strict=True, max_dwell=6, excess=1.0)
    stable = _res(feasible=True, any_strict=True, max_dwell=6, excess=0.0)
    idx, best, all_inf = select_candidate([fast_exit, stable])
    assert best is stable and not all_inf


# ── step-7 law 3: contact release AFTER strict-K6 is not penalised (both boundary definitions) ──
def test_release_after_k6_not_penalised():
    contact = np.array([True] * 6 + [False] * 4)                 # held through placement, released after
    dtz = np.array([0.3, 0.2, 0.1, 0.05, 0.02, 0.015, 0.015, 0.015, 0.015, 0.015])
    speed = np.array([0.4, 0.3, 0.2, 0.1, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02])
    dwell = np.array([0, 0, 0, 0, 1, 6, 6, 6, 6, 6])
    certified = np.array([False] * 5 + [True] * 5)               # k6 at step 5
    for boundary in ("stable_entry", "k6"):
        f = classify_feasibility(contact, dtz, speed, dwell, certified,
                                 cfg=FeasibilityConfig(boundary=boundary), carry_touched=True)
        assert f["feasible"] and not f["premature_required_contact_loss"]


def test_contact_loss_before_boundary_is_premature():
    n = 8
    contact = np.array([True] + [False] * 7)                     # acquired then abandoned, far from zone
    dtz = np.full(n, 0.3); speed = np.full(n, 0.4)
    dwell = np.zeros(n, int); certified = np.zeros(n, bool)
    f = classify_feasibility(contact, dtz, speed, dwell, certified,
                             cfg=FeasibilityConfig(boundary="k6"), carry_touched=True)
    assert f["premature_required_contact_loss"] and not f["feasible"] and f["contact_loss_step"] == 1


def test_illegal_exit_before_k6_flagged():
    contact = np.array([True] * 6)
    dtz = np.array([0.06, 0.04, 0.03, 0.08, 0.09, 0.10])         # enter at step 1, exit at step 3, never certified
    speed = np.full(6, 0.2); dwell = np.zeros(6, int); certified = np.zeros(6, bool)
    f = classify_feasibility(contact, dtz, speed, dwell, certified,
                             cfg=FeasibilityConfig(boundary="k6"), carry_touched=True)
    assert f["illegal_target_exit"] and f["exit_before_k6"] and not f["feasible"]


# ── step-7 law 4: when no candidate is feasible, the least-violating is selected and flagged ──
def test_all_infeasible_selects_least_violating_and_flags():
    two = _res(feasible=False, n_violations=2, any_strict=True, max_dwell=6)     # delivers but 2 violations
    one = _res(feasible=False, n_violations=1, any_strict=False, max_dwell=0)    # worse task, but 1 violation
    idx, best, all_inf = select_candidate([two, one])
    assert idx == 1 and best is one and all_inf                                  # fewer violations wins, flagged


# ── step-7 law 5: strict success outranks mere progress among feasible candidates ──
def test_strict_outranks_progress_when_feasible():
    strict = _res(feasible=True, any_strict=True, min_dtz=0.03)                  # delivers, farther min approach
    progress = _res(feasible=True, any_strict=False, min_dtz=0.005)             # closer, but never certifies
    idx, best, all_inf = select_candidate([progress, strict])
    assert best is strict and not all_inf and repaired_key(strict) > repaired_key(progress)
