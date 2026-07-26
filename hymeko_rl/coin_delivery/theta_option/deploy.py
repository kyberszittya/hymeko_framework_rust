"""STAGE 4 — update-0 no-regression deploy on the frozen 4-state panel.

The learned actor proposes θ_0 from each cradle's causal features; the SAME fixed bounded search selects θ_exec; the
frozen physical option runs; frozen K6 decides. The hard gate is K6 = 4/4 including both held-out cradles, with no motion
or safety regression, the same option semantics, the same structured-search budget, exact θ_0/θ_exec provenance, and no
hidden teacher fallback.

Central honesty control: the actor is only load-bearing if INFORMED θ_0 (the BC actor) beats an UNINFORMED θ_0 (the box
centre) at the SAME fixed budget. A large search can deliver regardless of θ_0 — that would make a "pass" a search
artifact, not the learned actor. We therefore sweep the budget {0,4,8} for BOTH conditions and report the gap. Budget 0
executes θ_0 directly (no search) — the purest test of the actor. The search budget is NEVER enlarged for the learned
actor only.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.theta_option.dataset import causal_history, flatten_features, structured_features
from hymeko_rl.coin_delivery.theta_option.proposal import BCProposal, Featurizer
from hymeko_rl.coin_delivery.theta_option.search import fixed_search_select, search_semantics
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG, DIM, ThetaBox
from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot

DEPLOY_BUDGETS = (0, 4, 8)
DEPLOY_BUDGET = 8                                    # the canonical deploy budget (the gate is read at this budget)
SEARCH_SEED_BASE = 80000


class PanelState:
    """One panel cradle: re-acquired snapshot + its causal features/history + the frozen teacher θ. Snapshot-bound (physics)."""

    def __init__(self, tag: str, split: str, seed: int, snap: CradleSnapshot, teacher_theta: np.ndarray):
        self.tag, self.split, self.seed, self.snap, self.teacher_theta = tag, split, seed, snap, teacher_theta
        feat = structured_features(snap)
        self.features = flatten_features(feat)
        self.history = causal_history(snap)


def build_panel(harness: Any, bank: dict[str, Any]) -> list[PanelState]:
    """Re-acquire the frozen 4-state panel (s1,s3,s4,s7) and attach causal features + the teacher θ. Verifies the
    re-acquired snapshot hash matches the bank (cross-stage replay). # Postconditions: 4 snapshot-bound panel states."""
    panel = []
    for e in bank["states"]:
        if "canonical_theta_vec" not in e:
            continue
        snap, _m = acquire_snapshot(harness, e["seed"])
        if snap is None:
            raise RuntimeError(f"update-0: could not re-acquire {e['tag']}")
        if snap.post_release_hash != e["snapshot"]["post_release_hash"]:
            raise RuntimeError(f"update-0: {e['tag']} snapshot hash drift vs teacher bank")
        panel.append(PanelState(e["tag"], e["split"], e["seed"], snap, np.asarray(e["canonical_theta_vec"], np.float64)))
    return panel


def _deploy_one(ps: PanelState, theta0: np.ndarray, budget: int, seed: int) -> dict[str, Any]:
    """Deploy a single θ_0 through the fixed search on this cradle. Returns the outcome + full θ_0/θ_exec provenance."""
    prov = fixed_search_select(ps.snap, theta0, np.random.default_rng(seed), budget=budget, cfg=DELIVERY_CFG)
    box = ThetaBox()
    return {"theta0": [round(float(x), 5) for x in prov.center],
            "theta_exec": [round(float(x), 5) for x in prov.selected],
            "search_displacement_norm": round(prov.displacement(box), 5),
            "k6_delivered": bool(prov.outcome["k6_delivered"]), "k6_max_dwell": int(prov.outcome["k6_max_dwell"]),
            "delivery_success": bool(prov.outcome["delivery_success"]), "dtz_end_mm": round(prov.outcome["dtz_end"] * 1000, 2),
            "terminal_coin_speed": round(prov.outcome["terminal_coin_speed"], 4),
            "peak_qdot": round(prov.outcome["peak_qdot"], 4), "peak_coin_speed": round(prov.outcome["peak_coin_speed"], 4),
            "budget": int(budget)}


def _box_center() -> np.ndarray:
    return ThetaBox().denorm(np.zeros(DIM, np.float32))


def evaluate_condition(panel: list[PanelState], theta0_fn: Callable[[PanelState], np.ndarray], budget: int) -> dict[str, Any]:
    """Evaluate one θ_0 source across the panel at a fixed budget (fixed per-state search seed). Returns per-state
    outcomes + the K6 count split by dev/held-out."""
    per = {}
    for i, ps in enumerate(panel):
        per[ps.tag] = {"split": ps.split, "teacher_theta": [round(float(x), 5) for x in ps.teacher_theta],
                       **_deploy_one(ps, theta0_fn(ps), budget, SEARCH_SEED_BASE + i * 131 + budget)}
    dev_k6 = sum(1 for t, r in per.items() if r["split"] == "development" and r["delivery_success"])
    held_k6 = sum(1 for t, r in per.items() if r["split"] == "held_out" and r["delivery_success"])
    return {"budget": int(budget), "per_state": per, "dev_k6": dev_k6, "held_out_k6": held_k6,
            "total_k6": dev_k6 + held_k6, "n_states": len(panel)}


def update_zero_eval(panel: list[PanelState], prop: BCProposal, fz: Featurizer, *,
                     budgets: tuple[int, ...] = DEPLOY_BUDGETS) -> dict[str, Any]:
    """Full update-0 evaluation: informed (BC actor) vs uninformed (box centre) θ_0, swept over budgets, on the frozen
    panel. The gate reads the INFORMED condition at ``DEPLOY_BUDGET``. Emits the verdict + the actor-load-bearing analysis."""
    center = _box_center()

    def informed(ps: PanelState) -> np.ndarray:
        return prop.center(fz.obs(ps.features, ps.history))

    def uninformed(_ps: PanelState) -> np.ndarray:
        return center

    def oracle(ps: PanelState) -> np.ndarray:                     # DIAGNOSTIC ONLY — the recorded teacher θ, NOT the actor
        return ps.teacher_theta

    informed_sweep = {b: evaluate_condition(panel, informed, b) for b in budgets}
    uninformed_sweep = {b: evaluate_condition(panel, uninformed, b) for b in budgets}
    gate_b = DEPLOY_BUDGET
    # Oracle validates the NEGATIVE (memory guard: trust a negative only if the positive oracle passes the SAME predicate):
    # if the teacher θ + the SAME fixed search delivers 4/4, the search+physics are fine and the sole cause of any
    # held-out miss is the actor's θ_0 (coverage), not the search/physics. Not part of the actor; uses no held-out FIT.
    oracle_gate = evaluate_condition(panel, oracle, gate_b)
    inf = informed_sweep[gate_b]
    unf = uninformed_sweep[gate_b]
    n = inf["n_states"]
    dev_ok = inf["dev_k6"] == 2
    held_ok = inf["held_out_k6"] == 2
    all_ok = inf["total_k6"] == n
    # actor load-bearing: informed strictly beats uninformed at the gate budget (and ideally at budget 0)
    actor_gap = inf["total_k6"] - unf["total_k6"]
    load_bearing = bool(actor_gap > 0)
    if all_ok and dev_ok and held_ok:
        verdict = ("UPDATE_ZERO_MOBILE_TEACHER_NO_REGRESSION_PASS" if load_bearing
                   else "UPDATE_ZERO_NO_REGRESSION_PASS_BUT_SEARCH_MASKED")
    elif dev_ok and not held_ok:
        verdict = "UPDATE_ZERO_REGRESSES_HELD_OUT"
    else:
        verdict = "UPDATE_ZERO_PARTIAL_REPRODUCTION"
    oracle_ok = bool(oracle_gate["total_k6"] == n)
    blocker = None
    if not all_ok:
        blocker = ("INSUFFICIENT_DATASET_COVERAGE_2_DEV_CRADLES" if oracle_ok
                   else "SEARCH_OR_PHYSICS_DEFECT")               # if even teacher θ fails, it's not coverage
    return {"contract": "COIN_6D_THETA_UPDATE_ZERO_V1", "base_commit": "a3459629",
            "variant": prop.variant, "deploy_budget": gate_b, "budgets": list(budgets),
            "search_std": None,  # filled by the caller
            "search_semantics": search_semantics(gate_b),
            "informed_sweep": informed_sweep, "uninformed_sweep": uninformed_sweep,
            "oracle_gate_diagnostic": oracle_gate,
            "gate": {"informed_total_k6": inf["total_k6"], "informed_dev_k6": inf["dev_k6"],
                     "informed_held_out_k6": inf["held_out_k6"], "n_states": n,
                     "uninformed_total_k6": unf["total_k6"], "oracle_total_k6": oracle_gate["total_k6"],
                     "actor_gap_vs_uninformed": actor_gap, "actor_load_bearing": load_bearing,
                     "oracle_validates_search_and_physics": oracle_ok, "no_hidden_teacher_fallback": True,
                     "passed": bool(all_ok)},
            "verdict": verdict, "diagnosed_blocker": blocker,
            "authorises_rl": bool(all_ok and verdict == "UPDATE_ZERO_MOBILE_TEACHER_NO_REGRESSION_PASS")}
