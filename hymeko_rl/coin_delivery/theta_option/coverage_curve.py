"""COVERAGE-ONLY CAUSAL EXPERIMENT — does growing the development-cradle set alone close the held-out regression?

The one question, held fixed to a single independent variable N (the number of unique K6-deliverable DEVELOPMENT cradles
the BC actor is fit on):

    N = 2  → s1, s3
    N = 4  → s1, s3 + two additional cradles, selected + FROZEN before any training by a NON-OUTCOME rule
    N = 6  → all six usable development cradles

Held-out is ALWAYS {s4, s7} at every N; the evaluation panel is the frozen 4-state {s1,s3,s4,s7}; the model (B0), feature
set, optimiser, update count, initialisation seed, search budget (8), search semantics, K6, and motion/collision/task
contract are IDENTICAL across N. Every N is trained from a FRESH matched initialisation (no N=2 → N=4 continuation), so the
curve is a genuine coverage curve, not a fine-tuning trajectory.

This module holds only the PURE, testable coverage logic (the non-outcome N=4 selection rule, the nested dev sets, the
actor→teacher θ distance, the per-N record, and the decision-tree verdict). The physics/fit orchestration lives in the
benchmark mode (`coin_theta_rl_benchmark --coverage-curve`), which reuses the frozen teacher-bank / dataset / BC / deploy
machinery unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np

FROZEN_DEV_SEEDS = (14250, 14750)                    # s1, s3 — the frozen N=2 development set (nested into every larger N)


def select_n4_additions(new_seeds: list[int], fps_by_seed: dict[int, np.ndarray],
                        frozen_dev_seeds: tuple[int, ...] = FROZEN_DEV_SEEDS, *, k: int = 2) -> dict[str, Any]:
    """Pick the ``k`` cradles that MOST enlarge the geometric coverage of the frozen dev set, by greedy farthest-point
    sampling in the frozen 42-D geometry-fingerprint space. This is the frozen, NON-OUTCOME selection rule for the N=4
    additions: it reads ONLY the cradle geometry (never delivery / held-out K6), so it cannot be tuned to flatter the
    result. Deterministic; ties break to the lower seed. # Preconditions: every seed in ``new_seeds`` ∪ ``frozen_dev_seeds``
    has a fingerprint. # Postconditions: ``selected`` has ``k`` distinct seeds ⊆ ``new_seeds``, ordered by pick."""
    chosen_fps = [np.asarray(fps_by_seed[s], np.float64) for s in frozen_dev_seeds]
    remaining = list(new_seeds)
    # min-distance of every candidate to the frozen dev set (the auditable geometry, before any greedy pick)
    to_frozen = {s: round(float(min(np.linalg.norm(np.asarray(fps_by_seed[s], np.float64) - f) for f in chosen_fps)), 4)
                 for s in new_seeds}
    picks: list[int] = []
    trace: list[dict[str, Any]] = []
    for _ in range(int(k)):
        best_seed, best_d = None, -1.0
        for s in sorted(remaining):                  # deterministic scan; > keeps the first (lowest-seed) max
            d = min(float(np.linalg.norm(np.asarray(fps_by_seed[s], np.float64) - f)) for f in chosen_fps)
            if d > best_d:
                best_d, best_seed = d, s
        assert best_seed is not None, "select_n4_additions: no candidate left (k exceeds |new_seeds|)"
        picks.append(best_seed)
        trace.append({"picked_seed": best_seed, "min_dist_to_current_set": round(best_d, 4)})
        chosen_fps.append(np.asarray(fps_by_seed[best_seed], np.float64))
        remaining.remove(best_seed)
    return {"rule": "greedy_farthest_point_in_geometry_fingerprint_space", "non_outcome_based": True,
            "selected": picks, "candidate_min_dist_to_frozen_dev": to_frozen, "pick_trace": trace}


def nested_dev_sets(new_seeds_ordered: list[int], n4_additions: list[int],
                    frozen_dev_seeds: tuple[int, ...] = FROZEN_DEV_SEEDS) -> dict[int, list[int]]:
    """The three NESTED development sets. N=2 ⊂ N=4 ⊂ N=6: N=2 is the frozen dev pair, N=4 adds the two frozen-selected
    cradles, N=6 is the frozen pair plus every usable new cradle. # Postconditions: set(N2) ⊆ set(N4) ⊆ set(N6); |N2|=2,
    |N4|=2+len(n4_additions), |N6|=2+len(new_seeds_ordered)."""
    n2 = list(frozen_dev_seeds)
    n4 = list(frozen_dev_seeds) + list(n4_additions)
    n6 = list(frozen_dev_seeds) + list(new_seeds_ordered)
    return {2: n2, 4: n4, 6: n6}


def theta_distance_summary(rows: list[dict[str, Any]], box: Any) -> dict[str, Any]:
    """Actor→teacher θ distance per panel state and averaged per split. ``rows`` = [{tag, split, proposed[6], teacher[6]}].
    Reports the legal-unit L2 and the box-normalised L2 (the scale-free one, comparable across θ components). A shrinking
    held-out normalised distance with N is the mechanistic signature of coverage helping generalisation, even short of the
    gate. # Postconditions: per-split mean over the states present (None if a split is absent)."""
    per_state: dict[str, Any] = {}
    for r in rows:
        p = np.asarray(r["proposed"], np.float64)
        t = np.asarray(r["teacher"], np.float64)
        per_state[r["tag"]] = {"split": r["split"],
                               "theta_l2_legal": round(float(np.linalg.norm(p - t)), 5),
                               "theta_l2_norm": round(float(np.linalg.norm(box.norm(p) - box.norm(t))), 5)}
    by_split: dict[str, "float | None"] = {}
    for split in ("development", "held_out"):
        vals = [v["theta_l2_norm"] for v in per_state.values() if v["split"] == split]
        by_split[split] = round(float(np.mean(vals)), 5) if vals else None
    return {"per_state": per_state, "mean_theta_l2_norm_by_split": by_split}


def _sweep_at(out: dict[str, Any], key: str, budget: int) -> dict[str, Any]:
    sw = out[key]
    return sw[str(budget)] if str(budget) in sw else sw[budget]


def coverage_record(n: int, dev_seeds: list[int], out: dict[str, Any], dist: dict[str, Any]) -> dict[str, Any]:
    """Assemble the per-N report row required by the coverage spec: the exact train-cradle list; dev / held-out(s4,s7) K6
    at the gate budget; the total update-0 K6/4; the proposal-only (budget 0) vs budget-8-search outcome; the actor→teacher
    θ distance; and the failure mode + motion-contract status of any held-out miss. Pure over one N's `update_zero_eval`
    output + its θ-distance summary."""
    from hymeko_rl.coin_delivery.theta_option.cradle_expansion import classify_failure_mode
    gate_b = out["deploy_budget"]
    inf_g = _sweep_at(out, "informed_sweep", gate_b)
    inf_0 = _sweep_at(out, "informed_sweep", 0)
    per = inf_g["per_state"]
    held = {t: r for t, r in per.items() if r["split"] == "held_out"}
    failure = {t: classify_failure_mode({"dtz_end_mm": r["dtz_end_mm"], "k6_max_dwell": r["k6_max_dwell"],
                                         "peak_qdot": r["peak_qdot"], "peak_coin_speed": r["peak_coin_speed"]})
               for t, r in held.items() if not r["delivery_success"]}
    motion_ok = {t: bool(r["peak_qdot"] <= 3.0 and r["peak_coin_speed"] <= 1.5) for t, r in per.items()}
    return {
        "N": int(n), "train_cradles": list(dev_seeds), "n_train_cradles": len(dev_seeds),
        "gate_budget": gate_b,
        "gate_informed_dev_k6": inf_g["dev_k6"], "gate_informed_held_out_k6": inf_g["held_out_k6"],
        "gate_informed_total_k6": inf_g["total_k6"], "n_panel": inf_g["n_states"],
        "held_out_per_state_k6": {t: int(r["delivery_success"]) for t, r in held.items()},
        "held_out_dtz_end_mm": {t: r["dtz_end_mm"] for t, r in held.items()},
        "proposal_only_total_k6": inf_0["total_k6"], "proposal_only_held_out_k6": inf_0["held_out_k6"],
        "search8_total_k6": inf_g["total_k6"], "search8_held_out_k6": inf_g["held_out_k6"],
        "uninformed_total_k6": _sweep_at(out, "uninformed_sweep", gate_b)["total_k6"],
        "oracle_total_k6": out["oracle_gate_diagnostic"]["total_k6"],
        "actor_load_bearing": out["gate"]["actor_load_bearing"],
        "held_out_theta_l2_norm": dist["mean_theta_l2_norm_by_split"].get("held_out"),
        "dev_theta_l2_norm": dist["mean_theta_l2_norm_by_split"].get("development"),
        "theta_distance": dist,
        "held_out_failure_mode": failure, "motion_contract_ok_per_state": motion_ok,
        "gate_passed": bool(out["gate"]["passed"]), "authorises_rl": bool(out["authorises_rl"]),
        "verdict": out["verdict"], "diagnosed_blocker": out["diagnosed_blocker"],
    }


def _monotone_nonincreasing(xs: list[float]) -> bool:
    return all(b <= a + 1e-9 for a, b in zip(xs, xs[1:]))


def _monotone_nondecreasing(xs: list[int]) -> bool:
    return all(b >= a for a, b in zip(xs, xs[1:]))


def coverage_verdict(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The decision-tree verdict over the ordered per-N records. Reproduces the coverage spec:

        any N reaches the frozen 4/4 update-0 gate  → COVERAGE_REACHES_UPDATE_ZERO_GATE (AUTHORISE_SAC_TD3)
        N=6 does NOT reach it                        → COVERAGE_ALONE_INSUFFICIENT (stop before changing the model)

    and separately flags the positive mechanistic finding — HELD_OUT_GENERALISATION_IMPROVES_WITH_CRADLE_COVERAGE — when
    held-out K6 is non-decreasing (and rises above N=2) OR the held-out actor→teacher θ distance shrinks monotonically with
    N, even if the gate is not yet reached. # Postconditions: ``authorises_sac_td3`` ⟺ some N passed the frozen gate."""
    recs = sorted(records, key=lambda r: r["N"])
    passing = [r for r in recs if r["gate_passed"]]
    held_k6 = [int(r["gate_informed_held_out_k6"]) for r in recs]
    held_dist = [r["held_out_theta_l2_norm"] for r in recs if r["held_out_theta_l2_norm"] is not None]
    k6_improves = _monotone_nondecreasing(held_k6) and (len(held_k6) >= 2 and held_k6[-1] > held_k6[0])
    dist_improves = len(held_dist) == len(recs) and len(held_dist) >= 2 and _monotone_nonincreasing(held_dist) \
        and held_dist[-1] < held_dist[0] - 1e-6
    generalisation_improves = bool(k6_improves or dist_improves)
    out: dict[str, Any] = {
        "held_out_k6_by_N": {r["N"]: r["gate_informed_held_out_k6"] for r in recs},
        "held_out_theta_l2_norm_by_N": {r["N"]: r["held_out_theta_l2_norm"] for r in recs},
        "held_out_k6_nondecreasing_and_rises": bool(k6_improves),
        "held_out_theta_distance_shrinks_monotonically": bool(dist_improves),
        "generalisation_improves_with_coverage": generalisation_improves,
    }
    if passing:
        first = min(passing, key=lambda r: r["N"])
        out.update({"verdict": "COVERAGE_REACHES_UPDATE_ZERO_GATE", "authorise_sac_td3": True,
                    "first_passing_N": first["N"], "next_action": "AUTHORISE_SAC_TD3 → fresh BC checkpoint → matched SAC vs TD3",
                    "stop": False})
        return out
    reached_n6 = any(r["N"] == 6 for r in recs)
    out.update({"authorise_sac_td3": False, "first_passing_N": None, "stop": True})
    if generalisation_improves:
        out["mechanistic_finding"] = "HELD_OUT_GENERALISATION_IMPROVES_WITH_CRADLE_COVERAGE_BUT_UPDATE_ZERO_GATE_NOT_YET_REACHED"
    if reached_n6:
        out["verdict"] = "COVERAGE_ALONE_INSUFFICIENT"
        out["next_action"] = "acceptable-set / multimodal proposal → update-0 retry → only then RL (do NOT run SAC/TD3 now)"
    else:
        out["verdict"] = "COVERAGE_CURVE_INCOMPLETE"                # should not happen if N=6 ran
        out["next_action"] = "N=6 record missing — re-run the full curve before concluding"
    return out
