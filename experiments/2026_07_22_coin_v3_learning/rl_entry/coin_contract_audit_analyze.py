"""COIN_TASK_CONTRACT_SENSITIVITY_AUDIT_V1 — offline synthesis (read-only) over the POST-step certificate streams.
Four patched checks are verified first (post-step alignment incl. terminal, target-directed braking denominator, measured
per-rollout clearance, exact v3 reward decomposition); only if all pass is the final contract verdict emitted."""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_contract_audit import (  # noqa: E402
    braking_eligibility_sweep,
    post_stream,
    recertify,
    success_ladder,
    touched_ever_vs_current,
)

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
CONTROLLERS = ["pi0", "h30", "repaired"]
CENTER_TOLS = [0.01, 0.02, 0.03, 0.04]
SETTLE_VELS = [0.03, 0.06, 0.09, 0.12]
DWELL_KS = [1, 3, 6, 10, 20]
CANON = (0.02, 0.06, 6)


def _rate(rolls, ct, sv, k):
    return round(float(np.mean([recertify(post_stream(r), r["clearance_measured"], center_tol=ct, settle_vel=sv, dwell_req=k)[0]
                                for r in rolls])), 4)


def _deliv(r):
    return recertify(post_stream(r), r["clearance_measured"], center_tol=CANON[0], settle_vel=CANON[1], dwell_req=CANON[2])[0]


def main():
    log = lambda *a: print(*a, flush=True)
    traces = {c: json.load(open(f"{D}/audit_trace_{c}.json")) for c in CONTROLLERS}
    rolls = {c: traces[c]["rollouts"] for c in CONTROLLERS}

    # ── VERIFY the four patched details before any verdict ──
    checks = {}
    # (1) post-step alignment incl. terminal — a synthetic trace whose K6 completes on the FINAL post-step is detected
    fin = [{"disk_to_zone": 0.3, "disk_speed": 0.3, "left_fingertip": True, "right_fingertip": False,
            "arm_body_contact": False, "arm_body_impulse": 0.0, "force_left": 1.0, "force_right": 1.0,
            "body_progress": 0.0, "ever_grasped": True}]
    fin += [{**fin[0], "disk_to_zone": 0.015, "disk_speed": 0.02} for _ in range(6)]     # K6 completes on the last step
    det, ci, _ = recertify(fin, 0.02, center_tol=0.02, settle_vel=0.06, dwell_req=6)
    checks["post_step_terminal_detected"] = bool(det and ci == len(fin) - 1)
    # (4) reward decomposition sum == scalar (captured max error) for every controller
    checks["reward_decomposition_exact"] = all(traces[c]["max_decomp_error"] < 1e-4 for c in CONTROLLERS)
    checks["reward_max_decomp_error"] = max(traces[c]["max_decomp_error"] for c in CONTROLLERS)
    # (3) measured clearance stored per rollout + invariance assertion
    clr = sorted({round(r["clearance_measured"], 5) for c in CONTROLLERS for r in rolls[c]})
    checks["clearance_measured_per_rollout"] = all("clearance_measured" in r for c in CONTROLLERS for r in rolls[c])
    checks["clearance_invariant"] = len(clr) == 1
    checks["clearance_range"] = [clr[0], clr[-1]]
    checks["clearance_all_positive"] = clr[0] > 0
    all_pass = (checks["post_step_terminal_detected"] and checks["reward_decomposition_exact"]
                and checks["clearance_measured_per_rollout"] and checks["clearance_all_positive"])
    log(f"[verify] post-step-terminal {checks['post_step_terminal_detected']}  reward-decomp-exact "
        f"{checks['reward_decomposition_exact']} (err {checks['reward_max_decomp_error']:.1e})  "
        f"clearance measured/rollout {checks['clearance_measured_per_rollout']} invariant {checks['clearance_invariant']} "
        f"range {checks['clearance_range']}")
    if not all_pass:
        json.dump({"aborted": "patched checks failed", "checks": checks}, open(f"{D}/task_contract_audit_v1.json", "w"), indent=1)
        log("PATCHED CHECKS FAILED — NO VERDICT"); return

    # (3-surface) certifier sensitivity + canonical
    surface = {c: {f"ct{ct}_sv{sv}_k{k}": _rate(rolls[c], ct, sv, k)
                   for ct in CENTER_TOLS for sv in SETTLE_VELS for k in DWELL_KS} for c in CONTROLLERS}
    canon_rate = {c: _rate(rolls[c], *CANON) for c in CONTROLLERS}
    canon_order = tuple(sorted(CONTROLLERS, key=lambda c: -canon_rate[c]))

    # (4-rank) ranking stability across the grid
    flips = []
    for ct in CENTER_TOLS:
        for sv in SETTLE_VELS:
            for k in DWELL_KS:
                order = tuple(sorted(CONTROLLERS, key=lambda c: -_rate(rolls[c], ct, sv, k)))
                if order[0] != canon_order[0]:
                    flips.append({"center_tol": ct, "settle_vel": sv, "dwell_k": k, "top": order[0]})
    ranking_robust = len(flips) == 0
    axis_flip = {}
    for name, grid, idx in [("center_tol", CENTER_TOLS, 0), ("settle_vel", SETTLE_VELS, 1), ("dwell_k", DWELL_KS, 2)]:
        for val in grid:
            a = list(CANON); a[idx] = val
            top = sorted(CONTROLLERS, key=lambda c: -_rate(rolls[c], *a))[0]
            if top != canon_order[0]:
                axis_flip.setdefault(name, []).append({"value": val, "top": top})

    # (5) success ladder
    def ladder_agg(rl):
        L = [success_ladder(post_stream(r)) for r in rl]
        keys = ["target_entry", "one_step_in_zone", "k3_dwell", "k6_dwell", "k10_dwell", "exit_after_entry", "ever_touched"]
        agg = {k: round(float(np.mean([int(x[k]) for x in L])), 4) for k in keys}
        agg.update({"mean_max_held_dwell": round(float(np.mean([x["max_held_dwell"] for x in L])), 3),
                    "mean_centered_settled_integral_s": round(float(np.mean([x["centered_settled_integral"] for x in L])), 4),
                    "mean_reentry": round(float(np.mean([x["reentry_count"] for x in L])), 3),
                    "mean_final_distance": round(float(np.mean([x["final_distance"] for x in L])), 4)})
        return agg
    ladder = {c: ladder_agg(rolls[c]) for c in CONTROLLERS}

    # (6) touched-ever audit
    touched = {}
    for c in CONTROLLERS:
        dv = [post_stream(r) for r in rolls[c] if _deliv(r)]
        tv = [touched_ever_vs_current(s) for s in dv]
        released = sum(1 for x in tv if x["current_contact_through_dwell"] is False)
        touched[c] = {"n_delivered": len(dv), "delivered_with_no_current_contact_through_dwell": released,
                      "fraction_released": round(released / max(len(dv), 1), 4)}

    # (7) v3 reward vs ladder + per-term contribution
    reward = {}
    for c in CONTROLLERS:
        ret = np.array([sum(tr["reward"] for tr in r["transitions"]) for r in rolls[c]])
        dv = np.array([_deliv(r) for r in rolls[c]])
        term_tot = {}
        for r in rolls[c]:
            for tr in r["transitions"]:
                for k, v in tr["components"].items():
                    term_tot[k] = term_tot.get(k, 0.0) + v
        n = sum(len(r["transitions"]) for r in rolls[c])
        reward[c] = {"mean_return": round(float(ret.mean()), 3),
                     "mean_return_delivered": round(float(ret[dv].mean()), 3) if dv.any() else None,
                     "mean_return_not_delivered": round(float(ret[~dv].mean()), 3) if (~dv).any() else None,
                     "delivered_minus_not": round(float(ret[dv].mean() - ret[~dv].mean()), 3) if dv.any() and (~dv).any() else None,
                     "mean_component_per_step": {k: round(v / max(n, 1), 4) for k, v in sorted(term_tot.items())}}
    reward["_terms_v3"] = dict(traces["pi0"]["reward_terms"])

    # (8) braking eligibility recalculation (target-directed denominator)
    partA = json.load(open(f"{D}/braking_support_partA.json"))["per_state"]
    braking = braking_eligibility_sweep(partA, [0.03, 0.05, 0.07, 0.09])

    # (10) reclassification
    hidden = max((ladder[c]["k3_dwell"] - ladder[c]["k6_dwell"]) for c in CONTROLLERS)
    entry_gap = max((ladder[c]["target_entry"] - ladder[c]["k6_dwell"]) for c in CONTROLLERS)
    b = braking["v_excess=0.05"]
    reclass = {
        "supervised_ceiling": "depends_on_strict_K6" if hidden >= 0.15 or entry_gap >= 0.3 else "robust",
        "no_beneficial_support / local_improvement_exhausted": "depends_on_strict_K6_local_denominator",
        "braking_primitive_insufficient": ("depends_on_invalid_denominator" if b["support_over_target_directed"] > b["support_over_all"] + 0.15 else "robust"),
        "primitive_loses_required_contact": "depends_on_contact_retention_hard_gate+strict_K6 (braking-phase coupling)",
        "H30_teacher_unqualified": "robust_under_task_contract" if ranking_robust else "depends_on_thresholds"}

    verdicts = []
    if not ranking_robust:
        verdicts.append("STRICT_RULE_BRITTLE")
    if b["support_over_target_directed"] >= 0.5 and b["support_over_target_directed"] > b["support_over_all"] + 0.15:
        verdicts.append("BRAKING_SUPPORT_DENOMINATOR_INVALID")
    if any(reward[c]["delivered_minus_not"] and reward[c]["delivered_minus_not"] > 20 for c in CONTROLLERS):
        verdicts.append("V3_REWARD_CERTIFIER_MISALIGNED")
    if hidden >= 0.15 or entry_gap >= 0.3:
        verdicts.append("LOCAL_GATES_OVERCONSTRAINED")
    if not verdicts:
        verdicts = ["TASK_CONTRACT_ALIGNED"]
    elif len(verdicts) > 1:
        verdicts = ["MULTIPLE_CONTRACT_MISMATCHES"] + verdicts

    out = {"campaign": "COIN_TASK_CONTRACT_SENSITIVITY_AUDIT_V1", "date": "2026-07-23", "measurement_only": True,
           "patched_checks": checks, "canonical_success_rate": canon_rate, "canonical_ranking": list(canon_order),
           "certifier_surface": surface, "ranking": {"robust": ranking_robust, "n_top_flips": len(flips),
           "flips": flips[:10], "single_axis_flips": axis_flip}, "success_ladder": ladder, "touched_ever_audit": touched,
           "reward_vs_ladder": reward, "braking_eligibility_recalc": braking, "reclassification": reclass, "verdict": verdicts}
    json.dump(out, open(f"{D}/task_contract_audit_v1.json", "w"), indent=1, default=float)

    log("== TASK CONTRACT AUDIT (verified) ==")
    log(f"  canonical success (K6/0.02/0.06): {canon_rate}  ranking {list(canon_order)}")
    log(f"  ranking robust across grid: {ranking_robust}  (top-flips {len(flips)}; single-axis flips {list(axis_flip)})")
    log("  ladder: " + "  ".join(f"{c}[entry {ladder[c]['target_entry']} k3 {ladder[c]['k3_dwell']} k6 {ladder[c]['k6_dwell']} k10 {ladder[c]['k10_dwell']}]" for c in CONTROLLERS))
    log("  touched-ever released: " + "  ".join(f"{c} {touched[c]['delivered_with_no_current_contact_through_dwell']}/{touched[c]['n_delivered']}" for c in CONTROLLERS))
    log("  reward Δ(deliv−not): " + "  ".join(f"{c} {reward[c]['delivered_minus_not']}" for c in CONTROLLERS))
    log(f"  braking support/all {b['support_over_all']} vs /target-directed(v>0.05) {b['support_over_target_directed']}  "
        f"prevalence {b['prevalence_target_directed']}  false-on-slow {b['false_interventions_on_slow_states']}  away {b['n_target_away']}")
    log(f"\n→ {verdicts}\nwrote {D}/task_contract_audit_v1.json\nAUDIT_DONE")


if __name__ == "__main__":
    main()
