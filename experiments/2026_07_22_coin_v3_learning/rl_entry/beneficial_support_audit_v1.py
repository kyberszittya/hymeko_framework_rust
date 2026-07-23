"""STAGE A — BENEFICIAL_SUPPORT_AUDIT_V1. Label-only audit of the re-materialized hold-sweep labels (no rollouts, no
critic). Is the K-hold leverage increase backed by BENEFICIAL support, or merely by growing HARM? Selects an eligible K
by beneficial support + separability, NOT by maximum absolute leverage.

Frozen definitions (below, recorded in the output) are fixed BEFORE analysis. State identity = the captured
obs/base/causal-state/gate stored in the labels; nothing is paired against a post-restore recomputed observation.
"""
import json

import numpy as np

LABELS = "experiments/2026_07_22_coin_v3_learning/rl_entry/hold_sweep_v1_labels.json"
OUT = "experiments/2026_07_22_coin_v3_learning/rl_entry/beneficial_support_audit_v1.json"
FAM = ("transport", "entry", "settling", "contact_retention")

FROZEN = {
    "beneficial_threshold": 1.0,           # dG > 1 ⇒ beneficial
    "beneficial_thresholds_reported": [1.0, 5.0, 10.0],
    "harm_definition": "dG < -1.0 OR contact_break OR target_exit",
    "contact_preserving": "outcome.contact_persist == True",
    "non_harmful": "contact_persist AND not target_exit AND dG >= -1.0",
    "separability_margin": 1.0,            # best − second-best (among a group's candidates) must be ≥ this
    "support_min": 0.30,                   # pooled fraction of groups with a good candidate to call support sufficient
    "underpowered_min_groups_per_family": 8,
    "K_selection_rule": "eligible K = argmax_K pooled fraction of groups with a beneficial contact-preserving non-exit "
                        "candidate that also clears the separability margin; require ≥ support_min; tie-break smaller K; "
                        "NOT chosen by maximum |ΔG|.",
}
BT, HT, SEP, SUPMIN = FROZEN["beneficial_threshold"], 1.0, FROZEN["separability_margin"], FROZEN["support_min"]


def _classify(rec_K):
    """Per-candidate boolean arrays for one group@K (index 0 = zero candidate)."""
    dG = np.asarray(rec_K["dG"], float); cp = np.asarray(rec_K["contact_persist"], bool)
    te = np.asarray(rec_K["target_exit"], bool); dwell = np.asarray(rec_K["max_dwell"], int)
    beneficial = dG > BT
    harmful = (dG < -HT) | (~cp) | te
    non_harmful = cp & (~te) & (dG >= -HT)
    dwell_inc = dwell > rec_K["dwell0"]
    return dG, cp, te, beneficial, harmful, non_harmful, dwell_inc


def _group_stats(rec_K):
    dG, cp, te, ben, harm, nonh, dwell_inc = _classify(rec_K)
    order = np.argsort(dG)[::-1]
    best_i = int(order[0]); second = float(dG[order[1]]) if len(order) > 1 else float("-inf")
    nonh_dG = dG[nonh]
    return {
        "has_dG_gt1": bool((dG[1:] > 1).any()), "has_dG_gt5": bool((dG[1:] > 5).any()), "has_dG_gt10": bool((dG[1:] > 10).any()),
        "has_beneficial_contact_preserving": bool((ben & cp)[1:].any()),
        "has_beneficial_non_exit": bool((ben & ~te)[1:].any()),
        "has_beneficial_dwell_increase": bool((ben & dwell_inc)[1:].any()),
        "only_neutral_or_harmful": bool(not (dG[1:] > 1).any()),
        "best_dG": round(float(dG[best_i]), 3), "best_non_harmful_dG": round(float(nonh_dG.max()), 3) if nonh_dG.size else 0.0,
        "best_minus_zero": round(float(dG[best_i]), 3), "best_minus_second": round(float(dG[best_i] - second), 3),
        "best_magnitude": rec_K["magnitude"][best_i], "best_dir": rec_K["dir"][best_i],
        "best_is_nonharmful": bool(nonh[best_i]), "best_stable": True,   # ×2 determinism certified in the sweep
        "good_candidate": bool((ben & cp & ~te)[1:].any() and (dG[best_i] - second) >= SEP and nonh[best_i]),
        "beneficial_names": [rec_K["names"][i] for i in range(1, len(dG)) if ben[i]],
        "beneficial_magnitudes": [rec_K["magnitude"][i] for i in range(1, len(dG)) if ben[i]],
    }


def main():
    L = json.load(open(LABELS)); groups = L["groups"]; KS = [int(k) for k in L["K_values"]]
    assert L.get("matches_completed_aggregates"), "materialized labels do not match completed sweep aggregates"
    per = {int(gid): {K: _group_stats(rec["K"][str(K)]) for K in KS} for gid, rec in groups.items()}
    fam_of = {int(gid): rec["family"] for gid, rec in groups.items()}
    n_by_fam = {f: sum(v == f for v in fam_of.values()) for f in FAM}

    def frac_count(K, fam, key):
        gids = [gid for gid in per if fam_of[gid] == fam]
        c = sum(per[gid][K][key] for gid in gids)
        return {"frac": round(c / max(len(gids), 1), 3), "count": c, "n": len(gids)}

    report = {"frozen": FROZEN, "n_by_family": n_by_fam,
              "state_manifest_sha": L["state_manifest_sha"], "matches_completed_aggregates": L["matches_completed_aggregates"],
              "by_K_family": {}, "K_selection": {}, "across_K": {}}
    for K in KS:
        report["by_K_family"][str(K)] = {}
        for fam in FAM:
            gids = [gid for gid in per if fam_of[gid] == fam]
            report["by_K_family"][str(K)][fam] = {
                "item1_has_dG_gt1": frac_count(K, fam, "has_dG_gt1"), "item1_has_dG_gt5": frac_count(K, fam, "has_dG_gt5"),
                "item1_has_dG_gt10": frac_count(K, fam, "has_dG_gt10"),
                "item2_beneficial_contact_preserving": frac_count(K, fam, "has_beneficial_contact_preserving"),
                "item2_beneficial_non_exit": frac_count(K, fam, "has_beneficial_non_exit"),
                "item2_beneficial_dwell_increase": frac_count(K, fam, "has_beneficial_dwell_increase"),
                "item2_only_neutral_or_harmful": frac_count(K, fam, "only_neutral_or_harmful"),
                "item3_median_best_dG": round(float(np.median([per[g][K]["best_dG"] for g in gids])), 3),
                "item3_median_best_non_harmful_dG": round(float(np.median([per[g][K]["best_non_harmful_dG"] for g in gids])), 3),
                "item3_median_best_minus_second": round(float(np.median([per[g][K]["best_minus_second"] for g in gids])), 3),
                "good_candidate_support": frac_count(K, fam, "good_candidate"),
            }

    # §4 across-K: pooled beneficial support, harmful trend, identity/sign agreement, beneficial magnitudes
    def pooled(K, key):
        return round(np.mean([per[gid][K][key] for gid in per]), 3)
    support = {K: pooled(K, "good_candidate") for K in KS}
    ben_cp = {K: pooled(K, "has_beneficial_contact_preserving") for K in KS}
    only_harm = {K: pooled(K, "only_neutral_or_harmful") for K in KS}
    # beneficial-candidate identity Jaccard across consecutive K (per group), mean
    def jaccard(a, b):
        A, B = set(a), set(b)
        return len(A & B) / len(A | B) if (A | B) else 1.0
    jac = {}
    for i in range(1, len(KS)):
        js = [jaccard(per[gid][KS[i - 1]]["beneficial_names"], per[gid][KS[i]]["beneficial_names"]) for gid in per]
        jac[f"{KS[i-1]}->{KS[i]}"] = round(float(np.mean(js)), 3)
    ben_mags = {K: sorted({m for gid in per for m in per[gid][K]["beneficial_magnitudes"]}) for K in KS}
    report["across_K"] = {"pooled_good_candidate_support": support, "pooled_beneficial_contact_preserving": ben_cp,
                          "pooled_only_neutral_or_harmful": only_harm, "beneficial_identity_jaccard_consecutiveK": jac,
                          "beneficial_magnitudes_present": {str(K): v for K, v in ben_mags.items()}}

    # K selection (frozen rule) — by support+separability, tie-break smaller K
    eligible = [K for K in KS if K != 1 and support[K] >= SUPMIN]
    if eligible:
        best_support = max(support[K] for K in eligible)
        chosen = min(K for K in eligible if support[K] == best_support)
    else:
        chosen = None
    report["K_selection"] = {"pooled_support_by_K": support, "support_min": SUPMIN,
                             "eligible_K": eligible, "selected_K": chosen,
                             "rationale": "highest pooled good-candidate support ≥ support_min, smallest K on ties; "
                                          "NOT chosen by maximum |ΔG|."}

    # verdict
    underpowered = any(n_by_fam[f] < FROZEN["underpowered_min_groups_per_family"] for f in FAM)
    max_support = max(support.values())
    if underpowered:
        verdict = "HOLD_SUPPORT_AUDIT_UNDERPOWERED"
    elif eligible and max_support >= SUPMIN:
        verdict = "BENEFICIAL_RESIDUAL_SUPPORT_CONFIRMED"
    elif max_support > 0.05:
        verdict = "BENEFICIAL_SUPPORT_SPARSE"
    else:
        verdict = "HOLD_SIGNAL_DOMINATED_BY_HARM"
    report["verdict"] = verdict
    json.dump(report, open(OUT, "w"), indent=1, default=float)

    print(f"n_by_family {n_by_fam}")
    print("pooled good-candidate support by K:", support)
    print("pooled beneficial-contact-preserving by K:", ben_cp)
    print("pooled only-neutral/harmful by K:", only_harm)
    print("beneficial magnitudes present by K:", {K: ben_mags[K] for K in KS})
    print("eligible_K", eligible, "selected_K", chosen)
    print("VERDICT:", verdict, "\nwrote", OUT)


if __name__ == "__main__":
    main()
