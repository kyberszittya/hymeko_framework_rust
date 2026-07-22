"""§2/§3/§5 matched H=15 vs H=30 comparison + corrective gate. Reads the two instrumented pilot_result.json files."""
import json
import sys

H15 = json.load(open(sys.argv[1]))   # instrumented H=15 baseline
H30 = json.load(open(sys.argv[2]))   # H=30 corrective
COMMITTED = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None


def rows_by_seed(res, bank):
    return {r["seed"]: r for r in res[bank]["rows"]}


def transitions(bank):
    a = rows_by_seed(H15, bank)
    b = rows_by_seed(H30, bank)
    t = {"fail_to_success": [], "success_to_success": [], "success_to_fail": [], "fail_to_diff_failure": []}
    settle_convert = []
    for s in sorted(a):
        ra, rb = a[s], b[s]
        pa, pb = ra["replay_certified"], rb["replay_certified"]
        if not pa and pb:
            t["fail_to_success"].append(s)
            if ra["failure_class"] in ("target_exit_failure", "dwell_recovery_failure", "settling_failure"):
                settle_convert.append(s)
        elif pa and pb:
            t["success_to_success"].append(s)
        elif pa and not pb:
            t["success_to_fail"].append(s)
        elif ra["failure_class"] != rb["failure_class"]:
            t["fail_to_diff_failure"].append((s, ra["failure_class"], rb["failure_class"]))
    return t, settle_convert, a, b


def main():
    out = {"config_h15": H15["config"], "config_h30": H30["config"],
           "obs_contract_sha": H30["obs_contract_sha"], "bundle_hash": H30["bundle_hash"]}
    # consistency: instrumented H=15 baseline must reproduce the committed H=15 outcomes
    if COMMITTED:
        out["h15_baseline_reproduces_committed"] = {
            "headline": H15["headline"]["replay_certified_success"] == COMMITTED["headline"]["replay_certified_success"],
            "train_query": H15["train_query"]["replay_certified_success"] == COMMITTED["train_query"]["replay_certified_success"],
            "committed": [COMMITTED["headline"]["replay_certified_success"], COMMITTED["train_query"]["replay_certified_success"]],
            "instrumented": [H15["headline"]["replay_certified_success"], H15["train_query"]["replay_certified_success"]]}
    for bank in ("headline", "train_query"):
        t, settle_convert, a, b = transitions(bank)
        # §3 within-plan mechanism: plan_any_strict_frac at seeds that FAILED under H=15
        h15_fail = [s for s in a if not a[s]["replay_certified"]]
        mech = []
        for s in h15_fail:
            mech.append({"seed": s, "h15_fail": a[s]["failure_class"], "h15_maxK": a[s]["max_dwell"],
                         "h15_any_strict": a[s].get("plan_any_strict_frac"), "h30_any_strict": b[s].get("plan_any_strict_frac"),
                         "h30_maxK": b[s]["max_dwell"], "h30_result": b[s]["replay_certified"],
                         "h30_fail": b[s]["failure_class"]})
        tax15 = H15[bank]["failure_taxonomy"]
        tax30 = H30[bank]["failure_taxonomy"]
        out[bank] = {
            "h15_replay_cert": H15[bank]["replay_certified_success"], "h30_replay_cert": H30[bank]["replay_certified_success"],
            "h30_planning": H30[bank]["planning_success"],
            "planning_eq_replay": H30[bank]["planning_success"] == H30[bank]["replay_certified_success"],
            "transitions": {k: (len(v) if isinstance(v, list) else v) for k, v in t.items()},
            "transition_detail": t, "settle_stage_conversions": settle_convert,
            "taxonomy_h15": tax15, "taxonomy_h30": tax30,
            "target_exit_h15": tax15.get("target_exit_failure", 0), "target_exit_h30": tax30.get("target_exit_failure", 0),
            "dwell_recovery_h15": tax15.get("dwell_recovery_failure", 0), "dwell_recovery_h30": tax30.get("dwell_recovery_failure", 0),
            "mechanism_probe_h15_failures": mech}
    # §5 corrective gate
    hl, tq = out["headline"], out["train_query"]
    settle_conv = len(tq["settle_stage_conversions"])
    settle_before = tq["target_exit_h15"] + tq["dwell_recovery_h15"]
    settle_after = tq["target_exit_h30"] + tq["dwell_recovery_h30"]
    gate = {
        "headline_ge_6": H30["headline"]["replay_certified_success"] >= 6,
        "train_query_ge_18": H30["train_query"]["replay_certified_success"] >= 18,
        "planning_eq_replay": hl["planning_eq_replay"] and tq["planning_eq_replay"],
        "ge_3_settle_conversions": settle_conv >= 3,
        "lost_le_1_h15_success": tq["transitions"]["success_to_fail"] <= 1,
        "settle_failures_decrease": settle_after < settle_before,
    }
    out["gate"] = gate
    out["gate_all_pass"] = all(gate.values())
    out["settle_summary"] = {"conversions": settle_conv, "settle_fail_before": settle_before, "settle_fail_after": settle_after}
    json.dump(out, open(sys.argv[4] if len(sys.argv) > 4 else "/dev/stdout", "w"), indent=1)
    print(f"\nH15 -> H30: headline {out['headline']['h15_replay_cert']}->{out['headline']['h30_replay_cert']}/9  "
          f"train_query {out['train_query']['h15_replay_cert']}->{out['train_query']['h30_replay_cert']}/30")
    print(f"settle conversions {settle_conv}, settle-fails {settle_before}->{settle_after}, "
          f"lost H15 successes {tq['transitions']['success_to_fail']}")
    print(f"GATE {gate}  ALL_PASS={out['gate_all_pass']}")


if __name__ == "__main__":
    main()
