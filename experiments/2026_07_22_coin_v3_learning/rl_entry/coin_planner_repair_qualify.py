"""REPAIR_H30_PLANNER_OBJECTIVE_V1 qualification (no training). Re-runs the IDENTICAL 31-state teacher-qualification
harness with the REPAIRED feasibility-gated planner, measures BOTH contact-boundary definitions (A=stable_entry,
B=k6), freezes the physically correct one, and emits the step-9 verdict. No student/TD3/SAC/chunk/final-test.

Usage: python -m ...coin_planner_repair_qualify [--smoke N] [--workers W] [--floor F]
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_baseline_reconstruction import pi0_policy, qualify_teacher, rollout_from_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_planner_repair import (  # noqa: E402
    FeasibilityConfig,
    RepairedPlannerPolicy,
    final_qualification,
    repaired_first_action_stability,
)
from hymeko_rl.coin_delivery.coin_transport_dwell import CONTROL_MODES  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
PI0 = f"{D}/frozen/pi0_shared_clip_actor.pt"
TDCFG = f"{D}/transport_dwell_config.json"
OUT = f"{D}/planner_repair_qualify_v1.json"
CEM = {"horizon": 30, "pop": 40, "iters": 6, "elite": 8}      # kept unchanged (contract step 6)
BOUNDARIES = ("stable_entry", "k6")


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _eval_one(args):
    row, horizon, floor = args
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    ls = LateStart(seed=row[0], prefix_steps=row[1], family=row[2], obs_sha=row[3], base_sha=row[4], causal_sha=row[5])
    b = rollout_from_handoff(pi0, ls, pi0_policy(pi0), horizon=horizon)
    out = {}
    for boundary in BOUNDARIES:
        pol = RepairedPlannerPolicy(cfg=FeasibilityConfig(boundary=boundary, contact_floor=floor), **CEM)
        c = rollout_from_handoff(pi0, ls, pol, horizon=horizon)
        inf = float(np.mean(pol.infeasible_steps)) if pol.infeasible_steps else 0.0
        out[boundary] = {"metrics": c, "infeasible_freq": round(inf, 4)}
    return {"seed": ls.seed, "family": ls.family, "prefix": ls.prefix_steps}, b, out


def _rate(rows, k):
    return round(float(np.mean([int(r[k]) for r in rows])), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--floor", type=float, default=0.75)
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)
    cfg = json.load(open(TDCFG)); horizon = cfg["horizon"]
    dev = [ls for m in CONTROL_MODES for ls in _bank(cfg["banks"]["dev"][m])]
    if args.smoke:
        dev = dev[:args.smoke]
    log(f"[{time.strftime('%H:%M:%S')}] REPAIR qualify: {len(dev)} dev states, CEM {CEM}, contact_floor {args.floor}")

    t = time.time()
    payload = [(ls.to_row(), horizon, args.floor) for ls in dev]
    if args.workers > 1 and len(dev) > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            results = pool.map(_eval_one, payload)
    else:
        results = [_eval_one(p) for p in payload]
    meta, b_rows, per = zip(*results)
    b_rows = list(b_rows)
    log(f"  ({time.time()-t:.0f}s) B(pi_0)+C'(repaired ×2 boundaries) done")

    boundary_summ = {}
    for boundary in BOUNDARIES:
        c_rows = [p[boundary]["metrics"] for p in per]
        inf = round(float(np.mean([p[boundary]["infeasible_freq"] for p in per])), 4)
        fq = final_qualification(b_rows, c_rows)
        boundary_summ[boundary] = {"final": fq, "frozen_qualify_teacher": qualify_teacher(b_rows, c_rows),
                                   "all_candidates_infeasible_freq": inf,
                                   "exit_before_k6_rate": _rate(c_rows, "exit_before_k6"),
                                   "strict_rate": _rate(c_rows, "strict_success"),
                                   "req_contact": round(float(np.mean([r["required_contact_retention"] for r in c_rows])), 4)}
        log(f"  [{boundary}] strict {boundary_summ[boundary]['strict_rate']} req_contact {boundary_summ[boundary]['req_contact']} "
            f"exit<K6 {boundary_summ[boundary]['exit_before_k6_rate']} infeasible {inf}  → {fq['verdict']}")

    # Freeze the physically-correct boundary: stable_entry is the minimal physical requirement (release after settling is
    # legal); freeze the stricter k6 ONLY if releasing at stable_entry demonstrably causes MORE premature exits before K6.
    exit_A, exit_B = boundary_summ["stable_entry"]["exit_before_k6_rate"], boundary_summ["k6"]["exit_before_k6_rate"]
    frozen = "k6" if (exit_A - exit_B) > 0.05 else "stable_entry"
    log(f"\n  boundary evidence: exit<K6 stable_entry {exit_A} vs k6 {exit_B} (Δ {round(exit_A-exit_B,4)}) → FROZEN = {frozen}")

    # first-action stability of the repaired planner under the frozen boundary (a few dev states)
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    stab = []
    for ls in dev[:min(3, len(dev))]:
        rl, _g, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
        stab.append({"seed": ls.seed, "family": ls.family,
                     **repaired_first_action_stability(rl, FeasibilityConfig(boundary=frozen, contact_floor=args.floor), n_seeds=6, **CEM)})
    log(f"  first-action stability (repaired, {frozen}): " + ", ".join(f"{s['family']}:cos={s['pairwise_cosine']}" for s in stab))

    final = boundary_summ[frozen]["final"]
    out = {"campaign": "REPAIR_H30_PLANNER_OBJECTIVE_V1", "date": "2026-07-23", "no_training": True, "smoke": bool(args.smoke),
           "kept_unchanged": {**CEM, "reward": "galambos_task_deliver_v3", "certifier": "strict-K6", "harness": "31-state qualify_teacher"},
           "contact_floor": args.floor, "boundary_measurement": {b: {k: v for k, v in boundary_summ[b].items() if k != "frozen_qualify_teacher"} for b in BOUNDARIES},
           "frozen_boundary": frozen, "frozen_boundary_evidence": {"exit_before_k6_stable_entry": exit_A, "exit_before_k6_k6": exit_B},
           "first_action_stability": stab, "final_qualification": final,
           "pi0_reference": {"strict_rate": _rate(b_rows, "strict_success"),
                             "req_contact": round(float(np.mean([r["required_contact_retention"] for r in b_rows])), 4),
                             "exit_before_k6": _rate(b_rows, "exit_before_k6")},
           "per_state": [{**m, "pi0": b, "repaired": {bd: per[i][bd] for bd in BOUNDARIES}} for i, (m, b) in enumerate(zip(meta, b_rows))],
           "verdict": final["verdict"]}
    outp = OUT if not args.smoke else OUT.replace(".json", "_smoke.json")
    json.dump(out, open(outp, "w"), indent=1, default=float)

    ag = final["aggregate"]
    log(f"\n== FINAL (frozen boundary={frozen}) {final['clauses']} ==")
    log(f"  req contact: repaired {ag['planner_req_contact']} vs pi_0 {ag['pi0_req_contact']} (Δ {ag['mean_d_contact_retention']}, new losses {ag['new_required_contact_losses']})")
    log(f"  exit<K6: repaired {ag['planner_exit_before_k6']} vs pi_0 {ag['pi0_exit_before_k6']} (Δ {ag['d_exit_before_k6_rate']})")
    log(f"  strict: repaired {ag['planner_strict_rate']} vs pi_0 {ag['pi0_strict_rate']} (Δ {ag['d_strict_rate']})  dwell Δ {ag['mean_d_dwell']}")
    log(f"\n→ {final['verdict']}\nwrote {outp}\nREPAIR_QUALIFY_DONE")


if __name__ == "__main__":
    main()
