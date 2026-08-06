"""CONTACT_STABILIZED_PRIMITIVE_MPC_V1 — one bounded 31-state development evaluation (no training). Compares A frozen
pi_0, B raw-action H=30 CEM planner (read from the baseline reconstruction — cached, not re-run), C primitive MPC.
Emits PRIMITIVE_MPC_QUALIFIED or a named mechanism. No student/TD3/SAC/PPO; no final-test seeds.

Usage: python -m ...coin_primitive_mpc_qualify [--smoke N] [--workers W]
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_baseline_reconstruction import pi0_policy, rollout_from_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_planner_repair import final_qualification  # noqa: E402
from hymeko_rl.coin_delivery.coin_primitive_mpc import (  # noqa: E402
    PrimitiveBounds,
    PrimitiveMPCPolicy,
    first_primitive_stability,
)
from hymeko_rl.coin_delivery.coin_transport_dwell import CONTROL_MODES  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
PI0 = f"{D}/frozen/pi0_shared_clip_actor.pt"
TDCFG = f"{D}/transport_dwell_config.json"
RECON = f"{D}/baseline_reconstruction_v1.json"
OUT = f"{D}/primitive_mpc_qualify_v1.json"
CEM = {"pop": 40, "iters": 6, "elite": 8, "horizon": 60}         # frozen (contract step 5); θ dim 10, replan every 4
BOUNDS = PrimitiveBounds()


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _eval_one(args):
    import torch
    torch.set_num_threads(1)                                     # avoid BLAS oversubscription across pool workers
    row, horizon = args
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    ls = LateStart(seed=row[0], prefix_steps=row[1], family=row[2], obs_sha=row[3], base_sha=row[4], causal_sha=row[5])
    a = rollout_from_handoff(pi0, ls, pi0_policy(pi0), horizon=horizon)
    pol = PrimitiveMPCPolicy(BOUNDS, pi0, plan_seed_base=0, replan_every=4, **CEM)
    c = rollout_from_handoff(pi0, ls, pol, horizon=horizon)
    return {"seed": ls.seed, "family": ls.family, "prefix": ls.prefix_steps}, a, c, pol.stats()


def _rate(rows, k):
    return round(float(np.mean([int(r[k]) for r in rows])), 4)


def _mean(vals):
    return round(float(np.mean(vals)), 4) if vals else 0.0


def _run_states(dev_states, horizon, log):
    """Serial (no multiprocessing) with per-state live progress — the pool proved unreliable for this heavy workload."""
    out = []
    for j, ls in enumerate(dev_states):
        st = time.time(); r = _eval_one((ls.to_row(), horizon)); out.append(r)
        log(f"    [{time.strftime('%H:%M:%S')}] {j+1}/{len(dev_states)} seed {r[0]['seed']} ({r[0]['family']}) "
            f"C strict={int(r[2]['strict_success'])} contact={r[2]['required_contact_retention']} "
            f"exit<K6={int(r[2]['exit_before_k6'])} fb={r[3]['guard_fallback_rate']} ({time.time()-st:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--islice", type=int, default=-1); ap.add_argument("--nslice", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)
    cfg = json.load(open(TDCFG)); horizon = cfg["horizon"]
    dev = [ls for m in CONTROL_MODES for ls in _bank(cfg["banks"]["dev"][m])]
    if args.smoke:
        dev = dev[:args.smoke]

    # slice mode: process dev[islice::nslice] serially in an independent process, dump partial → robust vs pool crashes
    if args.islice >= 0 and not args.merge:
        idxs = list(range(len(dev)))[args.islice::args.nslice]
        log(f"[{time.strftime('%H:%M:%S')}] SLICE {args.islice}/{args.nslice}: {len(idxs)} states {idxs}")
        results = _run_states([dev[i] for i in idxs], horizon, log)
        json.dump({"idxs": idxs, "results": results}, open(f"{D}/primitive_mpc_slice_{args.islice}.json", "w"), default=float)
        log(f"SLICE_{args.islice}_DONE"); return

    log(f"[{time.strftime('%H:%M:%S')}] PRIMITIVE MPC: {len(dev)} dev states, θ-dim 10, CEM {CEM}, replan every 4, guard 2-step")
    t = time.time()
    if args.merge:                                              # reassemble slices (written by the parallel slice jobs)
        pairs = []
        for i in range(args.nslice):
            sl = json.load(open(f"{D}/primitive_mpc_slice_{i}.json"))
            pairs += list(zip(sl["idxs"], sl["results"]))
        results = [r for _i, r in sorted(pairs)]
        log(f"  merged {len(results)} states from {args.nslice} slices")
    else:
        results = _run_states(dev, horizon, log)
    meta, a_rows, c_rows, cstats = zip(*results)
    a_rows, c_rows = list(a_rows), list(c_rows)
    log(f"  ({time.time()-t:.0f}s) A(pi_0)+C(primitive) assembled")

    # B: raw H=30 planner — read the cached per-state rows from the reconstruction (not re-run)
    b_rows = None
    try:
        recon = json.load(open(RECON))["per_state"]
        key = {(s["seed"], s["prefix"]): s["planner"] for s in recon}
        b_rows = [key.get((m["seed"], m["prefix"])) for m in meta]
        b_rows = b_rows if all(b_rows) else None
    except (OSError, KeyError):
        pass

    fq = final_qualification(a_rows, c_rows)                     # contact / exit-before-K6 / advantage clauses
    mean_fallback = _mean([s["guard_fallback_rate"] for s in cstats])
    mean_interv = _mean([s["guard_intervention_rate"] for s in cstats])
    guard_ok = mean_fallback < 0.9                              # step-10: must not depend almost entirely on α=0 fallback
    # first-primitive stability on a few states
    pi0 = load_frozen_clip_actor(PI0, freeze=True); stab = []
    for ls in dev[:min(3, len(dev))]:
        rl, _g, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
        stab.append({"seed": ls.seed, "family": ls.family,
                     **first_primitive_stability(rl, BOUNDS, pi0, n_seeds=6, **CEM)})
    theta_std = _mean([s["theta_std_norm"] for s in stab]); cem_stable = theta_std < 0.25

    clauses = {**fq["clauses"], "not_fallback_dependent": guard_ok, "cem_stable": cem_stable}
    qualified = fq["qualified"] and guard_ok and cem_stable
    reasons = []
    if not guard_ok:
        reasons.append("PRIMITIVE_GUARD_ALWAYS_FALLS_BACK")
    if not fq["clauses"]["contact"]:
        reasons.append("PRIMITIVE_MPC_LOSES_REQUIRED_CONTACT")
    if not fq["clauses"]["advantage"]:
        reasons.append("PRIMITIVE_SPACE_NO_BENEFICIAL_SUPPORT" if guard_ok else "PRIMITIVE_MPC_NO_TASK_GAIN")
    if not cem_stable:
        reasons.append("PRIMITIVE_CEM_UNSTABLE")
    verdict = "PRIMITIVE_MPC_QUALIFIED" if qualified else "PRIMITIVE_MPC_UNQUALIFIED:" + ",".join(reasons or ["EXIT_BEFORE_K6"])

    def occ(k):
        return {m: _mean([s["mode_occupancy"][m] for s in cstats]) for m in ("PUSH", "BRAKE", "SETTLE")}[k]
    gold = {
        "A_pi0": {"strict": _rate(a_rows, "strict_success"), "req_contact": _mean([r["required_contact_retention"] for r in a_rows]),
                  "exit_before_k6": _rate(a_rows, "exit_before_k6"),
                  "max_dwell": _mean([r["max_dwell"] for r in a_rows]), "entry_velocity": _mean([r["entry_velocity"] or 0 for r in a_rows]),
                  "total_return": _mean([r["total_return"] for r in a_rows])},
        "C_primitive": {"strict": _rate(c_rows, "strict_success"), "req_contact": _mean([r["required_contact_retention"] for r in c_rows]),
                        "exit_before_k6": _rate(c_rows, "exit_before_k6"), "max_dwell": _mean([r["max_dwell"] for r in c_rows]),
                        "entry_velocity": _mean([r["entry_velocity"] or 0 for r in c_rows]), "braking": _mean([r["braking"] for r in c_rows]),
                        "total_return": _mean([r["total_return"] for r in c_rows]),
                        "mode_occupancy": {m: occ(m) for m in ("PUSH", "BRAKE", "SETTLE")},
                        "guard_intervention_rate": mean_interv, "guard_fallback_rate": mean_fallback,
                        "all_fallback_plan_rate": _mean([s["all_fallback_plan_rate"] for s in cstats])}}
    if b_rows:
        gold["B_raw_h30"] = {"strict": _rate(b_rows, "strict_success"), "req_contact": _mean([r["required_contact_retention"] for r in b_rows]),
                             "target_exit": _rate(b_rows, "target_exit"), "max_dwell": _mean([r["max_dwell"] for r in b_rows]),
                             "source": "cached from baseline_reconstruction_v1 (not re-run; pre-dates exit_before_k6 metric)"}

    out = {"campaign": "CONTACT_STABILIZED_PRIMITIVE_MPC_V1", "date": "2026-07-23", "no_training": True, "smoke": bool(args.smoke),
           "theta_dim": 10, "cem": CEM, "replan_every": 4, "guard_horizon": 2, "bounds": {"lo": BOUNDS.lo, "hi": BOUNDS.hi},
           "gold_baseline": gold, "final_qualification": fq, "clauses": clauses, "first_primitive_stability": stab,
           "mean_theta_std_norm": theta_std, "verdict": verdict,
           "per_state": [{**m, "pi0": a, "primitive": c, "primitive_stats": st} for m, a, c, st in zip(meta, a_rows, c_rows, cstats)]}
    outp = OUT if not args.smoke else OUT.replace(".json", "_smoke.json")
    json.dump(out, open(outp, "w"), indent=1, default=float)

    ag = fq["aggregate"]
    log(f"\n== PRIMITIVE MPC {clauses} ==")
    log(f"  req contact: C {ag['planner_req_contact']} vs pi_0 {ag['pi0_req_contact']} (Δ {ag['mean_d_contact_retention']}, new losses {ag['new_required_contact_losses']})")
    log(f"  exit<K6: C {ag['planner_exit_before_k6']} vs pi_0 {ag['pi0_exit_before_k6']} (Δ {ag['d_exit_before_k6_rate']})")
    log(f"  strict: C {ag['planner_strict_rate']} vs pi_0 {ag['pi0_strict_rate']} (Δ {ag['d_strict_rate']})  dwell Δ {ag['mean_d_dwell']}")
    log(f"  mode occ {gold['C_primitive']['mode_occupancy']}  guard interv {mean_interv} fallback {mean_fallback}  θ-std/range {theta_std}")
    log(f"\n→ {verdict}\nwrote {outp}\nPRIMITIVE_MPC_DONE")


if __name__ == "__main__":
    main()
