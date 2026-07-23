"""COIN_FEEDBACK_BASELINE_RECONSTRUCTION_V1 (no training). Freezes the completed arc as historical evidence, verifies
controller/checkpoint identities, evaluates the four gold controllers in one harness, and runs the load-bearing
FULL-HORIZON H=30 teacher qualification. Emits the reconstruction JSON + verdict + the single next command.

Usage:  python -m ...coin_baseline_reconstruction [--smoke N] [--workers W]
  --smoke N : run only the first N dev states of B/C (production-scale timing before the full fan-out).
"""
import argparse
import hashlib
import json
import sys
import time

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_baseline_reconstruction import (  # noqa: E402
    CANONICAL_H30,
    ELEVEN_METRICS,
    pi0_policy,
    planner_policy,
    qualify_teacher,
    rollout_from_handoff,
)
from hymeko_rl.coin_delivery.coin_late_start import LateStart  # noqa: E402
from hymeko_rl.coin_delivery.coin_rl_env import CANONICAL_REWARD_FILE, HELD_DWELL  # noqa: E402
from hymeko_rl.coin_delivery.coin_transport_dwell import CONTROL_MODES  # noqa: E402
from hymeko_rl.coin_delivery.delivery_certificate import DeliveryThresholds  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
PI0 = f"{D}/frozen/pi0_shared_clip_actor.pt"
E_APPROACH = "experiments/2026_07_08_seed_stabilized/E_valselect_v2.pt"
TDCFG = f"{D}/transport_dwell_config.json"
D_PILOT = "experiments/2026_07_22_coin_v3_learning/receding_horizon/feedback_pilot_h30_result.json"
OUT = f"{D}/baseline_reconstruction_v1.json"

FROZEN_ARC = [
    ("PHASE_GATED_RESIDUAL_CRITIC_ROUTE_BLOCKED", "reports/2026-07-23-coin-push-delivery-evening-v2.json"),
    ("HOLD_SIGNAL_DOMINATED_BY_HARM", f"{D}/beneficial_support_audit_v1.json"),
    ("PHASE_SWITCHED_TD3_STAGE1_NO_IMPROVEMENT", f"{D}/td3_baseline_v1_results.json"),
    ("PHASE_SWITCHED_TD3_STAGE1B_NO_IMPROVEMENT", f"{D}/td3_stage1b_results.json"),
    ("PHASE_SWITCHED_TD3_STAGE1C_NO_IMPROVEMENT", f"{D}/td3_stage1c_results.json"),
    ("TRANSPORT_DWELL_TD3_NO_IMPROVEMENT", f"{D}/transport_dwell_results.json"),
    ("FEEDBACK_CHUNK_WARMSTART_V2_STILL_UNDERPERFORMS", f"{D}/feedback_chunk_warmstart_v2.json"),
    ("CHUNK_SUPERVISED_M1_FEEDBACK_NO_GAIN", f"{D}/chunk_m1_diagnostic.json"),
]


def _sha256(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except OSError:
        return None


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def freeze_arc():
    """Step 1 — record the accepted outcomes as read-only historical evidence (artifact path + sha; never modified)."""
    out = []
    for tok, path in FROZEN_ARC:
        sha = _sha256(path)
        out.append({"verdict": tok, "artifact": path, "exists": sha is not None, "sha256_16": sha[:16] if sha else None})
    return out


def verify_identities(cfg):
    """Step 3 — exact checkpoint/reward/certifier/seed-bank identities."""
    return {
        "pi0": {"path": PI0, "sha256": _sha256(PI0), "expected_prefix": "1902454c", "role": "frozen deployed late controller B"},
        "e_approach": {"path": E_APPROACH, "sha256": _sha256(E_APPROACH), "expected_prefix": "7dbbf1a7", "role": "recovered frozen approach A"},
        "planner_config": CANONICAL_H30 | {"scorer": "lexo strict>dwell>-min_dtz>-min_speed>-effort (NO contact term)"},
        "reward": {"file": CANONICAL_REWARD_FILE, "sha256_16": (_sha256(CANONICAL_REWARD_FILE) or "")[:16], "held_dwell": HELD_DWELL},
        "certifier": {"center_tol": DeliveryThresholds().center_tol, "settle_vel": DeliveryThresholds().settle_vel,
                      "dwell_req": DeliveryThresholds().dwell_req, "grading": "CoinRL4Dof._strict>=6 ∧ touched"},
        "seed_banks": {m: {"n": cfg["banks"]["dev"][m]["n"], "sha16": cfg["banks"]["dev"][m]["sha16"]} for m in CONTROL_MODES},
        "d_pilot": {"path": D_PILOT, "sha256_16": (_sha256(D_PILOT) or "")[:16]},
    }


def controller_A(seeds):
    """Recovered frozen E-approach from neutral: approach/grasp competence only (first contact / bilateral grasp)."""
    import numpy as np
    import torch

    from hymeko_rl.experiments.coin_neutral_start import _e_approach_actor, neutral_env
    e = _e_approach_actor(); env, cf = neutral_env(prefix_steps=0); inner = cf._env
    fc = bi = 0
    for s in seeds:
        env.set_stage(0); env.reset(seed=int(s)); got_fc = got_bi = False
        for _k in range(160):
            m = inner._planar_metrics
            got_fc = got_fc or bool(m.left_contact or m.right_contact); got_bi = got_bi or bool(m.left_contact and m.right_contact)
            if got_bi:
                break
            with torch.no_grad():
                a = e.action_mean(torch.as_tensor(np.asarray(inner.node_features(), np.float32)[None]))[0].numpy()
            inner.step(np.asarray(a, np.float32))
        fc += got_fc; bi += got_bi
    return {"start": "neutral", "n": len(seeds), "first_contact": fc, "bilateral_grasp": bi, "scope": "approach-only"}


def _eval_one(args):
    """Worker: reconstruct a dev late-start; roll B (pi_0) and C (H=30 planner) for the full horizon; return metrics."""
    row, horizon, cem = args
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    ls = LateStart(seed=row[0], prefix_steps=row[1], family=row[2], obs_sha=row[3], base_sha=row[4], causal_sha=row[5])
    b = rollout_from_handoff(pi0, ls, pi0_policy(pi0), horizon=horizon)
    c = rollout_from_handoff(pi0, ls, planner_policy(plan_seed_base=0, **cem), horizon=horizon)
    return {"seed": ls.seed, "family": ls.family, "prefix": ls.prefix_steps}, b, c


def _mean_row(rows, keys):
    import numpy as np
    return {k: round(float(np.mean([int(r[k]) if isinstance(r[k], bool) else (r[k] or 0.0) for r in rows])), 4) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)
    cfg = json.load(open(TDCFG)); horizon = cfg["horizon"]
    dev = [ls for m in CONTROL_MODES for ls in _bank(cfg["banks"]["dev"][m])]
    if args.smoke:
        dev = dev[:args.smoke]
    log(f"[{time.strftime('%H:%M:%S')}] dev states: {len(dev)}  horizon: {horizon}  H30: {CANONICAL_H30}")

    ident = verify_identities(cfg)
    log(f"  pi0 {ident['pi0']['sha256'][:8]} (exp 1902454c)  E {ident['e_approach']['sha256'][:8]} (exp 7dbbf1a7)")
    arc = freeze_arc(); log(f"  froze {sum(a['exists'] for a in arc)}/{len(arc)} arc artifacts")

    log(f"[{time.strftime('%H:%M:%S')}] controller A (E-approach, neutral)...")
    a_res = controller_A([1011, 1045, 1164, 1174, 1202, 1278, 1358, 1447, 1568])
    log(f"  A: first_contact {a_res['first_contact']}/9  bilateral {a_res['bilateral_grasp']}/9")

    log(f"[{time.strftime('%H:%M:%S')}] controllers B (pi_0) + C (H=30 planner) on {len(dev)} dev states "
        f"({args.workers} workers)... this is the load-bearing run")
    t = time.time()
    payload = [(ls.to_row(), horizon, CANONICAL_H30) for ls in dev]
    if args.workers > 1 and len(dev) > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            results = pool.map(_eval_one, payload)
    else:
        results = [_eval_one(p) for p in payload]
    meta, b_rows, c_rows = zip(*results)
    log(f"  ({time.time()-t:.0f}s) B/C rollouts done")

    qual = qualify_teacher(list(b_rows), list(c_rows))
    keys = list(ELEVEN_METRICS) + ["progress", "lost_required_contact"]
    gold = {"A_e_approach": a_res,
            "B_pi0": {"start": "dev-handoff", "n": len(b_rows), **_mean_row(b_rows, keys)},
            "C_h30_planner": {"start": "dev-handoff", "n": len(c_rows), **_mean_row(c_rows, keys)},
            "D_composed_h30": {"start": "neutral", "source": "read from feedback_pilot_h30 (not re-run)"}}
    try:
        dp = json.load(open(D_PILOT))["headline"]
        gold["D_composed_h30"].update({"n": dp["n"], "delivered": dp["planning_success"],
                                       "contact_loss_after_acq": dp["contact_loss_after_acq_count"],
                                       "target_exit_after_entry": dp["target_exit_after_entry_count"]})
    except (OSError, KeyError):
        pass

    next_cmd = ("STEP 7 — REPAIR the H=30 planner objective (lexicographic/constrained: 1 preserve required contact, "
                "2 prevent target exit, 3 progress, 4 brake, 5 settle+K6), then re-evaluate the planner itself until it "
                "qualifies. DO NOT train a student.") if not qual["qualified"] else \
               ("STEP 8 — build PHASE_CONDITIONED_FIRST_ACTION_DAGGER_V1 (single 4D action, per-step planner query in "
                "TRAINING ONLY, no chunks, no open-loop M) and require it to reproduce the qualified teacher on disjoint "
                "dev states before any RL.")

    out = {"campaign": "COIN_FEEDBACK_BASELINE_RECONSTRUCTION_V1", "date": "2026-07-23", "no_training": True,
           "smoke": bool(args.smoke), "horizon": horizon, "frozen_arc": arc, "identities": ident,
           "gold_baseline": gold, "teacher_qualification": qual,
           "per_state": [{**m, "pi0": b, "planner": c} for m, b, c in zip(meta, b_rows, c_rows)],
           "verdict": qual["verdict"], "next_command": next_cmd}
    json.dump(out, open(OUT if not args.smoke else OUT.replace(".json", "_smoke.json"), "w"), indent=1, default=float)

    ag = qual["aggregate"]
    log(f"\n== TEACHER QUALIFICATION ({qual['clauses']}) ==")
    log(f"  required contact: C {ag['planner_req_contact']} vs pi_0 {ag['pi0_req_contact']}  (Δ {ag['mean_d_contact_retention']}, "
        f"new losses {ag['new_required_contact_losses']})")
    log(f"  strict: C {ag['planner_strict_rate']} vs pi_0 {ag['pi0_strict_rate']}   exit: C {ag['planner_exit_rate']} vs pi_0 {ag['pi0_exit_rate']}")
    log(f"  dwell Δ {ag['mean_d_dwell']}  return Δ {ag['mean_d_return']}")
    log(f"\n→ {qual['verdict']}\nNEXT: {next_cmd}\nwrote {OUT if not args.smoke else OUT.replace('.json','_smoke.json')}\nRECONSTRUCTION_DONE")


if __name__ == "__main__":
    main()
