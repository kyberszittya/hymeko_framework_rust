"""§5 bounded receding-horizon feedback-expert PILOT (no RL, no learner training, no recovery demos).

Runs the closed-loop receding-horizon rollout (frozen E-approach → per-step CEM replanning) from true canonical
neutral on the frozen headline (9) + a preregistered train_query subset (30). Every apparent success is
REPLAY-CERTIFIED from neutral (replay the executed action sequence with NO replanning). Planning success and
replay-certified success are reported separately. Failures are classified into the §5 taxonomy. The gate:
``RECEDING_HORIZON_FEEDBACK_EXPERT_PASS`` iff replay-certified ≥6/9 headline AND ≥18/30 train_query.

Usage: python -m hymeko_rl.coin_delivery.coin_v3_feedback_pilot --bank both --workers 15 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from hymeko_rl.coin_delivery import coin_v3_seed_banks as sb
from hymeko_rl.coin_delivery.coin_v3_receding_horizon import receding_horizon_rollout, replay_certify
from hymeko_rl.coin_delivery.full_action_obs_history import contract_spec

PILOT_TRAIN_QUERY = tuple(range(6000, 6030))     # frozen preregistered 30-state train_query pilot subset


def _bank_hash(seeds):
    return hashlib.sha256(json.dumps(list(seeds)).encode()).hexdigest()[:16]


_CFG = {}   # populated by main; read by workers


def _run_one(seed):
    torch.set_num_threads(1)
    r = receding_horizon_rollout(seed, horizon=_CFG["horizon"], pop=_CFG["pop"], iters=_CFG["iters"],
                                 elite=_CFG["elite"], plan_seed_base=_CFG["plan_seed_base"],
                                 max_steps=_CFG["max_steps"])
    executed = r.pop("executed")
    r.pop("dwell_seq", None)
    if r["strict"]:                                          # replay-certify apparent successes from neutral
        cert, cstep, md = replay_certify(seed, executed, max_steps=_CFG["max_steps"])
        r["replay_certified"] = bool(cert)
        r["replay_cert_step"] = int(cstep)
        r["replay_max_dwell"] = int(md)
        if not cert:
            r["failure_class"] = "replay_nondeterminism"
    else:
        r["replay_certified"] = False
    return r


def _summarize(rows, bank_name, seeds, thresh):
    plan_success = sum(r["strict"] for r in rows)
    replay_success = sum(r["replay_certified"] for r in rows)
    taxonomy: dict = {}
    for r in rows:
        if not r["replay_certified"]:
            fc = r["failure_class"] or "unknown"
            taxonomy[fc] = taxonomy.get(fc, 0) + 1
    contact_loss = sum(r["contact_loss_after_acq"] for r in rows)
    target_exit = sum(r["target_exit_after_entry"] for r in rows)
    return {"bank": bank_name, "n": len(seeds), "bank_sha16": _bank_hash(seeds),
            "planning_success": plan_success, "replay_certified_success": replay_success,
            "threshold": thresh, "meets_threshold": replay_success >= thresh,
            "failure_taxonomy": taxonomy,
            "contact_loss_after_acq_count": contact_loss, "target_exit_after_entry_count": target_exit,
            "delivered_seeds": [r["seed"] for r in rows if r["replay_certified"]],
            "rows": rows}


def main():
    global _CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=["headline", "train_query", "both"], default="both")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--plan-seed-base", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=360)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    _CFG = {"horizon": args.horizon, "pop": args.pop, "iters": args.iters, "elite": args.elite,
            "plan_seed_base": args.plan_seed_base, "max_steps": args.max_steps}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    banks = {}
    if args.bank in ("headline", "both"):
        banks["headline"] = (sb.HEADLINE, 6)
    if args.bank in ("train_query", "both"):
        banks["train_query"] = (PILOT_TRAIN_QUERY, 18)
    import multiprocessing as mp
    torch.set_num_threads(1)
    contract = contract_spec()
    print(f"[pilot] bundle {sb.BUNDLE_HASH} | obs {contract['name']} sha {contract['sha256'][:12]} | "
          f"cfg {_CFG}", flush=True)
    summaries = {}
    for name, (seeds, thresh) in banks.items():
        print(f"[{name}] {len(seeds)} states, threshold {thresh}, bank_sha {_bank_hash(seeds)}", flush=True)
        with mp.Pool(args.workers) as pool:
            rows = []
            for r in pool.imap_unordered(_run_one, list(seeds)):
                rows.append(r)
                print(f"  seed {r['seed']}: strict={r['strict']} replay_cert={r['replay_certified']} "
                      f"reason={r['reason']} maxK={r['max_dwell']} fail={r['failure_class']} "
                      f"lat={r['mean_plan_latency']}s", flush=True)
        rows.sort(key=lambda r: r["seed"])
        summaries[name] = _summarize(rows, name, seeds, thresh)
        s = summaries[name]
        print(f"[{name}] plan {s['planning_success']}/{s['n']}  replay-cert {s['replay_certified_success']}/{s['n']} "
              f"(>= {thresh}? {s['meets_threshold']})  taxonomy {s['failure_taxonomy']}", flush=True)
    # decision rule
    hl = summaries.get("headline")
    tq = summaries.get("train_query")
    passed = all(summaries[k]["meets_threshold"] for k in summaries)
    verdict = "RECEDING_HORIZON_FEEDBACK_EXPERT_PASS" if (passed and args.bank == "both") else \
              ("RECEDING_HORIZON_FEEDBACK_EXPERT_FAIL" if args.bank == "both" else "PARTIAL_BANK_RUN")
    result = {"verdict": verdict, "bundle_hash": sb.BUNDLE_HASH, "obs_contract_sha": contract["sha256"],
              "config": _CFG, "rl_started": False,
              "headline": hl, "train_query": tq}
    json.dump(result, open(out / "pilot_result.json", "w"), indent=1)
    print(f"\n{verdict}  headline {hl['replay_certified_success'] if hl else '-'}/9  "
          f"train_query {tq['replay_certified_success'] if tq else '-'}/30  RL_started=False", flush=True)


if __name__ == "__main__":
    main()
