"""§3 bounded FEEDBACK-ONLY dataset generator (H=30 receding-horizon expert; NO open-loop suffix actions).

Runs the accepted canonical feedback expert (H=30, pop=40, iters=6, elite=8) from true neutral on the preregistered
pilot banks and stores, per replay-certified trajectory, the supervised labels in the frozen `FULL_ACTION_OBS_HISTORY_V1`
key: obs-history (152) → executed first action (4), plus phase, feedback-flag (E-approach prefix vs H=30 expert), and
the per-step planner strict-feasibility flag. Every label is a state-feedback action (E-approach or per-step-replanned
H=30 expert) — the quarantined open-loop CEM suffix actions never enter this dataset. Certification is neutral-reset
replay (no injection). Parallel + checkpointed.

Usage: python -m hymeko_rl.coin_delivery.coin_v3_feedback_dataset --bank both --workers 26 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.coin_delivery import coin_v3_seed_banks as sb
from hymeko_rl.coin_delivery.coin_v3_feedback_pilot import PILOT_TRAIN_QUERY
from hymeko_rl.coin_delivery.coin_v3_receding_horizon import receding_horizon_rollout, replay_certify
from hymeko_rl.coin_delivery.full_action_obs_history import contract_spec

_CFG = {"horizon": 30, "pop": 40, "iters": 6, "elite": 8, "plan_seed_base": 0, "max_steps": 360}


def _run_one(args_tuple):
    seed, outdir = args_tuple
    torch.set_num_threads(1)
    r = receding_horizon_rollout(seed, horizon=_CFG["horizon"], pop=_CFG["pop"], iters=_CFG["iters"],
                                 elite=_CFG["elite"], plan_seed_base=_CFG["plan_seed_base"], max_steps=_CFG["max_steps"])
    executed = r["executed"]
    cert = False
    if r["strict"]:
        cert = replay_certify(seed, executed, max_steps=_CFG["max_steps"])[0]
    if not cert:                                             # store ONLY replay-certified feedback trajectories
        return {"seed": int(seed), "certified": False, "strict": bool(r["strict"]), "steps": int(r["steps"]),
                "failure_class": r["failure_class"]}
    p = Path(outdir) / f"fb_{seed}.npz"
    np.savez_compressed(p, obs_hist=r["obs_hist"], act=executed, phase=r["phase"],
                        is_feedback=r["is_feedback"], plan_any_strict=r["plan_any_strict_step"])
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"seed": int(seed), "certified": True, "steps": int(r["steps"]),
            "n_feedback": int(r["is_feedback"].sum()), "n_approach": int((~r["is_feedback"]).sum()),
            "max_dwell": int(r["max_dwell"]), "sha256": sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=["headline", "train_query", "both"], default="both")
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = []
    if args.bank in ("headline", "both"):
        seeds += list(sb.HEADLINE)
    if args.bank in ("train_query", "both"):
        seeds += list(PILOT_TRAIN_QUERY)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    contract = contract_spec()
    print(f"[feedback-ds] bundle {sb.BUNDLE_HASH} obs {contract['name']} sha {contract['sha256'][:12]} | "
          f"{len(seeds)} states | cfg {_CFG}", flush=True)
    torch.set_num_threads(1)
    import multiprocessing as mp
    idx = []
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_run_one, [(s, str(out)) for s in seeds]):
            idx.append(r)
            print(f"  seed {r['seed']}: certified={r['certified']} "
                  f"{'feedback=' + str(r.get('n_feedback')) + ' approach=' + str(r.get('n_approach')) if r['certified'] else 'fail=' + str(r.get('failure_class'))}",
                  flush=True)
    n_cert = sum(x["certified"] for x in idx)
    n_fb = sum(x.get("n_feedback", 0) for x in idx)
    n_ap = sum(x.get("n_approach", 0) for x in idx)
    manifest = {"bank": args.bank, "bundle_hash": sb.BUNDLE_HASH, "obs_contract": contract["name"],
                "obs_contract_sha": contract["sha256"], "action_dim": 4, "feature_dim": contract["feature_dim"],
                "config": _CFG, "n_states": len(seeds), "n_certified_trajectories": n_cert,
                "n_feedback_labels": n_fb, "n_approach_labels": n_ap,
                "label_sources": {"e_approach": "state-feedback approach policy", "h30_expert": "per-step replanned H=30 receding-horizon"},
                "open_loop_suffix_labels": 0, "index": idx}
    (out / "feedback_dataset_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[feedback-ds] DONE: {n_cert}/{len(seeds)} certified | {n_fb} feedback + {n_ap} approach labels "
          f"| 0 open-loop suffix labels\nFEEDBACK_DATASET_DONE", flush=True)


if __name__ == "__main__":
    main()
