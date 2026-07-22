"""§5-7 / §8-10: phase-balanced full-action BC training + standalone deployment eval.

Trajectory-level train/val-loss split (never split one trajectory). Optionally MERGES DAgger corrective labels
(base ∪ dagger). Trains several seeds, then deploys each BC from NEUTRAL (u=policy(obs), no teacher) and grades
strict K=6 on the headline panel; the best BC is also evaluated on the VALIDATION bank.

Usage: python train_full_action_bc.py BASE_DIR OUT [--dagger DAGGER_DIR] [--seeds 0 1 2] [--eval-validation]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE, VALIDATION
from hymeko_rl.coin_delivery.full_action_bc import (
    eval_bc_delivery, eval_framestack_bc_delivery, load_trajectory_dataset, stack_frames, train_bc,
    train_bc_phase_balanced,
)

PHASES = ["APPROACH", "CONTACT_ACQUISITION", "BILATERAL_OR_STABLE_CONTACT", "TRANSPORT",
          "TARGET_ENTRY", "SETTLING", "STRICT_DWELL"]


def traj_split(traj, n_val=10, seed=0):
    ids = np.unique(traj)
    rng = np.random.default_rng(seed)
    val_ids = set(rng.choice(ids, size=min(n_val, len(ids) - 1), replace=False).tolist())
    val_mask = np.isin(traj, list(val_ids))
    return ~val_mask, val_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("out")
    ap.add_argument("--dagger", default=None, help="DAgger label dir to merge (dagger_*.npz)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-validation", action="store_true")
    ap.add_argument("--uniform", action="store_true", help="uniform sampling (no phase balancing) — ablation")
    ap.add_argument("--framestack", type=int, default=1, help="k-frame stacking (recovers coin velocity); 1 = reactive")
    args = ap.parse_args()

    dirs = [args.base_dir] + ([args.dagger] if args.dagger else [])
    patterns = ("traj_*.npz", "dagger_*.npz") if args.dagger else ("traj_*.npz",)
    d = load_trajectory_dataset(dirs, patterns=patterns)
    obs, act, phase, traj = d["obs"], d["act"], d["phase"], d["traj"]
    k = args.framestack
    if k > 1:
        obs = stack_frames(obs, traj, k)
        print(f"frame-stack k={k}: obs_dim -> {obs.shape[1]}", flush=True)
    tr_m, va_m = traj_split(traj, n_val=10, seed=0)
    print(f"dataset: {d['n_traj']} traj ({'base+dagger' if args.dagger else 'base'}), {len(obs)} transitions | "
          f"train {int(tr_m.sum())} / val {int(va_m.sum())}", flush=True)

    results = []
    for seed in args.seeds:
        if args.uniform:
            bc, hist = train_bc(obs[tr_m], act[tr_m], epochs=300, lr=1e-3, batch=256, seed=seed,
                                val=(obs[va_m], act[va_m]))
        else:
            bc, hist = train_bc_phase_balanced(
                obs[tr_m], act[tr_m], phase[tr_m], epochs=300, lr=1e-3, batch=256, seed=seed,
                steps_per_epoch=200, val=(obs[va_m], act[va_m], phase[va_m]))
        vloss = hist["val"][-1]
        ev = (eval_framestack_bc_delivery(bc, HEADLINE, k=k, horizon=360) if k > 1
              else eval_bc_delivery(bc, HEADLINE, horizon=360))
        print(f"seed {seed}: val {vloss:.2e} | HEADLINE fc {ev['first_contact']}/9 grasp {ev['grasp']}/9 "
              f"DELIVER {ev['deliver']}/9 {ev['delivered_seeds']}", flush=True)
        torch.save(bc.state_dict(), f"{args.out}/bc_seed{seed}.pt")
        results.append({"seed": seed, "val_loss": vloss, "headline_deliver": ev["deliver"],
                        "delivered_seeds": ev["delivered_seeds"]})

    best = max(results, key=lambda r: (r["headline_deliver"], -r["val_loss"]))
    print(f"\nBEST: seed {best['seed']} HEADLINE {best['headline_deliver']}/9 val {best['val_loss']:.2e}", flush=True)
    out = {"n_traj": d["n_traj"], "merged": bool(args.dagger), "results": results, "best_seed": best["seed"]}

    if args.eval_validation:
        from hymeko_rl.coin_delivery.full_action_bc import FullActionBC
        bc = FullActionBC()
        bc.load_state_dict(torch.load(f"{args.out}/bc_seed{best['seed']}.pt"))
        bc.eval()
        vev = (eval_framestack_bc_delivery(bc, VALIDATION, k=k, horizon=360) if k > 1
               else eval_bc_delivery(bc, VALIDATION, horizon=360))
        print(f"VALIDATION (best seed {best['seed']}): DELIVER {vev['deliver']}/{vev['n']} "
              f"rate {vev['deliver_rate']} {vev['delivered_seeds']}", flush=True)
        out["validation"] = {"deliver": vev["deliver"], "n": vev["n"], "rate": vev["deliver_rate"],
                             "delivered_seeds": vev["delivered_seeds"]}

    json.dump(out, open(f"{args.out}/bc_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
