"""Matched 4-config BC pilot (§2-§9): A instantaneous / B obs-history / C obs+action-history on the IDENTICAL
certified open-loop trajectories, matched arch/opt/budget/seeds; + D handoff baseline. Reports supervised val MSE
(teacher-forced proxy) and standalone CLOSED-LOOP strict-K6 success (headline + validation)."""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE, VALIDATION
from hymeko_rl.coin_delivery.full_action_bc import (
    FullActionBC, eval_action_history_bc_delivery, eval_bc_delivery, eval_framestack_bc_delivery,
    load_trajectory_dataset, stack_frames, train_bc_phase_balanced,
)
from hymeko_rl.coin_delivery.full_action_obs_history import build_history_features

BASE = sys.argv[1]
OUT = sys.argv[2]
SEEDS = [0, 1, 2]


def open_loop_corpus():
    d = load_trajectory_dataset([BASE], patterns=("traj_*.npz",))
    idx = [json.loads(x) for x in open(BASE + "/index.jsonl")]
    fseeds = [int(f.split("traj_")[1].split(".npz")[0]) for f in d["files"]]
    ol = {r["seed"] for r in idx if r.get("delivered") and r.get("teacher") == "search"}
    tids = {i for i, s in enumerate(fseeds) if s in ol}
    m = np.isin(d["traj"], list(tids))
    return d["obs"][m], d["act"][m], d["phase"][m], d["traj"][m]


def traj_split(traj, frac_val=0.2, seed=0):
    ids = np.unique(traj)
    rng = np.random.default_rng(seed)
    nval = max(1, int(len(ids) * frac_val))
    val = set(rng.choice(ids, size=nval, replace=False).tolist())
    vm = np.isin(traj, list(val))
    return ~vm, vm


def train_eval(inp, act, phase, tr, va, contract, seed):
    bc, hist = train_bc_phase_balanced(inp[tr], act[tr], phase[tr], epochs=300, lr=1e-3, batch=256,
                                       seed=seed, steps_per_epoch=200, val=(inp[va], act[va], phase[va]))
    vmse = hist["val"][-1]
    if contract == "A_instant":
        hl = eval_bc_delivery(bc, HEADLINE)
        vl = eval_bc_delivery(bc, VALIDATION)
    elif contract == "B_obs_hist":
        hl = eval_framestack_bc_delivery(bc, HEADLINE, k=3)
        vl = eval_framestack_bc_delivery(bc, VALIDATION, k=3)
    else:                                                    # C_obs_act_hist
        hl = eval_action_history_bc_delivery(bc, HEADLINE)
        vl = eval_action_history_bc_delivery(bc, VALIDATION)
    return {"seed": seed, "val_mse": vmse, "headline_deliver": hl["deliver"], "headline_grasp": hl["grasp"],
            "validation_deliver": vl["deliver"], "validation_n": vl["n"], "delivered_headline": hl["delivered_seeds"]}


def main():
    obs, act, phase, traj = open_loop_corpus()
    tr, va = traj_split(traj, seed=0)
    print(f"open-loop corpus: {len(np.unique(traj))} traj, {len(obs)} samples | train {int(tr.sum())} val {int(va.sum())}",
          flush=True)
    inputs = {"A_instant": obs, "B_obs_hist": stack_frames(obs, traj, 3),
              "C_obs_act_hist": build_history_features(obs, act, traj)}
    results = {}
    for contract, inp in inputs.items():
        rows = []
        for sd in SEEDS:
            r = train_eval(inp, act, phase, tr, va, contract, sd)
            rows.append(r)
            print(f"  {contract} seed {sd}: val_mse {r['val_mse']:.2e} | HEADLINE deliver {r['headline_deliver']}/9 "
                  f"grasp {r['headline_grasp']}/9 | VALIDATION {r['validation_deliver']}/{r['validation_n']}", flush=True)
        results[contract] = rows
    # D: handoff baseline (instantaneous, previously trained best) — re-eval for a matched closed-loop number
    try:
        bcD = FullActionBC()
        bcD.load_state_dict(torch.load(sys.argv[3]))
        bcD.eval()
        hlD = eval_bc_delivery(bcD, HEADLINE)
        vlD = eval_bc_delivery(bcD, VALIDATION)
        results["D_handoff_baseline"] = [{"seed": "frozen", "headline_deliver": hlD["deliver"],
                                          "headline_grasp": hlD["grasp"], "validation_deliver": vlD["deliver"],
                                          "validation_n": vlD["n"], "delivered_headline": hlD["delivered_seeds"]}]
        print(f"  D_handoff_baseline (frozen): HEADLINE {hlD['deliver']}/9 grasp {hlD['grasp']}/9 | "
              f"VALIDATION {vlD['deliver']}/{vlD['n']}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  D handoff baseline eval skipped: {e}", flush=True)
    json.dump(results, open(OUT, "w"), indent=1)
    # summary
    def best(c):
        return max(results.get(c, [{"headline_deliver": -1}]), key=lambda r: r["headline_deliver"])
    print("\n=== best-seed closed-loop HEADLINE / VALIDATION ===", flush=True)
    for c in ("A_instant", "B_obs_hist", "C_obs_act_hist", "D_handoff_baseline"):
        b = best(c)
        print(f"  {c:<20} headline {b.get('headline_deliver')}/9  grasp {b.get('headline_grasp')}/9  "
              f"validation {b.get('validation_deliver')}/{b.get('validation_n','?')}  val_mse {b.get('val_mse','-')}",
              flush=True)


if __name__ == "__main__":
    main()
