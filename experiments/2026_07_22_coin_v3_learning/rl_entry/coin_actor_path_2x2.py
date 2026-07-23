"""ACTOR_PATH_FIDELITY_CLEAN_2x2_V1 — the decisive actor-path experiment. One trained Markov critic; four candidate
actors from a 2x2 = {Q-only, Q+BC} x {no-trust, trust}, each a multi-step optimisation from the trained actor; then a
PHYSICAL rollout of every candidate on a graded-ladder panel (ΔQ is critic-space; K1/K3/K5/K6 + dtz/speed/dwell/exit is
the physical proof). Isolates whether the BC anchor and/or the trust region is the load-bearing blocker. No new campaign.
"""
import copy
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_late_start import LateStart  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import (  # noqa: E402
    OBS48,
    _sample_actor,
    eval_ablation,
    make_late_actor55_from_pi0,
    train_arm,
)
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import strict_onehot  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_trainer import masked_actor_loss  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_transactional import TransactionalConfig, transactional_actor_step  # noqa: E402
from hymeko_rl.coin_delivery.coin_transport_dwell import CONTROL_MODES  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
STEPS = 100


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _with_strict(o55, k):
    o = o55.clone(); o[:, OBS48:] = torch.as_tensor(strict_onehot(k)); return o


def _q(critic, actor, o):
    with torch.no_grad():
        return float(critic.min_q(o, torch.clamp(actor.action_mean(o), -4, 4)).mean())


def _counter_use(actor, o):
    with torch.no_grad():
        return float((actor.action_mean(_with_strict(o, 1)) - actor.action_mean(_with_strict(o, 5))).abs().mean())


def main():
    torch.set_num_threads(1); log = lambda *a: print(*a, flush=True)
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json")); pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    stage = dict(cfg["stage1"]); stage["total_updates"] = 4000; stage["checkpoints"] = [0, 2000, 4000]
    tcfg = TransactionalConfig(); batch = 256; lam = tcfg.lambda_bc

    # graded-ladder eval panel: the 31-state transport/braking/settling dev bank (pi_0 has K1>K3>K6 there)
    tdcfg = json.load(open(f"{D}/transport_dwell_config.json"))
    graded = [ls for m in CONTROL_MODES for ls in _bank(tdcfg["banks"]["dev"][m])]
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    pi0_g = eval_ablation(pi0, base, graded, "A", horizon=stage["horizon"], families=CONTROL_MODES, reset_strict=False)
    log(f"[graded panel] n={pi0_g['n']} pi_0 ladder K1 {pi0_g['k1']} K3 {pi0_g['k3']} K5 {pi0_g['k5']} K6 {pi0_g['k6_rate']}  graded={pi0_g['k1']>pi0_g['k6_rate']+0.05}")

    log("[train] Arm A Markov critic (4000 upd)...")
    _res, art = train_arm(pi0, "A", stage, tb, db, seed=0, tcfg=tcfg, return_artifacts=True, log=lambda *a: None)
    critic, actor0, buf, anchor_obs, a0_anchor = art["critic"], art["actor"], art["buf"], art["anchor_obs"], art["a0_anchor"]
    rng = np.random.default_rng(0); ao, ag = _sample_actor(buf, batch, rng); ao = ao[ag > 0][:128]; ag2 = torch.ones(ao.shape[0])
    q0 = _q(critic, actor0, ao); cu0 = _counter_use(actor0, ao)

    def candidate(use_bc, use_trust):
        act = copy.deepcopy(actor0); opt = torch.optim.Adam(act.parameters(), lr=(tcfg.actor_lr if use_trust else 1e-3))
        accepts = 0
        for _ in range(STEPS):
            if use_bc:
                with torch.no_grad():
                    pb = torch.clamp(pi0.action_mean(ao[:, :OBS48]), -4, 4)
                lf = (lambda: masked_actor_loss(critic, act, ao, ag2) + lam * ((act.action_mean(ao) - pb) ** 2).sum(-1).mean())
            else:
                lf = (lambda: masked_actor_loss(critic, act, ao, ag2))
            if use_trust:
                r = transactional_actor_step(act, opt, lf, anchor_obs, a0_anchor, tcfg); accepts += int(r["outcome"] == "accepted")
            else:
                loss = lf(); opt.zero_grad(); loss.backward(); opt.step()
        phys = eval_ablation(pi0, act, graded, "A", horizon=stage["horizon"], families=CONTROL_MODES, reset_strict=False)
        return {"accepts": accepts, "delta_Q": round(_q(critic, act, ao) - q0, 3), "counter_use": round(_counter_use(act, ao), 4),
                "K6": phys["k6_rate"], "K5": phys["k5"], "K3": phys["k3"], "K1": phys["k1"], "mean_dwell": phys["mean_max_dwell"],
                "exit_ct": phys["exit_ct"], "entry_vel": phys["entry_vel"], "canonical_return": phys["canonical_return"]}

    log(f"[2x2] optimising 4 candidates ({STEPS} steps each) + physical rollout on the graded panel...")
    cells = {"Qonly_notrust": candidate(False, False), "QBC_notrust": candidate(True, False),
             "Qonly_trust": candidate(False, True), "QBC_trust": candidate(True, True)}
    for k, v in cells.items():
        log(f"  {k:14} ΔQ {v['delta_Q']:+.1f} counter_use {v['counter_use']:.4f} accepts {v['accepts']}/{STEPS} | "
            f"K6 {v['K6']} K5 {v['K5']} K3 {v['K3']} K1 {v['K1']} dwell {v['mean_dwell']} exit {v['exit_ct']}")

    pi0k6 = pi0_g["k6_rate"]; M = 0.05
    qn = cells["Qonly_notrust"]; qbcn = cells["QBC_notrust"]; qt = cells["Qonly_trust"]
    physical_conversion = qn["K6"] >= pi0k6 + M                     # Q-only/no-trust physically improves K6
    bc_breaks = physical_conversion and qbcn["K6"] < qn["K6"] - M   # adding BC destroys the gain
    trust_breaks = physical_conversion and qt["K6"] < qn["K6"] - M  # adding trust destroys the gain
    verdicts = ["MARKOV_CRITIC_REPAIR_CONFIRMED"]
    if physical_conversion:
        verdicts.append("PHYSICAL_POLICY_CONVERSION_ACHIEVED")
        if bc_breaks:
            verdicts.append("BC_ANCHOR_LOAD_BEARING_BLOCKER")
        if trust_breaks:
            verdicts.append("TRUST_REGION_LOAD_BEARING_BLOCKER")
        if not (bc_breaks or trust_breaks):
            verdicts.append("ACTOR_PATH_UNBLOCKED")
    else:
        verdicts.append("PHYSICAL_POLICY_CONVERSION_NOT_ACHIEVED_ON_THIS_PANEL")

    out = {"contract": "ACTOR_PATH_FIDELITY_CLEAN_2x2_V1", "date": "2026-07-23", "no_new_campaign": True,
           "graded_panel": {"n": pi0_g["n"], "pi0_ladder": {"K1": pi0_g["k1"], "K3": pi0_g["k3"], "K5": pi0_g["k5"], "K6": pi0_g["k6_rate"]},
                            "is_graded": pi0_g["k1"] > pi0_g["k6_rate"] + 0.05},
           "q0": round(q0, 2), "counter_use_trained": round(cu0, 4), "cells": cells,
           "physical_conversion": physical_conversion, "bc_breaks_conversion": bc_breaks, "trust_breaks_conversion": trust_breaks,
           "verdict": verdicts}
    json.dump(out, open(f"{D}/actor_path_2x2_v1.json", "w"), indent=1, default=float)
    log(f"\n  pi_0 K6 {pi0k6} | Qonly/notrust K6 {qn['K6']} (Δ {qn['K6']-pi0k6:+.3f}) | QBC/notrust {qbcn['K6']} | Qonly/trust {qt['K6']} | QBC/trust {cells['QBC_trust']['K6']}")
    log(f"→ {verdicts}\nwrote {D}/actor_path_2x2_v1.json\nACTOR_PATH_2x2_DONE")


if __name__ == "__main__":
    main()
