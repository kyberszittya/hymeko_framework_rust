"""Cooperative two-contact launch benchmark — the (1) step: a two-contact REACHABILITY PLANNER + decoupled synchronized
close, to try to actually EXERCISE the cooperative twist allocation that BIMANUAL_V1 found un-exercised. Frozen V2/V4/
coast/B1. Per state: (a) reachability probe (can each arm individually reach the coin), (b) cooperative launch (both-tips
contact, launch quality, force-line-miss ω). Saves teacher records (reachability + acquisition outcome) for the structural
study. No RL.

The question this answers: is two-contact acquisition reliable enough to exercise the cooperative launch across the panel?
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.cooperative_launch import CooperativeConfig, cooperative_launch_carry, reachability_probe  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
HORIZON = 200
V_FRAC, CROSS_MAX, GATE = 0.85, 0.2, 3.45


def main(smoke=False):
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    v4 = json.load(open(f"{OUT}/dynamics_contract_v4.json"))["frozen_contract"]
    v2 = json.load(open(f"{OUT}/rubber_tip_v2_material.json"))["frozen_material"]
    mu = float(np.mean(list(v2["coin_floor_mu_eff_by_v0"].values())))
    stack = V3Stack(v4["qdot_soft"], v4["qdot_hard"], v4["armature"], v4["damping"], v4["friction"],
                    v4["kp"], v4["kv"], v4["tau_rate"], over_hard_brake=v4["over_hard_brake"])
    cfg = CooperativeConfig(coast_mu=mu)

    def make_v2(seed):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=seed, tries=3)
        tg, adr, _bt, _bd = setup_material_decoupling(rl)
        set_material(rl, tg, adr, v2["tip_coin_friction"], v2["coin_slide_viscous_damping"], v2["coin_slide_coulomb_frictionloss"])
        return rl, gate

    n_states = 4 if smoke else 8
    print(f"COOPERATIVE LAUNCH (reachability + synchronized close): μ={mu:.3f}", flush=True)
    rows = []
    for si in range(n_states):
        seed = 14000 + 250 * si
        rl, _g = make_v2(seed)
        reach = reachability_probe(rl, stack, cfg)
        rl, gate = make_v2(seed)
        o = cooperative_launch_carry(rl, gate, pi0, base, stack, horizon=HORIZON, cfg=cfg)
        vt = float(min(0.8, np.sqrt(max(0.0, 2.0 * mu * 9.81 * rl._dtz()))))
        directed = bool(o["peak_v_parallel"] >= V_FRAC * vt and o["cross_ratio"] < CROSS_MAX
                        and o["signed_target_displacement"] > 0 and o["peak_joint_vel"] <= GATE)
        rows.append({"state": si, "two_contact_reachable": reach["two_contact_reachable"],
                     "left_reachable": reach["left"]["reachable"], "right_reachable": reach["right"]["reachable"],
                     "both_tips_contact": o["both_tips_contact"], "both_contact_frames": o["both_contact_frames"],
                     "cross_ratio": o["cross_ratio"], "peak_omega": o["peak_omega"],
                     "signed_target_displacement": o["signed_target_displacement"], "directed": directed})
        print(f"  s{si}: reach L{int(reach['left']['reachable'])}/R{int(reach['right']['reachable'])}/2={int(reach['two_contact_reachable'])} "
              f"| both {o['both_tips_contact']}({o['both_contact_frames']}f) | xr {o['cross_ratio']} ω {o['peak_omega']} | dir {int(directed)}", flush=True)

    reach_rate = round(float(np.mean([r["two_contact_reachable"] for r in rows])), 3)
    both_rate = round(float(np.mean([r["both_tips_contact"] for r in rows])), 3)
    reachable = [r for r in rows if r["two_contact_reachable"]]
    both_on_reachable = round(float(np.mean([r["both_tips_contact"] for r in reachable])), 3) if reachable else 0.0
    if both_rate >= 0.5:
        verdict = "COOPERATIVE_TWO_CONTACT_LAUNCH_EXERCISED"        # (1) sufficient ⇒ proceed to (2)
    elif reach_rate < 0.6:
        verdict = "TWO_CONTACT_REACHABILITY_LIMITED_BY_EMBODIMENT_AND_START_CONFIG"
    else:
        verdict = "TWO_CONTACT_REACHABLE_BUT_SIMULTANEOUS_ACQUISITION_DEFEATED_BY_MOBILE_COIN"
    manifest = {"contract": "COOPERATIVE_TWO_CONTACT_LAUNCH", "date": "2026-07-25", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2",
                "coast_mu": round(mu, 3), "n_states": n_states, "two_contact_reachable_rate": reach_rate,
                "both_tips_contact_rate": both_rate, "both_tips_on_reachable_states": both_on_reachable,
                "n_directed": sum(r["directed"] for r in rows), "rows": rows, "teacher_records": rows, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/cooperative_launch_benchmark.json", "w"), indent=1, default=float)
    print("\n== COOPERATIVE two-contact launch ==")
    print(f"  two-contact reachable: {reach_rate} of panel | both-tips achieved: {both_rate} (on reachable states: {both_on_reachable})")
    print(f"  target-directed: {sum(r['directed'] for r in rows)}/{n_states}")
    print(f"\n  → {verdict}\n  artifact: {OUT}/cooperative_launch_benchmark.json\nCOOPERATIVE_LAUNCH_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
