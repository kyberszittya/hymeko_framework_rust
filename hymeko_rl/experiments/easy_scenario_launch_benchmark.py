"""EASY-SCENARIO cooperative launch — the first rung of the step-by-step walk. The full panel had two-contact unreachable
on 5/8 states (a start-config/spawn artifact). Here the coin is placed where BOTH arms can reach it (the fingertip
midpoint), isolating the cooperative launch from the reachability problem. Frozen V2/V4/coast/B1. No RL.

Question: given a two-arm-reachable coin, does the cooperative twist-Jacobian launch reliably acquire two contacts and
aim a target-directed launch? Arms A1 (symmetric close, no force balancing) vs A3 (force-balanced twist allocation).
Pre-registered gate: v_∥ ≥ 0.85·v_target ∧ |v_cross|/v_∥ < 0.2 ∧ signed_disp > 0 ∧ joint ≤ 3.45 (+ both tips, |ω| < 0.5).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.bimanual_launch import BimanualConfig, bimanual_launch_carry  # noqa: E402
from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.cooperative_launch import CooperativeConfig, cooperative_launch_carry, place_coin_at, reachability_probe, tip_midpoint  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
HORIZON = 200
V_FRAC, CROSS_MAX, GATE, OMEGA_MAX = 0.85, 0.2, 3.45, 0.5


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

    def make_easy(seed):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=seed, tries=3)
        tg, adr, _bt, _bd = setup_material_decoupling(rl)
        set_material(rl, tg, adr, v2["tip_coin_friction"], v2["coin_slide_viscous_damping"], v2["coin_slide_coulomb_frictionloss"])
        place_coin_at(rl, tip_midpoint(rl))                      # EASY: coin at the two-arm-reachable midpoint
        return rl, gate

    def directed(o, vt):
        return bool(o["peak_v_parallel"] >= V_FRAC * vt and o["cross_ratio"] < CROSS_MAX and o["peak_omega"] < OMEGA_MAX
                    and o["signed_target_displacement"] > 0 and o["peak_joint_vel"] <= GATE and o["both_tips_contact"] == 1)

    n_states = 4 if smoke else 8
    print(f"EASY-SCENARIO launch (coin at tip midpoint): μ={mu:.3f}, gate v_par≥{V_FRAC}·v_tgt ∧ cross<{CROSS_MAX} ∧ |ω|<{OMEGA_MAX} ∧ both", flush=True)
    rows = []
    for si in range(n_states):
        seed = 14000 + 250 * si
        rl, _g = make_easy(seed)
        reach = reachability_probe(rl, stack, CooperativeConfig(coast_mu=mu))
        vt = float(min(0.8, np.sqrt(max(0.0, 2.0 * mu * 9.81 * rl._dtz()))))
        rl, g = make_easy(seed)
        coop = cooperative_launch_carry(rl, g, pi0, base, stack, horizon=HORIZON, cfg=CooperativeConfig(coast_mu=mu))
        rl, g = make_easy(seed)
        a3 = bimanual_launch_carry(rl, g, pi0, base, stack, horizon=HORIZON, cfg=BimanualConfig(coast_mu=mu, state_dependent=True, force_balance=True))
        for o in (coop, a3):
            o["directed"] = directed(o, vt)
        rows.append({"state": si, "v_target": round(vt, 3), "two_contact_reachable": reach["two_contact_reachable"],
                     "cooperative": {k: coop[k] for k in ("both_tips_contact", "both_contact_frames", "peak_v_parallel", "cross_ratio", "peak_omega", "signed_target_displacement", "directed")},
                     "A3_balanced": {k: a3[k] for k in ("both_tips_contact", "both_contact_frames", "peak_v_parallel", "cross_ratio", "peak_omega", "signed_target_displacement", "directed")}})
        c = rows[-1]["cooperative"]
        print(f"  s{si}: reach2={int(reach['two_contact_reachable'])} | coop both{c['both_tips_contact']}({c['both_contact_frames']}f) "
              f"v_par{c['peak_v_parallel']} xr{c['cross_ratio']} ω{c['peak_omega']} dir{int(c['directed'])}", flush=True)

    both_rate = round(float(np.mean([r["cooperative"]["both_tips_contact"] for r in rows])), 3)
    coop_dir = sum(r["cooperative"]["directed"] for r in rows)
    a3_dir = sum(r["A3_balanced"]["directed"] for r in rows)
    best_dir = max(coop_dir, a3_dir)
    if best_dir >= max(3, n_states // 2 + 1):
        verdict = "EASY_SCENARIO_COOPERATIVE_LAUNCH_TARGET_DIRECTED"
    elif both_rate >= 0.5 and best_dir > 0:
        verdict = "COOPERATIVE_LAUNCH_EXERCISED__FORCE_ALLOCATION_TUNING_REMAINS"
    elif both_rate >= 0.5:
        verdict = "TWO_CONTACT_ACQUIRED_BUT_LAUNCH_NOT_YET_TARGET_DIRECTED"
    else:
        verdict = "COOPERATIVE_ACQUISITION_STILL_UNRELIABLE_EVEN_EASY"
    manifest = {"contract": "EASY_SCENARIO_COOPERATIVE_LAUNCH", "date": "2026-07-25", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2",
                "scenario": "coin placed at the fingertip midpoint (two-arm-reachable)", "coast_mu": round(mu, 3),
                "n_states": n_states, "both_tips_contact_rate": both_rate,
                "n_target_directed": {"cooperative": coop_dir, "A3_balanced": a3_dir}, "rows": rows, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/easy_scenario_launch_benchmark.json", "w"), indent=1, default=float)
    print("\n== EASY-SCENARIO cooperative launch ==")
    print(f"  both-tips contact rate: {both_rate} | target-directed: cooperative {coop_dir}/{n_states}, A3 {a3_dir}/{n_states}")
    print(f"\n  → {verdict}\n  artifact: {OUT}/easy_scenario_launch_benchmark.json\nEASY_SCENARIO_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
