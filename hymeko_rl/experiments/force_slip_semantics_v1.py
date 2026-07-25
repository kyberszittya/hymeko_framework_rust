"""FORCE_SLIP_SEMANTICS_V1 — a narrow, mechanistic ablation of contact-level manipulation semantics on the FROZEN
RUBBER_TIP_LOW_DRAG_COIN_V2 physics + frozen V4 motion contract. NO retraining, NO proposal, NO RL. Progressive stages
(each adds ONE semantic element):

  S0  position-following reference
  S1  + bounded normal preload
  S2  + target-directed tangential velocity (launch target from the CALIBRATED coast model)
  S3  + slip-aware modulation
  S4  + controlled impulse + coast observation
  S5  + predictive braking

Statistical discipline (user): the evaluation subset is the CONTROLLER-INDEPENDENT contact-capable set (an acquire-only
oracle establishes contact), pre-registered before running the stages — NOT states where a given stage happened to touch
the coin. Report the full panel, the contact-capable subset, and per-stage results.

Primary metrics (K6 is NOT primary here): coin velocity at the end of the controlled push, target-directed impulse, Ft/Fn,
transport distance, zone entry, stopping-distance prediction error. Secondary: K6.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_capable import contact_capable_subset  # noqa: E402
from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.force_slip_carry import ForceSlipConfig, force_slip_carry  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
HORIZON = 320


def _stages(mu):
    base = dict(coast_mu=mu)
    return {
        "S0_reference": ForceSlipConfig(**base, enable_preload=False, enable_target_velocity=False, enable_slip_aware=False,
                                        enable_impulse_gate=False, enable_predictive_brake=False),
        "S1_preload": ForceSlipConfig(**base, enable_target_velocity=False, enable_slip_aware=False,
                                      enable_impulse_gate=False, enable_predictive_brake=False),
        "S2_target_velocity": ForceSlipConfig(**base, enable_slip_aware=False, enable_impulse_gate=False,
                                              enable_predictive_brake=False),
        "S3_slip_aware": ForceSlipConfig(**base, enable_impulse_gate=False, enable_predictive_brake=False),
        "S4_impulse": ForceSlipConfig(**base, enable_predictive_brake=False),
        "S5_predictive_brake": ForceSlipConfig(**base),
    }


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

    n_states = 5 if smoke else 8
    seeds = [(si, 14000 + 250 * si) for si in range(n_states)]

    def make_v2(seed):                                            # fresh env + its MATCHING gate, V2 material applied
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=seed, tries=3)
        tg, adr, _bt, _bd = setup_material_decoupling(rl)
        set_material(rl, tg, adr, v2["tip_coin_friction"], v2["coin_slide_viscous_damping"], v2["coin_slide_coulomb_frictionloss"])
        return rl, gate

    # pre-registered contact-capable subset (controller-independent acquire oracle; the oracle uses only rl)
    cap = contact_capable_subset(lambda s: make_v2(s)[0], stack, seeds)
    capable = set(cap["contact_capable_states"])
    print(f"contact-capable subset (acquire oracle): {sorted(capable)} ({cap['n_capable']}/{cap['n_total']}); coast μ={mu:.3f}", flush=True)

    stages = _stages(mu)
    rows = []
    for si, seed in seeds:
        rec = {"state": si, "contact_capable": si in capable}
        for name, cfg in stages.items():
            rl, gate = make_v2(seed)                              # fresh env + MATCHING gate per stage
            o = force_slip_carry(rl, gate, pi0, base, stack, horizon=HORIZON, cfg=cfg)
            rec[name] = {k: o[k] for k in ("peak_coin_velocity", "coin_push_end_velocity", "target_directed_impulse",
                                           "ftfn", "transport_dist", "entered_zone", "k6", "stopping_distance_pred_error",
                                           "peak_joint_vel")}
        rows.append(rec)
        print(f"  s{si}{'*' if si in capable else ' '}: " + " ".join(
            f"{n.split('_')[0]} v{rec[n]['peak_coin_velocity']}/td{rec[n]['transport_dist']}/z{rec[n]['entered_zone']}"
            for n in stages), flush=True)

    def agg(subset, name, k):
        vals = [r[name][k] for r in rows if (subset is None or r["contact_capable"]) and r[name][k] is not None]
        return round(float(np.mean(vals)), 3) if vals else 0.0
    metrics = ("peak_coin_velocity", "coin_push_end_velocity", "target_directed_impulse", "ftfn", "transport_dist",
               "entered_zone", "k6", "stopping_distance_pred_error", "peak_joint_vel")
    summary = {scope: {name: {k: agg(scope, name, k) for k in metrics} for name in stages}
               for scope in ("capable", None)}

    cap_sum = summary["capable"]
    s0, s5 = cap_sum["S0_reference"], cap_sum["S5_predictive_brake"]
    dv = s5["peak_coin_velocity"] - s0["peak_coin_velocity"]      # controlled impulse: velocity actually imparted
    v_ok = s5["peak_joint_vel"] <= v4["abs_velocity_gate"]        # created the impulse WITHIN the motion contract
    zone_up = s5["entered_zone"] > s0["entered_zone"] + 0.15
    if dv > 0.1 and zone_up:
        verdict = "CONTACT_SEMANTIC_CONTROL_RECOVERS_TRANSPORT"
    elif dv > 0.1 and v_ok:
        verdict = "TRANSPORT_IMPULSE_RECOVERED__BRAKING_OR_TERMINAL_CONTROL_REMAINS_THE_WALL"
    elif not v_ok:
        verdict = "IMPULSE_ONLY_ACHIEVED_BY_VIOLATING_MOTION_CONTRACT"
    else:
        verdict = "SINGLE_TIP_POSITION_IMPEDANCE_INSUFFICIENT_FOR_REQUIRED_CONTROLLED_IMPULSE"
    manifest = {"contract": "FORCE_SLIP_SEMANTICS_V1", "date": "2026-07-25", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2 (frozen)",
                "motion_contract": v4["dynamics_contract"], "no_retraining": True, "coast_mu": round(mu, 3),
                "contact_capable_subset": cap, "summary_capable": summary["capable"], "summary_full_panel": summary[None],
                "rows": rows, "coin_velocity_delta_S5_minus_S0": round(dv, 3), "impulse_within_motion_contract": bool(v_ok),
                "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/force_slip_semantics_v1.json", "w"), indent=1, default=float)
    print("\n== FORCE_SLIP_SEMANTICS_V1 on the contact-capable subset (primary: coin push-end velocity) ==")
    for name in stages:
        s = cap_sum[name]
        print(f"  {name:22s} peak_coin_v {s['peak_coin_velocity']} | impulse {s['target_directed_impulse']} | Ft/Fn {s['ftfn']} | "
              f"transport {s['transport_dist']} | zone {s['entered_zone']} | K6 {s['k6']} | stop_err {s['stopping_distance_pred_error']}")
    print(f"\n  peak coin velocity S0→S5: {s0['peak_coin_velocity']}→{s5['peak_coin_velocity']} (Δ{round(dv, 3)}); "
          f"peak joint vel {s5['peak_joint_vel']} (≤{v4['abs_velocity_gate']}: {v_ok})")
    print(f"  → {verdict}\n  artifact: {OUT}/force_slip_semantics_v1.json\nFORCE_SLIP_SEMANTICS_V1_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
