"""BIMANUAL_TARGET_DIRECTED_LAUNCH_V1 benchmark — the two-arm cooperative launch is the essential task mechanism (single
tip is the CONTROL). Frozen: V2 material, V4 motion contract, coast launch target, B1 barrier, the 8-state panel, the
pre-registered target-directed gate. Deterministic teacher, no RL.

Arms:
  A0 single_tip_L3   — best reachable single point (V1 control)
  A1 bimanual_symmetric — two arms, symmetric ±e_cross acquire, twist allocation (no force balancing)
  A2 bimanual_state_dep — two arms, state-dependent acquire (each tip to its nearest coin surface point)
  A3 bimanual_balanced  — A2 + force balancing / zero-spin objective in the twist solve

Pre-registered gate: v_∥ ≥ 0.85·v_target ∧ |v_cross|/v_∥ < 0.2 ∧ signed_disp > 0 ∧ joint ≤ 3.45.
Bimanual mechanistic metrics: both tips contact + simultaneity, force imbalance |Fn_L−Fn_R|, force-line-miss ω_c.
Saves per-state teacher records (the relational contact-mode/force-allocation decision) for the later structural-prior study.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.bimanual_launch import BimanualConfig, bimanual_launch_carry  # noqa: E402
from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.directed_launch import DirectedLaunchConfig, directed_launch_carry  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
HORIZON = 220
V_FRAC, CROSS_MAX, GATE = 0.85, 0.2, 3.45


def _directed(o, vt):
    return bool(o["peak_v_parallel"] >= V_FRAC * vt and o["peak_v_parallel"] > 0
               and o["cross_ratio"] < CROSS_MAX and o["signed_target_displacement"] > 0 and o["peak_joint_vel"] <= GATE)


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

    def make_v2(seed):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=seed, tries=3)
        tg, adr, _bt, _bd = setup_material_decoupling(rl)
        set_material(rl, tg, adr, v2["tip_coin_friction"], v2["coin_slide_viscous_damping"], v2["coin_slide_coulomb_frictionloss"])
        return rl, gate

    arms = {
        "A1_symmetric": BimanualConfig(coast_mu=mu, state_dependent=False, force_balance=False),
        "A2_state_dep": BimanualConfig(coast_mu=mu, state_dependent=True, force_balance=False),
        "A3_balanced": BimanualConfig(coast_mu=mu, state_dependent=True, force_balance=True),
    }
    n_states = 4 if smoke else 8
    print(f"BIMANUAL LAUNCH V1: μ={mu:.3f}, gate v_par≥{V_FRAC}·v_tgt ∧ cross<{CROSS_MAX} ∧ signed>0 ∧ joint≤{GATE}", flush=True)
    rows, teacher = [], []
    for si in range(n_states):
        seed = 14000 + 250 * si
        rl, g = make_v2(seed)
        l3 = directed_launch_carry(rl, g, pi0, base, stack, horizon=HORIZON, cfg=DirectedLaunchConfig(coast_mu=mu, enable_contact_select=True))
        vt = l3["v_target"]
        rec = {"state": si, "v_target": round(vt, 3),
               "A0_single_tip": {**{k: l3[k] for k in ("peak_v_parallel", "cross_ratio", "signed_target_displacement", "peak_joint_vel")}, "directed": _directed(l3, vt)}}
        for name, cfg in arms.items():
            rl, g = make_v2(seed)
            o = bimanual_launch_carry(rl, g, pi0, base, stack, horizon=HORIZON, cfg=cfg)
            rec[name] = {**{k: o[k] for k in ("peak_v_parallel", "cross_ratio", "peak_omega", "both_tips_contact",
                                              "both_contact_frames", "force_imbalance", "signed_target_displacement", "peak_joint_vel")},
                         "directed": _directed(o, vt)}
        rows.append(rec)
        a3 = rec["A3_balanced"]
        teacher.append({"state": si, "v_target": round(vt, 3), "both_tips_contact": a3["both_tips_contact"],
                        "both_contact_frames": a3["both_contact_frames"], "force_imbalance": a3["force_imbalance"],
                        "force_line_miss_omega": a3["peak_omega"], "cross_ratio": a3["cross_ratio"], "directed": a3["directed"]})
        print(f"  s{si}: A0 xr{rec['A0_single_tip']['cross_ratio']}/{'D' if rec['A0_single_tip']['directed'] else '-'} | "
              f"A1 xr{rec['A1_symmetric']['cross_ratio']} | A2 xr{rec['A2_state_dep']['cross_ratio']} | "
              f"A3 xr{a3['cross_ratio']} ω{a3['peak_omega']} both{a3['both_tips_contact']} imb{a3['force_imbalance']}/{'D' if a3['directed'] else '-'}", flush=True)

    def n_dir(name):
        return sum(r[name]["directed"] for r in rows)

    def contact_rate(name):
        return round(float(np.mean([r[name].get("both_tips_contact", 0) for r in rows])), 3)
    dir_by = {n: n_dir(n) for n in ("A0_single_tip", "A1_symmetric", "A2_state_dep", "A3_balanced")}
    a0, a3d = dir_by["A0_single_tip"], dir_by["A3_balanced"]
    if a3d >= max(3, n_states // 2 + 1):
        verdict = "BIMANUAL_FORCE_ALLOCATION_RECOVERS_TARGET_DIRECTED_LAUNCH"
    elif a3d > a0:
        verdict = "BIMANUAL_CONTACT_IMPROVES_DIRECTIONAL_LAUNCH__REACHABILITY_OR_BALANCE_REMAINS_OPEN"
    elif contact_rate("A3_balanced") < 0.4:
        verdict = "BIMANUAL_ACQUISITION_UNRELIABLE__BOTH_TIPS_RARELY_ENGAGE"
    else:
        verdict = "CURRENT_EMBODIMENT_GEOMETRY_LIMITS_TARGET_DIRECTED_LAUNCH"
    manifest = {"contract": "BIMANUAL_TARGET_DIRECTED_LAUNCH_V1", "date": "2026-07-25", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2",
                "coast_mu": round(mu, 3), "gate": {"v_frac": V_FRAC, "cross_max": CROSS_MAX}, "n_states": n_states,
                "n_target_directed_by_arm": dir_by,
                "both_tips_contact_rate": {n: contact_rate(n) for n in ("A1_symmetric", "A2_state_dep", "A3_balanced")},
                "mean_force_imbalance_A3": round(float(np.mean([r["A3_balanced"]["force_imbalance"] for r in rows])), 3),
                "rows": rows, "teacher_records": teacher, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/bimanual_launch_benchmark.json", "w"), indent=1, default=float)
    print("\n== BIMANUAL — target-directed count | both-tips rate ==")
    for n in ("A0_single_tip", "A1_symmetric", "A2_state_dep", "A3_balanced"):
        cr = contact_rate(n) if n != "A0_single_tip" else "-"
        print(f"  {n:16s} directed {dir_by[n]}/{n_states} | both-tips {cr}")
    print(f"\n  → {verdict}\n  artifact: {OUT}/bimanual_launch_benchmark.json\nBIMANUAL_LAUNCH_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
