"""TARGET_DIRECTED_LAUNCH_V2 benchmark — structural contact-MODE comparison. Frozen reference = L3 best reachable single
point (from V1); test whether a TWO-POINT contact places the resultant force-line through the coin centre and recovers a
target-directed launch. Frozen: V2 material, V4 motion contract, coast launch target, B1 barrier, the 8-state panel, and
the pre-registered target-directed gate. No RL.

Arms:
  L3_single_point   — best reachable single-point (far-side acquire) — the V1 baseline
  L5a_two_point      — symmetric two-point (fixed φ), squeeze toward COM
  L5b_edge_aware     — edge-aware two-point (pick φ per state by a delivery-independent probe)

V2 acceptance gate (pre-registered): v_∥ ≥ 0.85·v_target ∧ |v_cross|/v_∥ < 0.2 ∧ signed_disp > 0 ∧ joint ≤ 3.45
  + object angular velocity |ω_c| < ω_max (force-line passes near the COM) + both tips actually acquired (two-point arms).

The force-line-miss proxy ω_c is the mechanistic evidence: if two-point contact REDUCES ω_c AND the cross-track together,
the force-line mechanism is proven (not a lucky parameter). Per state we also SAVE the contact-mode decision (chosen mode,
φ, both-contact, force-line proxy) — the deterministic teacher record for the later structural-prior (Kato/HyMeKo) study.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.directed_launch import DirectedLaunchConfig, directed_launch_carry  # noqa: E402
from hymeko_rl.coin_delivery.two_point_launch import TwoPointConfig, two_point_launch_carry  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
HORIZON = 240
V_FRAC, CROSS_MAX, GATE, OMEGA_MAX = 0.85, 0.2, 3.45, 0.5      # pre-registered thresholds (ω_max rad/s force-line proxy)


def _directed(o, two_point):
    ok = bool(o["peak_v_parallel"] >= V_FRAC * o.get("v_target", 0.0) and o["peak_v_parallel"] > 0
              and o["cross_ratio"] < CROSS_MAX and o["signed_target_displacement"] > 0 and o["peak_joint_vel"] <= GATE)
    if two_point:
        ok = ok and o.get("peak_omega", 9.9) < OMEGA_MAX and o.get("both_tips_contact", 0) == 1
    return ok


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

    n_states = 4 if smoke else 8
    print(f"DIRECTED LAUNCH V2 (contact-mode): μ={mu:.3f}, gate v_par≥{V_FRAC}·v_tgt ∧ cross<{CROSS_MAX} ∧ |ω|<{OMEGA_MAX} ∧ both-tips", flush=True)
    rows, teacher = [], []
    for si in range(n_states):
        seed = 14000 + 250 * si
        rl, g = make_v2(seed)
        l3 = directed_launch_carry(rl, g, pi0, base, stack, horizon=HORIZON,
                                   cfg=DirectedLaunchConfig(coast_mu=mu, enable_contact_select=True))
        v_target = l3["v_target"]
        rl, g = make_v2(seed)
        l5a = two_point_launch_carry(rl, g, pi0, base, stack, horizon=HORIZON, cfg=TwoPointConfig(coast_mu=mu))
        rl, g = make_v2(seed)
        l5b = two_point_launch_carry(rl, g, pi0, base, stack, horizon=HORIZON, cfg=TwoPointConfig(coast_mu=mu, edge_aware=True))
        for o in (l5a, l5b):
            o["v_target"] = v_target
        rec = {"state": si, "v_target": round(v_target, 3),
               "L3_single_point": {**{k: l3[k] for k in ("peak_v_parallel", "peak_v_cross", "cross_ratio", "signed_target_displacement", "peak_joint_vel")}, "directed": _directed(l3, False)},
               "L5a_two_point": {**{k: l5a[k] for k in ("peak_v_parallel", "peak_v_cross", "cross_ratio", "peak_omega", "both_tips_contact", "signed_target_displacement", "peak_joint_vel")}, "directed": _directed(l5a, True)},
               "L5b_edge_aware": {**{k: l5b[k] for k in ("peak_v_parallel", "peak_v_cross", "cross_ratio", "peak_omega", "both_tips_contact", "chosen_phi", "signed_target_displacement", "peak_joint_vel")}, "directed": _directed(l5b, True)}}
        rows.append(rec)
        # teacher record for the later structural-prior study
        teacher.append({"state": si, "v_target": round(v_target, 3), "chosen_mode": l5b["contact_mode"],
                        "chosen_phi": l5b["chosen_phi"], "both_tips_contact": l5b["both_tips_contact"],
                        "force_line_miss_omega": l5b["peak_omega"], "predicted_v_parallel": l5b["peak_v_parallel"],
                        "predicted_v_cross": l5b["peak_v_cross"]})
        print(f"  s{si}: L3 xr{rec['L3_single_point']['cross_ratio']} | L5a xr{rec['L5a_two_point']['cross_ratio']} "
              f"ω{rec['L5a_two_point']['peak_omega']} both{rec['L5a_two_point']['both_tips_contact']} | "
              f"L5b xr{rec['L5b_edge_aware']['cross_ratio']} ω{rec['L5b_edge_aware']['peak_omega']} φ{rec['L5b_edge_aware']['chosen_phi']}", flush=True)

    def n_dir(name):
        return sum(r[name]["directed"] for r in rows)

    def mean_om(name):
        return round(float(np.mean([r[name].get("peak_omega", 0.0) for r in rows])), 4)

    def contact_rate(name):
        return round(float(np.mean([r[name].get("both_tips_contact", 0) for r in rows])), 3)
    dir_by = {n: n_dir(n) for n in ("L3_single_point", "L5a_two_point", "L5b_edge_aware")}
    l5ad, l5bd = dir_by["L5a_two_point"], dir_by["L5b_edge_aware"]
    # does two-point reduce ω AND cross vs L3 (force-line mechanism)?
    mech = sum(r["L5a_two_point"]["peak_omega"] < 0.1 and r["L5a_two_point"]["both_tips_contact"] == 1
               and r["L5a_two_point"]["cross_ratio"] < r["L3_single_point"]["cross_ratio"] - 0.02 for r in rows)
    if l5ad >= max(2, n_states // 2):
        verdict = "TWO_POINT_CENTERLINE_CONTACT_RECOVERS_TARGET_DIRECTED_LAUNCH"
    elif l5bd > l5ad and l5bd >= max(2, n_states // 2):
        verdict = "EDGE_AWARE_CONTACT_PAIR_SELECTION_IS_LOAD_BEARING"
    elif l5ad > 0 or l5bd > 0 or mech > 0:
        verdict = "MULTI_CONTACT_DIRECTIONAL_CAPABILITY_EXISTS__REACHABILITY_OR_MODE_SELECTION_REMAINS_OPEN"
    else:
        verdict = "CURRENT_EMBODIMENT_GEOMETRY_LIMITS_TARGET_DIRECTED_LAUNCH"
    manifest = {"contract": "TARGET_DIRECTED_LAUNCH_V2", "date": "2026-07-25", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2",
                "coast_mu": round(mu, 3), "gate": {"v_frac": V_FRAC, "cross_max": CROSS_MAX, "omega_max": OMEGA_MAX},
                "n_states": n_states, "n_target_directed_by_mode": dir_by,
                "both_tips_contact_rate": {"L5a": contact_rate("L5a_two_point"), "L5b": contact_rate("L5b_edge_aware")},
                "mean_force_line_omega": {"L5a": mean_om("L5a_two_point"), "L5b": mean_om("L5b_edge_aware")},
                "n_forceline_mechanism_confirmed": mech, "rows": rows, "teacher_records": teacher, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/directed_launch_v2_benchmark.json", "w"), indent=1, default=float)
    print("\n== V2 contact-mode — target-directed count | both-tips rate | mean force-line ω ==")
    for n in ("L3_single_point", "L5a_two_point", "L5b_edge_aware"):
        print(f"  {n:18s} directed {dir_by[n]}/{n_states} | both-tips {contact_rate(n) if 'two' in n or 'edge' in n else '-'} | mean ω {mean_om(n)}")
    print(f"  force-line mechanism confirmed (ω<0.1 ∧ both-contact ∧ cross↓ vs L3): {mech}/{n_states}")
    print(f"\n  → {verdict}\n  artifact: {OUT}/directed_launch_v2_benchmark.json\nDIRECTED_LAUNCH_V2_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
