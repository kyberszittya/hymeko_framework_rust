"""V1-vs-V2 material re-run — how much of the coin transport wall was the (mis-calibrated) CONTACT PHYSICS vs the
controller? Run the SAME frozen controllers, WITHOUT retraining, on:

  V1 = SINGLE_TIP_LOW_FRICTION_COIN_V1  (as-loaded: tip μ 1.0, coin viscous damping 2.5 — ~15× too sticky)
  V2 = RUBBER_TIP_LOW_DRAG_COIN_V2      (calibrated: tip μ 2.0, coin viscous 0.02 + Coulomb 0.074 → μ_eff ≈ 0.15)

Same panel / seeds / frozen V4 motion contract. Controllers:
  searched_legacy_expert — the 128-shot open-loop macro (the strongest legacy)
  C1_closed_loop         — motion_robust_carry (continuous contact-retaining transport + braking)
  C2_intermittent        — intermittent_carry (impulse → coast → re-contact → brake → settle)

Metrics: transport distance, zone entry, K6, Ft/Fn, coin peak/terminal speed, contact episodes, torque saturation.
Verdict compares V2 vs V1 for the UNCHANGED controllers — the physics-correction-only delta. No delivery inspected during
the V2 freeze; here delivery is the OUTCOME we read, not an input.
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")

from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.intermittent_carry import IntermittentConfig, intermittent_carry  # noqa: E402
from hymeko_rl.coin_delivery.motion_robust_expert import CarryControllerConfig, motion_robust_carry  # noqa: E402
from hymeko_rl.experiments.coin_c2_ablation import EVAL_H, _coin_progress, _legacy_arm, _stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"


def _carry_arm(rl, gate, pi0, base, stack, fn):
    """Wrap a closed-loop controller (motion_robust_carry / intermittent_carry) with a coin-progress hook → phase-ladder."""
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    u, _n = rl.inner.direction_to_zone()
    prog = {"max": 0.0}

    def hook(_ph, _s):
        prog["max"] = max(prog["max"], _coin_progress(r2, disk0, np.asarray(u, np.float32)))
    r2 = copy.deepcopy(rl)
    o = fn(r2, copy.deepcopy(gate), pi0, base, stack, horizon=EVAL_H, frame_hook=hook)
    return {"k6": int(o["k6"]), "acquired": int(o["acquired_contact"]), "transport_dist": round(prog["max"], 4),
            "zone_entry": int(o["entered_zone"]), "episodes": o.get("n_contact_episodes", 0),
            "ftfn": round(o["peak_contact_tangential_force"] / (o["peak_contact_normal_force"] + 1e-6), 3),
            "peak_coin": o["peak_coin_speed"], "term_coin": o["terminal_coin_speed"], "sat": o["torque_saturation_frac"]}


def _run_controllers(rl, gate, pi0, base, stack):
    return {
        "searched_legacy_expert": _legacy_arm(rl, gate, pi0, base, stack, slow=False),
        "C1_closed_loop": _carry_arm(rl, gate, pi0, base, stack,
                                     lambda *a, horizon, frame_hook: motion_robust_carry(*a, horizon=horizon, cfg=CarryControllerConfig(), frame_hook=frame_hook)),
        "C2_intermittent": _carry_arm(rl, gate, pi0, base, stack,
                                      lambda *a, horizon, frame_hook: intermittent_carry(*a, horizon=horizon, cfg=IntermittentConfig(), frame_hook=frame_hook)),
    }


def _apply_v2(rl):
    v2 = json.load(open(f"{OUT}/rubber_tip_v2_material.json"))["frozen_material"]
    tg, adr, _bt, _bd = setup_material_decoupling(rl)
    set_material(rl, tg, adr, v2["tip_coin_friction"], v2["coin_slide_viscous_damping"], v2["coin_slide_coulomb_frictionloss"])


def main(smoke=False):
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    stack, _v4 = _stack()
    n_states = 4 if smoke else 8
    rows = []
    for si in range(n_states):
        rl_v1, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None,
                                   seed_lo=14000 + 250 * si, tries=3)
        v1 = _run_controllers(rl_v1, gate, pi0, base, stack)          # as-loaded material (deepcopies inside → rl_v1 intact)
        rl_v2, gate2 = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None,
                                    seed_lo=14000 + 250 * si, tries=3)
        _apply_v2(rl_v2)                                              # calibrated material
        v2 = _run_controllers(rl_v2, gate2, pi0, base, stack)
        rows.append({"state": si, "V1": v1, "V2": v2})
        print(f"  s{si}: " + " | ".join(
            f"{c[:4]} td {v1[c]['transport_dist']}->{v2[c]['transport_dist']} zone {v1[c]['zone_entry']}->{v2[c]['zone_entry']} "
            f"k6 {v1[c]['k6']}->{v2[c]['k6']}" for c in v1), flush=True)

    ctrls = list(rows[0]["V1"].keys())

    def agg(mat, c, k):
        return round(float(np.mean([r[mat][c].get(k, 0.0) for r in rows])), 3)   # legacy arm lacks ftfn/peak_coin keys
    summary = {c: {mat: {k: agg(mat, c, k) for k in ("transport_dist", "zone_entry", "k6", "ftfn", "peak_coin", "term_coin")}
                   for mat in ("V1", "V2")} for c in ctrls}
    # physics-correction delta on the strongest transport controller
    best_dtd = max(summary[c]["V2"]["transport_dist"] - summary[c]["V1"]["transport_dist"] for c in ctrls)
    v2_zone = max(summary[c]["V2"]["zone_entry"] for c in ctrls)
    v1_zone = max(summary[c]["V1"]["zone_entry"] for c in ctrls)
    if v2_zone > v1_zone + 0.2 and v2_zone > 0.3:
        verdict = "AS_LOADED_CONTACT_MODEL_WAS_PRIMARY_FAILURE"       # calibrated physics alone restores delivery
    elif best_dtd > 0.02:
        verdict = "PHYSICAL_MODEL_BOTTLENECK_PARTIALLY_REMOVED__OPTION_SEMANTICS_REMAINING_GAP"
    else:
        verdict = "MATERIAL_CORRECTION_INSUFFICIENT__DEEPER_LIMIT"
    manifest = {"contract": "COIN_V1_V2_MATERIAL_RERUN", "date": "2026-07-25", "no_retraining": True,
                "V1": "SINGLE_TIP_LOW_FRICTION_COIN_V1 (as-loaded)", "V2": "RUBBER_TIP_LOW_DRAG_COIN_V2 (calibrated)",
                "n_states": n_states, "summary": summary, "rows": rows,
                "best_transport_delta_V2_minus_V1": round(best_dtd, 4), "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/coin_v1_v2_material_rerun.json", "w"), indent=1, default=float)
    print("\n== V1 → V2 (as-loaded → calibrated material), same controllers, NO retraining ==")
    for c in ctrls:
        s = summary[c]
        print(f"  {c:22s} transport {s['V1']['transport_dist']}→{s['V2']['transport_dist']} | zone "
              f"{s['V1']['zone_entry']}→{s['V2']['zone_entry']} | K6 {s['V1']['k6']}→{s['V2']['k6']} | "
              f"Ft/Fn {s['V1']['ftfn']}→{s['V2']['ftfn']} | coin_pk {s['V1']['peak_coin']}→{s['V2']['peak_coin']}")
    print(f"\n  → {verdict}\n  artifact: {OUT}/coin_v1_v2_material_rerun.json\nCOIN_V1_V2_RERUN_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
