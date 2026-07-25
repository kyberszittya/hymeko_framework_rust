"""LAUNCH_FEASIBLE_ACQUISITION_SEARCH_V1 — the CORRECTED acquisition benchmark. E1 searched for Fn-balance; E2B proved that
is the wrong option-postcondition (balanced contact ≠ launch-capable contact — a zone-side balanced pair has zero forward
authority). Here the ACQUIRE option terminates successfully only when the next option (LAUNCH) is executable, via a
LEXICOGRAPHIC gate:
  G1  clean contact         — dual-contact dwell, bounded penetration, settled qdot, no torque saturation
  G2  null preload wrench   — realized NET contact force + coin torque ≈ 0 (the physical preload target, not Fn-balance)
  G3  launch feasibility    — forward direction inside the friction cone + directed grasp solve with F∥ ≥ min, low cross
The far-side sign is only a candidate-ordering prior; the decision is G1∧G2∧G3. Frozen V2/V4. No RL. O3 paused.

Verdicts:
  LAUNCH_FEASIBLE_ACQUISITION_EXISTS_ON_A_SUBSET            (≥1 state passes G1∧G2∧G3)
  NO_LAUNCH_FEASIBLE_ACQUISITION_UNDER_CURRENT_SEARCH       (0 states — clean preload / feasibility unmet in ±4 cm)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.cooperative_launch import CooperativeConfig, launch_feasible_acquisition_search  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"


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

    def make(seed):
        rl, _gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=seed, tries=3)
        tg, adr, _bt, _bd = setup_material_decoupling(rl)
        set_material(rl, tg, adr, v2["tip_coin_friction"], v2["coin_slide_viscous_damping"], v2["coin_slide_coulomb_frictionloss"])
        return rl

    n_states = 3 if smoke else 8
    print(f"LAUNCH-FEASIBLE ACQUISITION search (G1 clean ∧ G2 null-wrench ∧ G3 launch-feasible): μ={mu:.3f} "
          f"g2_F≤{cfg.g2_force_max} g2_τ≤{cfg.g2_torque_max} g3_F∥≥{cfg.g3_fpar_min}", flush=True)
    rows = []
    for si in range(n_states):
        seed = 14000 + 250 * si
        srch = launch_feasible_acquisition_search(make(seed), stack, cfg=cfg)
        srch.pop("_best_env", None)
        srch.pop("_best_saved", None)
        cs = srch["candidates"]
        funnel = {g: sum(c[g] for c in cs) for g in ("G1_clean_contact", "G2_null_preload_wrench", "G3_launch_feasible", "done")}
        best = srch["best"]
        rows.append({"state": si, "exists": srch["exists"], "best": best, "funnel": funnel, "candidates": cs})
        print(f"  s{si}: exists={int(srch['exists'])} funnel G1={funnel['G1_clean_contact']} G2={funnel['G2_null_preload_wrench']} "
              f"G3={funnel['G3_launch_feasible']} done={funnel['done']}" + (
              f" | best s={best['s']} F∥={best['launch_cert']['f_parallel']} wrench_F={best['realized_wrench']['force_norm']}"
              if best else ""), flush=True)

    exist_rate = round(float(np.mean([r["exists"] for r in rows])), 3)
    agg = {g: sum(r["funnel"][g] for r in rows) for g in ("G1_clean_contact", "G2_null_preload_wrench", "G3_launch_feasible", "done")}
    verdict = ("LAUNCH_FEASIBLE_ACQUISITION_EXISTS_ON_A_SUBSET" if exist_rate > 0
               else "NO_LAUNCH_FEASIBLE_ACQUISITION_UNDER_CURRENT_SEARCH")
    manifest = {"contract": "LAUNCH_FEASIBLE_ACQUISITION_SEARCH_V1", "date": "2026-07-25", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2",
                "coast_mu": round(mu, 3), "n_states": n_states,
                "gate": {"g2_force_max": cfg.g2_force_max, "g2_torque_max": cfg.g2_torque_max, "g3_fpar_min": cfg.g3_fpar_min,
                         "g3_cross_max": cfg.g3_cross_max, "search_span": cfg.search_span, "search_n": cfg.search_n},
                "exists_rate": exist_rate, "candidate_funnel_total": agg, "verdict": verdict, "rows": rows}
    json.dump(manifest, open(f"{OUT}/launch_feasible_acquisition.json", "w"), indent=1, default=float)
    print("\n== LAUNCH-FEASIBLE ACQUISITION search ==")
    print(f"  states with a G1∧G2∧G3 acquisition: {exist_rate}")
    print(f"  candidate funnel (of {agg and n_states * cfg.search_n} total): {agg}")
    print(f"  → {verdict}\n  artifact: {OUT}/launch_feasible_acquisition.json\nLFA_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
