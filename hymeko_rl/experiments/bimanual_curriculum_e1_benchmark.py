"""BIMANUAL_ACQUISITION_CURRICULUM_V1 — rung E1: does an AUTHORITY-BALANCED preload configuration EXIST? E0 showed the
geometric tip-midpoint is not in general the point where the two arms carry equal normal force (asymmetric Jacobian
conditioning / torque headroom / reachable force). Before building any force-balance controller, decide EXISTENCE: a
delivery-blind 1-D search along the fingertip→fingertip axis for a coin position with a clean, BALANCED, real-force
preload. Then E2: from that point, release-only sanity must show no spring residue. Frozen V2/V4. No RL. O3 paused.

Verdicts:
  E1  BALANCED_PRELOAD_CONFIGURATION_EXISTS (+ GEOMETRIC_MIDPOINT_IS_NOT_THE_AUTHORITY_CENTRE if the best s ≠ 0)
      | CURRENT_CONTACT_GEOMETRY_LACKS_BALANCED_AUTHORITY
  E2  NO_SPRING_RESIDUE_ON_RELEASE  |  SPRING_RESIDUE_REMAINS_AT_BALANCED_POINT
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.cooperative_launch import (  # noqa: E402
    CooperativeConfig, balanced_preload_search, release_only_sanity)
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
    print(f"E1 authority-balance existence oracle: sweep ±{cfg.search_span} m × {cfg.search_n} along tip-tip axis; "
          f"balanced = clean ∧ total_Fn≥{cfg.preload_min_total} ∧ imbalance≤{cfg.imbalance_max}", flush=True)
    rows = []
    for si in range(n_states):
        seed = 14000 + 250 * si
        srch = balanced_preload_search(make(seed), stack, cfg=cfg)
        best_env, best_saved = srch.pop("_best_env"), srch.pop("_best_saved")   # strip live envs before the row is serialised
        row = {"state": si, "exists": srch["exists"], "best": srch["best"], "candidates": srch["candidates"]}
        best = srch["best"]
        if srch["exists"]:                                        # E2 — release-only sanity from the EXACT validated preload,
            row["E2_hold"] = release_only_sanity(copy.deepcopy(best_env), stack, best_saved, cfg=cfg, retract=False)  # both controllers
            row["E2_retract"] = release_only_sanity(best_env, stack, best_saved, cfg=cfg, retract=True)
        imb_min = min((c["imbalance"] for c in srch["candidates"] if c["acquired"]), default=None)
        h, r = row.get("E2_hold") or {}, row.get("E2_retract") or {}
        print(f"  s{si}: exists={int(srch['exists'])} best_s={best['s'] if best else None} "
              f"imb={best['imbalance'] if best else None} total_fn={best['total_fn'] if best else None} | "
              f"E2 hold jump={int(bool(h.get('jumped'))) if h else '-'} disp={h.get('coin_displacement') if h else '-'} | "
              f"retract jump={int(bool(r.get('jumped'))) if r else '-'} disp={r.get('coin_displacement') if r else '-'}", flush=True)
        rows.append(row)

    exist_rate = round(float(np.mean([r["exists"] for r in rows])), 3)
    off_centre = [r["best"]["s"] for r in rows if r["exists"] and r["best"] is not None]
    midpoint_is_authority = bool(off_centre and all(abs(s) < 1e-6 for s in off_centre))
    e2_rows = [r for r in rows if r["exists"] and r.get("E2_retract")]
    hold_pass = round(float(np.mean([not r["E2_hold"]["jumped"] for r in e2_rows])), 3) if e2_rows else 0.0
    e2_pass = round(float(np.mean([not r["E2_retract"]["jumped"] for r in e2_rows])), 3) if e2_rows else 0.0
    exists_any = bool(any(r["exists"] for r in rows))
    if exist_rate >= 0.5:
        e1_verdict = ("BALANCED_PRELOAD_CONFIGURATION_EXISTS" if midpoint_is_authority
                      else "BALANCED_PRELOAD_CONFIGURATION_EXISTS__GEOMETRIC_MIDPOINT_IS_NOT_THE_AUTHORITY_CENTRE")
    elif exists_any:
        e1_verdict = "BALANCED_PRELOAD_EXISTS_BUT_RARE_AND_OFF_MIDPOINT__CONTACT_GEOMETRY_LACKS_BROAD_BALANCED_AUTHORITY"
    else:
        e1_verdict = "CURRENT_CONTACT_GEOMETRY_LACKS_BALANCED_AUTHORITY"
    e2_verdict = ("NO_SPRING_RESIDUE_ON_RELEASE" if e2_pass >= 0.75 and e2_rows
                  else "SPRING_RESIDUE_REMAINS_AT_BALANCED_POINT" if e2_rows else "NO_BALANCED_POINT_TO_RELEASE_TEST")
    manifest = {"contract": "BIMANUAL_ACQUISITION_CURRICULUM_V1_E1", "date": "2026-07-25", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2",
                "coast_mu": round(mu, 3), "n_states": n_states, "search": {"span": cfg.search_span, "n": cfg.search_n},
                "balance_gate": {"preload_min_total": cfg.preload_min_total, "imbalance_max": cfg.imbalance_max},
                "E1_exists_rate": exist_rate, "E1_verdict": e1_verdict, "E2_hold_pass_rate": hold_pass,
                "E2_retract_pass_rate": e2_pass, "E2_verdict": e2_verdict, "rows": rows}
    json.dump(manifest, open(f"{OUT}/bimanual_curriculum_e1.json", "w"), indent=1, default=float)
    print("\n== E1 authority-balance existence oracle ==")
    print(f"  balanced config exists: {exist_rate} (off-midpoint: {not midpoint_is_authority})  → {e1_verdict}")
    print(f"  E2 release no-jump — hold {hold_pass} vs retract {e2_pass}  → {e2_verdict}")
    print(f"  artifact: {OUT}/bimanual_curriculum_e1.json\nE1_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
