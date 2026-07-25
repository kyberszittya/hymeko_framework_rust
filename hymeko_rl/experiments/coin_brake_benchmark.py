"""Micro-test 2 — BRAKE-ONLY benchmark with CONTROLLED initial conditions. The coin is placed at a known remaining
distance from the zone (4/6/8/10/12 cm) and given a known velocity toward it (0.3/0.4/0.5/0.6 m/s); the arm starts from a
config from which a braking contact is executable (fingertip driven to the zone side). Progressive arms B0–B5 isolate
whether PREDICTIVE RE-CONTACT BRAKING can stop the coin INSIDE the zone. Frozen RUBBER_TIP_LOW_DRAG_COIN_V2 + V4; no
retraining. Reports the explicit event chain so a stage that never fires is not read as a stage that failed.

Decision tree (per overshoot cells, where free coast would overshoot the zone):
  brake lands the coin in the zone where coast overshoots          → PREDICTIVE_RECONTACT_BRAKING_CAPABILITY_ESTABLISHED
  re-contact acquired but coin not stopped in zone (impulse issue) → BRAKING_IMPULSE_EXISTS__TERMINAL_MODULATION_INSUFFICIENT
  brake condition crossed but re-contact never acquired            → RECONTACT_GEOMETRY_OR_TIMING_IS_THE_WALL
  coin stoppable but does not stay in the zone                     → LOW_SPEED_TERMINAL_CORRECTION_REQUIRED
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.brake_control import _MODES, BrakeConfig, brake_carry  # noqa: E402
from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
D_REMAINING = (0.04, 0.06, 0.08, 0.10, 0.12)
V0S = (0.3, 0.4, 0.5, 0.6)


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
    bcfg = BrakeConfig(coast_mu=mu)

    def make_v2(seed):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=seed, tries=3)
        tg, adr, _bt, _bd = setup_material_decoupling(rl)
        set_material(rl, tg, adr, v2["tip_coin_friction"], v2["coin_slide_viscous_damping"], v2["coin_slide_coulomb_frictionloss"])
        return rl, gate

    states = [14000] if smoke else [14000, 14250]
    ds = (0.04, 0.06) if smoke else (0.04, 0.05, 0.06)
    vs = (0.5, 0.6) if smoke else (0.5, 0.6)
    print(f"BRAKE-ONLY (controlled): μ={mu:.3f}, d_remaining {ds}, v0 {vs}, modes {len(_MODES)}", flush=True)
    rows = []
    for seed in states:
        for dr in ds:
            for v0 in vs:
                cell = {"seed": seed, "d_remaining": dr, "v0": v0, "coast_stop": round(v0 * v0 / (2 * mu * 9.81), 4)}
                for mode in _MODES:
                    rl, gate = make_v2(seed)
                    cell[mode] = brake_carry(rl, gate, pi0, base, stack, mode=mode, d_remaining=dr, v0=v0, cfg=bcfg)
                rows.append(cell)
                b0, b5 = cell["B0_coast"], cell["B5_settle"]
                print(f"  seed{seed} d{dr} v{v0} (coast {cell['coast_stop']}): B0 err {b0['signed_stop_error']} zone {b0['entered_zone']} "
                      f"| B5 err {b5['signed_stop_error']} zone {b5['entered_zone']} recon {b5['recontacted']} "
                      f"ci {b5['counter_impulse']} dwell {b5['in_zone_dwell']}", flush=True)

    # focus on OVERSHOOT cells (free coast would overshoot the zone ⇒ braking has something to correct)
    overshoot = [r for r in rows if r["B0_coast"]["overshoot"]]
    scope = overshoot or rows

    def zrate(mode):
        return round(float(np.mean([r[mode]["entered_zone"] for r in scope])), 3)
    zone_by_mode = {mode: zrate(mode) for mode in _MODES}
    recon_rate = round(float(np.mean([r["B5_settle"]["recontacted"] for r in scope])), 3)
    trigger_rate = round(float(np.mean([r["B5_settle"]["events"]["brake_condition_crossed"] for r in scope])), 3)
    dwell_rate = round(float(np.mean([r["B5_settle"]["in_zone_dwell"] > 5 for r in scope])), 3)
    b0z = zone_by_mode["B0_coast"]
    best_mode = max((m for m in _MODES if m != "B0_coast"), key=lambda m: zone_by_mode[m])
    best_z = zone_by_mode[best_mode]
    active_z = max(zone_by_mode[m] for m in ("B2_recontact", "B3_counter_impulse", "B4_terminal_correction", "B5_settle"))
    if best_z > b0z + 0.15 and best_z >= 0.5:
        # a braking mode lands the coin in the zone where coast does not
        verdict = ("PREDICTIVE_PASSIVE_BARRIER_BRAKING_CAPABILITY_ESTABLISHED" if best_mode == "B1_passive_landing"
                   else "PREDICTIVE_RECONTACT_BRAKING_CAPABILITY_ESTABLISHED")
    elif recon_rate > 0.3 and active_z < 0.5:
        verdict = "BRAKING_IMPULSE_EXISTS__TERMINAL_MODULATION_INSUFFICIENT" if dwell_rate > 0.2 \
            else "LOW_SPEED_TERMINAL_CORRECTION_REQUIRED"
    elif trigger_rate > 0.3 and recon_rate <= 0.3:
        verdict = "RECONTACT_GEOMETRY_OR_TIMING_IS_THE_WALL"
    else:
        verdict = "BRAKE_CONDITION_NOT_EXERCISED_IN_SCOPE"
    manifest = {"contract": "COIN_BRAKE_ONLY_BENCHMARK", "date": "2026-07-25", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2",
                "coast_mu": round(mu, 3), "d_remaining": list(ds), "v0s": list(vs), "n_overshoot_cells": len(overshoot),
                "zone_entry_by_mode": zone_by_mode, "recontact_rate": recon_rate, "brake_trigger_rate": trigger_rate,
                "in_zone_dwell_rate": dwell_rate, "best_braking_mode": best_mode, "best_mode_zone_entry": best_z, "rows": rows, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/coin_brake_benchmark.json", "w"), indent=1, default=float)
    print(f"\n== BRAKE-ONLY (overshoot cells: {len(overshoot)}/{len(rows)}) — zone entry by mode ==")
    for mode in _MODES:
        print(f"  {mode:26s} zone {zone_by_mode[mode]}")
    print(f"  trigger crossed {trigger_rate} | re-contact {recon_rate} | in-zone dwell {dwell_rate}")
    print(f"\n  → {verdict}\n  artifact: {OUT}/coin_brake_benchmark.json\nCOIN_BRAKE_BENCHMARK_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
