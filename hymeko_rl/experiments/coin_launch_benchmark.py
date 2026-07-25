"""Micro-test 1 — LAUNCH-ONLY benchmark. Isolates the target-directed impulse question from braking/terminal control:
bounded preload → target-directed impulse → release, with predictive braking OFF (observe the coast, do not brake). On
frozen RUBBER_TIP_LOW_DRAG_COIN_V2 + V4 motion contract; no retraining.

For each state, measure (peak_coin_velocity alone is NOT enough — a coin can move fast in the WRONG direction, cf. s7):
  v_target (coast-model launch), peak TARGET-projected velocity, peak CROSS-track velocity, target-directed impulse,
  lateral impulse, signed target displacement, torque saturation, motion-contract pass.

Per-state diagnostic categories (controller-independent → progressively stronger):
  CONTACT_CAPABLE               — the acquire oracle establishes contact
  IMPULSE_CAPABLE               — the coin reaches a launch-scale SPEED
  TARGET_DIRECTED_IMPULSE_CAPABLE — that speed is TOWARD the zone (cross-track small) AND ≥ the 10 cm coast-model launch
The full panel is still reported; this is a mechanism breakdown, not a re-selection.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_capable import acquire_oracle  # noqa: E402
from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.force_slip_carry import ForceSlipConfig, _launch_velocity, force_slip_carry  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
HORIZON = 320
SPEED_FLOOR = 0.45             # m/s — launch-scale speed (≈ 0.9× the 10 cm coast-model launch)
CROSS_FRAC = 0.5               # cross-track must be < this fraction of the target-projected velocity to be "directed"


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
    v_req = _launch_velocity(mu, 0.10, 2.0)                       # coast-model launch for a 10 cm delivery
    launch_cfg = ForceSlipConfig(coast_mu=mu, enable_predictive_brake=False)   # launch → release → observe (no brake)
    gate_lim = v4["abs_velocity_gate"]

    def make_v2(seed):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=seed, tries=3)
        tg, adr, _bt, _bd = setup_material_decoupling(rl)
        set_material(rl, tg, adr, v2["tip_coin_friction"], v2["coin_slide_viscous_damping"], v2["coin_slide_coulomb_frictionloss"])
        return rl, gate

    n_states = 5 if smoke else 8
    print(f"LAUNCH-ONLY benchmark: coast μ={mu:.3f}, 10 cm coast-model launch v_req={v_req:.3f} m/s, motion gate {gate_lim}", flush=True)
    rows = []
    for si in range(n_states):
        seed = 14000 + 250 * si
        rl_o, _g = make_v2(seed)
        orc = acquire_oracle(rl_o, stack)                        # controller-independent contact capability
        rl, gate = make_v2(seed)
        o = force_slip_carry(rl, gate, pi0, base, stack, horizon=HORIZON, cfg=launch_cfg)
        contact_capable = bool(orc["contact_capable"])
        motion_ok = bool(o["peak_joint_vel"] <= gate_lim)
        impulse_capable = bool(contact_capable and o["peak_coin_velocity"] >= SPEED_FLOOR and motion_ok)
        directed = bool(impulse_capable and o["peak_target_velocity"] >= 0.9 * v_req
                        and o["peak_cross_track_velocity"] < CROSS_FRAC * max(1e-6, o["peak_target_velocity"]))
        cat = ("TARGET_DIRECTED_IMPULSE_CAPABLE" if directed else "IMPULSE_CAPABLE" if impulse_capable
               else "CONTACT_CAPABLE" if contact_capable else "NOT_CONTACT_CAPABLE")
        rows.append({"state": si, "seed": seed, "category": cat, "contact_capable": contact_capable,
                     "v_target": o["launch_velocity_target"], "peak_target_velocity": o["peak_target_velocity"],
                     "peak_cross_track_velocity": o["peak_cross_track_velocity"], "peak_coin_velocity": o["peak_coin_velocity"],
                     "target_directed_impulse": o["target_directed_impulse"], "lateral_impulse": o["lateral_impulse"],
                     "signed_target_displacement": o["signed_target_displacement"], "transport_dist": o["transport_dist"],
                     "ftfn": o["ftfn"], "torque_saturation": o.get("governor_active_frac", 0.0),
                     "peak_joint_vel": o["peak_joint_vel"], "motion_contract_pass": motion_ok, "phase_log": o["phase_log"]})
        print(f"  s{si}: {cat:32s} v_tgt {o['launch_velocity_target']} | peak_target {o['peak_target_velocity']} "
              f"cross {o['peak_cross_track_velocity']} | tgt_imp {o['target_directed_impulse']} lat_imp {o['lateral_impulse']} "
              f"| signed_disp {o['signed_target_displacement']} | Ft/Fn {o['ftfn']} | motion {motion_ok}", flush=True)

    cats = {c: [r["state"] for r in rows if r["category"] == c]
            for c in ("TARGET_DIRECTED_IMPULSE_CAPABLE", "IMPULSE_CAPABLE", "CONTACT_CAPABLE", "NOT_CONTACT_CAPABLE")}
    n_directed = len(cats["TARGET_DIRECTED_IMPULSE_CAPABLE"])
    verdict = ("TARGET_DIRECTED_IMPULSE_CAPABILITY_EXISTS_ON_A_SUBSET" if n_directed
               else "IMPULSE_EXISTS_BUT_NOT_TARGET_DIRECTED" if cats["IMPULSE_CAPABLE"]
               else "LAUNCH_CONTROLLER_INSUFFICIENT")
    manifest = {"contract": "COIN_LAUNCH_ONLY_BENCHMARK", "date": "2026-07-25", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2",
                "coast_mu": round(mu, 3), "launch_velocity_required_10cm": round(v_req, 3), "n_states": n_states,
                "diagnostic_categories": cats, "n_target_directed": n_directed, "rows": rows, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/coin_launch_benchmark.json", "w"), indent=1, default=float)
    print("\n== LAUNCH-ONLY diagnostic categories ==")
    for c, states in cats.items():
        print(f"  {c:32s} {states}")
    print(f"\n  → {verdict}\n  artifact: {OUT}/coin_launch_benchmark.json\nCOIN_LAUNCH_BENCHMARK_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
