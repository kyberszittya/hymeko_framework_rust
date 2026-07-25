"""BIMANUAL CRADLE EMBODIMENT AUDIT — the decision the whole E3 arc turns on: is a STRADDLING, equilibrium-feasible
two-finger cradle REACHABLE with this arm placement, or must the contact strategy change? Every wrench-null controller
(proportional / soft-LS / hard-QP) nulled ‖w‖ by contact loss because the acquired grasp presses from the SAME side
(n_L·n_R > 0), where a net-zero internal force is geometrically impossible. Here we drive a STRADDLE-directed grasp (each
tip to the opposite side of the coin along a squeeze axis) and test the DEFINITIVE certificate: does an admissible internal
force exist (∃ f: G·f=0, cone-feasible, Fn ≥ F_min)? Frozen V2/V4. No RL. O3 paused.

Three outcomes (per the analysis):
  STRADDLING_CRADLE_REACHABLE                    — some (state, axis) reaches both-contact AND a feasible cradle certificate
  STRADDLE_FORCE_FEASIBLE_BUT_NOT_REACHABLE      — the straddle target is always force-feasible, but the arms cannot reach it
  NULL_PRELOAD_CRADLE_NOT_AVAILABLE_MORPHOLOGY   — (subsumed by the above; distinguishing needs posture/approach analysis)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.cooperative_launch import (  # noqa: E402
    CooperativeConfig, straddle_directed_acquire, tip_midpoint)
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
    print(f"CRADLE EMBODIMENT AUDIT — straddle-directed grasp + internal-force certificate: μ={mu:.3f}", flush=True)
    rows = []
    for si in range(n_states):
        seed = 14000 + 250 * si
        rl0 = make(seed)
        mid = tip_midpoint(rl0)
        u, _dtz = rl0.inner.direction_to_zone()
        e_par = np.asarray(u, np.float64)
        axis_opts = {"zone_cross": np.array([-e_par[1], e_par[0]]), "zone_par": e_par}
        best = None
        attempts = {}
        for name, ax in axis_opts.items():
            rl = make(seed)
            out = straddle_directed_acquire(rl, stack, mid.astype(np.float32), ax, cfg=cfg)
            cert = out["cradle_certificate"]
            reachable_cradle = bool(out["both_contact"] and cert["feasible"])
            attempts[name] = {"both_contact": out["both_contact"], "both_dwell": out["both_dwell"],
                              "n_dot": out["straddle"]["n_dot"], "cert_feasible": cert["feasible"],
                              "cone_admissible": cert["cone_admissible"], "reachable_cradle": reachable_cradle}
            if reachable_cradle and best is None:
                best = name
        rows.append({"state": si, "best_axis": best, "attempts": attempts})
        z, t = attempts["zone_cross"], attempts["zone_par"]
        print(f"  s{si}: zone_cross[both{int(z['both_contact'])} ndot{z['n_dot']} cert{int(z['cert_feasible'])}] "
              f"zone_par[both{int(t['both_contact'])} ndot{t['n_dot']} cert{int(t['cert_feasible'])}] → cradle={best}", flush=True)

    reachable = sum(1 for r in rows if r["best_axis"] is not None)
    any_straddle = sum(1 for r in rows if any(a["n_dot"] < 0 for a in r["attempts"].values()))
    rate = round(reachable / n_states, 3)
    if reachable > 0:
        verdict = "STRADDLING_CRADLE_REACHABLE"
    elif any_straddle > 0:
        verdict = "STRADDLE_REACHED_BUT_NO_FEASIBLE_CRADLE_CERTIFICATE"
    else:
        verdict = "NULL_PRELOAD_CRADLE_NOT_REACHABLE_UNDER_CURRENT_MORPHOLOGY"
    manifest = {"contract": "BIMANUAL_CRADLE_EMBODIMENT_AUDIT", "date": "2026-07-26", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2",
                "coast_mu": round(mu, 3), "n_states": n_states, "reachable_cradle_states": reachable,
                "straddle_reached_states": any_straddle, "reachable_rate": rate, "verdict": verdict, "rows": rows}
    json.dump(manifest, open(f"{OUT}/cradle_embodiment_audit.json", "w"), indent=1, default=float)
    print("\n== CRADLE EMBODIMENT AUDIT ==")
    print(f"  reachable straddling cradle: {reachable}/{n_states} | straddle geometry reached: {any_straddle}/{n_states}")
    print(f"  → {verdict}\n  artifact: {OUT}/cradle_embodiment_audit.json\nAUDIT_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
