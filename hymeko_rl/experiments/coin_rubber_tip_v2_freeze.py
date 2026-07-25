"""Freeze RUBBER_TIP_LOW_DRAG_COIN_V2 — a physically-motivated coin MATERIAL model, frozen WITHOUT inspecting delivery
(no K6 / zone). Mass fixed; V4_INTERMITTENT_CONTACT motion contract unchanged. Two contact relationships, each calibrated
on physics only:

  tip↔coin  : HIGH friction (rubberised finger). Set on the fingertip geom via geom_priority; the EFFECTIVE friction
              MuJoCo assigns to the fingertip↔disk contact is VERIFIED here (the user-requested separate check).
  coin↔floor: LOW, COULOMB-DOMINANT drag (smooth table). From the multi-velocity coast calibration: a small viscous
              dof_damping (numerical residual) + a Coulomb dof_frictionloss giving a speed-independent μ_eff ≈ 0.15
              (as-loaded was viscous 2.5 ⇒ μ_eff 0.4–1.7, speed-dependent and ~15× too sticky).

Delivery is NOT consulted. The frozen artifact records the material params, the coast-calibration evidence (μ_eff vs
speed), and the verified effective tip↔coin friction. SINGLE_TIP_LOW_FRICTION_COIN_V1 (as-loaded) is untouched.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_pair_scenario import (  # noqa: E402
    effective_tip_coin_friction, set_material, setup_material_decoupling)
from hymeko_rl.coin_delivery.motion_robust_expert import _acquire_direction  # noqa: E402
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation  # noqa: E402
from hymeko_rl.coin_delivery.motion_robust_expert import CarryControllerConfig  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
TIP_MU = 2.0                # rubberised finger — physically plausible high grip (rubber-on-metal μ ≈ 0.8–1.5+)
COIN_VISCOUS = 0.02         # numerical residual only (NOT the main table friction)
COIN_COULOMB = 0.074        # Coulomb table friction → μ_eff ≈ 0.15 (from the multi-velocity coast calibration)


def _verify_effective_tip_friction(pi0, base, forbidden, tip_mu, n_states=3):
    """Drive the tip into the coin (acquire) and read the EFFECTIVE fingertip↔disk contact friction MuJoCo assigns — must
    match the set tip_mu (confirming geom_priority wins the contact). Delivery-blind."""
    acfg = CarryControllerConfig()
    seen = []
    for si in range(n_states):
        rl, _g = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None,
                              seed_lo=14000 + 250 * si, tries=3)
        tg, adr, _bt, _bd = setup_material_decoupling(rl)
        set_material(rl, tg, adr, tip_mu, COIN_VISCOUS, COIN_COULOMB)
        for _ in range(80):                                    # acquire: close the tip→coin gap until contact
            a = 2.0 * _acquire_direction(rl, acfg)
            step_ablation(rl, np.asarray(a, np.float32), "A")
            mu = effective_tip_coin_friction(rl, tg, rl.inner._disk_geom)
            if mu is not None:
                seen.append(mu)
                break
    return seen


def main():
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    v4 = json.load(open(f"{OUT}/dynamics_contract_v4.json"))["frozen_contract"]
    coast = json.load(open(f"{OUT}/coin_coast_calibration.json"))
    assert coast["chosen_coulomb_model"], "no Coulomb model chosen by the coast calibration"

    print(f"verifying effective tip↔coin friction (set tip_mu={TIP_MU}) …", flush=True)
    seen = _verify_effective_tip_friction(pi0, base, forbidden, TIP_MU)
    eff_mu = round(float(np.mean(seen)), 4) if seen else None
    tip_ok = bool(seen and abs(eff_mu - TIP_MU) < 0.25 * TIP_MU)      # effective ≈ set (priority won the contact)
    print(f"  effective tip↔coin μ over {len(seen)} contacts: {seen} (mean {eff_mu}); matches set {TIP_MU}: {tip_ok}", flush=True)

    coast_ok = bool(coast["chosen_params"] and coast["as_loaded_kind"] == "VISCOUS")
    frozen = {"scenario": "RUBBER_TIP_LOW_DRAG_COIN_V2", "based_on_dynamics": v4["dynamics_contract"], "mass": "FIXED",
              "tip_coin_friction": TIP_MU, "effective_tip_coin_friction_measured": eff_mu,
              "coin_slide_viscous_damping": COIN_VISCOUS, "coin_slide_coulomb_frictionloss": COIN_COULOMB,
              "coin_floor_mu_eff_by_v0": coast["chosen_params"]["mu_by_v0"],
              "coin_floor_model": "Coulomb-dominant (dof_frictionloss) + small numerical viscous",
              "motion_contract": "COIN_DYNAMICS_CONTRACT_V4_INTERMITTENT_CONTACT (unchanged)"}
    verdict = "RUBBER_TIP_LOW_DRAG_COIN_V2_FROZEN" if (tip_ok and coast_ok) else "V2_FREEZE_BLOCKED_VERIFICATION_FAILED"
    manifest = {"contract": "RUBBER_TIP_LOW_DRAG_COIN_V2_FREEZE", "date": "2026-07-25",
                "discipline": "material model calibrated on PHYSICS only (coast + effective-friction), delivery NEVER inspected",
                "tip_friction_verification": {"set": TIP_MU, "measured_effective": eff_mu, "ok": tip_ok, "samples": seen},
                "coast_calibration": {"as_loaded_kind": coast["as_loaded_kind"], "as_loaded_mu_by_v0": coast["as_loaded_mu_by_v0"],
                                      "chosen": coast["chosen_coulomb_model"], "chosen_params": coast["chosen_params"]},
                "frozen_material": frozen if verdict.endswith("FROZEN") else None, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/rubber_tip_v2_material.json", "w"), indent=1, default=float)
    print(f"\n  → {verdict}")
    print(f"  material: tip {TIP_MU} (eff {eff_mu}) | coin drag viscous {COIN_VISCOUS} + coulomb {COIN_COULOMB} "
          f"(μ_eff {coast['chosen_params']['mu_by_v0']})")
    print(f"  artifact: {OUT}/rubber_tip_v2_material.json\nRUBBER_TIP_V2_FREEZE_DONE")
    return verdict.endswith("FROZEN")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
