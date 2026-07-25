"""BIMANUAL_ACQUISITION_CURRICULUM_V1 — rung E0, decomposed into three separately-validated stages (the earlier bundled
"co-contact 8/8" was a FALSE POSITIVE: the tips buried 18–34 mm into the hard-pinned coin and the positive launch velocity
was a pin-release spring explosion, not a clean bimanual impulse). Frozen V2/V4/coast. No RL.

  E0a  clean bounded bilateral PRELOAD  — pinned surface-target acquisition (r_coin+r_tip−δ, radii read from model),
                                          settle qdot, gate: bounded penetration ∧ balanced Fn ∧ settled ∧ no saturation.
  E0b  release-only SANITY              — from the clean preload, unpin with NO launch command; the coin must not jump
                                          (rules out any residual spring-release exploit).
  E0c  A0 vs A2 directed LAUNCH         — only from a clean, release-sane preload: A0 coin-twist allocation vs A2
                                          grasp-matrix resultant-force allocation.

Verdicts:
  CLEAN_BIMANUAL_PRELOAD_CAPABILITY_ESTABLISHED        (E0a)
  RELEASE_ONLY_SANITY_PASS                             (E0b)
  BIMANUAL_COOPERATIVE_LAUNCH_CAPABILITY_ESTABLISHED   (E0c — best allocator target-directed)

Launch gate: clean preload ∧ not jumped ∧ v_∥ ≥ 0.85·v_target ∧ |v_cross|/v_∥ < 0.2 ∧ |ω| < 0.5 ∧ joint ≤ 3.45.
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.cooperative_launch import (  # noqa: E402
    CooperativeConfig, GraspAllocator, TwistAllocator, acquire_clean_preload, cooperative_launch_carry, place_coin_at,
    release_only_sanity, release_pin, static_reachability_probe, tip_midpoint)
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
HORIZON = 180
V_FRAC, CROSS_MAX, GATE, OMEGA_MAX = 0.85, 0.2, 3.45, 0.5
ALLOCATORS = {"A0_twist": TwistAllocator, "A2_grasp": GraspAllocator}


def _directed(o, vt, clean, jumped):
    return bool(clean and not jumped and o["peak_v_parallel"] >= V_FRAC * vt and o["cross_ratio"] < CROSS_MAX
               and o["peak_omega"] < OMEGA_MAX and o["peak_joint_vel"] <= GATE and o["both_tips_contact"] == 1)


def _contact_fingerprint(rl):
    """The contact-model parameters that set the preload physics — saved so the clean-preload gate stays reproducible if
    the MuJoCo contact model, geometry, or timestep ever changes (per the reproducibility requirement)."""
    import mujoco
    m = rl.inner.model
    dj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    return {"timestep": float(m.opt.timestep), "coin_radius": float(m.geom_size[dj][0]), "tip_radius": float(m.geom_size[gl][0]),
            "disk_solref": m.geom_solref[dj].tolist(), "disk_solimp": m.geom_solimp[dj].tolist(),
            "disk_margin": float(m.geom_margin[dj]), "disk_gap": float(m.geom_gap[dj]), "disk_friction": m.geom_friction[dj].tolist()}


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
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=seed, tries=3)
        tg, adr, _bt, _bd = setup_material_decoupling(rl)
        set_material(rl, tg, adr, v2["tip_coin_friction"], v2["coin_slide_viscous_damping"], v2["coin_slide_coulomb_frictionloss"])
        return rl

    n_states = 4 if smoke else 8
    print(f"E0 decomposed (E0a preload → E0b release-sanity → E0c A0/A2 launch): μ={mu:.3f}, δ={cfg.preload_depth} m", flush=True)
    fingerprint = _contact_fingerprint(make(14000))
    rows = []
    for si in range(n_states):
        seed = 14000 + 250 * si
        rl0 = make(seed)
        place_coin_at(rl0, tip_midpoint(rl0))
        st = static_reachability_probe(rl0, stack, cfg)
        vt = float(min(0.8, np.sqrt(max(0.0, 2.0 * mu * 9.81 * rl0._dtz()))))
        acq = acquire_clean_preload(rl0, stack, cfg=cfg)              # E0a — rl0 left at settled soft-pinned preload
        clean, saved = acq["clean"], acq["pin_saved"]
        row = {"state": si, "v_target": round(vt, 3), "static_two_reachable": st["two_contact_reachable"],
               "E0a_acquired": acq["acquired"], "E0a_clean": clean, "preload": acq["preload"]}
        if acq["acquired"]:
            row["E0b_release"] = release_only_sanity(copy.deepcopy(rl0), stack, saved, cfg=cfg)   # E0b — releases the pin itself
            for name, Alloc in ALLOCATORS.items():                   # E0c — each from a fresh copy, pin released onto a free coin
                rlc = copy.deepcopy(rl0)
                release_pin(rlc, saved)
                o = cooperative_launch_carry(rlc, None, pi0, base, stack, horizon=HORIZON, cfg=cfg, allocator=Alloc(cfg))
                o["directed"] = _directed(o, vt, clean, row["E0b_release"]["jumped"])
                row[name] = {k: o[k] for k in ("both_tips_contact", "both_contact_frames", "peak_v_parallel",
                                               "cross_ratio", "peak_omega", "peak_joint_vel", "directed")}
        pl = acq["preload"] or {}
        rel = row.get("E0b_release", {})
        print(f"  s{si}: clean={int(clean)} pen[{pl.get('penetration_left')},{pl.get('penetration_right')}] "
              f"qdot={pl.get('qdot_prerelease')} fnbal={pl.get('fn_balance')} | E0b jump={int(bool(rel.get('jumped')))} "
              f"spd{rel.get('coin_peak_speed')} | " + (
              f"A0 v{row['A0_twist']['peak_v_parallel']} xr{row['A0_twist']['cross_ratio']} d{int(row['A0_twist']['directed'])} "
              f"A2 v{row['A2_grasp']['peak_v_parallel']} xr{row['A2_grasp']['cross_ratio']} d{int(row['A2_grasp']['directed'])}"
              if acq["acquired"] else "no-acquire"), flush=True)
        rows.append(row)

    clean_rate = round(float(np.mean([r["E0a_clean"] for r in rows])), 3)
    rel_rows = [r for r in rows if r["E0a_clean"] and "E0b_release" in r]
    release_pass = round(float(np.mean([not r["E0b_release"]["jumped"] for r in rel_rows])), 3) if rel_rows else 0.0
    dir_by = {name: sum(r.get(name, {}).get("directed", False) for r in rows) for name in ALLOCATORS}
    best = max(dir_by, key=dir_by.get)
    dir_rate = round(dir_by[best] / n_states, 3)
    preload_verdict = ("CLEAN_BIMANUAL_PRELOAD_CAPABILITY_ESTABLISHED" if clean_rate >= 0.75
                       else "CLEAN_BOUNDED_PRELOAD_NOT_YET_ESTABLISHED")
    release_verdict = ("RELEASE_ONLY_SANITY_PASS" if release_pass >= 0.75 and rel_rows
                       else "RELEASE_ONLY_SANITY_FAIL_SPRING_RESIDUE" if rel_rows else "RELEASE_ONLY_SANITY_NO_CLEAN_STATES")
    launch_verdict = ("BIMANUAL_COOPERATIVE_LAUNCH_CAPABILITY_ESTABLISHED" if dir_rate >= 0.5
                      else "COOPERATIVE_FORCE_ALLOCATION_NOT_YET_TARGET_DIRECTED")
    manifest = {"contract": "BIMANUAL_ACQUISITION_CURRICULUM_V1_E0", "date": "2026-07-25", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2",
                "coast_mu": round(mu, 3), "preload_depth": cfg.preload_depth, "n_states": n_states,
                "clean_gate": {"pen_min": cfg.clean_pen_min, "pen_max": cfg.clean_pen_max, "qdot_max": cfg.clean_qdot_max,
                               "fn_min": cfg.clean_fn_min, "fn_balance_min": cfg.clean_fn_balance_min},
                "contact_fingerprint": fingerprint,
                "E0a_clean_preload_rate": clean_rate, "E0b_release_pass_rate": release_pass,
                "E0c_directed_by_allocator": dir_by, "E0c_best_allocator": best, "E0c_directed_rate": dir_rate,
                "E0a_verdict": preload_verdict, "E0b_verdict": release_verdict, "E0c_verdict": launch_verdict,
                "rows": rows, "teacher_records": rows}
    json.dump(manifest, open(f"{OUT}/bimanual_curriculum_e0.json", "w"), indent=1, default=float)
    print("\n== E0 (decomposed) ==")
    print(f"  E0a clean bounded preload: {clean_rate}  → {preload_verdict}")
    print(f"  E0b release-only sanity (no jump | clean states): {release_pass}  → {release_verdict}")
    print(f"  E0c directed launch by allocator: {dir_by}  (best {best}: {dir_rate})  → {launch_verdict}")
    print(f"  artifact: {OUT}/bimanual_curriculum_e0.json\nE0_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
