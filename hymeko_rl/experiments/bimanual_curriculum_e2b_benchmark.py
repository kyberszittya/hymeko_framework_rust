"""E2B_BASELINE_SUBTRACTED_ALLOCATOR_V1 — the cheap diagnostic that asks whether the A2 grasp-matrix allocation actually
beats A0 twist, WITHOUT first assuming the preload release can be made clean (E3). E1 found 3 states with a clean balanced
preload (s1@+0.03, s5@+0.01, s7@−0.03); each has an irreducible ~1.5 cm passive-release residue (E2). Here, from each
validated preload snapshot we run THREE bit-identical branches over a SHORT fixed horizon — P (passive hold), A0 (twist),
A2 (grasp) — and subtract the common pin-supported drift: incremental_X = X_branch − X_passive. The short horizon keeps the
comparison BEFORE the trajectories diverge into different contact modes (baseline subtraction is not literally linear
afterwards). Frozen V2/V4. No RL. O3 paused.

Credited (primary): incremental target-directed coin velocity, lateral velocity, spin ω, motion-contract, saturation.
NOT the allocator proof: raw zone-entry / K6 (the passive drift makes threshold delivery metrics misleading).

Verdicts:
  GRASP_MATRIX_ALLOCATION_PRODUCES_SUPERIOR_BASELINE_SUBTRACTED_TARGET_WRENCH   (A2 > A0 target-directed, ≥2/3, no blowup)
  WRENCH_ALLOCATION_NOT_THE_CURRENT_BOTTLENECK                                  (A2 ≈ A0)
  ALLOCATOR_ADVANTAGE_DEPENDS_ON_CONTACT_FRAME_AND_LOCAL_ACTUATION_AUTHORITY    (sign flips by state)
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

import mujoco  # noqa: E402

from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.cooperative_launch import (  # noqa: E402
    CooperativeConfig, GraspAllocator, TwistAllocator, _tip_xy, balanced_preload_search, measure_release_branch)
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402


def _contact_frame_side(rl):
    """Which side of the coin (relative to the zone direction e_par) each tip acquired. (p_tip − coin)·e_par > 0 means the
    tip is on the ZONE-FACING side, where pressing (Fn ≥ 0) pushes the coin AWAY from the zone — no feasible +e_par wrench.
    A target-directed launch needs the tips on the FAR (−e_par) side. Reports each projection + whether the frame affords
    a forward push at all."""
    m = rl.inner.model
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    c = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2]
    u, _dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(u, np.float32)
    proj_l = float((_tip_xy(rl, gl) - c) @ e_par)
    proj_r = float((_tip_xy(rl, gr) - c) @ e_par)
    return {"proj_left": round(proj_l, 4), "proj_right": round(proj_r, 4),
            "far_side": bool(proj_l < 0 and proj_r < 0), "zone_side": bool(proj_l > 0 and proj_r > 0)}

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
HORIZON = 40
VALIDATED_STATES = [1, 5, 7]                                      # the E1 balanced-preload states
MARGIN = 0.03                                                    # m/s — target-directed advantage that counts as a win


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

    states = VALIDATED_STATES[:1] if smoke else VALIDATED_STATES
    print(f"E2B baseline-subtracted allocator (P vs A0 vs A2), horizon {HORIZON}, states {states}", flush=True)
    rows = []
    for si in states:
        seed = 14000 + 250 * si
        srch = balanced_preload_search(make(seed), stack, cfg=cfg)
        if not srch["exists"]:
            print(f"  s{si}: no balanced preload (skipped)", flush=True)
            continue
        env, saved = srch["_best_env"], srch["_best_saved"]
        side = _contact_frame_side(copy.deepcopy(env))            # which side of the coin the balanced preload acquired
        branches = {"P": None, "A0": TwistAllocator(cfg), "A2": GraspAllocator(cfg)}
        meas = {name: measure_release_branch(copy.deepcopy(env), stack, saved, cfg=cfg, allocator=al, horizon=HORIZON)
                for name, al in branches.items()}
        p = meas["P"]
        inc = {name: {"d_v_par": round(meas[name]["peak_v_parallel"] - p["peak_v_parallel"], 4),
                      "d_v_cross": round(meas[name]["peak_v_cross"] - p["peak_v_cross"], 4),
                      "d_omega": round(meas[name]["peak_omega"] - p["peak_omega"], 4)} for name in ("A0", "A2")}
        a2_engaged = bool(any(abs(inc["A2"][k]) > 1e-4 for k in ("d_v_par", "d_v_cross", "d_omega")))
        rows.append({"state": si, "best_s": srch["best"]["s"], "imbalance": srch["best"]["imbalance"],
                     "contact_frame_side": side, "a2_engaged": a2_engaged, "measured": meas, "incremental": inc})
        print(f"  s{si}: frame far_side={int(side['far_side'])} zone_side={int(side['zone_side'])} "
              f"proj[{side['proj_left']},{side['proj_right']}] | passive v_par {p['peak_v_parallel']} | "
              f"inc_A0 dv{inc['A0']['d_v_par']} dx{inc['A0']['d_v_cross']} | inc_A2 dv{inc['A2']['d_v_par']} dx{inc['A2']['d_v_cross']} "
              f"engaged={int(a2_engaged)}", flush=True)

    # per-state: does A2 add more target-directed impulse than A0 (with margin), without blowing lateral/spin?
    def a2_wins(r):
        a2, a0 = r["incremental"]["A2"], r["incremental"]["A0"]
        directed = a2["d_v_par"] > a0["d_v_par"] + MARGIN
        no_blowup = a2["d_v_cross"] <= max(a0["d_v_cross"], 0.0) + abs(a2["d_v_par"]) and a2["d_omega"] < 0.5
        return bool(directed and no_blowup and r["measured"]["A2"]["motion_contract_pass"])

    def a0_wins(r):
        a0, a2 = r["incremental"]["A0"], r["incremental"]["A2"]
        return bool(a0["d_v_par"] > a2["d_v_par"] + MARGIN)

    n = len(rows)
    a2w = sum(a2_wins(r) for r in rows)
    a0w = sum(a0_wins(r) for r in rows)
    similar = n - a2w - a0w
    zone_side = sum(r["contact_frame_side"]["zone_side"] for r in rows)
    a2_engaged_n = sum(r["a2_engaged"] for r in rows)
    if n and a2_engaged_n == 0 and zone_side >= max(2, (n + 1) // 2):
        # A2 correctly refuses (zero force) because the balanced frame is on the ZONE-facing side — no feasible +e_par push
        verdict = "BALANCED_PRELOAD_FRAME_DOES_NOT_AFFORD_TARGET_DIRECTED_WRENCH__CONTACT_SIDE_DOMINATES_ALLOCATOR"
    elif n and a2w >= max(2, (n + 1) // 2) and a0w == 0:
        verdict = "GRASP_MATRIX_ALLOCATION_PRODUCES_SUPERIOR_BASELINE_SUBTRACTED_TARGET_WRENCH"
    elif a2w > 0 and a0w > 0:
        verdict = "ALLOCATOR_ADVANTAGE_DEPENDS_ON_CONTACT_FRAME_AND_LOCAL_ACTUATION_AUTHORITY"
    else:
        verdict = "WRENCH_ALLOCATION_NOT_THE_CURRENT_BOTTLENECK"
    manifest = {"contract": "E2B_BASELINE_SUBTRACTED_ALLOCATOR_V1", "date": "2026-07-25", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2",
                "coast_mu": round(mu, 3), "horizon": HORIZON, "margin": MARGIN, "n_states": n,
                "a2_wins": a2w, "a0_wins": a0w, "similar": similar, "zone_side": zone_side, "a2_engaged": a2_engaged_n,
                "verdict": verdict, "rows": rows}
    json.dump(manifest, open(f"{OUT}/bimanual_curriculum_e2b.json", "w"), indent=1, default=float)
    print("\n== E2B baseline-subtracted allocator ==")
    print(f"  A2 engaged {a2_engaged_n}/{n} | zone-side frames {zone_side}/{n} | A2 wins {a2w}/{n}, A0 wins {a0w}/{n}")
    print(f"  → {verdict}\n  artifact: {OUT}/bimanual_curriculum_e2b.json\nE2B_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
