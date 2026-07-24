"""C2 — coin option-language ablation on the FROZEN V4_CONTACT_AGILE dynamics. The physics is now capable (safe + agile
+ contact-robust); this isolates WHICH control element restores the last ~10 cm of transport. Same panel/seeds/V4/
governor for every arm; the primary result is the PHASE LADDER, not final K6.

Arms (per the pre-registered plan):
  A legacy            — the open-loop push→brake→release macro (expert search) under V4
  B slower open-loop  — the same macro, amplitudes scaled down (gentler, longer push)
  C feedback transport — closed-loop contact-retaining transport, NO braking (enable_braking=False, no replan)
  D C + vel-aware brake/release — closed-loop + braking-distance-gated brake/release (no replan)
  E D + receding-horizon — closed-loop + brake/release + Jacobian replanning (full C1)

Phase ladder: acquire → retained transport distance (coin displacement toward the zone) → zone entry → low-speed release
→ settling → K6. Also logs peak joint velocity (motion-contract check on every rollout).
"""
import copy
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")

from hymeko_rl.coin_delivery.coin_carry_structured import (  # noqa: E402
    CENTER_TOL, structured_carry_rollout, structured_random_best_with_support)
from hymeko_rl.coin_delivery.motion_robust_expert import CarryControllerConfig, motion_robust_carry  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.env.motion_contract import govern_torque  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
EVAL_H = 320


def _stack():
    v4 = json.load(open(f"{OUT}/dynamics_contract_v4.json"))["frozen_contract"]
    return V3Stack(v4["qdot_soft"], v4["qdot_hard"], v4["armature"], v4["damping"], v4["friction"],
                   v4["kp"], v4["kv"], v4["tau_rate"], over_hard_brake=v4["over_hard_brake"]), v4


def _coin_progress(rl, disk0, zone_dir):
    """Signed coin displacement toward the zone since the start (retained transport distance)."""
    disp = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2] - disk0
    return float(disp @ zone_dir)


def _legacy_arm(rl, gate, pi0, base, stack, *, slow):
    """Open-loop legacy macro under V4 (governor active). ``slow`` scales the amplitudes down (arm B)."""
    m = rl.inner.model
    m.dof_armature[:4] = stack.armature
    m.dof_damping[:4] = stack.damping
    m.dof_frictionloss[:4] = stack.friction

    def cb(_model, data):
        data.ctrl[:4] = govern_torque(data.ctrl[:4], data.qvel[:4], stack.gov)
    mujoco.set_mjcb_control(cb)
    try:
        th, _o, _s = structured_random_best_with_support(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base,
                                                         np.random.default_rng(7), shots=128, horizon=EVAL_H)
        if slow:
            th = th.copy()
            th[:12] *= 0.55                                   # gentler amplitudes ⇒ slower, longer push
        disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
        u, _n = rl.inner.direction_to_zone()
        prog = {"max": 0.0}

        def hook(_ph, _s):
            prog["max"] = max(prog["max"], _coin_progress(r2, disk0, np.asarray(u, np.float32)))
        r2 = copy.deepcopy(rl)
        o = structured_carry_rollout(r2, copy.deepcopy(gate), pi0, base, th, horizon=EVAL_H, frame_hook=hook)
    finally:
        mujoco.set_mjcb_control(None)
    return {"k6": int(o["k6"]), "acquired": int(o["touched"]), "transport_dist": round(prog["max"], 4),
            "zone_entry": int(o["reached_handoff"] or o["max_dwell"] > 0), "max_dwell": int(o["max_dwell"])}


def _closed_loop_arm(rl, gate, pi0, base, stack, cfg):
    """Closed-loop C1 (arms C/D/E via cfg toggles). motion_robust_carry sets V4 dynamics + governor itself."""
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    u, _n = rl.inner.direction_to_zone()
    prog = {"max": 0.0}

    def hook(_ph, _s):
        prog["max"] = max(prog["max"], _coin_progress(r2, disk0, np.asarray(u, np.float32)))
    r2 = copy.deepcopy(rl)
    o = motion_robust_carry(r2, copy.deepcopy(gate), pi0, base, stack, horizon=EVAL_H, cfg=cfg, frame_hook=hook)
    return {"k6": int(o["k6"]), "acquired": int(o["acquired_contact"]), "transport_dist": round(prog["max"], 4),
            "zone_entry": int(o["entered_zone"]), "max_dwell": int(o["max_dwell"]), "peak_vel": o["peak_joint_vel"]}


ARMS = {
    "A_legacy": lambda rl, g, p, b, s: _legacy_arm(rl, g, p, b, s, slow=False),
    "B_legacy_slow": lambda rl, g, p, b, s: _legacy_arm(rl, g, p, b, s, slow=True),
    "C_feedback": lambda rl, g, p, b, s: _closed_loop_arm(rl, g, p, b, s, CarryControllerConfig(enable_braking=False, replan_every=10 ** 9)),
    "D_brake": lambda rl, g, p, b, s: _closed_loop_arm(rl, g, p, b, s, CarryControllerConfig(enable_braking=True, replan_every=10 ** 9)),
    "E_replan": lambda rl, g, p, b, s: _closed_loop_arm(rl, g, p, b, s, CarryControllerConfig(enable_braking=True, replan_every=4)),
}


def main(smoke=False):
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    stack, v4 = _stack()
    n_states = 6 if smoke else 16
    rows = []
    for si in range(n_states):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None,
                                seed_lo=14000 + 250 * si, tries=3)
        rec = {"state": si}
        for name, fn in ARMS.items():
            rec[name] = fn(rl, gate, pi0, base, stack)
        rows.append(rec)
        print(f"  s{si}: " + " | ".join(f"{k[:1]} acq{rec[k]['acquired']} td{rec[k]['transport_dist']} zone{rec[k]['zone_entry']} k6{rec[k]['k6']}"
                                        for k in ARMS), flush=True)

    def agg(name, key):
        return round(float(np.mean([r[name][key] for r in rows])), 3)
    summary = {name: {k: agg(name, k) for k in ("acquired", "transport_dist", "zone_entry", "k6")} for name in ARMS}
    td = {n: summary[n]["transport_dist"] for n in ARMS}
    ze = {n: summary[n]["zone_entry"] for n in ARMS}
    k6 = {n: summary[n]["k6"] for n in ARMS}
    # pre-registered verdict on the phase ladder (transport distance / zone entry / K6)
    if k6["E_replan"] > 0.3 and k6["E_replan"] > k6["D_brake"] + 0.15:
        verdict = "RECEDING_HORIZON_REPLANNING_REQUIRED_UNDER_CONTACT_COVARIATE_SHIFT"
    elif k6["D_brake"] > 0.3:
        verdict = "VELOCITY_AWARE_BRAKING_AND_RELEASE_LOAD_BEARING"
    elif ze["C_feedback"] > max(ze["A_legacy"], ze["B_legacy_slow"]) + 0.15:
        verdict = "CLOSED_LOOP_CONTACT_RETENTION_RECOVERS_TRANSPORT"
    elif td["B_legacy_slow"] > td["A_legacy"] + 0.02:
        verdict = "LEGACY_OPTION_TIMING_TOO_IMPULSIVE"
    else:
        verdict = "CURRENT_TRANSPORT_OPTION_LANGUAGE_INSUFFICIENT_UNDER_REALISTIC_CONTACT_DYNAMICS"
    manifest = {"contract": "COIN_C2_OPTION_LANGUAGE_ABLATION", "date": "2026-07-25", "smoke": smoke,
                "dynamics_contract": v4, "n_states": n_states, "summary": summary, "rows": rows, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/c2_ablation.json", "w"), indent=1, default=float)
    print("\n== C2 phase ladder (mean) ==")
    for n in ARMS:
        print(f"  {n:16s} acquire {summary[n]['acquired']} transport-dist {summary[n]['transport_dist']} "
              f"zone-entry {summary[n]['zone_entry']} K6 {summary[n]['k6']}")
    print(f"\n  → {verdict}\n  artifact: {OUT}/c2_ablation.json\nC2_ABLATION_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
