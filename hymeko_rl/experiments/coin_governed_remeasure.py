"""G1 — re-measure the frozen coin baselines under the FROZEN COIN_DYNAMICS_CONTRACT_V2, WITHOUT retraining.

The dynamics contract was calibrated on physics only (G0) and is now frozen; here we ONLY measure. Same panel + paired
seeds + budgets as the legacy runs, but under the governed dynamics (armature/damping on the model + the directional
torque governor via mjcb_control, set-then-reset — the frozen baseline files are never mutated). The expert is RE-SEARCHED
under the new dynamics (never a legacy-fast θ replay). Emits the pre-registered 3-outcome verdict + a dynamics_contract
manifest.
"""
import copy
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")

from hymeko_rl.coin_delivery.coin_carry_proposal import load_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_carry_rollout, structured_random_best_with_support  # noqa: E402
from hymeko_rl.env.motion_contract import TorqueGovernorConfig, govern_torque  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

from coin_balltip_proposal import D  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
CYL_PROP = f"{D}/carry_proposal_balltip_fresh_v1.pt"
N, EVAL_H = 4, 260
V2 = {"qdot_soft": 1.5, "qdot_hard": 3.0, "armature": 0.4, "damping": 15.0, "friction": 0.1,
      "tau_rate": 25.0, "control_dt": 0.01, "substeps": 20}
GOV = TorqueGovernorConfig(V2["qdot_soft"], V2["qdot_hard"])


def _apply_v2(rl):
    m = rl.inner.model
    m.dof_armature[:N] = V2["armature"]
    m.dof_damping[:N] = V2["damping"]
    m.dof_frictionloss[:N] = V2["friction"]


def _governor_cb(model, data):
    data.ctrl[:N] = govern_torque(data.ctrl[:N], data.qvel[:N], GOV)


def _measure(rl, gate, pi0, base, theta):
    """Roll a committed θ under the governor; return K6 + peak/terminal joint velocity + contact fraction."""
    vv = {"peak": 0.0, "contact": 0, "n": 0, "term": 0.0}
    dxadr = rl.inner._disk_x_adr

    def hook(_ph, _s):
        d = r2.inner.data
        v = float(np.max(np.abs(d.qvel[:N])))
        vv["peak"] = max(vv["peak"], v)
        vv["term"] = v
        mtr = r2.inner._planar_metrics
        vv["contact"] += int(bool(mtr.left_contact or mtr.right_contact))
        vv["n"] += 1
        vv["obj_speed"] = float(np.linalg.norm(d.qvel[dxadr:dxadr + 2]))
    r2 = copy.deepcopy(rl)
    o = structured_carry_rollout(r2, copy.deepcopy(gate), pi0, base, theta, horizon=EVAL_H, frame_hook=hook)
    return {"k6": int(o["k6"]), "max_dwell": int(o["max_dwell"]), "reached_handoff": int(o["reached_handoff"]),
            "peak_joint_vel": round(vv["peak"], 2), "terminal_joint_vel": round(vv["term"], 3),
            "contact_frac": round(vv["contact"] / max(1, vv["n"]), 2), "obj_speed": round(vv.get("obj_speed", 0.0), 2)}


def _arm(rl, gate, pi0, base, kind, prop):
    """Re-SEARCH under the governed dynamics (governor active during search + eval), then measure the committed θ."""
    mujoco.set_mjcb_control(_governor_cb)
    try:
        if kind == "deploy":
            th, _o = search_select(copy.deepcopy(rl), copy.deepcopy(gate), prop.theta(rl.obs()), pi0, base,
                                   np.random.default_rng(7), b=8, horizon=EVAL_H)
        else:  # expert — re-searched under the new dynamics
            th, _o, _s = structured_random_best_with_support(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base,
                                                             np.random.default_rng(7), shots=192, horizon=EVAL_H)
        m = _measure(rl, gate, pi0, base, th)
    finally:
        mujoco.set_mjcb_control(None)
    return m


def main(smoke=False):
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    prop = load_proposal(CYL_PROP)
    n_states = 6 if smoke else 16
    rows = []
    for si in range(n_states):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None,
                                seed_lo=14000 + 250 * si, tries=3)
        _apply_v2(rl)
        rows.append({"state": si, "deploy": _arm(rl, gate, pi0, base, "deploy", prop),
                     "expert": _arm(rl, gate, pi0, base, "expert", prop)})
        print(f"  state {si}: deploy k6 {rows[-1]['deploy']['k6']} vel {rows[-1]['deploy']['peak_joint_vel']} | "
              f"expert k6 {rows[-1]['expert']['k6']} vel {rows[-1]['expert']['peak_joint_vel']}", flush=True)

    def agg(arm, key):
        return round(float(np.mean([r[arm][key] for r in rows])), 3)
    summary = {a: {"k6_rate": agg(a, "k6"), "mean_peak_vel": agg(a, "peak_joint_vel"),
                   "mean_terminal_vel": agg(a, "terminal_joint_vel"), "mean_contact_frac": agg(a, "contact_frac"),
                   "mean_obj_speed": agg(a, "obj_speed")} for a in ("deploy", "expert")}
    dep, exp = summary["deploy"]["k6_rate"], summary["expert"]["k6_rate"]
    if exp >= 0.5 and dep >= 0.4:
        verdict = "GOVERNED_COIN_CONTROL_RECHARACTERIZED"
    elif exp >= 0.5 and dep < 0.4:
        verdict = "PHYSICAL_CAPABILITY_RETAINED__DEPLOY_POLICY_MISMATCH_UNDER_GOVERNED_DYNAMICS"
    else:
        verdict = "LEGACY_COIN_SOLUTION_DEPENDED_ON_UNREALISTIC_DYNAMICS"
    manifest = {"contract": "COIN_GOVERNED_REMEASURE_G1", "date": "2026-07-25", "smoke": smoke,
                "dynamics_contract": {"name": "COIN_DYNAMICS_CONTRACT_V2", "motion_limit_version": "V1", **V2},
                "note": "expert re-searched under governed dynamics; NO retraining; frozen files not mutated",
                "n_states": n_states, "summary": summary, "rows": rows, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/coin_governed_remeasure.json", "w"), indent=1, default=float)
    print(f"\n== G1 governed re-measurement ==  deploy K6 {dep} | expert K6 {exp}\n  → {verdict}")
    print(f"  artifact: {OUT}/coin_governed_remeasure.json\nCOIN_GOVERNED_REMEASURE_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
