"""TARGET_DIRECTED_LAUNCH_V1 benchmark. Frozen V2 material + V4 motion contract + the B1 passive barrier; the only thing
that varies is how the tip aims the (already-proven) impulse. Progressive arms:
  L0 tuned launch (push toward the zone)
  L1 target-frame parallel objective        (drive v_∥ → v_∥*)
  L2 L1 + cross-track suppression            (drive v_⊥ → 0)
  L3 L2 + contact-point selection (far side) (contact opposite the zone so the push is target-directed)

PRE-REGISTERED acceptance gate — a launch is TARGET-DIRECTED iff (thresholds fixed before the run):
  v_parallel ≥ 0.85 · v_target   AND   v_parallel > 0   AND   |v_cross| / v_parallel < 0.2
  AND signed target displacement > 0   AND   peak joint velocity ≤ 3.45 rad/s

Per-state metrics: v_target, v_parallel, v_cross, launch-angle error, target-directed vs lateral impulse, signed target
displacement. Two headline plots: v_parallel vs v_target; cross_ratio per state. Verdicts per the decision tree.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.directed_launch import DirectedLaunchConfig, directed_launch_carry  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
HORIZON = 320
V_FRAC, CROSS_MAX, GATE = 0.85, 0.2, 3.45      # pre-registered acceptance thresholds


def _stages(mu):
    return {
        "L0_tuned": DirectedLaunchConfig(coast_mu=mu, enable_directed=False, enable_cross_suppress=False),
        "L1_parallel": DirectedLaunchConfig(coast_mu=mu, enable_directed=True, enable_cross_suppress=False),
        "L2_cross_suppress": DirectedLaunchConfig(coast_mu=mu, enable_directed=True, enable_cross_suppress=True),
        "L3_contact_select": DirectedLaunchConfig(coast_mu=mu, enable_directed=True, enable_cross_suppress=True, enable_contact_select=True),
    }


def _directed(o):
    return bool(o["peak_v_parallel"] >= V_FRAC * o["v_target"] and o["peak_v_parallel"] > 0
                and o["cross_ratio"] < CROSS_MAX and o["signed_target_displacement"] > 0 and o["peak_joint_vel"] <= GATE)


def _plot(rows, stages, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = {"L0_tuned": "#888", "L1_parallel": "#37a", "L2_cross_suppress": "#e73", "L3_contact_select": "#2a7"}
    for name in stages:
        vt = [r[name]["v_target"] for r in rows]
        vp = [r[name]["peak_v_parallel"] for r in rows]
        ax1.scatter(vt, vp, c=colors[name], label=name, s=36)
        ax2.plot([r["state"] for r in rows], [r[name]["cross_ratio"] for r in rows], "o-", c=colors[name], label=name)
    lim = max(max(r[n]["v_target"] for r in rows for n in stages), 0.6)
    ax1.plot([0, lim], [0, lim], "k--", lw=0.8, label="v_par = v_target")
    ax1.plot([0, lim], [0, 0.85 * lim], "r:", lw=0.8, label="0.85 gate")
    ax1.set_xlabel("v_target (coast model)")
    ax1.set_ylabel("peak v_parallel achieved")
    ax1.set_title("target-directed velocity: achieved vs required")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    ax2.axhline(CROSS_MAX, color="r", ls=":", label=f"gate {CROSS_MAX}")
    ax2.set_xlabel("state")
    ax2.set_ylabel("cross_ratio |v_cross|/v_parallel")
    ax2.set_title("cross-track leakage per state (lower = better aimed)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


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
    stages = _stages(mu)

    def make_v2(seed):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=seed, tries=3)
        tg, adr, _bt, _bd = setup_material_decoupling(rl)
        set_material(rl, tg, adr, v2["tip_coin_friction"], v2["coin_slide_viscous_damping"], v2["coin_slide_coulomb_frictionloss"])
        return rl, gate

    n_states = 4 if smoke else 8
    print(f"DIRECTED LAUNCH: μ={mu:.3f}, gate v_par≥{V_FRAC}·v_target ∧ cross<{CROSS_MAX} ∧ signed_disp>0 ∧ joint≤{GATE}", flush=True)
    rows = []
    for si in range(n_states):
        seed = 14000 + 250 * si
        rec = {"state": si}
        for name, cfg in stages.items():
            rl, gate = make_v2(seed)
            o = directed_launch_carry(rl, gate, pi0, base, stack, horizon=HORIZON, cfg=cfg)
            o["target_directed"] = _directed(o)
            rec[name] = o
        rows.append(rec)
        print("  s%d: " % si + " | ".join(
            f"{n.split('_')[0]} vpar{rec[n]['peak_v_parallel']}/xr{rec[n]['cross_ratio']}/{'DIR' if rec[n]['target_directed'] else '-'}"
            for n in stages), flush=True)

    def n_dir(name):
        return sum(r[name]["target_directed"] for r in rows)
    dir_by_stage = {name: n_dir(name) for name in stages}
    l0, l2, l3 = dir_by_stage["L0_tuned"], dir_by_stage["L2_cross_suppress"], dir_by_stage["L3_contact_select"]

    def mean_xr(name):
        return round(float(np.mean([r[name]["cross_ratio"] for r in rows])), 3)
    xr_by_stage = {name: mean_xr(name) for name in stages}
    # decision tree
    if l2 > l0 and l2 >= max(2, n_states // 2):
        verdict = "TARGET_FRAME_FORCE_DIRECTION_CONTROL_RECOVERS_LAUNCH"
    elif l3 > l2 and l3 >= max(2, n_states // 2):
        verdict = "CONTACT_POINT_SELECTION_IS_LOAD_BEARING"
    elif (l3 > 0 or l2 > 0) or xr_by_stage["L3_contact_select"] < xr_by_stage["L0_tuned"] - 0.05:
        verdict = "DIRECTIONAL_CAPABILITY_EXISTS__STATE_CONDITIONING_REMAINS_INSUFFICIENT"
    else:
        verdict = "SINGLE_POINT_CONTACT_GEOMETRY_LIMITS_DIRECTIONAL_CONTROL"
    manifest = {"contract": "TARGET_DIRECTED_LAUNCH_V1", "date": "2026-07-25", "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2",
                "coast_mu": round(mu, 3), "acceptance_gate": {"v_frac": V_FRAC, "cross_max": CROSS_MAX, "joint_gate": GATE},
                "n_states": n_states, "n_target_directed_by_stage": dir_by_stage, "mean_cross_ratio_by_stage": xr_by_stage,
                "rows": rows, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/directed_launch_benchmark.json", "w"), indent=1, default=float)
    _plot(rows, stages, f"{OUT}/directed_launch_benchmark.png")
    print("\n== TARGET-DIRECTED LAUNCH — target-directed count / mean cross-ratio by stage ==")
    for name in stages:
        print(f"  {name:20s} directed {dir_by_stage[name]}/{n_states} | mean cross_ratio {xr_by_stage[name]}")
    print(f"\n  → {verdict}\n  artifact: {OUT}/directed_launch_benchmark.json\nDIRECTED_LAUNCH_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
