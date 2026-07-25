"""RUBBER_TIP_LOW_DRAG_COIN_V1 — does a physically-motivated MATERIAL change (not an algorithmic trick) unlock the coin
transport wall found under SINGLE_TIP_LOW_FRICTION_COIN_V1? Mass is held FIXED; only the two contact relationships vary:

    tip↔coin friction  (rubberised finger — grip)      : {1.0, 1.5, 2.0, 3.0}× baseline (fingertip geom, priority-won)
    coin↔floor drag     (smooth table — easy to slide)  : {1.0, 0.75, 0.5, 0.25}× baseline (coin slide dof_damping ≈ 2.5)

Run with the frozen V4_INTERMITTENT_CONTACT dynamics + the full intermittent controller. Per-cell metrics (over N states):
transport distance, zone entry, K6, peak coin speed (not a projectile), terminal coin speed (braking/settling still
possible), re-contact episodes, Ft/Fn, motion-legal-in-contact, torque saturation.

Selection is NOT by K6 alone (§3 metric integrity): first find the PHYSICALLY MEANINGFUL region where Ft/Fn or transport
rises, the coin does NOT become a projectile (peak coin speed bounded), and braking/settling stay possible (terminal coin
speed low). Versioned SEPARATELY from SINGLE_TIP_LOW_FRICTION_COIN_V1 — the negative result is preserved, so this cleanly
shows whether the material change (not the controller) moves the wall.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.intermittent_carry import IntermittentConfig, intermittent_carry  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
TIP_SCALES = (1.0, 1.5, 2.0, 3.0)
DRAG_SCALES = (1.0, 0.75, 0.5, 0.25)
N_STATES, HORIZON = 5, 320
COIN_LAUNCH_MAX = 1.0          # m/s — above this the coin is a projectile (pre-declared, from V4)
COIN_SETTLE_MAX = 0.15         # m/s — terminal coin speed below which braking/settling remains possible
ABS_GATE = 3.45                # arm contact-phase peak-velocity gate (from V4)


def _cell(pi0, base, forbidden, stack, tip_scale, drag_scale):
    cfg = IntermittentConfig()
    rows, applied = [], None
    for si in range(N_STATES):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None,
                                seed_lo=14000 + 250 * si, tries=3)
        tg, adr, base_tip, base_drag = setup_material_decoupling(rl)
        set_material(rl, tg, adr, base_tip * tip_scale, base_drag * drag_scale)
        applied = {"tip_mu": round(base_tip * tip_scale, 3), "coin_slide_damping": round(base_drag * drag_scale, 3)}
        o = intermittent_carry(rl, gate, pi0, base, stack, horizon=HORIZON, cfg=cfg)
        rows.append({"transport": o["transport_dist"], "zone": o["entered_zone"], "k6": o["k6"],
                     "peak_coin": o["peak_coin_speed"], "term_coin": o["terminal_coin_speed"],
                     "episodes": o["n_contact_episodes"], "peak_in_contact": o["peak_joint_vel_in_contact"],
                     "Fn": o["peak_contact_normal_force"], "Ft": o["peak_contact_tangential_force"],
                     "sat": o["torque_saturation_frac"], "acq": o["acquired_contact"]})

    def mean(k):
        return round(float(np.mean([r[k] for r in rows])), 4)
    peak_coin, term_coin = mean("peak_coin"), mean("term_coin")
    motion_legal = bool(max(r["peak_in_contact"] for r in rows) <= ABS_GATE)
    return {"applied": applied, "transport": mean("transport"), "zone": mean("zone"), "k6": mean("k6"),
            "peak_coin": peak_coin, "term_coin": term_coin, "episodes": mean("episodes"), "acq": mean("acq"),
            "Fn": mean("Fn"), "Ft": mean("Ft"), "ftfn": round(mean("Ft") / (mean("Fn") + 1e-6), 3),
            "sat": mean("sat"), "motion_legal": motion_legal,
            # physically-sensible ⇔ not a projectile ∧ can still settle ∧ arm motion-legal
            "physically_sensible": bool(peak_coin <= COIN_LAUNCH_MAX and term_coin <= COIN_SETTLE_MAX and motion_legal),
            "per_state": rows}


def _plot(grid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("transport", "transport dist (m)"), ("zone", "zone entry"), ("ftfn", "Ft/Fn"),
            ("peak_coin", "peak coin speed (m/s)"), ("term_coin", "terminal coin speed"), ("sat", "torque saturation")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (k, title) in zip(axes.ravel(), keys):
        M = np.array([[grid[f"t{ts}_d{ds}"][k] for ds in DRAG_SCALES] for ts in TIP_SCALES])
        im = ax.imshow(M, cmap="viridis", origin="lower", aspect="auto")
        ax.set_xticks(range(len(DRAG_SCALES)))
        ax.set_xticklabels([f"{d}×" for d in DRAG_SCALES])
        ax.set_yticks(range(len(TIP_SCALES)))
        ax.set_yticklabels([f"{t}×" for t in TIP_SCALES])
        ax.set_xlabel("coin↔floor drag scale")
        ax.set_ylabel("tip↔coin friction scale")
        ax.set_title(title, fontsize=9)
        for i in range(len(TIP_SCALES)):
            for j in range(len(DRAG_SCALES)):
                ax.text(j, i, f"{M[i, j]:.3g}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("RUBBER_TIP_LOW_DRAG_COIN: tip friction × coin slide-drag (frozen V4, intermittent controller)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    v4 = json.load(open(f"{OUT}/dynamics_contract_v4.json"))["frozen_contract"]
    assert v4["dynamics_contract"] == "COIN_DYNAMICS_CONTRACT_V4_INTERMITTENT_CONTACT", v4["dynamics_contract"]
    stack = V3Stack(v4["qdot_soft"], v4["qdot_hard"], v4["armature"], v4["damping"], v4["friction"],
                    v4["kp"], v4["kv"], v4["tau_rate"], over_hard_brake=v4["over_hard_brake"])
    grid = {}
    print("RUBBER_TIP_LOW_DRAG sweep (tip friction × coin slide-drag), frozen V4, intermittent controller:", flush=True)
    for ts in TIP_SCALES:
        for ds in DRAG_SCALES:
            c = _cell(pi0, base, forbidden, stack, ts, ds)
            grid[f"t{ts}_d{ds}"] = c
            print(f"  tip {ts}× drag {ds}× (mu {c['applied']['tip_mu']} damp {c['applied']['coin_slide_damping']}): "
                  f"transport {c['transport']} zone {c['zone']} k6 {c['k6']} | coin_pk {c['peak_coin']} term {c['term_coin']} "
                  f"| Ft/Fn {c['ftfn']} | legal {c['motion_legal']} sensible {c['physically_sensible']} | sat {c['sat']}", flush=True)
    base_cell = grid["t1.0_d1.0"]
    # physically-sensible cells that IMPROVE transport over the baseline material (not chosen by K6 alone)
    improved = {k: v for k, v in grid.items()
                if v["physically_sensible"] and v["transport"] > base_cell["transport"] + 0.005}
    best = max(improved, key=lambda k: grid[k]["transport"]) if improved else None
    if best is None:
        verdict = "MATERIAL_CHANGE_DOES_NOT_UNLOCK_TRANSPORT_WITHIN_SENSIBLE_REGION"
    elif grid[best]["zone"] > base_cell["zone"] + 0.15:
        verdict = "RUBBER_TIP_LOW_DRAG_UNLOCKS_TRANSPORT_AND_ZONE_ENTRY"
    else:
        verdict = "RUBBER_TIP_LOW_DRAG_IMPROVES_TRANSPORT_BUT_NOT_YET_ZONE_ENTRY"
    manifest = {"contract": "RUBBER_TIP_LOW_DRAG_COIN_V1", "date": "2026-07-25", "based_on_dynamics": v4["dynamics_contract"],
                "mass": "FIXED", "material_model": "tip↔coin friction (fingertip geom, priority) × coin↔floor slide drag (disk dof_damping)",
                "tip_scales": list(TIP_SCALES), "drag_scales": list(DRAG_SCALES),
                "selection": "physically-sensible (coin not launched, terminal settle-able, arm motion-legal) AND transport improves; NOT K6 alone",
                "baseline_material_cell": "t1.0_d1.0", "grid": grid, "best_sensible_improved_cell": best, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/rubber_tip_lowdrag.json", "w"), indent=1, default=float)
    _plot(grid, f"{OUT}/rubber_tip_lowdrag.png")
    print(f"\n  baseline material (t1.0_d1.0): transport {base_cell['transport']} zone {base_cell['zone']} k6 {base_cell['k6']}")
    print(f"  best sensible improved cell: {best}" + (f" → transport {grid[best]['transport']} zone {grid[best]['zone']} k6 {grid[best]['k6']}" if best else ""))
    print(f"  → {verdict}\n  artifact: {OUT}/rubber_tip_lowdrag.json\nRUBBER_TIP_LOWDRAG_DONE")
    return verdict


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
