"""Mass × friction characterization of the coin CONTACT class (delivery-independent). The intermittent-contact finding
raised the question: is the coin push-and-coast because of the OBJECT physics (low mass, low floor friction, single-tip
point contact) rather than the actuation stack? This sweeps object mass {0.5×, 1×, 2×} × floor/coin friction {0.5×, 1×,
2×} and measures, under the SAME delivery-agnostic oscillating-press driver (governor-only velocity control), the contact
regime — NOT delivery. K6 / zone are never read.

Per-cell metrics (user-specified):
  * contact time (frames);
  * contact normal impulse (∫Fn dt);
  * object speed after contact (peak coin speed);
  * target-directed displacement (max coin displacement from start along the push axis);
  * re-contact (number of contact episodes, mean gap);
  * actuator saturation fraction;
  * contact force DIRECTION — peak NORMAL vs peak TANGENTIAL (a normal peak alone is a press/collision spike, not the
    shear that actually drags a low-friction coin).

Expected pattern (hypothesis, to be confirmed): low mass / low friction → object escapes, short contact; high mass /
friction → longer contact but higher force needed; too high → actuator saturates, object barely moves. A physically
sensible intermediate regime is expected.
"""
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.motion_robust_expert import CarryControllerConfig, motion_robust_carry  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
MASS_SCALES = (0.5, 1.0, 2.0)
FRIC_SCALES = (0.5, 1.0, 2.0)
HORIZON, N_STATES = 200, 3


def _scale_coin(rl, mass_scale, fric_scale):
    """Scale the coin body mass+inertia and the coin geom's sliding friction on the model in place. Returns the (base,
    scaled) mass and friction actually applied. mj_forward recomputes the derived mass matrix / subtree quantities."""
    m = rl.inner.model
    gid = rl.inner._disk_geom
    bid = int(m.geom_bodyid[gid])
    base_mass = float(m.body_mass[bid])
    base_fric = float(m.geom_friction[gid, 0])
    m.body_mass[bid] = base_mass * mass_scale
    m.body_inertia[bid] *= mass_scale
    m.geom_friction[gid, 0] = base_fric * fric_scale       # component 0 = sliding (tangential) friction coefficient
    mujoco.mj_forward(m, rl.inner.data)
    return {"base_mass": round(base_mass, 5), "scaled_mass": round(base_mass * mass_scale, 5),
            "base_friction": round(base_fric, 4), "scaled_friction": round(base_fric * fric_scale, 4)}


def _cell(pi0, base, forbidden, stack, mass_scale, fric_scale):
    cfg = CarryControllerConfig(sustained_press=True, enable_braking=False, replan_every=4)
    rows, applied = [], None
    for si in range(N_STATES):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None,
                                seed_lo=14000 + 300 * si, tries=2)
        applied = _scale_coin(rl, mass_scale, fric_scale)
        disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
        maxdisp = {"d": 0.0}

        def hook(_ph, _s, _rl=rl, _d0=disk0, _md=maxdisp):        # max coin displacement from start (target-directed proxy)
            disp = float(np.linalg.norm(np.asarray(_rl.inner._planar_metrics.disk_pos, np.float32)[:2] - _d0))
            _md["d"] = max(_md["d"], disp)
        o = motion_robust_carry(rl, gate, pi0, base, stack, horizon=HORIZON, cfg=cfg, frame_hook=hook)
        ep = o["n_contact_episodes"]
        rows.append({"contact_frames": o["contact_frames"], "normal_impulse": o["contact_normal_impulse"],
                     "peak_coin_speed": o["peak_coin_speed"], "max_coin_disp": round(maxdisp["d"], 4),
                     "n_episodes": ep, "mean_gap": round(o["contact_frames"] / ep, 1) if ep else 0.0,
                     "sat_frac": o["torque_saturation_frac"], "peak_Fn": o["peak_contact_normal_force"],
                     "peak_Ft": o["peak_contact_tangential_force"]})

    def mean(k):
        return round(float(np.mean([r[k] for r in rows])), 4)
    return {"applied": applied, "contact_frames": mean("contact_frames"), "normal_impulse": mean("normal_impulse"),
            "peak_coin_speed": mean("peak_coin_speed"), "max_coin_disp": mean("max_coin_disp"),
            "n_episodes": mean("n_episodes"), "sat_frac": mean("sat_frac"), "peak_Fn": mean("peak_Fn"),
            "peak_Ft": mean("peak_Ft"), "tangential_ratio": round(mean("peak_Ft") / (mean("peak_Fn") + 1e-6), 3),
            "per_state": rows}


def _plot(grid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("contact_frames", "contact frames"), ("peak_coin_speed", "peak coin speed (m/s)"),
            ("max_coin_disp", "max coin disp (m)"), ("sat_frac", "actuator saturation"),
            ("normal_impulse", "normal impulse (N·s)"), ("tangential_ratio", "Ft/Fn ratio")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (k, title) in zip(axes.ravel(), keys):
        M = np.array([[grid[f"m{ms}_f{fs}"][k] for fs in FRIC_SCALES] for ms in MASS_SCALES])
        im = ax.imshow(M, cmap="viridis", origin="lower", aspect="auto")
        ax.set_xticks(range(len(FRIC_SCALES)))
        ax.set_xticklabels([f"{f}×" for f in FRIC_SCALES])
        ax.set_yticks(range(len(MASS_SCALES)))
        ax.set_yticklabels([f"{m}×" for m in MASS_SCALES])
        ax.set_xlabel("friction")
        ax.set_ylabel("mass")
        ax.set_title(title, fontsize=9)
        for i in range(len(MASS_SCALES)):
            for j in range(len(FRIC_SCALES)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Coin contact-class characterization: mass × friction (delivery-independent oscillating press)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    v3 = json.load(open(f"{OUT}/dynamics_contract_v3_agile.json"))["frozen_contract"]
    stack = V3Stack(v3["qdot_soft"], v3["qdot_hard"], v3["armature"], v3["damping"], v3["friction"],
                    v3["kp"], v3["kv"], v3["tau_rate"], over_hard_brake=1.5)
    grid = {}
    print("mass×friction characterization (delivery-independent oscillating press):", flush=True)
    for ms in MASS_SCALES:
        for fs in FRIC_SCALES:
            c = _cell(pi0, base, forbidden, stack, ms, fs)
            grid[f"m{ms}_f{fs}"] = c
            print(f"  mass {ms}× fric {fs}×: frames {c['contact_frames']} | coin_speed {c['peak_coin_speed']} | "
                  f"disp {c['max_coin_disp']} | episodes {c['n_episodes']} | sat {c['sat_frac']} | "
                  f"Fn {c['peak_Fn']} Ft {c['peak_Ft']} (Ft/Fn {c['tangential_ratio']}) | imp {c['normal_impulse']}", flush=True)
    # coarse regime read (measured, not a verdict): where is contact longest and the coin controllable (not launched)?
    best = max(grid, key=lambda k: grid[k]["contact_frames"] * (grid[k]["peak_coin_speed"] <= 1.0))
    manifest = {"contract": "COIN_MASS_FRICTION_CHARACTERIZATION", "date": "2026-07-25",
                "delivery_independent": True, "driver": "oscillating press (sustained_press), governor-only velocity control",
                "mass_scales": list(MASS_SCALES), "friction_scales": list(FRIC_SCALES), "grid": grid,
                "longest_contact_controllable_cell": best}
    json.dump(manifest, open(f"{OUT}/coin_mass_friction.json", "w"), indent=1, default=float)
    _plot(grid, f"{OUT}/coin_mass_friction.png")
    print(f"\nlongest-contact controllable cell: {best}")
    print(f"artifact: {OUT}/coin_mass_friction.json\nCOIN_MASS_FRICTION_DONE")
    return grid


if __name__ == "__main__":
    main()
