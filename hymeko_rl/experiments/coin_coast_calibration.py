"""Coin coast calibration (multi-velocity, model-discriminating) — is the coin↔floor drag PHYSICALLY realistic, and is it
VISCOUS or COULOMB? These are not interchangeable: viscous ``dof_damping`` gives a force ∝ velocity (so the Coulomb-
equivalent μ_eff RISES with speed), while Coulomb ``dof_frictionloss`` gives a ≈ constant sliding force (μ_eff FLAT across
speed). A hard coin on a smooth table is Coulomb with μ ≈ 0.1–0.2; the as-loaded model uses viscous damping 2.5.

Method: inject a fixed coin speed into the free scene (arm held), and measure the EARLY average deceleration over a short,
distance-capped clean window (before the coin can leave its lane / touch the arm — which contaminated the earlier
full-stop test at low drag). μ_eff = avg_decel / g. Sweep candidate drag models × v0 ∈ {0.2, 0.5, 1.0, 1.5}:
  * μ_eff RISING with v0        ⇒ VISCOUS (wrong for a sliding coin);
  * μ_eff FLAT across v0        ⇒ COULOMB (the physically-correct table friction).
Pick the Coulomb ``dof_frictionloss`` whose (flat) μ_eff sits in the realistic band. Read the coin position from RAW
``qpos`` (``_planar_metrics`` is cached, only recomputes in ``step_ablation``).
"""
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
G = 9.81
V0S = (0.2, 0.5, 1.0, 1.5)
CLEAN_DIST = 0.05          # m — measure the early decel over this unobstructed distance (before lane/arm contamination)
MAX_STEPS = 4000
REALISTIC_MU = (0.05, 0.25)
# candidate drag models: (label, viscous dof_damping, Coulomb dof_frictionloss)
MODELS = [("viscous_asloaded_2.5", 2.5, 0.0), ("viscous_0.1", 0.1, 0.0),
          ("coulomb_mu0.075", 0.02, 0.037), ("coulomb_mu0.15", 0.02, 0.074), ("coulomb_mu0.30", 0.02, 0.147)]


def _early_decel(rl, adr, v0, viscous, coulomb):
    """Inject v0 into the free coin, step until it has travelled CLEAN_DIST (or slowed/stopped), and return the average
    deceleration over that unobstructed window + μ_eff = decel/g. Robust to later contamination (window ends early)."""
    m, d = rl.inner.model, rl.inner.data
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    m.dof_damping[adr:adr + 2] = viscous
    m.dof_frictionloss[adr:adr + 2] = coulomb
    d.ctrl[:4] = d.qpos[:4]
    d.qvel[adr:adr + 2] = [v0, 0.0]
    mujoco.mj_forward(m, d)
    p0 = d.qpos[adr:adr + 2].copy()
    t, dist, v = 0.0, 0.0, v0
    for _ in range(MAX_STEPS):
        d.ctrl[:4] = d.qpos[:4]
        mujoco.mj_step(m, d)
        v = float(np.linalg.norm(d.qvel[adr:adr + 2]))
        dist = float(np.linalg.norm(d.qpos[adr:adr + 2] - p0))
        t += m.opt.timestep
        if dist >= CLEAN_DIST or v < 0.02:
            break
    decel = (v0 - v) / t if t > 1e-6 else 0.0
    return {"v0": v0, "avg_decel": round(decel, 3), "mu_eff": round(decel / G, 3), "window_dist": round(dist, 4),
            "window_t": round(t, 4), "v_end": round(v, 3)}


def _base_adr():
    pi0, base, forbidden = _setup()
    rl, _g = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=14300, tries=2)
    return rl, int(rl.inner._disk_x_adr)


def _classify(mu_by_v0):
    """VISCOUS ⇒ μ_eff rises with v0 (ratio hi/lo ≫ 1); COULOMB ⇒ ~flat (ratio ≈ 1)."""
    lo, hi = mu_by_v0[V0S[0]], mu_by_v0[V0S[-1]]
    ratio = hi / (lo + 1e-6)
    kind = "VISCOUS" if ratio > 1.6 else ("COULOMB" if ratio < 1.35 else "MIXED")
    return kind, round(ratio, 2)


def _plot(grid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, cells in grid.items():
        ax.plot(list(V0S), [cells["mu_by_v0"][v] for v in V0S], "o-", label=f"{label} [{cells['kind']}]")
    ax.axhspan(REALISTIC_MU[0], REALISTIC_MU[1], alpha=0.15, color="g", label="realistic μ 0.05–0.25")
    ax.set_xlabel("initial coin speed v0 (m/s)")
    ax.set_ylabel("effective μ (early decel / g)")
    ax.set_title("Coin coast calibration: μ_eff vs speed — FLAT ⇒ Coulomb, RISING ⇒ viscous")
    ax.set_ylim(0, min(4, max(cells["mu_by_v0"][V0S[-1]] for cells in grid.values()) * 1.1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    rl, adr = _base_adr()
    print(f"coin slide DOF adr={adr}; as-loaded viscous 2.5. Discriminate viscous(μ∝v0) vs Coulomb(μ flat); realistic {REALISTIC_MU}", flush=True)
    grid = {}
    for label, vis, coul in MODELS:
        cells = {v0: _early_decel(rl, adr, v0, vis, coul) for v0 in V0S}
        mu_by_v0 = {v0: cells[v0]["mu_eff"] for v0 in V0S}
        kind, ratio = _classify(mu_by_v0)
        grid[label] = {"viscous": vis, "coulomb": coul, "kind": kind, "mu_ratio_hi_lo": ratio,
                       "mu_by_v0": mu_by_v0, "cells": cells}
        print(f"  {label:22s} (visc {vis}, coul {coul}): μ_eff by v0 " +
              " ".join(f"{v0}:{mu_by_v0[v0]}" for v0 in V0S) + f"  ⇒ {kind} (hi/lo {ratio})", flush=True)

    # pick the Coulomb-DOMINANT model (COULOMB or MIXED — i.e. NOT viscous-dominant, so μ_eff is ~flat across speed)
    # whose mean μ_eff is nearest the middle of the realistic band. A flat curve that sits ABOVE the band is worse than a
    # slightly-mixed curve INSIDE it.
    target = 0.5 * (REALISTIC_MU[0] + REALISTIC_MU[1])

    def _mean_mu(k):
        return float(np.mean(list(grid[k]["mu_by_v0"].values())))
    candidates = {k: v for k, v in grid.items() if v["kind"] in ("COULOMB", "MIXED") and REALISTIC_MU[0] <= _mean_mu(k) <= REALISTIC_MU[1]}
    if not candidates:                                          # none in-band ⇒ fall back to the flattest Coulomb
        candidates = {k: v for k, v in grid.items() if v["kind"] == "COULOMB"}
    chosen = min(candidates, key=lambda k: abs(_mean_mu(k) - target)) if candidates else None
    asloaded = grid["viscous_asloaded_2.5"]
    manifest = {"contract": "COIN_COAST_CALIBRATION_MULTIVELOCITY", "date": "2026-07-25", "gravity": G,
                "method": "early average deceleration over an unobstructed CLEAN_DIST window; μ_eff = decel/g",
                "clean_dist_m": CLEAN_DIST, "v0s": list(V0S), "realistic_mu_band": list(REALISTIC_MU),
                "as_loaded_kind": asloaded["kind"], "as_loaded_mu_by_v0": asloaded["mu_by_v0"],
                "chosen_coulomb_model": chosen,
                "chosen_params": ({"dof_damping": grid[chosen]["viscous"], "dof_frictionloss": grid[chosen]["coulomb"],
                                   "mu_by_v0": grid[chosen]["mu_by_v0"]} if chosen else None),
                "grid": grid}
    json.dump(manifest, open(f"{OUT}/coin_coast_calibration.json", "w"), indent=1, default=float)
    _plot(grid, f"{OUT}/coin_coast_calibration.png")
    print(f"\n  as-loaded (viscous 2.5): {asloaded['kind']}, μ_eff by v0 {asloaded['mu_by_v0']} — SPEED-DEPENDENT + too high")
    print(f"  chosen realistic COULOMB model: {chosen}" +
          (f" → dof_damping {grid[chosen]['viscous']} + dof_frictionloss {grid[chosen]['coulomb']}, μ_eff≈{grid[chosen]['mu_by_v0']}" if chosen else ""))
    print(f"  artifact: {OUT}/coin_coast_calibration.json\nCOIN_COAST_CALIBRATION_DONE")
    return grid


if __name__ == "__main__":
    main()
