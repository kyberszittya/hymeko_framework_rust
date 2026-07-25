"""Coin coast calibration — is the coin↔floor drag PHYSICALLY realistic? Inject a fixed initial coin speed into the FREE
scene (arm held, no contact), let the coin slide, and measure stopping time, stopping distance, and the Coulomb-equivalent
effective friction μ_eff = v0² / (2 g d). A hard coin on a smooth table should have μ_eff ≈ 0.1–0.2 (a 1.5 m/s coin coasts
~0.5–1 m). If it stops within ~10 cm, the model is far too sticky / over-damped.

This is the physical prerequisite the user flagged: the legacy "success" was likely a high-speed IMPACT on an over-damped
coin (27 rad/s arm → coin launched ~1.5 m/s → strong drag stops it within ~10 cm → lands in the zone), not clean
transport — which is why the scripted push worked in fast physics and collapsed at realistic speed. Calibrate the contact
physics BEFORE judging the controller.

The disk's in-plane resistance is a MIX of viscous ``dof_damping`` (≈2.5 as-loaded) and Coulomb ``dof_frictionloss`` (0 as
loaded). We read the coin position from the RAW ``qpos`` (``_planar_metrics`` only recomputes inside ``step_ablation``, so
raw ``mj_step`` leaves it stale). Sweep the viscous level and a Coulomb variant; report μ_eff for each.
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
V0S = (0.5, 1.0, 1.5)                         # initial coin speeds (1.5 ≈ the legacy launched-coin speed)
VISCOUS = (2.5, 1.0, 0.5, 0.1, 0.05, 0.02)   # dof_damping levels (2.5 = as-loaded)
COULOMB = (0.0, 0.02, 0.05, 0.1)             # dof_frictionloss (N) added on top of a LOW viscous base (realistic model)
STOP_SPEED, MAX_STEPS = 0.01, 4000
REALISTIC_MU = (0.05, 0.25)                  # a hard coin on a smooth table: μ_eff in roughly this band


def _coast(rl, adr, v0, viscous, coulomb):
    """Inject v0 along +x into the free coin, hold the arm, raw-step until it (asymptotically) stops. Read the coin
    position from RAW qpos. Returns stopping distance, stopping time, and Coulomb-equivalent μ_eff."""
    m, d = rl.inner.model, rl.inner.data
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    m.dof_damping[adr:adr + 2] = viscous
    m.dof_frictionloss[adr:adr + 2] = coulomb
    d.ctrl[:4] = d.qpos[:4]                                        # hold the arm (position actuators) — no coin contact
    d.qvel[adr:adr + 2] = [v0, 0.0]
    mujoco.mj_forward(m, d)
    p0 = d.qpos[adr:adr + 2].copy()
    steps = MAX_STEPS
    for k in range(MAX_STEPS):
        d.ctrl[:4] = d.qpos[:4]
        mujoco.mj_step(m, d)
        if float(np.linalg.norm(d.qvel[adr:adr + 2])) < STOP_SPEED:
            steps = k + 1
            break
    dist = float(np.linalg.norm(d.qpos[adr:adr + 2] - p0))
    mu_eff = v0 * v0 / (2 * G * dist) if dist > 1e-6 else float("inf")
    truncated = steps >= MAX_STEPS                                # still moving at the window end (μ_eff not meaningful)
    return {"v0": v0, "stop_dist_m": round(dist, 4), "stop_time_s": round(steps * m.opt.timestep, 3),
            "truncated": bool(truncated), "mu_eff": round(mu_eff, 3) if np.isfinite(mu_eff) else 999.0}


def _base_adr():
    pi0, base, forbidden = _setup()
    rl, _g = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=14300, tries=2)
    return rl, int(rl.inner._disk_x_adr)


def _plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for v0 in V0S:
        vis = [r for r in rows if r["model"] == "viscous" and r["v0"] == v0]
        ax1.plot([r["param"] for r in vis], [r["stop_dist_m"] for r in vis], "o-", label=f"v0={v0}")
        ax2.plot([r["param"] for r in vis], [min(r["mu_eff"], 10) for r in vis], "o-", label=f"v0={v0}")
    ax1.axhspan(0.5, 1.0, alpha=0.12, color="g", label="realistic coast (0.5–1 m)")
    ax1.set_xlabel("viscous dof_damping")
    ax1.set_ylabel("stopping distance (m)")
    ax1.set_xscale("log")
    ax1.set_title("coast distance vs coin slide-damping")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.axhspan(REALISTIC_MU[0], REALISTIC_MU[1], alpha=0.15, color="g", label="realistic μ_eff 0.05–0.25")
    ax2.set_xlabel("viscous dof_damping")
    ax2.set_ylabel("effective μ (capped at 10)")
    ax2.set_xscale("log")
    ax2.set_title("Coulomb-equivalent μ_eff vs slide-damping")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle("Coin coast calibration — as-loaded damping 2.5 is far too sticky", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    rl, adr = _base_adr()
    print(f"coin slide DOF adr={adr}; as-loaded damping 2.5, frictionloss 0. Realistic target μ_eff∈{REALISTIC_MU}", flush=True)
    rows = []
    print("\n-- viscous damping sweep (frictionloss 0) --", flush=True)
    for c in VISCOUS:
        for v0 in V0S:
            r = _coast(rl, adr, v0, c, 0.0)
            rows.append({"model": "viscous", "param": c, **r})
            print(f"  damping {c:5}: v0 {v0} → dist {r['stop_dist_m']} m, t {r['stop_time_s']} s, μ_eff {r['mu_eff']}", flush=True)
    print("\n-- Coulomb frictionloss sweep (low viscous base 0.05) --", flush=True)
    for fl in COULOMB:
        for v0 in V0S:
            r = _coast(rl, adr, v0, 0.05, fl)
            rows.append({"model": "coulomb", "param": fl, **r})
            print(f"  frictionloss {fl:5}: v0 {v0} → dist {r['stop_dist_m']} m, t {r['stop_time_s']} s, μ_eff {r['mu_eff']}", flush=True)

    asloaded = next(r for r in rows if r["model"] == "viscous" and r["param"] == 2.5 and r["v0"] == 1.5)
    # nearest realistic damping — CLEAN (non-truncated) v0=1.5 measurements only; low-damping free-coast is contaminated by
    # workspace/arm contact (the coin does not have an unobstructed lane), so those μ_eff are not reliable.
    target = 0.5 * (REALISTIC_MU[0] + REALISTIC_MU[1])
    clean15 = [r for r in rows if r["model"] == "viscous" and r["v0"] == 1.5 and not r["truncated"]]
    best = min(clean15, key=lambda r: abs(r["mu_eff"] - target)) if clean15 else asloaded
    verdict = ("AS_LOADED_COIN_DRAG_UNREALISTICALLY_STICKY" if asloaded["mu_eff"] > REALISTIC_MU[1]
               else "AS_LOADED_COIN_DRAG_WITHIN_REALISTIC_BAND")
    manifest = {"contract": "COIN_COAST_CALIBRATION", "date": "2026-07-25", "gravity": G,
                "as_loaded": {"damping": 2.5, "frictionloss": 0.0, "coast_at_1.5ms": asloaded},
                "realistic_mu_band": list(REALISTIC_MU), "rows": rows,
                "low_damping_caveat": "free-coast at low damping is contaminated by workspace/arm contact (short or "
                                      "truncated coasts); an unobstructed coast lane is needed to pin the exact realistic damping",
                "nearest_realistic_clean_damping": {"damping": best["param"], "mu_eff_at_1.5": best["mu_eff"],
                                                    "stop_dist_at_1.5": best["stop_dist_m"]},
                "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/coin_coast_calibration.json", "w"), indent=1, default=float)
    _plot(rows, f"{OUT}/coin_coast_calibration.png")
    print(f"\n  as-loaded (damping 2.5): a 1.5 m/s coin stops in {asloaded['stop_dist_m']} m ⇒ μ_eff {asloaded['mu_eff']} "
          f"(realistic {REALISTIC_MU}) — {int(asloaded['mu_eff'] / REALISTIC_MU[1])}× too sticky")
    print(f"  nearest-realistic CLEAN viscous damping: {best['param']} → μ_eff {best['mu_eff']}, coast {best['stop_dist_m']} m at 1.5 m/s")
    print(f"  → {verdict}\n  artifact: {OUT}/coin_coast_calibration.json\nCOIN_COAST_CALIBRATION_DONE")
    return verdict


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
