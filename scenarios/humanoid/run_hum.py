"""CIP-HUM-01 gate runner: drive the CIP-0 lifecycle and evaluate HUM-0..4 HONESTLY.

Usage::  PYTHONPATH=. python -m scenarios.humanoid.run_hum

The HyMeKo humanoid is fixed-base, so balance is untestable: HUM-2 (support
shift) and HUM-4 (recover stance after perturbation) are reported BLOCKED, and
HUM-3's support component is vacuous. Genuine ceiling: HUM-1. No RL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hymeko_control.cip.protocol import CIP0Adapter
from hymeko_control.cip.runtime import CIP0Runtime

from . import load_model
from .adapter import CHAIN_H, HumanoidCIPAdapter

_OUT = Path("reports/2026-07-27-cip-humanoid")


def _summary(records: list) -> dict[str, Any]:
    visited = [r.mode for r in records]
    reached = any(r.trace.provenance.get("reached_target") for r in records)
    recover_ok = any(r.trace.provenance.get("recover_home_ok") for r in records)
    limits_all = all(r.trace.provenance.get("limits_ok") for r in records) if records else False
    no_div = not any(r.trace.provenance.get("diverged") for r in records)
    stand_recs = [r for r in records if r.mode == "STAND"]
    stand_stable = bool(stand_recs) and all(
        r.trace.provenance.get("torso_upright") and not r.trace.provenance.get("diverged")
        for r in stand_recs
    )
    steps = sum(int(r.trace.provenance.get("steps", 0)) for r in records)
    return {
        "visited_modes": visited, "reached_target": reached,
        "recover_home_ok": recover_ok, "limits_ok_all": limits_all,
        "no_divergence": no_div, "stand_stable": stand_stable,
        "cycle_reached_recover": "RECOVER" in visited,
        "n_ticks": len(records), "n_steps": steps,
    }


def _gates(adapter, model, summ: dict) -> dict:
    hum0 = isinstance(adapter, CIP0Adapter) and model.name.startswith("cip_hum_01")
    hum1 = summ["stand_stable"] and summ["no_divergence"]
    # BLOCKED on the fixed-base model (balance untestable):
    hum2 = False
    hum3 = False  # reach genuine, but "while support certificate holds" is vacuous
    hum4 = False
    genuine = {"HUM-0": hum0, "HUM-1": hum1, "HUM-2": hum2, "HUM-3": hum3, "HUM-4": hum4}
    highest = "none"
    for name in ("HUM-0", "HUM-1", "HUM-2", "HUM-3", "HUM-4"):
        if genuine[name]:
            highest = name
        else:
            break
    return {
        "gates": genuine,
        "highest_genuine_passed": highest,
        "blocked": {
            "HUM-2": "no floating base: support-shift instability is impossible to test",
            "HUM-3": "reach is genuine but 'support certificate holds' is vacuous (fixed base)",
            "HUM-4": "no balance/recovery controller for a floating-base humanoid",
        },
        "informational": {
            "kinematic_reach_completed": summ["reached_target"],
            "kinematic_recover_completed": summ["recover_home_ok"],
            "note": "reach cycle runs, but is NOT a genuine HUM-4 (balance vacuous)",
        },
    }


def _plot(records: list, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    steps = [s for r in records for s in r.trace.signals]
    if not steps:
        return False
    xs = range(len(steps))
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(xs, [s["ee_to_target"] for s in steps], label="ee_to_target (m)")
    ax[0].plot(xs, [s["ee_to_home"] for s in steps], label="ee_to_home (m)")
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel("m")
    ax[0].set_title("CIP-HUM-01 fixed-base stand-reach-recover (balance vacuous)")
    ax[1].plot(xs, [s["torso_z"] for s in steps], label="torso_z (m)")
    ax[1].plot(xs, [s["max_qvel"] for s in steps], label="max|qvel|")
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel("sim step")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _gif(path: Path) -> str | None:
    """Best-effort GIF of the fixed-base stand-reach-recover cycle (reuses adapter PD)."""
    try:
        import imageio.v2 as imageio
        import mujoco
        a = HumanoidCIPAdapter(model=load_model())
        sim = a._sim
        sim.model.vis.global_.offwidth = 960
        sim.model.vis.global_.offheight = 720
        renderer = mujoco.Renderer(sim.model, height=720, width=960)
        frames = []
        k = 0
        for mode in CHAIN_H:
            qt = a._q_target_for(mode)
            for step in range(1, a.max_mode_steps + 1):
                sim.pd_step(qt, a.kp, a.kd)
                if k % 8 == 0:
                    renderer.update_scene(sim.data)
                    frames.append(renderer.render())
                k += 1
                if a._exit_condition(mode, a._signals(), step):
                    break
        imageio.mimsave(path, frames, fps=15)
        return str(path)
    except Exception as exc:
        return f"gif_unavailable: {type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)
    model = load_model()
    adapter = HumanoidCIPAdapter(model=model)
    runtime = CIP0Runtime(model=model, adapter=adapter)
    records = runtime.run(max_ticks=12)
    summ = _summary(records)
    result = _gates(adapter, model, summ)
    result["summary"] = summ
    result["plot"] = "hum_trajectory.png" if _plot(records, _OUT / "hum_trajectory.png") else None
    if args.gif:
        result["gif"] = _gif(_OUT / "hum_trajectory.gif")
    (_OUT / "hum_gates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": result["gates"],
                      "highest_genuine_passed": result["highest_genuine_passed"],
                      "reach_completed": summ["reached_target"],
                      "recover_completed": summ["recover_home_ok"],
                      "visited": summ["visited_modes"]}, indent=2))


if __name__ == "__main__":
    main()
