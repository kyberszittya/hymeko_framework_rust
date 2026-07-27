"""AIBO-SIM-01/02/03 harness (SIMULATION ONLY) + gates A0-A3.

Drives the CIP-AIBO-01 adapter (bidirectional steered trot) through the CIP-0
runtime for three simple scenarios and logs, per run: initial pose, realised
forward velocity + yaw rate, target-distance error, orientation error, stopping
time, final body speed, stability, and the external certificate result.

Usage::  PYTHONPATH=. python -m scenarios.aibo.run_aibo_sim [--gif]

Simulation-certified only; NOT hardware. No RL.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from hymeko_control.cip.runtime import CIP0Runtime

from . import load_model
from .adapter import AIBOCIPAdapter

_OUT = Path("reports/2026-07-27-aibo-simple-scenarios")
_DT = 0.005  # frame_skip(5) * timestep(0.001)


def run_one(bearing_deg: float, seed: int = 0) -> dict[str, Any]:
    model = load_model()
    adapter = AIBOCIPAdapter(model=model, seed=seed, goal_bearing_deg=bearing_deg)
    env = adapter._env
    x0, y0 = float(env.data.xpos[env.torso, 0]), float(env.data.xpos[env.torso, 1])
    runtime = CIP0Runtime(model=model, adapter=adapter)
    records = runtime.run(max_ticks=12)

    steps = [s for r in records for s in r.trace.signals]
    last = records[-1].trace.provenance if records else {}
    # realised yaw rate over the ALIGN option; forward speed over WALK
    align = next((r for r in records if r.mode == "ALIGN"), None)
    walk = next((r for r in records if r.mode == "WALK"), None)
    yaw_rate = None
    if align and len(align.trace.signals) > 1:
        sg = align.trace.signals
        yaw_rate = math.degrees(sg[-1]["body_yaw"] - sg[0]["body_yaw"]) / (len(sg) * _DT)
    fwd_speed = (sum(s["speed"] for s in walk.trace.signals) / len(walk.trace.signals)
                 if walk and walk.trace.signals else None)
    stop_rec = next((r for r in records if r.mode == "STOP"), None)
    return {
        "bearing_deg": bearing_deg,
        "initial_pose": {"x": x0, "y": y0, "yaw_deg": 0.0},
        "realised_forward_speed_mps": fwd_speed,
        "realised_yaw_rate_dps": yaw_rate,
        "target_distance_error_m": float(last.get("dist_to_goal", 9.9)),
        "orientation_error_deg": math.degrees(float(last.get("orientation_error", math.pi))),
        "stopping_steps": int(stop_rec.trace.provenance.get("steps", 0)) if stop_rec else None,
        "final_body_speed_mps": float(last.get("speed", 9.9)),
        "min_uprightness": min((s["uprightness"] for s in steps), default=0.0),
        "no_fall": not any(r.trace.provenance.get("fell") for r in records),
        "certificate_passed": bool(records[-1].certificate.passed) if records else False,
        "reached": bool(last.get("reached")),
        "oriented": bool(last.get("oriented")),
        "halted": bool(last.get("halted")),
        "held": bool(last.get("held")),
        "visited_modes": [r.mode for r in records],
        "_steps": steps,
    }


def _gates(sim01, sim02, sim03) -> dict[str, Any]:
    a0 = sim01["certificate_passed"] and sim01["reached"] and sim01["held"] and sim01["no_fall"]
    left = next(r for r in sim02 if r["bearing_deg"] > 0)
    right = next(r for r in sim02 if r["bearing_deg"] < 0)
    # A1: both turn the correct way (left run yaws +, right run yaws -) while upright
    a1 = (left["realised_yaw_rate_dps"] is not None and left["realised_yaw_rate_dps"] > 1.0
          and right["realised_yaw_rate_dps"] is not None and right["realised_yaw_rate_dps"] < -1.0
          and left["min_uprightness"] > 0.5 and right["min_uprightness"] > 0.5)
    a2 = all(r["certificate_passed"] and r["no_fall"] for r in sim02)
    a3 = all(r["certificate_passed"] and r["no_fall"] for r in sim03)
    gates = {"A0": bool(a0), "A1": bool(a1), "A2": bool(a2), "A3": bool(a3)}
    highest = "none"
    for name in ("A0", "A1", "A2", "A3"):
        if gates[name]:
            highest = name
        else:
            break
    return {"gates": gates, "highest_passed": highest}


def _plot(runs: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for r in runs:
        steps = r["_steps"]
        if not steps:
            continue
        # reconstruct planar path is not stored; plot dist/heading proxy instead
        ax.plot([math.degrees(s["heading_error"]) for s in steps],
                [s["dist_to_goal"] for s in steps], lw=1.0,
                label=f"bearing {r['bearing_deg']:+.0f}°")
    ax.axhline(0.42, ls="--", c="gray", lw=0.8, label="stop radius")
    ax.set_xlabel("heading error (deg)")
    ax.set_ylabel("distance to waypoint (m)")
    ax.set_title("AIBO-SIM approach-align-stop (bidirectional yaw, SIMULATED)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _gif(bearing_deg: float, path: Path) -> str | None:
    try:
        import imageio.v2 as imageio
        import mujoco
        import numpy as np
        adapter = AIBOCIPAdapter(model=load_model(), seed=0, goal_bearing_deg=bearing_deg)
        env = adapter._env
        env.model.vis.global_.offwidth = 960
        env.model.vis.global_.offheight = 720
        renderer = mujoco.Renderer(env.model, height=720, width=960)
        frames = []
        k = 0
        from .adapter import CHAIN_A
        for mode in CHAIN_A:
            adapter._mode = mode
            for step in range(1, adapter.max_mode_steps + 1):
                s = adapter._signals()
                env.step(np.asarray(adapter._control(mode, s), dtype=np.float64))
                if k % 10 == 0:
                    renderer.update_scene(env.data)
                    frames.append(renderer.render())
                k += 1
                if adapter._exit(mode, adapter._signals(), step):
                    break
        imageio.mimsave(path, frames, fps=15)
        return str(path)
    except Exception as exc:
        return f"gif_unavailable: {type(exc).__name__}: {exc}"


def _clean(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "_steps"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)

    sim01 = run_one(0.0)
    sim02 = [run_one(+30.0), run_one(-30.0)]
    sim03 = [run_one(b) for b in (+10.0, -10.0, +20.0, -20.0)]
    result = _gates(sim01, sim02, sim03)
    result["physical_vs_simulated"] = "SIMULATED (22-DOF ERS-1000 sim; NOT hardware)"
    result["AIBO_SIM_01_forward_stop_hold"] = _clean(sim01)
    result["AIBO_SIM_02_turn_align_stop"] = [_clean(r) for r in sim02]
    result["AIBO_SIM_03_approach_align_stop_offsets"] = [_clean(r) for r in sim03]
    result["plot"] = "aibo_sim_paths.png" if _plot([sim01, *sim02, *sim03], _OUT / "aibo_sim_paths.png") else None
    if args.gif:
        result["gif"] = _gif(20.0, _OUT / "aibo_sim_approach.gif")

    (_OUT / "aibo_sim_gates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": result["gates"], "highest_passed": result["highest_passed"],
                      "sim02_yaw_dps": [round(r["realised_yaw_rate_dps"], 2) if r["realised_yaw_rate_dps"] else None for r in sim02],
                      "sim03_orient_err_deg": [round(r["orientation_error_deg"], 1) for r in sim03],
                      "sim03_reached": [r["reached"] for r in sim03],
                      "sim03_upright": [round(r["min_uprightness"], 2) for r in sim03]}, indent=2))


if __name__ == "__main__":
    main()
