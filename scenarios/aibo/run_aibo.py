"""CIP-AIBO-01 gate runner: interface-response audit + authority measurement +
certified approach->stop->hold, evaluated HONESTLY. SIMULATION ONLY (no hardware).

Usage::  PYTHONPATH=. python -m scenarios.aibo.run_aibo [--gif]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_control.cip.protocol import CIP0Adapter
from hymeko_control.cip.runtime import CIP0Runtime

from . import load_model
from .adapter import AIBOCIPAdapter
from .locomotion_gait import SteeredTrotGait, body_yaw

_OUT = Path("reports/2026-07-27-cip-aibo")
_DT = 0.005  # frame_skip(5) * timestep(0.001) s per env.step


def _interface_audit(seed: int = 0) -> dict[str, Any]:
    """AIBO-1/2: commanded forward/yaw -> measured body response (SIMULATOR)."""
    from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
    gait = SteeredTrotGait()

    def run(yaw, drive, steps):
        env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=5.0, max_steps=steps + 5)
        env.reset(seed=seed)
        x0 = float(env.data.xpos[env.torso, 0])
        y0 = body_yaw(env)
        first_move = None
        up = 1.0
        for k in range(steps):
            env.step(gait.action(env, yaw_cmd=yaw, drive=drive))
            up = min(up, env._torso_uprightness())
            if first_move is None and abs(float(env.data.xpos[env.torso, 0]) - x0) > 0.01:
                first_move = k
        dx = float(env.data.xpos[env.torso, 0]) - x0
        dyaw = body_yaw(env) - y0
        return {"dx_m": dx, "fwd_speed_mps": dx / (steps * _DT),
                "dyaw_deg": float(np.degrees(dyaw)), "yaw_rate_dps": float(np.degrees(dyaw)) / (steps * _DT),
                "latency_steps": first_move, "min_upright": up}

    return {
        "forward": run(0.0, 1.0, 800),
        "turn_stable": run(-0.5, 0.7, 800),
        "turn_other": run(+0.5, 0.7, 400),   # documents the one-directional stability limit
    }


def _summary(records: list) -> dict[str, Any]:
    visited = [r.mode for r in records]
    last = records[-1].trace.provenance if records else {}
    fell = any(r.trace.provenance.get("fell") for r in records)
    return {
        "visited_modes": visited,
        "final_mode": records[-1].next_mode if records else None,
        "final_certificate_passed": bool(records[-1].certificate.passed) if records else False,
        "reached": bool(last.get("reached")), "halted": bool(last.get("halted")),
        "held": bool(last.get("held")), "fell": fell,
        "final_dist": float(last.get("dist_to_goal", 9.9)),
        "n_ticks": len(records),
        "n_steps": sum(int(r.trace.provenance.get("steps", 0)) for r in records),
    }


def _gates(adapter, model, audit: dict, summ: dict) -> dict:
    fwd = audit["forward"]
    turn = audit["turn_stable"]
    aibo0 = isinstance(adapter, CIP0Adapter) and model.name.startswith("cip_aibo_01")
    aibo1 = fwd["fwd_speed_mps"] > 0.0 and fwd["min_upright"] > 0.5 and fwd["latency_steps"] is not None
    aibo2 = fwd["fwd_speed_mps"] > 0.0 and abs(turn["yaw_rate_dps"]) > 1.0  # nonzero yaw authority
    # on-axis approach+stop+hold is certified (informational), but AIBO-3 demands a
    # robust approach-ALIGN-stop; one-directional yaw stability => not genuinely met.
    aibo3 = False
    aibo4 = False
    gates = {"AIBO-0": aibo0, "AIBO-1": aibo1, "AIBO-2": aibo2, "AIBO-3": aibo3, "AIBO-4": aibo4}
    highest = "none"
    for name in ("AIBO-0", "AIBO-1", "AIBO-2", "AIBO-3", "AIBO-4"):
        if gates[name]:
            highest = name
        else:
            break
    return {
        "gates": gates, "highest_genuine_passed": highest,
        "physical_vs_simulated": "SIMULATED (no physical AIBO hardware this session)",
        "blocked": {
            "AIBO-3": "robust approach-ALIGN-stop: scripted yaw is stable in one direction only; "
                      "arbitrary-bearing pursuit not reliable on the weak scripted gait",
            "AIBO-4": "needs a robust steerable gait (tuned or RL walk policy) AND/OR physical AIBO + SDK",
        },
        "informational": {
            "onaxis_approach_stop_hold_certified": summ["final_certificate_passed"] and summ["held"],
            "yaw_authority_measured_dps": turn["yaw_rate_dps"],
            "note": "Aibo CAN turn (yaw authority demonstrated); on-axis approach+stop+hold is certified. "
                    "These are SIMULATED, not hardware.",
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
    ax[0].plot(xs, [s["dist_to_goal"] for s in steps], label="dist_to_goal (m)")
    ax[0].axhline(0.35, ls="--", c="gray", lw=0.8, label="stop radius")
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel("m")
    ax[0].set_title("CIP-AIBO-01 approach->stop->hold (SIMULATED)")
    ax[1].plot(xs, [s["speed"] for s in steps], label="planar speed")
    ax[1].plot(xs, [s["uprightness"] for s in steps], label="uprightness")
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel("env step")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _gif(seed: int, path: Path) -> str | None:
    try:
        import imageio.v2 as imageio
        import mujoco
        adapter = AIBOCIPAdapter(model=load_model(), seed=seed)
        env = adapter._env
        env.model.vis.global_.offwidth = 960
        env.model.vis.global_.offheight = 720
        renderer = mujoco.Renderer(env.model, height=720, width=960)
        # replay the certified trajectory deterministically, capturing frames:
        frames = []
        k = 0
        for mode in ("STAND", "ALIGN", "WALK", "DECELERATE", "STOP", "HOLD"):
            adapter._mode = mode
            for step in range(1, adapter.max_mode_steps + 1):
                s = adapter._signals()
                env.step(adapter._control(mode, s))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)

    audit = _interface_audit(args.seed)
    model = load_model()
    adapter = AIBOCIPAdapter(model=model, seed=args.seed)
    runtime = CIP0Runtime(model=model, adapter=adapter)
    records = runtime.run(max_ticks=12)
    summ = _summary(records)
    result = _gates(adapter, model, audit, summ)
    result["seed"] = args.seed
    result["interface_audit"] = audit
    result["summary"] = summ
    result["plot"] = "aibo_trajectory.png" if _plot(records, _OUT / "aibo_trajectory.png") else None
    if args.gif:
        result["gif"] = _gif(args.seed, _OUT / "aibo_trajectory.gif")
    (_OUT / "aibo_gates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": result["gates"],
                      "highest_genuine_passed": result["highest_genuine_passed"],
                      "physical_vs_simulated": result["physical_vs_simulated"],
                      "fwd_speed_mps": round(audit["forward"]["fwd_speed_mps"], 4),
                      "yaw_rate_dps": round(audit["turn_stable"]["yaw_rate_dps"], 3),
                      "onaxis_certified": result["informational"]["onaxis_approach_stop_hold_certified"],
                      "visited": summ["visited_modes"]}, indent=2))


if __name__ == "__main__":
    main()
