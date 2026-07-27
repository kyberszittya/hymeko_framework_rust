"""AIBO Lyapunov verification: does the state-dependent high-level loop
(approach-align-stop pursuit) satisfy the Lyapunov conditions?

This is the refined-hypothesis test: the pH/Lyapunov high-level loop should pay on a
STATE-DEPENDENT embodiment (AIBO balance/alignment), unlike PnP's uniform bias. A
negative control (constant forward, no align/stop) must FAIL the Lyapunov certificate
-- a certificate everything passes is worthless (metric integrity).

Usage::  PYTHONPATH=. python -m scenarios.aibo.run_aibo_lyapunov
SIMULATION. NOT RL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hymeko_control.cip.runtime import CIP0Runtime

from . import load_model
from .adapter import AIBOCIPAdapter
from .locomotion_gait import SteeredTrotGait, body_yaw, wrap
from .lyapunov import AIBOLyapunov, evaluate_lyapunov

_OUT = Path("reports/2026-07-27-aibo-lyapunov")
_OFFSETS = (10.0, -10.0, 20.0, -20.0)


def _pursuit_V_series(bearing_deg: float, V) -> list[float]:
    model = load_model()
    adapter = AIBOCIPAdapter(model=model, seed=0, goal_bearing_deg=bearing_deg)
    records = CIP0Runtime(model=model, adapter=adapter).run(max_ticks=12)
    return [V(s) for r in records for s in r.trace.signals]


def _negative_V_series(bearing_deg: float, V, steps: int = 2500) -> list[float]:
    """Constant forward, no align/stop -> does not converge (Lyapunov must reject)."""
    from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
    env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8,
                           reach_radius=0.12, max_steps=steps + 5)
    env.reset(seed=0)
    th = np.radians(bearing_deg)
    tx, ty = float(env.data.xpos[env.torso, 0]), float(env.data.xpos[env.torso, 1])
    env.goal = np.array([tx + 0.8 * np.cos(th), ty + 0.8 * np.sin(th)], np.float32)
    env._prev_dist = env.dist_to_goal()
    gait = SteeredTrotGait()
    series = []
    for _ in range(steps):
        env.step(gait.action(env, yaw_cmd=0.0, drive=1.0))   # blind forward, no steer/stop
        v = float(np.hypot(env.data.qvel[0], env.data.qvel[1]))
        sig = {"dist_to_goal": float(env.dist_to_goal()),
               "heading_error": _herr(env), "speed": v}
        series.append(V(sig))
    return series


def _herr(env) -> float:
    from .locomotion_gait import goal_bearing
    return float(wrap(goal_bearing(env) - body_yaw(env)))


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    V = AIBOLyapunov()

    pursuit = {}
    all_pass = True
    for b in _OFFSETS:
        res = evaluate_lyapunov(_pursuit_V_series(b, V))
        pursuit[f"offset_{int(b):+d}deg"] = res
        all_pass = all_pass and res["passes"]

    neg = evaluate_lyapunov(_negative_V_series(20.0, V))

    verdict = ("AIBO_STATE_DEPENDENT_LOOP_SATISFIES_LYAPUNOV"
               if all_pass and not neg["passes"]
               else "LYAPUNOV_INCONCLUSIVE")
    result = {
        "verdict": verdict,
        "lyapunov_V": "0.5*(w_d*max(0,d-reach)^2 + w_theta*herr^2 + w_v*speed^2)",
        "pursuit_state_dependent": pursuit,
        "negative_control_constant_forward": neg,
        "discriminates": all_pass and not neg["passes"],
        "note": "SIMULATION. Refined hypothesis: pH/Lyapunov high-level loop pays on the "
                "STATE-DEPENDENT AIBO (unlike PnP's uniform bias). The Lyapunov certificate is "
                "reward-independent and generalizes stability_certificate (core-promotion candidate).",
    }
    (_OUT / "lyapunov_gates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "discriminates": result["discriminates"],
                      "pursuit_pass": {k: v["passes"] for k, v in pursuit.items()},
                      "pursuit_Vfinal": {k: v["Vfinal"] for k, v in pursuit.items()},
                      "negative_passes": neg["passes"], "negative_Vfinal": neg["Vfinal"]}, indent=2))


if __name__ == "__main__":
    main()
