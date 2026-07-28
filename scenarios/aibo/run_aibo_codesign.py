"""Co-design loop runner — search stance × gait for the best multi-position reach, report the frontier.

Profiles a grid of (stance width × stride sign × hip amplitude) points (forward propulsion, turn
authority, uprightness), ranks by a composite that needs BOTH walk and stable turn, then evaluates
multi-position reach on the top candidates. Reports the Pareto frontier + the best (model, gait) pair.

Usage::  PYTHONPATH=. python -m scenarios.aibo.run_aibo_codesign
"""

from __future__ import annotations

import json
from pathlib import Path

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv

from .agile_embodiment import widen_stance
from .codesign import CoDesignPoint, multi_position_reach, profile
from .motion_contract import JointVelocityGovernor

_OUT = Path("reports/2026-07-28-aibo-codesign")
_ROBOTICS = Path("data/robotics")
_GRID = [(0.5, 0), (0.5, 25), (0.5, -25), (0.6, 40), (0.6, -40)]
_STANCES = (0.062, 0.085, 0.10)
_SIGNS = (1.0, -1.0)
_AMPS = (0.7, 1.1)


def _variant_path(width: float) -> Path:
    """Write (once) the stance variant next to meta_kinematics.hymeko so the import resolves."""
    if width == 0.062:
        return _ROBOTICS / "quadruped.hymeko"
    dest = _ROBOTICS / f"_codesign_{int(width * 1000)}.hymeko"
    if not dest.exists():
        dest.write_text(widen_stance((_ROBOTICS / "quadruped.hymeko").read_text(), width))
    return dest


def _env(width: float, max_steps: int = 1600) -> QuadrupedGoalEnv:
    return QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12,
                            max_steps=max_steps, hymeko_path=str(_variant_path(width)))


def _score(prof) -> float:
    """Composite needing BOTH forward propulsion AND a stable turn (upright): min(walk, turn) gated."""
    if prof.turn_upright < 0.6:
        return -1.0                                   # tipping disqualifies
    return min(abs(prof.forward) / 0.4, abs(prof.turn_deg_per_1000) / 40.0)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    gov = JointVelocityGovernor(v_max=8.0)
    rows = []
    for w in _STANCES:
        env = _env(w)
        for sign in _SIGNS:
            for amp in _AMPS:
                pt = CoDesignPoint(stance_width=w, stride_sign=sign, hip_amp=amp)
                pr = profile(env, pt, gov)
                rows.append({"stance": w, "sign": sign, "amp": amp, "forward": pr.forward,
                             "turn": pr.turn_deg_per_1000, "turn_upright": pr.turn_upright,
                             "score": round(_score(pr), 3)})
    rows.sort(key=lambda r: r["score"], reverse=True)

    # multi-position reach on the top-3 candidates + the baseline
    baseline = CoDesignPoint(stance_width=0.062, stride_sign=1.0, hip_amp=0.7)
    base_reach = multi_position_reach(_env(0.062), baseline, gov, _GRID)
    evaluated = []
    for r in rows[:3]:
        pt = CoDesignPoint(stance_width=r["stance"], stride_sign=r["sign"], hip_amp=r["amp"])
        reach = multi_position_reach(_env(r["stance"]), pt, gov, _GRID)
        evaluated.append({**r, **reach})
    best = max(evaluated, key=lambda r: (r["upright_reach_rate"], -r["mean_min_dist"]))

    delta = round(best["upright_reach_rate"] - base_reach["upright_reach_rate"], 3)
    result = {
        "verdict": ("CODESIGN_IMPROVES_MULTI_POSITION_REACH" if delta > 0.15
                    else "CODESIGN_MODEST_ON_PARETO_FRONTIER" if delta > 0.0
                    else "CODESIGN_NO_IMPROVEMENT_TRADEOFF_FUNDAMENTAL"),
        "baseline_upright_reach": base_reach["upright_reach_rate"],
        "baseline_mean_min_dist": base_reach["mean_min_dist"],
        "best_point": {k: best[k] for k in ("stance", "sign", "amp", "forward", "turn",
                                            "turn_upright")},
        "best_upright_reach": best["upright_reach_rate"],
        "best_mean_min_dist": best["mean_min_dist"],
        "reach_delta": delta,
        "profile_frontier": rows,
        "evaluated_top3": evaluated,
        "note": "SIMULATION. Scripted co-design: stance width x gait (sign, amp), turn-then-walk "
                "pursuit. Upright reach on a held-out (dist x bearing) grid. Baseline = canonical trot.",
    }
    (_OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("verdict", "baseline_upright_reach", "best_point",
                                             "best_upright_reach", "reach_delta")}, indent=2))
    # keep only the best stance variant; clean the rest of the search
    for f in _ROBOTICS.glob("_codesign_*.hymeko"):
        if best["stance"] != 0.062 and f.name == f"_codesign_{int(best['stance'] * 1000)}.hymeko":
            f.rename(_ROBOTICS / f"quadruped_codesign_{int(best['stance'] * 1000)}.hymeko")
        else:
            f.unlink()


if __name__ == "__main__":
    main()
