"""Bounded KINEMATIC feasibility probe for a clearance-safe retracted seed + path (FANUC pick-place v2).

Question: does a config-space transition exist from the over-extended ``arm_home`` into a usable RETRACTED branch
from which the object hover pose is reachable by short Cartesian hops — the whole path staying table-clear? If yes,
a ``HOME_RETRACT_OR_PRESHAPE`` phase is justified; if no, the scene/home may need a versioned fallback.

This is a probe, NOT a controller. It is pure kinematics — ``fk_tool`` / ``solve`` / ``solve_collision_free`` on a
scratch ``MjData`` — so it isolates *reachability + geometric clearance* from the servo sag / gains (deferred). No
physics stepping, no reward, no training, no scene/home change.

    python -m hymeko_rl.eval.pick_retract_probe --seed0 50000 --episodes 3 \
        --out reports/figures/pick_place_clean_expert/retract_probe
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import hypot, inf, isfinite
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.viz.render_pick_place import fanuc_pick_env

_HOVER_REACHED_HORIZ = 0.02    # tool centred over the object (matches the controller's align tolerance)
_DISTMAX = 1.0


@dataclass(frozen=True)
class SegmentResult:
    """A path segment's clearance summary: minimum signed finger↔table clearance and the first forbidden step."""

    min_clearance: float
    first_forbidden: "int | None"
    steps: int
    ok: bool


@dataclass(frozen=True)
class CandidateResult:
    name: str
    tool_xyz: "tuple[float, float, float]"
    radius: float
    horiz_to_obj: float
    seed_clearance: float
    seed_collision_free: bool
    home_to_seed: SegmentResult
    seed_to_hover_reached: bool
    seed_to_hover: SegmentResult
    hover_final_horiz: float
    feasible: bool


class _StaticChecker:
    """Static clearance/validity of an arm config on a scratch ``MjData`` (object kept at its reset pose).

    # Invariants reads only; the live env data is never mutated (a private scratch ``MjData`` is used).
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self.n = env.n_actions - 1
        self.scratch = mujoco.MjData(env.model)
        self.table_geom = int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "table"))
        self.finger_geoms = [g for b in (int(env._b_fl), int(env._b_fr))
                             for g in range(int(env.model.body_geomadr[b]),
                                            int(env.model.body_geomadr[b]) + int(env.model.body_geomnum[b]))]

    def status(self, q: np.ndarray) -> "tuple[float, bool]":
        """(signed finger↔table clearance, self/floor-collision-free) at arm config ``q``."""
        d = self.scratch
        d.qpos[:] = self.env.data.qpos
        d.qpos[: self.n] = np.asarray(q[: self.n], dtype=np.float64)
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, d)
        clr = inf
        if self.table_geom >= 0:
            for g in self.finger_geoms:
                clr = min(clr, float(mujoco.mj_geomDistance(self.env.model, d, g, self.table_geom, _DISTMAX, None)))
        return clr, bool(self.env._ik_pose_valid(d))

    def forbidden(self, q: np.ndarray) -> "tuple[bool, float]":
        # "at least table-clear": a step is forbidden only on finger↔table PENETRATION (clr < 0). The benign
        # link_2↔link_4 self-overlap that exists in the home posture (measured −3 mm; v1 runs from it) is NOT a
        # table contact and must not fail the path — self-collision-free is reported separately, not gated.
        clr, _valid = self.status(q)
        return (clr < 0.0), clr


def _interp_segment(chk: _StaticChecker, q0: np.ndarray, q1: np.ndarray, steps: int = 40) -> SegmentResult:
    """Joint-linear interpolate ``q0 → q1`` and report the clearance minimum + first forbidden step."""
    minc, first = inf, None
    for i in range(steps + 1):
        a = i / steps
        forbidden, clr = chk.forbidden((1.0 - a) * q0 + a * q1)
        minc = min(minc, clr)
        if forbidden and first is None:
            first = i
    return SegmentResult(round(minc, 5) if isfinite(minc) else minc, first, steps + 1, first is None)


def _hop_to_hover(env: Any, chk: _StaticChecker, q_seed: np.ndarray, hover: np.ndarray, *,
                  hop: float, rate: float, max_hops: int = 200) -> "tuple[bool, SegmentResult, float]":
    """Simulate the seed-frame short-hop command chain from ``q_seed`` toward ``hover`` (kinematic only, the same
    rule as ``_v2_ik_step``); report whether it reaches the hover and the clearance along the way."""
    q = np.asarray(q_seed, dtype=np.float64)
    minc, first, horiz = inf, None, inf
    reached, used = False, 0
    for i in range(max_hops):
        used = i + 1
        tool = env._ik.fk_tool(q)
        horiz = float(hypot(tool[0] - hover[0], tool[1] - hover[1]))
        if horiz <= _HOVER_REACHED_HORIZ:
            reached = True
            break
        tgt = tool + np.clip(hover - tool, -hop, hop)
        q_sol = np.asarray(env._ik.solve(q, tgt, down=True, iters=120), dtype=np.float64)
        q = q + np.clip(q_sol - q, -rate, rate)
        forbidden, clr = chk.forbidden(q)
        minc = min(minc, clr)
        if forbidden and first is None:
            first = i
    seg = SegmentResult(round(minc, 5) if isfinite(minc) else minc, first, used, first is None)
    return reached, seg, round(horiz, 4)


def _candidates(env: Any, obj: np.ndarray, z_hover: float) -> "list[tuple[str, np.ndarray]]":
    """Candidate retracted seeds: arm_home (baseline), multi-start collision-free configs at the object hover and
    at a retracted mid-radius pose, and a manual elbow-up guess."""
    home = np.asarray(env._arm_home, dtype=np.float64)
    hover = np.array([obj[0], obj[1], z_hover])
    mid = np.array([0.55 * obj[0], 0.55 * obj[1], z_hover + 0.06])   # closer to base + higher = generous envelope
    cf_hover = np.asarray(env._ik.solve_collision_free(home, hover, env._ik_pose_valid, down=True, n_starts=40),
                          dtype=np.float64)
    cf_mid = np.asarray(env._ik.solve_collision_free(home, mid, env._ik_pose_valid, down=True, n_starts=40),
                        dtype=np.float64)
    manual = home + np.array([0.0, 0.4, 0.5, 0.0, 0.3, 0.0])         # more shoulder/elbow flexion → retract inward
    return [("arm_home", home), ("cf_hover", cf_hover), ("cf_mid_retract", cf_mid), ("manual_elbow_up", manual)]


def _evaluate_candidate(env: Any, chk: _StaticChecker, obj: np.ndarray, z_hover: float,
                        name: str, q_seed: np.ndarray) -> CandidateResult:
    home = np.asarray(env._arm_home, dtype=np.float64)
    hover = np.array([obj[0], obj[1], z_hover])
    tool = env._ik.fk_tool(q_seed)
    clr, valid = chk.status(q_seed)
    home_seg = _interp_segment(chk, home, q_seed)
    reached, hover_seg, final_horiz = _hop_to_hover(env, chk, q_seed, hover,
                                                    hop=env._V2_HOP, rate=env._V2_CENTER_RATE)
    # feasibility = table-clear seed + a table-clear (no-penetration) path home→seed→hover that reaches the hover.
    # self-collision-free (``valid``) is reported for info, not gated (per "collision-free OR at least table-clear").
    feasible = bool(clr > 0.0 and home_seg.ok and reached and hover_seg.ok)
    return CandidateResult(
        name=name, tool_xyz=(round(float(tool[0]), 3), round(float(tool[1]), 3), round(float(tool[2]), 3)),
        radius=round(float(hypot(tool[0], tool[1])), 3),
        horiz_to_obj=round(float(hypot(tool[0] - obj[0], tool[1] - obj[1])), 3),
        seed_clearance=round(float(clr), 5) if isfinite(clr) else float(clr), seed_collision_free=bool(valid),
        home_to_seed=home_seg, seed_to_hover_reached=bool(reached), seed_to_hover=hover_seg,
        hover_final_horiz=final_horiz, feasible=feasible)


def probe_episode(seed: int) -> "dict[str, Any]":
    """Run the feasibility probe for one object spawn (seed). # Postconditions returns per-candidate results +
    the best feasible seed (if any) + the object geometry."""
    env = fanuc_pick_env(expert_version=2)
    env.reset(seed=seed)
    obj = np.asarray(env._obj_xyz(), dtype=np.float64)
    grasp_z = env._surf + env.box_half + env._GRASP_TOOL_DZ
    z_hover = grasp_z + env._V2_HOVER_DZ
    chk = _StaticChecker(env)
    cands = [_evaluate_candidate(env, chk, obj, z_hover, name, q) for name, q in _candidates(env, obj, z_hover)]
    feasible = [c for c in cands if c.feasible and c.name != "arm_home"]
    best = max(feasible, key=lambda c: min(c.home_to_seed.min_clearance, c.seed_to_hover.min_clearance),
               default=None)
    close = getattr(env, "close", None)
    if callable(close):
        close()
    return {
        "seed": seed, "obj_radius": round(float(hypot(obj[0], obj[1])), 3), "z_hover": round(z_hover, 3),
        "candidates": [asdict(c) for c in cands],
        "best_seed": best.name if best is not None else None,
        "home_retract_feasible": best is not None,
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", type=str, default="reports/figures/pick_place_clean_expert/retract_probe")
    a = ap.parse_args(argv)

    print(f"[retract_probe] kinematic feasibility · seeds {a.seed0}..{a.seed0 + a.episodes - 1} "
          f"(no physics, no gains, no scene change)", flush=True)
    episodes = [probe_episode(a.seed0 + i) for i in range(a.episodes)]
    for ep in episodes:
        print(f"\nseed {ep['seed']}  obj_r={ep['obj_radius']}  z_hover={ep['z_hover']}  "
              f"HOME_RETRACT_feasible={ep['home_retract_feasible']}  best={ep['best_seed']}")
        for c in ep["candidates"]:
            h, s = c["home_to_seed"], c["seed_to_hover"]
            print(f"  {c['name']:16s} tool={c['tool_xyz']} r={c['radius']:.3f} seed_clr={c['seed_clearance']} "
                  f"cf={c['seed_collision_free']} | home→seed minclr={h['min_clearance']} ok={h['ok']} | "
                  f"seed→hover reached={c['seed_to_hover_reached']} minclr={s['min_clearance']} "
                  f"first_forbidden={s['first_forbidden']} final_horiz={c['hover_final_horiz']} => "
                  f"{'FEASIBLE' if c['feasible'] else 'no'}")
    n_feasible = sum(ep["home_retract_feasible"] for ep in episodes)
    verdict = {"episodes": len(episodes), "feasible_episodes": n_feasible,
               "home_retract_or_preshape_justified": n_feasible == len(episodes)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"episodes": episodes, "verdict": verdict}, indent=2),
                                        encoding="utf-8")
    print(f"\n[verdict] feasible on {n_feasible}/{len(episodes)} seeds → "
          f"HOME_RETRACT_OR_PRESHAPE justified = {verdict['home_retract_or_preshape_justified']}")
    print(f"[wrote] {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
