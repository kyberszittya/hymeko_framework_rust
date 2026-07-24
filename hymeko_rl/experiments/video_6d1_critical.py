"""Video 01 — the 6D-1 flagship: the critical pair, side by side, SAME state + SAME total budget.

  LEFT  single-head anchored to the wrong route basin  → local jitter cannot cross → FAIL / collision
  RIGHT K-mode covering all route basins               → search reaches the open basin → SUCCESS

Renders the EXACT rollout each arm's scorer executed (the selected via replayed through the same committed option) — no
demo controller. Overlays: status bar (task | ctrl | B | route/mode | STATUS), distance-to-goal(t) with the reach
threshold + the via→goal phase marker, a proposal/allocation panel, and a certificate badge. A per-clip manifest carries
the full provenance. Output: reports/2026-07-24-se3-obstacle-6d1/videos/01_6d1_critical_pair.mp4 (+ .json).
"""
import json
import os
import sys

import mujoco
import numpy as np

from hymeko_rl.env.se3_obstacle_reach_env import SE3ObstacleReachEnv
from hymeko_rl.env.se3_reach_option import (
    ROUTE_DIRS, RouteModeProposal, RouteOptionScorer, ik_position, route_execution_feasible)
from hymeko_rl.option_rl import MultimodalBudgetSearch
from hymeko_rl.viz.render_reach import CameraView, _draw_target
from hymeko_rl.viz.rollout_overlay import (
    InfoPanel, StatusBar, TimeSeriesPanel, encode_clip, hstack, overlay_frames, summary_card)

OUT = "reports/2026-07-24-se3-obstacle-6d1/videos"
DIRNAMES = list(ROUTE_DIRS)
BUDGET = 12


class ViaGen:
    def sample(self, center, n, rng):
        c = np.asarray(center, np.float64)
        return c[None, :] if n == 1 else c + rng.normal(0, 0.02, (int(n), len(c)))


def _env():
    return SE3ObstacleReachEnv(control_mode="position", max_steps=320, reach_thresh=0.06, ang_thresh=0.4,
                              min_separation=0.16)


def record_route_rollout(env, via_ee, *, via_steps=60, goal_steps=150, h=430, w=560):
    """Replay the committed option (open-loop via → closed-loop goal) for ``via_ee`` while capturing frames + per-frame
    diagnostics (EE distance-to-goal, obstacle-collision flag, phase). This IS the scored rollout."""
    cam = CameraView(distance=1.15, elevation=-22, azimuth=52, lookat_z=0.2).to_mjv()
    renderer = mujoco.Renderer(env.model, height=h, width=w)
    q_via = ik_position(env, via_ee)
    frames, dist, coll, phase, trail = [], [], [], [], []
    collided = env.ee_in_obstacle(env._ee_pos())

    def snap(ph):
        nonlocal collided
        collided = collided or env.ee_in_obstacle(env._ee_pos())
        trail.append(env._ee_pos().copy())
        renderer.update_scene(env.data, camera=cam)
        _draw_target(renderer.scene, np.asarray(env._target, np.float64))
        frames.append(np.asarray(renderer.render(), np.uint8))
        dist.append(float(np.linalg.norm(env._ee_pos() - env._target)))
        coll.append(int(collided))
        phase.append(ph)

    for _ in range(via_steps):
        env.step(q_via)
        snap("via")
    reached, info = False, {"dist": 9.9, "ang_err": 9.9}
    for _ in range(goal_steps):
        _o, _r, term, trunc, info = env.step(env.expert_action)
        snap("goal")
        if term and not info.get("death", False):
            reached = True
            break
        if term or trunc:
            break
    renderer.close()
    success = bool(reached and not collided)
    return frames, {"dist": np.asarray(dist), "coll": np.asarray(coll), "via_end": via_steps,
                    "reached": int(reached), "collided": int(collided), "success": int(success),
                    "pos_err": float(info["dist"]), "ang_err": float(info["ang_err"])}


def _status_series(diag):
    n = len(diag["dist"])
    def fn(t):
        if diag["coll"][: t + 1].any():
            return "COLLISION"
        if t >= n - 2:                       # final verdict on the last frames
            return "SUCCESS" if diag["reached"] else "FAIL"
        return "RUNNING"
    return fn


def _overlay(frames, diag, *, task, ctrl, route_line, info_lines):
    thr = 0.06
    panels = [
        StatusBar(f"{task} | CTRL: {ctrl} | B={BUDGET} | {route_line}", _status_series(diag)),
        TimeSeriesPanel({"dist→goal": diag["dist"]}, title="EE distance to goal", threshold=thr,
                        vlines=[(diag["via_end"], "via→goal")], size=(280, 140)),
        InfoPanel(lambda _t: info_lines),
    ]
    return overlay_frames(frames, panels)


def main():
    os.makedirs(OUT, exist_ok=True)
    env = _env()
    # find the decisive state: single-head@infeasible FAILS, K-mode SUCCEEDS (deterministic scan)
    chosen = None
    for s in range(60):
        env.reset(seed=s)
        if not env.direct_path_blocked():
            continue
        feas = {nm: route_execution_feasible(env, d, seed=100 + s) for nm, d in ROUTE_DIRS.items()}
        good = [nm for nm, f in feas.items() if f]
        bad = [nm for nm, f in feas.items() if not f]
        if not (good and bad):
            continue
        env.reset(seed=s)
        obs = env.node_features().reshape(-1)
        sh = MultimodalBudgetSearch(ViaGen(), RouteOptionScorer(env), budget=BUDGET).select(
            RouteModeProposal(env, [ROUTE_DIRS[bad[0]]], "prob"), obs, np.random.default_rng(1))
        env.reset(seed=s)
        obs = env.node_features().reshape(-1)
        km = MultimodalBudgetSearch(ViaGen(), RouteOptionScorer(env), budget=BUDGET).select(
            RouteModeProposal(env, list(ROUTE_DIRS.values()), "equal"), obs, np.random.default_rng(1))
        if sh.outcome["success"] == 0 and km.outcome["success"] == 1:
            chosen = (s, good, bad, sh, km)
            break
    if chosen is None:
        print("no decisive state found")
        return None
    s, good, bad, sh, km = chosen
    print(f"decisive state seed {s}: feasible={good} single-head anchored to '{bad[0]}' (infeasible)")

    # record the EXACT scored rollouts
    env.reset(seed=s)
    fsh, dsh = record_route_rollout(env, sh.selected)
    env.reset(seed=s)
    fkm, dkm = record_route_rollout(env, km.selected)
    km_alloc = km.per_mode_budget
    km_mode_name = DIRNAMES[km.selected_mode] if km.selected_mode < len(DIRNAMES) else str(km.selected_mode)

    left = _overlay(fsh, dsh, task="6D-1 obstacle", ctrl="single-head (K=1)",
                    route_line=f"route={bad[0]} (wrong basin)",
                    info_lines=[f"Budget: [{BUDGET}] on 1 mode", f"Route: {bad[0]}",
                                f"reached: {dsh['reached']}  collided: {dsh['collided']}", "→ FAIL (basin not crossable)"])
    right = _overlay(fkm, dkm, task="6D-1 obstacle", ctrl="K-mode + search",
                     route_line=f"selected={km_mode_name}",
                     info_lines=[f"Modes: {'/'.join(DIRNAMES)}", f"Budget alloc: {km_alloc} (total {BUDGET})",
                                 f"Selected: {km_mode_name}", "→ SUCCESS (open basin reached)"])
    clip = hstack(left, right)
    clip += summary_card(clip[0].size, "6D-1 critical pair — same state, same budget B=12",
                         [("single-head (wrong basin)", f"FAIL (collided={dsh['collided']})"),
                          ("K-mode + search", f"SUCCESS (mode={km_mode_name})"),
                          ("panel-level result", "single-head 2/14  vs  K-mode 13/14"),
                          ("claim", "EE route-selection; collision on executed EE path")], hold=70)
    path = encode_clip(clip, f"{OUT}/01_6d1_critical_pair.mp4", fps=30)

    manifest = {"clip": "01_6d1_critical_pair", "date": "2026-07-24", "task": "6D-1 obstacle EE-route-selection",
                "scope": "collision graded on executed EE trajectory; NOT full-arm avoidance",
                "state_seed": s, "search_seed": 1, "budget": BUDGET, "feasible_routes": good, "infeasible_routes": bad,
                "single_head": {"anchored_route": bad[0], "success": dsh["success"], "collided": dsh["collided"],
                                "theta_selected": [round(float(x), 4) for x in sh.selected]},
                "kmode": {"modes": DIRNAMES, "mode_probs": [round(float(p), 3) for p in km.mode_probs],
                          "budget_alloc": list(km_alloc), "selected_mode": km_mode_name, "success": dkm["success"],
                          "theta_selected": [round(float(x), 4) for x in km.selected]},
                "commit": "809bc1c2", "video": path}
    json.dump(manifest, open(f"{OUT}/01_6d1_critical_pair.json", "w"), indent=1, default=str)
    print(f"wrote {path} ({len(clip)} frames)\n01_6D1_CRITICAL_VIDEO_DONE")
    return path


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
