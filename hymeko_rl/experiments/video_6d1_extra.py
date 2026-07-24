"""Videos 02 + 03 — the rest of the 6D-1 trilogy, reusing the overlay framework + the recorder (no new rendering code).

  02_6d1_route_modes.mp4  — one SUCCESSFUL execution-eligible example per route family (LEFT / RIGHT / OVER / UNDER),
                            concatenated with a title card each. Caption: EE route-selection; collision on executed EE path.
  03_6d1_equal_budget_k1_vs_kmode.mp4 — the FAIR deploy pair: single-head@ARGMAX (its own best-guess route) vs K-mode,
                            SAME total budget B=12, side by side, with the integer allocation on screen.

Every clip renders the EXACT scored rollout + writes a manifest.
"""
import json
import os
import sys

import numpy as np

from hymeko_rl.env.se3_reach_option import ROUTE_DIRS, RouteModeProposal, RouteOptionScorer, route_execution_feasible
from hymeko_rl.option_rl import MultimodalBudgetSearch
from hymeko_rl.viz.rollout_overlay import (
    InfoPanel, StatusBar, TimeSeriesPanel, encode_clip, hstack, overlay_frames, summary_card)

from hymeko_rl.experiments.video_6d1_critical import (  # reuse the recorder + helpers (no duplication)
    BUDGET, DIRNAMES, OUT, ViaGen, _env, _status_series, record_route_rollout)


def _clip(frames, diag, *, task, ctrl, route_line, info_lines):
    panels = [StatusBar(f"{task} | CTRL: {ctrl} | B={BUDGET} | {route_line}", _status_series(diag)),
              TimeSeriesPanel({"dist→goal": diag["dist"]}, title="EE distance to goal", threshold=0.06,
                              vlines=[(diag["via_end"], "via→goal")], size=(280, 140)),
              InfoPanel(lambda _t: info_lines)]
    return overlay_frames(frames, panels)


def _title(size, text, hold=24):
    return summary_card(size, text, [], hold=hold)


def route_montage():
    """One successful example per route family, from execution-eligible states (K-mode picks + delivers that route)."""
    env = _env()
    want = {nm: None for nm in DIRNAMES}
    for s in range(80):
        if all(want.values()):
            break
        env.reset(seed=s)
        if not env.direct_path_blocked():
            continue
        feas = {nm: route_execution_feasible(env, d, seed=100 + s) for nm, d in ROUTE_DIRS.items()}
        if not any(feas.values()):
            continue
        env.reset(seed=s)
        obs = env.node_features().reshape(-1)
        km = MultimodalBudgetSearch(ViaGen(), RouteOptionScorer(env), budget=BUDGET).select(
            RouteModeProposal(env, list(ROUTE_DIRS.values()), "equal"), obs, np.random.default_rng(1))
        nm = DIRNAMES[km.selected_mode] if km.selected_mode < len(DIRNAMES) else None
        if nm and want.get(nm) is None and km.outcome["success"]:
            want[nm] = (s, km)
    clip, manifest = [], []
    size = None
    for nm in DIRNAMES:
        if want[nm] is None:
            continue
        s, km = want[nm]
        env.reset(seed=s)
        f, d = record_route_rollout(env, km.selected)
        oc = _clip(f, d, task="6D-1 route montage", ctrl="K-mode + search", route_line=f"ROUTE: {nm.upper()}",
                   info_lines=[f"seed {s}  budget alloc {km.per_mode_budget}", f"selected route: {nm}",
                               "→ SUCCESS (collision-free EE path)"])
        size = oc[0].size
        clip += _title(size, f"ROUTE FAMILY: {nm.upper()}", hold=20) + oc
        manifest.append({"route": nm, "state_seed": s, "budget_alloc": list(km.per_mode_budget),
                         "success": km.outcome["success"]})
    if not clip:
        return None
    clip += summary_card(size, "6D-1 route montage — state-dependent route selection",
                         [("routes shown", "  ".join(m["route"] for m in manifest)),
                          ("claim", "EE route-selection; collision on executed EE path")], hold=50)
    path = encode_clip(clip, f"{OUT}/02_6d1_route_modes.mp4", fps=30)
    json.dump({"clip": "02_6d1_route_modes", "examples": manifest, "commit": "bbffe8da"},
              open(f"{OUT}/02_6d1_route_modes.json", "w"), indent=1, default=str)
    print(f"wrote {path} ({len(clip)} frames; routes {[m['route'] for m in manifest]})")
    return path


def equal_budget_pair():
    """FAIR pair: single-head@ARGMAX vs K-mode at SAME budget. Find a state where the argmax single-head FAILS but
    K-mode succeeds (the deploy-failure case that motivates multimodality)."""
    env = _env()
    chosen = None
    for s in range(80):
        env.reset(seed=s)
        if not env.direct_path_blocked():
            continue
        feas = {nm: route_execution_feasible(env, d, seed=100 + s) for nm, d in ROUTE_DIRS.items()}
        if not (any(feas.values()) and not all(feas.values())):
            continue
        env.reset(seed=s)
        obs = env.node_features().reshape(-1)
        k1 = MultimodalBudgetSearch(ViaGen(), RouteOptionScorer(env), budget=BUDGET).select(
            RouteModeProposal(env, list(ROUTE_DIRS.values()), "prob", k=1), obs, np.random.default_rng(2))
        env.reset(seed=s)
        obs = env.node_features().reshape(-1)
        km = MultimodalBudgetSearch(ViaGen(), RouteOptionScorer(env), budget=BUDGET).select(
            RouteModeProposal(env, list(ROUTE_DIRS.values()), "equal"), obs, np.random.default_rng(2))
        if k1.outcome["success"] == 0 and km.outcome["success"] == 1:
            chosen = (s, k1, km)
            break
    if chosen is None:
        print("no fair-deploy decisive state found")
        return None
    s, k1, km = chosen
    k1_name = DIRNAMES[k1.selected_mode] if k1.selected_mode < len(DIRNAMES) else str(k1.selected_mode)
    km_name = DIRNAMES[km.selected_mode] if km.selected_mode < len(DIRNAMES) else str(km.selected_mode)
    env.reset(seed=s)
    f1, d1 = record_route_rollout(env, k1.selected)
    env.reset(seed=s)
    fk, dk = record_route_rollout(env, km.selected)
    left = _clip(f1, d1, task="6D-1 equal budget", ctrl="single-head@argmax", route_line=f"argmax route={k1_name}",
                 info_lines=[f"Budget: [{BUDGET}] on 1 mode (argmax)", f"route: {k1_name}", "→ FAIL (argmax basin wrong)"])
    right = _clip(fk, dk, task="6D-1 equal budget", ctrl="K-mode + search", route_line=f"selected={km_name}",
                  info_lines=[f"Budget alloc: {km.per_mode_budget} (total {BUDGET})", f"selected: {km_name}",
                              "→ SUCCESS (coverage finds open basin)"])
    clip = hstack(left, right)
    clip += summary_card(clip[0].size, "6D-1 equal budget B=12 — argmax single-head vs K-mode",
                         [("single-head@argmax", f"FAIL (route {k1_name})"),
                          ("K-mode + search", f"SUCCESS (route {km_name}, alloc {km.per_mode_budget})"),
                          ("why", "same 12 candidates; K-mode spends them across basins")], hold=70)
    path = encode_clip(clip, f"{OUT}/03_6d1_equal_budget_k1_vs_kmode.mp4", fps=30)
    json.dump({"clip": "03_6d1_equal_budget_k1_vs_kmode", "state_seed": s, "budget": BUDGET,
               "single_head": {"argmax_route": k1_name, "success": d1["success"]},
               "kmode": {"selected": km_name, "alloc": list(km.per_mode_budget), "success": dk["success"]},
               "commit": "bbffe8da"}, open(f"{OUT}/03_6d1_equal_budget_k1_vs_kmode.json", "w"), indent=1, default=str)
    print(f"wrote {path} ({len(clip)} frames; state {s}, K1@{k1_name} FAIL vs K-mode@{km_name} SUCCESS)")
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    route_montage()
    equal_budget_pair()
    print("6D1_EXTRA_VIDEOS_DONE")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
