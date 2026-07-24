"""6D-1 — EE-route-selection benchmark on the frozen runtime (NOT full-arm collision avoidance; obstacle is contype=0
and collision is graded by the end-effector path's AABB intersection). Route modes are GEOMETRICALLY-SEPARATED /
locally-disconnected proposal basins (a left-mode's local jitter cannot slip to the right/over route), not strict
homotopy classes.

Closes the critical validity pair (the gate before any budget×K×allocation grid):
  on the SAME eligible state / budget / RNG / executor / certificate,
    single-head anchored to an execution-INFEASIBLE route  → should FAIL,
    K-mode covering an execution-FEASIBLE route            → should RECOVER.
Plus: (a) a fixed, controller-independent eligible panel (direct blocked ∧ ≥1 exec-feasible ∧ ≥1 exec-infeasible route),
(b) the mode-non-crossing check (a mode's candidates stay in its route-family region), (c) realised allocation logging
(requested prob-reweighting → integer per-mode candidate counts, e.g. top_probe B=12,K=4 → [9,1,1,1]).
"""
import json
import os
import sys

import mujoco
import numpy as np

from hymeko_rl.env.se3_obstacle_reach_env import SE3ObstacleReachEnv
from hymeko_rl.env.se3_reach_option import (
    ROUTE_DIRS, RouteModeProposal, RouteOptionScorer, execute_route_option, route_execution_feasible, route_via)
from hymeko_rl.option_rl import MultimodalBudgetSearch, allocate_budget

OUT = "reports/2026-07-24-se3-obstacle-6d1"


class ViaGen:
    """Local via jitter (3-D), small enough to stay inside the mode's route-family basin."""

    def __init__(self, std=0.02):
        self.std = std

    def sample(self, center, n, rng):
        c = np.asarray(center, np.float64)
        return c[None, :] if n == 1 else c + rng.normal(0, self.std, (int(n), len(c)))


def _env():
    return SE3ObstacleReachEnv(control_mode="position", max_steps=320, reach_thresh=0.06, ang_thresh=0.4,
                              min_separation=0.16)


def build_eligible_panel(env, n_scan=60):
    """Fixed, controller-independent eligibility: direct path blocked ∧ ≥1 execution-feasible route ∧ ≥1 execution-
    infeasible route (so a right AND a wrong choice both exist). Returns [(seed, feasible[], infeasible[])] + rate."""
    panel = []
    for s in range(n_scan):
        env.reset(seed=s)
        if not env.direct_path_blocked():
            continue
        feas = {nm: route_execution_feasible(env, d, seed=100 + s) for nm, d in ROUTE_DIRS.items()}
        good = [nm for nm, f in feas.items() if f]
        bad = [nm for nm, f in feas.items() if not f]
        if good and bad:
            panel.append({"seed": s, "feasible": good, "infeasible": bad})
    return panel, round(len(panel) / max(1, n_scan), 3)


def mode_non_crossing(env, direction, n=32, std=0.02):
    """A mode's jittered candidates must stay in its route-family region: the via offset from the midpoint must keep the
    SAME sign along `direction` as the mode centre. Returns the fraction that stay in-basin (want 1.0)."""
    mid = 0.5 * (env._start_ee + np.asarray(env._target, np.float32))
    d = np.asarray(direction, np.float32)
    base = route_via(env, direction) - mid
    cands = ViaGen(std).sample(route_via(env, direction), n, np.random.default_rng(0))
    proj = (cands - mid) @ d
    return float(np.mean(np.sign(proj) == np.sign(base @ d)))


def critical_pair(env, panel, budget=12, seed_rng=1):
    """On each eligible state: single-head@(an infeasible route) vs K-mode covering all routes (equal alloc), SAME
    budget / RNG / executor / certificate."""
    sh, km, rows = 0, 0, []
    for item in panel:
        s = item["seed"]
        env.reset(seed=s)
        obs = env.node_features().reshape(-1)
        bad_dir = ROUTE_DIRS[item["infeasible"][0]]
        prov_sh = MultimodalBudgetSearch(ViaGen(), RouteOptionScorer(env), budget=budget).select(
            RouteModeProposal(env, [bad_dir], "prob"), obs, np.random.default_rng(seed_rng))
        env.reset(seed=s)
        obs = env.node_features().reshape(-1)
        prov_km = MultimodalBudgetSearch(ViaGen(), RouteOptionScorer(env), budget=budget).select(
            RouteModeProposal(env, list(ROUTE_DIRS.values()), "equal"), obs, np.random.default_rng(seed_rng))
        sh += prov_sh.outcome["success"]
        km += prov_km.outcome["success"]
        rows.append({"seed": s, "feasible": item["feasible"], "single_head_wrong_success": prov_sh.outcome["success"],
                     "kmode_success": prov_km.outcome["success"], "kmode_selected_mode": int(prov_km.selected_mode)})
    return sh, km, rows


def _deploy_success(env, panel, K, alloc, B, seed_rng):
    """Success rate over the eligible panel for one (K, allocation, budget) cell — the FAIR deploy setting: the proposal
    picks its own top-K routes by prior (K=1 = single-head@argmax). B=0 = execute the top-prior via directly (no search).
    Every arm at a fixed B spends exactly B candidate rollouts (equal compute)."""
    dirs = list(ROUTE_DIRS.values())
    succ = 0
    for item in panel:
        env.reset(seed=item["seed"])
        obs = env.node_features().reshape(-1)
        prop = RouteModeProposal(env, dirs, alloc, k=K)
        if B == 0:
            modes = prop.modes(obs)
            q, v = env.data.qpos.copy(), env.data.qvel.copy()
            out = execute_route_option(env, modes[0].center)
            env.data.qpos[:], env.data.qvel[:] = q, v
            mujoco.mj_forward(env.model, env.data)
            env._step = 0
            succ += out["success"]
        else:
            prov = MultimodalBudgetSearch(ViaGen(), RouteOptionScorer(env), budget=B).select(
                prop, obs, np.random.default_rng(seed_rng))
            succ += prov.outcome["success"]
    return succ / max(1, len(panel))


def fair_grid(env, panel, budgets=(0, 4, 8, 12, 24), ks=(1, 2, 3, 4),
              allocations=("prob", "equal", "top_probe"), seeds=(1, 2)):
    """The FAIR deploy grid: K=1 single-head@argmax vs K>1 × allocation, at equal total budget B. 2 search-seeds ⇒
    PILOT (seed-aware). Returns {cell: mean success over states×seeds}."""
    grid = {}
    for B in budgets:
        for K in ks:
            for alloc in (["prob"] if K == 1 else allocations):
                rates = [_deploy_success(env, panel, K, alloc, B, s) for s in seeds]
                grid[f"B{B}_K{K}_{alloc}"] = round(float(np.mean(rates)), 3)
    return grid


def plot_grid(grid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    budgets = [0, 4, 8, 12, 24]
    series = {"K1 single-head": ("K1_prob", "#444", "o-"), "K4 prob": ("K4_prob", "#37a", "s-"),
              "K4 equal": ("K4_equal", "#3a7", "^-"), "K4 top_probe": ("K4_top_probe", "#e73", "d-")}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for label, (suf, col, sty) in series.items():
        ys = [grid.get(f"B{b}_{suf}") for b in budgets]
        if all(y is not None for y in ys):
            ax.plot(budgets, ys, sty, color=col, label=label, linewidth=2, markersize=6)
    ax.set_xlabel("total candidate budget B (equal across arms)")
    ax.set_ylabel("collision-free pose-reach success (eligible panel)")
    ax.set_title("6D-1 EE-route-selection: single-head@argmax vs K-mode at EQUAL budget\n"
                 "(obstacle contype=0, EE-AABB collision — NOT full-arm avoidance)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def _grid_verdict(grid):
    """Map the fair grid to the pre-registered verdicts."""
    eq = [(b, grid.get(f"B{b}_K4_top_probe", 0) - grid.get(f"B{b}_K1_prob", 0)) for b in (4, 8, 12)]
    big = grid.get("B24_K4_top_probe", 0) - grid.get("B24_K1_prob", 0)
    if any(d > 0.1 for _b, d in eq):
        return "MULTIMODAL_POLICY_SEARCH_VALIDATED_ON_DISJOINT_PATH_MODES"
    if big > 0.1:
        return "MULTIMODAL_REPRESENTATION_VALID__SEARCH_BUDGET_DOMINANT"
    return "MODE_ROUTING_OR_TRAINING_INSUFFICIENT_OR_NOT_A_DEPLOY_LEVER"


def realized_allocations():
    """Prove the requested allocation strategy → the intended integer per-mode candidate counts (frozen allocate_budget
    on the reweighted probs). Especially top_probe should be [bulk,1,1,1]."""
    out = {}
    priors = {"equal": [0.25, 0.25, 0.25, 0.25], "top_probe": [0.97, 0.01, 0.01, 0.01], "prob": [0.5, 0.3, 0.15, 0.05]}
    for B in (0, 4, 8, 12, 24):
        out[B] = {name: allocate_budget(p, B) for name, p in priors.items()}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    env = _env()
    panel, rate = build_eligible_panel(env, n_scan=60)
    print(f"eligible panel: {len(panel)} states (rate {rate}); route families vary by state:")
    for it in panel[:8]:
        print(f"  seed {it['seed']}: feasible={it['feasible']} infeasible={it['infeasible']}")

    # mode non-crossing (basin locality) on a few eligible states
    ncx = []
    for it in panel[:6]:
        env.reset(seed=it["seed"])
        ncx += [mode_non_crossing(env, ROUTE_DIRS[nm]) for nm in ("left", "over")]
    print(f"\nmode non-crossing (candidates stay in route-family basin): min {min(ncx):.3f} mean {np.mean(ncx):.3f} (want 1.0)")

    alloc = realized_allocations()
    print(f"\nrealised allocations (B=12): {alloc[12]}")
    assert alloc[12]["top_probe"] == [9, 1, 1, 1], alloc[12]["top_probe"]
    print("  top_probe B=12,K=4 = [9,1,1,1] ✓")

    sh, km, rows = critical_pair(env, panel, budget=12)
    n = len(panel)
    print(f"\n== CRITICAL PAIR (n={n} eligible, budget=12, same RNG/executor/cert) ==")
    print(f"  single-head anchored to an INFEASIBLE route: {sh}/{n} success")
    print(f"  K-mode covering all routes (equal alloc):    {km}/{n} success")
    crit_verdict = ("SINGLE_WRONG_MODE_FAILS_KMODE_RECOVERS" if km > sh + max(1, n // 5)
                    else "CRITICAL_PAIR_NOT_YET_SEPARATED")
    print(f"  → {crit_verdict}")

    print("\n== FAIR DEPLOY GRID (single-head@argmax vs K-mode × allocation × budget; 2 seeds = PILOT) ==")
    grid = fair_grid(env, panel)
    for b in (0, 4, 8, 12, 24):
        row = "  ".join(f"{k.split('_',1)[1]}={grid[k]}" for k in grid if k.startswith(f"B{b}_"))
        print(f"  B={b:2d}: {row}")
    grid_verdict = _grid_verdict(grid)
    print(f"  → FAIR-GRID VERDICT: {grid_verdict}")
    plot_grid(grid, f"{OUT}/obstacle_6d1_grid.png")

    manifest = {"contract": "SE3_OBSTACLE_6D1_EE_ROUTE_SELECTION", "date": "2026-07-24",
                "claim_scope": "EE-route-selection (obstacle contype=0, EE-AABB collision); NOT full-arm avoidance",
                "route_modes": "geometrically-separated / locally-disconnected proposal basins",
                "eligibility": "direct blocked AND >=1 exec-feasible route AND >=1 exec-infeasible route (execution-based)",
                "eligible_panel": panel, "eligibility_rate": rate,
                "mode_non_crossing_min": round(min(ncx), 3), "realized_allocations": alloc,
                "critical_pair": {"n": n, "single_head_wrong": sh, "kmode": km, "rows": rows, "verdict": crit_verdict},
                "fair_grid": grid, "fair_grid_verdict": grid_verdict}
    json.dump(manifest, open(f"{OUT}/obstacle_6d1_critical.json", "w"), indent=1, default=str)
    print(f"\nartifact: {OUT}/obstacle_6d1_critical.json\nSE3_OBSTACLE_6D1_DONE")
    return grid_verdict


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
