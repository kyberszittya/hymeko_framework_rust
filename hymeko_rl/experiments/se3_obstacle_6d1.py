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


def build_eligible_panel(env, state_seeds=range(60), feas_seed_base=100, want=None):
    """Fixed, controller-independent eligibility over an EXPLICIT set of state seeds (so replications use FRESH panels):
    direct path blocked ∧ ≥1 execution-feasible route ∧ ≥1 execution-infeasible route (a right AND a wrong choice both
    exist). Feasibility is EXECUTION-based (not the geometric polyline). Returns [(seed, feasible[], infeasible[])] + rate."""
    panel, scanned = [], 0
    for s in state_seeds:
        scanned += 1
        env.reset(seed=int(s))
        if not env.direct_path_blocked():
            continue
        feas = {nm: route_execution_feasible(env, d, seed=feas_seed_base + int(s)) for nm, d in ROUTE_DIRS.items()}
        good = [nm for nm, f in feas.items() if f]
        bad = [nm for nm, f in feas.items() if not f]
        if good and bad:
            panel.append({"seed": int(s), "feasible": good, "infeasible": bad})
            if want and len(panel) >= want:
                break
    return panel, round(len(panel) / max(1, scanned), 3)


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


def deploy_state(env, item, K, alloc, B, seed_rng):
    """One (state, K, allocation, budget) deploy outcome — success + failure mode (collided / not-reached) + selected
    route family. B=0 = direct proposal (top mode, no search)."""
    env.reset(seed=item["seed"])
    obs = env.node_features().reshape(-1)
    prop = RouteModeProposal(env, list(ROUTE_DIRS.values()), alloc, k=K)
    if B == 0:
        modes = prop.modes(obs)
        q, v = env.data.qpos.copy(), env.data.qvel.copy()
        out = execute_route_option(env, modes[0].center)
        env.data.qpos[:], env.data.qvel[:] = q, v
        mujoco.mj_forward(env.model, env.data)
        env._step = 0
        return {**out, "selected_mode": int(modes[0].mode_id)}
    prov = MultimodalBudgetSearch(ViaGen(), RouteOptionScorer(env), budget=B).select(prop, obs, np.random.default_rng(seed_rng))
    return {**prov.outcome, "selected_mode": int(prov.selected_mode)}


def _hier_bootstrap(per_seed_deltas, iters=5000, boot_seed=0):
    """Hierarchical seed→state bootstrap of the mean paired Δ (K-mode − single-head): resample SEEDS with replacement,
    then STATES within each resampled seed, average the per-state Δ. Returns median + 95% CI. per_seed_deltas = list (per
    seed) of per-state Δ arrays."""
    rng = np.random.default_rng(boot_seed)
    seeds = [np.asarray(d, np.float64) for d in per_seed_deltas if len(d)]
    if not seeds:
        return {"median": 0.0, "lo": 0.0, "hi": 0.0}
    boot = []
    for _ in range(iters):
        chosen = [seeds[i] for i in rng.integers(0, len(seeds), len(seeds))]
        vals = [s[rng.integers(0, len(s), len(s))].mean() for s in chosen]
        boot.append(float(np.mean(vals)))
    return {"median": round(float(np.median(boot)), 4), "lo": round(float(np.percentile(boot, 2.5)), 4),
            "hi": round(float(np.percentile(boot, 97.5)), 4)}


def harden(n_seeds=6, budgets=(0, 4, 8, 12, 24), headline_B=12, kmode_alloc="equal", kmode_K=4, env_factory=None):
    """Seed-hardening: N replications, each a FRESH eligible panel (disjoint state-seed range) + a paired search seed.
    Headline Δ = KMODE(K=kmode_K, alloc) − single-head@argmax at ``headline_B``, per (seed, state). Reports per-seed Δ,
    seed median/IQR, hierarchical bootstrap, fresh critical-pair replication, route-family + failure decomposition.
    ``env_factory`` (default fast ``_env``) lets the realistic velocity-limited env + physical horizons be injected — the
    eligible panel is then RECOMPUTED under that limited executor (not reused from the unlimited run)."""
    env_factory = env_factory or _env
    per_seed, curves = [], {"K1": {b: [] for b in budgets}, "KM": {b: [] for b in budgets}}
    per_seed_deltas = []
    for i in range(n_seeds):
        env = env_factory()
        lo = i * 90
        panel, rate = build_eligible_panel(env, state_seeds=range(lo, lo + 55), feas_seed_base=300 + i, want=14)
        srng = 40 + i
        # fresh critical pair on THIS panel
        sh, km, _rows = critical_pair(env, panel, budget=12, seed_rng=srng)
        # per-state deploy bits across budgets
        deltas, fam = [], {"left": 0, "right": 0, "over": 0, "under": 0}
        fails = {"collided": 0, "timeout": 0, "not_reached": 0}   # timeout = ran out of physical time (separate category)
        for it in panel:
            for nm in it["feasible"]:
                fam[nm] += 1
            for b in budgets:
                k1 = deploy_state(env, it, 1, "prob", b, srng)
                kmv = deploy_state(env, it, kmode_K, kmode_alloc, b, srng)
                curves["K1"][b].append(k1["success"])
                curves["KM"][b].append(kmv["success"])
                if b == headline_B:
                    deltas.append(kmv["success"] - k1["success"])
                    if not kmv["success"]:
                        key = "collided" if kmv.get("collided") else ("timeout" if kmv.get("timeout") else "not_reached")
                        fails[key] += 1
        per_seed_deltas.append(deltas)
        per_seed.append({"seed_group": i, "eligibility_rate": rate, "n_states": len(panel),
                         "critical_pair": {"single_head_wrong": sh, "kmode": km},
                         "delta_headline_mean": round(float(np.mean(deltas)) if deltas else 0.0, 3),
                         "route_family_feasible": fam, "kmode_failure_decomp": fails})
    seed_means = [p["delta_headline_mean"] for p in per_seed]
    n_pos = sum(m > 0 for m in seed_means)
    agg = {"n_seeds": n_seeds, "headline_budget": headline_B, "kmode": f"K{kmode_K}_{kmode_alloc}",
           "per_seed_delta_mean": seed_means, "seed_median_delta": round(float(np.median(seed_means)), 3),
           "seed_iqr": [round(float(np.percentile(seed_means, 25)), 3), round(float(np.percentile(seed_means, 75)), 3)],
           "n_seeds_positive": int(n_pos), "hier_bootstrap": _hier_bootstrap(per_seed_deltas),
           "budget_curve_K1": {b: round(float(np.mean(curves["K1"][b])), 3) for b in budgets},
           "budget_curve_KM": {b: round(float(np.mean(curves["KM"][b])), 3) for b in budgets}}
    strong = (n_pos >= max(5, n_seeds - 1) and agg["seed_median_delta"] > 0.05 and agg["hier_bootstrap"]["lo"] > 0
              and all(p["critical_pair"]["kmode"] > p["critical_pair"]["single_head_wrong"] for p in per_seed))
    agg["verdict"] = ("MULTIMODAL_POLICY_SEARCH_VALIDATED_ON_GEOMETRICALLY_SEPARATED_ROUTE_BASINS" if strong
                      else "STILL_PILOT")
    return {"per_seed": per_seed, "aggregate": agg}


def obstacle_shift_control(budgets=(4, 12), kmode_K=4, kmode_alloc="equal", env_for_half=None):
    """Generalisation control: does the K-mode advantage survive a bounded change of obstacle GEOMETRY (not just fresh
    seeds of the same box)? Vary half-extents + a small center shift; a fresh eligible panel per geometry; check
    K-mode still beats single-head. Guards against memorising one obstacle layout. ``env_for_half`` (default fast) builds
    the env for a given obstacle half-extent — inject a realistic velocity-limited variant to run under motion limits."""
    def _default_for_half(half):
        return SE3ObstacleReachEnv(control_mode="position", max_steps=320, reach_thresh=0.06, ang_thresh=0.4,
                                   min_separation=0.16, obstacle_half=half)
    env_for_half = env_for_half or _default_for_half
    geoms = {"wider": (0.040, 0.040, 0.075), "taller": (0.028, 0.028, 0.11), "narrow": (0.020, 0.020, 0.06)}
    out = {}
    for name, half in geoms.items():
        env = env_for_half(half)
        panel, rate = build_eligible_panel(env, state_seeds=range(700, 760), feas_seed_base=900, want=10)
        row = {"eligibility_rate": rate, "n_states": len(panel)}
        for b in budgets:
            k1 = np.mean([deploy_state(env, it, 1, "prob", b, 7)["success"] for it in panel]) if panel else 0.0
            km = np.mean([deploy_state(env, it, kmode_K, kmode_alloc, b, 7)["success"] for it in panel]) if panel else 0.0
            row[f"B{b}"] = {"single_head": round(float(k1), 3), "kmode": round(float(km), 3),
                           "delta": round(float(km - k1), 3)}
        out[name] = row
    return out


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


def plot_harden(agg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    budgets = sorted(agg["budget_curve_K1"], key=int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax1.plot(budgets, [agg["budget_curve_K1"][b] for b in budgets], "o-", color="#444", label="single-head (K1)", lw=2)
    ax1.plot(budgets, [agg["budget_curve_KM"][b] for b in budgets], "s-", color="#2a7", label=f"K-mode {agg['kmode']}", lw=2)
    ax1.set_xlabel("total budget B (equal)")
    ax1.set_ylabel("collision-free success (all seeds×states)")
    ax1.set_title("6D-1 hardened: single-head vs K-mode across budgets", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.02)
    means = agg["per_seed_delta_mean"]
    ax2.bar(range(len(means)), means, color=["#2a7" if m > 0 else "#c44" for m in means])
    ax2.axhline(0, color="k", lw=0.8)
    hb = agg["hier_bootstrap"]
    ax2.axhline(agg["seed_median_delta"], color="#37a", ls="--", label=f"seed-median {agg['seed_median_delta']:+.3f}")
    ax2.axhspan(hb["lo"], hb["hi"], alpha=0.15, color="#37a", label=f"hier-boot95 [{hb['lo']:+.2f},{hb['hi']:+.2f}]")
    ax2.set_xlabel("seed replication")
    ax2.set_ylabel(f"Δ success (K-mode − single-head) @ B={agg['headline_budget']}")
    ax2.set_title(f"per-seed Δ ({agg['n_seeds_positive']}/{agg['n_seeds']} positive)", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main_harden():
    os.makedirs(OUT, exist_ok=True)
    print("== 6D-1 SEED HARDENING (6 replications, fresh panels, paired search seeds) ==")
    hard = harden(n_seeds=6)
    agg = hard["aggregate"]
    for p in hard["per_seed"]:
        print(f"  seed {p['seed_group']}: elig {p['eligibility_rate']} n {p['n_states']} "
              f"critical {p['critical_pair']['single_head_wrong']}→{p['critical_pair']['kmode']} "
              f"Δ@B12 {p['delta_headline_mean']:+.3f} fails {p['kmode_failure_decomp']}")
    print(f"\n  per-seed Δ: {agg['per_seed_delta_mean']}")
    print(f"  seed-median Δ {agg['seed_median_delta']} IQR {agg['seed_iqr']} | positive {agg['n_seeds_positive']}/{agg['n_seeds']}")
    print(f"  hierarchical seed→state bootstrap: {agg['hier_bootstrap']}")
    print(f"  budget curve K1 {agg['budget_curve_K1']}")
    print(f"  budget curve KM {agg['budget_curve_KM']}")
    print("\n== OBSTACLE-SHIFT GENERALISATION CONTROL (fresh geometry, not just fresh seeds) ==")
    shift = obstacle_shift_control()
    for name, row in shift.items():
        print(f"  {name}: elig {row['eligibility_rate']} " + " ".join(f"{k}Δ={row[k]['delta']}" for k in row if k.startswith("B")))
    print(f"\n  → VERDICT: {agg['verdict']}")
    manifest = {"contract": "SE3_OBSTACLE_6D1_HARDENING", "date": "2026-07-24",
                "claim": "multimodal policy search improves EE route selection in geometrically-separated local proposal basins (NOT full-arm avoidance)",
                "harden": hard, "obstacle_shift_control": shift, "verdict": agg["verdict"]}
    json.dump(manifest, open(f"{OUT}/obstacle_6d1_harden.json", "w"), indent=1, default=str)
    plot_harden(agg, f"{OUT}/obstacle_6d1_harden.png")
    print(f"\nartifact: {OUT}/obstacle_6d1_harden.json\nSE3_OBSTACLE_6D1_HARDEN_DONE")
    return agg["verdict"]


if __name__ == "__main__":
    if "--harden" in sys.argv:
        sys.exit(0 if main_harden() else 1)
    sys.exit(0 if main() else 1)
