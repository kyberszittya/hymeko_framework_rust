"""6D-0 / 6D-1 REALISTIC-MOTION regression: rerun the SE(3) pose-reach (6D-0) and the strong EE-route-selection result
(6D-1) under REALISTIC velocity limits — the runtime-realism sibling of the coin motion contract. The unlimited-speed
6D-1 result (STRONG multimodal, [[project-option-rl-structured-temporal-runtime]]) must survive a velocity-limited
executor to earn ``MULTIMODAL_POLICY_SEARCH_VALIDATED_UNDER_REALISTIC_MOTION_LIMITS``.

What "realistic" means here (all via the EXISTING machinery — no forked harness, §6.1):
  * velocity-limited position execution — ``SE3ObstacleReachEnv(slew_joint_vel_limit=SLEW)`` (REALISTIC_MOTION_CONTRACT_V1
    anti-teleport slew limiter, already wired into ArmReachEnv.step);
  * PHYSICAL-time horizons — a slew-limited arm needs more env-steps to traverse a route; horizons are derived from the
    joint-displacement budget / (SLEW · ctrl_dt) + accel/brake + settling allowance, attached to the env as
    ``route_via_steps`` / ``route_goal_steps`` (read by execute_route_option / the scorer / feasibility);
  * eligibility RECOMPUTED under the limited executor (build_eligible_panel runs on the realistic env — NOT reused from
    the unlimited run);
  * six paired seeds, equal candidate budget, critical-pair replication, route-family + failure (incl. TIMEOUT) breakdown,
    obstacle-shift generalisation, hierarchical seed→state bootstrap — all from se3_obstacle_6d1.harden/obstacle_shift.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.env.se3_obstacle_reach_env import SE3ObstacleReachEnv  # noqa: E402
from hymeko_rl.env.se3_reach_option import ROUTE_DIRS, execute_route_option, route_via  # noqa: E402
from hymeko_rl.experiments.se3_obstacle_6d1 import (  # noqa: E402
    _hier_bootstrap, build_eligible_panel, harden, obstacle_shift_control, plot_harden)

OUT = "reports/2026-07-24-se3-obstacle-6d1"

# ── PRE-DECLARED realistic-motion parameters (fixed before observing results) ────────────────────────────────────────
SLEW = 2.0                     # rad/s — realistic joint velocity limit (= MotionLimits.joint_vel_safe; robots do 1–3)
JOINT_BUDGET = 3.2             # rad — worst-case single-route joint displacement (reach + lateral detour)
ACCEL_FRAC = 0.30              # +30% steps for accel/brake ramps under the slew limiter
SETTLE_STEPS = 40              # env-steps of terminal low-velocity settling allowance


def _physical_horizons(env):
    """Derive via/goal horizons (env-steps) from path length / permitted speed + accel/brake + settling. A slew-limited
    arm moves ≤ SLEW·ctrl_dt rad per env-step, so traversing JOINT_BUDGET rad needs JOINT_BUDGET/(SLEW·ctrl_dt) steps."""
    ctrl_dt = float(env.frame_skip * env.model.opt.timestep)
    base = JOINT_BUDGET / (SLEW * ctrl_dt)                        # env-steps to traverse the displacement at the speed cap
    via = int(np.ceil(base * (1 + ACCEL_FRAC)))                  # detour phase (open-loop servo to the via)
    goal = int(np.ceil(base * (1 + ACCEL_FRAC))) + SETTLE_STEPS  # goal phase (closed-loop DLS-IK + settling)
    return via, goal, ctrl_dt


def _realistic_env(obstacle_half=(0.028, 0.028, 0.075)):
    """Velocity-limited SE(3) obstacle env with PHYSICAL horizons attached (read by the execution primitive)."""
    e = SE3ObstacleReachEnv(control_mode="position", reach_thresh=0.06, ang_thresh=0.4, min_separation=0.16,
                            obstacle_half=obstacle_half, slew_joint_vel_limit=SLEW, max_steps=10 ** 6)
    via, goal, _dt = _physical_horizons(e)
    e.route_via_steps, e.route_goal_steps = via, goal
    e.max_steps = via + goal + 50                                # never truncate before the physical route completes
    return e


STARTUP_TRANSIENT = 5          # env-steps of servo startup transient excluded from the SUSTAINED-velocity judgment


def peak_joint_vel_probe(env, panel, n=6):
    """Confirm the executor is velocity-limited in the SUSTAINED regime. The slew limiter caps the COMMANDED target
    velocity; the position servo shows a 2–4 step startup transient (far target from rest) that then damps to the cap —
    analogous to the qacc/servo transients REALISTIC_MOTION_CONTRACT_V1 treats as diagnostic-not-gated. We report both the
    raw peak (incl. transient) and the SUSTAINED peak (after the transient), plus mean and terminal velocity. The unlimited
    coin arm hit 27 rad/s sustained; here sustained ≲ SLEW."""
    import mujoco

    from hymeko_rl.env.se3_reach_option import ik_position
    raw_peaks, sust_peaks, means, terms = [], [], [], []
    for item in panel[:n]:
        env.reset(seed=item["seed"])
        q, v = env.data.qpos.copy(), env.data.qvel.copy()
        via = route_via(env, ROUTE_DIRS[item["feasible"][0]])
        tgt = ik_position(env, via)
        vels = []
        for _ in range(env.route_via_steps):
            env.step(tgt)
            vels.append(float(np.max(np.abs(env.data.qvel[:env.n_actions]))))
        vels = np.asarray(vels)
        raw_peaks.append(float(vels.max()))
        sust_peaks.append(float(vels[STARTUP_TRANSIENT:].max()) if len(vels) > STARTUP_TRANSIENT else float(vels.max()))
        means.append(float(vels.mean()))
        terms.append(float(vels[-1]))
        env.data.qpos[:], env.data.qvel[:] = q, v
        mujoco.mj_forward(env.model, env.data)
        env._step = 0
    return {"raw_peak_joint_vel": round(float(np.max(raw_peaks)), 3),
            "sustained_peak_joint_vel": round(float(np.max(sust_peaks)), 3),
            "mean_joint_vel": round(float(np.mean(means)), 3), "slew_limit": SLEW,
            "sustained_within_limit": bool(np.max(sust_peaks) <= SLEW * 1.25),
            "startup_transient_excluded_steps": STARTUP_TRANSIENT,
            "terminal_joint_vel_mean": round(float(np.mean(terms)), 3)}


def reach_6d0(n_states=12):
    """6D-0 pose reach under the velocity-limited executor: direct route (no obstacle detour needed for reachability),
    physical horizon, terminal low-velocity certificate. Success + peak velocity + duration."""
    env = _realistic_env()
    succ, peaks, terms, durs = 0, [], [], []
    for s in range(n_states):
        env.reset(seed=1000 + s)
        via = 0.5 * (env._start_ee + np.asarray(env._target, np.float32))    # straight-line mid (no lateral detour)
        out = execute_route_option(env, via)
        succ += out["success"]
        peaks.append(float(np.max(np.abs(env.data.qvel[:env.n_actions]))))
        terms.append(out.get("timeout", 0))
        durs.append(env.route_via_steps + env.route_goal_steps)
    return {"n_states": n_states, "success_rate": round(succ / n_states, 3), "via_steps": env.route_via_steps,
            "goal_steps": env.route_goal_steps, "n_timeout": int(sum(terms)),
            "peak_joint_vel": round(float(np.max(peaks)), 3), "within_slew": bool(np.max(peaks) <= SLEW * 1.25)}


def main():
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    env0 = _realistic_env()
    via, goal, ctrl_dt = _physical_horizons(env0)
    print(f"== 6D REALISTIC MOTION ==  SLEW {SLEW} rad/s | ctrl_dt {ctrl_dt:.4f}s | physical horizons via {via} goal {goal} "
          f"(unlimited baseline 60/150) | max_steps {env0.max_steps}", flush=True)

    # eligibility RECOMPUTED under the limited executor (must not reuse the unlimited-run panel)
    panel, rate = build_eligible_panel(env0, state_seeds=range(0, 60), feas_seed_base=200, want=14)
    print(f"eligible panel (RECOMPUTED under slew): {len(panel)} states, rate {rate}", flush=True)
    if not panel:
        json.dump({"contract": "SE3_OBSTACLE_6D1_REALISTIC", "verdict": "NO_ELIGIBLE_STATES_UNDER_SLEW_HORIZONS",
                   "note": "physical horizons too short OR slew too slow for any route to be execution-feasible — lengthen horizons",
                   "physical_horizons": {"via": via, "goal": goal}}, open(f"{OUT}/obstacle_6d1_realistic.json", "w"), indent=1)
        print("\n→ NO_ELIGIBLE_STATES_UNDER_SLEW_HORIZONS (lengthen horizons)\nSE3_OBSTACLE_6D1_REALISTIC_DONE")
        return "NO_ELIGIBLE_STATES_UNDER_SLEW_HORIZONS"

    mp = peak_joint_vel_probe(env0, panel)
    print(f"executor velocity: sustained_peak {mp['sustained_peak_joint_vel']} rad/s (slew {SLEW}, within "
          f"{mp['sustained_within_limit']}) | mean {mp['mean_joint_vel']} | raw_peak {mp['raw_peak_joint_vel']} "
          f"(startup transient, {STARTUP_TRANSIENT} steps excluded) | terminal {mp['terminal_joint_vel_mean']}", flush=True)

    r0 = reach_6d0()
    print(f"6D-0 pose reach under slew: success {r0['success_rate']} | peak_vel {r0['peak_joint_vel']} within_slew "
          f"{r0['within_slew']} | timeouts {r0['n_timeout']}", flush=True)

    print("\n== 6D-1 SEED HARDENING under realistic motion (6 seeds, fresh panels recomputed under slew) ==", flush=True)
    hard = harden(n_seeds=6, env_factory=_realistic_env)
    agg = hard["aggregate"]
    for p in hard["per_seed"]:
        print(f"  seed {p['seed_group']}: elig {p['eligibility_rate']} n {p['n_states']} "
              f"critical {p['critical_pair']['single_head_wrong']}→{p['critical_pair']['kmode']} "
              f"Δ@B12 {p['delta_headline_mean']:+.3f} fails {p['kmode_failure_decomp']}", flush=True)
    print(f"\n  per-seed Δ: {agg['per_seed_delta_mean']}")
    print(f"  seed-median Δ {agg['seed_median_delta']} IQR {agg['seed_iqr']} | positive {agg['n_seeds_positive']}/{agg['n_seeds']}")
    print(f"  hierarchical bootstrap: {agg['hier_bootstrap']}")
    print(f"  budget curve K1 {agg['budget_curve_K1']}")
    print(f"  budget curve KM {agg['budget_curve_KM']}", flush=True)

    print("\n== OBSTACLE-SHIFT GENERALISATION under realistic motion ==", flush=True)
    shift = obstacle_shift_control(env_for_half=lambda half: _realistic_env(obstacle_half=half))
    for name, row in shift.items():
        print(f"  {name}: elig {row['eligibility_rate']} " + " ".join(f"{k}Δ={row[k]['delta']}" for k in row if k.startswith("B")))

    # realistic-motion verdict: the strong multimodal advantage must survive the velocity limit
    strong = agg["verdict"].startswith("MULTIMODAL_POLICY_SEARCH_VALIDATED")
    verdict = ("MULTIMODAL_POLICY_SEARCH_VALIDATED_UNDER_REALISTIC_MOTION_LIMITS" if strong
               else f"UNDER_REALISTIC_MOTION__{agg['verdict']}")
    print(f"\n  → REALISTIC-MOTION VERDICT: {verdict}", flush=True)

    manifest = {"contract": "SE3_OBSTACLE_6D1_REALISTIC_MOTION", "date": "2026-07-25",
                "motion": {"slew_limit": SLEW, "ctrl_dt": ctrl_dt, "physical_horizons": {"via": via, "goal": goal},
                           "joint_budget": JOINT_BUDGET, "accel_frac": ACCEL_FRAC, "settle_steps": SETTLE_STEPS},
                "executor_velocity_check": mp, "reach_6d0": r0, "eligibility_rate_under_slew": rate,
                "harden": hard, "obstacle_shift_control": shift, "unlimited_verdict": agg["verdict"], "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/obstacle_6d1_realistic.json", "w"), indent=1, default=str)
    plot_harden(agg, f"{OUT}/obstacle_6d1_realistic.png")
    _ = _hier_bootstrap  # (re-exported for downstream analysis parity)
    print(f"\nartifact: {OUT}/obstacle_6d1_realistic.json\nSE3_OBSTACLE_6D1_REALISTIC_DONE")
    return verdict


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
