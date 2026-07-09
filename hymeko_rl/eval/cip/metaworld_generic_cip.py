"""Generic MetaWorld CIP runner — sweep CIP over ANY MetaWorld task via its native ``info`` signals.

The coffee-push real-env path (``metaworld_cip.run_metaworld_cip_real``) hard-codes an obs → ``CoffeePushMonitor``
schema mapping, so it is one task only. This runner is **task-agnostic**: every MetaWorld V3 env emits the same
``info`` dict (``success``, ``near_object``, ``grasp_success``, ``in_place_reward``, ``obj_to_target``,
``unscaled_reward``), so those signals are a ready-made CIP variable set for all 48 MT50 tasks. Roll a scripted
policy + per-episode action noise (the observed exogenous input), build a :class:`RolloutFrame` from the info
signals, run DirectLiNGAM, and declare + cross-view-verify the discovered DAG — exactly the coffee-push pipeline,
minus the per-task monitor.

**Scope / honesty:** the task-truth here is **MetaWorld's own** ``success`` / ``in_place_reward`` (not a HyMeKo
runtime monitor — those exist only for coffee-push / dial-turn templates). The discovered causal DAG is still
declared in ``.hymeko`` and engine-verified. So HyMeKo remains the source of truth for the *causal model's
representation*; the *task variables* here are MetaWorld-native (the trade for covering all 48 tasks). No training;
read-only scripted rollouts. All DAGs are PROPOSED; MetaWorld's env randomisation is seed-uncontrolled, so a single
run is a point estimate (multi-seed tightening is a follow-up).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# task → scripted V3 policy (all present in metaworld.policies; one representative per task family).
GENERIC_TASKS: dict[str, str] = {
    "push": "SawyerPushV3Policy",              # push family (no grasp)
    "pick-place": "SawyerPickPlaceV3Policy",   # grasp family (grasp_success fires)
    "door-open": "SawyerDoorOpenV3Policy",     # articulated / hinge
    "button-press": "SawyerButtonPressV3Policy",  # contact → press
    "reach": "SawyerReachV3Policy",            # approach only (no contact term)
}
# Continuous CIP variables read from MetaWorld info; action_noise is the observed exogenous input, total_reward the
# sink under test. in_place_reward is used only for the reward↔task disagreement (kept out of the frame — it is
# near-collinear with obj_to_target_delta).
_GENERIC_CONTINUOUS = ("action_noise", "near_fraction", "grasp_fraction", "obj_to_target_delta", "total_reward")


def _generic_rollout(task: str, policy_name: str, seed: int, noise: float, max_steps: int = 180,
                     ) -> "tuple[dict[str, float], int, float]":
    """Roll one episode of ``task`` (scripted policy + action noise) → (continuous CIP vars, mw_success, in_place)."""
    import warnings

    import metaworld.policies as mp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]  # no stubs
        env: Any = ENVS[f"{task}-v3-goal-observable"](render_mode=None)
        policy = getattr(mp, policy_name)()
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        near = grasp = steps = success = 0
        total = 0.0
        d0: "float | None" = None
        d_final = in_place = 0.0
        for _ in range(max_steps):
            act = np.clip(np.asarray(policy.get_action(obs), np.float32)
                          + rng.normal(0, noise, 4).astype(np.float32), -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(act)
            steps += 1
            total += float(reward)
            near += int(info.get("near_object", 0))
            grasp += int(info.get("grasp_success", 0))
            success = max(success, int(info.get("success", 0)))
            d_final = float(info.get("obj_to_target", 0.0))
            in_place = float(info.get("in_place_reward", 0.0))
            if d0 is None:
                d0 = d_final
            if terminated or truncated:
                break
    cont = {"action_noise": float(noise), "near_fraction": near / max(1, steps),
            "grasp_fraction": grasp / max(1, steps),
            "obj_to_target_delta": float((d0 or 0.0) - d_final), "total_reward": total}
    return cont, success, in_place


def _disagreement(total_reward: "list[float]", task_score: "list[float]", task: str) -> float:
    """Reward↔task disagreement (``1 − concordance``) from the RewardConsistencyMonitor over (reward, in_place)."""
    from hymeko_rl.eval.task_monitor.consistency import RewardConsistencyMonitor
    rows = [{"policy": f"{task}_{i}", "total_reward": total_reward[i], "monitor_score": task_score[i]}
            for i in range(len(total_reward))]
    return round(1.0 - float(RewardConsistencyMonitor().check_reward_alignment(rows).score), 6)


def run_generic_cip(task: str, n: int, seed: int, out_dir: Path, *, noise_max: float = 0.7,
                    policy_name: "str | None" = None) -> "dict[str, Any]":
    """Run CIP over ``n`` scripted rollouts of a MetaWorld ``task`` (info-signal variables) → DAG + cross-view."""
    from hymeko_rl.eval.causal import CausalDiagnosis, RolloutFrame
    from hymeko_rl.eval.cip.metaworld_cip import _fit_declare_render

    policy_name = policy_name or GENERIC_TASKS[task]
    rng = np.random.default_rng(seed)
    noises = rng.uniform(0.0, noise_max, n)
    cols: dict[str, list[float]] = {name: [] for name in _GENERIC_CONTINUOUS}
    successes: list[int] = []
    in_place: list[float] = []
    for i in range(n):
        cont, succ, ip = _generic_rollout(task, policy_name, seed + i, float(noises[i]))
        for name in _GENERIC_CONTINUOUS:
            cols[name].append(cont[name])
        successes.append(succ)
        in_place.append(ip)
        print(f"[cip-mw-gen] {task:12s} ep {i + 1:3d}/{n} | noise={noises[i]:.2f} success={succ} "
              f"grasp={cont['grasp_fraction']:.2f} d_delta={cont['obj_to_target_delta']:+.3f} "
              f"reward={cont['total_reward']:.0f}", flush=True)

    disagreement = _disagreement(cols["total_reward"], in_place, task)
    continuous = {name: np.asarray(cols[name], dtype=np.float64) for name in _GENERIC_CONTINUOUS}
    continuous["reward_progress_disagreement"] = np.full(n, disagreement, dtype=np.float64)
    frame = RolloutFrame(
        continuous=continuous,
        categorical={"task": [task] * n, "mw_success": [bool(s) for s in successes]},
        missing={"reward_progress_disagreement": np.zeros(n, dtype=bool)}, n=n)
    report = CausalDiagnosis().run(frame)
    _matrix, kept, dropped = frame.continuous_matrix()
    summary: dict[str, Any] = {
        "task": task, "source": "generic MetaWorld info-signals (scripted policy + action noise)", "n": n,
        "seed": seed, "policy": policy_name, "mw_success_rate": float(np.mean(successes)),
        "reward_monitor_disagreement": disagreement, "continuous_kept": kept, "continuous_dropped": dropped,
        "diagnosis": report.as_dict(),
        "_disclaimer": "PROPOSED; task-truth is MetaWorld-native (not a HyMeKo monitor); single run is a point "
                       "estimate (env randomisation is seed-uncontrolled). Cross-view proves representation only.",
    }
    tag = f"{task.replace('-', '_')}_generic"     # hyphens are invalid in .hymeko identifiers / graph names
    _fit_declare_render(frame, tag, out_dir, summary, note=f"real MetaWorld {task} (info-signals)")
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[cip-mw-gen] {task}: success={summary['mw_success_rate']:.2f} "
          f"disagreement={disagreement:.3f} -> {out_dir}", flush=True)
    return summary


def _reward_parents(summary: "dict[str, Any]") -> "list[list[Any]]":
    """The discovered edges whose effect is total_reward (its causal parents), from a summary's strongest_edges."""
    return [e for e in summary.get("strongest_edges", []) if e[1] == "total_reward"]


def run_generic_sweep(tasks: "list[str]", n: int, seed: int, out_dir: Path) -> "dict[str, Any]":
    """Run :func:`run_generic_cip` over several tasks and write a cross-task comparison of where total_reward sits."""
    per_task: dict[str, Any] = {}
    for task in tasks:
        s = run_generic_cip(task, n, seed, out_dir)
        per_task[task] = {
            "causal_order": s.get("causal_order"), "reward_parents": _reward_parents(s),
            "reward_monitor_disagreement": s["reward_monitor_disagreement"],
            "mw_success_rate": s["mw_success_rate"],
            "cross_view_agree": s.get("cross_view", {}).get("agree"),
            "canonical_hash": s.get("cross_view", {}).get("canonical_hash"),
            "continuous_dropped": s["continuous_dropped"]}
    comparison = {"kind": "generic MetaWorld CIP sweep", "tasks": tasks, "n_per_task": n, "seed": seed,
                  "per_task": per_task,
                  "cross_view_all_pass": all(v["cross_view_agree"] for v in per_task.values())}
    (out_dir / "sweep_comparison.json").write_text(json.dumps(comparison, indent=2))
    print(f"[cip-mw-gen] sweep done: {tasks} | cross_view_all_pass={comparison['cross_view_all_pass']} -> {out_dir}",
          flush=True)
    return comparison


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    from hymeko_rl.eval.evaluate import experiment_dir
    parser = argparse.ArgumentParser(description="Generic MetaWorld CIP (info-signals) over any task / a sweep")
    parser.add_argument("--task", choices=[*GENERIC_TASKS, "sweep"], default="sweep")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="reports/figures")
    args = parser.parse_args(argv)
    out_dir = experiment_dir(args.out, "cip_metaworld_generic")
    if args.task == "sweep":
        run_generic_sweep(list(GENERIC_TASKS), int(args.n), int(args.seed), out_dir)
    else:
        run_generic_cip(args.task, int(args.n), int(args.seed), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
