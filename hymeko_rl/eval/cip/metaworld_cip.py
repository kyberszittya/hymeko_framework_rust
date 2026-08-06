"""CIP over the MetaWorld task templates (coffee-push, dial-turn) — the Ito+Kato scenario, template level.

The Ito+Kato task (`docs/task/20260702_task_ito_kato`) asks for CIP over MetaWorld scenarios. The real MetaWorld
env is **not installed** and the monitor (`task_monitor/metaworld.py`) is deliberately synthetic-driven (a real-env
wrapper is out of scope there). So this runs the same pipeline the coin PoC ran — **monitor → RolloutFrame →
DirectLiNGAM → `.hymeko` cross-view** — on synthetic trajectories generated to each task's monitor *story*:

* **coffee-push**: ``approach → contact → object moves toward target → target reached``;
* **dial-turn**: ``engage → rotate toward target angle → reach angle (without overshoot)``.

Each trajectory is drawn from a latent *skill* that plausibly drives the chain (higher skill → lower approach
error → more contact → more progress → success). The **monitor** (unchanged, read-only) extracts the continuous
outcomes; DirectLiNGAM then discovers the structure among {skill inputs, monitor outputs, a shaped reward proxy}.
A MetaWorld-style dense reward proxy (reach + in-place + contact bonus) is included so the same reward↔progress
axis the coin PoC surfaced is visible here.

**Scope (honest label):** this is a **template-level method validation** — it demonstrates the monitor+CIP+HyMeKo
pipeline reproduces the coffee-push / dial-turn causal story and machine-verifies the DAG. It is **not** a claim
about a real MetaWorld policy; real-env rollouts require installing the ``metaworld`` package (a §1 dependency,
gated). The discovered DAG is PROPOSED; controlled ablation decides — unchanged doctrine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class _Rollout:
    """One synthetic rollout: its monitor verdict + the controlled/derived continuous scalars for the frame."""

    verdict: Any
    continuous: dict[str, float]


def _coffee_rollout(rng: np.random.Generator, steps: int = 24) -> _Rollout:
    """A coffee-push trajectory drawn from a latent skill → (verdict, {approach_error, contact_fraction, reward}).

    The monitor's ``progress_score`` is the manipulation OUTCOME (not re-added as a column — that would duplicate
    ``object_target_distance_delta`` and muddy the causal order)."""
    from hymeko_rl.eval.task_monitor import CoffeePushMonitor

    # Acyclic non-Gaussian SEM (no hidden confounder — causal sufficiency): approach_error is exogenous, contact
    # is caused by it, progress (the monitor's score) is caused by contact, reward is the downstream sink.
    approach_error = float(np.clip(abs(rng.laplace(0.10, 0.05)), 0.0, 0.35))            # exogenous root
    contact_fraction = float(np.clip(0.95 - 2.2 * approach_error + rng.laplace(0, 0.05), 0.0, 1.0))  # ← approach
    d0 = float(np.clip(0.30 + rng.laplace(0, 0.01), 0.24, 0.36))
    push = float(np.clip(contact_fraction * d0 + rng.laplace(0, 0.01), 0.0, d0))        # contact → push (progress)
    dists = d0 - push * np.clip(np.linspace(0.0, 1.0, steps) + rng.normal(0, 0.02, steps), 0.0, 1.0)
    traj = [{"object_xy": [float(d), 0.0], "target_xy": [0.0, 0.0], "contact": bool(rng.uniform() < contact_fraction)}
            for d in dists]
    verdict = CoffeePushMonitor(success_radius=0.05, min_progress=0.01).evaluate(traj)
    reward_proxy = float(-approach_error + 2.0 * verdict.object_target_distance_delta
                         + 1.5 * contact_fraction + rng.laplace(0, 0.05))               # downstream sink
    return _Rollout(verdict, {
        "approach_error": approach_error, "contact_fraction": contact_fraction, "reward_proxy": reward_proxy})


def _dial_rollout(rng: np.random.Generator, steps: int = 24) -> _Rollout:
    """A dial-turn trajectory drawn from a latent skill → (verdict, {engage_error, rotation_fraction, reward})."""
    from hymeko_rl.eval.task_monitor import DialTurnMonitor

    # Acyclic non-Gaussian SEM: engage_error exogenous → rotation → progress (monitor) → reward sink.
    engage_error = float(np.clip(abs(rng.laplace(0.18, 0.10)), 0.0, 0.8))               # exogenous root
    rotation_fraction = float(np.clip(0.98 - 1.4 * engage_error + rng.laplace(0, 0.05), 0.0, 1.0))  # ← engage
    a0 = float(np.clip(0.65 + rng.laplace(0, 0.03), 0.45, 0.85))
    turn = float(np.clip(rotation_fraction * a0 + rng.laplace(0, 0.01), 0.0, a0))       # rotation → progress
    angles = a0 - turn * np.clip(np.linspace(0.0, 1.0, steps) + rng.normal(0, 0.02, steps), 0.0, 1.0)
    traj = [{"dial_angle": float(a), "target_angle": 0.0,
             "contact": bool(rng.uniform() < rotation_fraction)} for a in angles]
    verdict = DialTurnMonitor().evaluate(traj)
    error_delta = float(verdict.target_error_initial - verdict.target_error_final)
    reward_proxy = float(-engage_error + 2.0 * error_delta + 1.5 * rotation_fraction + rng.laplace(0, 0.05))
    return _Rollout(verdict, {
        "engage_error": engage_error, "rotation_fraction": rotation_fraction, "reward_proxy": reward_proxy})


@dataclass(frozen=True)
class TaskTemplate:
    """A MetaWorld task template: its name, synthetic rollout generator, and the story chain (for the report)."""

    name: str
    rollout: Callable[[np.random.Generator], _Rollout]
    story: str


TEMPLATES: dict[str, TaskTemplate] = {
    "coffee_push": TaskTemplate("coffee_push", _coffee_rollout,
                                "approach_error → contact_fraction → progress_score (object toward target)"),
    "dial_turn": TaskTemplate("dial_turn", _dial_rollout,
                              "engage_error → rotation_fraction → progress_score (dial toward target angle)"),
}


def _fit_declare_render(frame: Any, task: str, out_dir: Path, summary: dict[str, Any],
                        note: str = "synthetic template") -> None:
    """Fit DirectLiNGAM, declare the DAG as ``.hymeko`` (cross-view verified), render it, and record into summary.

    A frame with too few varying continuous columns leaves ``summary`` without the causal fields (diagnosis only)."""
    from hymeko_rl.eval.causal import CausalHypergraph, DirectLiNGAM, cross_view_verify
    from hymeko_rl.experiments.cip_lingam_demo import render_dag

    matrix, kept, _dropped = frame.continuous_matrix()
    if len(kept) < 2 or matrix.shape[0] <= len(kept):
        return
    result = DirectLiNGAM().fit(matrix, kept)
    cg = CausalHypergraph.from_lingam(result, f"MetaWorld{task.title().replace('_', '')}")
    xview = cross_view_verify(cg, out_dir / f"causal_{task}.hymeko")
    render_dag(result.order, result.adjacency, kept, out_dir / f"dag_{task}.png",
               f"MetaWorld {task} causal ({note}, N={matrix.shape[0]}, PROPOSED)")
    summary["causal_order"] = result.ordered_names()
    # store ALL discovered edges (not just a top-k) so the multi-seed aggregation never misses a key edge
    summary["strongest_edges"] = [[c, e, round(w, 6)] for c, e, w in result.strongest_edges(64)]
    summary["cross_view"] = xview.as_dict()
    print(f"[cip-mw] {task}: order={result.ordered_names()} | cross-view agree={xview.agree} "
          f"hash={xview.canonical_hash[:24]}", flush=True)


def run_metaworld_cip(task: str, n: int, seed: int, out_dir: Path) -> "dict[str, Any]":
    """Run the CIP pipeline over ``n`` synthetic ``task`` rollouts → diagnosis + DAG + ``.hymeko`` cross-view.

    # Preconditions ``task in TEMPLATES``; ``n`` large enough for DirectLiNGAM (``n > n_continuous_vars``).
    # Postconditions writes ``<task>_summary.json`` + a DAG png + a ``.hymeko`` into ``out_dir``.
    """
    from hymeko_rl.eval.causal import CausalDiagnosis, RolloutFrame

    if task not in TEMPLATES:
        raise ValueError(f"unknown task {task!r}; known: {sorted(TEMPLATES)}")
    template = TEMPLATES[task]
    rng = np.random.default_rng(seed)
    rollouts = [template.rollout(rng) for _ in range(n)]
    extra_names = list(rollouts[0].continuous)
    pass_rate = float(np.mean([r.verdict.monitor_pass for r in rollouts]))
    print(f"[cip-mw] {task}: {n} synthetic rollouts | pass_rate={pass_rate:.2f} | vars={extra_names}", flush=True)

    maps = [r.verdict.as_dict() for r in rollouts]
    extra = {name: [r.continuous[name] for r in rollouts] for name in extra_names}
    frame = RolloutFrame.from_verdicts(maps, extra_continuous=extra, categoricals={"task": [task] * n})
    report = CausalDiagnosis().run(frame)
    _matrix, kept, dropped = frame.continuous_matrix()

    summary: dict[str, Any] = {
        "task": task, "scope": "template-level method validation (synthetic; NOT a real MetaWorld policy)",
        "story": template.story, "n": n, "seed": seed, "monitor_pass_rate": pass_rate,
        "continuous_kept": kept, "continuous_dropped": dropped, "diagnosis": report.as_dict(),
        "_disclaimer": "PROPOSED structure over synthetic template rollouts; the .hymeko cross-view proves "
                       "declared ≡ engine tensor view, not causal truth. Real-env rollouts need the metaworld dep.",
    }
    _fit_declare_render(frame, task, out_dir, summary)
    (out_dir / f"{task}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[cip-mw] {task}: wrote {task}_summary.json + DAG/.hymeko to {out_dir}", flush=True)
    return summary


# ── real-env path (requires the `metaworld` package; coffee-push maps cleanly obs→monitor schema) ─────────────
# coffee-push is a PUSH task (grasp never fires), so `near_object` is the contact/approach proxy; the scripted
# V3 policy is perturbed with per-episode action noise = the OBSERVED exogenous input (no hidden confounder).
_REAL_TASKS: dict[str, tuple[str, str]] = {
    "coffee_push": ("coffee-push-v3-goal-observable", "SawyerCoffeePushV3Policy"),
}


def _real_rollout(env_key: str, policy_name: str, seed: int, noise: float, max_steps: int = 200,
                  ) -> "tuple[_Rollout, int]":
    """One real MetaWorld coffee-push episode (scripted policy + action noise) → (monitor rollout, mw_success)."""
    import warnings

    import metaworld.policies as mp
    from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]  # no py.typed

    from hymeko_rl.eval.task_monitor import CoffeePushMonitor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = ENVS[env_key](render_mode=None)
        policy = getattr(mp, policy_name)()
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        traj: list[dict[str, Any]] = []
        total = 0.0
        near = success = steps = 0
        for _ in range(max_steps):
            act = np.clip(np.asarray(policy.get_action(obs), np.float32)
                          + rng.normal(0, noise, 4).astype(np.float32), -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(act)
            total += float(reward)
            steps += 1
            near += int(info.get("near_object", 0))
            success = max(success, int(info.get("success", 0)))
            traj.append({"object_xy": [float(obs[4]), float(obs[5])], "target_xy": [float(obs[-3]), float(obs[-2])],
                         "gripper_xy": [float(obs[0]), float(obs[1])], "contact": bool(info.get("near_object", 0))})
            if terminated or truncated:
                break
    verdict = CoffeePushMonitor(success_radius=0.05, min_progress=0.01).evaluate(traj)
    rollout = _Rollout(verdict, {"action_noise": float(noise), "near_fraction": near / max(1, steps),
                                 "total_reward": total})
    return rollout, success


def _roll_real_batch(task: str, n: int, seed: int, noise_max: float,
                     ) -> "tuple[list[_Rollout], list[int], np.ndarray]":
    """Roll ``n`` real MetaWorld episodes (scripted policy + per-episode action noise). Returns
    ``(rollouts, mw_successes, noises)``. # Preconditions ``task in _REAL_TASKS``; ``metaworld`` importable."""
    if task not in _REAL_TASKS:
        raise ValueError(f"no real-env mapping for {task!r}; available: {sorted(_REAL_TASKS)}")
    env_key, policy_name = _REAL_TASKS[task]
    rng = np.random.default_rng(seed)
    noises = rng.uniform(0.0, noise_max, n)
    rollouts: list[_Rollout] = []
    successes: list[int] = []
    for i in range(n):
        ro, succ = _real_rollout(env_key, policy_name, seed + i, float(noises[i]))
        rollouts.append(ro)
        successes.append(succ)
        print(f"[cip-mw-real] {task} ep {i + 1:3d}/{n} | noise={noises[i]:.2f} pass={int(ro.verdict.monitor_pass)} "
              f"prog={ro.verdict.progress_score:+.3f} reward={ro.continuous['total_reward']:.0f} "
              f"mw_success={succ}", flush=True)
    return rollouts, successes, noises


def _reward_monitor_disagreement(rollouts: "list[_Rollout]", task: str) -> float:
    """Reward↔monitor disagreement (``1 − concordance``) over the batch, from the RewardConsistencyMonitor (§6.1)."""
    from hymeko_rl.eval.task_monitor.consistency import RewardConsistencyMonitor
    rows = [{"policy": f"{task}_{i}", "total_reward": r.continuous["total_reward"],
             "monitor_score": float(r.verdict.monitor_score)} for i, r in enumerate(rollouts)]
    return round(1.0 - float(RewardConsistencyMonitor().check_reward_alignment(rows).score), 6)


def run_metaworld_cip_real(task: str, n: int, seed: int, out_dir: Path, noise_max: float = 0.7) -> "dict[str, Any]":
    """Real-env CIP: roll a scripted MetaWorld policy (+per-episode action noise) → monitor → DAG + cross-view.

    The Phase-2 analog for MetaWorld: real physics, real dense reward, the reward-independent monitor. Action
    noise is the OBSERVED exogenous input (no hidden confounder). # Preconditions ``task in _REAL_TASKS``;
    the ``metaworld`` package is importable.
    """
    from hymeko_rl.eval.causal import CausalDiagnosis, RolloutFrame

    rollouts, successes, _noises = _roll_real_batch(task, n, seed, noise_max)
    extra_names = list(rollouts[0].continuous)
    maps = [r.verdict.as_dict() for r in rollouts]
    extra = {name: [r.continuous[name] for r in rollouts] for name in extra_names}
    frame = RolloutFrame.from_verdicts(maps, extra_continuous=extra,
                                       categoricals={"task": [task] * n, "mw_success": [bool(s) for s in successes]})
    report = CausalDiagnosis().run(frame)
    _matrix, kept, dropped = frame.continuous_matrix()
    summary: dict[str, Any] = {
        "task": task, "source": "real-env (metaworld scripted policy + action noise)",
        "scope": "real MetaWorld rollouts through the HyMeKo monitor — the Phase-2 analog", "n": n, "seed": seed,
        "monitor_pass_rate": float(np.mean([r.verdict.monitor_pass for r in rollouts])),
        "mw_success_rate": float(np.mean(successes)),
        "reward_monitor_disagreement": _reward_monitor_disagreement(rollouts, task),
        "continuous_kept": kept, "continuous_dropped": dropped, "diagnosis": report.as_dict(),
        "_disclaimer": "PROPOSED structure over real rollouts; controlled ablation decides. Cross-view proves "
                       "declared ≡ engine tensor view, not causal truth.",
    }
    _fit_declare_render(frame, f"{task}_real", out_dir, summary, note="real MetaWorld rollouts")
    (out_dir / f"{task}_real_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[cip-mw-real] {task}: pass_rate={summary['monitor_pass_rate']:.2f} "
          f"mw_success={summary['mw_success_rate']:.2f} disagreement={summary['reward_monitor_disagreement']:.3f} "
          f"-> wrote to {out_dir}", flush=True)
    return summary


# Unordered key variable pairs whose edge (either direction) the multi-seed aggregation tracks (coffee-push vars).
_KEY_PAIRS: tuple[tuple[str, str], ...] = (
    ("near_fraction", "total_reward"),
    ("progress_score", "total_reward"),
    ("action_noise", "total_reward"),
    ("progress_score", "near_fraction"),
    ("action_noise", "near_fraction"),
    ("action_noise", "progress_score"),
)


def _edge_in(edges: "list[list[Any]]", a: str, b: str) -> "tuple[float, str] | tuple[None, None]":
    """The signed weight + direction (``"a->b"``) of the edge between ``a`` and ``b`` in ``edges``, else (None, None)."""
    for cause, effect, weight in edges:
        if {cause, effect} == {a, b} and float(weight) != 0.0:
            return float(weight), f"{cause}->{effect}"
    return None, None


def _median_iqr(values: "list[float]") -> "dict[str, Any]":
    arr = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(arr)), "iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
            "n": int(arr.size)}


def _collect_edge(summaries: "list[dict[str, Any]]", a: str, b: str) -> "tuple[list[float], list[str]]":
    """Collect the (weight, direction) of the ``a``–``b`` edge across each batch that has it."""
    weights: list[float] = []
    directions: list[str] = []
    for s in summaries:
        w, d = _edge_in(s.get("strongest_edges", []), a, b)
        if w is not None and d is not None:
            weights.append(w)
            directions.append(d)
    return weights, directions


def _aggregate_edge(summaries: "list[dict[str, Any]]", a: str, b: str, min_presence: float) -> "dict[str, Any]":
    """Aggregate one key edge across batch summaries → presence, median/IQR weight, direction split, stability."""
    weights, directions = _collect_edge(summaries, a, b)
    presence = len(weights) / max(1, len(summaries))
    if not weights:
        return {"presence": 0.0, "n_present": 0, "stable": False}
    arr = np.asarray(weights)
    lo, hi = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr_excludes_zero = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    dir_counts = {d: directions.count(d) for d in set(directions)}
    return {"presence": round(presence, 3), "n_present": len(weights),
            "median_weight": round(float(np.median(arr)), 4), "iqr": [round(lo, 4), round(hi, 4)],
            "directions": dir_counts, "dominant_direction": max(dir_counts, key=lambda k: dir_counts[k]),
            "sign_consistent": bool(np.all(arr > 0) or np.all(arr < 0)),
            "stable": bool(presence >= min_presence and iqr_excludes_zero)}


def _aggregate_batches(summaries: "list[dict[str, Any]]", key_pairs: "tuple[tuple[str, str], ...]" = _KEY_PAIRS,
                       min_presence: float = 0.6) -> "dict[str, Any]":
    """Aggregate per-batch summaries → edge stability + median/IQR of pass-rate & disagreement + cross-view-all-pass.

    A key edge is ``stable`` iff it appears in ``>= min_presence`` of batches AND its weight IQR does not cross zero.
    """
    edges = {f"{a}--{b}": _aggregate_edge(summaries, a, b, min_presence) for a, b in key_pairs}
    cross_view_all = all(s.get("cross_view", {}).get("agree", False) for s in summaries) and bool(summaries)
    return {
        "n_batches": len(summaries),
        "edges": edges,
        "monitor_pass_rate": _median_iqr([s["monitor_pass_rate"] for s in summaries if "monitor_pass_rate" in s]),
        "reward_monitor_disagreement": _median_iqr(
            [s["reward_monitor_disagreement"] for s in summaries if "reward_monitor_disagreement" in s]),
        "cross_view_all_pass": cross_view_all,
        "stable_edges": [k for k, v in edges.items() if v.get("stable")],
    }


def run_metaworld_multiseed(task: str, batches: int, n: int, seed0: int, out_dir: Path, noise_max: float = 0.7,
                            ) -> "dict[str, Any]":
    """Multi-seed real-env aggregation: run ``batches`` independent coffee-push batches, aggregate edge stability.

    Each batch is an independent real-env run (its own seed) written to ``out_dir/batch_<b>/`` (prior single-run
    artifacts are never overwritten). # Preconditions ``task in _REAL_TASKS``; ``metaworld`` importable.
    """
    if task not in _REAL_TASKS:
        raise ValueError(f"no real-env mapping for {task!r}; available: {sorted(_REAL_TASKS)}")
    batch_summaries: list[dict[str, Any]] = []
    for b in range(batches):
        bdir = out_dir / f"batch_{b}"
        bdir.mkdir(parents=True, exist_ok=True)
        print(f"[cip-mw-multiseed] === batch {b + 1}/{batches} (seed {seed0 + b * 1000}) ===", flush=True)
        batch_summaries.append(run_metaworld_cip_real(task, n, seed0 + b * 1000, bdir, noise_max=noise_max))

    aggregate = _aggregate_batches(batch_summaries)
    out = {
        "task": task, "kind": "multi-seed real-env aggregation (NO training)", "batches": batches, "n_per_batch": n,
        "seed0": seed0, "noise_max": noise_max,
        "per_batch": [{"batch": b, "seed": seed0 + b * 1000, "monitor_pass_rate": s.get("monitor_pass_rate"),
                       "reward_monitor_disagreement": s.get("reward_monitor_disagreement"),
                       "cross_view_agree": s.get("cross_view", {}).get("agree"),
                       "causal_order": s.get("causal_order"),
                       "edges": {f"{a}--{c}": _edge_in(s.get("strongest_edges", []), a, c)[0] for a, c in _KEY_PAIRS}}
                      for b, s in enumerate(batch_summaries)],
        "aggregate": aggregate,
        "_disclaimer": "PROPOSED; a key edge is called stable only if it recurs across batches with a sign-consistent "
                       "IQR that excludes zero AND cross-view passes for every batch. MetaWorld env randomisation is "
                       "seed-uncontrolled, so batches are the replication unit.",
    }
    (out_dir / f"{task}_multiseed_summary.json").write_text(json.dumps(out, indent=2))
    print(f"[cip-mw-multiseed] {task}: {batches}x{n} | stable edges={aggregate['stable_edges']} | "
          f"cross_view_all_pass={aggregate['cross_view_all_pass']} -> {out_dir}", flush=True)
    return out


def _run_single(source: str, task: str, n: "int | None", seed: int, out_dir: Path) -> None:
    """Dispatch a single (non-multiseed) run: real metaworld rollouts or synthetic templates."""
    if source == "real":
        for t in (list(_REAL_TASKS) if task == "all" else [task]):
            run_metaworld_cip_real(t, n if n is not None else 80, seed, out_dir)
    else:
        for t in (list(TEMPLATES) if task == "all" else [task]):
            run_metaworld_cip(t, n if n is not None else 120, seed, out_dir)


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    from hymeko_rl.eval.evaluate import experiment_dir
    parser = argparse.ArgumentParser(description="CIP over MetaWorld tasks (coffee-push, dial-turn)")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic",
                        help="synthetic templates (no metaworld dep) or real metaworld rollouts")
    parser.add_argument("--task", choices=[*TEMPLATES, *_REAL_TASKS, "all"], default="all")
    parser.add_argument("--n", type=int, default=None, help="rollouts per task/batch (synthetic 120 / real 80)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multiseed", type=int, default=0,
                        help="number of independent real-env batches to aggregate (0 = single run)")
    parser.add_argument("--out", type=str, default="reports/figures")
    args = parser.parse_args(argv)
    if args.multiseed > 0:
        out_dir = experiment_dir(args.out, "cip_metaworld_multiseed")
        task = "coffee_push" if args.task == "all" else args.task
        run_metaworld_multiseed(task, args.multiseed, args.n if args.n is not None else 80, int(args.seed), out_dir)
        return 0
    _run_single(args.source, args.task, args.n, int(args.seed), experiment_dir(args.out, f"cip_metaworld_{args.source}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
