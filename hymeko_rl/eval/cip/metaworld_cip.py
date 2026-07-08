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


def _fit_declare_render(frame: Any, task: str, out_dir: Path, summary: dict[str, Any]) -> None:
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
               f"MetaWorld {task} causal (synthetic template, N={matrix.shape[0]}, PROPOSED)")
    summary["causal_order"] = result.ordered_names()
    summary["strongest_edges"] = [[c, e, round(w, 6)] for c, e, w in result.strongest_edges(6)]
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


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    from hymeko_rl.eval.evaluate import experiment_dir
    parser = argparse.ArgumentParser(description="CIP over MetaWorld task templates (coffee-push, dial-turn)")
    parser.add_argument("--task", choices=[*TEMPLATES, "all"], default="all")
    parser.add_argument("--n", type=int, default=120, help="synthetic rollouts per task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="reports/figures")
    args = parser.parse_args(argv)
    out_dir = experiment_dir(args.out, "cip_metaworld")
    tasks = list(TEMPLATES) if args.task == "all" else [args.task]
    for task in tasks:
        run_metaworld_cip(task, int(args.n), int(args.seed), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
