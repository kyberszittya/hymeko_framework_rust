"""A/B: per-node actor head vs pooled actor vs MLP — the RL payoff test for the multidim-readout finding.

The supervised probe (``reports/2026-06-26-multidim-readout.md``) showed a per-node head crushes
collapse-then-expand and beats a flat MLP on a per-node vector target. Here is the control-task test: the SAC
actor's ``μ`` is read **per joint** from its own message-passed node embedding + global context
(:class:`hymeko_rl.sac.PerNodeSquashedGaussianActor`) instead of mean-pool → ``Linear(feat, action_dim)``.
Same SAC, same HSiKAN backbone, params-matched MLP — only the actor readout differs.

Three configs: ``hsikan_pooled`` (the current actor), ``hsikan_pernode`` (the new head), ``mlp`` (flat
baseline, params-matched to the pooled HSiKAN actor). Reports DELIVERY (the comparable metric) + return + a
GIF + a scoreboard (§9). Reuses ``train_sac``/``evaluate``/``render_actor_gif``/``plot_scoreboard`` — no copied
loop, no re-implemented rendering (§6.1).

    python -m hymeko_rl.exp_pernode_actor_ab --task galambos --mode smoke   # proves plumbing + 3-form output
    python -m hymeko_rl.exp_pernode_actor_ab --task galambos --mode full    # the result (1 seed)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from hymeko_rl.campaign_viz import _greedy_action_fn, render_actor_gif
from hymeko_rl.env.arm_world import actuator_vertices
from hymeko_rl.evaluate import EvalStats, evaluate, experiment_dir, now_stamp, plot_scoreboard, results_to_csv
from hymeko_rl.offpolicy_eval import _env_dims, match_mlp_hidden
from hymeko_rl.reach_arch_compare import _iqr   # one IQR helper for the whole package (no re-impl)
from hymeko_rl.sac import SACConfig, build_sac, train_sac

_TASKS: dict[str, Callable[[], Any]] = {
    "galambos": lambda: __import__("hymeko_rl.env.planar_grasp_env", fromlist=["PlanarGraspEnv"])
    .PlanarGraspEnv.from_hymeko(max_steps=160, difficulty=0.3),
    "quadruped": lambda: __import__("hymeko_rl.env.quadruped_env", fromlist=["QuadrupedGoalEnv"])
    .QuadrupedGoalEnv(),
}

# (config name, backbone kind, actor_head, skip) — the contestants. ``skip="highway"`` engages the HSiKAN
# alpha-gate (T·H + (1−T)·carry, the Schmidhuber highway = the "H") so we test the readout fix AND the
# feature-collection gate the RL backbone otherwise leaves OFF (skip="none").
_CONFIGS: tuple[tuple[str, str, str, str], ...] = (
    ("hsikan_pooled", "hsikan", "pooled", "none"),
    ("hsikan_pernode", "hsikan", "per_node", "none"),
    ("hsikan_pernode_hw", "hsikan", "per_node", "highway"),
    ("sa_hsikan", "sa_hsikan", "pooled", "none"),     # the Bᴸ-collapse (launch-bound fix; 2.6x faster, 11x fewer params)
    ("mlp", "mlp", "pooled", "none"),
)


def _build_actor(name: str, kind: str, actor_head: str, skip: str, env: Any, cfg: SACConfig, *,
                 hidden: int) -> Any:
    nv, feat, action_dim, action_scale = _env_dims(env)
    kw: dict[str, Any] = {} if kind == "mlp" else {"hg_state": env.hg, "skip": skip}
    if actor_head == "per_node":
        kw["act_vertices"] = actuator_vertices(env.model)
    actor, critics = build_sac(kind, obs_dim=feat, flat_dim=nv * feat, action_dim=action_dim,
                               action_scale=action_scale, n_critics=cfg.n_critics, hidden=hidden,
                               actor_head=actor_head, **kw)
    return actor, critics


def run_config(name: str, kind: str, actor_head: str, skip: str, *, task: str,
               env_factory: Callable[[], Any], cfg: SACConfig, hidden: int, n_eval: int, out: Path,
               render_gif: bool = True) -> tuple[EvalStats, dict[str, Any]]:
    """Train one (config, seed) cell, evaluate delivery/return; render the GIF only for the representative
    seed (``render_gif``, to bound the count). Returns (EvalStats, result-dict incl. the seed)."""
    env = env_factory()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    actor, critics = _build_actor(name, kind, actor_head, skip, env, cfg, hidden=hidden)
    train_sac(actor, critics, env, cfg)
    stats = evaluate(env_factory(), _greedy_action_fn(actor), source=name, n_episodes=n_eval, seed0=30_000)
    if render_gif:
        try:
            render_actor_gif(env_factory(), actor, out / f"{task}_{name}")
        except Exception as exc:   # noqa: BLE001 — viz best-effort; a GL failure must not kill the A/B
            print(f"  [gif {name} skipped: {type(exc).__name__}: {exc}]", flush=True)
    result = dict(delivery=round(stats.success_rate, 3), mean_return=round(stats.mean_return, 2),
                  deaths=stats.deaths, skip=skip, seed=int(cfg.seed),
                  n_params=int(sum(p.numel() for p in actor.parameters())), n_eval=n_eval)
    print(f"  [{name}/s{cfg.seed}] {stats.summary()}  params={result['n_params']}", flush=True)
    return stats, result


def _config_worker(spec: "tuple[str, str, str, str, str, SACConfig, int, int, str, bool]",
                   ) -> "tuple[str, int, EvalStats, dict[str, Any]]":
    """Process-pool worker: train+evaluate ONE (config, seed) cell (rebuilds its own env from the task name).
    Pins one BLAS thread per worker so N workers spread across cores without oversubscription (CLAUDE.md §3 RL
    carve-out). Returns picklable ``(name, seed, EvalStats, result-dict)``."""
    name, kind, head, skip, task, cfg, hidden, n_eval, out_str, render_gif = spec
    torch.set_num_threads(1)
    stats, result = run_config(name, kind, head, skip, task=task, env_factory=_TASKS[task], cfg=cfg,
                               hidden=hidden, n_eval=n_eval, out=Path(out_str), render_gif=render_gif)
    return name, int(cfg.seed), stats, result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="galambos", choices=list(_TASKS))
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--steps", type=int, default=None, help="override total_steps (else 2k smoke / 20k full)")
    ap.add_argument("--hidden", type=int, default=64, help="HSiKAN width (MLP params-matched)")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds per config (median/IQR — the real verdict)")
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--configs", default=None,
                    help="comma-separated subset of config names to run (default: all four)")
    ap.add_argument("--out-dir", default="experiments",
                    help="parent dir; the run lands in <out-dir>/YYYY_MM_DD_HH_MM_pernode_<task>/")
    ap.add_argument("--workers", type=int, default=None,
                    help="max parallel worker processes (cap for page-file/commit safety; default cpu-1). "
                         "Each worker is a torch process reserving VAS; too many can exhaust the page file.")
    a = ap.parse_args(argv)

    steps = a.steps if a.steps is not None else (2_000 if a.mode == "smoke" else 20_000)
    selected = [c for c in _CONFIGS if a.configs is None or c[0] in a.configs.split(",")]
    mlp_hidden, hk_params, mk_params = match_mlp_hidden(a.task, a.hidden)
    print(f"  [params-match] {a.task}: hsikan@{a.hidden}={hk_params} ~ mlp@{mlp_hidden}={mk_params}", flush=True)
    hidden_of = {name: (mlp_hidden if kind == "mlp" else a.hidden) for name, kind, _, _ in selected}
    out = experiment_dir(a.out_dir, f"pernode_{a.task}")   # one timestamped folder for all this run's artifacts
    start = time.perf_counter()

    # cells = config × seed; independent → parallel processes (one BLAS thread each, CLAUDE.md §3). GIF only
    # for seed 0 (representative) to bound the count.
    cells = [(name, kind, head, skip, a.task,
              SACConfig(total_steps=steps, eval_every=steps, n_eval=5, seed=sd),
              hidden_of[name], a.n_eval, str(out), sd == 0)
             for name, kind, head, skip in selected for sd in range(a.seeds)]
    requested = a.workers if a.workers is not None else max(1, (os.cpu_count() or 2) - 1)
    workers = min(len(cells), max(1, requested))   # cap: each worker is a torch process committing VAS
    print(f"  [parallel] {len(selected)} configs × {a.seeds} seeds = {len(cells)} cells across {workers} "
          f"workers ({steps} steps each)", flush=True)

    by_config: dict[str, list[dict[str, Any]]] = {c[0]: [] for c in selected}
    stats_seed0: dict[str, EvalStats] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_config_worker, c): (c[0], c[5].seed) for c in cells}
        for fut in as_completed(futs):
            cname, csd = futs[fut]
            try:
                nm, sd, stats, result = fut.result()
                by_config[nm].append(result)
                if sd == 0:
                    stats_seed0[nm] = stats
            except Exception as exc:   # noqa: BLE001 — one cell failing must not lose the rest
                print(f"  [cell {cname}/s{csd} FAILED: {type(exc).__name__}: {exc}]", flush=True)

    # Aggregate per config: median/IQR of delivery over seeds — the verdict that survives galambos's variance.
    results: dict[str, Any] = {}
    for name, _kind, _head, _skip in selected:
        cells_n = sorted(by_config[name], key=lambda r: r["seed"])
        if not cells_n:
            continue
        dels = sorted(float(r["delivery"]) for r in cells_n)
        rets = sorted(float(r["mean_return"]) for r in cells_n)
        results[name] = dict(
            delivery_median=round(statistics.median(dels), 3), delivery_iqr=round(_iqr(dels), 3),
            delivery_worst=round(min(dels), 3), delivery_best=round(max(dels), 3),
            per_seed_delivery=[round(d, 3) for d in dels],
            return_median=round(statistics.median(rets), 2), n_seeds=len(cells_n),
            skip=cells_n[0]["skip"], n_params=cells_n[0]["n_params"])
        print(f"  [{name}] delivery median {results[name]['delivery_median']} "
              f"IQR {results[name]['delivery_iqr']} (n={len(cells_n)} seeds: {results[name]['per_seed_delivery']})",
              flush=True)

    stats_list = [stats_seed0[n] for n, _, _, _ in selected if n in stats_seed0]
    if stats_list:
        plot_scoreboard(stats_list, out / f"{a.task}_scoreboard",
                        title=f"{a.task}: per-node vs pooled vs MLP ({steps} steps, {a.seeds} seeds; bars=seed0)")
    try:
        import psutil
        rss_mb: float | None = round(psutil.Process().memory_info().rss / 1e6, 1)
    except ImportError:
        rss_mb = None
    report = dict(timestamp=now_stamp(), task=a.task, mode=a.mode, seeds=a.seeds, steps=steps,
                  wall_s=round(time.perf_counter() - start, 1), rss_mb=rss_mb, result=results)
    (out / f"{a.task}_pernode_ab.json").write_text(json.dumps(report, indent=2))
    results_to_csv(out / f"{a.task}_results", results)          # the aggregate table as CSV (alongside JSON/GIFs)
    print(f"  [artifacts] {out}", flush=True)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
