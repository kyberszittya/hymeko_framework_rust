"""A/B: flat weighted-sum reward vs HTL-robustness reward, on the galambos collaborative task.

Same env, same HSiKAN backbone, same SAC budget — only the reward *composition* differs: the flat
``Σ wᵢ·termᵢ`` (``galambos_task.hymeko``) vs the instantaneous robustness ``ρ`` of the temporal-geometric
spec ``data/robotics/galambos_spec.htl`` (:class:`hymeko_rl.control.htl_reward.HtlRewardSpec`). The comparable
metric is **delivery** (the two reward scales differ, so raw return is not comparable); we also report the
per-episode HTL delivery **verdict** ``F[0,T](in_zone)`` — the same formula's temporal reading — to show
the dual use (one spec → reward + accountability). Output is three-form (§9): JSON numbers, a scoreboard
plot, and a GIF per variant. Reuses the SAC trainer, ``evaluate()``, and the shared renderers — no copied
train loop, no re-implemented rendering (CLAUDE.md §6.1/§6.5#3).

    python -m hymeko_rl.experiments.exp_htl_reward_ab --mode smoke   # short, 1 seed — proves plumbing + 3-form output
    python -m hymeko_rl.experiments.exp_htl_reward_ab --mode full    # longer single seed
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.viz.campaign_viz import _greedy_action_fn, render_actor_gif
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.eval.evaluate import EvalStats, evaluate, experiment_dir, now_stamp, plot_scoreboard, results_to_csv
from hymeko_rl.control.htl_reward import HtlRewardSpec
from hymeko_rl.eval.offpolicy_eval import _env_dims
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac

_DIFFICULTY = 0.3


def make_env(*, htl: bool) -> PlanarGraspEnv:
    """A fresh galambos env; ``htl`` swaps the flat reward for the HTL-robustness reward (the only diff)."""
    env = PlanarGraspEnv.from_hymeko(max_steps=160, difficulty=_DIFFICULTY)
    if htl:
        env.reward_spec = HtlRewardSpec()      # drop-in (duck-typed RewardSpec.evaluate); no env change
    return env


def train_hsikan_sac(env: PlanarGraspEnv, cfg: SACConfig, *, hidden: int) -> Any:
    """Train one HSiKAN SAC actor on ``env`` (reusing the shared trainer); returns the trained actor."""
    nv, feat, action_dim, action_scale = _env_dims(env)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    actor, critics = build_sac("hsikan", obs_dim=feat, flat_dim=nv * feat, action_dim=action_dim,
                               action_scale=action_scale, n_critics=cfg.n_critics, hidden=hidden,
                               hg_state=env.hg)
    train_sac(actor, critics, env, cfg)
    return actor


def htl_verdict_rate(env: PlanarGraspEnv, actor: Any, *, n: int, seed0: int = 30_000) -> float:
    """Fraction of greedy episodes whose HTL delivery verdict ``F[0,T](in_zone>0.5)`` is satisfied —
    the temporal reading of the same spec, scored per episode (the accountability signal, not the reward)."""
    spec = env.reward_spec if isinstance(env.reward_spec, HtlRewardSpec) else HtlRewardSpec()
    act = _greedy_action_fn(actor)
    satisfied = 0
    for k in range(n):
        mon = spec.episode_monitor(horizon=env.max_steps + 1)
        obs, _ = env.reset(seed=seed0 + k)
        for t in range(env.max_steps):
            obs, _r, term, trunc, _info = env.step(act(env, obs))
            mon.observe(spec.event(env, t))
            if term or trunc:
                break
        satisfied += int(mon.satisfied())
    return satisfied / max(1, n)


def run_variant(*, htl: bool, cfg: SACConfig, hidden: int, n_eval: int, out: Path,
                ) -> tuple[EvalStats, dict[str, Any]]:
    """Train + evaluate one reward variant; render its GIF; return its (EvalStats, result-dict)."""
    name = "htl" if htl else "flat"
    actor = train_hsikan_sac(make_env(htl=htl), cfg, hidden=hidden)
    stats = evaluate(make_env(htl=htl), _greedy_action_fn(actor), source=name,
                     n_episodes=n_eval, seed0=30_000)
    verdict = htl_verdict_rate(make_env(htl=htl), actor, n=n_eval)
    try:
        render_actor_gif(make_env(htl=htl), actor, out / f"galambos_{name}")
    except Exception as exc:   # noqa: BLE001 — viz is best-effort; a GL failure must not kill the A/B
        print(f"  [gif {name} skipped: {type(exc).__name__}: {exc}]", flush=True)
    result = dict(delivery=round(stats.success_rate, 3), mean_return=round(stats.mean_return, 2),
                  deaths=stats.deaths, htl_verdict_satisfied=round(verdict, 3), n_eval=n_eval)
    print(f"  [{name}] {stats.summary()}  htl_verdict={verdict:.0%}", flush=True)
    return stats, result


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--out-dir", default="experiments",
                    help="parent dir; the run lands in <out-dir>/YYYY_MM_DD_HH_MM_htl_reward_ab/")
    a = ap.parse_args(argv)

    steps = 2_000 if a.mode == "smoke" else 30_000
    cfg = SACConfig(total_steps=steps, eval_every=steps, n_eval=5, seed=a.seed)
    out = experiment_dir(a.out_dir, "htl_reward_ab")   # one timestamped folder for all this run's artifacts
    start = time.perf_counter()

    stats_list: list[EvalStats] = []
    results: dict[str, Any] = {}
    for htl in (False, True):
        stats, result = run_variant(htl=htl, cfg=cfg, hidden=a.hidden, n_eval=a.n_eval, out=out)
        stats_list.append(stats)
        results["htl" if htl else "flat"] = result

    plot_scoreboard(stats_list, out / "scoreboard",
                    title=f"galambos reward A/B — flat Σw vs HTL ρ ({a.mode}, seed {a.seed})")
    try:
        import psutil
        rss_mb: float | None = round(psutil.Process().memory_info().rss / 1e6, 1)
    except ImportError:
        rss_mb = None
    report = dict(timestamp=now_stamp(), mode=a.mode, seed=a.seed, steps=steps,
                  wall_s=round(time.perf_counter() - start, 1), rss_mb=rss_mb, result=results)
    (out / "htl_reward_ab.json").write_text(json.dumps(report, indent=2))
    results_to_csv(out / "results", results)           # the results table as CSV (alongside the JSON/GIFs)
    print(f"  [artifacts] {out}", flush=True)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
