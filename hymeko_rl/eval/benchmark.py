"""RL benchmark harness — every (algorithm × backbone × scenario) cell, multi-seed, one leaderboard.

The point (Hajdu, 2026-07-01): test the structural architectures (HSiKAN / SA-HSiKAN backbones, and — as they
land — per-node / tree heads and the TreeChannel) **against vanilla TD3 and the other algorithms**, on the same
scenarios, with the same discipline (multi-seed median/IQR, §3). One config-driven entry, not a function per cell
(§6.5 #1): a `BenchCell` names the axes, `train_and_eval` dispatches to the right trainer, `run_benchmark` sweeps
+ aggregates + plots + prints the leaderboard. Everything flows through the simulator ecosystem (`tasks.py`) and
the unified evaluator (`evaluate.eval_metric`), so adding a scenario or an algorithm is one line, not a new driver.

    from hymeko_rl.eval.benchmark import BenchCell, run_benchmark
    run_benchmark([BenchCell("cartpole", "td3", "hsikan"), BenchCell("cartpole", "td3", "mlp")], seeds=[0,1,2])
"""
from __future__ import annotations

import statistics as st
from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.train.ddpg import OffPolicyConfig, build_offpolicy, td3_config, train_offpolicy
from hymeko_rl.agents.policy import build_policy
from hymeko_rl.train.ppo import PPOConfig, train_ppo
from hymeko_rl.eval.tasks import evaluate_task, get_task

_OFF_POLICY = {"ddpg", "td3"}                 # from-scratch off-policy (td3_bc/sac/rlpd add here once demo-wired)
_ALGOS = _OFF_POLICY | {"ppo"}
# arm_reach reports a distance (lower is better); everything else is a rate/steps/return (higher is better).
_LOWER_BETTER = {"arm_reach"}


@dataclass(frozen=True)
class BenchCell:
    """One benchmark cell: a scenario × algorithm × backbone at a fixed interaction budget."""
    scenario: str
    algo: str        # ppo / ddpg / td3
    backbone: str    # mlp / hsikan / sa_hsikan / signedkan / mixture
    steps: int = 20_000

    def label(self) -> str:
        return f"{self.algo}/{self.backbone}"


def _skip_eval(_env: Any, _ac: Any) -> float:
    return 0.0                                 # the benchmark evaluates the FINAL policy via evaluate_task, not in-loop


def train_and_eval(cell: BenchCell, seed: int, *, n_eval: int = 8) -> float:
    """Train one cell for ``seed`` and return the scenario's aggregate metric (median over ``n_eval`` episodes).

    # Preconditions ``cell.algo in _ALGOS``; ``cell.scenario`` is registered; ``cell.backbone in POLICY_KINDS``."""
    if cell.algo not in _ALGOS:
        raise ValueError(f"unknown algo {cell.algo!r}; expected {sorted(_ALGOS)}")
    spec = get_task(cell.scenario)
    env = spec.make_env()
    ss = env.observation_space.shape
    nv, feat = int(ss[0]), int(ss[1])
    n_act = int(env.action_space.shape[0])         # universal (n_actions isn't on every env, e.g. cartpole)
    kw: dict[str, Any] = {} if cell.backbone == "mlp" else {"hg_state": env.hg}

    if cell.algo == "ppo":
        obs_dim = nv * feat if cell.backbone == "mlp" else feat
        ac: Any = build_policy(cell.backbone, obs_dim=obs_dim, action_dim=n_act, **kw)
        # Vectorized rollout is the measured-faster default (batched forward; learning preserved, test_ppo_vec).
        train_ppo(ac, env, PPOConfig(n_iters=max(1, cell.steps // 2048), n_steps=2048, seed=seed),
                  n_envs=8, make_env=spec.make_env)
        policy = ac
    else:
        scale = float(np.max(np.abs(env.action_space.high)))
        n_crit = 2 if cell.algo == "td3" else 1
        actor, critics = build_offpolicy(cell.backbone, obs_dim=feat, flat_dim=nv * feat, action_dim=n_act,
                                         action_scale=scale, n_critics=n_crit, **kw)
        cfg = (td3_config(total_steps=cell.steps, seed=seed) if cell.algo == "td3"
               else OffPolicyConfig(total_steps=cell.steps, seed=seed))
        train_offpolicy(actor, critics, env, cfg, eval_fn=_skip_eval)
        policy = actor

    res = evaluate_task(cell.scenario, policy, n_episodes=n_eval, seed=20_000)
    vals = [float(r[0]) if isinstance(r, tuple) else float(r) for r in res]   # (lift,place)->lift; else scalar
    return float(np.median(vals))


def run_cell(cell: BenchCell, seeds: "list[int]", *, n_eval: int = 8) -> "dict[str, Any]":
    """Median/IQR of a cell's scenario metric across ``seeds``."""
    scores = [train_and_eval(cell, s, n_eval=n_eval) for s in seeds]
    return {"scenario": cell.scenario, "algo": cell.algo, "backbone": cell.backbone, "label": cell.label(),
            "scores": scores, "median": st.median(scores),
            "iqr": (float(np.percentile(scores, 25)), float(np.percentile(scores, 75)))}


def run_benchmark(cells: "list[BenchCell]", seeds: "list[int]", *, out: "str | None" = None) -> "list[dict[str, Any]]":
    """Sweep every cell over ``seeds``, print a per-scenario leaderboard, optionally plot. Returns the rows."""
    rows = [run_cell(c, seeds) for c in cells]
    for scenario in sorted({r["scenario"] for r in rows}):
        sub = [r for r in rows if r["scenario"] == scenario]
        sub.sort(key=lambda r: r["median"], reverse=scenario not in _LOWER_BETTER)
        arrow = "lower=better" if scenario in _LOWER_BETTER else "higher=better"
        print(f"\n=== {scenario} ({arrow}) ===")
        for rank, r in enumerate(sub, 1):
            print(f"  {rank}. {r['label']:22} median={r['median']:+.3f}  "
                  f"IQR[{r['iqr'][0]:+.3f},{r['iqr'][1]:+.3f}]  seeds={[round(s, 3) for s in r['scores']]}")
    if out is not None:
        _plot(rows, out)
    return rows


def _plot(rows: "list[dict[str, Any]]", out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenarios = sorted({r["scenario"] for r in rows})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4.5), dpi=140, squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        sub = [r for r in rows if r["scenario"] == scenario]
        sub.sort(key=lambda r: r["median"], reverse=scenario not in _LOWER_BETTER)
        xs = list(range(len(sub)))
        meds = [r["median"] for r in sub]
        lo = [r["median"] - r["iqr"][0] for r in sub]
        hi = [r["iqr"][1] - r["median"] for r in sub]
        ax.bar(xs, meds, yerr=[lo, hi], capsize=5, color="#3477a8")
        for x, r in zip(xs, sub):
            ax.scatter([x] * len(r["scores"]), r["scores"], color="black", s=14, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([r["label"] for r in sub], rotation=30, ha="right", fontsize=8)
        ax.set_title(scenario)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("RL benchmark — median/IQR per scenario (structural backbones vs vanilla)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, bbox_inches="tight")
    print(f"\nwrote {out}")
