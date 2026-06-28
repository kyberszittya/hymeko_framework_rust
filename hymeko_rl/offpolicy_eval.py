"""Multi-seed off-policy architecture eval on a real-topology task.

The experiment cart-pole cannot give. PPO/DDPG/TD3/SAC all saturate the 2-vertex cart-pole, and a single
Galambos seed is too noisy to settle it (``reports/2026-06-22-galambos-structure-vs-capacity.md``). This driver
runs an **off-policy** learner (SAC the strongest; DDPG/TD3 available) on a **real-topology** task (the Galambos
6-vertex kinematic hypergraph), across ``>=5`` seeds, comparing the **HSiKAN backbone vs a params-matched MLP**,
scored by **curve-max** (the roadmap's explicit fix for the noisy final-snapshot metric) with median/IQR/worst.

It is env-agnostic and reuses everything: the trainers (``train_sac``/``train_offpolicy``, with their additive
``eval_fn`` seam), the builders (``build_sac``/``build_offpolicy``), and the episode-return scorer
(``evaluate.evaluate``). No new env, no new trainer, no copied train loop (CLAUDE.md §6.5).

    python -m hymeko_rl.offpolicy_eval --task galambos --algo sac --mode smoke   # 1 seed, cheap
    python -m hymeko_rl.offpolicy_eval --task galambos --algo sac --mode full    # 5 seeds, the result
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.ddpg import OffPolicyConfig, build_offpolicy, td3_config, train_offpolicy
from hymeko_rl.evaluate import evaluate, now_stamp
from hymeko_rl.hsikan_diagnose import Diagnosis, diagnose
from hymeko_rl.policy import POLICY_KINDS
from hymeko_rl.ppo import PPOEvalConfig, build_ppo, train_ppo_eval
from hymeko_rl.reach_arch_compare import _iqr   # one IQR helper for the whole package (no re-impl)
from hymeko_rl.sac import SACConfig, build_sac, train_sac

EnvFactory = Callable[[], Any]


# ----------------------------------------------------------------------------------------------------
# Task registry (Strategy) — a task name -> a fresh-env factory. Imports are lazy so importing this
# module stays cheap (mujoco is heavy); each call builds an independent env (fresh per seed).
# ----------------------------------------------------------------------------------------------------
def _cartpole_env() -> Any:
    """Inverted pendulum — the 2-vertex control floor (structure not load-bearing; the negative control)."""
    from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
    return InvertedPendulumEnv(mjcf=emit_cartpole_mjcf())


# Coin spawn difficulty for the Galambos architecture A/B. At d=1 the coin spawns across the whole reachable
# table (only one-arm-reachable) → the two-arm grasp+deliver is ~never achieved, so the comparison ties at
# FAILURE and is uninformative. A solvable difficulty keeps the coin in easy two-arm reach near the zone, so the
# goal is achievable and the HSiKAN-vs-MLP gap is meaningful (see reports/2026-06-24-galambos-hyperedge-ab.md).
_GALAMBOS_DIFFICULTY = 0.3


def _galambos_env() -> Any:
    """Galambos planar coin-grasp — 6-vertex two-arm kinematic hypergraph (coin in reach, solvable)."""
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    return PlanarGraspEnv.from_hymeko(max_steps=160, difficulty=_GALAMBOS_DIFFICULTY)


def _galambos_taskgraph_env() -> Any:
    """Galambos with the coin+zone IN the graph (grasp/goal hyperedges) — the structural ablation vs the
    robot-only ``galambos`` task: does putting the task objects in the graph let HSiKAN beat the MLP?"""
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    return PlanarGraspEnv.from_hymeko(max_steps=160, task_graph=True, difficulty=_GALAMBOS_DIFFICULTY)


def _arm6dof_env() -> Any:
    """6-DOF anthropomorphic arm reaching (emitted from .hymeko; position control, EE body 'tool')."""
    from hymeko_rl.env.arm_reach_env import ArmReachEnv
    return ArmReachEnv.from_hymeko("data/robotics/anthropomorphic_arm.hymeko",
                                   name="arm", control_mode="position", ee_body="tool")


def _quadruped_env() -> Any:
    """Quadruped goal-reaching — 14-vertex branching topology (planar base, the learnable walker)."""
    from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
    return QuadrupedGoalEnv()


def _fanuc_pick_env() -> Any:
    """FANUC 7-DOF arm+gripper pick-and-place — hard-exploration manipulation. NOTE: from-scratch off-policy is
    expected to struggle (the known-working path is BC warm-start, a separate experiment); the curve still
    reports the algo/backbone comparison at this regime."""
    from hymeko_rl.render_pick_place import fanuc_pick_env
    return fanuc_pick_env()


TASKS: dict[str, EnvFactory] = {
    "cartpole": _cartpole_env,      # inverted pendulum (2-vtx control floor)
    "galambos": _galambos_env,      # planar coin-grasp (6-vtx, robot-only graph)
    "galambos_taskgraph": _galambos_taskgraph_env,   # + coin/zone in the graph (grasp/goal hyperedges)
    "arm6dof": _arm6dof_env,        # anthropomorphic reaching (6-vtx, 6 DOF)
    "quadruped": _quadruped_env,    # goal-reaching quadruped (14-vtx, branching)
    "fanuc": _fanuc_pick_env,       # FANUC pick-and-place (7-DOF arm+gripper; hard-exploration)
}


# ----------------------------------------------------------------------------------------------------
# Algorithm registry (Strategy) — one (build, train, config) triple per off-policy member. build_sac and
# build_offpolicy share a kwargs signature, and the three configs share the fields the driver overrides,
# so there is exactly one build call site and one train call site (no per-algo wrapper; §6.5 #1).
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Algo:
    build: Callable[..., tuple[Any, list[Any]]]
    train: Callable[..., list[float]]
    make_cfg: Callable[..., Any]


_ALGOS: dict[str, _Algo] = {
    "ppo": _Algo(build_ppo, train_ppo_eval, lambda **kw: PPOEvalConfig(**kw)),   # on-policy, adapted in
    "sac": _Algo(build_sac, train_sac, lambda **kw: SACConfig(**kw)),
    "ddpg": _Algo(build_offpolicy, train_offpolicy, lambda **kw: OffPolicyConfig(**kw)),
    "td3": _Algo(build_offpolicy, train_offpolicy, lambda **kw: td3_config(**kw)),
}


@dataclass(frozen=True)
class Budget:
    """Per-cell training/eval budget (one cell = one (algo, backbone, seed) run).

    # Invariants ``total_steps >= eval_every >= 1``; ``n_eval >= 1``; ``hidden`` maps a backbone kind to its
    width, so HSiKAN vs MLP can be **params-matched** by widening the MLP."""

    total_steps: int
    eval_every: int
    n_eval: int
    hidden: dict[str, int]


def greedy_return_eval(n_eval: int, *, seed: int = 20_000) -> Callable[[Any, Any], float]:
    """An env-agnostic ``eval_fn(env, actor) -> mean episode return`` over ``n_eval`` greedy rollouts.

    The deterministic-mean action (``actor.action_mean``) is fed exactly as the trainers feed it (batch dim
    added, squeezed back), so the curve is a faithful return at the policy's greedy operating point.

    # Preconditions ``actor`` exposes ``action_mean(Tensor) -> Tensor``; ``env`` is a gym 5-tuple env with
    ``max_steps``. # Postconditions returns a finite float (mean summed reward)."""

    def _eval(env: Any, actor: Any) -> float:
        dev = next(actor.parameters()).device                      # follow the actor onto CPU or CUDA
        def action_fn(_env: Any, obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                a = actor.action_mean(torch.as_tensor(obs[None], dtype=torch.float32, device=dev))
            return np.asarray(a.squeeze(0).cpu().numpy(), dtype=np.float32)

        return evaluate(env, action_fn, source="eval", n_episodes=n_eval, seed0=seed).mean_return

    return _eval


class _Diverged(Exception):
    """Raised from the guarded eval the first time actor/critic params read structurally DIVERGED.

    Aborts the cell at the next eval checkpoint instead of grinding the full budget in denormal arithmetic —
    the recurring off-policy failure (large-reward Q-blow-up) that was indistinguishable from "slow" before.
    """


def _first_divergence(actor: Any, critics: Sequence[Any], sample_obs: torch.Tensor) -> Diagnosis | None:
    """The first unhealthy :class:`Diagnosis` among (actor, *critics), or ``None`` if all are healthy.

    The actor is diagnosed *with* ``sample_obs`` (param health + the HSiKAN structural forward → a named
    bad component); critics are param-only (no forward → no false positive from a critic's distinct call
    signature). # Postconditions a returned diagnosis has ``healthy is False``."""
    da = diagnose(actor, sample_obs=sample_obs)
    if not da.healthy:
        return da
    for c in critics:
        dc = diagnose(c)
        if not dc.healthy:
            return dc
    return None


def _env_dims(env: Any) -> tuple[int, int, int, float]:
    """``(n_vertices, feat, action_dim, action_scale)`` derived from an env's gym spaces.

    # Postconditions ``action_scale`` is finite and positive (raises otherwise — a saturating tanh guard)."""
    space = env.observation_space.shape
    assert space is not None
    n_vertices, feat = int(space[0]), int(space[1])
    act_space = env.action_space
    action_dim = int(np.prod(act_space.shape))
    action_scale = float(np.max(np.abs(np.asarray(act_space.high, dtype=np.float64))))
    if not np.isfinite(action_scale) or action_scale <= 0.0:
        raise ValueError(f"action_scale must be finite and positive; got {action_scale}")
    return n_vertices, feat, action_dim, action_scale


def _actor_params(kind: str, env: Any, hidden: int, *, n_critics: int = 2) -> int:
    """Parameter count of the SAC actor of ``kind`` at ``hidden`` on ``env`` (no training; for matching)."""
    nv, feat, ad, asc = _env_dims(env)
    kw: dict[str, Any] = {} if kind == "mlp" else {"hg_state": env.hg}
    actor, _ = build_sac(kind, obs_dim=feat, flat_dim=nv * feat, action_dim=ad,
                         action_scale=asc, n_critics=n_critics, hidden=hidden, **kw)
    return int(sum(p.numel() for p in actor.parameters()))


def match_mlp_hidden(task: str, hsikan_hidden: int, *, search: range = range(32, 401, 2),
                     ) -> tuple[int, int, int]:
    """Pick the MLP width whose param count is closest to HSiKAN@``hsikan_hidden`` on ``task``.

    The actor's log-std head is identical across backbones, so matching the SAC actor also keeps the TD3
    (deterministic) actor fair. # Postconditions ``(mlp_hidden, hsikan_params, mlp_params)``."""
    env = TASKS[task]()
    hk = _actor_params("hsikan", env, hsikan_hidden)
    best = min(search, key=lambda h: abs(_actor_params("mlp", env, h) - hk))
    return best, hk, _actor_params("mlp", env, best)


def _run_cell(factory: EnvFactory, algo_name: str, kind: str, seed: int, budget: Budget, *,
              gif_dir: Path | None = None, task: str = "", device: str = "cpu") -> dict[str, Any]:
    """Train one (algo, backbone, seed) cell on a fresh env; return its curve + curve-max + params.

    With ``gif_dir`` set, seed-0 cells also render a high-res GIF of the trained policy (the §9 animated output;
    one per (task, algo, backbone) to bound the count). # Preconditions ``algo_name in _ALGOS``;
    ``kind in POLICY_KINDS``. # Postconditions the dict carries ``curve`` (list), ``curve_max``/``final``
    (mean return) and a positive ``n_params``."""
    if kind not in POLICY_KINDS:
        raise ValueError(f"unknown backbone {kind!r}; expected {POLICY_KINDS}")
    env = factory()
    n_vertices, feat, action_dim, action_scale = _env_dims(env)
    algo = _ALGOS[algo_name]
    cfg = algo.make_cfg(total_steps=budget.total_steps, eval_every=budget.eval_every,
                        n_eval=budget.n_eval, seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw: dict[str, Any] = {} if kind == "mlp" else {"hg_state": env.hg}
    actor, _critics = algo.build(kind, obs_dim=feat, flat_dim=n_vertices * feat, action_dim=action_dim,
                                 action_scale=action_scale, n_critics=cfg.n_critics,
                                 hidden=budget.hidden[kind], device=device, **kw)
    ev = greedy_return_eval(budget.n_eval)
    sample_obs = torch.as_tensor(env.reset(seed=0)[0][None], dtype=torch.float32, device=device)
    curve: list[float] = []
    diag_caught: Diagnosis | None = None

    @torch.no_grad()
    def guarded_eval(e_env: Any, ac: Any) -> float:                # abort+localise on divergence, don't grind
        nonlocal diag_caught
        bad = _first_divergence(ac, _critics, sample_obs)
        if bad is not None:
            diag_caught = bad
            raise _Diverged
        v = ev(e_env, ac)
        curve.append(v)
        return v

    diverged = False
    try:
        algo.train(actor, _critics, env, cfg, eval_fn=guarded_eval)
    except _Diverged:
        diverged = True
        roles = [p.role for p in diag_caught.bad] if diag_caught else []
        print(f"  [DIVERGED {task}/{algo_name}/{kind}/{seed} → {roles or 'forward-unrunnable'}]", flush=True)
    final = (curve[-1] if curve else 0.0) if diverged else ev(env, actor)
    if gif_dir is not None and seed == 0 and not diverged:
        from hymeko_rl.campaign_viz import render_actor_gif
        actor.to("cpu")                                            # render on CPU (cell done; GIF not perf-critical)
        try:
            render_actor_gif(factory(), actor, gif_dir / f"{task}_{algo_name}_{kind}")
        except Exception as exc:   # noqa: BLE001 — viz is best-effort; a GL/camera failure must not kill the cell
            print(f"  [gif {task}/{algo_name}/{kind} skipped: {type(exc).__name__}: {exc}]", flush=True)
    return dict(curve=[round(float(c), 3) for c in curve],
                curve_max=round(max(curve), 3) if curve else round(final, 3),
                final=round(float(final), 3),
                diverged=diverged,
                localized_to=[p.role for p in diag_caught.bad] if diag_caught else [],
                n_params=int(sum(p.numel() for p in actor.parameters())))


def aggregate_records(records: Sequence[dict[str, Any]], *, seed_set: set[int] | None = None,
                      ) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Group per-cell records by ``(task, algo, backbone)`` → curve-max stats over seeds.

    The single aggregator used by both the live driver and the offline table generator (no re-impl).
    'Worst' is the minimum (reward); 'best' the maximum. # Postconditions one entry per group present in
    ``records`` (optionally filtered to ``seed_set``), each with median/IQR/worst/best + per-seed vectors."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in records:
        if seed_set is not None and int(r["seed"]) not in seed_set:
            continue
        groups.setdefault((str(r["task"]), str(r["algo"]), str(r["backbone"])), []).append(r)
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, cells in groups.items():
        cells = sorted(cells, key=lambda c: int(c["seed"]))
        maxes = sorted(float(c["curve_max"]) for c in cells)
        out[key] = dict(
            n_params=int(cells[0]["n_params"]), n_seeds=len(cells),
            curve_max_median=round(statistics.median(maxes), 3),
            curve_max_iqr=round(_iqr(maxes), 3),
            curve_max_worst=round(min(maxes), 3),
            curve_max_best=round(max(maxes), 3),
            per_seed_max=[round(m, 3) for m in maxes],
            per_seed_final=[round(float(c["final"]), 3) for c in cells],
        )
    return out


def _load_journal(journal: Path | None) -> dict[str, dict[str, Any]]:
    """Read a per-cell journal (one JSON object per line) into ``{key: record}`` (resume support)."""
    done: dict[str, dict[str, Any]] = {}
    if journal is not None and journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[str(rec["key"])] = rec
    return done


def compare_offpolicy(task: str, *, algos: Sequence[str], backbones: Sequence[str],
                      seeds: Sequence[int], budget: Budget,
                      journal: Path | None = None,
                      gif_dir: Path | None = None, device: str = "cpu") -> dict[str, dict[str, Any]]:
    """Run every (algo, backbone, seed) cell; aggregate curve-max as median / IQR / worst over seeds.

    'Worst' for a reward metric is the **minimum** over seeds; 'best' the maximum. The HSiKAN-vs-MLP verdict
    is read off the per-(algo,backbone) median gap **relative to the cross-seed IQR** (a gap inside the IQR is
    a tie, not a win) — never off a single seed or the noisy final snapshot.

    ``journal`` (optional) makes the campaign **resumable** (CLAUDE.md §4): each finished cell is appended as
    one JSON line keyed ``task/algo/backbone/seed``; on a rerun, already-journalled cells are skipped and only
    the missing ones run. Aggregation reads journalled + freshly-run cells together.

    # Preconditions ``task in TASKS``; each algo in ``_ALGOS``; ``len(seeds) >= 1``.
    # Postconditions one entry per ``"algo/backbone"`` with the curve-max statistics, the per-seed vectors,
    and ``n_params`` (so the params-matched claim is auditable)."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected {sorted(TASKS)}")
    factory = TASKS[task]
    done = _load_journal(journal)
    records: list[dict[str, Any]] = list(done.values())
    for algo_name in algos:
        for kind in backbones:
            for s in seeds:
                key = f"{task}/{algo_name}/{kind}/{s}"
                if key in done:
                    print(f"  [skip {key} — journalled]", flush=True)
                    continue
                print(f"  [run  {key}]", flush=True)
                cell = _run_cell(factory, algo_name, kind, s, budget, gif_dir=gif_dir, task=task, device=device)
                rec = dict(key=key, task=task, algo=algo_name, backbone=kind, seed=int(s),
                           timestamp=now_stamp(), **cell)
                records.append(rec)
                if journal is not None:
                    journal.parent.mkdir(parents=True, exist_ok=True)
                    with journal.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")

    agg = aggregate_records(records, seed_set={int(s) for s in seeds})
    return {f"{algo}/{kind}": stats for (t, algo, kind), stats in agg.items() if t == task}


_SMOKE = dict(total_steps=2_000, eval_every=1_000, n_eval=5)
_FULL = dict(total_steps=30_000, eval_every=3_000, n_eval=20)

# Per-task FULL total_steps override (cart-pole solves in ~4k → 30k is wasted wall; the rest keep 30k).
_TASK_STEPS: dict[str, int] = {"cartpole": 15_000}


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default=["galambos"], nargs="+", choices=list(TASKS),
                    help="one or more tasks (the whole campaign in one resumable command)")
    ap.add_argument("--algo", default=["sac"], nargs="+", choices=list(_ALGOS))
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--hidden", type=int, default=64, help="HSiKAN width")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="override the diagnostic/eval cadence (steps); small = catch divergence fast")
    ap.add_argument("--mlp-hidden", default="auto",
                    help="MLP width: 'auto' (params-match HSiKAN, the default) or an int override")
    ap.add_argument("--seeds", type=int, default=None, help="override the seed count (default: mode's)")
    ap.add_argument("--journal", default=None,
                    help="per-cell resume journal (.jsonl); finished cells are skipped on rerun")
    ap.add_argument("--out", default=None, help="write the aggregate JSON report to this path")
    ap.add_argument("--gif", action="store_true",
                    help="render a high-res GIF of the seed-0 policy per (task,algo,backbone) (§9 animated output)")
    ap.add_argument("--gif-dir", default="reports/gifs/campaign", help="where the GIFs go (with --gif)")
    ap.add_argument("--plot", action="store_true",
                    help="write a grouped-bar curve-max plot per task/algo (§9 plotted output; needs --journal)")
    ap.add_argument("--device", default="cpu", help="cpu (default, proven) or cuda — moves the nets + batches")
    a = ap.parse_args(argv)

    n_seeds = a.seeds if a.seeds is not None else (1 if a.mode == "smoke" else 5)
    seeds = list(range(n_seeds))
    base = _SMOKE if a.mode == "smoke" else _FULL
    journal = Path(a.journal) if a.journal else None
    gif_dir = Path(a.gif_dir) if a.gif else None
    if gif_dir is not None:
        gif_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    result: dict[str, dict[str, Any]] = {}
    per_task_hidden: dict[str, dict[str, int]] = {}
    for task in a.task:
        steps = int(base["total_steps"]) if a.mode == "smoke" else _TASK_STEPS.get(task, int(base["total_steps"]))
        if a.mlp_hidden == "auto":
            mlp_hidden, hk, mk = match_mlp_hidden(task, a.hidden)
            print(f"  [params-match] {task}: hsikan@{a.hidden}={hk} ~ mlp@{mlp_hidden}={mk}", flush=True)
        else:
            mlp_hidden = int(a.mlp_hidden)
        budget = Budget(total_steps=steps, eval_every=a.eval_every or int(base["eval_every"]),
                        n_eval=int(base["n_eval"]),
                        hidden={"hsikan": a.hidden, "signedkan": a.hidden, "mlp": mlp_hidden})
        per_task_hidden[task] = budget.hidden
        task_result = compare_offpolicy(task, algos=a.algo, backbones=["hsikan", "mlp"],
                                        seeds=seeds, budget=budget, journal=journal, gif_dir=gif_dir,
                                        device=a.device)
        for k, v in task_result.items():
            result[f"{task}/{k}"] = v
    wall = time.perf_counter() - start
    try:
        import psutil
        rss_mb: float | None = round(psutil.Process().memory_info().rss / 1e6, 1)
    except ImportError:
        rss_mb = None

    report: dict[str, Any] = dict(
        timestamp=now_stamp(), tasks=a.task, algos=a.algo, mode=a.mode, seeds=seeds,
        hidden=per_task_hidden, wall_s=round(wall, 1), rss_mb=rss_mb, result=result)
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
    if a.plot and journal is not None and journal.exists():
        from hymeko_rl.campaign_viz import plot_metric
        recs = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        png = (gif_dir or Path("reports")) / "campaign_curvemax.png"
        try:
            plot_metric(recs, png, metric="curve_max", title="curve-max: HSiKAN vs MLP")
            print(f"  [plot] {png}", flush=True)
        except Exception as exc:   # noqa: BLE001 — plotting best-effort, must not fail a finished run
            print(f"  [plot skipped: {type(exc).__name__}: {exc}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
