"""Designed-structure control on real MuJoCo plants (Phase 3 / Kato's real-plant test).

Trains the SAME SAC actor over different controller hypergraphs on a balancing (unstable) plant: the plant's
**kinematic** graph (baseline), the kinematic graph **augmented** with a combinatorial design (Steiner /
sunflower over the body indices — the design adds structural coverage the mechanics lack), and a flat **MLP**.
The thesis: a designed combinatorial structure gives the controller a complete feature basis *by construction*,
so it balances at least as well — more robustly — without learning it.

Reuses the existing A/B harness wholesale (``exp_pernode_actor_ab.run_config`` → ``build_sac``/``train_sac``/
``evaluate``/GIF): the *only* new thing is :class:`DesignAugmentedEnv`, which augments the kinematic hypergraph
and zero-pads the per-vertex obs for the design hubs. No copied trainer, no per-arm code (§6.5).

    python -m hymeko_rl.experiments.exp_designed_control --plant pendulum --mode smoke   # harness validation (rung 0/1)
    python -m hymeko_rl.experiments.exp_designed_control --plant quadruped --mode full   # the thesis test (rung 2)
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hymeko_rl.eval.evaluate import experiment_dir, plot_scoreboard
from hymeko_rl.experiments.exp_pernode_actor_ab import run_config
from hymeko_rl.control.hypergraph_designs import steiner_blocks, sunflower_cover
from hymeko_rl.experiments.reach_arch_compare import _iqr
from hymeko_rl.train.sac import SACConfig


class DesignAugmentedEnv(gym.Wrapper):
    """Augment the wrapped env's kinematic hypergraph with design hyperedges over its body indices (one hub per
    block, via ``with_task_hyperedges``) and zero-pad the per-vertex obs for the hubs. The base vertices keep
    their physical ``[q, q̇]``; the design hubs add structural coverage. # Invariants observation rows
    ``0..N_kin-1`` are the physical obs; the rest are zeros (relation hubs carry no measurement)."""

    def __init__(self, base: gym.Env, blocks: "list[tuple[int, ...]]") -> None:
        super().__init__(base)
        self.hg = base.hg.with_task_hyperedges(           # type: ignore[attr-defined]
            new_entities=[], hyperedges=[(f"design{i}", tuple(b)) for i, b in enumerate(blocks)])
        self._n_pad = self.hg.n_vertices - base.hg.n_vertices   # type: ignore[attr-defined]
        feat = base.observation_space.shape[1]            # type: ignore[index]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.hg.n_vertices, feat), dtype=np.float32)

    def _pad(self, obs: np.ndarray) -> np.ndarray:
        if self._n_pad == 0:
            return obs
        return np.concatenate([obs, np.zeros((self._n_pad, obs.shape[1]), dtype=obs.dtype)], axis=0)

    def reset(self, **kw: Any) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kw)
        return self._pad(obs), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, term, trunc, info = self.env.step(action)
        return self._pad(obs), reward, term, trunc, info

    def __getattr__(self, name: str) -> Any:
        # Forward base-env attributes the harness reads (max_steps, model, render, …) — this gymnasium Wrapper
        # does not auto-delegate them. Only called on normal-lookup miss; guard ``env``/private names so a miss
        # before ``self.env`` is set cannot recurse.
        if name == "env" or name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)


def _design_blocks(n_bodies: int, design: str) -> "list[tuple[int, ...]]":
    """Blocks (subsets of ``range(n_bodies)``) for ``design`` ∈ {steiner, sunflower}. Steiner needs
    ``n ∈ {7,9}``; otherwise it falls back to a sunflower cover (so the augmentation is always defined)."""
    if design == "steiner" and n_bodies in (7, 9):
        return steiner_blocks(n_bodies)
    return sunflower_cover(n_bodies, core_size=max(1, n_bodies // 3))


# (arm name, backbone kind, design) — swap only the hypergraph; pooled head + no skip everywhere.
_ARMS: tuple[tuple[str, str, "str | None"], ...] = (
    ("kinematic", "hsikan", None),
    ("steiner_aug", "hsikan", "steiner"),
    ("sunflower_aug", "hsikan", "sunflower"),
    ("mlp", "mlp", None),
)

_PLANTS: dict[str, Callable[[], gym.Env]] = {}


def _register_plants() -> None:
    """Lazy import (MuJoCo envs construct a sim at import) → keep the registry cheap until used."""
    from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv
    _PLANTS["pendulum"] = InvertedPendulumEnv
    try:
        # Rung 2 wants *standing* (balance): quadruped_stand is the free-base stay-upright task (STAND_REWARD),
        # the postural regime where declared topology is hypothesised to pay. "quadruped" (walk-to-goal) is
        # kept as the non-cyclic control (flat HSiKAN already wins there — the negative reference).
        from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
        _PLANTS["quadruped"] = QuadrupedGoalEnv
        _PLANTS["quadruped_stand"] = lambda: QuadrupedGoalEnv(base="free", task="stand", max_steps=250)
    except (ImportError, AttributeError):
        pass


def _arm_factory(base_factory: Callable[[], gym.Env], design: "str | None") -> Callable[[], gym.Env]:
    if design is None:
        return base_factory
    n = base_factory().hg.n_vertices                      # type: ignore[attr-defined]
    blocks = _design_blocks(n, design)
    return lambda: DesignAugmentedEnv(base_factory(), blocks)


def run_designed_control(*, plant: str, steps: int, hidden: int, seeds: int, n_eval: int, out: Path,
                         arms: "list[str] | None" = None) -> dict[str, Any]:
    """Train each arm (kinematic / design-augmented / MLP) on ``plant``; report balance return median/IQR over
    seeds. # Postconditions one entry per arm with ``return_median``, ``return_iqr``, ``per_seed_return``."""
    base_factory = _PLANTS[plant]
    selected = [a for a in _ARMS if arms is None or a[0] in arms]
    results: dict[str, Any] = {}
    stats0: dict[str, Any] = {}
    for name, kind, design in selected:
        factory = _arm_factory(base_factory, design)
        rets: list[float] = []
        nv = factory().hg.n_vertices                      # type: ignore[attr-defined]
        for sd in range(seeds):
            cfg = SACConfig(total_steps=steps, eval_every=steps, n_eval=5, seed=sd)
            stats, result = run_config(name, kind, "pooled", "none", task=plant, env_factory=factory, cfg=cfg,
                                       hidden=hidden, n_eval=n_eval, out=out, render_gif=(sd == 0))
            rets.append(float(result["mean_return"]))
            if sd == 0:
                stats0[name] = stats
        results[name] = dict(return_median=round(statistics.median(rets), 2), return_iqr=round(_iqr(rets), 2),
                             per_seed_return=[round(r, 2) for r in rets], n_vertices=int(nv), kind=kind,
                             design=design, n_seeds=seeds)
        print(f"  [{name}] return median {results[name]['return_median']} "
              f"(nv={nv}, seeds={results[name]['per_seed_return']})", flush=True)
    if stats0:
        plot_scoreboard(list(stats0.values()), out / f"{plant}_scoreboard",
                        title=f"{plant}: kinematic vs designed vs MLP ({steps} steps, {seeds} seeds)")
    return results


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _register_plants()
    ap.add_argument("--plant", default="pendulum", choices=list(_PLANTS))
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--arms", default=None, help="comma-separated arm subset (default: all)")
    ap.add_argument("--out-dir", default="experiments")
    a = ap.parse_args(argv)
    steps = a.steps if a.steps is not None else (1_500 if a.mode == "smoke" else 30_000)
    arms = a.arms.split(",") if a.arms else None
    out = experiment_dir(a.out_dir, f"designed_{a.plant}")
    print(f"  [designed-control] plant={a.plant} steps={steps} seeds={a.seeds} -> {out}", flush=True)
    start = time.perf_counter()
    results = run_designed_control(plant=a.plant, steps=steps, hidden=a.hidden, seeds=a.seeds,
                                   n_eval=a.n_eval, out=out, arms=arms)
    report = dict(plant=a.plant, steps=steps, seeds=a.seeds, wall_s=round(time.perf_counter() - start, 1),
                  results=results)
    (out / "designed_control.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
