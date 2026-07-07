"""One reusable off-policy RL campaign — the BC→TD3+BC loop, factored out of the per-experiment copy-paste.

Every recent experiment re-implemented the same body: build env → collect demos → behaviour-clone → refine with
``train_offpolicy`` under a best-checkpoint ``eval_fn`` → measure → write ``results.json`` + a GIF. That is the
``§6.5 #3`` per-experiment-scaffold duplication. This module lifts it into a single :class:`Campaign` so a new
experiment is a *config plus two closures* (an env factory and a policy builder), not a 60-line script.

Design (composition over inheritance, strategies as plain callables):
  * ``make_env: () -> env``              — the environment factory (a thunk; rollout/eval build fresh copies).
  * ``build: (env) -> (actor, critics)`` — the architecture strategy (e.g. joint ``build_offpolicy`` vs the
    collaborative ``build_collaborative_offpolicy``). Swapping architectures is swapping this closure.
  * ``measure: (make_env, actor) -> {metric: value}`` — one evaluation, many scalars (e.g. delivery + arm-crash).
    Best-checkpoint selects on :attr:`CampaignConfig.select`.
  * ``demos: (env, n, seed) -> (obs, acts) | None`` — the BC teacher (e.g. ``collect_galambos_demos``); ``None``
    trains pure TD3.

Everything self-contains in one :func:`~hymeko_rl.eval.evaluate.experiment_dir`: ``policies/`` (best per seed),
``gifs/`` (best seed), ``results.json``, and — the piece that was missing — **``run.log``** (all stdout tee'd, so
the log lives with the artifacts, not in an ephemeral scratchpad). Compare architectures with :func:`compare`.
"""
from __future__ import annotations

import copy
import json
import statistics as st
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

from hymeko_rl.train.bc import behaviour_clone
from hymeko_rl.train.ddpg import OffPolicyConfig, td3_bc_config, td3_config, train_offpolicy
from hymeko_rl.eval.evaluate import experiment_dir

# ── strategy signatures (plain callables — no class hierarchy needed) ─────────
MakeEnv = Callable[[], Any]
BuildPolicy = Callable[[Any], "tuple[Any, list[Any]]"]
Measure = Callable[[MakeEnv, Any], "dict[str, float]"]
Demos = Callable[[Any, int, int], "tuple[np.ndarray, np.ndarray]"]


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


@dataclass(frozen=True)
class CampaignConfig:
    """What a campaign needs beyond its closures. ``select`` is the metric best-checkpoint maximises."""
    name: str
    select: str = "delivery"
    seeds: "tuple[int, ...]" = (0, 1, 2)
    total_steps: int = 200_000
    eval_every: int = 25_000
    n_demos: int = 200
    bc_epochs: int = 200
    n_eval: int = 50
    n_envs: int = 8
    device: str = "auto"
    offpolicy: "dict[str, float] | None" = None   # off-policy knob overrides (noise_scale, bc_coef, …) — comes
                                               # from the experiment profile's @offpolicy term; None = defaults
    bc_batch: int = 128                        # BC minibatch (512 + device="auto" = the measured GPU 3.9×)


@dataclass
class SeedResult:
    """One seed's outcome: the best-checkpoint metrics (``peak``) and the full eval ``curve``."""
    seed: int
    peak: "dict[str, Any]"
    curve: "list[dict[str, Any]]" = field(default_factory=list)
    wall_s: float = 0.0


@contextmanager
def tee_stdout(path: Path) -> Iterator[None]:
    """Duplicate stdout to ``path`` for the duration — so a run's log is saved with its artifacts (not lost to an
    ephemeral scratchpad). # Postconditions ``path`` holds every line printed inside the ``with`` block."""
    class _Tee:
        def __init__(self, *streams: Any) -> None:
            self._streams = streams

        def write(self, s: str) -> int:
            for st_ in self._streams:
                st_.write(s)
            return len(s)

        def flush(self) -> None:
            for st_ in self._streams:
                st_.flush()

    handle = path.open("w", encoding="utf-8")
    prev = sys.stdout
    sys.stdout = _Tee(prev, handle)
    try:
        yield
    finally:
        sys.stdout = prev
        handle.close()


class Campaign:
    """A multi-seed BC→TD3+BC experiment for ONE architecture, with best-checkpoint selection and self-contained
    artifacts. Compose several via :func:`compare` for an A/B.

    # Preconditions ``build(make_env())`` returns ``(actor, critics)`` satisfying the ``train_offpolicy`` contract;
      ``measure`` returns a dict containing :attr:`CampaignConfig.select`.
    # Postconditions :meth:`run` writes ``results.json`` + ``policies/`` + ``gifs/`` + ``run.log`` into one
      ``experiment_dir`` and returns the summary dict.
    """

    def __init__(self, cfg: CampaignConfig, make_env: MakeEnv, build: BuildPolicy, measure: Measure, *,
                 demos: "Demos | None" = None, gif: bool = True) -> None:
        self.cfg = cfg
        self.make_env = make_env
        self.build = build
        self.measure = measure
        self.demos = demos
        self.gif = gif

    def _train_seed(self, seed: int, exp: Path) -> SeedResult:
        demo_env = self.make_env()
        try:
            offline = self.demos(demo_env, self.cfg.n_demos, seed) if self.demos is not None else None
        finally:
            _close_env(demo_env)
        torch.manual_seed(seed)
        np.random.seed(seed)
        build_env = self.make_env()
        try:
            actor, critics = self.build(build_env)
        finally:
            _close_env(build_env)
        if offline is not None:
            behaviour_clone(actor, offline[0], offline[1], n_epochs=self.cfg.bc_epochs, seed=seed,
                            batch=self.cfg.bc_batch, device=self.cfg.device)
        curve: "list[dict[str, Any]]" = []
        best: "dict[str, Any]" = {self.cfg.select: -1e18}
        best_path = exp / "policies" / f"{self.cfg.name}_s{seed}.pt"

        def eval_fn(_env: Any, ac: Any) -> float:
            snapshot = copy.deepcopy(ac).cpu().eval()
            stage = ("bc_step0" if offline is not None and not curve
                     else "rl_refined" if offline is not None
                     else "rl_from_scratch")
            pt: "dict[str, Any]" = {"step": float(len(curve) * self.cfg.eval_every), "stage": stage}
            pt.update({k: round(v, 4) for k, v in self.measure(self.make_env, snapshot).items()})
            curve.append(pt)
            print(f"CURVE {self.cfg.name} s{seed} " + json.dumps(pt), flush=True)
            if pt[self.cfg.select] > best[self.cfg.select]:
                best.update(pt)
                torch.save(snapshot.state_dict(), best_path)
            return pt[self.cfg.select]

        op: OffPolicyConfig = (td3_bc_config if offline is not None else td3_config)(
            total_steps=self.cfg.total_steps, seed=seed, update_every=2,
            eval_every=self.cfg.eval_every, device=self.cfg.device, **(self.cfg.offpolicy or {}))
        print(f"\n===== {self.cfg.name} seed {seed} =====", flush=True)
        if offline is not None:
            eval_fn(None, actor)   # step-0 point: the BC warm-start floor must be in the curve/checkpoint race
        t0 = time.perf_counter()
        train_env = self.make_env()
        train_offpolicy(actor, critics, train_env, op, eval_fn=eval_fn, offline_data=offline,
                        n_envs=self.cfg.n_envs, make_env=self.make_env)
        wall = time.perf_counter() - t0
        print(f"{self.cfg.name} s{seed} DONE peak[{self.cfg.select}]={best[self.cfg.select]} ({wall:.0f}s)",
              flush=True)
        return SeedResult(seed=seed, peak=best, curve=curve, wall_s=round(wall, 1))

    def run(self, base: "str | Path" = "experiments") -> "dict[str, Any]":
        exp = experiment_dir(base, self.cfg.name)
        (exp / "policies").mkdir(exist_ok=True)
        (exp / "gifs").mkdir(exist_ok=True)
        with tee_stdout(exp / "run.log"):
            rows = [self._train_seed(s, exp) for s in self.cfg.seeds]
            peaks = sorted(r.peak[self.cfg.select] for r in rows)
            summary: "dict[str, Any]" = {
                "name": self.cfg.name, "select": self.cfg.select, "total_steps": self.cfg.total_steps,
                f"{self.cfg.select}_median": st.median(peaks), f"{self.cfg.select}_per_seed": peaks,
                "seeds": [{"seed": r.seed, "peak": r.peak, "curve": r.curve, "wall_s": r.wall_s} for r in rows],
            }
            if self.gif:
                self._render_best(rows, exp)
            (exp / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"\n{self.cfg.name}: {self.cfg.select} median={summary[f'{self.cfg.select}_median']:.3f} "
                  f"per-seed={peaks}\nartifacts -> {exp}", flush=True)
        summary["dir"] = str(exp)
        return summary

    def _render_best(self, rows: "list[SeedResult]", exp: Path) -> None:
        best = max(rows, key=lambda r: r.peak[self.cfg.select])
        try:
            from hymeko_rl.viz.campaign_viz import render_actor_gif
            build_env = self.make_env()
            try:
                actor, _ = self.build(build_env)
            finally:
                _close_env(build_env)
            actor.load_state_dict(torch.load(exp / "policies" / f"{self.cfg.name}_s{best.seed}.pt",
                                             map_location="cpu"))
            gif_env = self.make_env()
            try:
                render_actor_gif(gif_env, actor, str(exp / "gifs" / f"{self.cfg.name}_s{best.seed}"), seed=9000)
            finally:
                _close_env(gif_env)
        except Exception as exc:   # noqa: BLE001 — viz is best-effort; a GL/camera failure must not fail the run
            print(f"  [gif {self.cfg.name} skipped: {type(exc).__name__}: {exc}]", flush=True)


def compare(base_cfg: CampaignConfig, make_env: MakeEnv, builds: "dict[str, BuildPolicy]", measure: Measure, *,
            demos: "Demos | None" = None, base: "str | Path" = "experiments") -> "dict[str, Any]":
    """Run one :class:`Campaign` per architecture in ``builds`` (same env/measure/demos) and return the A/B — the
    ``select``-metric median per architecture. Each architecture keeps its own self-contained experiment dir."""
    out: "dict[str, Any]" = {}
    for arch, build in builds.items():
        out[arch] = Campaign(replace(base_cfg, name=f"{base_cfg.name}_{arch}"), make_env, build, measure,
                             demos=demos).run(base)
    sel = base_cfg.select
    verdict = {arch: out[arch][f"{sel}_median"] for arch in builds}
    print("\n=== A/B ===  " + "  ".join(f"{a}={v:.3f}" for a, v in verdict.items()), flush=True)
    return {"verdict": verdict, "runs": out}
