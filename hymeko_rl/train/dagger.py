"""Lock-step DAgger (Ross, Gordon & Bagnell 2011) — push past the BC ceiling in the imitation regime.

DAgger fixes behaviour cloning's covariate shift: BC trains on the *expert's* state distribution and fails on
the states the *learner* drifts into. DAgger rolls the learner, queries the expert on those learner-visited
states, aggregates the new (state, expert-action) labels, and re-clones. Crucially it introduces **no critic /
no Q**, so the value-drift that collapsed TD3+BC on this task (reports/2026-07-06-fanuc-pick-td3bc.md) cannot
recur — DAgger converges toward the expert's own ceiling, not past it.

**Lock-step labelling.** The FANUC pick expert (`PickPlaceEnv.expert_action`) is not purely stateless — it
latches `_lift_xy` (the grasp point to lift from) once per commit and resets it when not committed. So the
expert MUST be queried on *every* learner step, in lock-step, for the latch to track the rollout coherently;
sparse labelling would return stale/history-torn labels. A `label_sanity` gate verifies this before any run.

Composes with (does not fork) the `Campaign` scaffolding: reuses `behaviour_clone`, `experiment_dir`,
`tee_stdout`, `render_actor_gif`, and an `eval_success`-style measure. Evidence-complete (per
`feedback-evidence-complete-learned-policy`): every round's checkpoint + config + seed + curve + aggregate
sizes + a gif are saved into one `experiment_dir`.
"""
from __future__ import annotations

import copy
import json
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from hymeko_rl.eval.evaluate import experiment_dir, greedy_action_fn
from hymeko_rl.train.bc import behaviour_clone
from hymeko_rl.train.campaign import tee_stdout

MakeEnv = Callable[[], Any]
BuildActor = Callable[[Any], Any]                      # (env) -> ActorCritic (imitation policy; no critics)
Demos = Callable[[Any, int, int], "tuple[np.ndarray, np.ndarray]"]
Measure = Callable[[Any, Any], "dict[str, float]"]     # (env, actor) -> {metric: value}
ExpertFn = Callable[[Any], np.ndarray]                 # (env) -> expert action at the CURRENT state


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


@dataclass(frozen=True)
class DaggerConfig:
    """DAgger hyperparameters. ``select`` is the metric best-checkpoint maximises. ``expert_replay_ratio`` is the
    integer oversample factor for the original expert demos in each round's aggregated retrain set."""

    name: str
    select: str = "place"
    seeds: "tuple[int, ...]" = (0,)
    n_demos: int = 18
    bc_epochs: int = 80
    dagger_iters: int = 4
    rollouts_per_iter: int = 12
    beta: float = 0.5
    beta_decay: float = 0.5
    expert_replay_ratio: float = 1.0
    aggregate_cap_rounds: int = 0      # 0 = keep ALL relabels (textbook DAgger); >0 = keep only D0 + the last K
                                       # rounds' relabels (recency cap — stops the growing aggregate from letting
                                       # a stale/bad round poison later re-BCs)
    warm_start: bool = False           # True: continue re-BC from the prior round's weights (smoother, less D1
                                       # dip); False: fresh policy each round (textbook DAgger, higher variance)
    n_eval: int = 8
    bc_batch: int = 128
    device: str = "auto"
    profile: "str | None" = None


def label_sanity(env: Any, expert_fn: ExpertFn, *, n_episodes: int = 2,
                 action_lo: "np.ndarray | None" = None, action_hi: "np.ndarray | None" = None,
                 ) -> "dict[str, Any]":
    """Verify the lock-step expert labels are usable BEFORE a production run. Rolls with the **expert acting**
    (act = the queried label) so the trajectory reaches the commit/lift phases — the only phases where the
    ``_lift_xy`` latch is live and thus the only place its coherence can be checked. Because the latch logic is
    a function of the current state queried every step (not of *whose* action drives the sim), an expert roll is
    a valid pre-flight for the coherence the learner rollouts will rely on. Checks:
      - every label finite; ``deterministic``: re-querying the SAME state returns the SAME label;
      - ``latch_coherent``: whenever committed, ``_lift_xy`` is a finite pair (never stale/torn);
      - ``in_bounds``: labels overshoot the action box by ≤ one action-range (the env CLIPS in ``step``, and
        the existing BC clones these same raw IK actions — a strict box check is wrong; garbage still fails);
      - ``committed_steps > 0``: the roll actually reached the latched phase (else the latch went untested).

    # Postconditions returns a report dict with per-check bools + ``max_over_box_ranges`` + ``ok``."""
    finite = deterministic = latch_coherent = True
    has_latch = False              # does this env expose a latch at all? a STATELESS expert (e.g. PD-hold) has none
    n_steps = committed_steps = 0
    max_over = 0.0
    for ep in range(n_episodes):
        env.reset(seed=90_000 + ep)
        done = False
        while not done:
            label = np.asarray(expert_fn(env), dtype=np.float64)
            relabel = np.asarray(expert_fn(env), dtype=np.float64)     # re-query the SAME state
            deterministic = deterministic and bool(np.allclose(label, relabel, atol=1e-6))
            finite = finite and bool(np.all(np.isfinite(label)))
            if action_lo is not None and action_hi is not None:
                rng = np.maximum(action_hi - action_lo, 1e-9)
                over = np.maximum(0.0, np.maximum(action_lo - label, label - action_hi) / rng)
                max_over = max(max_over, float(np.max(over)))
            lx = getattr(env, "_lift_xy", "absent")
            if lx != "absent":
                has_latch = True
                if lx is not None:
                    committed_steps += 1
                    latch_coherent = latch_coherent and bool(np.all(np.isfinite(np.asarray(lx, dtype=np.float64))))
            _obs, _r, term, trunc, _info = env.step(label)             # act with the EXPERT (reach commit/lift)
            n_steps += 1
            done = bool(term or trunc)
    in_bounds = max_over <= 1.0
    # A latched expert must actually REACH the latched phase (committed_steps>0); a stateless expert has no latch
    # to reach, so that requirement is vacuous for it (the finite/deterministic/in_bounds checks still gate).
    latch_reached = committed_steps > 0 if has_latch else True
    return {"finite": finite, "in_bounds": in_bounds, "deterministic": deterministic,
            "latch_coherent": latch_coherent, "has_latch": has_latch, "max_over_box_ranges": round(max_over, 3),
            "n_steps": n_steps, "committed_steps": committed_steps,
            "ok": bool(finite and in_bounds and deterministic and latch_coherent and latch_reached)}


class Dagger:
    """A multi-seed lock-step DAgger run for one imitation architecture, evidence-complete and best-checkpointed.

    # Preconditions ``build(make_env())`` returns an ActorCritic with ``action_mean``; ``measure`` returns a dict
      containing ``cfg.select``; ``expert_fn(env)`` returns the expert action at the env's current state.
    # Postconditions ``run`` writes results.json + per-round policies + a gif + run.log into one experiment_dir.
    """

    def __init__(self, cfg: DaggerConfig, *, make_env: MakeEnv, build: BuildActor, demos: Demos,
                 measure: Measure, expert_fn: ExpertFn, gif: bool = True,
                 action_bounds: "tuple[np.ndarray, np.ndarray] | None" = None) -> None:
        self.cfg = cfg
        self.make_env = make_env
        self.build = build
        self.demos = demos
        self.measure = measure
        self.expert_fn = expert_fn
        self.gif = gif
        self.action_bounds = action_bounds

    def _rollout_with_expert_labels(self, env: Any, learner: Any, beta: float, rng: "np.random.Generator",
                                    ) -> "tuple[np.ndarray, np.ndarray]":
        """Roll ``learner`` for ``rollouts_per_iter`` episodes; at EVERY step query the expert (lock-step) as the
        label; act with the expert w.p. ``beta`` else the learner. Returns expert-labelled ``(obs, acts)``."""
        act = greedy_action_fn(learner)
        obs_all: "list[np.ndarray]" = []
        act_all: "list[np.ndarray]" = []
        for _ep in range(self.cfg.rollouts_per_iter):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done = False
            while not done:
                label = np.asarray(self.expert_fn(env), dtype=np.float32)   # lock-step label (advances _lift_xy)
                obs_all.append(obs)
                act_all.append(label)
                a = label if rng.random() < beta else act(env, obs)
                obs, _r, term, trunc, _info = env.step(a)
                done = bool(term or trunc)
        return np.asarray(obs_all, dtype=np.float32), np.asarray(act_all, dtype=np.float32)

    def _eval_save(self, actor: Any, stage: str, round_idx: int, agg_size: int, seed: int,
                   curve: "list[dict[str, Any]]", best: "dict[str, Any]", pol_dir: Path) -> None:
        """Eval a CPU snapshot, log a CURVE line, save this round's checkpoint (evidence-complete), and race
        best-checkpoint on ``cfg.select`` (so a later-round regression can never lose the best policy)."""
        snap = copy.deepcopy(actor).cpu().eval()
        eval_env = self.make_env()
        try:
            metrics = self.measure(eval_env, snap)
        finally:
            _close_env(eval_env)
        pt = {"round": round_idx, "stage": stage, "agg_size": agg_size,
              **{k: round(v, 4) for k, v in metrics.items()}}
        curve.append(pt)
        print(f"CURVE {self.cfg.name} s{seed} " + json.dumps(pt), flush=True)
        torch.save(snap.state_dict(), pol_dir / f"{self.cfg.name}_s{seed}_{stage}.pt")
        if pt[self.cfg.select] > best[self.cfg.select]:
            best.update(pt)
            torch.save(snap.state_dict(), pol_dir / f"{self.cfg.name}_s{seed}_best.pt")

    def _train_seed(self, seed: int, exp: Path) -> "dict[str, Any]":
        cfg = self.cfg
        pol_dir = exp / "policies"
        demo_env = self.make_env()
        try:
            e_obs, e_acts = self.demos(demo_env, cfg.n_demos, seed)     # D0 = expert demos (the replay pool)
        finally:
            _close_env(demo_env)
        d_obs, d_acts = e_obs.copy(), e_acts.copy()                    # the aggregated set (D0 at bc0)
        relabels: "list[tuple[np.ndarray, np.ndarray]]" = []           # per-round expert relabels (for the recency cap)
        rng = np.random.default_rng(seed)
        curve: "list[dict[str, Any]]" = []
        best: "dict[str, Any]" = {cfg.select: -1e18}

        torch.manual_seed(seed)
        np.random.seed(seed)
        build_env = self.make_env()
        try:
            actor = self.build(build_env)
        finally:
            _close_env(build_env)
        print(f"\n===== {cfg.name} seed {seed} (D0={len(d_obs)}) =====", flush=True)
        behaviour_clone(actor, d_obs, d_acts, n_epochs=cfg.bc_epochs, seed=seed,
                        batch=cfg.bc_batch, device=cfg.device)
        self._eval_save(actor, "bc0", 0, len(d_obs), seed, curve, best, pol_dir)

        reps = max(1, int(round(cfg.expert_replay_ratio)))
        for r in range(1, cfg.dagger_iters + 1):
            beta_r = cfg.beta * (cfg.beta_decay ** (r - 1))
            roll_env = self.make_env()
            try:
                new_obs, new_acts = self._rollout_with_expert_labels(roll_env, actor, beta_r, rng)
            finally:
                _close_env(roll_env)
            relabels.append((new_obs, new_acts))
            kept = relabels[-cfg.aggregate_cap_rounds:] if cfg.aggregate_cap_rounds > 0 else relabels
            d_obs = np.concatenate([e_obs, *[o for o, _ in kept]], axis=0)   # D0 anchor + (capped) relabels
            d_acts = np.concatenate([e_acts, *[a for _, a in kept]], axis=0)
            agg_obs = np.concatenate([d_obs, *([e_obs] * (reps - 1))], axis=0) if reps > 1 else d_obs
            agg_acts = np.concatenate([d_acts, *([e_acts] * (reps - 1))], axis=0) if reps > 1 else d_acts
            print(f"[dagger] round {r}/{cfg.dagger_iters} beta={beta_r:.3f} "
                  f"agg={len(agg_obs)} (+{len(new_obs)} new labels) "
                  f"{'warm' if cfg.warm_start else 'fresh'}-reBC", flush=True)
            if not cfg.warm_start:                                     # fresh policy each round (textbook DAgger)
                torch.manual_seed(seed + r)
                rebuild_env = self.make_env()
                try:
                    actor = self.build(rebuild_env)
                finally:
                    _close_env(rebuild_env)
            # warm_start: keep `actor` and continue BC from its weights (a fresh Adam, prior params retained)
            behaviour_clone(actor, agg_obs, agg_acts, n_epochs=cfg.bc_epochs, seed=seed + r,
                            batch=cfg.bc_batch, device=cfg.device)
            self._eval_save(actor, f"d{r}", r, len(agg_obs), seed, curve, best, pol_dir)

        torch.save(copy.deepcopy(actor).cpu().eval().state_dict(), pol_dir / f"{cfg.name}_s{seed}_final.pt")
        return {"seed": seed, "peak": best, "final": curve[-1], "curve": curve}

    def _render_best(self, rows: "list[dict[str, Any]]", exp: Path) -> None:
        best = max(rows, key=lambda r: r["peak"][self.cfg.select])
        try:
            from hymeko_rl.viz.campaign_viz import render_actor_gif
            build_env = self.make_env()
            try:
                actor = self.build(build_env)
            finally:
                _close_env(build_env)
            actor.load_state_dict(torch.load(exp / "policies" / f"{self.cfg.name}_s{best['seed']}_best.pt",
                                             map_location="cpu"))
            gif_env = self.make_env()
            try:
                render_actor_gif(gif_env, actor, str(exp / "gifs" / f"{self.cfg.name}_s{best['seed']}"), seed=9_000)
            finally:
                _close_env(gif_env)
        except Exception as exc:   # noqa: BLE001 — viz is best-effort; a GL/camera failure must not fail the run
            print(f"  [gif {self.cfg.name} skipped: {type(exc).__name__}: {exc}]", flush=True)

    def _run_sanity_gate(self) -> "dict[str, Any]":
        probe_env = self.make_env()
        try:
            lo, hi = self.action_bounds if self.action_bounds is not None else (None, None)
            report = label_sanity(probe_env, self.expert_fn, n_episodes=2, action_lo=lo, action_hi=hi)
        finally:
            _close_env(probe_env)
        print(f"[label-sanity] {json.dumps(report)}", flush=True)
        if not report["ok"]:
            raise ValueError(f"label-sanity gate FAILED (lock-step labels not usable): {report}")
        return report

    def run(self, base: "str | Path" = "experiments") -> "dict[str, Any]":
        cfg = self.cfg
        exp = experiment_dir(base, cfg.name)
        (exp / "policies").mkdir(exist_ok=True)
        (exp / "gifs").mkdir(exist_ok=True)
        with tee_stdout(exp / "run.log"):
            sanity = self._run_sanity_gate()
            rows = [self._train_seed(s, exp) for s in cfg.seeds]
            peaks = sorted(r["peak"][cfg.select] for r in rows)
            finals = sorted(r["final"][cfg.select] for r in rows)
            summary: "dict[str, Any]" = {
                "name": cfg.name, "algorithm": "dagger", "select": cfg.select, "profile": cfg.profile,
                "label_sanity": sanity,
                f"{cfg.select}_median": st.median(peaks), f"{cfg.select}_per_seed": peaks,
                f"final_{cfg.select}_median": st.median(finals),
                "budget": {"seeds": list(cfg.seeds), "n_demos": cfg.n_demos, "bc_epochs": cfg.bc_epochs,
                           "dagger_iters": cfg.dagger_iters, "rollouts_per_iter": cfg.rollouts_per_iter,
                           "beta": cfg.beta, "beta_decay": cfg.beta_decay,
                           "expert_replay_ratio": cfg.expert_replay_ratio, "n_eval": cfg.n_eval},
                "seeds": rows}
            if self.gif:
                self._render_best(rows, exp)
            (exp / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"\n{cfg.name}: {cfg.select} median={summary[f'{cfg.select}_median']:.3f} "
                  f"per-seed={peaks}\nartifacts -> {exp}", flush=True)
        summary["dir"] = str(exp)
        return summary
