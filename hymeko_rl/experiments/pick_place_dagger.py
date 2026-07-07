"""FANUC pick-place via a DECLARED training strategy — reads `pick_place_dagger.hymeko`, dispatches on the
declared `algorithm`, and (for `dagger`) runs the lock-step DAgger loop.

This is the entrypoint half of "the training strategy is data": `TrainingSpec.from_hymeko` parses the declared
`algorithm ∈ {bc, td3_bc, dagger}` + its knobs, and this module binds the FANUC plant (`fanuc_pick_env`) + the
scripted expert and routes to the matching loop — `dagger` → `Dagger` (new), `td3_bc` → the `Campaign` wrapper,
`bc` → the BC-only runner (both unchanged). DAgger targets the measured cloning gap (BC place 0.625 → expert
ceiling 0.875) without a critic, so the TD3+BC value-drift collapse cannot recur.

    python -m hymeko_rl.experiments.pick_place_dagger --smoke                 # tiny path check (1 seed)
    python -m hymeko_rl.experiments.pick_place_dagger --kind hsikan --seeds 1 # 1-seed production DAgger
    python -m hymeko_rl.experiments.pick_place_dagger --kind hsikan           # 3-seed (declared) DAgger
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from gymnasium.spaces import Box

from hymeko_rl.experiments.gripper_pick_bc import build as build_bc_actor, collect_demos, eval_success
from hymeko_rl.experiments.training_spec import TrainingSpec
from hymeko_rl.train.dagger import Dagger, DaggerConfig
from hymeko_rl.viz.render_pick_place import fanuc_pick_env

_PROFILE = "data/robotics/pick_place_dagger.hymeko"
_EVAL_SEED = 20_000
_LIFT_THRESH = 0.035
# measured 2026-07-06 (reports/2026-07-06-fanuc-pick-td3bc.md), for provenance in the launch banner:
_EXPERT_CEILING = "place 0.875 / lift 0.958"
_BC_FLOOR = "place 0.625 (median)"


def _close(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _measure_factory(n_eval: int) -> "Any":
    def _measure(env: Any, actor: Any) -> "dict[str, float]":
        lift, place = eval_success(env, actor, n_eval, seed=_EVAL_SEED, lift_thresh=_LIFT_THRESH)
        return {"lift": round(lift, 4), "place": round(place, 4)}
    return _measure


def _demos(env: Any, n: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    return collect_demos(env, n, seed, only_success=True, lift_thresh=_LIFT_THRESH)


def _action_bounds(env: Any) -> "tuple[np.ndarray, np.ndarray]":
    sp = env.action_space
    assert isinstance(sp, Box)
    return np.asarray(sp.low, dtype=np.float64), np.asarray(sp.high, dtype=np.float64)


def _run_dagger(spec: TrainingSpec, *, kind: str, hidden: int, smoke: bool,
                seeds: "tuple[int, ...] | None", warm_start: "bool | None" = None,
                base: "str | Any" = "experiments") -> "dict[str, Any]":
    b, s = spec.budget, spec.strategy
    seed_list = tuple(seeds) if seeds is not None else tuple(b.get("seeds", (0,)))
    ws = bool(s.get("warm_start", 0.0)) if warm_start is None else warm_start   # override wins (tests); else declared
    dcfg = DaggerConfig(
        name=f"pick_place_dagger_{kind}", select="place",
        seeds=(seed_list[:1] if smoke else seed_list),
        n_demos=min(6, int(b.get("n_demos", 18))) if smoke else int(b.get("n_demos", 18)),
        bc_epochs=min(5, int(b.get("bc_epochs", 80))) if smoke else int(b.get("bc_epochs", 80)),
        dagger_iters=min(2, int(s.get("dagger_iters", 4))) if smoke else int(s.get("dagger_iters", 4)),
        rollouts_per_iter=min(2, int(s.get("rollouts_per_iter", 12))) if smoke else int(s.get("rollouts_per_iter", 12)),
        beta=float(s.get("beta", 0.5)), beta_decay=float(s.get("beta_decay", 0.5)),
        expert_replay_ratio=float(s.get("expert_replay_ratio", 1.0)), warm_start=ws,
        n_eval=min(3, int(s.get("n_eval", 24))) if smoke else int(s.get("n_eval", 24)),
        device="auto", profile=spec.profile)
    probe = fanuc_pick_env()
    lo, hi = _action_bounds(probe)
    _close(probe)
    dagger = Dagger(dcfg, make_env=fanuc_pick_env, build=lambda env: build_bc_actor(kind, env, hidden),
                    demos=_demos, measure=_measure_factory(dcfg.n_eval),
                    expert_fn=lambda env: env.expert_action, gif=not smoke, action_bounds=(lo, hi))
    return dagger.run(base)


def dispatch(spec: TrainingSpec, *, kind: str = "hsikan", hidden: int = 64, smoke: bool = False,
             seeds: "tuple[int, ...] | None" = None, base: "str | Any" = "experiments") -> "dict[str, Any]":
    """Route the declared strategy to its loop. `dagger` is the new loop; `bc`/`td3_bc` delegate to the
    existing runners UNCHANGED (the declarative layer selects, it does not re-implement)."""
    if spec.algorithm == "dagger":
        return _run_dagger(spec, kind=kind, hidden=hidden, smoke=smoke, seeds=seeds, base=base)
    if spec.algorithm == "td3_bc":
        from hymeko_rl.experiments.pick_place_td3bc import PickConfig
        from hymeko_rl.experiments.pick_place_td3bc import run as run_td3bc
        b = spec.budget
        return run_td3bc(PickConfig(kind=kind, smoke=smoke,
                                    seeds=tuple(seeds) if seeds is not None else b.get("seeds"),
                                    n_demos=b.get("n_demos"), bc_epochs=b.get("bc_epochs"),
                                    total_steps=int(b["total_steps"]) if "total_steps" in b else None))
    if spec.algorithm == "bc":
        from hymeko_rl.experiments.pick_place_bc import run_pick_place_bc
        b = spec.budget
        seed0 = seeds[0] if seeds else tuple(b.get("seeds", (0,)))[0]
        return run_pick_place_bc(kind, n_demos=int(b.get("n_demos", 18)),
                                 n_epochs=int(b.get("bc_epochs", 80)), seed=int(seed0), algo="bc")
    raise ValueError(f"unknown algorithm {spec.algorithm!r}")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", default="hsikan", choices=("hsikan", "mlp", "sa_hsikan"))
    ap.add_argument("--profile", default=_PROFILE)
    ap.add_argument("--smoke", action="store_true", help="tiny path check (1 seed, capped budget)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    a = ap.parse_args(argv)
    spec = TrainingSpec.from_hymeko(a.profile)
    print(f"[dagger] declared strategy: algorithm={spec.algorithm!r} profile={spec.profile}", flush=True)
    print(f"[dagger] reference (measured 2026-07-06): expert ceiling {_EXPERT_CEILING} | BC floor {_BC_FLOOR}",
          flush=True)
    summary = dispatch(spec, kind=a.kind, smoke=a.smoke,
                       seeds=tuple(a.seeds) if a.seeds is not None else None)
    print("\n=== FANUC pick-place (declared strategy) ===")
    print(f"  {a.kind} [{spec.algorithm}]: place median = {summary.get('place_median', '?')} "
          f"per-seed={summary.get('place_per_seed', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
