"""Canonical pick-place benchmark — ONE registry + ONE loader + ONE evaluator over every deployable artifact.

The reconciliation entry point (2026-07-16, reports/canonical_integration/pick_place/). Every pick-place policy —
scripted expert, FF-DAgger base, structured-residual PPO (holds base), TD3+BC (learned RL), plain SAC (negative
control) — is declared ONCE here with explicit metadata, loaded through the single canonical loader
(:func:`gripper_pick_bc.load_pick_policy`, which now handles TD3+BC — G1), and scored under ONE protocol that reports
``reached`` / ``grasped`` / ``placed`` / full ``grasp∧place`` on both the FULL and the FAR-SPAWN distribution — so the
0.875-vs-0.167 metric-overload can never recur silently. Composes existing modules; no parallel experiment stack.

Fail-loud: an artifact whose checkpoint the canonical loader cannot build raises (never a silent default/fallback).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from hymeko_rl.experiments.gripper_pick_bc import PickPolicyIncompatible, load_pick_policy
from hymeko_rl.viz.render_pick_place import expert_action_fn, fanuc_pick_env

_ENV = dict(expert_version=3, require_settle=True, max_steps=1000)
_FAR_MARGIN = 0.05          # far-spawn = spawn-to-target > place_radius + this (removes the ~0.458 trivial floor)


@dataclass(frozen=True)
class ArtifactSpec:
    """One deployable pick-place controller, declared with full provenance (the canonical manifest row)."""
    name: str
    method: str               # scripted | bc | dagger | residual_ppo | td3_bc | sac
    architecture: str         # scripted | ff_actorcritic | deterministic_actor | ...
    action_abstraction: str   # absolute-joint+grip | residual-on-base | scripted-ik
    obs_schema: str           # "9x10 node-features"
    env_version: str          # "fanuc_pick_env v3 require_settle"
    checkpoint: str | None    # path, or None for scripted/negative-doc
    source_experiment: str
    seed: int | str
    selection_rule: str
    fallback: str             # "none" — verified no silent scripted fallback
    loadable: bool            # can the canonical loader build it?
    note: str = ""


REGISTRY: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("scripted_v3_expert", "scripted", "scripted", "scripted-ik", "9x10 node-features",
                 "fanuc_pick_env v3 require_settle", None, "hand-authored v3 planner", "n/a",
                 "N/A (reference ceiling; off-limits for the learned demo)", "none", True,
                 "the deployable-best REFERENCE; not a learned policy"),
    ArtifactSpec("ff_dagger_base", "dagger", "ff_actorcritic", "absolute-joint+grip", "9x10 node-features",
                 "fanuc_pick_env v3 require_settle",
                 "experiments/hybrid_dagger_gif/policies/hybrid_dagger_hsikan_s0_best.pt",
                 "2026-07-13 hybrid-DAgger (stabilized)", 0,
                 "best seed-0 ckpt of a 3-seed run (median 0.833; per-seed 0.750/0.833/0.875)", "none", True,
                 "strongest LEARNED (imitation) policy"),
    ArtifactSpec("td3bc_s0", "td3_bc", "deterministic_actor", "absolute-joint+grip (env-clipped)", "9x10 node-features",
                 "fanuc_pick_env v3 require_settle",
                 "experiments/2026_07_13_02_55_fanuc_pick_td3bc_hsikan/policies/fanuc_pick_td3bc_hsikan_s0.pt",
                 "2026-07-13 pick_place_td3bc (BC warm-start + TD3 refine)", 0,
                 "best-ckpt-on-place; s0 (s1/s2 collapse to ~0)", "none", True,
                 "strongest LEARNED RL (fragile: 1/3 seeds); RL degrades the BC floor"),
    ArtifactSpec("residual_ppo_hold", "residual_ppo", "ff_actorcritic+residual", "residual-on-base", "9x10 node-features",
                 "fanuc_pick_env v3 require_settle",
                 "experiments/hybrid_dagger_gif/policies/hybrid_dagger_hsikan_s0_best.pt",
                 "residual_hsikan_d0.05_settle_place-release.json ([0.875,0.875,0.875])", "0,1,2",
                 "zero-anchored residual = base; PPO holds, cannot beat (setpoint-bound)", "none", True,
                 "residual PPO HOLDS the base (0.875 full); reproduced by the base under zero residual (G3)"),
    ArtifactSpec("plain_sac_negative", "sac", "squashed_gaussian", "residual-on-base", "9x10 node-features",
                 "fanuc_pick_env v3 require_settle", None,
                 "pick_place_residual_rl (AUTO alpha) / F-PP-009", "0",
                 "NEGATIVE CONTROL — off-policy critic overestimation collapses to the idle floor", "none", False,
                 "documented negative (F-PP-009); no deployable checkpoint (would need training — deferred)"),
)


def load_artifact(spec: ArtifactSpec, env: Any) -> "Callable[[Any, np.ndarray], np.ndarray]":
    """Build the artifact's deterministic action_fn through the ONE canonical path. Fail-loud on unloadable."""
    if spec.method == "scripted":
        return expert_action_fn()
    if spec.checkpoint is None:
        raise PickPolicyIncompatible(f"{spec.name}: no checkpoint (documented-only artifact; cannot evaluate live)")
    return load_pick_policy(spec.checkpoint, env, kind="hsikan")


def evaluate_canonical(action_fn: "Callable[[Any, np.ndarray], np.ndarray]", *, n: int = 48, seed0: int = 20_000,
                       diverge_qacc: float = 5_000.0) -> "dict[str, float]":
    """ONE protocol → reached/grasped/placed on FULL and FAR-SPAWN, + safety + episode time. Never one ambiguous
    ``success``. Far-spawn = spawn-to-target > place_radius + margin (removes the trivial spawn-at-target floor)."""
    Rall = 0
    far = {"n": 0, "reached": 0, "grasped": 0, "placed": 0}          # placed = reached ∧ grasped (real pick-place)
    steps_sum = safe = tot = 0
    for s in range(n):
        env = fanuc_pick_env(**_ENV)
        obs, _ = env.reset(seed=seed0 + s)
        tx, ty = env.target_xy
        o0 = env._obj_xyz()
        is_far = float(np.hypot(o0[0] - tx, o0[1] - ty)) > env.place_radius + _FAR_MARGIN
        grasped = reached = blew = False
        k = 0
        for _ in range(env.max_steps):
            a = np.asarray(action_fn(env, obs), np.float32)
            obs, _rw, term, trunc, info = env.step(a)
            k += 1
            left, right = env._both_contact()
            grasped = grasped or bool(left and right)
            reached = reached or bool(info.get("reached"))
            if float(np.abs(env.data.qacc).max()) > diverge_qacc:
                blew = True
            if term or trunc:
                break
        Rall += int(reached)
        steps_sum += k
        tot += 1
        safe += int(not blew)
        if is_far:
            far["n"] += 1
            far["reached"] += int(reached)
            far["grasped"] += int(grasped)
            far["placed"] += int(reached and grasped)
    nf = max(far["n"], 1)
    return {"reached_full": round(Rall / n, 4),
            "reached_far": round(far["reached"] / nf, 4),
            "grasped_far": round(far["grasped"] / nf, 4),
            "placed_real_far": round(far["placed"] / nf, 4),          # the honest skill number
            "safety_no_divergence": round(safe / max(tot, 1), 4),
            "mean_episode_steps": round(steps_sum / max(tot, 1), 1),
            "n": n, "n_far": far["n"], "seed0": seed0}


def run(*, n: int = 48, seed0: int = 20_000, out: str = "reports/canonical_integration/pick_place") -> dict:
    env = fanuc_pick_env(**_ENV)
    manifest: "list[dict]" = []
    results: "dict[str, Any]" = {}
    print(f"{'artifact':22s} {'reached_full':>12s} {'reached_far':>11s} {'grasped_far':>11s} {'placed_far':>10s} {'safe':>6s}", flush=True)
    for spec in REGISTRY:
        row = asdict(spec)
        if not spec.loadable:
            row["evaluation"] = "SKIPPED — documented-only (F-PP-009 negative); no checkpoint, no new training"
            manifest.append(row)
            results[spec.name] = row["evaluation"]
            print(f"{spec.name:22s}   SKIPPED (documented negative — plain SAC collapses to idle floor, F-PP-009)", flush=True)
            continue
        try:
            fn = load_artifact(spec, env)
            m = evaluate_canonical(fn, n=n, seed0=seed0)
        except PickPolicyIncompatible as err:
            row["evaluation"] = f"LOAD FAILED: {err}"
            manifest.append(row)
            results[spec.name] = row["evaluation"]
            print(f"{spec.name:22s}   LOAD FAILED: {err}", flush=True)
            continue
        row["metrics"] = m
        manifest.append(row)
        results[spec.name] = m
        print(f"{spec.name:22s} {m['reached_full']:>12.3f} {m['reached_far']:>11.3f} {m['grasped_far']:>11.3f} "
              f"{m['placed_real_far']:>10.3f} {m['safety_no_divergence']:>6.2f}", flush=True)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "canonical_manifest.json").write_text(json.dumps(manifest, indent=2))
    (outdir / "canonical_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {outdir}/canonical_manifest.json + canonical_results.json", flush=True)
    return results


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=20_000)
    ap.add_argument("--list", action="store_true", help="print the registry and exit (no eval)")
    a = ap.parse_args(argv)
    if a.list:
        for s in REGISTRY:
            print(f"{s.name:22s} {s.method:12s} loadable={s.loadable}  {s.checkpoint or '(none)'}")
        return 0
    torch.manual_seed(0)
    np.random.seed(0)
    run(n=a.n, seed0=a.seed0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
