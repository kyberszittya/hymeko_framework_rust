"""Coin physical-contact SAC-vs-TD3 rerun (2026-07-22).

Both algorithms improve the SAME BC-initialized residual policy — ``u_exec = clip(grasp_carry + delta*tanh(policy))``
— under the CORRECTED coin↔arm-link collision model (ARM_LEGALITY 1/3). The BC init is the zero-residual head (the
faithful clone of the scripted ``grasp_carry`` base, verified to reproduce the 9/9 native / 6/9 strict ceiling); SAC
(``mu`` head) and TD3 (``head``) init to it identically (max|Δ| = 0 on a probe).

Thin composition root — reuses ``coin_two_arm_sac`` (env / demos / phase-stratified replay / eval / disjoint seed
splits + reward certify) and ``train.sac`` / ``train.ddpg`` and ``coin_delivery_rl``. NO new trainer / env / replay /
rollout. §8 discipline: **checkpoint selection is on the NATIVE metric only** (``zone_rate``); the strict certified
count is recorded alongside for the report, never used to pick the checkpoint (a metric a failure can inflate must not
drive selection).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from hymeko_rl.experiments.coin_two_arm_sac import (
    _DEMO_SEEDS,
    _PROGRESS_MIN,
    _TRAIN_SEEDS,
    _VAL_SEEDS,
    certify_or_abort,
    collect_demos,
    direct_env,
    evaluate,
    stratify_seed,
)


def bc_init_zero_residual(actor: Any) -> str:
    """Zero the deterministic action head so the policy starts AT the scripted ``grasp_carry`` (zero residual) —
    the faithful BC of the scripted base. Handles both the SAC (``mu``) and TD3 (``head``) architectures; the
    resulting behaviour is identical (residual = 0) which is what makes the SAC/TD3 init behaviour-equivalent.

    # Preconditions ``actor`` exposes exactly one of ``mu`` / ``head`` / ``actor_mean`` with ``weight`` and ``bias``.
    # Postconditions that head's weight and bias are 0 → ``action_mean(·) = 0`` everywhere.
    # Errors ``AttributeError`` if no known head is present (never a silent no-op)."""
    for name in ("mu", "head", "actor_mean"):
        head = getattr(actor, name, None)
        if head is not None and hasattr(head, "weight"):
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
            return name
    raise AttributeError(f"actor {type(actor).__name__} exposes no known residual head (mu/head/actor_mean)")


def _competence_bc_coef_fn(comp: dict[str, Any]):
    """Milestone-driven BC anchor (identical schedule to ``coin_two_arm_sac``): anchor hard until the policy makes
    progress, loosen as strict deliveries accumulate. NOT a step-decay (competence-gated)."""
    def fn(_step: int) -> float:
        if comp["consec_strict"] >= 3:
            return 0.05
        if comp["first_strict"]:
            return 0.1
        if comp["progress_ok"]:
            return 0.3
        return 1.0
    return fn


def run_one(algo: str, seed: int, steps: int, out: str | Path, *, eval_every: int = 5_000) -> dict[str, Any]:
    """Train one (algo, seed) from the BC zero-residual init; select the best checkpoint on NATIVE ``zone_rate``;
    record the strict certified count separately. Returns the best record."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    certify_or_abort()
    env = direct_env(train_seed_pool=_TRAIN_SEEDS)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    demos = collect_demos(env, _DEMO_SEEDS + _VAL_SEEDS[:4])
    seed_replay = stratify_seed(demos)
    eval_env = direct_env()
    comp = {"progress_ok": False, "first_strict": False, "consec_strict": 0}
    bc_fn = _competence_bc_coef_fn(comp)
    best: dict[str, Any] = {"native": -1.0, "step": 0, "metrics": None}
    hist: list[dict] = []

    def eval_fn(_train_env: Any, ac: Any) -> float:
        m = evaluate(eval_env, ac, _VAL_SEEDS)
        if m["mean_progress"] >= _PROGRESS_MIN:
            comp["progress_ok"] = True
        if m["strict_count"] >= 1:
            comp["first_strict"] = True
            comp["consec_strict"] += 1
        else:
            comp["consec_strict"] = 0
        m.update(bc_coef=bc_fn(0), consec_strict=comp["consec_strict"])
        hist.append(m)
        native = float(m["zone_rate"])                                   # §8: NATIVE metric drives selection
        print(f"  [eval#{len(hist)}] NATIVE zone={native:.2f} | strict={m['strict_rate']:.2f}({m['strict_count']}) "
              f"prog={m['mean_progress']:.4f} both={m['both_frac']:.2f} L/R={m['lc']:.2f}/{m['rc']:.2f}", flush=True)
        if native > best["native"]:
            best.update(native=native, step=len(hist) * eval_every, metrics=m)
            torch.save(ac.state_dict(), out / f"{algo.lower()}_actor_best.pt")
        return native

    if algo == "SAC":
        from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
        actor, critics = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim, action_dim=act_dim, action_scale=1.0)
        head = bc_init_zero_residual(actor)
        cfg = SACConfig.stable(total_steps=steps, seed=seed, bc_coef=1.0,
                               log_every=min(1_000, eval_every), eval_every=eval_every)
        print(f"[rerun] SAC seed={seed} steps={steps} BC-init(zero {head}) bc_coef competence-gated | "
              f"demos={len(demos[0])} seeded={len(seed_replay[0])}", flush=True)
        curve = train_sac(actor, critics, env, cfg, eval_fn=eval_fn,
                          offline_data=(demos[0], demos[1]), init_transitions=seed_replay, bc_coef_fn=bc_fn)
    elif algo == "TD3":
        from hymeko_rl.train.ddpg import build_offpolicy, td3_bc_config, train_offpolicy
        actor, critics = build_offpolicy("mlp", obs_dim=obs_dim, flat_dim=obs_dim, action_dim=act_dim,
                                         action_scale=1.0, n_critics=2)
        head = bc_init_zero_residual(actor)
        cfg = td3_bc_config(total_steps=steps, seed=seed,
                            log_every=min(1_000, eval_every), eval_every=eval_every)
        print(f"[rerun] TD3 seed={seed} steps={steps} BC-init(zero {head}) td3+bc anchor (warm_start) | "
              f"demos={len(demos[0])}", flush=True)
        curve = train_offpolicy(actor, critics, env, cfg, eval_fn=eval_fn, offline_data=(demos[0], demos[1]))
    else:
        raise ValueError(f"algo {algo!r} unknown; expected SAC or TD3")

    torch.save(actor.state_dict(), out / f"{algo.lower()}_actor_final.pt")
    (out / "run.json").write_text(json.dumps(dict(
        algo=algo, seed=seed, steps=steps, obs_dim=obs_dim, act_dim=act_dim,
        physics="corrected coin<->arm-link ARM_LEGALITY 1/3", bc_init="zero-residual head (scripted grasp_carry)",
        selection_metric="native zone_rate", n_demos=int(len(demos[0])), n_seeded=int(len(seed_replay[0])),
        curve=curve, best_native=best["native"], best_step=best["step"], best_metrics=best["metrics"],
        eval_history=hist), indent=1, default=float))
    print(f"[done] {algo} seed={seed} best_native={best['native']:.3f} @step{best['step']} "
          f"(strict at best = {(best['metrics'] or {}).get('strict_count')})", flush=True)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["SAC", "TD3"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_one(a.algo, a.seed, a.steps, a.out, eval_every=a.eval_every)


if __name__ == "__main__":
    main()
