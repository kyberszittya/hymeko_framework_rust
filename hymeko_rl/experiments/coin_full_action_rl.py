"""Standalone full-action SAC / TD3 from the BC clone (2026-07-22, step 5).

Both algorithms improve the SAME competent standalone BC (``BC_FULL_ACTION_PHYSICAL_V1``) with the scripted base
DISABLED: ``u_exec = clip(policy(observation))`` — no grasp_carry base, no residual, no scripted prefix during the
rollout. The primary baseline is the standalone BC (NOT the scripted controller). Checkpoint selection on the NATIVE
metric (center_rate) at the declared horizon; strict + temporal recorded separately (§8).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.train.coin_full_action import collect_expert_dataset, eval_full_action, make_full_action_env

_BC_CKPT = "experiments/2026_07_22_coin_full_action/BC_FULL_ACTION_PHYSICAL_V1.pt"
_OBS, _ACT = 41, 6
_TRAIN = tuple(range(1100, 1300))
_VAL = tuple(range(1300, 1330))          # selection — disjoint from headline (panel + 1000-1099 held-out)


def _bc_state_dict():
    return torch.load(_BC_CKPT, weights_only=True)


def bc_init(algo: str):
    """Build (actor, critics) initialised from the standalone full-action BC. SAC loads the BC exactly; TD3 copies the
    shared backbone and maps the SAC ``mu`` head to the TD3 ``head``. Returns (actor, critics, delta_vs_bc)."""
    from hymeko_rl.train.sac import build_sac
    bc, _ = build_sac("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0)
    bc.load_state_dict(_bc_state_dict())
    bc.eval()
    probe = torch.as_tensor(np.random.default_rng(0).standard_normal((256, _OBS)).astype(np.float32))
    with torch.no_grad():
        y = bc.action_mean(probe)
    if algo == "SAC":
        actor, critics = build_sac("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0)
        actor.load_state_dict(bc.state_dict())
    elif algo == "TD3":
        from hymeko_rl.train.ddpg import build_offpolicy
        actor, critics = build_offpolicy("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0,
                                         n_critics=2)
        sd, bs = actor.state_dict(), bc.state_dict()
        for k in ("backbone.1.weight", "backbone.1.bias", "backbone.3.weight", "backbone.3.bias"):
            sd[k] = bs[k].clone()
        sd["head.weight"], sd["head.bias"] = bs["mu.weight"].clone(), bs["mu.bias"].clone()
        actor.load_state_dict(sd)
    else:
        raise ValueError(f"algo {algo!r} unknown; expected SAC or TD3")
    actor.eval()
    with torch.no_grad():
        delta = float((actor.action_mean(probe) - y).abs().max())
    return actor, critics, delta


def _train_env(seed_pool, *, geom: str = "POINT", horizon: int = 160):
    env = make_full_action_env(fingertip_geometry=geom, horizon=horizon)
    it = itertools.cycle(seed_pool)
    orig = env.reset

    def reset(*, seed=None):
        return orig(seed=int(next(it)) if seed is None else int(seed))
    env.reset = reset          # rotate the training seed pool when train_sac/ train_offpolicy resets without a seed
    return env


def greedy_fn(actor):
    def fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).numpy()[0]
    return fn


def run_one(algo: str, seed: int, steps: int, out: str | Path, *, eval_every: int = 10_000,
            horizon: int = 160) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    actor, critics, delta = bc_init(algo)
    print(f"[fa-rl] {algo} seed={seed} BC-init delta_vs_BC={delta:.2e} steps={steps}", flush=True)
    demos = collect_expert_dataset(_TRAIN[:120], fingertip_geometry="POINT", horizon=horizon)
    env = _train_env(_TRAIN, horizon=horizon)
    eval_env = make_full_action_env(fingertip_geometry="POINT", horizon=horizon)
    best: dict[str, Any] = {"native": -1.0, "step": 0, "metrics": None}
    hist: list[dict] = []

    def eval_fn(_te, ac) -> float:
        m = eval_full_action(greedy_fn(ac), _VAL, eval_env)
        native = float(m["center_rate"])                                  # §8: NATIVE selection
        hist.append({"center_rate": native, "strict": m["strict_count"], "auc": m["success_curve_auc"],
                     "tts_median": m["tts_median"]})
        print(f"  [eval#{len(hist)}] NATIVE center={native:.2f} strict={m['strict_count']} "
              f"auc={m['success_curve_auc']} tts={m['tts_median']}", flush=True)
        if native > best["native"]:
            best.update(native=native, step=len(hist) * eval_every, metrics=hist[-1])
            torch.save(ac.state_dict(), out / f"{algo.lower()}_actor_best.pt")
        return native

    if algo == "SAC":
        from hymeko_rl.train.sac import SACConfig, train_sac
        cfg = SACConfig.stable(total_steps=steps, seed=seed, bc_coef=1.0,
                               log_every=min(2_000, eval_every), eval_every=eval_every)
        curve = train_sac(actor, critics, env, cfg, eval_fn=eval_fn, offline_data=(demos["obs"], demos["act"]))
    else:
        from hymeko_rl.train.ddpg import td3_bc_config, train_offpolicy
        cfg = td3_bc_config(total_steps=steps, seed=seed, log_every=min(2_000, eval_every), eval_every=eval_every)
        curve = train_offpolicy(actor, critics, env, cfg, eval_fn=eval_fn, offline_data=(demos["obs"], demos["act"]))

    torch.save(actor.state_dict(), out / f"{algo.lower()}_actor_final.pt")
    (out / "run.json").write_text(json.dumps(dict(
        algo=algo, seed=seed, steps=steps, horizon=horizon, bc_init_delta=delta,
        contract="u_exec = clip(policy(obs)); scripted base DISABLED; primary baseline = standalone BC",
        selection_metric="native center_rate", n_demos=int(demos["obs"].shape[0]),
        curve=curve, best_native=best["native"], best_step=best["step"], best_metrics=best["metrics"],
        eval_history=hist), indent=1, default=float))
    print(f"[done] {algo} seed={seed} best_native={best['native']:.3f}", flush=True)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["SAC", "TD3"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_one(a.algo, a.seed, a.steps, a.out, eval_every=a.eval_every)


if __name__ == "__main__":
    main()
