"""COIN STAGE-2 n-step transport experiment — matched CONTROL (n_step=1) vs TREATMENT (n_step=3) continuation from the
best CURRICULUM checkpoint, training on the frozen STAGE-2 corpus with the committed 70/15/15 mix. ONLY variable:
n_step. Actor/critic/SAC/reward/strict predicate/obs/action/BC anchor/replay ratios/generator/STAGE-2 corpus unchanged.
Reuses coin_clearance_curriculum + coin_generator_exp + the canonical restore/rollout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.coin_delivery.provenance.state_identity import snapshot_hash
from hymeko_rl.env.planar_snapshot import snapshot_planar
from hymeko_rl.experiments.coin_clearance_curriculum import _CURDIR, _GENDIR, _STAGES, _eval_pool, _stage_metrics
from hymeko_rl.experiments.coin_generator_exp import _greedy, _restore_generated, direct_env
from hymeko_rl.experiments.coin_problem_generator import load_configs
from hymeko_rl.experiments.coin_two_arm_sac import policy_strict
from hymeko_rl.train.coin_delivery_actor import rollout
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac

_CKPT = Path("experiments/2026_07_21_coin_clearance_curriculum/run_s0/actor_best.pt")
_STRONG_HASH = "04870b0e0357ecb5"                                # the STAGE1 +0.0253 retention state
_FIXED_TRAIN = tuple(range(64_000, 64_056))


def _find_strong(held_all):
    env = direct_env()
    for c in held_all:
        _restore_generated(env, c.snapshot)
        if snapshot_hash(snapshot_planar(env.inner)).startswith(_STRONG_HASH[:12]):
            return c
    return None


def run(n_step: int, seed: int, rep: int, steps: int, out: Path) -> dict:
    s2_train = load_configs(_CURDIR / "STAGE2_train.pkl")
    earlier = load_configs(_CURDIR / "STAGE0_train.pkl") + load_configs(_CURDIR / "STAGE1_train.pkl")
    orig_cert = [c for c in load_configs(_GENDIR / "train_configs.pkl") if c.family == "CERTIFIED_NEIGHBORHOOD"]
    held = {s: load_configs(_CURDIR / f"{s}_held.pkl") for s in _STAGES}
    strong = _find_strong(held["STAGE1"])

    run_seed = seed * 100 + rep
    env = direct_env()
    env._base_override = lambda inner, t: np.zeros(env.action_space.shape[0], np.float32)
    env._delta_override = 1.0
    rng = np.random.default_rng(run_seed)
    _orig = env.reset

    def _reset(*, seed=None):                                    # committed 70/15/15: STAGE2 / earlier / orig-cert
        if seed is not None:
            return _orig(seed=seed)
        u = rng.random()
        pool = s2_train if u < 0.70 else (earlier if u < 0.85 else orig_cert)
        _restore_generated(env, pool[rng.integers(len(pool))].snapshot)
        return env._last_obs, {}
    env.reset = _reset

    actor, critics = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    actor.load_state_dict(torch.load(_CKPT, map_location="cpu"))
    cfg = SACConfig.stable(total_steps=steps, seed=run_seed, bc_coef=1.0, log_every=2500, eval_every=2500, n_step=n_step)
    comp = {"progress_ok": False, "first_strict": False, "consec_strict": 0}

    def bc_coef_fn(_s):
        return 0.05 if comp["consec_strict"] >= 3 else 0.1 if comp["first_strict"] else 0.3 if comp["progress_ok"] else 1.0

    eval_env = direct_env()
    eval_env._base_override = lambda inner, t: np.zeros(6, np.float32)
    eval_env._delta_override = 1.0
    hist, best = [], {"key": (-1, -9.9, -1, -1), "step": 0, "m": None}
    arm = f"n{n_step} s{seed}r{rep}"

    def _strong_strict(ac):
        _restore_generated(eval_env, strong.snapshot)
        return bool(policy_strict(rollout(eval_env, _greedy(ac), max_steps=60)))

    def eval_fn(_e, ac):
        s1 = _stage_metrics(_eval_pool(eval_env, ac, held["STAGE1"]))     # retention
        s2 = _stage_metrics(_eval_pool(eval_env, ac, held["STAGE2"]))     # transport target
        eval_env.reset(seed=64_102)
        r64102 = bool(policy_strict(rollout(eval_env, _greedy(ac), max_steps=60)))
        strong_ok = _strong_strict(ac)
        m = dict(eval=len(hist) + 1, step=(len(hist) + 1) * 2500, stage1=s1, stage2=s2,
                 s64102_strict=r64102, strong_strict=strong_ok, bc_coef=bc_coef_fn(0))
        if s2["mean_progress"] >= 0.02 or s2["strict"] >= 1:
            comp["progress_ok"] = True
        comp["consec_strict"] = comp["consec_strict"] + 1 if (s1["strict"] >= 1 or s2["strict"] >= 1) else 0
        if s1["strict"] >= 1 or s2["strict"] >= 1:
            comp["first_strict"] = True
        hist.append(m)
        # §9 checkpoint priority: STAGE2 cov, max STAGE2 cert clearance, STAGE1 retention, 64102 retention
        key = (s2["coverage"], s2["max_certified_clearance"], s1["coverage"], int(r64102) + int(strong_ok))
        if key > best["key"]:
            best.update(key=key, step=m["step"], m=m)
            torch.save(ac.state_dict(), out / "actor_best.pt")
        print(f"  [{arm} ev#{len(hist)}] S2:cov{s2['coverage']}/{s2['n']} L{s2['loose']} mc{s2['max_certified_clearance']:+.2f} "
              f"prog{s2['mean_progress']:.3f} | S1ret:cov{s1['coverage']} strong={strong_ok} 64102={r64102}", flush=True)
        return float(s2["coverage"] + s2["loose"] * 0.1 + s1["coverage"] * 0.01)

    out.mkdir(parents=True, exist_ok=True)
    curve = train_sac(actor, critics, env, cfg, eval_fn=eval_fn, bc_coef_fn=bc_coef_fn)
    torch.save(actor.state_dict(), out / "actor_final.pt")
    result = dict(n_step=n_step, seed=seed, rep=rep, steps=steps, source_checkpoint=str(_CKPT),
                  strong_state_hash=_STRONG_HASH, curve=curve, best_step=best["step"], best_metrics=best["m"],
                  eval_history=hist)
    (out / "run.json").write_text(json.dumps(result, indent=1, default=float))
    bm = best["m"]
    print(f"[{arm}] done | best S2 cov={bm['stage2']['coverage'] if bm else 0} "
          f"maxCertClr={bm['stage2']['max_certified_clearance'] if bm else -9.9:+.3f} "
          f"S1ret={bm['stage1']['coverage'] if bm else 0} strong={bm['strong_strict'] if bm else False}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nstep", type=int, choices=[1, 3], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.nstep, a.seed, a.rep, a.steps, Path(a.out))


if __name__ == "__main__":
    main()
