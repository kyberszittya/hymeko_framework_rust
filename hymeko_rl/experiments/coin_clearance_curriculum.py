"""COIN clearance-transport curriculum (§7): ONE continuous SAC from the best GENERATOR checkpoint, trained through 4
progressive footprint-clearance stages (coin moved visibly OUTSIDE the target). Only variable = the generated
training-config distribution. Actor/critic/SAC/reward/strict predicate/obs/action/BC anchor/replay sampler unchanged;
no hold shaping, no new reward, no n-step. Reuses coin_generator_exp + the canonical restore/rollout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.experiments.coin_generator_exp import _greedy, _restore_generated, direct_env
from hymeko_rl.experiments.coin_problem_generator import CURRICULUM_STAGES, load_configs
from hymeko_rl.experiments.coin_two_arm_sac import policy_strict
from hymeko_rl.train.coin_delivery_actor import _attribution_from_trace, rollout
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac

_FOOT = 0.06
_STAGES = list(CURRICULUM_STAGES)                                # STAGE0..STAGE3
_CURDIR = Path("experiments/2026_07_21_coin_clearance_curriculum")
_GENDIR = Path("experiments/2026_07_20_coin_problem_generator")
_CKPT = Path("experiments/2026_07_20_coin_generator_generator_s2r0/actor_best.pt")
_STAGE_STEPS, _EVAL_EVERY = 25_000, 2_500


def _clearance(inner) -> float:
    return float(inner.planar_metrics.disk_to_zone) - _FOOT


def _eval_pool(env, actor, cfgs):
    rows = []
    for c in cfgs:
        _restore_generated(env, c.snapshot)
        clr = _clearance(env.inner)
        tr = rollout(env, _greedy(actor), max_steps=60)
        att = _attribution_from_trace(tr)
        ff = att.fingertip_fraction
        clean = (min(att.alpha_L, att.alpha_R) / (ff + 1e-9)) >= 0.15
        rows.append(dict(clr=clr, strict=policy_strict(tr), loose=tr.loose, attribution=ff, clean=clean,
                         bilateral=tr.both_frac > 0, body=att.alpha_body, dwell=tr.best_dwell, settle=tr.settle_vel,
                         progress=tr.progress))
    return rows


def _stage_metrics(rows):
    z = [r for r in rows if r["loose"]]
    nz = max(1, len(z))
    strict = [r for r in rows if r["strict"]]
    return dict(n=len(rows), strict=len(strict), coverage=len(strict),
                loose=sum(r["loose"] for r in rows), loose_rate=round(sum(r["loose"] for r in rows) / max(1, len(rows)), 3),
                mean_progress=round(float(np.mean([r["progress"] for r in rows])), 4),
                max_certified_clearance=round(max([r["clr"] for r in strict], default=-9.9), 4),
                P_attr=round(sum(r["attribution"] >= 0.60 for r in z) / nz, 3),
                P_clean=round(sum(r["clean"] for r in z) / nz, 3),
                P_bilat=round(sum(r["bilateral"] for r in z) / nz, 3))


def run(seed: int, out: Path) -> dict:
    train_pools = {s: load_configs(_CURDIR / f"{s}_train.pkl") for s in _STAGES}
    held_pools = {s: load_configs(_CURDIR / f"{s}_held.pkl") for s in _STAGES}
    orig_cert = [c for c in load_configs(_GENDIR / "train_configs.pkl") if c.family == "CERTIFIED_NEIGHBORHOOD"]

    env = direct_env()
    env._base_override = lambda inner, t: np.zeros(env.action_space.shape[0], np.float32)
    env._delta_override = 1.0
    rng = np.random.default_rng(seed)
    curr = {"idx": 0, "stage_start_eval": 0}
    _orig = env.reset

    def _reset(*, seed=None):                                    # §6 mix: 70% current stage / 15% earlier / 15% orig-cert
        if seed is not None:
            return _orig(seed=seed)
        u = rng.random()
        si = curr["idx"]
        if u < 0.70 or si == 0:
            pool = train_pools[_STAGES[si]]
        elif u < 0.85:
            pool = train_pools[_STAGES[rng.integers(0, si)]]
        else:
            pool = orig_cert
        _restore_generated(env, pool[rng.integers(len(pool))].snapshot)
        return env._last_obs, {"stage": _STAGES[si]}
    env.reset = _reset

    actor, critics = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    actor.load_state_dict(torch.load(_CKPT, map_location="cpu"))
    cfg = SACConfig.stable(total_steps=len(_STAGES) * _STAGE_STEPS, seed=seed, bc_coef=1.0,
                           log_every=2500, eval_every=_EVAL_EVERY)
    comp = {"progress_ok": False, "first_strict": False, "consec_strict": 0}

    def bc_coef_fn(_s):
        return 0.05 if comp["consec_strict"] >= 3 else 0.1 if comp["first_strict"] else 0.3 if comp["progress_ok"] else 1.0

    eval_env = direct_env()
    eval_env._base_override = lambda inner, t: np.zeros(6, np.float32)
    eval_env._delta_override = 1.0
    hist, best = [], {"key": (-9.9, -1), "step": 0, "m": None}

    def eval_fn(_e, ac):
        evn = len(hist)
        # evaluate current stage + done stages + the next stage (to detect look-ahead generalization)
        upto = min(curr["idx"] + 2, len(_STAGES))
        by_stage = {s: _stage_metrics(_eval_pool(eval_env, ac, held_pools[s])) for s in _STAGES[:upto]}
        # 64102 retention
        eval_env.reset(seed=64_102)
        r64102 = policy_strict(rollout(eval_env, _greedy(ac), max_steps=60))
        cur = by_stage[_STAGES[curr["idx"]]]
        m = dict(eval=evn + 1, step=(evn + 1) * _EVAL_EVERY, active_stage=_STAGES[curr["idx"]],
                 by_stage={s: v for s, v in by_stage.items()}, s64102_strict=bool(r64102),
                 bc_coef=bc_coef_fn(0))
        # global best clear-start: max certified clearance across all evaluated stages
        max_cert = max((v["max_certified_clearance"] for v in by_stage.values()), default=-9.9)
        m["max_certified_clearance"] = round(float(max_cert), 4)
        if cur["mean_progress"] >= 0.02 or max_cert > 0:
            comp["progress_ok"] = True
        comp["consec_strict"] = comp["consec_strict"] + 1 if cur["strict"] >= 1 else 0
        if cur["strict"] >= 1:
            comp["first_strict"] = True
        hist.append(m)
        # checkpoint priority: (max certified clearance, coverage at that band)
        key = (max_cert, sum(v["coverage"] for v in by_stage.values()))
        if key > best["key"]:
            best.update(key=key, step=m["step"], m=m)
            torch.save(ac.state_dict(), out / "actor_best.pt")
        # §7 stage advancement: >=25% loose AND >=2 certified on the current held stage, OR full stage budget elapsed
        advanced = ""
        elapsed = (evn + 1 - curr["stage_start_eval"]) * _EVAL_EVERY
        if curr["idx"] < len(_STAGES) - 1:
            crit = cur["loose_rate"] >= 0.25 and cur["strict"] >= 2
            flat = cur["loose_rate"] < 0.05 and cur["mean_progress"] < 0.005
            if crit or (elapsed >= _STAGE_STEPS and not flat):
                curr["idx"] += 1
                curr["stage_start_eval"] = evn + 1
                advanced = f" -> ADVANCE to {_STAGES[curr['idx']]} ({'criterion' if crit else 'budget'})"
        print(f"  [curr ev#{evn+1} {m['active_stage']}] " +
              " ".join(f"{s[-1]}:cov{v['coverage']}/{v['n']}L{v['loose']}mc{v['max_certified_clearance']:+.2f}"
                       for s, v in by_stage.items()) +
              f" | maxCertClr={max_cert:+.3f} 64102={m['s64102_strict']}{advanced}", flush=True)
        return float(max(max_cert, -1) + sum(v["coverage"] for v in by_stage.values()) * 0.01)

    out.mkdir(parents=True, exist_ok=True)
    curve = train_sac(actor, critics, env, cfg, eval_fn=eval_fn, bc_coef_fn=bc_coef_fn)
    torch.save(actor.state_dict(), out / "actor_final.pt")
    result = dict(seed=seed, source_checkpoint=str(_CKPT), curve=curve, best_step=best["step"],
                  best_max_certified_clearance=best["key"][0], best_metrics=best["m"], eval_history=hist)
    (out / "run.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"[curr] done | best max certified clearance={best['key'][0]:+.4f} @ step {best['step']}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/2026_07_21_coin_clearance_curriculum/run_s0")
    a = ap.parse_args()
    run(a.seed, Path(a.out))


if __name__ == "__main__":
    main()
