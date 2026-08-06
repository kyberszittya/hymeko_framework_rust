"""COIN structured-problem-generator experiment — matched CONTROL vs GENERATOR (§7). Only variable: the distribution
of TRAINING initial states (CONTROL = fixed states; GENERATOR = 50% fixed + 50% generated configs). Actor, semantic
critic, corrected SAC, reward, strict predicate, obs, action space, BC anchor, replay sampler all unchanged. Reuses
coin_two_arm_sac + coin_contact_replay + the canonical restore path; the generator only supplies initial configs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.env.planar_snapshot import restore_planar
from hymeko_rl.experiments.coin_contact_replay import _EVAL_SEEDS, build_corpus, corpus_stratum_counts
from hymeko_rl.experiments.coin_problem_generator import load_configs
from hymeko_rl.experiments.coin_two_arm_sac import certify_or_abort, direct_env, policy_strict
from hymeko_rl.train.coin_delivery_actor import _attribution_from_trace, rollout
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac

_ATTR = 0.60
_GENDIR = Path("experiments/2026_07_20_coin_problem_generator")
_CKPT = Path("experiments/2026_07_20_coin_two_arm_sac_100k/sac_actor_best.pt")
_FIXED_TRAIN = tuple(range(64_000, 64_056))


def _restore_generated(env, snap) -> np.ndarray:
    env.reset(seed=64_000)                                        # init the wrapper (phase flags, tracking)
    restore_planar(env.inner, snap)
    env._reset_state()                                            # re-anchor _start_dtz/_prev_dtz to the restored state
    env._last_obs = np.asarray(env.env._obs(np.zeros(4, np.float32)), np.float32)
    return env._last_obs


def gen_env(train_configs, *, generated_frac: float, seed: int):
    """Direct env whose auto-reset draws `generated_frac` of episodes from `train_configs` (restore) and the rest from
    the fixed training seeds — the ONLY thing that differs between CONTROL (frac 0) and GENERATOR (frac 0.5)."""
    env = direct_env()
    env._base_override = lambda inner, t: np.zeros(env.action_space.shape[0], np.float32)
    env._delta_override = 1.0
    rng = np.random.default_rng(seed)
    _orig = env.reset

    def _reset(*, seed=None):
        if seed is None and train_configs and rng.random() < generated_frac:
            _restore_generated(env, train_configs[rng.integers(len(train_configs))].snapshot)
            return env._last_obs, {"source": "generated"}
        return _orig(seed=int(rng.choice(_FIXED_TRAIN)) if seed is None else seed)
    env.reset = _reset
    return env, rng


def _greedy(actor):
    def g(inner, t, obs):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).numpy()[0]
    return g


def _endpoints(rows):
    z = [r for r in rows if r["zone"]]
    nz = max(1, len(z))
    strict = [r for r in rows if r["strict"]]
    return dict(strict=len(strict), coverage=len(strict), zone=sum(r["zone"] for r in rows),
                P_attr=round(sum(r["attribution"] >= _ATTR for r in z) / nz, 3),
                P_clean=round(sum(r["clean"] for r in z) / nz, 3),
                P_bilat=round(sum(r["bilateral"] for r in z) / nz, 3), n=len(rows))


def _roll_row(env, actor, *, seed=None, snap=None):
    if snap is not None:
        _restore_generated(env, snap)
    else:
        env.reset(seed=int(seed))
    tr = rollout(env, _greedy(actor), max_steps=60)
    att = _attribution_from_trace(tr)
    ff = att.fingertip_fraction
    clean = (min(att.alpha_L, att.alpha_R) / (ff + 1e-9)) >= 0.15
    return dict(strict=policy_strict(tr), zone=tr.loose, attribution=float(ff), clean=bool(clean),
                bilateral=bool(tr.both_frac > 0), aL=att.alpha_L, aR=att.alpha_R,
                lc=float(np.mean([s.left_contact for s in tr.steps])) if tr.steps else 0.0,
                rc=float(np.mean([s.right_contact for s in tr.steps])) if tr.steps else 0.0)


def evaluate_all(eval_env, actor, held_configs) -> dict:
    fixed = [_roll_row(eval_env, actor, seed=s) for s in _EVAL_SEEDS]              # 18 fixed
    held = [dict(_roll_row(eval_env, actor, snap=c.snapshot), family=c.family) for c in held_configs]  # 48 generated
    s64102 = _roll_row(eval_env, actor, seed=64_102)["strict"]
    by_fam = {}
    for fam in sorted({c.family for c in held_configs}):
        by_fam[fam] = _endpoints([r for r in held if r["family"] == fam])
    return dict(fixed=_endpoints(fixed), held=_endpoints(held), held_by_family=by_fam, s64102_strict=bool(s64102),
                lc=round(float(np.mean([r["lc"] for r in fixed])), 3), rc=round(float(np.mean([r["rc"] for r in fixed])), 3))


def run(arm: str, seed: int, rep: int, steps: int, out: Path) -> dict:
    certify_or_abort()
    train_cfgs = load_configs(_GENDIR / "train_configs.pkl")
    held_cfgs = load_configs(_GENDIR / "held_configs.pkl")
    frac = 0.5 if arm == "generator" else 0.0
    run_seed = seed * 100 + rep                                  # distinct init per repetition, matched across arms
    env, _ = gen_env(train_cfgs, generated_frac=frac, seed=run_seed)
    corpus = build_corpus(env)
    print(f"[{arm} s{seed}r{rep}] generated_frac={frac} train_cfgs={len(train_cfgs)} held={len(held_cfgs)} "
          f"corpus={len(corpus[0])} strata={corpus_stratum_counts(corpus[5])}", flush=True)

    actor, critics = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    actor.load_state_dict(torch.load(_CKPT, map_location="cpu"))
    cfg = SACConfig.stable(total_steps=steps, seed=run_seed, bc_coef=1.0, log_every=2000, eval_every=2500)
    comp = {"progress_ok": False, "first_strict": False, "consec_strict": 0}

    def bc_coef_fn(_s):
        return 0.05 if comp["consec_strict"] >= 3 else 0.1 if comp["first_strict"] else 0.3 if comp["progress_ok"] else 1.0

    eval_env, _ = gen_env([], generated_frac=0.0, seed=run_seed)
    hist, best = [], {"score": -1.0, "m": None}

    def eval_fn(_e, ac):
        m = evaluate_all(eval_env, ac, held_cfgs)
        if m["fixed"]["strict"] >= 1 or m["held"]["strict"] >= 3:
            comp["progress_ok"] = True
        comp["consec_strict"] = comp["consec_strict"] + 1 if m["fixed"]["strict"] >= 1 else 0
        if m["fixed"]["strict"] >= 1:
            comp["first_strict"] = True
        hist.append(m)
        score = m["fixed"]["coverage"] * 1e3 + m["held"]["coverage"] * 1e1 + m["fixed"]["zone"]
        if score > best["score"]:
            best.update(score=score, m=m, step=len(hist) * 2500)
            torch.save(ac.state_dict(), out / "actor_best.pt")
        print(f"  [{arm} s{seed}r{rep} ev#{len(hist)}] fixed:cov={m['fixed']['coverage']} zone={m['fixed']['zone']} "
              f"Pattr={m['fixed']['P_attr']} | held:cov={m['held']['coverage']}/{m['held']['n']} "
              f"byfam={ {k[:4]:v['coverage'] for k,v in m['held_by_family'].items()} } 64102={m['s64102_strict']}", flush=True)
        return float(m["fixed"]["coverage"] + m["held"]["coverage"] * 0.1)

    out.mkdir(parents=True, exist_ok=True)
    curve = train_sac(actor, critics, env, cfg, eval_fn=eval_fn, offline_data=(corpus[0], corpus[1]),
                      init_transitions=corpus[:5], bc_coef_fn=bc_coef_fn)
    torch.save(actor.state_dict(), out / "actor_final.pt")
    result = dict(arm=arm, seed=seed, rep=rep, steps=steps, generated_frac=frac,
                  curve=curve,
                  best_step=best.get("step"), best_metrics=best["m"], eval_history=hist)
    (out / "run.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"[{arm} s{seed}r{rep}] done | best fixed cov={best['m']['fixed']['coverage'] if best['m'] else 0} "
          f"held cov={best['m']['held']['coverage'] if best['m'] else 0}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["control", "generator"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.arm, a.seed, a.rep, a.steps, Path(a.out))


if __name__ == "__main__":
    main()
