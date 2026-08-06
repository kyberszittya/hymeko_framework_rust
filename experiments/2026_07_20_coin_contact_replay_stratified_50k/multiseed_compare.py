"""Matched multi-seed replication analysis (seeds 0-3): per-seed CONTROL vs STRATIFIED best-checkpoint endpoints,
paired deltas (STRATIFIED - CONTROL), median/IQR + sign counts, and the §-decision classification. Read-only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.experiments.coin_contact_replay import _EVAL_SEEDS
from hymeko_rl.experiments.coin_two_arm_sac import direct_env, policy_strict
from hymeko_rl.train.coin_delivery_actor import (
    _ONE_FINGER_MAX,
    _attribution_from_trace,
    rollout,
)
from hymeko_rl.train.sac import build_sac

_ATTR = 0.60
_BASE = "experiments/2026_07_20_coin_contact_replay"


def _dir(arm: str, seed: int) -> Path:
    return Path(f"{_BASE}_{arm}_50k") if seed == 0 else Path(f"{_BASE}_{arm}_50k_s{seed}")


def load(p):
    a, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    a.load_state_dict(torch.load(p, map_location="cpu"))
    return a


def decompose(env, actor, seed):
    env.reset(seed=int(seed))
    def g(inner, t, obs):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).numpy()[0]
    tr = rollout(env, g, max_steps=60)
    att = _attribution_from_trace(tr)
    ff = att.fingertip_fraction
    clean = (min(att.alpha_L, att.alpha_R) / (ff + 1e-9)) >= _ONE_FINGER_MAX
    return dict(seed=int(seed), zone=tr.loose, strict=policy_strict(tr), attribution=float(ff),
                body=float(att.alpha_body), clean=bool(clean), bilateral=bool(tr.both_frac > 0),
                dwell=int(tr.best_dwell), settle=float(tr.settle_vel), attr_margin=float(ff - _ATTR))


def summarize(rows):
    zone = [r for r in rows if r["zone"]]
    nz = max(1, len(zone))
    strict = [r for r in rows if r["strict"]]
    return dict(strict_count=len(strict), coverage=len({r["seed"] for r in strict}),
                zone_count=sum(r["zone"] for r in rows),
                P_attr=sum(r["attribution"] >= _ATTR for r in zone) / nz,
                P_clean=sum(r["clean"] for r in zone) / nz,
                P_bilat=sum(r["bilateral"] for r in zone) / nz,
                attr_margin_mean=float(np.mean([r["attr_margin"] for r in zone])) if zone else 0.0,
                s64102=next((r["strict"] for r in rows if r["seed"] == 64102), None),
                strict_states=sorted(r["seed"] for r in strict))


def _iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def main():
    env = direct_env()
    per_seed = {}
    for seed in (0, 1, 2, 3):
        try:
            c = summarize([decompose(env, load(_dir("control", seed) / "actor_best.pt"), s) for s in _EVAL_SEEDS])
            s = summarize([decompose(env, load(_dir("stratified", seed) / "actor_best.pt"), s) for s in _EVAL_SEEDS])
        except FileNotFoundError as e:
            print(f"seed {seed}: missing checkpoint ({e}) — skipping")
            continue
        per_seed[seed] = dict(control=c, stratified=s, delta=dict(
            strict_count=s["strict_count"] - c["strict_count"], coverage=s["coverage"] - c["coverage"],
            P_clean=round(s["P_clean"] - c["P_clean"], 3), P_attr=round(s["P_attr"] - c["P_attr"], 3),
            zone_count=s["zone_count"] - c["zone_count"]))
    seeds = sorted(per_seed)
    dkeys = ["strict_count", "coverage", "P_clean", "P_attr", "zone_count"]
    agg = {}
    for k in dkeys:
        d = [per_seed[s]["delta"][k] for s in seeds]
        agg[k] = dict(values=d, median=float(np.median(d)), iqr=_iqr(d),
                      pos=sum(x > 0 for x in d), zero=sum(x == 0 for x in d), neg=sum(x < 0 for x in d))
    # decision rule
    sc, cov = agg["strict_count"], agg["coverage"]
    worse_strict_or_cov = max(sc["neg"], cov["neg"])
    better_strict_or_cov = max(sc["pos"], cov["pos"])
    mech_consistent_up = agg["P_clean"]["pos"] >= 3 or agg["P_attr"]["pos"] >= 3
    s64102 = {s: per_seed[s]["stratified"]["s64102"] for s in seeds}
    retains = sum(bool(v) for v in s64102.values())
    if better_strict_or_cov >= 3 and retains >= 3 and mech_consistent_up:
        cls = "REVISED_POSITIVE"
    elif worse_strict_or_cov >= 3 and (sc["median"] < 0 or cov["median"] < 0) and not mech_consistent_up:
        cls = "CONFIRMED_NEGATIVE"
    elif abs(sc["median"]) <= 0.5 and abs(agg["P_clean"]["median"]) <= 0.05 and sc["pos"] <= 1 and sc["neg"] <= 1:
        cls = "NO_EFFECT"
    else:
        cls = "SEED_SENSITIVE"
    out = dict(seeds=seeds, per_seed=per_seed, aggregate=agg, s64102_retention=s64102, classification=cls)
    Path("experiments/2026_07_20_coin_contact_replay_stratified_50k/multiseed_comparison.json").write_text(
        json.dumps(out, indent=1, default=float))
    for s in seeds:
        c, st, d = per_seed[s]["control"], per_seed[s]["stratified"], per_seed[s]["delta"]
        print(f"seed{s}: CTRL strict={c['strict_count']} cov={c['coverage']} Pattr={c['P_attr']:.2f} Pclean={c['P_clean']:.2f} 64102={c['s64102']}"
              f" | STRAT strict={st['strict_count']} cov={st['coverage']} Pattr={st['P_attr']:.2f} Pclean={st['P_clean']:.2f} 64102={st['s64102']}"
              f" | Δstrict={d['strict_count']:+d} Δcov={d['coverage']:+d} ΔPclean={d['P_clean']:+.2f}")
    print("--- paired deltas (STRAT - CTRL) across seeds", seeds, "---")
    for k in dkeys:
        a = agg[k]
        print(f"  {k}: values={a['values']} median={a['median']:+.3g} IQR={a['iqr']:.3g} (+{a['pos']}/0={a['zero']}/-{a['neg']})")
    print(f"64102 retention (STRAT): {s64102}")
    print(f"=== CLASSIFICATION: {cls}")


if __name__ == "__main__":
    main()
