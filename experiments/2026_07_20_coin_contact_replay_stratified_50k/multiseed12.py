"""Twelve-seed (0-11) matched CONTROL vs STRATIFIED pooled analysis: per-seed best-checkpoint endpoints, paired deltas,
mean/median/IQR/min/max + sign counts + deterministic bootstrap 95% CI, and the basin-interaction test (Spearman of the
strict delta vs CONTROL strict; weak/strong group medians). Read-only; classifies per the extended decision rule."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.experiments.coin_contact_replay import _EVAL_SEEDS
from hymeko_rl.experiments.coin_two_arm_sac import direct_env, policy_strict
from hymeko_rl.train.coin_delivery_actor import _ONE_FINGER_MAX, _attribution_from_trace, rollout
from hymeko_rl.train.sac import build_sac

_ATTR, _BODY, _BOOT_SEED, _B = 0.60, 0.20, 20260720, 10000
_BASE = "experiments/2026_07_20_coin_contact_replay"


def _dir(arm, seed):
    return Path(f"{_BASE}_{arm}_50k") if seed == 0 else Path(f"{_BASE}_{arm}_50k_s{seed}")


def load(p):
    a, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    a.load_state_dict(torch.load(p, map_location="cpu"))
    return a


def _row(env, actor, seed):
    env.reset(seed=int(seed))
    def g(inner, t, obs):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).numpy()[0]
    tr = rollout(env, g, max_steps=60)
    att = _attribution_from_trace(tr)
    ff = att.fingertip_fraction
    clean = (min(att.alpha_L, att.alpha_R) / (ff + 1e-9)) >= _ONE_FINGER_MAX
    return dict(strict=policy_strict(tr), zone=tr.loose, attribution=float(ff), body=float(att.alpha_body),
                clean=bool(clean), bilateral=bool(tr.both_frac > 0), margin=float(ff - _ATTR))


def summ(rows):
    z = [r for r in rows if r["zone"]]
    nz = max(1, len(z))
    strict = [r for r in rows if r["strict"]]
    return dict(strict=len(strict), coverage=len(strict), zone=sum(r["zone"] for r in rows),
                P_attr=sum(r["attribution"] >= _ATTR for r in z) / nz, P_clean=sum(r["clean"] for r in z) / nz,
                P_bilat=sum(r["bilateral"] for r in z) / nz, P_body=sum(r["body"] <= _BODY for r in z) / nz,
                margin=float(np.mean([r["margin"] for r in z])) if z else 0.0)


def _boot_ci(x, seed, b=_B):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), size=(b, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _stats(d, seed):
    d = list(d)
    lo, hi = _boot_ci(d, seed)
    return dict(values=d, mean=float(np.mean(d)), median=float(np.median(d)),
                iqr=float(np.percentile(d, 75) - np.percentile(d, 25)), min=min(d), max=max(d),
                pos=sum(x > 0 for x in d), zero=sum(x == 0 for x in d), neg=sum(x < 0 for x in d),
                boot95=[round(lo, 3), round(hi, 3)])


def main():
    env = direct_env()
    seeds, per = [], {}
    for s in range(12):
        try:
            c = summ([_row(env, load(_dir("control", s) / "actor_best.pt"), st) for st in _EVAL_SEEDS])
            t = summ([_row(env, load(_dir("stratified", s) / "actor_best.pt"), st) for st in _EVAL_SEEDS])
        except FileNotFoundError:
            continue
        seeds.append(s)
        per[s] = dict(control=c, stratified=t,
                      delta={k: round(t[k] - c[k], 4) for k in ("strict", "coverage", "P_clean", "P_attr", "P_bilat", "zone", "margin")})
    agg = {k: _stats([per[s]["delta"][k] for s in seeds], _BOOT_SEED + i)
           for i, k in enumerate(["strict", "coverage", "P_clean", "P_attr", "P_bilat", "zone", "margin"])}
    ctrl_strict = [per[s]["control"]["strict"] for s in seeds]
    d_strict = [per[s]["delta"]["strict"] for s in seeds]
    rho = _spearman(ctrl_strict, d_strict)
    weak = [d for d, c in zip(d_strict, ctrl_strict) if c <= 2]
    strong = [d for d, c in zip(d_strict, ctrl_strict) if c >= 4]
    basin = dict(spearman_ctrlstrict_vs_delta=round(rho, 3),
                 weak_group_ctrl_le2_n=len(weak), weak_median_delta=float(np.median(weak)) if weak else None,
                 strong_group_ctrl_ge4_n=len(strong), strong_median_delta=float(np.median(strong)) if strong else None)
    # decision
    ms, mc = agg["strict"]["median"], agg["coverage"]["median"]
    ci = agg["strict"]["boot95"]
    ci_spans_zero = ci[0] <= 0 <= ci[1]
    contact_up = agg["P_attr"]["median"] > 0.03 and agg["P_clean"]["median"] >= -0.02 and agg["zone"]["median"] >= -0.5
    inverse = rho <= -0.4 and (basin["weak_median_delta"] or 0) > 0 and (basin["strong_median_delta"] or 0) < 0
    nonneg = agg["strict"]["pos"] + agg["strict"]["zero"]
    if ci_spans_zero and inverse and abs(ms) <= 0.5:
        cls = "BASIN_DEPENDENT"
    elif (ms > 0 or mc > 0) and nonneg >= 8 and agg["strict"]["pos"] > agg["strict"]["neg"] and contact_up:
        cls = "AVERAGE_POSITIVE"
    elif (ms < 0 or mc < 0) and agg["strict"]["neg"] > agg["strict"]["pos"] and not contact_up:
        cls = "AVERAGE_NEGATIVE"
    else:
        cls = "NO_AVERAGE_EFFECT"
    out = dict(seeds=seeds, bootstrap_seed=_BOOT_SEED, n_bootstrap=_B, per_seed=per, aggregate=agg,
               basin_interaction=basin, classification=cls)
    Path(f"{_BASE}_stratified_50k/twelveseed_comparison.json").write_text(json.dumps(out, indent=1, default=float))
    for s in seeds:
        c, t, d = per[s]["control"], per[s]["stratified"], per[s]["delta"]
        print(f"s{s:>2}: CTRL strict={c['strict']} Pattr={c['P_attr']:.2f} | STRAT strict={t['strict']} Pattr={t['P_attr']:.2f} "
              f"| Dstrict={d['strict']:+d} Dcov={d['coverage']:+d} DPattr={d['P_attr']:+.2f} DPclean={d['P_clean']:+.2f}")
    print(f"--- {len(seeds)}-seed paired deltas ---")
    for k in ("strict", "coverage", "P_attr", "P_clean", "zone"):
        a = agg[k]
        print(f"  {k}: mean={a['mean']:+.3g} median={a['median']:+.3g} IQR={a['iqr']:.3g} min/max={a['min']}/{a['max']} "
              f"(+{a['pos']}/0={a['zero']}/-{a['neg']}) boot95={a['boot95']}")
    print(f"basin: Spearman(CTRLstrict,Dstrict)={basin['spearman_ctrlstrict_vs_delta']} | "
          f"weak(<=2) n={basin['weak_group_ctrl_le2_n']} med={basin['weak_median_delta']} | "
          f"strong(>=4) n={basin['strong_group_ctrl_ge4_n']} med={basin['strong_median_delta']}")
    print(f"=== CLASSIFICATION: {cls}")


if __name__ == "__main__":
    main()
