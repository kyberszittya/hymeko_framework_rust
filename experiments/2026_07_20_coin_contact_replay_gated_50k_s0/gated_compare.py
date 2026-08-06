"""GATED vs UNIFORM 12-seed paired analysis: per-seed switch history + best-checkpoint endpoints, paired deltas
(GATED - UNIFORM), bootstrap CI, and the basin-coupling test vs the prior fixed-stratification -0.825. Read-only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.experiments.coin_contact_replay import _EVAL_SEEDS
from hymeko_rl.experiments.coin_two_arm_sac import direct_env, policy_strict
from hymeko_rl.train.coin_delivery_actor import _ONE_FINGER_MAX, _attribution_from_trace, rollout
from hymeko_rl.train.sac import build_sac

_ATTR, _BOOT_SEED, _B = 0.60, 20260720, 10000
_BASE = "experiments/2026_07_20_coin_contact_replay"


def _dir(arm, seed):
    return Path(f"{_BASE}_{arm}_50k_s{seed}")


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
    return dict(strict=policy_strict(tr), zone=tr.loose, attribution=float(ff), clean=bool(clean),
                bilateral=bool(tr.both_frac > 0))


def summ(rows):
    z = [r for r in rows if r["zone"]]
    nz = max(1, len(z))
    return dict(strict=sum(r["strict"] for r in rows), coverage=sum(r["strict"] for r in rows),
                zone=sum(r["zone"] for r in rows),
                P_attr=sum(r["attribution"] >= _ATTR for r in z) / nz, P_clean=sum(r["clean"] for r in z) / nz,
                P_bilat=sum(r["bilateral"] for r in z) / nz)


def _boot(x, seed, b=_B):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), size=(b, len(x)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return round(float(np.corrcoef(ra, rb)[0, 1]), 3)


def _stats(d, seed):
    return dict(values=list(d), mean=round(float(np.mean(d)), 3), median=float(np.median(d)),
                iqr=float(np.percentile(d, 75) - np.percentile(d, 25)), min=min(d), max=max(d),
                pos=sum(x > 0 for x in d), zero=sum(x == 0 for x in d), neg=sum(x < 0 for x in d), boot95=_boot(d, seed))


def main():
    env = direct_env()
    seeds, per = [], {}
    for s in range(12):
        try:
            u = summ([_row(env, load(_dir("uniform", s) / "actor_best.pt"), st) for st in _EVAL_SEEDS])
            g = summ([_row(env, load(_dir("gated", s) / "actor_best.pt"), st) for st in _EVAL_SEEDS])
            gate = json.load(open(_dir("gated", s) / "run.json")).get("gate", {})
        except FileNotFoundError:
            continue
        seeds.append(s)
        per[s] = dict(uniform=u, gated=g, gate=gate,
                      delta={k: round(g[k] - u[k], 4) for k in ("strict", "coverage", "zone", "P_attr", "P_clean")})
    agg = {k: _stats([per[s]["delta"][k] for s in seeds], _BOOT_SEED + i)
           for i, k in enumerate(["strict", "coverage", "zone", "P_attr", "P_clean"])}
    u_strict = [per[s]["uniform"]["strict"] for s in seeds]
    d_strict = [per[s]["delta"]["strict"] for s in seeds]
    rho = _spearman(u_strict, d_strict)
    weak = [d for d, c in zip(d_strict, u_strict) if c <= 2]
    strong = [d for d, c in zip(d_strict, u_strict) if c >= 4]
    basin = dict(spearman=rho, prior_fixed=-0.825, coupling_reduced=abs(rho) < 0.825 - 0.15,
                 weak_n=len(weak), weak_median=float(np.median(weak)) if weak else None,
                 strong_n=len(strong), strong_median=float(np.median(strong)) if strong else None)
    ms, mc = agg["strict"]["median"], agg["coverage"]["median"]
    contact_ok = agg["P_attr"]["median"] >= -0.03 and agg["P_clean"]["median"] >= -0.03
    wk = (basin["weak_median"] or 0) > 0
    st_ok = (basin["strong_median"] if basin["strong_median"] is not None else 0) >= 0
    if wk and st_ok and (ms >= 0 or mc >= 0) and basin["coupling_reduced"] and contact_ok:
        cls = "GATED_POSITIVE"
    elif wk and not st_ok:
        cls = "WEAK_ONLY"
    elif ms < 0 and agg["strict"]["neg"] > agg["strict"]["pos"] and not contact_ok:
        cls = "NEGATIVE"
    else:
        cls = "NO_EFFECT"
    out = dict(seeds=seeds, per_seed=per, aggregate=agg, basin=basin, classification=cls, bootstrap_seed=_BOOT_SEED)
    Path(f"{_BASE}_gated_50k_s0/gated_comparison.json").write_text(json.dumps(out, indent=1, default=float))
    for s in seeds:
        u, g, d, ga = per[s]["uniform"], per[s]["gated"], per[s]["delta"], per[s]["gate"]
        sw = f"switch@ev{ga.get('switch_eval')}/step{ga.get('switch_step')}" if ga.get("switched") else "no-switch"
        print(f"s{s:>2}: UNIF strict={u['strict']} | GATED strict={g['strict']} [{sw}] | Dstrict={d['strict']:+d} Dcov={d['coverage']:+d} DPattr={d['P_attr']:+.2f}")
    print("--- 12-seed paired deltas (GATED - UNIFORM) ---")
    for k in ("strict", "coverage", "P_attr", "P_clean", "zone"):
        a = agg[k]
        print(f"  {k}: mean={a['mean']:+.3g} median={a['median']:+.3g} IQR={a['iqr']:.3g} (+{a['pos']}/0={a['zero']}/-{a['neg']}) boot95={a['boot95']}")
    print(f"basin: Spearman(UNIFstrict,Dstrict)={rho} (prior fixed -0.825; reduced={basin['coupling_reduced']}) | "
          f"weak(<=2) n={basin['weak_n']} med={basin['weak_median']} | strong(>=4) n={basin['strong_n']} med={basin['strong_median']}")
    print(f"=== CLASSIFICATION: {cls}")


if __name__ == "__main__":
    main()
