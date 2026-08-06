"""Matched n_step=3 (TREATMENT) vs n_step=1 (CONTROL) paired analysis (8 pairs): per-pair best-checkpoint deltas on
STAGE-2 transport + STAGE-1/64102 retention, bootstrap CIs, and the §10/§12 classification. Read-only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_BASE = "experiments/2026_07_21_coin_nstep"
_BOOT_SEED, _B = 20260721, 10000
_PAIRS = [(s, r) for s in range(4) for r in range(2)]


def _best(arm, s, r):
    p = Path(f"{_BASE}_{arm}_s{s}r{r}/run.json")
    return json.loads(p.read_text())["best_metrics"] if p.exists() else None


def _boot(x, seed, b=_B):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), size=(b, len(x)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def _stats(d, seed):
    return dict(values=[round(float(v), 3) for v in d], mean=round(float(np.mean(d)), 3),
                median=round(float(np.median(d)), 3), iqr=round(float(np.percentile(d, 75) - np.percentile(d, 25)), 3),
                pos=int(sum(x > 0 for x in d)), zero=int(sum(x == 0 for x in d)), neg=int(sum(x < 0 for x in d)),
                boot95=_boot(d, seed))


def main():
    per, keys = [], ["s2_cov", "s2_loose", "s2_maxclr", "s1_ret", "strong_ret", "r64102"]
    for s, r in _PAIRS:
        c, t = _best("n1", s, r), _best("n3", s, r)
        if c is None or t is None:
            continue
        def mc(m):
            v = m["stage2"]["max_certified_clearance"]
            return v if v > -9 else 0.0
        per.append(dict(seed=s, rep=r,
                        s2_cov=t["stage2"]["coverage"] - c["stage2"]["coverage"],
                        s2_loose=t["stage2"]["loose"] - c["stage2"]["loose"],
                        s2_maxclr=round(mc(t) - mc(c), 4),
                        s1_ret=t["stage1"]["coverage"] - c["stage1"]["coverage"],
                        strong_ret=int(t["strong_strict"]) - int(c["strong_strict"]),
                        r64102=int(t["s64102_strict"]) - int(c["s64102_strict"]),
                        t_s2cov=t["stage2"]["coverage"], c_s2cov=c["stage2"]["coverage"],
                        t_maxclr=mc(t), c_maxclr=mc(c), t_s1=t["stage1"]["coverage"], c_s1=c["stage1"]["coverage"],
                        t_strong=t["strong_strict"], c_strong=c["strong_strict"]))
    agg = {k: _stats([p[k] for p in per], _BOOT_SEED + i) for i, k in enumerate(keys)}
    # §12 classification
    s2cov = agg["s2_cov"]
    any_s2_certified = any(p["t_s2cov"] > 0 for p in per)
    s2_cov_above0 = s2cov["boot95"][0] > 0
    progress_up = agg["s2_loose"]["boot95"][0] > 0 or agg["s2_maxclr"]["boot95"][0] > 0
    degrades = (agg["s1_ret"]["boot95"][1] < 0 or agg["strong_ret"]["boot95"][1] < 0) and s2cov["median"] <= 0
    if any_s2_certified and s2_cov_above0:
        cls = "NSTEP_CLEAR_START_POSITIVE"
    elif degrades:
        cls = "NEGATIVE"
    elif progress_up:
        cls = "NSTEP_PROGRESS_POSITIVE"
    else:
        cls = "NO_EFFECT"
    out = dict(pairs=per, aggregate=agg, classification=cls, n_pairs=len(per), bootstrap_seed=_BOOT_SEED,
               any_stage2_certified=any_s2_certified)
    Path(f"{_BASE}_n3_s0r0/nstep_comparison.json").write_text(json.dumps(out, indent=1, default=float))
    for p in per:
        print(f"s{p['seed']}r{p['rep']}: S2cov {p['c_s2cov']}->{p['t_s2cov']}(D{p['s2_cov']:+d}) maxclr {p['c_maxclr']:+.3f}->{p['t_maxclr']:+.3f} "
              f"| S1ret {p['c_s1']}->{p['t_s1']}(D{p['s1_ret']:+d}) strong {p['c_strong']}->{p['t_strong']}")
    print(f"--- {len(per)} pairs, paired deltas (n3 - n1) ---")
    for k in keys:
        a = agg[k]
        print(f"  {k}: mean={a['mean']:+.3g} median={a['median']:+.3g} (+{a['pos']}/0={a['zero']}/-{a['neg']}) boot95={a['boot95']}")
    print(f"any STAGE2 certified (n3): {any_s2_certified}")
    print(f"=== CLASSIFICATION: {cls}")


if __name__ == "__main__":
    main()
