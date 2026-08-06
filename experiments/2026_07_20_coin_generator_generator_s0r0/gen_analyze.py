"""Matched GENERATOR vs CONTROL paired analysis (8 pairs = 4 seeds x 2 reps): per-pair best-checkpoint deltas,
mean/median/IQR/bootstrap CI, held-out-by-family, 64102 retention, and the §12 classification. Read-only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_BASE = "experiments/2026_07_20_coin_generator"
_BOOT_SEED, _B = 20260720, 10000
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
                min=round(float(min(d)), 3), max=round(float(max(d)), 3),
                pos=int(sum(x > 0 for x in d)), zero=int(sum(x == 0 for x in d)), neg=int(sum(x < 0 for x in d)),
                boot95=_boot(d, seed))


def main():
    per, keys = [], ["fixed_strict", "fixed_cov", "held_strict", "held_cov", "P_attr_fixed", "P_clean_fixed"]
    fam_keys = ["CERTIFIED_NEIGHBORHOOD", "ATTRIBUTION_BOUNDARY", "LEFT_RIGHT_SYMMETRY"]
    fam_delta = {k: [] for k in fam_keys}
    ret = {"control": [], "generator": []}
    for s, r in _PAIRS:
        c, g = _best("control", s, r), _best("generator", s, r)
        if c is None or g is None:
            continue
        d = dict(seed=s, rep=r,
                 fixed_strict=g["fixed"]["strict"] - c["fixed"]["strict"],
                 fixed_cov=g["fixed"]["coverage"] - c["fixed"]["coverage"],
                 held_strict=g["held"]["strict"] - c["held"]["strict"],
                 held_cov=g["held"]["coverage"] - c["held"]["coverage"],
                 P_attr_fixed=round(g["fixed"]["P_attr"] - c["fixed"]["P_attr"], 3),
                 P_clean_fixed=round(g["fixed"]["P_clean"] - c["fixed"]["P_clean"], 3),
                 g_fixed=c["fixed"]["coverage"], gg=g["fixed"]["coverage"],
                 g_held=g["held"]["coverage"], c_held=c["held"]["coverage"],
                 g_64102=g["s64102_strict"], c_64102=c["s64102_strict"])
        for fk in fam_keys:
            fam_delta[fk].append(g["held_by_family"][fk]["coverage"] - c["held_by_family"][fk]["coverage"])
        ret["control"].append(c["s64102_strict"])
        ret["generator"].append(g["s64102_strict"])
        per.append(d)
    agg = {k: _stats([p[k] for p in per], _BOOT_SEED + i) for i, k in enumerate(keys)}
    fam_agg = {fk: _stats(fam_delta[fk], _BOOT_SEED + 100 + i) for i, fk in enumerate(fam_keys)}
    # §12 classification
    fx, hc = agg["fixed_cov"], agg["held_cov"]
    cov_above_zero = fx["boot95"][0] > 0 or hc["boot95"][0] > 0
    ret_ok = sum(ret["generator"]) >= sum(ret["control"]) and sum(ret["generator"]) > 0
    zone_only = agg["fixed_strict"]["median"] <= 0 and agg["held_strict"]["median"] <= 0     # gain only in loose zone
    mech_up = agg["P_attr_fixed"]["boot95"][0] > 0 or agg["P_clean_fixed"]["boot95"][0] > 0
    degrades = (fx["boot95"][1] < 0 or hc["boot95"][1] < 0) and sum(ret["generator"]) < sum(ret["control"])
    if cov_above_zero and ret_ok and not zone_only:
        cls = "GENERATOR_POSITIVE"
    elif degrades:
        cls = "GENERATOR_NEGATIVE"
    elif mech_up and not cov_above_zero:
        cls = "GENERATOR_MECHANISM_POSITIVE"
    else:
        cls = "NO_EFFECT"
    out = dict(pairs=per, aggregate=agg, held_by_family_delta=fam_agg, s64102=ret, classification=cls,
               n_pairs=len(per), bootstrap_seed=_BOOT_SEED)
    Path(f"{_BASE}_generator_s0r0/generator_comparison.json").write_text(json.dumps(out, indent=1, default=float))
    for p in per:
        print(f"s{p['seed']}r{p['rep']}: fixedcov {p['g_fixed']}->{p['gg']} (D{p['fixed_cov']:+d}) | "
              f"heldcov {p['c_held']}->{p['g_held']} (D{p['held_cov']:+d}) | 64102 C={p['c_64102']} G={p['g_64102']}")
    print(f"--- {len(per)} matched pairs, paired deltas (GENERATOR - CONTROL) ---")
    for k in keys:
        a = agg[k]
        print(f"  {k}: mean={a['mean']:+.3g} median={a['median']:+.3g} IQR={a['iqr']} (+{a['pos']}/0={a['zero']}/-{a['neg']}) boot95={a['boot95']}")
    print("--- held-out certified coverage delta by family ---")
    for fk in fam_keys:
        a = fam_agg[fk]
        print(f"  {fk}: median={a['median']:+.3g} boot95={a['boot95']} (+{a['pos']}/0={a['zero']}/-{a['neg']})")
    print(f"64102 retention: control={sum(ret['control'])}/{len(ret['control'])} generator={sum(ret['generator'])}/{len(ret['generator'])}")
    print(f"=== CLASSIFICATION: {cls}")


if __name__ == "__main__":
    main()
