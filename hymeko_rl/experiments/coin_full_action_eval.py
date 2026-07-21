"""Full-action final + temporal + generalization eval and verdicts (2026-07-22, step 6).

Three questions, the PRIMARY baseline being the standalone BC the RL was initialised from (NOT the scripted expert):

  A FINAL TASK SUCCESS  — scripted / BC / SAC / TD3 / zero-action: native center + strict, separately.
  B TEMPORAL            — first zone-entry / first strict step, success-by-time, success-curve AUC, TTS median/IQR.
  C GENERALIZATION      — frozen panel + >=50 held-out POINT transport states (disjoint from the VAL selection set).

Verdict per algorithm (§8), decided against the STANDALONE BC:
  FULL_ACTION_RL_SUCCESS_POSITIVE   median strict > BC by a margin at the matched horizon
  FULL_ACTION_RL_TEMPORAL_POSITIVE  no worse final strict AND materially faster (higher AUC / lower TTS)
  FULL_ACTION_RL_NO_EFFECT / _REGRESSION / RUN_INVALID
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.experiments.coin_full_action_rl import _BC_CKPT, greedy_fn
from hymeko_rl.train.coin_delivery_rl import p_grasp_carry
from hymeko_rl.train.coin_full_action import eval_full_action, make_full_action_env

_PANEL = (1011, 1045, 1164, 1174, 1202, 1278, 1358, 1447, 1568)
_HELD = tuple(s for s in range(1000, 1100) if s not in _PANEL)[:50]
_OBS, _ACT = 41, 6


def _load(algo: str, path: str):
    if algo == "SAC" or algo == "BC":
        from hymeko_rl.train.sac import build_sac
        ac, _ = build_sac("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0)
    else:
        from hymeko_rl.train.ddpg import build_offpolicy
        ac, _ = build_offpolicy("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0, n_critics=2)
    ac.load_state_dict(torch.load(path, weights_only=True))
    ac.eval()
    return ac


def _summ(m: dict) -> dict:
    return {"center_rate": m["center_rate"], "strict": m["strict_count"], "auc": m["success_curve_auc"],
            "tts_median": m["tts_median"], "first_zone_median": m["first_zone_median"]}


def _verdict(bc: dict, med: dict) -> str:
    ds = med["strict"] - bc["strict"]          # strict count delta (out of len(seeds)); on panel(9)+held(50)=59
    dauc = med["auc"] - bc["auc"]
    if med["center_rate"] != med["center_rate"]:
        return "RUN_INVALID"
    if ds >= 3:
        return "FULL_ACTION_RL_SUCCESS_POSITIVE"
    if ds <= -3:
        return "FULL_ACTION_RL_REGRESSION"
    # no material final-success change → check temporal (faster without worse final strict)
    if ds >= 0 and dauc >= 0.05:
        return "FULL_ACTION_RL_TEMPORAL_POSITIVE"
    return "FULL_ACTION_RL_NO_EFFECT"


def evaluate(campaign_dir: str, out: str, *, horizon: int = 160) -> dict[str, Any]:
    env = make_full_action_env(fingertip_geometry="POINT", horizon=horizon)
    dists = {"panel": _PANEL, "heldout": _HELD}
    all_seeds = _PANEL + _HELD

    def on_all(fn):
        m = eval_full_action(fn, all_seeds, env)
        return _summ(m)

    res: dict[str, Any] = {"horizon": horizon, "distributions": {k: len(v) for k, v in dists.items()},
                           "combined_n": len(all_seeds)}
    # A: reference sources
    res["scripted"] = on_all(lambda _o: np.asarray(p_grasp_carry(env.inner, env._suffix_t), np.float32))
    res["zero_action"] = on_all(lambda _o: np.zeros(_ACT, np.float32))
    bc = _load("BC", _BC_CKPT)
    res["BC"] = on_all(greedy_fn(bc))
    print("scripted:", res["scripted"], "\nBC:", res["BC"], "\nzero:", res["zero_action"], flush=True)
    # SAC / TD3 across seeds
    per: dict[str, list] = {"SAC": [], "TD3": []}
    for rj in sorted(glob.glob(f"{campaign_dir}/*/run.json")):
        meta = json.loads(Path(rj).read_text())
        algo, seed = meta["algo"], meta["seed"]
        pt = Path(rj).parent / f"{algo.lower()}_actor_best.pt"
        if not pt.is_file():
            continue
        s = on_all(greedy_fn(_load(algo, str(pt))))
        per[algo].append({"seed": seed, **s})
        print(f"  {algo}_s{seed}: center={s['center_rate']} strict={s['strict']} auc={s['auc']} tts={s['tts_median']}",
              flush=True)
    res["runs"] = per
    res["verdicts"] = {}
    for algo, rows in per.items():
        if not rows:
            continue
        med = {"center_rate": float(np.median([r["center_rate"] for r in rows])),
               "strict": float(np.median([r["strict"] for r in rows])),
               "auc": float(np.median([r["auc"] for r in rows])),
               "tts_median": float(np.median([r["tts_median"] for r in rows if r["tts_median"] is not None]))
               if any(r["tts_median"] is not None for r in rows) else None}
        v = _verdict(res["BC"], med)
        res["verdicts"][algo] = {"median": med, "verdict": v, "n_seeds": len(rows)}
        print(f"VERDICT {algo}: {v} | BC strict {res['BC']['strict']} -> {med['strict']}, "
              f"AUC {res['BC']['auc']} -> {med['auc']} (median/{len(rows)})", flush=True)
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    print("WROTE", out, flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    evaluate(a.campaign, a.out)


if __name__ == "__main__":
    main()
