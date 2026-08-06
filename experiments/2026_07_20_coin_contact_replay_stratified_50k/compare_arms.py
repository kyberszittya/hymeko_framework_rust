"""Matched CONTROL vs STRATIFIED comparison on the best checkpoints: strict count/coverage + conditional contact-quality
endpoints on the 18-state eval set, state-64102 retention, and the §11 classification. Read-only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.experiments.coin_contact_replay import _EVAL_SEEDS
from hymeko_rl.experiments.coin_two_arm_sac import direct_env, policy_strict
from hymeko_rl.train.coin_delivery_actor import (
    _BODY_SHOVE_MAX,
    _DWELL_STEPS,
    _ONE_FINGER_MAX,
    _SETTLE_VEL,
    _attribution_from_trace,
    rollout,
)
from hymeko_rl.train.sac import build_sac

_ATTR_MIN = 0.60
_ARMS = {"CONTROL": Path("experiments/2026_07_20_coin_contact_replay_control_50k/actor_best.pt"),
         "STRATIFIED": Path("experiments/2026_07_20_coin_contact_replay_stratified_50k/actor_best.pt")}


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
    la = float(np.mean([abs(s.action[0]) + abs(s.action[3]) for s in tr.steps])) if tr.steps else 0.0
    return dict(seed=int(seed), zone=tr.loose, strict=policy_strict(tr), attribution=float(ff),
                body=float(att.alpha_body), clean=bool(clean), bilateral=bool(tr.both_frac > 0),
                dwell=int(tr.best_dwell), settle=float(tr.settle_vel), attr_margin=float(ff - _ATTR_MIN),
                lc=float(np.mean([s.left_contact for s in tr.steps])) if tr.steps else 0.0,
                rc=float(np.mean([s.right_contact for s in tr.steps])) if tr.steps else 0.0, act_mag=la)


def summarize(rows):
    zone = [r for r in rows if r["zone"]]
    nz = max(1, len(zone))
    strict = [r for r in rows if r["strict"]]
    return dict(
        strict_count=len(strict), strict_coverage=len({r["seed"] for r in strict}),
        zone_count=sum(r["zone"] for r in rows), zone_rate=sum(r["zone"] for r in rows) / len(rows),
        P_attr_given_zone=sum(r["attribution"] >= _ATTR_MIN for r in zone) / nz,
        P_clean_given_zone=sum(r["clean"] for r in zone) / nz,
        P_bilateral_given_zone=sum(r["bilateral"] for r in zone) / nz,
        P_body_given_zone=sum(r["body"] <= _BODY_SHOVE_MAX for r in zone) / nz,
        P_dwell_given_zone=sum(r["dwell"] >= _DWELL_STEPS for r in zone) / nz,
        P_settle_given_zone=sum(r["settle"] <= _SETTLE_VEL for r in zone) / nz,
        attr_margin_mean=float(np.mean([r["attr_margin"] for r in zone])) if zone else None,
        attr_margin_median=float(np.median([r["attr_margin"] for r in zone])) if zone else None,
        s64102_strict=next((r["strict"] for r in rows if r["seed"] == 64102), None),
        lc=float(np.mean([r["lc"] for r in rows])), rc=float(np.mean([r["rc"] for r in rows])),
        act_mag=float(np.mean([r["act_mag"] for r in rows])),
        strict_states=sorted(r["seed"] for r in strict))


def main():
    env = direct_env()
    out = {}
    for arm, ck in _ARMS.items():
        rows = [decompose(env, load(ck), s) for s in _EVAL_SEEDS]
        out[arm] = summarize(rows)
    c, s = out["CONTROL"], out["STRATIFIED"]
    # §11 classification
    more_strict = s["strict_coverage"] > c["strict_coverage"] or s["strict_count"] > c["strict_count"]
    mech_up = (s["P_attr_given_zone"] > c["P_attr_given_zone"] + 0.05) or (s["P_clean_given_zone"] > c["P_clean_given_zone"] + 0.05)
    zone_down = s["zone_rate"] < c["zone_rate"] - 0.1
    cover_down = s["strict_coverage"] < c["strict_coverage"]
    if zone_down or cover_down:
        cls = "NEGATIVE"
    elif more_strict and mech_up:
        cls = "POSITIVE"
    elif mech_up and not more_strict:
        cls = "MECHANISM_ONLY_POSITIVE"
    else:
        cls = "NO_EFFECT"
    out["classification"] = dict(verdict=cls, more_strict=bool(more_strict), mechanism_up=bool(mech_up),
                                 zone_down=bool(zone_down), coverage_down=bool(cover_down))
    Path("experiments/2026_07_20_coin_contact_replay_stratified_50k/comparison.json").write_text(
        json.dumps(out, indent=1, default=float))
    for arm in ("CONTROL", "STRATIFIED"):
        m = out[arm]
        print(f"[{arm}] strict={m['strict_count']} coverage={m['strict_coverage']} states={m['strict_states']} "
              f"zone={m['zone_rate']:.2f} | P(attr|zone)={m['P_attr_given_zone']:.2f} P(clean|zone)={m['P_clean_given_zone']:.2f} "
              f"P(bilat|zone)={m['P_bilateral_given_zone']:.2f} 64102={m['s64102_strict']} "
              f"attrMargin med={m['attr_margin_median']:.3f} L/R={m['lc']:.2f}/{m['rc']:.2f}")
    print(f"=== CLASSIFICATION: {cls} | {out['classification']}")


if __name__ == "__main__":
    main()
