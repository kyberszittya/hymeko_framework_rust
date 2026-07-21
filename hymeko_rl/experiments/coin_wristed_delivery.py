"""Wristed-independent-pad delivery integration + 2×2 physical oracle.

Wires the schema-aware wrist/closure motor path (``env.pad_actuation``) into the canonical Coin Delivery env
(``CoinDeliveryTrainEnv → WristedPadContactFormationEnv → PlanarGraspEnv``) and runs the explicit APPROACH → WRIST_ALIGN
→ PAD_CLOSE → FORCE_HOLD → TRANSPORT → BRAKE → RELEASE → WITHDRAW → SETTLE oracle over the E0/E1/E2/E3 embodiments to
isolate the load-bearing DoF. The arm is the canonical cooperative ``grasp_carry`` (zero residual); the wrist/closure
are driven by the bounded controllers; the reward/strict predicate/coin/target are unchanged.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.env.pad_actuation import Phase, PadLimits, build_wristed_contact_env
from hymeko_rl.experiments.exp_v3_handoff_gate import _roll_env
from hymeko_rl.train.coin_delivery_rl import CoinDeliveryTrainEnv, DeliveryRLConfig

_CENTER_TOL = 0.02      # canonical delivery center-reach tolerance (DeliveryRLConfig.center_tol)
_SETTLE_VEL = 0.06      # coin speed below which it has "settled" (strict low-velocity dwell condition)

_GEOMS = ("E0", "E1", "E2", "E3")
_GEOM_MAP = {"E0": "CONCAVE_CLAMP", "E1": "E1_WRIST", "E2": "E2_CLOSURE", "E3": "E3_WRIST_CLOSURE"}


def make_wristed_delivery_env(geom: str, *, limits: PadLimits | None = None):
    """Build the canonical delivery env over the wristed planar env; return (delivery_env, wristed_contact_env)."""
    from hymeko_rl.experiments.pedc_selection import _C1_HORIZON, _ctx, _load_pkl_bank, c1_config
    planar = _roll_env(_GEOM_MAP[geom])
    cf = build_wristed_contact_env(planar, _load_pkl_bank("c1_heldseed_bank.pkl", holdout=False),
                                   _ctx()["contract"], horizon=_C1_HORIZON, cfg=c1_config(), limits=limits)
    return CoinDeliveryTrainEnv(cf, DeliveryRLConfig()), cf


def _both_pad_contact(cf) -> "tuple[bool, float, float]":
    from hymeko_rl.env.pad_actuation import _pad_normal_force
    m = cf._pad.m
    d = cf._pad.d
    fl = _pad_normal_force(m, d, "left", cf._pad._disk)
    fr = _pad_normal_force(m, d, "right", cf._pad._disk)
    return (fl > 0.1 and fr > 0.1), fl, fr


def oracle_rollout(env, cf, *, max_steps: int = 200, seed: int | None = None) -> dict:
    """Drive the explicit phase machine; the arm runs the canonical grasp_carry (zero residual), wrist/closure the
    bounded controllers. Returns the outcome + per-phase reached flags + a failure class."""
    env.reset(seed=seed)
    inner = cf._env
    has_wrist = len(cf._pad.groups["WRIST_YAW"]) > 0
    phase = Phase.APPROACH
    ph_box = {"p": phase}

    def _base(innr, tt):                                            # ARM base is phase-aware: HOLD during acquisition,
        d, _n = innr.direction_to_zone()                           # grasp_carry-CARRY during transport (so closure can
        p = ph_box["p"]                                            # form a bilateral grip without the arm shoving the coin)
        if p in (Phase.APPROACH,):
            return np.array([d[0], d[1], 0.0, 0.6, 0.0, 0.0], np.float32)         # approach + light squeeze
        if p in (Phase.WRIST_ALIGN, Phase.PAD_CLOSE, Phase.FORCE_HOLD):
            return np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0], np.float32)           # HOLD (no translation) while gripping
        if p in (Phase.TRANSPORT, Phase.BRAKE):
            return np.array([d[0], d[1], 0.0, 0.7, 0.0, 0.0], np.float32)         # carry the gripped coin to the zone
        return np.array([0.0, 0.0, -0.6, -0.5, 0.0, 0.0], np.float32)            # RELEASE/WITHDRAW/SETTLE: open + still
    env._base_override = _base
    reached = {p.value: False for p in Phase}
    log = dict(force_hold_steps=0, min_dtz=9.9, wrist_err_at_close=9.9)
    stable = 0
    centered_and_settled = False
    body_shove_ever = False
    for t in range(max_steps):
        reached[phase.value] = True
        ph_box["p"] = phase
        cf.set_phase(phase)
        dtz = float(inner._planar_metrics.disk_to_zone)
        log["min_dtz"] = min(log["min_dtz"], dtz)
        met = inner._planar_metrics
        both_ft = bool(met.left_contact and met.right_contact)
        both_pad, fl, fr = _both_pad_contact(cf)
        werr = float(np.mean(cf._pad.log.wrist_err[-2:])) if cf._pad.log.wrist_err else 9.9
        cvel = float(np.linalg.norm(inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2]))
        if dtz <= _CENTER_TOL and cvel < _SETTLE_VEL:
            centered_and_settled = True                            # canonical strict conditions: centered + settled
        lg = getattr(met, "legality", None)
        if lg is not None and getattr(lg, "arm_body_contact", False):
            body_shove_ever = True
        # transitions (public named measurements)
        if phase is Phase.APPROACH and (both_ft or (met.left_contact or met.right_contact)):
            phase = Phase.WRIST_ALIGN
        elif phase is Phase.WRIST_ALIGN and (werr < 0.25 or not has_wrist):    # no-wrist embodiments skip alignment
            phase = Phase.PAD_CLOSE
        elif phase is Phase.PAD_CLOSE and both_pad:
            log["wrist_err_at_close"] = werr
            phase = Phase.FORCE_HOLD
        elif phase is Phase.FORCE_HOLD:
            in_band = min(fl, fr) > 0.5 * cf._pad.lim.force_target
            log["force_hold_steps"] += int(in_band)
            if log["force_hold_steps"] >= 3:
                phase = Phase.TRANSPORT
            elif not (met.left_contact or met.right_contact):
                phase = Phase.PAD_CLOSE
        elif phase is Phase.TRANSPORT and dtz <= inner._zone_half * 1.4:
            phase = Phase.BRAKE
        elif phase is Phase.BRAKE and cvel < 0.05:
            phase = Phase.RELEASE
        elif phase is Phase.RELEASE and (fl + fr) < 0.2:
            phase = Phase.WITHDRAW
        elif phase is Phase.WITHDRAW:
            phase = Phase.SETTLE
        elif phase is Phase.SETTLE:
            stable += 1
        _o, _r, term, trunc, _i = env.step(np.zeros(env.action_space.shape[0], np.float32))
        if term or trunc or (phase is Phase.SETTLE and stable > 5):
            break
    strict = bool(centered_and_settled and not body_shove_ever)
    fail = _classify_fail(reached, log, strict)
    return dict(strict=int(strict), min_dtz=round(log["min_dtz"], 4), reached=reached,
                force_hold_steps=log["force_hold_steps"], wrist_err_at_close=round(log["wrist_err_at_close"], 3),
                body_shove=int(body_shove_ever), failure=fail)


def _clearance(inner) -> float:
    disk_r = float(inner.model.geom_size[inner._disk_geom][0])
    return float(inner.planar_metrics.disk_to_zone) - (disk_r + float(inner._zone_half))


def run_2x2(seeds: int, out) -> dict:
    """The matched E0/E1/E2/E3 physical oracle over seeds, bucketed by signed initial clearance band."""
    import json
    from collections import Counter
    from pathlib import Path
    bands = [("+0.018-0.030", 0.018, 0.030), ("+0.030-0.045", 0.030, 0.045),
             ("+0.045-0.060", 0.045, 0.060), ("+0.060-0.080", 0.060, 0.080)]
    result = {}
    for g in _GEOMS:
        env, cf = make_wristed_delivery_env(g)
        by_band = {name: dict(n=0, strict=0, fails=Counter(), min_dtz=9.9) for name, _lo, _hi in bands}
        for s in range(seeds):
            env.reset(seed=70_000 + s)
            clr = _clearance(cf._env)
            band = next((nm for nm, lo, hi in bands if lo <= clr < hi), None)
            if band is None:
                continue
            r = oracle_rollout(env, cf, max_steps=200, seed=70_000 + s)
            b = by_band[band]
            b["n"] += 1
            b["strict"] += r["strict"]
            b["fails"][r["failure"]] += 1
            b["min_dtz"] = min(b["min_dtz"], r["min_dtz"])
        for nm in by_band:
            by_band[nm]["fails"] = dict(by_band[nm]["fails"])
            by_band[nm]["min_dtz"] = round(by_band[nm]["min_dtz"], 4)
        result[g] = by_band
        tot = sum(by_band[nm]["strict"] for nm in by_band)
        strict030 = sum(by_band[nm]["strict"] for nm in by_band if not nm.startswith("+0.018"))
        print(f"[{g}] total strict={tot} strict>=+0.030={strict030} | "
              + " ".join(f"{nm}:{by_band[nm]['strict']}/{by_band[nm]['n']}(md{by_band[nm]['min_dtz']})" for nm in by_band),
              flush=True)

    def s030(g):
        return sum(result[g][nm]["strict"] for nm in result[g] if not nm.startswith("+0.018"))
    e0, e1, e2, e3 = (s030(g) for g in _GEOMS)
    if e3 == 0 and e1 == 0 and e2 == 0:
        verdict = "NO_FORCE_CLOSURE"
    elif e3 > max(e1, e2, e0):
        verdict = "WRIST_CLOSURE_POSITIVE"
    elif e1 > e0 and e1 >= e2:
        verdict = "WRIST_POSITIVE"
    elif e2 > e0:
        verdict = "CLOSURE_POSITIVE"
    else:
        verdict = "NO_FORCE_CLOSURE"
    summary = dict(strict_ge_030={g: s030(g) for g in _GEOMS}, verdict=verdict, bands=result)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "wristed_oracle.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"[oracle] strict>=+0.030 E0={e0} E1={e1} E2={e2} E3={e3}\n=== VERDICT: {verdict}", flush=True)
    return summary


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--out", default="experiments/2026_07_21_coin_wristed_pad/oracle")
    a = ap.parse_args()
    run_2x2(a.seeds, a.out)


def _classify_fail(reached, log, strict) -> str:
    if strict:
        return "OK"
    if not reached["WRIST_ALIGN"]:
        return "APPROACH_FAILURE"
    if not reached["PAD_CLOSE"]:
        return "WRIST_ALIGNMENT_FAILURE"
    if not reached["FORCE_HOLD"]:
        return "PAD_CONTACT_FAILURE"
    if not reached["TRANSPORT"]:
        return "FORCE_HOLD_FAILURE"
    if not reached["BRAKE"]:
        return "TRANSPORT_FAILURE"
    if not reached["RELEASE"]:
        return "SETTLE_FAILURE"
    return "SETTLE_FAILURE"


if __name__ == "__main__":
    main()
