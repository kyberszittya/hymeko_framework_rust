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
_DWELL_REQ = 6          # consecutive in-zone + low-velocity steps required for the strict certificate (dwell)

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


def oracle_rollout(env, cf, *, max_steps: int = 260, seed: int | None = None, variant: str = "A") -> dict:
    """Certify-before-release oracle. Variants differ ONLY in post-transport sequencing (identical approach/grasp/carry):
    A = CLOSED_HOLD_CERTIFY (hold pads closed + arm fixed in the zone, accumulate the strict dwell, terminate on cert —
    NEVER release); B = CERTIFY_THEN_RELEASE; C = CERTIFY_RELEASE_WITHDRAW; D = CURRENT (release/withdraw before dwell).
    Logs the per-step strict components + the first failing one + the best pre-release dwell/velocity."""
    env.reset(seed=seed)
    inner = cf._env
    has_wrist = len(cf._pad.groups["WRIST_YAW"]) > 0
    brake_r = float(inner._zone_half) * 1.5                         # brake radius from the target geometry (§4)
    phase = Phase.APPROACH
    ph_box = {"p": phase}

    def _base(innr, _tt):
        d, _n = innr.direction_to_zone()
        p = ph_box["p"]
        dtz_b = float(innr._planar_metrics.disk_to_zone)
        if p is Phase.APPROACH:
            return np.array([d[0], d[1], 0.0, 0.6, 0.0, 0.0], np.float32)
        if p in (Phase.WRIST_ALIGN, Phase.PAD_CLOSE, Phase.FORCE_HOLD):
            return np.array([0.0, 0.0, 0.0, 0.55, 0.0, 0.0], np.float32)          # HOLD grip
        if p is Phase.TRANSPORT:
            scale = float(min(1.0, dtz_b / brake_r))                             # brake profile: slow as the coin nears
            return np.array([d[0] * scale, d[1] * scale, 0.0, 0.6, 0.0, 0.0], np.float32)
        if p in (Phase.BRAKE,):                                                   # CLOSED_HOLD: zero transport, keep grip
            return np.array([0.0, 0.0, 0.0, 0.6, 0.0, 0.0], np.float32)
        return np.array([0.0, 0.0, -0.7, -0.6, 0.0, 0.0], np.float32)            # RELEASE/WITHDRAW: open + still
    env._base_override = _base
    reached = {p.value: False for p in Phase}
    dwell = best_dwell = 0
    min_vel_in_zone = 9.9
    body_shove_ever = certified = False
    cert_step = -1
    steps = dict(min_dtz=9.9, force_hold_steps=0)
    first_fail = "NONE"
    for t in range(max_steps):
        reached[phase.value] = True
        ph_box["p"] = phase
        cf.set_phase(phase if phase is not Phase.BRAKE else Phase.FORCE_HOLD)     # keep closure regulated during hold
        met = inner._planar_metrics
        dtz = float(met.disk_to_zone)
        steps["min_dtz"] = min(steps["min_dtz"], dtz)
        both_pad, fl, fr = _both_pad_contact(cf)
        werr = float(np.mean(cf._pad.log.wrist_err[-2:])) if cf._pad.log.wrist_err else 9.9
        v = inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2]
        cvel = float(np.linalg.norm(v))
        lg = getattr(met, "legality", None)
        body = bool(lg is not None and getattr(lg, "arm_body_contact", False))
        body_shove_ever = body_shove_ever or body
        in_zone = dtz <= _CENTER_TOL
        if in_zone:
            min_vel_in_zone = min(min_vel_in_zone, cvel)
        # strict dwell certificate: consecutive in-zone + low-velocity + no body shove
        if in_zone and cvel < _SETTLE_VEL and not body:
            dwell += 1
            best_dwell = max(best_dwell, dwell)
        else:
            dwell = 0
        if dwell >= _DWELL_REQ and not certified:
            certified, cert_step = True, t
        if not certified and first_fail == "NONE" and in_zone and cvel >= _SETTLE_VEL:
            first_fail = "settle_velocity"                          # in the zone but too fast to accumulate dwell
        # transitions
        if phase is Phase.APPROACH and (met.left_contact or met.right_contact):
            phase = Phase.WRIST_ALIGN
        elif phase is Phase.WRIST_ALIGN and (werr < 0.25 or not has_wrist):
            phase = Phase.PAD_CLOSE
        elif phase is Phase.PAD_CLOSE and both_pad:
            phase = Phase.FORCE_HOLD
        elif phase is Phase.FORCE_HOLD:
            steps["force_hold_steps"] += int(both_pad or (met.left_contact and met.right_contact))  # bilateral contact
            if steps["force_hold_steps"] >= 3:
                phase = Phase.TRANSPORT
            elif not (met.left_contact or met.right_contact):
                phase = Phase.PAD_CLOSE
        elif phase is Phase.TRANSPORT and dtz <= brake_r:
            phase = Phase.BRAKE                                     # BRAKE == CLOSED_HOLD (grip kept, transport zero)
        elif phase is Phase.BRAKE:
            if certified and variant in ("B", "C"):
                phase = Phase.RELEASE
            elif variant == "D":                                   # current: release before dwell certifies
                phase = Phase.RELEASE
            # variant A: stay in CLOSED_HOLD until cert/timeout (never release)
        elif phase is Phase.RELEASE and (fl + fr) < 0.2:
            phase = Phase.WITHDRAW if variant in ("C", "D") else Phase.SETTLE
        elif phase in (Phase.WITHDRAW, Phase.SETTLE):
            phase = Phase.SETTLE
        env.step(np.zeros(env.action_space.shape[0], np.float32))
        if certified and (variant == "A" or (variant != "A" and phase in (Phase.SETTLE, Phase.WITHDRAW))):
            if variant == "A":
                break                                              # A: terminate on certification
    fail = _classify_fail_v2(reached, certified, best_dwell, min_vel_in_zone, both_pad_ever(cf), body_shove_ever)
    return dict(variant=variant, strict=int(certified), cert_step=cert_step, best_dwell=best_dwell,
                min_dtz=round(steps["min_dtz"], 4), min_vel_in_zone=round(min_vel_in_zone, 4),
                body_shove=int(body_shove_ever), first_fail=first_fail, failure=fail)


def both_pad_ever(cf) -> bool:
    fl = np.array(cf._pad.log.force_left or [0.0])
    fr = np.array(cf._pad.log.force_right or [0.0])
    n = min(len(fl), len(fr))
    return bool(n and np.any((fl[:n] > 0.1) & (fr[:n] > 0.1)))


def _classify_fail_v2(reached, certified, best_dwell, min_vel, grasped, body) -> str:
    """§5 taxonomy: GRASP / TRANSPORT / HOLD_SETTLE / RELEASE_DISTURBANCE / STRICT_SUCCESS."""
    if certified:
        return "STRICT_SUCCESS"
    if not grasped:
        return "GRASP_FAILURE"
    if not reached["BRAKE"]:
        return "TRANSPORT_FAILURE"
    if best_dwell >= 1 and reached["RELEASE"]:
        return "RELEASE_DISTURBANCE"
    return "HOLD_SETTLE_FAILURE"


def _clearance(inner) -> float:
    disk_r = float(inner.model.geom_size[inner._disk_geom][0])
    return float(inner.planar_metrics.disk_to_zone) - (disk_r + float(inner._zone_half))


_BANDS = [("+0.018-0.030", 0.018, 0.030), ("+0.030-0.045", 0.030, 0.045),
          ("+0.045-0.060", 0.045, 0.060), ("+0.060-0.080", 0.060, 0.080)]


def _run_cell(geom: str, variant: str, seeds: int) -> dict:
    from collections import Counter
    env, cf = make_wristed_delivery_env(geom)
    by_band = {nm: dict(n=0, strict=0, best_dwell=0, min_vel=9.9, fails=Counter(), first_fail=Counter())
               for nm, _lo, _hi in _BANDS}
    for s in range(seeds):
        env.reset(seed=70_000 + s)
        clr = _clearance(cf._env)
        band = next((nm for nm, lo, hi in _BANDS if lo <= clr < hi), None)
        if band is None:
            continue
        r = oracle_rollout(env, cf, seed=70_000 + s, variant=variant)
        b = by_band[band]
        b["n"] += 1
        b["strict"] += r["strict"]
        b["best_dwell"] = max(b["best_dwell"], r["best_dwell"])
        b["min_vel"] = min(b["min_vel"], r["min_vel_in_zone"])
        b["fails"][r["failure"]] += 1
        b["first_fail"][r["first_fail"]] += 1
    for nm in by_band:
        by_band[nm]["fails"] = dict(by_band[nm]["fails"])
        by_band[nm]["first_fail"] = dict(by_band[nm]["first_fail"])
        by_band[nm]["min_vel"] = round(by_band[nm]["min_vel"], 4)
    return by_band


def _strict030(cell) -> int:
    return sum(cell[nm]["strict"] for nm in cell if not nm.startswith("+0.018"))


def run_2x2(seeds: int, out) -> dict:
    """§1 CERTIFY-BEFORE-RELEASE test: E3 × {A,B,C,D} (release sequencing) + variant A × {E0..E3} (DoF isolation)."""
    import json
    from pathlib import Path
    e3 = {v: _run_cell("E3", v, seeds) for v in ("A", "B", "C", "D")}
    for v, cell in e3.items():
        print(f"[E3-{v}] strict>=+0.030={_strict030(cell)} | "
              + " ".join(f"{nm}:{cell[nm]['strict']}/{cell[nm]['n']}(dwell{cell[nm]['best_dwell']},v{cell[nm]['min_vel']})"
                         for nm in cell), flush=True)
    dof = {g: _run_cell(g, "A", seeds) for g in _GEOMS}
    for g, cell in dof.items():
        print(f"[{g}-A] strict>=+0.030={_strict030(cell)} best_dwell={max(cell[nm]['best_dwell'] for nm in cell)} "
              f"minVel={min(cell[nm]['min_vel'] for nm in cell)}", flush=True)

    a030 = _strict030(e3["A"])
    a_dwell = max(e3["A"][nm]["best_dwell"] for nm in e3["A"])
    a_minvel = min(e3["A"][nm]["min_vel"] for nm in e3["A"])
    grasp_ok = any(f in ("HOLD_SETTLE_FAILURE", "RELEASE_DISTURBANCE", "STRICT_SUCCESS")
                   for nm in e3["A"] for f in e3["A"][nm]["fails"])
    d030 = _strict030(e3["D"])
    if a030 >= 1 and a_dwell >= _DWELL_REQ:
        verdict = "CERTIFY_BEFORE_RELEASE_POSITIVE"
    elif not grasp_ok:
        verdict = "FORCE_CLOSURE_BLOCKED"
    elif a030 == 0 and a_minvel >= _SETTLE_VEL:
        verdict = "HOLD_SETTLE_BLOCKED"
    elif a030 > d030:
        verdict = "RELEASE_DISTURBANCE_CONFIRMED"
    else:
        verdict = "HOLD_SETTLE_BLOCKED"
    summary = dict(verdict=verdict, e3_variants={v: {"strict030": _strict030(e3[v]), "bands": e3[v]} for v in e3},
                   dof_A={g: {"strict030": _strict030(dof[g]), "bands": dof[g]} for g in dof},
                   e3_A_best_dwell=a_dwell, e3_A_min_vel=a_minvel, dwell_required=_DWELL_REQ)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "certify_oracle.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"[oracle] E3-A strict>=+0.030={a030} best_dwell={a_dwell}/{_DWELL_REQ} minVel={a_minvel} | E3-D={d030}\n"
          f"=== VERDICT: {verdict}", flush=True)
    return summary


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--out", default="experiments/2026_07_21_coin_wristed_pad/oracle")
    a = ap.parse_args()
    run_2x2(a.seeds, a.out)



if __name__ == "__main__":
    main()
