"""§4 physical feasibility oracle — does FLAT_PAD fingertip geometry permit stable transport that POINT does not?

Before any RL: run the CANONICAL scripted push/plow actors (actuator-limited, no teleport, no coin manipulation, no
reward override) on matched clear-start states under POINT vs FLAT_PAD fingertip geometry, and measure whether the
finite-area pad produces a clean STRICT transport from signed clearance ≥ +0.030 where the point contact cannot. Fresh
per-geometry states (a FLAT_PAD snapshot is never a restored POINT snapshot); matched seeds give matched coin/arm/target
layouts (only the fingertip geom differs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hymeko_rl.experiments.coin_clearance_curriculum import _clearance
from hymeko_rl.experiments.coin_generator_exp import _restore_generated
from hymeko_rl.experiments.coin_problem_generator import move_to_clearance
from hymeko_rl.experiments.coin_two_arm_sac import direct_env, policy_strict
from hymeko_rl.env.planar_snapshot import snapshot_planar
from hymeko_rl.train.coin_delivery_actor import ActorParams, DeliveryActor, actor_action, rollout

from hymeko_rl.train.coin_delivery_rl import p_grasp_carry

_ACTORS = [DeliveryActor.A0_SYM_PUSH, DeliveryActor.A1_VPLOW, DeliveryActor.A2_ASYM_PUSH,
           DeliveryActor.A3_SETUP_PUSH, DeliveryActor.A4_RECOVERY, "GRASP_CARRY"]   # grasp_carry = the clamp controller
_BANDS = [("P0_+0.018-0.030", 0.024), ("P1_+0.030-0.045", 0.037),
          ("P2_+0.045-0.060", 0.052), ("P3_+0.060-0.080", 0.070)]


def _scripted(actor):
    p = ActorParams()

    def fn(inner, t, _obs):
        return p_grasp_carry(inner, t) if actor == "GRASP_CARRY" else actor_action(inner, t, actor, p)
    return fn


def _best_transport(env, snap) -> dict:
    """Run every scripted actor from ``snap``; return the best (strict, persistence, targetward progress). Also records
    settle velocity + a §4 RELEASE_SETTLE diagnosis (reached the zone loosely but failed the low-velocity dwell)."""
    best = dict(strict=0, both_frac=0.0, progress=-9.9, actor=None, loose=0, settle_vel=1.0, release_settle_fail=0)
    for actor in _ACTORS:
        _restore_generated(env, snap)
        tr = rollout(env, _scripted(actor), max_steps=60)
        st = int(bool(policy_strict(tr)))
        if (st, tr.both_frac, tr.progress) > (best["strict"], best["both_frac"], best["progress"]):
            rs_fail = int(bool(tr.loose) and not st and tr.settle_vel > 0.1)   # in zone but too fast to settle
            best = dict(strict=st, both_frac=round(float(tr.both_frac), 3), progress=round(float(tr.progress), 4),
                        loose=int(bool(tr.loose)), actor=getattr(actor, "value", actor),
                        settle_vel=round(float(tr.settle_vel), 4),
                        release_settle_fail=rs_fail)
    return best


def run(seeds: int, out: Path, geometries: "tuple[str, ...]" = ("POINT", "FLAT_PAD")) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for geom in geometries:
        env = direct_env(fingertip_geometry=geom)
        env._base_override = lambda _i, _t: np.zeros(6, np.float32)
        env._delta_override = 1.0
        by_band = {}
        for band, tgt_clr in _BANDS:
            rows = []
            for s in range(seeds):
                env.reset(seed=64_000 + s)                          # fresh per-geometry state (matched seed)
                snap0 = snapshot_planar(env.inner)
                snap = move_to_clearance(snap0, tgt_clr, lateral=((s % 3) - 1) * 0.02)   # push coin outward to the band
                _restore_generated(env, snap)
                clr = _clearance(env.inner)
                if clr < 0.018:                                     # generation didn't reach the band → skip
                    continue
                b = _best_transport(env, snap)
                b["clearance"] = round(float(clr), 4)
                rows.append(b)
            strict = sum(r["strict"] for r in rows)
            by_band[band] = dict(n=len(rows), strict=strict,
                                 loose=sum(r.get("loose", 0) for r in rows),
                                 release_settle_fail=sum(r.get("release_settle_fail", 0) for r in rows),
                                 mean_both_frac=round(float(np.mean([r["both_frac"] for r in rows])), 3) if rows else 0.0,
                                 mean_settle_vel=round(float(np.mean([r["settle_vel"] for r in rows])), 4) if rows else 1.0,
                                 mean_progress=round(float(np.mean([r["progress"] for r in rows])), 4) if rows else 0.0,
                                 max_strict_clearance=round(max([r["clearance"] for r in rows if r["strict"]],
                                                               default=-9.9), 4))
            print(f"[{geom} {band}] n={by_band[band]['n']} strict={strict} loose={by_band[band]['loose']} "
                  f"rs_fail={by_band[band]['release_settle_fail']} both_frac={by_band[band]['mean_both_frac']} "
                  f"settleV={by_band[band]['mean_settle_vel']} maxStrictClr={by_band[band]['max_strict_clearance']}",
                  flush=True)
        result[geom] = by_band

    def strict_ge_030(res):
        return sum(res[b]["strict"] for b in res if not b.startswith("P0"))

    def max_strict(res):
        return round(max((res[b]["max_strict_clearance"] for b in res), default=-9.9), 4)
    per_geom = {g: dict(strict_ge_030=strict_ge_030(result[g]), max_strict_clearance=max_strict(result[g]),
                        rs_fail=sum(result[g][b]["release_settle_fail"] for b in result[g])) for g in geometries}
    # §3/§12 clamp verdict: the target geometry (last in the list) vs the others
    tgt = geometries[-1]
    baselines = [g for g in geometries if g != tgt]
    tgt_030 = per_geom[tgt]["strict_ge_030"]
    base_030 = max((per_geom[g]["strict_ge_030"] for g in baselines), default=0)
    tgt_max = per_geom[tgt]["max_strict_clearance"]
    advantage = tgt_030 > base_030 and tgt_030 >= 1 and tgt_max >= 0.030
    if tgt == "CONCAVE_CLAMP":
        any_transport = any(result[tgt][b]["loose"] > 0 or result[tgt][b]["mean_progress"] > 0.02 for b in result[tgt])
        verdict = ("CLAMP_FEASIBLE_PROCEED_TO_RL" if advantage else
                   "RELEASE_SETTLE_LIMITED" if (tgt_030 == 0 and per_geom[tgt]["rs_fail"] > base_030) else
                   "NO_FORCE_CLOSURE" if not any_transport else "NO_CLAMP_ADVANTAGE")
    else:
        verdict = "FEASIBLE_PROCEED_TO_RL" if advantage else "NO_PAD_ADVANTAGE"
    summary = dict(per_geom=per_geom, target=tgt, advantage=advantage, verdict=verdict, bands=result)
    (out / "clamp_oracle.json").write_text(json.dumps(summary, indent=1, default=float))
    for g in geometries:
        print(f"[oracle] {g:14s} strict≥+0.030={per_geom[g]['strict_ge_030']} "
              f"maxStrictClr={per_geom[g]['max_strict_clearance']:+.4f} rs_fail={per_geom[g]['rs_fail']}", flush=True)
    print(f"=== VERDICT: {verdict}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--geometries", default="POINT,FLAT_PAD,CONCAVE_CLAMP")
    ap.add_argument("--out", default="experiments/2026_07_21_coin_concave_clamp/oracle")
    a = ap.parse_args()
    run(a.seeds, Path(a.out), tuple(a.geometries.split(",")))


if __name__ == "__main__":
    main()
