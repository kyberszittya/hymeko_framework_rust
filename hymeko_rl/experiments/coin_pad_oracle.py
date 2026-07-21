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

_ACTORS = [DeliveryActor.A0_SYM_PUSH, DeliveryActor.A1_VPLOW, DeliveryActor.A2_ASYM_PUSH,
           DeliveryActor.A3_SETUP_PUSH, DeliveryActor.A4_RECOVERY]
_BANDS = [("P0_+0.018-0.030", 0.024), ("P1_+0.030-0.045", 0.037),
          ("P2_+0.045-0.060", 0.052), ("P3_+0.060-0.080", 0.070)]


def _scripted(actor: DeliveryActor):
    p = ActorParams()

    def fn(inner, t, _obs):
        return actor_action(inner, t, actor, p)
    return fn


def _best_transport(env, snap) -> dict:
    """Run every scripted actor from ``snap``; return the best (strict, persistence, targetward progress)."""
    best = dict(strict=0, both_frac=0.0, progress=-9.9, actor=None)
    for actor in _ACTORS:
        _restore_generated(env, snap)
        tr = rollout(env, _scripted(actor), max_steps=60)
        st = int(bool(policy_strict(tr)))
        if (st, tr.both_frac, tr.progress) > (best["strict"], best["both_frac"], best["progress"]):
            best = dict(strict=st, both_frac=round(float(tr.both_frac), 3), progress=round(float(tr.progress), 4),
                        loose=int(bool(tr.loose)), actor=actor.value)
    return best


def run(seeds: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for geom in ("POINT", "FLAT_PAD"):
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
                                 mean_both_frac=round(float(np.mean([r["both_frac"] for r in rows])), 3) if rows else 0.0,
                                 mean_progress=round(float(np.mean([r["progress"] for r in rows])), 4) if rows else 0.0,
                                 max_strict_clearance=round(max([r["clearance"] for r in rows if r["strict"]],
                                                               default=-9.9), 4))
            print(f"[{geom} {band}] n={by_band[band]['n']} strict={strict} loose={by_band[band]['loose']} "
                  f"both_frac={by_band[band]['mean_both_frac']} prog={by_band[band]['mean_progress']} "
                  f"maxStrictClr={by_band[band]['max_strict_clearance']}", flush=True)
        result[geom] = by_band

    # feasibility signal: FLAT_PAD strict from >= +0.030 that POINT lacks
    def strict_ge_030(res):
        return sum(res[b]["strict"] for b in res if not b.startswith("P0"))
    pad_030 = strict_ge_030(result["FLAT_PAD"])
    point_030 = strict_ge_030(result["POINT"])
    pad_max = max((result["FLAT_PAD"][b]["max_strict_clearance"] for b in result["FLAT_PAD"]), default=-9.9)
    feasible = pad_030 >= 1 and pad_max >= 0.030
    verdict = ("FEASIBLE_PROCEED_TO_RL" if feasible and pad_030 > point_030 else
               "PAD_ORACLE_ONLY" if pad_030 > point_030 else "NO_PAD_ADVANTAGE")
    summary = dict(point_strict_ge_030=point_030, pad_strict_ge_030=pad_030, pad_max_strict_clearance=round(pad_max, 4),
                   point_max_strict_clearance=round(max((result["POINT"][b]["max_strict_clearance"]
                                                         for b in result["POINT"]), default=-9.9), 4),
                   feasible=feasible, verdict=verdict, bands=result)
    (out / "pad_oracle.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"[oracle] POINT strict≥+0.030={point_030}  FLAT_PAD strict≥+0.030={pad_030}  "
          f"pad_max_strict_clr={pad_max:+.4f}\n=== VERDICT: {verdict}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--out", default="experiments/2026_07_21_coin_fingertip_pad/oracle")
    a = ap.parse_args()
    run(a.seeds, Path(a.out))


if __name__ == "__main__":
    main()
