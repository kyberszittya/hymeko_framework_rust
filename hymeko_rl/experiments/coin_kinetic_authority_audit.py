"""Authority audit (learning-free) — is ≤ 30 mm reachable from a healthy R2 frontier, and with what residual authority?

Runs a bounded CEM reachability search from the healthy R2-champion frontiers (57.3, 53.84 mm) for three authority families —
A0 (current 4-D per-joint, α = 0.15), A1 (same basis, larger α), A2 (structured coin-following basis) — and reports the SMALLEST
family that reaches `AUTHORITY_REACHABILITY_PASS`. No policy is trained. This decides whether the R3-B wall is (a) a learnability
gap in the current basis (A0 already reaches), (b) a residual-bound gap (A1 reaches), (c) an action-basis gap (only A2 reaches),
or (d) none reach (RL not yet justified — inspect the teacher-torque span).

Run: ``python -m hymeko_rl.experiments.coin_kinetic_authority_audit``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from hymeko_rl.coin_delivery.theta_option import kinetic_authority as ka
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
from hymeko_rl.coin_delivery.theta_option import kinetic_rl3 as krl3
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import AUG_DIM
from hymeko_rl.coin_delivery.theta_option.residual_option_env import distill_zero_residual
from hymeko_rl.experiments.coin_kinetic_r2_rl import _clone_augs, _load_clone
from hymeko_rl.option_rl.agents import make_actor

OUT = Path("reports/2026-07-28-coin-r9-authority-audit")
FAMILIES = [("A0", 0.15), ("A1", 0.20), ("A1", 0.25), ("A1", 0.30), ("A2", 0.25)]   # ordered smallest → largest authority


def run(seed: int = 0, r2_options: int = 300) -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    model, norm = _load_clone()
    harness = load_harness()
    snap, meta = acquire_snapshot(harness, kc.S1_SEED)
    if snap is None:
        raise SystemExit(f"could not acquire s1 straddle: {meta}")
    bounds = ResidualBounds(alpha=0.15)

    def clone_factory() -> CloneActor:
        return CloneActor(model, norm)

    r2_actor = make_actor("td3", AUG_DIM, krl2.ACT_DIM)
    distill_zero_residual(r2_actor, _clone_augs(snap, clone_factory(), bounds), seed=seed)
    r2_champ, _ = krl2.train_perstep("td3", snap, clone_factory(), bounds, krl2.Reward2Weights(),
                                     krl2.PerStepConfig(total_options=r2_options), seed=seed)
    frontiers, boundary = krl3.capture_frontiers(snap, clone_factory(), r2_champ, bounds)
    if not frontiers:
        raise SystemExit(f"no healthy frontiers (boundary {len(boundary)}) — cannot audit authority")

    results: list[dict] = []
    for f in frontiers:
        for fam, alpha in FAMILIES:
            r = ka.authority_cem(f, model, norm, fam, alpha, bounds=bounds)
            results.append({"frontier_mm": f.dtz_mm, **r})

    def _passes(fam: str, alpha: float) -> bool:                  # reachable from at least one healthy frontier
        return any(r["family"] == fam and r["alpha"] == round(alpha, 3) and r["authority_reachability_pass"] for r in results)

    smallest = next(((fam, a) for fam, a in FAMILIES if _passes(fam, a)), None)
    if smallest is None:
        verdict = "CURRENT_RESIDUAL_AUTHORITY_INSUFFICIENT"       # none reach — inspect the teacher-torque span before RL
    elif smallest[0] == "A0":
        verdict = "CURRENT_BASIS_REACHES_RL_LEARNABILITY_GAP"     # a solution exists at α=0.15 — the RL, not the basis, fell short
    elif smallest[0] == "A1":
        verdict = "EXPANDED_KINETIC_AUTHORITY_REACHES_CLOSE_MOVING_MANIFOLD"   # larger α reaches; no new basis needed
    else:
        verdict = "EXPANDED_BASIS_REQUIRED_STRUCTURED_COIN_FOLLOWING"          # only the structured basis reaches

    out = {"contract": "COIN_KINETIC_AUTHORITY_AUDIT_V1", "seed": seed, "corridor_mm": ka.CORRIDOR_MM,
           "frontiers_mm": [f.dtz_mm for f in frontiers], "families": [{"family": f, "alpha": a} for f, a in FAMILIES],
           "results": results, "smallest_passing": ({"family": smallest[0], "alpha": smallest[1]} if smallest else None),
           "verdict": verdict, "wall_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "authority_audit.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = run()
    print(f"\nAUTHORITY AUDIT VERDICT: {r['verdict']}  (smallest passing {r['smallest_passing']}; wall {r['wall_s']}s)")
    print(f"  frontiers {r['frontiers_mm']}mm  corridor ≤ {r['corridor_mm']}mm\n")
    for res in r["results"]:
        print(f"  {res['family']}@{res['alpha']:<4} frontier {res['frontier_mm']:5}mm  →  min_dtz {res['min_dtz_mm']:6.1f}mm  "
              f"exit_v {res['exit_v_par']:+.3f}  exit_fn {res['exit_fn']:.3f}  stall/rev/clamp {res['stalls']}/{res['reversals']}/{res['clamps']}  "
              f"safe {res['safe']!s:5}  REACH {res['authority_reachability_pass']}")
