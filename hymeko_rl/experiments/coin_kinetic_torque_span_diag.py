"""Teacher-torque-span diagnostic — WHY the bounded residual saturates ~6 mm short of the 30 mm corridor (learning-free).

Two complementary, read-only analyses (no policy trained, no state edited, no teacher in any deployed loop):

  Part A — authority ceiling: re-run the audit CEM on the A0 per-joint basis at growing α up to full per-step override
    (α = 2.0 ⇒ the residual sets any admissible action, the frozen clone contributes nothing). Reaches ≤ 30 mm ⇒ the wall is the
    residual BOUND (a bigger residual would deliver); still short at full override ⇒ the wall is the SCAFFOLD (the clone's
    contact-decaying state trajectory the residual is bounded around).

  Part B — the span projection: decompose the DELIVERING teacher's per-step action against the clone's counterfactual action at
    the same states, in the shared slew-normalised space, into (i) magnitude vs the residual bound α and (ii) the component
    OUTSIDE the A2 structured basis span. Names the missing correction: too large (magnitude), out-of-basis (direction), or within
    reach (learnability).

Run: ``python -m hymeko_rl.experiments.coin_kinetic_torque_span_diag``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from hymeko_rl.coin_delivery.theta_option import kinetic_authority as ka
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
from hymeko_rl.coin_delivery.theta_option import kinetic_rl3 as krl3
from hymeko_rl.coin_delivery.theta_option import kinetic_torque_span as kts
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import AUG_DIM
from hymeko_rl.coin_delivery.theta_option.residual_option_env import distill_zero_residual
from hymeko_rl.experiments.coin_kinetic_r1_rl import CLONE_CKPT
from hymeko_rl.experiments.coin_kinetic_r2_rl import _clone_augs, _load_clone
from hymeko_rl.option_rl.agents import make_actor

OUT = Path("reports/2026-07-28-coin-r9-torque-span")
THETA_DELIVER = [0.0, 0.2714, -0.057, 16.5378, 8.7978, 3.4604]     # entry+full_cem, K6 True, min_dtz 15.68 mm (K0 artifact)
CEIL_ALPHAS = [0.15, 0.5, 1.0, 2.0]                                # control → full per-step override (α = 2.0)


def _regen_r2_frontiers(snap, model, norm, bounds, *, seed: int, r2_options: int):
    """Deterministically reproduce the clean R2 champion and its healthy frontiers (same path as the audit; the 4-line regen is
    duplicated once — extract to a shared R2 helper on the third occurrence)."""
    def clone() -> CloneActor:
        return CloneActor(model, norm)
    r2_actor = make_actor("td3", AUG_DIM, krl2.ACT_DIM)
    distill_zero_residual(r2_actor, _clone_augs(snap, clone(), bounds), seed=seed)
    champ, _ = krl2.train_perstep("td3", snap, clone(), bounds, krl2.Reward2Weights(),
                                  krl2.PerStepConfig(total_options=r2_options), seed=seed)
    return krl3.capture_frontiers(snap, clone(), champ, bounds)


def _ceiling_sweep(frontiers, model, norm, bounds) -> list[dict]:
    """Part A: A0 per-joint basis (family 'A1') at each α from the healthy frontiers; α = 2.0 is full per-step override. A
    horizon control at the largest α rules out the CEM horizon as the limiter."""
    rows: list[dict] = []
    for f in frontiers:
        for alpha in CEIL_ALPHAS:
            r = ka.authority_cem(f, model, norm, "A1", alpha, bounds=bounds)
            rows.append({"frontier_mm": f.dtz_mm, "horizon": ka.AuthorityCEMConfig().horizon, **r})
        rc = ka.authority_cem(f, model, norm, "A1", 2.0, bounds=bounds, cfg=ka.AuthorityCEMConfig(horizon=22))
        rows.append({"frontier_mm": f.dtz_mm, "horizon": 22, "note": "horizon_control", **rc})
    return rows


def _verdict(ceiling: list[dict], summ: dict) -> dict:
    """Combine Part A (bound vs scaffold) and Part B (magnitude vs direction) into the diagnostic verdict."""
    reached = [r for r in ceiling if r["authority_reachability_pass"]]
    smallest_alpha = min((r["alpha"] for r in reached), default=None)
    if smallest_alpha is None:
        part_a = "SCAFFOLD_LIMITED_FULL_AUTHORITY_INSUFFICIENT"     # even full per-step override misses the corridor
    elif smallest_alpha <= 0.30:
        part_a = "AUDIT_INCONSISTENCY_SMALL_ALPHA_REACHES"          # should not happen — the audit saturated ≤ 0.30
    else:
        part_a = "BOUND_LIMITED_LARGER_RESIDUAL_REACHES"           # a bigger residual reaches; RL with α = smallest_alpha
    direction_missing = summ.get("max_a2_ortho", 0.0) > 1e-3
    magnitude_gap = summ.get("frac_exceeds_2alpha0", 0.0) >= 0.5
    part_b = ("TEACHER_CORRECTION_A2_ORTHOGONAL_DIRECTION_MISSING" if direction_missing else
              ("TEACHER_CORRECTION_MAGNITUDE_GAP_IN_A2_SPAN" if magnitude_gap else
               "TEACHER_CORRECTION_WITHIN_BOUND_LEARNABILITY_GAP"))
    return {"part_a": part_a, "part_b": part_b, "smallest_reaching_alpha": smallest_alpha,
            "direction_missing": direction_missing, "magnitude_gap": magnitude_gap}


def run(seed: int = 0, r2_options: int = 300) -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    model, norm = _load_clone()
    harness = load_harness()
    snap, meta = acquire_snapshot(harness, kc.S1_SEED)
    if snap is None:
        raise SystemExit(f"could not acquire s1 straddle: {meta}")
    entry = kc.freeze_kinetic_entry(snap, seed=kc.S1_SEED)

    # Part B — teacher-vs-clone span projection along the delivering trajectory (uses the frozen K2 clone checkpoint)
    tmodel, tnorm = kts.load_clone(CLONE_CKPT)
    tm, steps = kts.decompose_teacher_vs_clone(entry.tsnap, THETA_DELIVER, tmodel, tnorm)
    summ = kts.summarize_decomposition(steps)

    # Part A — authority ceiling from the healthy R2 frontiers (regenerated deterministically, as in the audit)
    bounds = ResidualBounds(alpha=0.15)
    frontiers, boundary = _regen_r2_frontiers(snap, model, norm, bounds, seed=seed, r2_options=r2_options)
    if not frontiers:
        raise SystemExit(f"no healthy frontiers (boundary {len(boundary)}) — cannot run the ceiling sweep")
    ceiling = _ceiling_sweep(frontiers, model, norm, bounds)

    verdict = _verdict(ceiling, summ)
    out = {"contract": "COIN_KINETIC_TORQUE_SPAN_DIAG_V1", "seed": seed,
           "entry": {"dtz_mm": round(entry.entry_dtz * 1000, 2), "v_par": round(entry.entry_v_par, 4)},
           "theta_deliver": THETA_DELIVER, "teacher_delivers": bool(tm["k6_delivered"]),
           "teacher_dtz_end_mm": round(tm["dtz_end"] * 1000, 2), "frontiers_mm": [f.dtz_mm for f in frontiers],
           "part_b_summary": summ, "part_b_steps": [s.__dict__ for s in steps],
           "part_a_ceiling": ceiling, "verdict": verdict, "wall_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "torque_span.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = run()
    v = r["verdict"]
    print(f"\nTORQUE-SPAN DIAG  part_a={v['part_a']}  part_b={v['part_b']}  (wall {r['wall_s']}s)")
    print(f"  teacher delivers {r['teacher_delivers']} to {r['teacher_dtz_end_mm']}mm; frontiers {r['frontiers_mm']}mm")
    s = r["part_b_summary"]
    print(f"  Part B (transport steps {s['transport_steps']}): mean|d|∞ {s['mean_d_inf']} max {s['max_d_inf']}  "
          f"frac>α0 {s['frac_exceeds_alpha0']} frac>2α0 {s['frac_exceeds_2alpha0']}  "
          f"A2-ortho mean {s['mean_a2_ortho']} max {s['max_a2_ortho']}")
    print("  Part A ceiling (family A1 = A0 per-joint basis; α=2.0 = full override):")
    for row in r["part_a_ceiling"]:
        tag = " [horizon_control]" if row.get("note") == "horizon_control" else ""
        print(f"    α={row['alpha']:<4} h{row['horizon']:<2} frontier {row['frontier_mm']:5}mm → min_dtz {row['min_dtz_mm']:6.1f}mm "
              f"exit_v {row['exit_v_par']:+.3f} s/r/c {row['stalls']}/{row['reversals']}/{row['clamps']} "
              f"REACH {row['authority_reachability_pass']}{tag}")
