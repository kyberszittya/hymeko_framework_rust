# Galambos reverse curriculum — an honest null result

*2026-06-20 · Aiko (Claude Code) for Dr. Csaba Hajdu*
*Plan: [docs/plans/2026-06-20-galambos-curriculum/](../docs/plans/2026-06-20-galambos-curriculum/)*

## Summary

Added a reverse (start-state) curriculum to the Galambos planar grasper — coin spawned
near the zone early, annealed to the full range — to lift the 5/8 goal rate. **It did
not.** The curriculum is correctly implemented and tested, but the goal rate is
**unchanged at 5/8**, and it is the *same* 5 episodes that succeed and 3 that fail.
Reported as a null, with the corrected diagnosis of why.

## What was built (kept — sound, tested infrastructure)

- `PlanarGraspEnv(difficulty ∈ [0,1])` — caps how far from the zone the coin spawns;
  `difficulty=1` is byte-identical to the prior full-range sampling (so the baseline
  comparison is fair and other consumers are unaffected).
- `train_ppo(..., on_iteration=callback)` — a general Observer hook called once per
  iteration; the seat for any curriculum that mutates env state on a schedule.
- `train_planar_grasp --curriculum-iters K` — anneals `difficulty` 0→1 over the first
  `K` iterations via the hook.

Tests: difficulty-spawns-nearer, default-preserves-range, out-of-range-rejected,
hook-called-per-iteration-with-indices. 16/16 in the two modules; ruff + mypy clean.

## Result (retrain 150 it, curriculum 0→1 over 60 it; eval at difficulty=1)

| | Baseline (freed shoulder) | + curriculum |
|---|---|---|
| Goals reached | **5 / 8** | **5 / 8** (unchanged) |
| Episodes that succeed | ep 0,3,5,6,7 | ep 0,3,5,6,7 (identical) |
| Median coin displacement | 0.055 m | 0.063 m |
| Both-finger contact | 0 | 0 |

Training return ran 10.85 → 1.15 (high early on easy coins, lower as difficulty
annealed up — expected, not a degradation).

## Why it was null (corrected diagnosis)

The prior report framed the 3 misses as "far-spawn." The diagnostic shows they are
**heterogeneous control-precision failures**, which a start-state curriculum cannot
touch:

- **ep4 — overshoot:** pushes the coin 0.171 m but ends 0.093 m from the zone.
- **ep2 — wrong direction:** moves it 0.060 m yet disk→zone barely changes (0.093→0.092).
- **ep1 — undershoot:** reaches 0.077 m and stalls short of the 0.055 m zone.

The successful episodes were already robust to start state; the failures are about
*precision of the push*, not *where the coin starts*. Making the start easier does not
teach a more precise push, so the curriculum moved nothing for exactly these cases. The
"far-spawn" correlation in the prior report was real but not the operative cause.

## Files touched

| File | Δ | Note |
|------|---|------|
| `hymeko_rl/env/planar_grasp_env.py` | +~12 | `difficulty` knob + capped spawn |
| `hymeko_rl/ppo.py` | +~8 | `on_iteration` Observer hook (general) |
| `hymeko_rl/train_planar_grasp.py` | +~8 | `--curriculum-iters` schedule |
| `hymeko_rl/tests/{test_planar_grasp_env,test_ppo}.py` | +~35 | curriculum + hook tests |

## CORE.YAML / dependencies

**None.** All under `hymeko_rl/` (non-core). The `ppo.py` change is additive
(`on_iteration` defaults to `None`).

## Decision: keep or revert

**Kept.** The `difficulty` knob and the `on_iteration` hook are correct, tested, general
infrastructure (the hook is reusable for any scheduled env mutation), and `difficulty=1`
is a behaviour-neutral default. Reverting would discard tested capability to undo a
hypothesis that simply didn't pay off. The null is a result, not a regression.

## Next lever (not started — named, not done)

The residual failures are push-precision. The targeted levers, in order:
1. **Settle/overshoot reward shaping** — a small velocity penalty on the coin when it is
   near the zone, so the policy decelerates instead of overshooting (addresses ep4) and a
   directional term (addresses ep2). This is reward work, not curriculum.
2. **More compute** — the same policy may close the gap with more iterations; cheap to test.
3. **A true two-sided pinch** (currently the coin is *pushed*, `both_contact`=0) would give
   finer control than open-loop pushing, but is a larger change.

## Provenance

Git branch `soma-vision`; tree dirty (pre-existing unrelated changes). CPU MuJoCo, no
GPU. Seeds fixed (env seed 0; diagnostic seeds 1000–1007). Checkpoints:
`ppo_curriculum.pt` (this run), `ppo_freed.pt` (baseline, 5/8).
