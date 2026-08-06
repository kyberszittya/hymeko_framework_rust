# Farther-start transport — amendment to the geometry-generalization design (SPEC + measured preconditions)

Amends the post-factorial geometry experiment (companion to `next_factorial_spec.md`, sha 58b63b72). **The
actor×critic factorial is NOT running** — the n-step task ended BLOCKED and the factorial was not launched; this file
is a design-of-record amendment plus the concrete preconditions I could measure now, not a change to a live run.

## Clearance bands (physically disjoint footprints; base radius 0.020, footprint touch at dtz=0.060)
| band | signed clearance | dtz range |
|---|---|---|
| NEAR (unchanged factorial corpus) | +0.030 – 0.060 | 0.090 – 0.120 |
| MEDIUM | +0.060 – 0.100 | 0.120 – 0.160 |
| FAR | +0.100 – 0.160 | 0.160 – 0.220 |
| EXTENDED (optional; reject on reachability/workspace fail) | +0.160 – 0.240 | 0.220 – 0.300 |

The NEAR factorial corpus stays fixed for comparability. Target/strict thresholds/control timestep unchanged.

## MEASURED preconditions (this is what gates the amendment, done 2026-07-21 on HEAD f1d31e8)

### 1. Physical reachability of generated far starts (30 samples/band, `move_to_clearance` + `validate`)
| band | valid (reachable, disjoint, deterministic) |
|---|---|
| NEAR | 30/30 |
| MEDIUM | 30/30 |
| FAR | 28/30 |
| EXTENDED | **10/30** (20 rejected on workspace/reachability — EXTENDED is largely out of the K0 workspace) |

### 2. Horizon sufficiency → **the horizon is NOT the limiter**
No scripted/oracle primitive transports the coin to the zone from ANY band, even at a 200-step probe (3.3× the 60-step
rollout horizon): grasp_carry, A0_sym_push, A3_setup_push, A1_v_plow all reach the zone **0/15** at NEAR/MEDIUM/FAR.
Per §"Episode horizon" ("increase the horizon only when demonstrably too short for a **valid** trajectory"), there is
**no demonstrable valid scripted trajectory whose duration exceeds the horizon** — so a horizon increase is **not
justified by the measurement**, and would be changing a controlled task parameter without cause. Report the horizon as
unchanged (60 rollout / `_C1_HORIZON` env), controlled and equal across arms.

### 3. Why no scripted trajectory exists (the real generation gap)
`move_to_clearance` displaces ONLY the coin (qpos[4:6]); the arms stay in the adjacent-coin pose the parent was captured
in. So a far start has the coin outside the fingertips with the arms NOT positioned at it — the scripted primitives
(which push/carry an already-pinched coin) have nothing to act on. A far start is therefore an **approach-THEN-transport**
problem, not the last-increment placement the current post-acquisition policy + primitives do. This is the same wall the
n-step experiment hit at +0.030 (STAGE2 unreached by both arms), and MEDIUM/FAR/EXTENDED are strictly harder.

## Required generator change BEFORE farther-start training is meaningful
A valid farther-start config must reposition the arms to a **pre-approach pose** consistent with the displaced coin (or
generate from a pre-acquisition parent), so an approach+transport trajectory is at least geometrically possible. Without
this, farther-start training targets configs no policy/primitive can act on, and the metrics would all read 0 — an
artifact, not a measurement. This arm-repositioning is a generator refinement, on the canonical `PlanarSnapshot` path,
to add when the geometry experiment is built.

## Balanced corpus (when built): radius {0.015,0.020,0.025} × clearance {NEAR,MEDIUM,FAR[,EXTENDED where valid]} ×
target sector {L,C,R} × start mode {left-lead,sym,right-lead}; every valid radius×band gets train + hash-disjoint held
examples; NEAR must not dominate the aggregate (report per band, never pooled).

## Progressive NEAR→MEDIUM→FAR→EXTENDED, one continuous policy, retain earlier configs+certified trajectories; advance on
repeatable loose entry OR ≥1 reproducible certified delivery (not target-distance alone); stop a band that stays
unreachable with no target-progress across the eval window.

## Distance-specific reporting (per band, never pooled): certified coverage, loose coverage, mean target-distance
reduction, max reproducibly certified clearance, first-contact success, bilateral rate, fingertip attribution, clean
mechanism, timeout rate.

## Presentation criterion: headline demo starts at signed clearance ≥ +0.080 (MEDIUM presentation-ready; > +0.100
preferred strong transport), ≥8/10 certified, genuine bilateral + clean mechanism, identical initial-state hash;
comparison video over {initial ckpt, 1×1 baseline, selected factorial policy, zero-action}, first frame paused with the
measured clearance. **A +0.030 result is NOT long-range transport and will not be labeled as such.**

## Honest status
Recorded as design; NOT executed. Preconditions measured: far starts are physically generatable (except most of
EXTENDED), the horizon is not the blocker, and the generator needs arm-repositioning before farther-start training is a
real test rather than a guaranteed-zero artifact. The factorial itself remains unbuilt (see `next_factorial_spec.md`).
