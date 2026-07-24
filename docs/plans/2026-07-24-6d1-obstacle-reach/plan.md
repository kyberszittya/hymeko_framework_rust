# 6D-1 obstacle reach — the falsifiable test of multimodal policy search on topologically-disjoint route modes

**Created-at:** 2026-07-24 22:15 JST
**ETA:** env + disjoint-mode validation ≈ 40 min (the gate); the full budget×K×allocation grid + report ≈ +40 min.
**Design source:** the user's 2026-07-24 spec (verbatim requirements below); this doc adds the engineering + risk.

## Scope / goal
The single open question after 6D-0 (`RUNTIME_GENERAL_ON_SE3`) and Track B (`KMODE_NO_DEPLOY_ADVANTAGE` on boxes):
**does the multimodal proposal help when the working solutions are genuinely distant, topologically-separated routes?**
A contact-free obstacle reach isolates this from contact physics / friction / unstable teacher / retention / object
rotation / mesh+inertia error — a cleaner decisive test than the O3 triangle.

## Env (SE3ObstacleReachEnv(SE3ReachEnv))
- Inject a static box obstacle `hk_obstacle` into the worldbody (reuse the `with_collision_floor` injection pattern);
  at reset seat it (via `model.geom_pos`) between the start-EE and goal-EE so the DIRECT EE path is blocked.
- Committed option = an **EE-waypoint path** `start → via → goal` (interpolated, each waypoint IK'd + servoed) so the
  EE follows the intended route closely (a joint-space straight line would swing through the box and muddy the modes).
- **Route modes are genuinely disjoint:** a mode = a via-point = goal ± a lateral offset in one direction
  {left −y, right +y, over +z, (under −z)}. Left-via and right-via paths are different homotopy classes around the box;
  local jitter around a left-via stays left → single-head anchored to one side CANNOT reach via the other. This is the
  property that makes the test decisive (and the risk to validate first).
- Certificate (unchanged shape): goal pose reached (pos ∧ ang) ∧ EE-path collision-free ∧ stable final.

## The three allocation strategies WITHOUT unfreezing the runtime
`MultimodalBudgetSearch` stays frozen. `allocate_budget` already gives ≥1/mode + remainder ∝ prob, so the allocation
strategy is just how the proposal sets **mode probabilities**:
- **A. probability-weighted** — modes carry the true classifier/route probs.
- **B. equal-minimum per mode** — modes carry equal probs (1/K) ⇒ equal split B/K.
- **C. top-refined + one probe per alternate** — one dominant prob + tiny alternates ⇒ [bulk, 1, 1, …]. Likely the
  most sensible (every mode gets ≥1 physical probe; the rest refines the most promising).

## Matched controls (identical across arms)
start/goal panel · obstacle geometry · total candidate budget · random seeds · certificate · latency reporting.

## Measured
success · collision · route-family coverage · alternate-mode recovery · success-per-budget · selected-mode calibration ·
latency · direct-proposal vs search-selected. **Separate (mode correctness ≠ deploy success):** mode recall ·
within-mode localization · physical success.

## Design matrix
budget B ∈ {0, 4, 8, 12, 24} × K ∈ {1, 2, 3, 4} × allocation ∈ {A, B, C}. K=1 = single-head baseline (allocation inert).

## Pre-registered verdicts (from the user)
- K-mode improves at EQUAL budget → **MULTIMODAL_POLICY_SEARCH_VALIDATED_ON_DISJOINT_PATH_MODES** (runtime hypothesis
  correct; O2's wall was contact candidate-localization) → then O3 triangle → pick-place → AIBO.
- Only improves with LARGER budget → **MULTIMODAL_REPRESENTATION_VALID + SEARCH_BUDGET_DOMINANT** (modes useful,
  allocation mechanism not yet good enough).
- Explicit modes exist but K-mode doesn't improve → **MODE_ROUTING_OR_TRAINING_INSUFFICIENT** (proposal-head / teacher-
  routing weak, not necessarily the runtime).
- No multimodal advantage at all → **MULTIMODAL_PROPOSAL_NOT_ESTABLISHED_AS_DEPLOY_LEVER** (no central learning claim on
  it; runtime stays a general interface, main strength = certified bounded search).

## Affected files (planned)
- `hymeko_rl/env/se3_obstacle_reach_env.py` (new) — SE3ObstacleReachEnv + obstacle injection + EE-path collision.
- `hymeko_rl/env/se3_reach_option.py` (extend) — EE-waypoint-path committed option + RouteModeProposal(allocation).
- `hymeko_rl/experiments/se3_obstacle_6d1.py` (new) — the matched grid + metrics + plot + GIF.
- tests: `hymeko_rl/tests/test_se3_obstacle_reach.py` (env + disjoint-mode property + allocation-prob mapping).

## CORE.YAML items touched
None (verified: CORE.YAML lists no `hymeko_rl` env/option_rl paths).

## Risk / what to validate FIRST (the gate)
The whole experiment is meaningless if the env's route modes are not genuinely disjoint. **Discriminating test before
any grid:** construct a state where the direct + one lateral route are blocked; assert single-head anchored to the
blocked side fails while a K-mode proposal covering the open side recovers. If modes are NOT disjoint (single-head jitter
crosses to the other route), fix the geometry / execution before measuring. Worst-case: 4-DOF reachability may not admit
all of {left,right,over} for every state — sample states where ≥2 routes are feasible, and LOG per-state route feasibility.

## Rollback
New files + an additive proposal class; delete the new modules and the SE3ObstacleReachEnv to revert. Runtime + 6D-0
unaffected.

## Frozen (per user)
runtime API · coin/object baselines · pick-place & AIBO learning campaigns. O3 mesh = environment prep only, no campaign.
