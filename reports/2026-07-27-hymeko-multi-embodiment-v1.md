# HyMeKo Multi-Embodiment Integration — v1

**Date:** 2026-07-27 (JST)
**Branch:** `integration/hymeko-multi-embodiment-v1` (`b9d02d0d`)
**Base → Profile → Core:** `819f35fc` → `2210e4c9` (`hymeko-control-profile-v0`) → `66d0d7a3` (`hymeko-control-core-v1`)

---

## What this is

One common **HyMeKo control language + CIP-0 runtime profile** (a stdlib-only,
torch-free, scenario-agnostic shared core), instantiated sequentially on three
embodiments — pick-and-place, humanoid, AIBO — each as an independent adapter
that depends on the core and never the reverse. All work lives in separate
worktrees; the main tree (coin-toss, another session) was never touched.

## Final conformance matrix

| dimension | PNP | HUM | AIBO |
|---|:--:|:--:|:--:|
| schema validation | ✓ | ✓ | ✓ |
| causal observation | ✓ | ✓ | ✓ |
| mode legality | ✓ | ✓ | ✓ |
| intent bounds | ✓ | ✓ | ✓ |
| authority provenance | ✓ | ✓ | ✓ |
| decoder determinism | ✓ | ✓ | ✓ |
| option provenance | ✓ | ✓ | ✓ |
| external certificate | ✓ | ✓ | ✓ |
| no hidden state edit | ✓ | ✓ | ✓ |
| core isolation | ✓ | ✓ | ✓ |
| **scenario smoke** | **PNP-4** certified acquire-carry-place | **HUM-1** (balance vacuous) | **AIBO-2** (+ certified on-axis approach-stop-hold) |

**Integrated test suite: 30 passed / 0 failed** (15 core conformance + 5×3
scenario conformance). `ruff` clean. Import isolation verified: core imports no
scenario / no torch; no scenario imports another (all `[]`).

## Highest gate per scenario (honest)

- **Pick-and-place → PNP-4 (complete, externally certified).** `PickPlaceEnv` +
  scripted expert v3, `require_settle`+`target_bin`: APPROACH→GRASP→LIFT(0.060 m)→
  CARRY→PLACE→RELEASE→SETTLE, `placed_stable`, certificate passed, no drop/death.
  Tagged `cip-pick-place-v0`. **Simulation.**
- **Humanoid → HUM-1 (genuine ceiling).** The only HyMeKo humanoid is **fixed-base**
  (welded pelvis) so balance/support-margin/no-fall is **vacuous**; a fixed-base
  stand→reach→recover cycle runs and is limit/divergence-certified, but HUM-2/3/4
  are **blocked** (need a floating base + balance controller). **No tag.**
- **AIBO → AIBO-2 (genuine ceiling).** Constructed 22-DOF ERS-1000 sim; a NEW
  scripted `SteeredTrotGait` **adds yaw** (measured −3.64 °/s; forward 0.115 m/s);
  on-axis approach→stop→hold is certified. Robust arbitrary-bearing align (AIBO-3/4)
  is **blocked** (one-directional scripted yaw stability; no hardware). **No tag.**
  **Simulation, never claimed as hardware.**

## Physical vs simulated

**All three scenarios are simulated.** No physical hardware (arm, humanoid, or
AIBO) was accessed. No simulated result is labeled as hardware success.

## Core promotions (into `hymeko-control-core-v1`)

Promoted (additive, stdlib-only, reward-independence preserved, 15 tests):

1. `threshold_certificate(extract, lower/upper)` — the recurring "keep a quantity
   within a bound" predicate, parameterised by a caller-supplied extractor so the
   core names no scenario signal. Implementers: AIBO `speed_bounded_at_stop`,
   pick-place bounded-terminal, humanoid joint-velocity.
2. `stability_certificate(uprightness, min)` — generic no-fall / support safety
   cert; genuine implementer = AIBO free-base uprightness; a floating-base humanoid
   is the second implementer.

Deferred (not promoted): the humanoid support cert (vacuous on a fixed base until a
floating base exists), and the MuJoCo-dependent PD realizer / response-audit helper
(scenario-side, not stdlib core).

## Merge provenance

`--no-ff` throughout, `rerere` enabled. Order: core→{pick_place,humanoid,aibo},
then {pick-place,humanoid,aibo}→multi-embodiment. Merges were clean (disjoint
paths). No scenario regressed post-merge (20/20 each before integration; 30/30
integrated). Details in `merge_provenance.json`.

## RL readiness (per scenario) — no RL was run

| scenario | RL authorised? | prerequisite |
|---|---|---|
| pick-place | Not yet | learned arm pick-place checkpoints absent in checkout; the scripted baseline is the certified teacher. Update-zero + unchanged certificate discipline before any RL. |
| humanoid | No | floating-base humanoid + a balance/support-shift controller producing a certified stand-reach-recover baseline. |
| aibo | No | a robust bidirectionally-steerable gait (tuned or the branch's SAC walk policy) and/or physical AIBO + SDK, with a certified deterministic approach-align-stop baseline. |

This campaign's deliverable was the **language, adapters, certificates and baseline
trajectories** — not SAC/TD3. That is what shipped.

## Honest overall verdict

The common CIP-0 contract is real, tested (30/30), and genuinely reused across three
very different embodiments with one-way dependencies enforced by a machine check.
Pick-and-place reaches a full externally-certified trajectory. Humanoid and AIBO
deliver the complete contract + adapter boundary + conformance and stop at their
**genuine** ceilings, with the exact missing prerequisite recorded in each case —
no manufactured physical success, no faked hardware, no unearned RL.

## Exact next action per scenario

- **Pick-place:** wire a learned decoder (retrain the arm pick-place policy; the
  scripted expert is the update-zero teacher) — or extend to multi-seed success-rate.
- **Humanoid:** add a free-joint root to `humanoid.hymeko` (or adopt `Humanoid-v5`)
  and a balance controller; then HUM-2..4 become testable.
- **AIBO:** tune the steered trot for stable bidirectional yaw (or load the SAC walk
  policy) to make robust off-axis align+reach; and/or connect physical AIBO for a
  hardware command-response contract.

## Artifacts

`reports/2026-07-27-hymeko-multi-embodiment-v1/{shared_contract, scenario_matrix,
core_promotions, merge_provenance, final_tests}.json`; per-scenario reports +
plots/gifs under each `reports/2026-07-27-cip-*`.
