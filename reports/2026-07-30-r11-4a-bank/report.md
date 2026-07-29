# R11.4A integration re-measure — the R11.5 starting distribution

**2026-07-29 · grasp-aware DEFAULT capture over the 64-scenario bank · branch `feature/r11-4a-target-conditioned-delivery-teacher` @ `9aaa57c2` · run on kato14 (Linux, CPU) · non-core · no new deps**

## Summary

Re-measure of the full 64-scenario coin/target bank under the **grasp-aware default capture** (the R11.4A integration, `PipelineConfig.grasp_objective=GraspObjective()`), producing the new bank **side-by-side** (the R11.3 bank is untouched) and the metrics that define R11.5's starting distribution. The headline separates the two problems cleanly, for the first time free of reach/capture noise:

> **Grasping is now reliable (58/64 = 91% certified bilateral grasp), with zero nudge-only K6. Delivery is the bottleneck: the frozen R2 delivers only 7/64 of those certified grasps; 51/64 are certified-grasp → delivery-failure.**

## Per-scenario-best (n = 64)

| outcome class | n/64 | share |
|---|---|---|
| **certified bilateral grasp acquired** | **58** | **91%** |
| — of which enter KINETIC | 36 | 56% |
| `K6_WITH_VALID_DELIVERY_MODE` (certified → KINETIC → valid K6) | **7** | 11% |
| `DELIVERY_FAILURE_AFTER_VALID_GRASP` (certified grasp, R2 falls short) | **51** | 80% |
| `SETTLE_FAILURE_AFTER_VALID_GRASP` | 0 | — |
| `K6_WITHOUT_DELIVERY_MODE_TRANSITION` (ungrasped nudge-K6) | **0** | — |
| `CAPTURE_FAIL` (never grasped across seeds) | **6** | 9% |

Attempt-level (185 attempts, 3 seeds/scenario, early-exit on certified grasped-K6): certified 0.519, KINETIC 0.27, valid-K6 0.038 — lower than per-scenario-best because it counts every failing seed; the per-scenario-best is the meaningful figure.

## The three regions

- **Delivered (7, all train split):** `bank_c0_3`, `bank_c3_r5_a-15`, `bank_c1_-0.01_-0.02`, `bank_c2_-0.025_-0.015`, `bank_c2_-0.025_+0.000`, `bank_c2_-0.015_-0.025`, `bank_c2_-0.015_-0.015`. The geometries the frozen R2 already generalizes to are all near-canonical (train). 0/10 dev, 0/9 test.
- **R11.5 target — certified grasp, delivery falls short (51):** delivery `min_dtz` **median 40.6 mm** (min 27.6, max 151.1) vs the 20 mm K6 zone. These are clean downstream counterexamples: certified bilateral grasp → (KINETIC or not) → frozen R2 insufficient target progress. This 51-scenario block is the delivery-generalization training/target set for R11.5.
- **Capture still fails (6):** clusters at `+/+` coin–target offsets (`bank_c1_+0.01_+0.03`, `+0.03_+0.02`, `+0.03_+0.03`, `bank_c2_+0.015_+0.025`, `+0.025_+0.015`, `+0.025_+0.025`) — a specific geometric corner where even the grasp-aware capture doesn't seat a grasp within the seed budget. A small residual capture-support gap, distinct from delivery.

## What this establishes

1. **The grasp-aware objective did its job at population scale:** 91% certified grasp, **nudge-only-K6 eliminated (0)**. Compare R11.3, where the "8/64 K6" was mostly captures that released or nudged.
2. **Grasp reliability and delivery generalization are now cleanly separable.** 51/64 certified grasps that the frozen R2 can't deliver = the R11.5 problem, isolated from capture noise. The non-invasive energy instrumentation built in R11.4A can now work on this certified-grasp input distribution.

## Provenance

- Code: branch `feature/r11-4a-target-conditioned-delivery-teacher` @ **`9aaa57c2`** (grasp-aware ranking `9de5789d` + hybrid elite `1e3e8177` + integration `58251131` + runner `fd16f5c7`/`9aaa57c2`). Rust `hymeko` CLI built from this source on kato14 (cargo 1.96.1).
- Host: **kato14** (Linux x86-64, 32 cores, 125 GB RAM), venv torch 2.12.0 / mujoco 3.10.0 / numpy 2.4.6 (matches the Mac stack). CPU-only (MuJoCo workload; no GPU benefit). 16-worker parallel (single-threaded each), wall ~48 min.
- Data: `bank.jsonl`, 185 attempts, seeds (0, 1, 2)/scenario, sha256 `6cacd30b6780727d5d2034a8b9bf96020add12ef839e7b3a4f8428647687cffe`. R11.3 bank untouched.
- Caveat: the K6 flag is `characterize_delivery`'s windowed settled-zone predicate, which can differ from the reported terminal `deliver_dtz_mm` in edge cases (one delivered scenario reports 24.99 mm); counts use the predicate. Cross-platform (Linux vs the Mac's ARM), so not bit-identical to a Mac run — the rates, not bit-exactness, are the deliverable.

## Next (R11.5)

Target-conditioned delivery teacher over the 51-scenario `DELIVERY_FAILURE_AFTER_VALID_GRASP` set: current coin→target direction, delivery magnitude/duration, braking, release, settle, target-entry energy — trained/evaluated on the certified-grasp starting distribution, with the R11.4A energy instrumentation. Also: the 6 `+/+` `CAPTURE_FAIL` scenarios are a small separate capture-support follow-up.
