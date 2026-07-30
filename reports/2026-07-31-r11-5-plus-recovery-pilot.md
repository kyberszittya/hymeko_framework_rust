# R11.5+ Phase 5 — 3-arm Residual-Recovery Pilot

**Date:** 2026-07-31
**Verdict:** `R11_5_PLUS_PIPELINE_PASS_RESIDUAL_RECOVERY_INSUFFICIENT` — infrastructure works, recovery short
(**1/12 recovered**, gate ≥ 6/12). **Stop without the full residual run** (per the pre-registered rule).

## Setup
The taxonomy's bounded 12 (4 capture-support, 4 negative-x `CONTACT_LOSS`, 4 `INSUFFICIENT_TRANSPORT_PROGRESS`). The 4
capture-support are the established Phase-2 honest-negative (bilateral contact never forms) and are short-circuited (not
re-run). The 8 deliverable scenarios each run three arms on the same certified grasp: frozen R2
(`characterize_delivery`), single-stage (`solve_delivery`, R = 5), and two-stage (`solve_delivery_two_stage`,
TARGET_RELATIVE_ALIGNMENT_PHASE, R = 5). `two_stage_adds` = two-stage K6 where single-stage is not (isolates ALIGN).

## Results (8 deliverable; R2 K6 = False for all)
| scenario | class | single dtz (mm) | two-stage dtz (mm) | align verdict | recovered |
|---|---|---|---|---|---|
| bank_c1_-0.01_+0.03 | CONTACT_LOSS | 172.04 | 172.04 | ALIGNMENT_FAILURE | ✗ |
| bank_c1_-0.03_+0.02 | CONTACT_LOSS | 190.71 | 190.71 | ALIGNMENT_FAILURE | ✗ |
| bank_c1_-0.03_+0.03 | CONTACT_LOSS | 159.34 | 159.34 | ALIGNMENT_FAILURE | ✗ |
| bank_c2_-0.025_+0.015 | CONTACT_LOSS | 137.39 | 137.39 | ALIGNMENT_FAILURE | ✗ |
| bank_c3_r9_a-45 | INSUFFICIENT | 56.12 | 56.12 | ALIGNED | ✗ |
| bank_c3_r9_a+15 | INSUFFICIENT | 36.48 | 36.48 | ALIGNED | ✗ |
| bank_c3_r7_a+15 | INSUFFICIENT | 22.21 | 22.21 | ALIGNED | ✗ (2.2 mm short of the 20 mm zone) |
| **bank_c2_-0.015_+0.015** | INSUFFICIENT | **18.60** | 18.60 | ALIGNMENT_FAILURE | **✓ (K6, via single-stage)** |

Gate: recovered **1/12**; negative-x recovered **0/4**; **two_stage_adds 0**; safety_ok true; energy_complete true.

## Findings
1. **The two-stage ALIGN adds nothing (`two_stage_adds = 0`).** It never beat single-stage. On every `CONTACT_LOSS` case
   the align phase itself loses the grasp (`ALIGNMENT_FAILURE`, adverse ~145° push geometry); on every `INSUFFICIENT`
   case the two-stage ties single-stage (`two_stage_dtz == single_dtz`). The two-stage never regressed (its guarantee),
   but it never helped — a clean negative for the new coordinate on these residuals.
2. **The one recovery is grasp-quality, not the coordinate.** `bank_c2_-0.015_+0.015` reached **18.6 mm K6 via
   single-stage** — yet its taxonomy replay stalled at 70.5 mm. The difference is the **grasp** (a different, more
   deliverable certified grasp from the pilot's capture): its align verdict was `ALIGNMENT_FAILURE`, so the two-stage
   did not contribute. This **softens "coordinate-bound" to "grasp-quality-sensitive"** for some INSUFFICIENT cases.
3. **CONTACT_LOSS remains unrecoverable** in scope: single, two-stage, and (from the audit) extended transport all leave
   the coin 137–191 mm out; the grasp squirts backward.

## Interpretation
The pilot confirms the residuals do not recover under the sanctioned Phase-4 coordinate. But finding (2) is a genuine
open thread: **capture-grasp quality**, not the transport coordinate, gates at least some INSUFFICIENT recoveries. That
points the R11.5++ frontier at the *capture/regrasp* mechanism (out of this task's "no capture-controller change" scope),
consistent with the Phase-2 capture-support finding.

## Provenance
Deterministic; R = 5 restarts; capture `teacher_budget=1` (finds a certified grasp for the delivery arms; the 4
capture-support use the Phase-2 audit result). Energy diagnostic-only. Mac (kato14 unreachable off-lab), venv-equivalent.
`reports/2026-07-31-r11-5-plus-recovery-pilot.{jsonl,json}`.
