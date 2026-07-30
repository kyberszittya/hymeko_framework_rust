# R11.5++ — Deliverability-Ranked Capture-Grasp Selection (A/B on the 10 INSUFFICIENT residuals)

**Date:** 2026-07-31 (kato14, official box)
**Verdict:** **`CAPTURE_DELIVERABILITY_RANKING_CONTRACT_GAP`** — grasp selection is **load-bearing** (a deliverable grasp
exists that current selection discards), but the pilot did not clear the full gate (material dtz-improvement 2/10 < 6;
K6 5/10 met the ≥5 sub-gate). `SUPPORT_INSUFFICIENT` is **not** triggered (`any_deliverable = true`).

## Setup
Transport **frozen** (`solve_delivery` / `full_transport_spec` / objective / K6 + safety contracts unchanged); only the
capture-candidate **selection** varies. Per INSUFFICIENT scenario: generate up to N=10 certified grasps from the capture
population (controller frozen), deliver **each** with the frozen teacher at R=5 restarts (the baseline transport budget),
then A/B: Arm C (current = first certified grasp) vs Arm D (deliverability-ranked, `select_deliverable_grasp`).
**Teacher-only oracle** — `selection_kind = DELIVERY_ORACLE_RANKED_CAPTURE_TEACHER`, `teacher_only = true`. Each candidate
logs the pre-delivery handoff descriptor (q, q̇, prev_τ, coin pose/vel, dwell, contact relvels, seating) for a future
`handoff → deliverability` surrogate.

## Results (10 scenarios)
| scenario | grasps | deliverable | current dtz (K6) | ranked dtz (K6) | ranking recovers |
|---|---|---|---|---|---|
| c2_-0.015_+0.015 | 2 | 1 | 17.25 ✓ | 17.25 ✓ | — (current already K6) |
| c2_-0.015_+0.025 | 4 | 0 | 39.65 ✗ | 39.65 ✗ | — (no deliverable grasp) |
| **c2_-0.025_+0.025** | 5 | 2 | **163.86 ✗** | **10.28 ✓** | **✓ decisive** |
| c3_r5_a-45 | 2 | 0 | 20.03 ✗ | 20.03 ✗ | — (no deliverable) |
| c3_r6_a-45 | 2 | 0 | 32.58 ✗ | 32.58 ✗ | — (no deliverable) |
| c3_r7_a+15 | 3 | 3 | 19.94 ✓ | 14.26 ✓ | — (both K6; ranked 5.7 mm closer) |
| c3_r7_a-45 | 2 | 0 | 52.01 ✗ | 51.22 ✗ | — (no deliverable) |
| c3_r9_a+15 | 3 | 2 | 19.14 ✓ | 17.0 ✓ | — (both K6) |
| c3_r9_a+30 | 3 | 2 | 16.62 ✓ | 16.62 ✓ | — (current already K6) |
| c3_r9_a-45 | 2 | 0 | 71.24 ✗ | 71.24 ✗ | — (no deliverable) |

**Gate:** ranked-K6 **5/10** · current-K6 4/10 · ranking net gain **+1** · material dtz-improve **2/10** · all safe · all
certified · any-deliverable true.

## Findings
1. **Grasp selection is load-bearing — decisively.** `c2_-0.025_+0.025`: the first certified grasp (Arm C) delivered
   163.86 mm (total failure); two grasps in the population deliver K6 (~10–13 mm); the ranking picks the 10.28 mm one.
   The transport was never the problem here — the grasp was.
2. **The smoke-driven fix (stability gate, not raw `-dwell`) is what makes it work.** In that case the failing grasps
   have the *higher* dwell (seed 1 = 7, seed 5 = 8); a raw-`-dwell` tier-3 would have picked one of them and failed. As a
   binary stability gate, all adequately-stable grasps tie on tier 3 and **delivered dtz decides** → the 10.28 mm grasp.
3. **The net gain over current is modest (+1 K6 at R=5), for two informative reasons:** (a) the current arm at R=5 already
   recovers 4/10 (restart budget + grasp variance flip several near-boundary scenarios — consistent with the cross-host
   story); (b) **5/10 scenarios found only 2–4 certified grasps and none delivered** — the deliverable-grasp *population is
   under-sampled*, not proven barren. The `a-45` angled scenarios cluster in the barren set.
4. **The bottleneck has shifted** from the transport coordinate to the **size/diversity of the deliverable-grasp
   population.** The ranking recovers whenever a deliverable grasp is present; growing the population (more capture seeds,
   or a delivery-aware capture that generates more diverse grasps) is the lever that would convert the 5 barren scenarios.

## Coverage implication (provisional)
5/10 INSUFFICIENT reached K6 in this pilot (via current or ranked, R=5) → suggests ~43/64 is reachable on kato14 (from
the 38/64 baseline) with the existing single-stage transport and better grasps. This is **not** counted coverage — it is
a pilot at R=5 / N=10, not the frozen full-51 protocol; a deliverability-ranked full re-run is needed to certify it.

## Claims / non-claims
- **Claim:** deliverability-ranked selection recovers at least one INSUFFICIENT scenario (164 → 10 mm K6) that current
  selection fails, with the transport bit-identical — grasp selection is a real, load-bearing lever.
- **Non-claim:** this is a full PASS (it is not — 2/10 material) or a coverage number (teacher-only oracle, single R=5
  realization). The oracle uses full-rollout delivered-dtz to *select*; deployment needs a handoff-descriptor surrogate
  (the per-candidate log is the training set for it).
- No two-stage ALIGN, extra delivery restarts beyond R=5, CONTACT_LOSS lever, new controller, BC, or RL. Transport frozen.

## Next lever (evidence-backed)
Grow the deliverable-grasp population for the barren 5/10: more capture seeds and/or a delivery-aware capture that
diversifies grasps, then re-A/B. The ranking is proven; the supply of deliverable grasps is the new limiter. The
`a-45`-cluster barrenness may need a genuinely harder capture-geometry change (parked with CONTACT_LOSS / +/+ support).

## Provenance
kato14 (Linux x86-64, 10-way fan-out, one process per scenario, N=10 capture seeds, R=5 delivery restarts; ~16 min).
Deterministic; teacher-only oracle; energy diagnostic-only. Artifacts `reports/2026-07-31-r11-5pp-deliverability-kato14/`
(merged.json + 10 shards with per-candidate handoff logs). Code `1cecc17a`. §2 plan on disk
`docs/plans/2026-07-31-r11-5pp-deliverability-grasp/`. No CORE.YAML, no deps.
