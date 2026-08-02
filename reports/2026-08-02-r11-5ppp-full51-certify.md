# R11.5+++ Official Full-51 Certification — Deliverability-Ranked Enlarged-Population Delivery Teacher

**Date:** 2026-08-02 (Mac, off-lab — kato14 unreachable at launch)
**Verdict:** **`R11_5_TARGET_CONDITIONED_DELIVERY_TEACHER_COVERAGE_PASS`** — **56/64** overall coverage.
**Interpretation (pre-registered):** 56 ≥ 47 ⇒ **strong margin → direct GO to R11.4B conditioned BC.**

This closes the official coverage number for the R11.5 target-conditioned delivery teacher. It is not new research: the
frozen protocol re-runs the already-committed certification runner (`cdc8f721`) over **all 51 `DELIVERY_FAILURE`
scenarios** at the frozen transport budget, and reuses the pre-registered `coverage_gate`.

## Frozen protocol (exactly as specified)
- **Capture population:** N=40 certified-grasp seeds per scenario.
- **Selection:** deliverability-ranked, teacher-only oracle (`DELIVERY_ORACLE_RANKED_CAPTURE_TEACHER`), lexicographic key
  (safety > certified > bilateral-stability **gate** (dwell ≥ 4, binary) > KINETIC entry > delivered dtz > progress > seating).
- **Delivery:** single-stage target-conditioned transport (`full_transport_spec` / `_deliver_best`), CEM restarts **R=11**.
- **No ALIGN, no new controller, no new score/coordinate.** Transport bit-identical to the frozen teacher.

## Gate (authoritative)
```json
{
  "scenarios": 51, "certified": 50, "teacher_recovered": 49,
  "overall_coverage": "56/64",
  "coverage_by_class": { "c0": 1.0, "c1": 0.812, "c2": 0.8, "c3": 0.958 },
  "recovered_by_split": { "train": 37, "dev": 7, "test": 5 },
  "nudge_only_k6": 0, "safety_ok": true, "energy_complete_all": true,
  "verdict": "R11_5_TARGET_CONDITIONED_DELIVERY_TEACHER_COVERAGE_PASS"
}
```
Every gate condition met: overall **56 ≥ 45**; every C-class **≥ 0.50** (lowest c2 = 0.80); **dev = 7 ≥ 1** and
**test = 5 ≥ 1** positive; **0** nudge-only K6 (every input is a certified grasp); **0** safety regression; **100 %**
energy ledgers complete.

## Recovery over the 51 re-attempted scenarios
| C-class | recovered / attempted |
|---|---|
| c0 | 3 / 3 |
| c1 | 12 / 12 |
| c2 | 12 / 13 |
| c3 | 22 / 23 |
| **total** | **49 / 51** |

- Delivered dtz (49 recovered): min **1.26 mm**, median **8.45 mm**, max **19.98 mm** (10 under 6 mm).
- **56/64 = 7 frozen-R2 baseline positives + 49 recovered here.** This is **+13 over the baseline's 33/51** two-stage
  first-certified recoveries — the ranking lifts *all* classes, not only the 10 INSUFFICIENT the re-A/B measured.

## The residual (2 non-recoveries = the known parked frontier, nothing new failed)
| scenario | split | state | reading |
|---|---|---|---|
| `bank_c3_r9_a-45` | dev | certified, dtz 30.46 mm | **geometry-limited a-45** — grasp forms, single-stage transport can't reach K6 (ALIGN already refuted dead, `two_stage_adds=0`). |
| `bank_c2_+0.015_+0.015` | train | **uncertified** (no bilateral grasp in N=40) | **+/+ capture-support tail** — bilateral grasp never forms even at N=40 (the parked capture-geometry frontier). |

Both were already flagged parked before this run. The a-45 near-boundary case `bank_c3_r6_a-45`, fragile on the earlier
Mac re-A/B, **recovered cleanly here (16.11 mm)** — the +1 that separated the robust-46 from the optimistic-47 is now a
firm K6.

## Mechanism: the enlarged-population ranking is load-bearing
- **33 / 49 recoveries used a grasp at seed ≥ 10** (median selected seed **17**, max **39**) — deep-bank picks that the
  baseline first-certified selection (seeds 0–2) never reaches. Population **depth alone** was shown to add 0 in the re-A/B
  (A0 = A1); the **ranking on the enlarged bank** is the lever, confirmed here at scale.

## Files touched
- `reports/2026-08-02-r11-5ppp-full51-certify.md` (this report).
- `reports/2026-08-02-r11-5ppp-cert-mac/` — merged gate + 9 shards with per-candidate handoff descriptors (the
  **demonstration bank**, 2.6 MB): every candidate's `handoff descriptor → oracle delivered dtz / K6`, retained for the
  two learning targets (coin/target-conditioned delivery policy; cheap handoff→deliverability surrogate to replace the
  oracle at deployment).
- **No source changed** — the runner (`r11_5ppp_full51_certify.py`), ranking (`deliverability_ranking.py`), and gate
  (`r11_5_full_coverage.py`) were committed and tested in `cdc8f721`.

## CORE.YAML items touched
**None.** No source, no dependency change. All touched Python is non-core.

## Test results
- `hymeko_rl/tests/test_r11_5ppp_full51_certify.py` — 2/2 pass in 0.61 s (certified path selects the ranked grasp +
  emits gate fields; no-grasp path returns uncertified/not-recovered).
- `coverage_gate` covered by `tests/test_r11_5_full_coverage.py` (unchanged).
- Pre-existing flaky `test_grasp_split_mechanism_seed0_vs_seed1` (bank_c0_3 seed-split at teacher_budget=1) fails on the
  clean baseline too — **not** introduced here.

## Performance
- **Wall time:** 5 h 46 m 41 s (9-way fan-out, 51 scenarios; hard non-recovering scenarios run all R=11 restarts with no
  early-K6 exit → 67–81 min each; fast recoveries ~34 min).
- **Peak RSS:** **221 MB** max single worker, **1.6 GB** aggregate across 9 workers — far under the 16 GB per-process cap
  (§4). `MemoryMax` unnecessary at this footprint.
- No wall-time §11 breach: per-scenario anchored at 2039 s on the first completion, consistent with the ~5.5–6.5 h
  projection and under the ~7 h budget.

## Provenance
- **Code SHA:** `cdc8f721` (branch `feature/r11-4a-target-conditioned-delivery-teacher`, worktree `hymeko_coin_r9_wt`).
- **Bank:** `reports/2026-07-30-r11-4a-bank/bank.jsonl`, md5 `473244de4795254f5de99f4ca7732714`.
- **Seeds:** capture seeds 0–39 (N=40, deterministic per scenario); delivery CEM restarts R=11.
- **Host:** Mac, 48 GB RAM, Apple Silicon; 9 workers × `OMP_NUM_THREADS=2`, single-thread BLAS. Deterministic;
  teacher-only oracle; energy diagnostic-only (never in any objective).
- **Config:** `PipelineConfig(teacher_budget=1)`, `TransitConfig(substeps=6, hold_steps=160)`, `GraspObjective()` default.

## Claims / non-claims
- **Claim:** the frozen target-conditioned delivery teacher, with deliverability-ranked selection over an N=40 capture
  population and single-stage R=11 transport, certifies **56/64** teacher-bank coverage — clearing the pre-registered
  45/64 BC-gate with strong (≥47) margin, all C-classes ≥50 %, dev+test positive, 0 nudge, 0 safety regression.
- **Non-claim:** a *deployment* result. A2 selection is a teacher-only oracle (uses full-rollout delivered dtz to rank).
  Deployment needs a `handoff → deliverability` surrogate; the retained per-candidate descriptors are its training set.
- **Guards preserved:** ALIGN dead (`two_stage_adds=0`), off-antipodal refuted, squeeze abandoned, transport frozen +
  bit-identical, energy never in an objective. Two parked cases remain the capture-geometry frontier
  (`c3_r9_a-45` transport, `c2_+0.015_+0.015` capture-support) — a separate held-out stress panel, not a campaign blocker.

## Next
Per the pre-registered rule — **"Ha hivatalosan megvan a 45/64, nem húzzuk tovább: indul a BC"** — the official gate is
met with margin. **R11.4B: coin/target-conditioned BC** starts from the demonstration bank (49 recovered teacher
trajectories + the full per-candidate handoff→deliverability logs). The two parked cases move to a held-out stress panel.
