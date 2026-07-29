# R11.5 — Target-Conditioned Delivery Teacher, Full-51 Coverage

**Date:** 2026-07-30
**Verdict (pre-registered gate):** `R11_5_TEACHER_COVERAGE_INSUFFICIENT` — **40/64** covered (threshold ≥ 45/64).
**Campaign framing (the accurate name — teacher-generalization is *proven*, this is a narrow recovery boundary):**
`R11_5_BASE_TRANSPORT_COORDINATE_REACHES_40_OF_64` + `R11_5_PLUS_RESIDUAL_RECOVERY_REQUIRED`. The base transport
coordinate covers 40/64; closing the last 5 needs targeted recovery (§ *Complete failure taxonomy* + *R11.5+ recovery
plan* below), **not** a new theory, more restarts, or BC at 40/64.
**Arc:** R11 generalized hybrid delivery — R11.4A (grasp reliability) → **R11.5 (delivery generalization)**.

---

## Summary

Ran the frozen full-51 protocol: for each of the 51 certified-grasp `DELIVERY_FAILURE` states in the R11.4A bank,
regenerate a certified bilateral grasp (grasp-aware capture, ≤ 5 capture seeds), then run the target-conditioned
delivery teacher — `solve_delivery` over `full_transport_spec()`, the **same transport coordinate/objective as the
re-gate** (no settle, no new parameter/controller/score) — with **R = 11** CEM restarts, early-exit on strict K6,
keeping every scenario in the ledger. Energy is measured on the winner only (diagnostic, never in the objective).

The teacher recovered **33/51** delivery failures. With the 7 frozen-R2 successes that is **40/64 overall — 5 short of
the 45/64 gate.** The result is a clean negative, and it **corrects two of my own prior predictions** (below).

Pre-registered non-count conditions all held: **0 nudge-only K6, 0 safety regressions, energy ledgers 100 % complete,
dev + test splits both positive.** The gate fails purely on the overall count, which by the pre-registered logic makes
this `TEACHER_COVERAGE_INSUFFICIENT` (the `GLOBAL_COVERAGE_PASS_WITH_HARD_GEOMETRY_GAP` verdict requires overall ≥ 45,
which is not reached — so it does not apply).

---

## Coverage breakdown

### Overall
| | count |
|---|---|
| DELIVERY_FAILURE states attempted | 51 |
| certified grasp formed (≤ 5 seeds) | 47 |
| **teacher-recovered to strict K6** | **33** |
| + frozen-R2 successes | 7 |
| **overall coverage** | **40 / 64** |
| gate threshold | ≥ 45 / 64 → **FAIL (−5)** |

### Per-class (total / R2 / certified / recovered / coverage)
| class | total | R2 | certified | recovered | coverage |
|---|---|---|---|---|---|
| c0 (nominal)              |  4 | 1 |  3 |  3 | **4/4 = 1.000** |
| c1 (coin-xy perturbation) | 16 | 1 | 10 |  7 | **8/16 = 0.500** |
| c2 (coin-xy perturbation) | 20 | 4 | 11 |  7 | **11/20 = 0.550** |
| c3 (far / angled)         | 24 | 1 | 23 | 16 | **17/24 = 0.708** |

**All four classes clear the ≥ 50 % per-class bar** (c1 exactly at 0.500). The shortfall is in the *volume* of the
c1/c2 perturbation classes, not in any class falling below threshold.

### Splits (recovered / attempted)
| split | recovered |
|---|---|
| train | 23 / 38 |
| dev   | 7 / 8 |
| test  | 3 / 5 |

Held-out generalization is **not** the problem: dev 88 %, test 60 %, both well positive. The drag is train-side c1/c2.

---

## Two corrected predictions

1. **I predicted the C3 far/angled tail would be the drag. That is WRONG.** c3 is the **strongest** perturbation class
   at **70.8 %** (16/23 certified grasps delivered). The coverage gap is in **c1/c2 — the coin-position perturbations**
   (especially negative-x offsets), not the hard geometry. I called the C3 tail repeatedly during the run; the data
   refutes it.

2. **The near-miss residual is NOT CEM-search variance.** The earlier single-scenario read (`p = 0.25 → R = 11`)
   suggested more restarts would convert near-misses. At population scale this is **refuted:** of 33 recoveries,
   **32 landed at restart ≤ 4** (24 on the very first try), and only **1** needed restart ≥ 5.
   **R = 11 bought +1 over R = 5** (39/64 → 40/64). The recoverable states recover almost immediately; the
   non-recoverable ones do not recover in 11 restarts. Budgeting restarts is a spent lever.

---

## Complete failure taxonomy — all 24 uncovered (not the partial 9)

The **24 uncovered = 64 − 40** = 6 baseline `CAPTURE_FAIL` (never certified in the R11.4A re-measure, never attempted) +
18 non-recovered attempts (4 that failed to re-certify a grasp in 5 seeds + 14 certified-but-not-delivered). Every one is
assigned exactly one mutually-exclusive category **from its measured trace** (`hymeko_rl/experiments/r11_5_failure_taxonomy.py`,
`taxonomy.json`). The discriminator that matters is **`gap_closed` sign** — forward coin motion with a *widening* gap is
directional, not progress. The 24 collapse to **three** categories (5 of the 8 defined categories are empty):

| category | n | signature | scenarios |
|---|---|---|---|
| **CAPTURE_SUPPORT_FAILURE** | **10** | no certified grasp reached the delivery teacher | 6 systematic **+/+** (`c1_+0.01_+0.03`, `c1_+0.03_+0.02/+0.03`, `c2_+0.015_+0.025`, `c2_+0.025_+0.015/+0.025`) + 4 stochastic-regen (`c1_+0.01_+0.02`, `c1_-0.01_+0.00`, `c2_+0.015_+0.015`, `c2_-0.015_+0.000`) |
| **DIRECTIONAL_BIAS** | **4** | driven, but net **away** from target (`gap_closed < 0`, dtz 127–161 mm) — all **negative-x** | `c1_-0.01_+0.03`, `c1_-0.03_+0.02`, `c1_-0.03_+0.03`, `c2_-0.025_+0.015` |
| **INSUFFICIENT_PROGRESS** | **10** | moved toward target, closed 32–70 % of the gap, stalled short (`gap_closed > 0`, `entry_speed = 0`) | `c3_r7_a+15` (22 mm), `c3_r5/r6/r7_a-45`, `c3_r9_a+15/+30/-45`, `c2_-0.015_+0.015/+0.025`, `c2_-0.025_+0.025` |

**Empty categories (also a finding):** `HANDOFF_TO_KINETIC_FAILURE = 0` — every certified case *did* get the coin
moving (`coin_progress ≠ 0`), so the R2/kinetic handoff is **not** the bottleneck; `ZONE_*_FAILURE = 0` — nothing entered
the 20 mm zone (all `entry_speed = 0`, so no too-fast-to-settle case); `CONTACT_LOSS = 0` as a distinct cause —
`contact_lost_steps` is uniformly high (57–86/90), which is **kinetic-normal** (post-push flight), not discriminative.

**Correction to this report's own earlier draft:** the capture-support gap is **10, not 4** — the 6 systematic +/+
baseline cases were invisible in the 51-run because they were never attempted. And the residual is **direction +
magnitude of transport**, not a physical limit: the negative-x coin→target line simply does not coincide with the force
direction the grasp can transfer, so a single PUSH drives the coin off-axis. That is a **control-program** fix, not new
physics. Restart-index histogram confirms budget is spent: `r0:24 · r1:5 · r2:1 · r3:1 · r4:1 · r5:1` → R=5→R=11 gain **+1**.

---

## R11.5+ targeted recovery plan (development starts now — the Mac run is valid for diagnosis; kato14 rerun is a later reproduction gate, not a precondition)

**A. Capture-support recovery** (the 10 `CAPTURE_SUPPORT_FAILURE`). Reuse the already-PASS grasp-aware objective; audit
whether a good candidate is generated and, if so, why it never reaches the elite. **No controller change** — only
proposal/support or elite-diversity, with **scenario-relative** capture sampling (no hand offsets). Target ≥ 2/4 new
certified grasps → same target-conditioned delivery teacher → strict K6.

**B. Negative-x transport coordinate** (the 4 `DIRECTIONAL_BIAS` + progress-lifting for the 10 `INSUFFICIENT_PROGRESS`).
A **two-phase target-relative program** reusing the *same* primitive in two short segments — **ALIGN / RECENTER →
TRANSPORT TOWARD TARGET → BRAKE / RELEASE** — plus profile/horizon/braking-onset freedom for the stall cases. Not a
single longer PUSH, not new physics, not a whole new controller.

**Bounded 12-scenario pilot** (`select_pilot`, deterministic — not a re-run of all 51): 4 capture-support (2 systematic
+/+ hardest, 2 stochastic-regen), 4 negative-x (the whole `DIRECTIONAL_BIAS` tail, structurally all-train), 4
`INSUFFICIENT_PROGRESS` (dev `c3_r9_a-45` + test `c3_r9_a+15` + nearest/farthest train — so the transport fix is
validated off-train). **Gate:** ≥ 6/12 recovery (2-scenario margin over the +5 needed), ≥ 2 capture-support, ≥ 2
negative-x, safety 12/12, 0 nudge-only K6, energy/provenance complete → **`R11_5_PLUS_RESIDUAL_RECOVERY_PILOT_PASS`**.
Then run the improved teacher on the **24** failures only (not the stable 40) + a regression control panel; **final gate ≥
47/64** (2-scenario margin for the kato14 reproduction), C0–C3 each ≥ 50 %, dev/test positive, 0 nudge-K6, 0 safety
regression, 100 % provenance/energy.

---

## Test results

| suite | result |
|---|---|
| `test_r11_5_full_coverage.py` (gate logic + 3-verdict routing) | 4 passed |
| `test_r11_5_delivery_pilot.py` | 7 passed |
| `ruff check` (runner + fan-out python paths) | clean |
| `radon cc -a -nc` | no C-or-worse blocks |

Regression test added for the 3-verdict routing (`test_coverage_verdict_three_way`): a far-C3 class shortfall must route
to `GLOBAL_COVERAGE_PASS_WITH_HARD_GEOMETRY_GAP`, and must **not** collapse to `INSUFFICIENT`, whenever overall ≥ 45.
(In this run overall < 45, so the INSUFFICIENT branch is exercised on real data.)

## Performance

- **Wall:** ~28.5 min, 16-way fan-out (start 16:48:35Z → workers done 17:17:07Z).
- **RSS:** per-process well under the 16 GB cap — single-env MuJoCo workers (~hundreds of MB each); 16 procs on a 48 GB
  host. Thread-pinned (`OMP/OPENBLAS/MKL/VECLIB/NUMEXPR = 1`) to avoid oversubscribing the 18 cores.

## Provenance

- **Host:** Mac (darwin 25.5.0), 18 cores (6P + 12E), 48 GB. **NOT kato14** — kato14 was unreachable at run time
  (SSH `kato14.katolab.nitech.ac.jp:22` timed out; off-lab network). The Mac is venv-equivalent to kato14
  (torch 2.12.0 CPU, mujoco 3.10.0, numpy 2.4.6 — the same stack that produced the pilot and both re-gates with
  identical numerics). The run is idempotent and can be re-fired on kato14 for the official numbers when reachable.
- **Git SHA:** `02f025ed` (runner: transport-only solve, R=11, capture-seeds 5, 3-verdict gate). Working tree dirty;
  files relevant to this task: `hymeko_rl/experiments/r11_5_full_coverage_fanout.sh` (new launcher),
  `reports/2026-07-30-r11-5-coverage/` (this run's shards + merged `coverage.jsonl` + worker logs), this report.
  Unrelated untracked artifacts (`experiments/.../*.pt`, other `reports/**` dirs) were **not** touched.
- **Seeds:** capture seeds 0–4; CEM restart seeds 0–10. Deterministic.
- **Dataset:** `reports/2026-07-30-r11-4a-bank/bank.jsonl`, sha256 `6cacd30b6780727d5d2034a8b9bf96020add12ef839e7b3a4f8428647687cffe`.
- **Objective:** energy diagnostic-only (never in the objective); no new parameter/controller/score vs the re-gate.
- **CORE.YAML items touched:** none (`hymeko_rl` + `forward_displacement.py` are non-core).
- **Dependencies:** none added/removed.

## Open issues / follow-ups

1. **Capture reliability on small c1/c2 perturbations** (4 states, up to +4 coverage) — upstream of delivery.
2. **Far/negative-x handoff geometry** (4-state 126–161 mm tail) — structural transport-coordinate limit; needs a new
   R11.5+ coordinate/objective design decision, not a re-run or more budget.
3. **`c3_r7_a+15`** (22.4 mm) — single zone-near case; candidate for a tolerance/budget nudge.
4. **Re-fire on kato14** for official numbers once the lab network is reachable (idempotent; same protocol).
