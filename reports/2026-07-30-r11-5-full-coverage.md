# R11.5 — Target-Conditioned Delivery Teacher, Full-51 Coverage

**Date:** 2026-07-30
**Verdict:** `R11_5_TEACHER_COVERAGE_INSUFFICIENT` — **40/64** covered (pre-registered threshold ≥ 45/64).
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

## Failure attribution (51 = 33 recovered + 4 capture-quality + 14 certified-not-delivered)

### Capture-quality failures (4) — no certified grasp in 5 seeds
All small c1/c2 perturbations (train): `bank_c1_-0.01_+0.00`, `bank_c1_+0.01_+0.02`, `bank_c2_-0.015_+0.000`,
`bank_c2_+0.015_+0.015`. These never reached the delivery teacher — the grasp itself could not be re-established. This
is an **upstream capture-reliability** gap, and it directly explains part of the c1/c2 drag.

### Certified-but-not-delivered (14) — grasp OK, delivery CEM (R=11) missed K6
Sorted by final coin-to-zone distance:
| band | count | scenarios |
|---|---|---|
| **zone-near** (≤ 30 mm) | **1** | `bank_c3_r7_a+15` (22.4 mm) — only plausibly budget-fixable case |
| mid (31–70 mm) | 9 | c3 angled (`r5_a-45`, `r9_a+15`, `r7_a-45`, `r9_a+30`, `r6_a-45`, `r9_a-45`) + c2 (`-0.025_+0.025`, `-0.015_+0.025`, `-0.015_+0.015`) |
| **far tail (126–161 mm)** | **4** | `bank_c1_-0.01_+0.03` (127), `bank_c1_-0.03_+0.03` (159), `bank_c2_-0.025_+0.015` (161), `bank_c1_-0.03_+0.02` (161) — all c1/c2 **negative-x** |

The far tail is **not** near-miss and **not** variance: the teacher leaves the coin 12–16 cm from the zone. That is a
**structural limit of the current transport coordinate** for the far/negative-x handoff geometry — a redesign lever,
not a budget lever.

### Restart-index histogram (winning restart of first K6)
`r0: 24 · r1: 5 · r2: 1 · r3: 1 · r4: 1 · r5: 1` → R=5-vs-R=11 marginal gain **+1**.

---

## What the next lever is (and is not)

- **NOT more restarts** — proven ~useless at population scale (+1 over R=5).
- **Capture reliability on small c1/c2 perturbations** — 4 states never formed a certified grasp; recovering them is
  worth up to +4 coverage and is upstream of delivery.
- **Delivery-teacher support for the far/negative-x handoff geometry** — the 126–161 mm tail is a structural transport
  limit; closing it needs a coordinate/objective change, which is a *new* R11.5+ design decision (out of this frozen
  protocol's scope), not a re-run.
- **1 genuine zone-near** (`c3_r7_a+15`, 22.4 mm) is the only case a modest budget/tolerance nudge might flip.

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
