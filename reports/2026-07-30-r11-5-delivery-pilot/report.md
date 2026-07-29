# R11.5 pilot — target-conditioned delivery teacher recovers off-canonical delivery

**2026-07-29 · 12 certified-grasp DELIVERY_FAILURE states · branch `feature/r11-4a-target-conditioned-delivery-teacher` @ `0d5e0e4f` · run on kato14 · non-core · no new deps**

## Verdict: `R11_5_TARGET_CONDITIONED_DELIVERY_TEACHER_PILOT_PASS`

On the same certified-grasp state, the target-conditioned teacher (`solve_delivery`, full-transport spec) recovers **7/12** previously-frozen-R2-failing scenarios to strict K6 — **train 3, dev 2, test 2** — with **no safety regression** and **all energy ledgers complete**. Frozen R2 delivered **0/12** of these (all were `DELIVERY_FAILURE`). Every pre-registered gate condition is met (≥6 recovered, ≥1 dev, ≥1 test, safety, energy).

**This is the first evidence the delivery layer generalizes off-canonical:** the frozen R2 delivered 0/19 dev+test in the R11.4A re-measure; the teacher recovers dev and test here — including `bank_c3_r9_a+45`, a 90 mm target R2 left at 85 mm, delivered to K6 @ 14.7 mm.

## Per-scenario A/B (same certified grasp; frozen R2 vs teacher)

| scenario | split | R2 dtz | teacher dtz | recovered | safe |
|---|---|---|---|---|---|
| bank_c0_0 | train | 51.6 | **19.06** | ✅ | ✓ |
| bank_c1_+0.01_-0.02 | train | 40.5 | **16.51** | ✅ | ✓ |
| bank_c3_r6_a+45 | train | 53.2 | **15.27** | ✅ | ✓ |
| bank_c2_+0.015_+0.015 | train | 62.1 | 15.99 | ✖ (in-zone, not settled) | ✓ |
| bank_c0_1 | dev | 51.9 | **19.31** | ✅ | ✓ |
| bank_c3_r9_a-15 | dev | 47.1 | **13.86** | ✅ | ✓ |
| bank_c1_+0.03_-0.02 | dev | 39.3 | 9.05 | ✖ (in-zone, not settled) | ✓ |
| bank_c3_r9_a-45 | dev | 38.8 | 71.24 | ✖ (fell short) | ✓ |
| bank_c2_+0.025_+0.000 | test | 39.5 | **12.13** | ✅ | ✓ |
| bank_c3_r9_a+45 | test | 85.0 | **14.66** | ✅ | ✓ |
| bank_c1_+0.03_+0.00 | test | 44.8 | 16.06 | ✖ (in-zone, not settled) | ✓ |
| bank_c3_r9_a+30 | test | 68.5 | 33.64 | ✖ (fell short) | ✓ |

Recovered K6 median dtz **15.3 mm**; all 12 target-entry speeds 0.000 (coin settles to rest — no throw-through); coin transported 54–82 mm; 12/12 safe.

## The bug that had to be fixed first (implementation, not the primitive)

The shelved teacher ran **Phase-A** (froze the push, `horizon=36`). Traced on `bank_c0_0`, the coin starts **85.6 mm** from the zone and is still moving toward it (26.3 mm) when 36 steps end — the frozen R2 uses 80. Opening the whole push (all 6 θ) + `horizon=90` (`full_transport_spec`) delivers the same state to K6 @ 18.75 mm. The coin always reached the zone; I had starved the teacher of steps.

## The non-recoveries are informative

- **3 near-misses** (`c2_+0.015_+0.015` 16 mm, `c1_+0.03_-0.02` 9 mm, `c1_+0.03_+0.00` 16 mm): the coin **reaches the K6 zone** (dtz < 20 mm) but doesn't certify — a **settle/dwell** limit, not a reach limit. These are exactly the states the extended settle coordinate (settle damping / settle dwell / rebound suppression) targets. If converted, coverage → 10/12.
- **2 shortfalls** (`c3_r9_a-45` 71 mm, `c3_r9_a+30` 33 mm): genuinely fell short — the harder far/angled targets. Candidates for more search restarts or the direction-correction coordinate.

## Provenance

Code @ `0d5e0e4f` (`full_transport_spec` + `r11_5_delivery_teacher_pilot`). Run on **kato14** (Linux, 32 cores; venv torch 2.12.0 / mujoco 3.10.0 / numpy 2.4.6), 12-way parallel, ~10 min, 0 worker failures. Data `pilot.jsonl` (12 rows, seeds: capture 0–2, teacher restarts 0–4, sha256 `dc7e929e4d33d6e282d5de39f1081c8c7b54baaacb5bd44960e5fe943fa8f68c`). Energy diagnostic-only (winner), never in the objective (R11.4A contract). No CORE.YAML items. No new deps.

## Next (per the R11.5 plan)

Pilot passed → run the full **51** `DELIVERY_FAILURE` scenarios. Conditioned BC only if overall positive coverage ≥ 70 %, ≥ 50 % in each of C0–C3, and dev + test both have positive trajectories (not canonical-orbit-only). The 3 settle-limited near-misses motivate adding the settle-coordinate before the full run to lift coverage.
