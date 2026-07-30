# R11.5+ — Targeted Residual Recovery: Final Coverage

**Date:** 2026-07-31
**Baseline:** `R11_5_BASE_TRANSPORT_COORDINATE_REACHES_40_OF_64` (33/51 teacher recoveries + 7 frozen-R2; 0 nudge-K6;
0 safety regressions; 100% energy/provenance). Frozen at `c58f8d39`; pushed.
**Target:** ≥ 47/64.
**Final coverage:** **40/64 — held** (a single pilot recovery, `bank_c2_-0.015_+0.015` → 18.6 mm K6, is *grasp-quality*
and awaits reproduction under the frozen protocol; not counted). The two-stage ALIGN coordinate **adds nothing**
(`two_stage_adds = 0` across all 8 deliverable scenarios); capture-support and CONTACT_LOSS are honest-negatives within
scope. This is a **bounded-evidence honest negative**, not a claim of physical infeasibility — the pilot's one recovery
localizes the remaining lever to **capture-grasp quality / regrasp**, a *mechanism* change this task's scope forbids.
**Pilot verdict:** `R11_5_PLUS_PIPELINE_PASS_RESIDUAL_RECOVERY_INSUFFICIENT` (1/12; Phase 5, below).

---

## What was tried (all three sanctioned levers, empirically, real MuJoCo)

### Complete residual taxonomy (Phase 1 — `R11_5_PLUS_RESIDUAL_TAXONOMY_COMPLETE`, `863d6554`)
All 24 uncovered classified from **trajectory replay** (the ledger's `dtz` is `dtz_end`, not the trajectory minimum):

| category | n | ground-truth signature |
|---|---|---|
| `CAPTURE_SUPPORT_FAILURE` | 10 | no certified grasp forms (all +/+ coin offsets) |
| `CONTACT_LOSS_DURING_DELIVERY` | 4 | grasp lost for >80% of the pre-release window (`lostBR` 39–44/47), coin flies to 126–161 mm; all rel-target `[-0.070,+0.032]` |
| `INSUFFICIENT_TRANSPORT_PROGRESS` | 10 | grasp held, moved toward target, stalled 11–50 mm short (`entry_speed`=0) |

Empty categories are findings too: `HANDOFF_TO_KINETIC_FAILURE`=0 (every certified case moved the coin), `ZONE_*`=0
(nothing reached the 20 mm zone; closest 22.4 mm). **Re-diagnosis:** the 4 negative-x cases are contact-loss, not
directional drift (the final-state guess) — the grasp squirts out when pushed off-axis.

### A. Capture-support (10) — honest negative
Candidate-hook audit of the 6 systematic +/+ scenarios (×2 seeds, ~7 400 candidates each): **bilateral contact never
forms** — best grasp class `SINGLE`, `bestContacts=1`, 0 `BITRANS`/`CERT`. There is no grasp candidate to rank or
diversify, so the elite/ranking lever cannot apply (this is a proposal/geometry bound, not a rank-then-reject bug). The
4 stochastic-regen scenarios (certified once in the R11.4A re-measure) recovered **0/4** with a fresh seed budget (5–9)
→ `CAPTURE_SUPPORT_HONEST_NEGATIVE`. Per the pre-registered rule, the honest negative is preserved and only the transport
pilot continues.

### B. Two-stage `TARGET_RELATIVE_ALIGNMENT_PHASE` (Phase 4 — implemented, `cceae486`)
Built as specified: a separate HyMeKo mode with its own grasp-preserving guard, a virtual align target
`x_align = x_coin + ℓ_a·d_a` (`d_a` = the *true* coin→target direction rotated by a searched offset — scenario-relative),
reusing the `forward_displacement` primitive over two segments (ALIGN → post-align snapshot → TRANSPORT via
`rollout_primitive` **unchanged**; zero regression risk to the frozen baseline). `solve_delivery_two_stage` returns the
better of {single, two-stage} and can never regress; `ALIGNMENT_FAILURE` (grasp not preserved) falls back to single.
Opt-in and additive — the baseline pipeline is bit-exact.

**Empirical result (smokes + pilot):** the machinery runs correctly but **does not recover the residuals**:
- `CONTACT_LOSS` (4): the ALIGN phase *also* loses the grasp → `ALIGNMENT_FAILURE`. The Phase-3 audit measured the
  adverse geometry: pushing toward the zone drives the coin **132–172°** away (backward). Any push (align or transport)
  with this grasp squirts the coin out.
- `INSUFFICIENT` (10): the two-stage found no align candidate beating single → fell back to single (dtz unchanged).

### C. Extended transport (Phase 3/4 audit) — coordinate-bound
For `INSUFFICIENT`, an extended-transport coordinate (horizon 150, ramp ≤ 60, forward ≤ 0.55) made every case **worse**
(Δ = −1.2, −8.2, −2.3, −0.8 mm), not better. A longer/harder push does not move the coin closer — it stalls at a
force-transfer limit. These are `HARD_GEOMETRY_WITHIN_CURRENT_COORDINATE`, not horizon-limited.

**Phase-3 honesty correction:** the first-move velocity-vs-target angle does **not** discriminate success from failure —
two frozen-R2 *successes* (`bank_c0_0`, `bank_c0_2`) show the same ~150–163° early backward jiggle yet reach K6. The
discriminator is the *sustained* transport (successes recover toward the target; the residuals do not), which the
taxonomy's `gap_closed`/`min_dtz` captures. The audit does not, by itself, prove ALIGN is the fix — the pilot tests it.

---

## Phase 5 — 3-arm recovery pilot (`reports/2026-07-31-r11-5-plus-recovery-pilot.md`)

12 scenarios (4 capture-support honest-negative short-circuited; 8 deliverable × {R2, single-stage R=5, two-stage R=5}):

| result | value |
|---|---|
| recovered | **1/12** (gate ≥ 6) |
| negative-x recovered | 0/4 |
| **two_stage_adds** (two-stage K6 where single is not) | **0** |
| safety_ok / energy_complete | true / true |

- All 4 `CONTACT_LOSS`: `ALIGNMENT_FAILURE`, single = two-stage (137–191 mm), not recovered.
- 3 of 4 `INSUFFICIENT`: `ALIGNED` but two-stage = single (no gain); `r7_a+15` reached 22.2 mm (2.2 mm short).
- 1 `INSUFFICIENT` (`bank_c2_-0.015_+0.015`) recovered to **18.6 mm K6 via single-stage** — a **grasp-quality** effect
  (70.5 mm in its taxonomy replay; a different, more deliverable pilot grasp), align verdict `ALIGNMENT_FAILURE` so the
  two-stage did not contribute.

**Gate:** `R11_5_PLUS_PIPELINE_PASS_RESIDUAL_RECOVERY_INSUFFICIENT`. The two-stage ALIGN coordinate adds no recovery; the
lone recovery localizes the remaining lever to **capture-grasp quality**, not the transport coordinate.

---

## Exact claims / non-claims

- **Claim:** within the current grip+push primitive and the sanctioned coordinate changes (capture proposal/budget,
  target-relative alignment, transport horizon/magnitude/profile/braking), the 24 residuals do not recover. Coverage is
  40/64. The two-stage ALIGN coordinate is implemented, correct, tested, and baseline-bit-exact, but adds no recovery on
  these adverse-geometry residuals.
- **Non-claim:** the residuals are NOT asserted physically infeasible. The +/+ capture forms only single-tip contact and
  the negative-x grasp transfers force ~backward — both are properties of the *current grasp/reach geometry*. A different
  capture, a regrasp, or an arm-reposition maneuver (all outside this task's "no new controller / no capture-controller
  change" scope) could plausibly solve them. That is the R11.5++ frontier.
- No BC, SAC, TD3, PPO, Hamiltonian reward, or teacher-free deployment was started. Energy stayed diagnostic-only.

## Coverage breakdown (unchanged from 40/64)
Per-class: c0 4/4 · c1 8/16 · c2 11/20 · c3 17/24. Splits: train 30/38 · dev 7/8 · test 3/5. The residual 24 =
6 baseline capture-fail + 4 uncertified-in-run + 14 certified-not-delivered, fully classified above.

## Validation (Phase 8)
- `ruff` clean on all changed modules; `radon` no D+ block (a few C-warns < 15 fail-line); `mypy --strict` clean on the
  3 new experiment modules (`r11_5_failure_taxonomy`, `r11_5_plus_residual_taxonomy`, `r11_5_plus_recovery_pilot`).
- Fast R11.5+ suite: 41 passed (taxonomy, two-stage never-regresses contract, capture candidate-hook bit-exactness,
  pilot gate logic). Frozen-primitive + delivery-teacher: 22 passed.
- **Pre-existing failure (not introduced here):** `test_r11_4a_delivery_teacher.py::test_grasp_split_mechanism_seed0_vs_seed1`
  fails on the clean baseline too (verified by stash) — a flaky stochastic grasp-split on `bank_c0_3` at `teacher_budget=1`.

## Provenance
Branch `feature/r11-4a-target-conditioned-delivery-teacher`. Host: Mac (darwin 25.5.0, 18 cores, 48 GB); per-process RSS
≪ 16 GB. kato14 unreachable (off-lab) — the Mac is venv-equivalent (torch 2.12 CPU / mujoco 3.10 / numpy 2.4.6);
idempotent. Bank sha `6cacd30b`; taxonomy sha in `reports/2026-07-31-r11-5-plus-residual-taxonomy.json`. Deterministic
seeds. Energy diagnostic-only. No CORE.YAML edits, no dependency changes.

## Recommended BC GO / NO-GO
**NO-GO for BC.** Coverage (40/64) is below the 45-formal / 47-margin gate, and the residual is a *mechanism* limit, not
a data/tuning gap — BC on 40/64 would bake in the coordinate-bound ceiling. Recommended next frontier (R11.5++, a new
gated task): capture-geometry / regrasp for +/+ and negative-x targets — i.e. the mechanism change the bounded evidence
now justifies.
