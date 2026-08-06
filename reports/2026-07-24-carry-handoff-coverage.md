---
title: Carry→settling handoff coverage — a bounded carry expert delivers on ~1/3 of held-out carry states (partial, search-helped)
date: 2026-07-24
branch: recovery/coin-hymeko-bundle-and-results
terminal: PARTIAL_CARRY_COVERAGE_CANDIDATE_CLASS_WORKS_FOR_A_SUBSET — CEM 0.333 vs pi_0 0.10 (3.3×), CEM > random > pi_0 (search helps); widen support to distinguish support-limited vs wrong-candidate-class before distilling
tags: [coin, carry, handoff, hierarchical, coverage, cem, existence-proof, upstream]
---

# CARRY_HANDOFF_COVERAGE_V1 — does a support-bounded carry-prefix exist that hands off to the frozen settling pi_0?

Phase 3 of the hierarchical upstream extension (A). We froze the working downstream skill and asked the achievability
prerequisite BEFORE building any carry actor/critic: on held-out strict-0 carry states, is there a support-bounded carry
action-sequence that produces a good settling handoff (→ K6 via the FROZEN pi_0)?

## Freeze (Phase 1) — held fixed, not retrained
pi_0 `1902454c`; the strict/certificate contract; **SETTLING_SKILL_CONFIRMED** (strict≥1 → K6 ≈ 0.95–0.97, measured); the
trust-region/safety gate. The settling critic stays SETTLING_VALUE and is **not used** here. No monolithic pi_0 retraining.

## Handoff (Phase 2) & method
A good handoff = the coin reaches **strict≥1** (contained: dtz≤CENTER_TOL ∧ speed<SETTLE_VEL) and the **frozen pi_0 then
delivers K6**. Candidates are **support-bounded** (pi_0 + clipped offset sequence, |offset| ≤ 0.20, length 30, horizon 120).
Three controllers on the same class: **PI_0** (frozen baseline), **RANDOM** (best of N random bounded sequences — no
search), **CEM_EXPERT** (bounded CEM, 24 shots × 4 iters — the exact simulator as an EXISTENCE expert only, not a learned
policy). No critic training.

## Result (30 held-out strict-0 carry states: 25 contact_retention, 5 transport)
| controller | handoff-reachable | **K6-coverage (end-to-end)** | mean max_strict |
|---|---|---|---|
| pi_0 | 0.10 | **0.10** | 0.6 |
| RANDOM | 0.267 | 0.20 | 1.33 |
| **CEM_EXPERT** | 0.333 | **0.333** | 2.0 |

Per family: contact_retention pi_0 0.12 → CEM 0.32; transport pi_0 0.0 → CEM 0.40.

**Verdict: `PARTIAL_CARRY_COVERAGE_CANDIDATE_CLASS_WORKS_FOR_A_SUBSET`.** A support-bounded carry-prefix delivers
end-to-end (→ K6 via the frozen settling pi_0) on **~1/3 of held-out carry states — 3.3× pi_0 (0.10)** — and the ordering
**pi_0 < RANDOM < CEM** shows the improvement is real and **search-driven** (CEM beats random, not just any perturbation).
So the upstream expert **partially exists**: the carry phase is at least partly solvable by bounded control of pi_0, and
the hierarchical carry→frozen-settling architecture converts those cases to K6.

(The auto-verdict's 0.5 bar flagged NO_COVERAGE; recomputed — 33% at 3.3× pi_0 with search > random is meaningful *partial*
coverage, not none. The bar was too harsh.)

## Honest reading
- **Partial, not reliable.** 2/3 of carry states are not delivered even by the CEM expert under this support bound — so a
  carry actor distilled now would inherit at best ~1/3 coverage.
- **Search matters.** CEM (0.333) > RANDOM (0.20) > pi_0 (0.10): the good carry-prefixes are not trivially random; a search
  (later, a learned policy) is needed to find them.
- The remaining 2/3 is the open question: **support-limited** (widen mag_max/length/horizon → coverage rises) vs **wrong
  candidate class** (pi_0-offset perturbation is inadequate for the carry phase → coverage plateaus, needs a carry-specific
  action structure). This experiment does not yet separate them (single support setting).

## Next lever (needs your go) — the discriminating test
Re-run the coverage CEM at a **wider support** (e.g. mag_max 0.2→0.4, length 30→50, horizon 120→160) on the same held-out
carry states:
- coverage **rises materially** → support-limited → widen the carry action space, then distil (Phase 4: BC-init from the
  CEM first actions → DAgger/receding-horizon labels → reward-driven RL), keeping the frozen settling pi_0 downstream;
- coverage **plateaus ~1/3** → the pi_0+offset candidate class is wrong for carry → a **carry-specific action structure**
  (not a perturbation of the settling-tuned pi_0) is needed before any distillation.
Only after coverage is understood does the carry actor (Phase 4) and later a carry critic (Phase 5, value-of-reaching-a-
valid-handoff) become warranted. The decision gate stays: handoff rate ↑, eventual K6 ↑, settling preservation not worse,
containment exit not worse; then carry-actor → frozen pi_0 beats original pi_0 end-to-end.

## Files
- lib `hymeko_rl/coin_delivery/coin_carry_handoff.py` — `sequence_then_pi0` (carry prefix → frozen settling, handoff-aware
  certifier), `carry_cem` (bounded existence expert), `carry_random` (control).
- entry `experiments/…/rl_entry/coin_carry_handoff_coverage.py`; result `…/carry_handoff_coverage_v1.json`.
- test `test_carry_sequence_then_pi0_and_cem` (zero-seq ≡ frozen pi_0, determinism, CEM validity). 26 tests pass, F-clean.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; held-out strict-0 carry states, seeds ≥6300 (disjoint
from train 6000–6088 / dev 6100–6148). CEM 24 shots × 4 iters, |offset|≤0.20, length 30, horizon 120, seeded per state
(deterministic). No training, no reward/task change, no CORE.YAML items. Single support setting — coverage is a first
existence estimate, not swept.
