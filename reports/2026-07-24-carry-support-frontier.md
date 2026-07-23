---
title: Carry support-frontier — the pi_0+offset candidate class plateaus at ~40% for contact_retention; a carry-specific action structure is needed
date: 2026-07-24
branch: recovery/coin-hymeko-bundle-and-results
terminal: CANDIDATE_CLASS_PLATEAU_PI0_OFFSET_LIKELY_WRONG_FOR_CARRY_CONTACT_RETENTION — no support axis (horizon/temporal/amplitude/budget) materially raises coverage above ~0.40, CEM optimizer stable (Jaccard 0.82/0.66), budget-matched random ties CEM
tags: [coin, carry, frontier, cem, candidate-class, plateau, budget-matched-random, multi-seed-stability]
---

# CARRY_SUPPORT_FRONTIER_V1 — which axis limits carry coverage? None of the tested ones; the candidate class plateaus.

CARRY_HANDOFF_COVERAGE found ~1/3 coverage at one support setting. This factorized frontier separates whether the limit is
magnitude, temporal length, horizon, or search budget — rather than a single 0.4/50/160 run that would only show "something
helped." All five review corrections are applied.

## Method (corrected)
Same **manifest-verified** 30 held-out strict-0 carry states (sha16 `d7f052ff`; each verified strict==0 ∧ gate_mult==1.0 ∧
family∈carry), same per-state CEM seeds, **budget-matched random** control (shots×iters rollouts). Factorized configs:
C0 (.20/30/120 ref), C1 (horizon 160), C2 (temporal len 50), C3 (amplitude .40), C4 (combined), CB (2× CEM iters).
Corrections: **NET** material gate (n_new−n_lost ≥ 5); **horizon-fair** safety = `any_exit` (fraction with ≥1 containment
exit), not raw counts; **multi-seed optimizer stability** (C0 & C4, 3 CEM seeds) to separate a real plateau from
CEM-instability; contact_retention-primary caveat. `carry_cem`/`carry_random` verified not to mutate the shared state. No
critic training.

## Result (30 states, 25 contact_retention + 5 transport)
| config | CEM K6 | random (budget-matched) | handoff | any_exit | net vs C0 |
|---|---|---|---|---|---|
| C0 .20/30/120 | 0.333 | 0.30 | 0.333 | — | ref |
| C1 horizon 160 | 0.40 | 0.30 | 0.40 | — | +2 |
| C2 temporal 50 | 0.40 | 0.367 | 0.40 | — | +2 |
| C3 amplitude .40 | 0.40 | 0.367 | 0.467 | 0.067 | +2 (+3/−1) |
| C4 combined | 0.367 | 0.40 | 0.433 | 0.067 | +1 (+2/−1) |
| CB 2× budget | 0.40 | 0.40 | 0.40 | 0.00 | +2 (+2/−0) |

- **No axis materially helps.** Max net gain **+2** (C1/C3/CB), well below the pre-registered material bar of **+5**. Horizon
  adds 2 states (0.333→0.40); temporal length, amplitude, and doubled search budget add nothing further; the combined C4 is
  *worse* (0.367). Amplitude raises *handoff* (0.467) but not K6 — bigger perturbations get more coins to the zone but they
  do not stay/deliver.
- **Budget-matched random ties/beats CEM** at C2–C4 and CB (rand 0.367–0.40 ≈ CEM 0.367–0.40). The "search helps" signal
  from the coverage experiment (CEM 0.333 vs random 0.20) was largely a **budget artifact** — the earlier random had fewer
  rollouts; at matched budget the CEM advantage disappears.
- **The plateau is real, not optimizer noise.** Multi-seed CEM stability: **C0 Jaccard 0.818, C4 0.659** (both ≥0.6); the
  CEM solves ~the same 9–10 states each seed. **Solved core (all configs) = 8 states**; union = 15; the other ~15 never
  solve under any support or seed.
- Per family: contact_retention 0.32 → 0.40 ceiling; **transport flat 0.40 (n=5, underpowered)**; braking absent.

## Verdict — `CANDIDATE_CLASS_PLATEAU_PI0_OFFSET_LIKELY_WRONG_FOR_CARRY_CONTACT_RETENTION`
For contact_retention, the **pi_0 + support-bounded offset candidate class plateaus at ~40%** (a stable ~8–12 state core),
and *no* tested support axis or extra search budget lifts it — while a stable CEM optimizer and a budget-matched random
control rule out "under-searched" and "search-lucky" explanations. So the remaining ~60% of carry states are **not reachable
by any support-bounded perturbation of the settling-tuned pi_0**. The limit is the candidate class, not magnitude / length /
horizon / budget.

## What this decides for Phase 4
Not "distil the widened-CEM" (no widening helps). The DAgger/expert should use a **carry-SPECIFIC action structure** — an
explicit push / brake / release parametrization (or a residual over a coarse carry primitive), not a perturbation of pi_0 —
because pi_0 is tuned for settling (minimal intervention) and its neighborhood cannot carry ~60% of the coins into the zone.
The stable 8-state solved core is the "easy" carry subset a pi_0-offset expert could already label; the hard majority needs
the new action structure.

## Honest limitations
- **contact_retention-primary.** n=25; transport (n=5) is exploratory (flat 0.40), braking absent — the candidate-class
  conclusion is about contact_retention, not "carry" in general.
- These 30 states are now a **dev panel** (config was compared on them); the eventual carry-specific expert must be
  re-confirmed on a **fresh** held-out carry panel.
- CEM at 24×4 is a moderate expert; a much larger search could shift the plateau, but the doubled-budget CB (0.40) and the
  stable multi-seed core argue the ceiling is the class, not the budget.

## Files
- entry `experiments/…/rl_entry/coin_carry_support_frontier.py`; result `…/carry_support_frontier_v1.json` (manifest sha16,
  per-config any_exit/net_new, multi-seed Jaccard, solved core/union, per-family).
- lib `coin_carry_handoff.py` (reused). test `test_carry_sequence_then_pi0_and_cem` extended with a no-mutation guard.
  26 tests pass, ruff F-clean.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; held-out strict-0 carry panel sha16 `d7f052ff`, seeds
≥6300 (disjoint from train/dev). CEM 24 shots × {4, 8} iters, seeds {200,500,700}+i (deterministic); budget-matched random
shots×iters. No training, no reward/task change, no CORE.YAML items.
