---
title: BALLTIP_COLLISION_ON_V1 — Stage B5: option-level stochastic-Gaussian SAC (+ full ball-embodiment synthesis)
date: 2026-07-24
branch: feat/balltip-interarm-filtered-v1
baseline: executable-hymeko-option-rl-v1 @ 772a11a4
status: BALLTIP_SAC_NO_IMPROVEMENT_OVER_UPDATE0 (both seeds negative, seed-aware; local RL caps at the search ceiling)
---

# BALLTIP_COLLISION_ON_V1 — Stage B5 (2026-07-24)

Reward-driven option-level RL on the honest collision-on ball, using the Stage-5 safe pipeline verbatim (state → ball
proposal-init actor → fixed b=8 search → committed option → frozen clamp pi_0 settling → K6), reused from
`coin_carry_option_rl`. The claim is PAIRED against the ball's OWN update-0 (never the clamp). Frozen baseline untouched.

## Mandatory reward gate (CLAUDE.md) — PASSED
`certify_reward` on ball training states: `delivers=True`, R_option K6-mean **9.19** ≫ non-K6-mean **−0.79**; option-return
separation R_success 9.97 vs R_fail −0.18. The certificate-aligned option reward ranks ball delivery above non-delivery, so
RL was authorised.

## Result — the RL does NOT improve over the ball update-0
Authoritative **seed-aware** eval from the saved checkpoints (`coin_balltip_b5_eval.py`, search-seed-paired, per-(seed,
state,search-seed) bits, order-invariance ✓):

| algo | seed 0 ΔK6 | seed 1 ΔK6 | median | hier-bootstrap CI95 |
|---|---:|---:|---:|---:|
| **SAC** | −0.083 | −0.097 | **−0.09** | (−0.222, +0.035) |
| TD3 (control) | −0.139 | −0.097 | −0.118 | — |

Update-0 ball b8 (paired) = 0.236; SAC RL b8 ≈ 0.10–0.15. **Both SAC seeds negative**, TD3 negative — verdict
`BALLTIP_SAC_NO_IMPROVEMENT_OVER_UPDATE0`. The distill init was good (MSE 0.001), but training moved the actor *below* the
update-0: local policy-improvement RL degrades, rather than tightens, the search-based proposal on this task. Consistent
with the broader coin arc (local RL caps at the supervised/search ceiling; only non-local exact-rollout search exceeds it).

## Methodology corrections applied (review)
1. **Selection bias removed** — the aggregate no longer picks the best seed then reads its CI; it reports per-seed ΔK6 →
   median/IQR → hierarchical (seed→state) bootstrap. **2 seeds ⇒ PILOT cap**; `STATISTICALLY_ESTABLISHED` needs more seeds.
   (Here both seeds are negative, so the fix and the old logic agree on the verdict, but the reporting is now honest.)
2. **Baselines separated** — the one-shot update-0 (single search seed 9000+i) is labelled a DIAGNOSTIC; the claim uses only
   the search-seed-PAIRED RL−update-0 (same search rng for both at each (state, search-seed)).
3. **Per-(seed,state,search-seed) bits saved** to `b5_sac_eval.json`, not just the mean.
4. **Order-invariance check** on the SearchWrapperEnv/search_select path — the paired matrix is bit-identical on re-run
   (no gate contamination, the bug class caught earlier). ✓
5. **Naming** — "stochastic Gaussian SAC", not "distributional RL".

## Full ball-embodiment synthesis (B1→B5)
| stage | finding |
|---|---|
| **B1** | ball SOLVABLE — strong expert 16/24 (> clamp ref 5/24); settling\|handoff ≈0.93 (clamp pi_0 transfers); action language works. Case D + A. |
| **B3** | refit ball proposal beats clamp zero-shot (b=8 5/24 vs 0/24) but is a WEAK update-0 (b=0=0). |
| **B3-iter** | robust filter: **95% of ball options are FRAGILE** (5/99 robust). DAgger lifts b=0 0→3, b=8→5; deterministic proposal absorbs the search only slowly. |
| **B5** | option-level SAC does NOT beat the ball update-0 (both seeds negative). Local RL caps at the search ceiling. |

**Net:** the collision-on ball is a *viable* embodiment — its deployable controller (proposal + b=8 ≈ 6/24) is comparable
to the clamp (5/24), it needs no filtering exploit, no settling adaptation, no action-language change, and it has a
*higher* search ceiling (16/24). But the deployable gap to that ceiling is NOT closed by DAgger or SAC — it requires the
non-local search (not deployable at b=8), because the ball's winning options are intrinsically fragile/multimodal.

## Embodiment decision (per the plan)
- `BALLTIP_EMBODIMENT_SOLVABLE` ✓, `BALLTIP_SETTLING_ADAPTATION_REQUIRED` ✗, `BALLTIP_ACTION_LANGUAGE_ADAPTED` ✗ (both
  transfer), `BALLTIP_PROPOSAL_REFIT_SUFFICIENT` (narrowly).
- The ball can serve as an **object-generalization baseline embodiment** (higher ceiling, no clamp-specific cupping, no
  exploit) — but reaching its ceiling at deploy needs a stronger-than-b=8 or non-local search, not more local RL.
- `CLAMP_REMAINS_TASK_SPECIFIC_STRONG_BASELINE` for the coin specifically (comparable deploy, fewer fragile options).

## Files
- **NEW** `coin_balltip_b5_sac.py` (training, seed-aware), `coin_balltip_b5_eval.py` (authoritative eval)
- **artifacts** `b5_sac.json`, `b5_sac_eval.json`, `carry_rl_balltip_{sac,td3}_seed{0,1}_bestval.pt`
- **CORE.YAML items touched:** none. Frozen clamp pi_0 + option language unchanged.
