---
title: BALLTIP_COLLISION_ON_V1 — Stage B1 capability decomposition + B2 adaptation-boundary diagnosis
date: 2026-07-24
branch: feat/balltip-interarm-filtered-v1
baseline: executable-hymeko-option-rl-v1 @ 772a11a4
status: BALLTIP_EMBODIMENT_SOLVABLE + BALLTIP_PROPOSAL_TRANSFER_FAILURE (Case D + Case A)
---

# BALLTIP_COLLISION_ON_V1 — Stage B1 (2026-07-24)

The **physically honest** ball-tip robot — spherical fingertip r0.020, inter-arm collision **ENABLED**, no filtering, no
penalty — decomposed across four controllers on a matched held-out panel (24 states, collision-on physics throughout), so
Stage B2 names the adaptation boundary instead of reading the clamp controller's transfer number as the ball's ceiling.
The frozen clamp robot + clamp `pi_0` are preserved unchanged.

## Decomposition (`reports/2026-07-24-balltip-b1-capability/b1_capability.{json,png}`)
| controller | candidate support | contact | handoff | **settling\|handoff** | K6 | contain-exit | ia-contact rate | option dur |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clamp_proposal_b8 (frozen, deployed) | 0.021 | 1.00 | 0.167 | 1.00 | 4/24 | 0.04 | 0.161 | 136 |
| random_shooting_64 | 0.009 | 1.00 | 0.375 | 0.89 | 8/24 | 0.08 | 0.018 | 117 |
| **expert_192** (strong search, ceiling) | 0.008 | 1.00 | **0.583** | **0.93** | **13/24** | 0.04 | 0.044 | 92 |
| geometry_probed (explicit, where practical) | n/a | 1.00 | 0.000 | n/a | 0/24 | 0.08 | 0.004 | 160 |

Solved sets (K6): clamp **reference** (clamp robot + clamp proposal) 5/24 {0,11,15,18,23}; ball under expert_192 **13/24**
{3,4,5,9,10,11,13,15,16,17,18,19,20}. Overlap with clamp = {11,15,18} — the ball solves **10 states the clamp cannot**.

## Reading the decomposition
1. **The ball embodiment is highly solvable.** A strong structured search reaches **13/24 (54%)** on the ball — more than
   twice the frozen clamp controller's 4/24, and above the clamp robot's own 5/24. Handoff rate rises monotonically with
   search strength (0.167 → 0.375 → 0.583): winning push→brake→release options EXIST for the ball; the frozen clamp
   proposal simply fails to find them.
2. **The frozen clamp `pi_0` settling skill TRANSFERS to the ball.** `settling|handoff` is 0.89–1.00 across all
   controllers — once a valid handoff is reached, the frozen settling policy dwells the ball to K6 ~90–100% of the time.
   **Settling is not the bottleneck.**
3. **The action language is adequate.** `push→brake→release` produces winning options on the ball (handoff 0.583, K6
   13/24 under the expert). Per-shot support is low in absolute terms (<1% of uniform-random shots are admissible) —
   which is exactly what a proposal is for (localise the search), not a sign the parametrization is wrong.
4. **The explicit geometry-probed push failed (0/24).** In the 4-DoF joint-torque space the joint→disk map is
   contact-mediated; a finite-difference-Jacobian push is a weak hand-controller, not a ceiling. The **search** controllers
   are the authoritative ceiling probe (as anticipated: "explicit controller where practical").

## §B2 diagnosis
- **Case D — `BALLTIP_EMBODIMENT_SOLVABLE`.** The strong search reaches a high ceiling (13/24, > clamp 5/24). ✓
- **Case A — `BALLTIP_PROPOSAL_TRANSFER_FAILURE`.** The full expert succeeds (13/24) where the frozen clamp proposal fails
  (4/24). The proposal is the transfer bottleneck. ✓
- **Case B — `BALLTIP_SETTLING_SKILL_REQUIRED`: RULED OUT.** `settling|handoff ≈ 0.93` — the clamp `pi_0` settles the ball.
  Do NOT train a new settling skill; do NOT overwrite the clamp `pi_0`.
- **Case C — `BALLTIP_ACTION_LANGUAGE_REQUIRES_ADAPTATION`: RULED OUT.** The push→brake→release language yields winning
  options for the ball; preserve it.

## Recommended next stage (Case D + Case A action — GATED on your go-ahead, it is a training stage)
**Stage B3 — ball-tip proposal.** Generate a DISJOINT ball-tip teacher bank with the strong structured expert (shots≈192,
the ceiling controller here), train a template+residual ball-tip proposal, evaluate at b=0 / b=8 / full-expert budget,
compare against the frozen clamp proposal zero-shot, and save a ball-tip update-0 checkpoint. Keep the option language and
the frozen clamp `pi_0` settling policy (B1 shows both transfer). Only after a strong update-0 proposal exists →
**Stage B5** option-level SAC, compared against its OWN ball-tip update-0 baseline (never against the frozen clamp proposal).

## Confounds / honesty
- All numbers are single-search-seed, 24-state, first-pass (provisional; §3 no-verdict-from-first-pass). The direction
  (expert ≫ frozen proposal; settling transfers) is robust across the three search strengths, but the exact ceiling (13/24)
  is one estimate — B3's teacher bank will re-measure it at scale.
- The expert ceiling (13/24) is itself a *lower bound* on what a trained proposal + search could reach — it is uniform
  random shooting, not a learned search distribution.

## Files
- **NEW** `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_balltip_b1_capability.py`
- **EDIT** `hymeko_rl/coin_delivery/coin_carry_structured.py` (+`structured_random_best_with_support`, delegating; no dup)
- **EDIT** `hymeko_rl/coin_delivery/coin_robot_variant.py` (+`interarm_contact_count`)
- **CORE.YAML items touched:** none.
