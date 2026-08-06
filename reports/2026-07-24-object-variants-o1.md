---
title: OBJECT_TO_TARGET_VARIANTS_V1 — O1 disk-size ladder (ball-tip fixed) + state-distribution finding
date: 2026-07-24
branch: feat/balltip-interarm-filtered-v1
baseline: executable-hymeko-option-rl-v1 @ 772a11a4
status: O1_EXPERT_CEILING_HOLDS_ACROSS_SIZES + PROPOSAL_STATE_DISTRIBUTION_MISMATCH_FOUND (transplant vs fresh-reconstruct)
---

# OBJECT_TO_TARGET_VARIANTS_V1 — O1 (2026-07-24)

First step of the object matrix: keep the frozen collision-on ball-tip embodiment FIXED and vary only the manipuland
size (cylinder radius), which changes mass/inertia/braking but not the contact topology. Per size, reconstruct strict==0
carry handoffs PER OBJECT (pi_0 replayed on the variant coin — not transplant, which would interpenetrate for a bigger
coin) and run the core ladder: full structured expert (192-shot, ceiling) + frozen ball proposal zero-shot + b=8.

## Result (`reports/2026-07-24-object-variants-o1/o1_disk_size.json`)
| size (radius) | n | ball proposal zero-shot b8 | structured expert ceiling |
|---|---:|---:|---:|
| small 0.014 | 16 | 0/16 (0.00) | 9/16 (0.56) |
| canonical 0.020 | 16 | 0/16 (0.00) | 11/16 (0.69) |
| large 0.028 | 16 | 1/16 (0.06) | 8/16 (0.50) |

## Two findings — one clean, one a methodology caveat
1. **`O1_EXPERT_CEILING_HOLDS_ACROSS_SIZES`.** The structured expert solves 0.50–0.69 across all three sizes (slight peak
   at the canonical 0.020, as expected — the pipeline was tuned there). Object *size* does not break solvability; the
   contact topology is preserved, so the ladder's ceiling is roughly size-invariant. Size is the *least* informative axis
   (as anticipated — the informative shapes are O2 square/rect, O3 triangle, O4 ring).
2. **`PROPOSAL_STATE_DISTRIBUTION_MISMATCH_FOUND` (important).** The frozen ball proposal scores **0/16 even at the
   canonical size** on fresh per-object reconstruction — but a direct check gives **0.25 on TRANSPLANT states** (the B3/B5
   distribution). So the 0/16 is a **train/eval state-distribution mismatch**, not size-sensitivity: the proposal was fit
   and evaluated (B3, B5, the freeze's 0.236) on *transplant* handoffs (canonical clamp reconstruct → transplant qpos),
   whereas pi_0 replayed directly on the ball reaches *harder* carry handoffs it does not cover. The expert (fresh search)
   handles both distributions; the frozen proposal does not.

## Caveat this puts on BALLTIP_COIN_BASELINE_V1
The freeze's deployed number (paired b8 0.236) was measured on the **transplant** distribution — a legitimate distribution
for the *robot comparison* (B1: clamp vs ball at matched states) but NOT the ball's own true deploy distribution (pi_0
reconstructed on the ball). On the fresh-reconstruct distribution the frozen proposal is ≈0. **The deployed ball baseline
is only operational on the transplant distribution; its true fresh-reconstruct deploy performance is an open item.** The
expert ceiling (0.5–0.69) shows the states ARE solvable — the gap is the proposal's, not the object's.

## Implication for the geometry matrix (before O2)
The proposal-transfer column of the ladder must be measured on a CONSISTENT distribution. Two clean options for O2–O4:
- (a) evaluate proposal transfer on **transplant** states where valid (matched to how the proposal was fit), OR
- (b) **re-fit** the ball proposal on **fresh-reconstruct** states to get an honest true-deploy baseline, then transfer.
Recommend (b) for a true-deploy story; (a) is faster for a like-for-like transfer signal. Either way the **expert ceiling**
and the **explicit geometry controller** are distribution-agnostic and remain the clean per-shape signals.

## Files
- **NEW** `coin_object_variants.py` (O1 disk-size ladder) — committed with the object enabler
- object enabler: `disk_radius_override`/`coin_shape` threaded through PlanarGraspEnv → make_coin_env → neutral_env →
  CoinRL4Dof, and `reconstruct_handoff`/`replay_pi0` (per-object reconstruction). Canonical unchanged (19 tests green).
- **CORE.YAML items touched:** none.
