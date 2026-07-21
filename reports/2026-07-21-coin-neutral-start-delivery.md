---
campaign: COIN neutral-start delivery — connect APPROACH → GRASP → TRANSPORT → CERTIFY
title: NO_EFFECT — the neutral→grip APPROACH is the load-bearing gap; a curriculum bootstrapped from grasp_carry cannot supply it (delivery 0/9 from neutral)
date: 2026-07-21
branch: exp/coin-neutral-start-delivery
source_commit: f6890fa
classification: NO_EFFECT (explicit neutral-start curriculum did not improve the complete task; approach demonstrator inadequate)
---

# Neutral-start Coin Delivery — reverse-curriculum attempt

**Created-at:** 2026-07-21 16:03 JST. Goal: train the complete task from the canonical neutral pose (APPROACH →
GRASP_ACQUIRE → TRANSPORT → BRAKE_HOLD_CERTIFY), reusing the proven transport checkpoint. Preserved: audit `9aae3a6`,
transport `learned_delivery_positive.pt` (8bd73d8c), P&P `d2da720a`, Beni `4630b537`. No contact-geometry / wrist /
critic / replay / n-step / force-closure change; no framework canonicalization rerun.

## §2 Hidden task-completion removed from reset (verified)
`NeutralCoinDeliveryEnv` (`hymeko_rl/experiments/coin_neutral_start.py`) resets from the **canonical neutral pose**
(`PlanarGraspEnv.reset` — arm joints [0,0,0,0], coin placed per seed, pads open, **no contact**) with **no bank-snapshot
restore** and a **controllable** scripted-prefix count (reverse-curriculum knob, 0 = true neutral). It **fails loud** if
a `prefix_steps=0` run begins with contact. The former hidden `grasp_carry` prefix is now explicit curriculum data.
Verified: prefix=0 → arm [0,0,0,0], no contact, true neutral clearance +0.0144 (seed 1174); rising prefix rotates the
arm toward the coin.

## §5 The load-bearing gap: there is no neutral→grip APPROACH demonstrator
Measured from true neutral (9 headline states): the scripted `grasp_carry` reaches **first-contact 2/9** (median 48
steps), **bilateral 0/9**, **handoff 0/9**. `grasp_carry` only *grips from a near-coin state* — it is **not** the
approach. The approach that built the contact-prepared bank was a **separate pipeline** (`sample_precontact_snapshots`
/ PEDC planner in `pedc_selection.py`), whose policy is not a loadable E0 checkpoint here (checkpoint `94601ea4` is not
present on disk). So neither the scripted prefix nor an available checkpoint supplies the neutral approach.

## §6–§8 Reverse-curriculum training (transport init + bank rehearsal + neutral curriculum + BC demos)
- Actor initialised from the proven **TRANSPORT** checkpoint (8bd73d8c).
- Curriculum env samples **35% contact-prepared bank rehearsal** (transport-suffix distribution) + neutral starts with
  a random scripted-approach prefix in [0, 110].
- BC anchor on explicit demos: scripted `grasp_carry` APPROACH (partial) + learned TRANSPORT from bank starts.
- α-floor 0.0367 (the stabilization finding), persistent bc_coef=1.0. 40k steps, seed 0.

## §9–§11 Result — NO_EFFECT (delivery 0/9 from neutral, every eval)

| policy on true-neutral headline (9) | first-contact | bilateral grasp | zone | **strict delivery** |
|---|---|---|---|---|
| zero-action (control) | — | 0/9 | 0/9 | **0/9** |
| transport-only (init) | 3/9 | 0/9 | 1/9 | **0/9** |
| scripted `grasp_carry` + learned transport | 2/9 | 0/9 | 0/9 | **0/9** |
| initial composed (transport init + BC) | 3/9 | 0/9 | 1/9 | **0/9** |
| **fully trained neutral-start (40k)** | **3→4/9** (noise) | **0/9** | 0–1/9 | **0/9** |

Delivery eval curve: `[0,0,0,0,0,0,0,0,0,0]`. First-contact curve: `[3,3,3,4,4,3,3,3,3,3]` — oscillates 3–4, no
sustained gain. **Bilateral grasp never occurred (0/9) at any eval.** True neutral clearances (not post-prefix):
[0.079, 0.011, 0.015, 0.014, 0.076, 0.089, 0.105, 0.039, 0.019].

## §15 Classification: **NO_EFFECT**
The explicit neutral-start curriculum did not improve the complete task: delivery stayed 0/9, bilateral grasp 0/9,
first-contact 3→4/9 (within noise). Not APPROACH_ONLY_POSITIVE (first-contact did not meaningfully or stably rise);
not GRASP_INTEGRATION_POSITIVE (grasp never formed); not BLOCKED (execution was clean).

## Diagnosis (honest, and what would unblock it — NOT a transport limitation)
The transport competence is intact (5/9 from contact-prepared); the gap is the **neutral→bilateral-grip APPROACH**,
which:
1. `grasp_carry` cannot supply from neutral (it grips from near-contact only), so BC on it teaches "drift toward coin,"
   not "approach + grip";
2. the **delivery reward gives no gradient for the approach phase** — it rewards coin→zone, but from neutral the arm
   must first *reach and grip* the coin before any coin motion earns reward (sparse-approach problem);
3. the original approach was an MLP/planner in the bank-generation pipeline, whose policy is not a loadable checkpoint.

Per the task constraints (no reward/certificate change), a shaped approach reward was not added. **The correct next
step is to recover or train a dedicated neutral-approach policy** (arm→coin reach + grip, e.g. an approach reward on the
already-present ACTOR_FIELDS approach-distance, or the PEDC/`sample_precontact_snapshots` approach), then compose
APPROACH → the proven TRANSPORT and reverse-curriculum-tune. That is a separate approach-training effort, not a tweak.

## §14 Demo gate: not met
No neutral-start delivery reached ≥8/10, so no `coin_delivery_neutral_start_*` video is produced (the gate is
explicit). The honest `neutral_start_attempt_real_time.mp4` from the pose-audit report already shows a neutral start
with no delivery.

## §17 Provenance
- Code: `hymeko_rl/experiments/coin_neutral_start.py` (NeutralCoinDeliveryEnv + curriculum + demos + eval + trainer);
  reuses `train.sac`, `coin_delivery_e0_campaign`, `delivery_certificate`, `p_grasp_carry`. ruff clean.
- Data: `experiments/2026_07_21_coin_neutral/run.json` (b00fcf11), `neutral_best.pt` (6b0cf15e), `neutral_final.pt`.
- Commits: env+baseline **f6890fa**; this report additive. Branch `exp/coin-neutral-start-delivery`.
- True neutral problem hashes: seeds 1011/1045/1164/1174/1202/1278/1358/1447/1568, arm [0,0,0,0], pads open, no contact,
  clearances above. Host Apple M5 Pro; threads pinned; ~400 steps/s; RSS < 1 GB. RL not bit-reproducible (BLAS).

## Honest bottom line
This is one bounded curriculum attempt whose setup has an **identified inadequacy** (no real neutral-approach
demonstrator, sparse approach reward) — so it is NO_EFFECT *for this setup*, not proof that neutral-start delivery is
unlearnable. The neutral-start env, the explicit-prefix curriculum knob, and the causal harness are in place and
reusable; the missing piece is a working APPROACH policy, which is the next task.
