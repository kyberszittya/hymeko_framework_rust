---
campaign: COIN E0 initial-pose audit + playback-timing + neutral-start benchmark
title: The learned win is CONTACT-PREPARED transport (5/9), not neutral-start delivery (0/9); the video slowness is PLAYBACK_SLOW (real ~1.2 s), not the policy
date: 2026-07-21
branch: exp/coin-e0-competence-stabilization
source_commit: fe550ce
classification: CONTACT_PREPARED_START + PLAYBACK_SLOW (learned competence is transport-of-gripped-coin; neutral-start delivery unsolved)
---

# E0 initial-pose audit, playback timing, neutral-start benchmark

**Created-at:** 2026-07-21 15:38 JST. Freeze first: the 16-run campaign is complete and committed (**fe550ce**);
`campaign.json` (a156a2605209), deployed `learned_delivery_positive.pt` (8bd73d8c), `state_split.json` (045a4864),
CONTROL headline≥8/10 [2,3,4,5,5,5,5,6] / STABILIZED [0,2,2,2,5,5,5,5] — all unchanged, not reinterpreted here.

The user's suspicion was correct: the videos do not start from a neutral pose.

## §1 Initial-pose audit → **CONTACT_PREPARED_START**
Two prepositioning sources, both measured on seed 1174:
1. **`ContactFormationEnv.reset` restores a contact-prepared BANK snapshot** (`c1_heldseed_bank.pkl`) — the arm is
   already at the coin with **both pads in contact** at t=0.
2. **`CoinDeliveryTrainEnv.reset` then runs a scripted `grasp_carry` PREFIX** (≤40 steps, until `handoff_ready`) that
   grips and even *moves* the coin.

| | arm joints (a_j0..3) | both-pad contact @ t0 | coin clearance | source |
|---|---|---|---|---|
| CANONICAL_ZERO / neutral (`PlanarGraspEnv.reset`) | **[0, 0, 0, 0]** | **no** | **+0.0144** | zero pose |
| CURRENT rollout/video start (delivery `reset`) | [0.94, −1.96, 0.89, 0.73] | **yes** | +0.0699 | bank snapshot + prefix, `handoff=True` |

The "+0.0698" headline clearance I reported earlier is the **post-prefix** value; the **true neutral clearance is
+0.0144**. Classification: **CONTACT_PREPARED_START** — not neutral, not merely prepositioned; both fingertips already
contact the coin. (A snapshot stored in a generated bank is not "neutral".)

## §2 Neutral-start benchmark (arm zero pose, no bank snapshot, no prefix, pads open, no initial contact)
Same deployed checkpoint, 9 headline states, true neutral inner reset:

| start | init contact @ t0 | first-contact reached | **strict delivery** |
|---|---|---|---|
| CONTACT-PREPARED (current, authoritative) | 9/9 | 9/9 | **5/9** |
| **NEUTRAL (arm zero)** | **0/9** | 6/9 | **0/9** |

**The learned policy delivers 0/9 from a true neutral start.** It was trained and evaluated exclusively on
contact-prepared bank snapshots, so it never saw the approach; from neutral the observation is out-of-distribution and
no coin is delivered (it drifts into contact in 6/9 but never transports+settles). **Honest re-scope: the
LEARNED_DELIVERY_POSITIVE result is a TRANSPORT-of-an-already-gripped-coin positive (5/9), NOT full delivery from
neutral.** The prepositioned benchmark remains a valid *transport* benchmark; it is not a neutral-start delivery.

## §3 Playback vs physical timing → **PLAYBACK_SLOW**
- sim timestep 0.5 ms × **frame_skip 20** ⇒ **control_dt = 10 ms** (the env raises the substep count to hold the
  control interval fixed). Real delivery = **~120–172 control steps = ~1.2–1.7 s** of simulated time.
- The earlier videos were exported at **16 fps** → 121 frames play over 7.6 s ⇒ **~6.25× slow-motion.**
- Classification: **PLAYBACK_SLOW** — the physical rollout is fast (~1.2 s); the export FPS was wrong. **Not**
  POLICY_SLOW. (Real-time FPS = 1/10 ms = **100 fps**.)

## §4 Playback fixed (policy unchanged, per the rule)
- `current_start_delivery_real_time.mp4` — 121 frames @ **100 fps = 1.21 s** (real-time scientific video).
- `current_start_delivery_2x_presentation.mp4` — @ 50 fps = 2.42 s (2× slow-mo, labelled, for viewing).
- The policy was **not** changed to make the video faster.

## §5–§6 Speed fine-tuning: **not triggered**
Because the slowness is PLAYBACK, not POLICY, per §4 ("do not change the policy merely to make the video faster") no
speed fine-tuning was run. The real physical delivery (~1.2 s, transport+dwell) is already fast; there is no genuine
per-phase time waste to optimize. Fine-tuning is deferred unless a *policy-slow* case is measured.

## §7 Videos (honest)
- `current_start_delivery_real_time.mp4` (0b507e64) — contact-prepared start, delivers, real-time.
- `current_start_delivery_2x_presentation.mp4` (d3a56b02) — 2× slow-mo presentation.
- `neutral_start_attempt_real_time.mp4` (a4ea1bf4) — true neutral start (arm zero, no contact); the learned policy
  **does not deliver** (labelled honestly). No fabricated neutral-start delivery or before/after video is produced,
  because neutral-start delivery does not exist yet and no speed fine-tuning occurred.
- Retained: `coin_delivery_oracle_vs_learned.{gif,mp4}`, `coin_delivery_learned_clear_start.{gif,mp4}`.

## Report
- Completed-campaign verdict: **LEARNED_DELIVERY_POSITIVE for CONTACT-PREPARED transport** (5/9 headline, deployed
  checkpoint), now correctly scoped by this audit; **neutral-start delivery = 0/9 (unsolved)**.
- Selected checkpoint + hash: `learned_delivery_positive.pt` **8bd73d8c**.
- Current vs neutral initial pose: [0.94,−1.96,0.89,0.73] both-contact @ +0.0699 vs [0,0,0,0] no-contact @ +0.0144.
- 10-run success both starts: contact-prepared 5/9; neutral 0/9.
- Median completion time: **~120–172 control steps = 1.2–1.7 s** (no before/after — no fine-tuning).
- Real-time video FPS: **100 fps** (control_dt 10 ms).
- Video paths/hashes: above.
- Commit: this audit is additive on **fe550ce**; no campaign change. Report: this file.

## Honest bottom line
My earlier "clear-start" wording overclaimed: the learned policy solves **transport of a coin already gripped by both
pads** into the zone (5/9, up to the post-prefix +0.070), reproducibly and fast (~1.2 s real). It does **not** solve
delivery from a neutral arm pose (0/9) — that needs training that includes the approach phase (the scripted prefix /
bank snapshot currently supplies it), a separate campaign not run here. The apparent slowness was purely export FPS.
