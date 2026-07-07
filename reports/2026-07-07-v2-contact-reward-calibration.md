# v2 contact/reward calibration A/B — reward FAILS to penalize the body-shove exploit

**Date:** 2026-07-07 · **No learning run.** Three scripted controllers × arm_body penalty sweep {0.5, 1, 2, 4}
in the v2 graded-contact env (`galambos_task_deliver_v2`, N=12, seeds 9000+). Harness
`scratchpad/v2_calibration.py`; data `reports/figures/v2_calibration/calibration.{json,png}`.

## Result

| metric | scripted_2f | one_fingertip | body_shove (exploit) |
|---|---|---|---|
| raw_delivery | **0.75** | 0.0 | 0.583 |
| fingertip_dominant_delivery | **0.75** | 0.0 | 0.083 |
| zero_body_contact_delivery | 0.083 | 0.0 | 0.0 |
| body_assisted_delivery | 0.0 | 0.0 | 0.083 |
| body_driven_exploit_delivery | 0.0 | 0.0 | **0.417** |
| arm_body_contact_rate | 0.917 | 0.583 | 0.917 |
| body_progress (coin moved by body) | **0.0006** | 0.0017 | **0.0337** |
| fingertip_contact | 0.17 | 0.054 | 0.021 |
| both_contact | 0.02 | 0.0 | 0.01 |
| dist_delta | +0.017 | −0.014 | +0.004 |
| coin_vel_to_zone | +0.0071 | −0.005 | +0.0023 |
| body-only-contact steps | 37.1 | 79.1 | 16.7 |
| base_reward (penalty=0) | −106.6 | −335.0 | **−84.5** |

**mean_reward by weight (want scripted > body_shove):**

| weight | scripted_2f | body_shove | scripted − exploit |
|---|---|---|---|
| 0.5 | −125.2 | −92.8 | **−32.4 (exploit wins)** |
| 1.0 | −143.7 | −101.1 | **−42.6** |
| 2.0 | −180.8 | −117.8 | **−63.0** |
| 4.0 | −255.0 | −151.1 | **−103.9** |

## Acceptance (2 of 4 pass)

- ✅ **scripted retains high raw & fingertip-dominant delivery** (0.75 / 0.75).
- ✅ **one-fingertip is much worse than two** (raw 0.0, dist_delta −0.014, vel −0.005 — the coin moves *away*;
  the two-arm "két robot ereje" affordance holds).
- ✅ **body-driven exploit is CLASSIFIED correctly** — body_shove grades 0.417 exploit + 0.083 assisted, with
  body_progress 0.0337 (56× the scripted's 0.0006) and fingertip_contact only 0.021.
- ❌ **exploit is NOT penalized** — the body-shove **out-scores the clean scripted controller at every weight**,
  and **raising the penalty makes it worse** (gap −32 → −104).
- ❌ **the penalty hurts the scripted demonstrator** — it fires on the scripted's 37 incidental body-only-contact
  steps (hand grazes the coin during approach with no fingertip gripping) *more* than on the exploit's 16.7.

**⇒ the reward calibration does NOT pass. Do not proceed to BC0/DAgger-for-RL or any reward-based learning until
the reward is fixed and re-calibrated.** (The metrics/affordance/classification pass, so the env + demonstrator +
grading are sound; the defect is purely in the reward shaping.)

## Root cause (two compounding defects)

1. **Base reward already prefers the exploit** (−84.5 vs −106.6, before any penalty): the fast body-shove
   earns `terminal_deliver` for delivering, while the scripted's dense `both_approach` (−max(l,r) distance)
   accrues more negative over its careful two-arm approach. The delivery bonus is awarded regardless of *how*
   the coin was delivered.
2. **The penalty fires on body-only CONTACT presence, not body-only PROGRESS.** `arm_body_coin_contact` is
   −1 per step of body-only contact; the scripted incurs 37 such steps (incidental hand-graze during approach)
   vs the exploit's 16.7, so more penalty lands on the clean controller — exactly backwards.

## Proposed fix (reward redesign, gated on user decision — reward changes edit the .hymeko)

- **Scale the penalty by body-only PROGRESS, not contact-step count.** body_progress separates the exploit from
  the clean controller by **56×** (0.0337 vs 0.0006) — a clean discriminator that the step-count proxy destroys.
  Change `arm_body_coin_contact` to return the toward-zone displacement accrued under body-only contact
  (penalise the body *moving* the coin, not merely touching it), matching the metric's grade.
- **And/or gate `terminal_deliver` by the fingertip-dominant grade** — a body-driven delivery should not earn the
  full delivery bonus. This fixes the base-reward preference at its source.
- Re-run this calibration after the fix; require scripted > exploit at the chosen weight before BC0.

## Frozen v1/v2 summary (unchanged)

scripted ~0.84 ceiling / BC ~0.5 / off-policy RL below BC (frozen). v2 graded-contact env validated with the
scripted controller (raw 0.75, fingertip_dominant 0.75). RL stays frozen.
