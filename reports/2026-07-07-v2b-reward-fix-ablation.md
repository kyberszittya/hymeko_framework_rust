# v2b reward fix — A/B/C/D ablation: both fixes PASS the calibration

**Date:** 2026-07-07 · No learning run. Versioned reward `galambos_task_deliver_v2b.hymeko` (v2 kept as the
failed baseline). Harness `scratchpad/v2b_ablation.py`; data `reports/figures/v2_calibration/v2b_ablation.{json,png}`.
Follows `reports/2026-07-07-v2-contact-reward-calibration.md` (the v2 failure).

## The two versioned fixes

1. **`body_progress_penalty`** (replaces the step-count `arm_body_coin_contact`): penalises body-only PROGRESS —
   the toward-zone coin speed during a non-fingertip arm↔coin contact (above ε=0.002), NOT contact-step count.
   An incidental graze that does not move the coin costs ~0; a body shove is penalised in proportion.
2. **`terminal_deliver_graded`** (replaces `terminal_deliver`): the delivery bonus is SCALED by the contact
   grade — **+1** fingertip-dominant, **+0.2** body-assisted, **−0.5** body-driven exploit. A body-shove
   delivery cannot earn the clean delivery bonus.

## Ablation (total reward, N=12; want scripted > exploit)

| variant | scripted_2f | one_fingertip | body_shove | scripted > exploit? |
|---|---|---|---|---|
| A: v2 (failed) | −180.8 | −493.2 | −117.8 | **NO** |
| B: progress penalty only | −106.9 | −335.9 | −101.6 | NO (close) |
| C: terminal gate only | −180.8 | −493.2 | −138.6 | NO |
| **D: both fixes** | **−106.9** | −335.9 | **−122.3** | **YES** |

**Both fixes are necessary and together sufficient** — B and C each fail alone.

## D decomposition (why it works)

| controller | total | terminal | zone_prog | body_pen | raw | ft_dom | exploit |
|---|---|---|---|---|---|---|---|
| scripted_2f | −106.9 | **+22.5** | +2.2 | **−0.21** | 0.75 | 0.75 | 0.0 |
| one_fingertip | −335.9 | 0.0 | +3.0 | −0.88 | 0.0 | 0.0 | 0.0 |
| body_shove | −122.3 | **−3.2** | +1.8 | **−17.1** | 0.583 | 0.083 | 0.417 |

## Acceptance — ALL MET (D)

- ✅ **scripted > exploit** (−106.9 > −122.3).
- ✅ **exploit gets no full terminal** — body_shove terminal is **−3.2** (a penalty) vs **+17.5** under A.
- ✅ **one-fingertip ≪ two-fingertip** (−335.9 ≪ −106.9; raw 0.0 vs 0.75, the two-arm affordance holds).
- ✅ **incidental graze does not dominate the penalty** — scripted body_pen is **−0.21** under D vs **−74.2** under
  A (the progress-scaled penalty spares the graze; the step-count penalty crushed it).
- ✅ **body-driven progress penalised strongly** — body_shove body_pen −17.1 + terminal −3.2 ⇒ the exploit is
  **not** reward-optimal (below the clean scripted controller).

## Provenance / status

- `galambos_task_deliver_v2b.hymeko`: `+ terminalgraded 30 + bodyprogpen 5 + zoneprog 10 + approach 4 +
  bothapproach 4 + oob 2`. Oracle `certify → delivers=True` (return 25.40).
- Env: `_fingertip_progress`/`_body_progress` accumulators (planar_grasp_env); terms in reward.py; 72 reward/env
  tests pass; ruff clean. Margin scripted−exploit = +15.4 (passes; can be widened by raising `bodyprogpen`).
- **v2b passes the calibration ⇒ v2 BC0 is now unblocked** (MLP + HSiKAN, same seeds, contact-quality split).
  RL stays frozen; no TD3/SAC/residual/off-policy.
