# Galambos grasp-gate: reward-shaping for grasping is falsified

**When:** 2026-06-27 23:31 JST · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu

## Question

The multi-seed verdict reported ~45% *delivery* but negative return. User asked: *has the coin ever actually been
grasped (fingertips contact), or is it knocked into the zone?* And then: gate success on grasp and see if that
makes deliveries be grasps.

## Method

- **Diagnostic** (`hymeko_rl/diag_contact.py`): train a policy, re-roll greedily with per-step contact logging
  (the env's `info["both_contact"]`), classify each episode as delivered-with-grasp / delivered-by-knock /
  death / timeout. (`evaluate` only logs the outcome and saves no policy, so this had to retrain.)
- **The grasp-gate** (the one untried reward lever): new `grasp_deliver` term (non-core `reward.py`) = +1 only
  when `in_zone AND _ever_grasped`; `galambos_task.hymeko` gated `in_zone 10→1` (knock token) `+ grasp_deliver 12`
  — so a knock earns +1, a real grasp-and-deliver +13. Three layers (term / vocab / bundle), 14 reward tests green.

## Result

| outcome (50 greedy eps, MLP) | ungated @25k | **gated @100k** |
|---|---|---|
| delivered-with-grasp | 1 | 1 |
| delivered-by-knock | 17 | 17 |
| death | 3 | **28** |
| timeout | 29 | 4 |
| **grasp-fraction of deliveries** | **0.056** | **0.056** |
| total fingertip-contact steps (50 eps) | 4 | 2 |

**Grasping did not increase.** The grasp-fraction is identical; contact dropped. Removing the knock reward and
quadrupling the budget made the policy *more aggressive* (deaths 3→28 — it shoves the coin off the table — and it
stopped freezing), but it still will not perform a two-finger pinch.

## Conclusion (measured vs inferred)

- **Measured:** the coin is essentially never grasped (1/18 deliveries, ~2–4 contact-steps per 50 episodes), and
  grasp-gating the reward does not change that.
- **Inferred:** grasping here is a **structural / exploration** problem, not a reward-weight problem. The two-finger
  pinch is a hard-to-discover behavior that reward-shaping alone cannot bootstrap — consistent with `both_contact≈0`
  across every reward variant tried since 2026-06-20 ("every patch reveals a new degenerate shortcut").
- **Caveat:** 100k steps ≈ 1/200 of proper PPO scale here, so "no grasp at 100k" is not "never." But the gate
  *removes* the easier alternative (knocking), so it should help *directionally* if reward were the lever — it did
  not. Reward-shaping for grasping is falsified as a near-term path.

## Decision

- **Keep the gate** — it is honest reward design (a knock should not earn the success bonus); the policy now fails
  honestly rather than succeeding by knocking.
- **Stop reward-grinding Galambos for grasping** — that path is dead. The real levers are structural: the
  attractor-field controller (flow that structurally cannot knock the coin out), a grasp primitive / coin-as-grasp
  -hyperedge, a demonstrated pinch (BC), or 20M-scale training. Otherwise accept Galambos-delivers-by-knock honestly
  and move to the DTC rung-2 quadruped (the larger prize).

## Files touched

- `hymeko_rl/env/reward.py` — `+_term_grasp_deliver` + registry (non-core).
- `data/robotics/meta_reward.hymeko` — declared `grasp_deliver`.
- `data/robotics/galambos_task.hymeko` — gated success (`in_zone 10→1`, `+grasp_deliver 12`).
- `hymeko_rl/diag_contact.py` — **new**, the grasp/knock classifier.
- `hymeko_rl/tests/test_reward.py` — new `grasp_deliver` test + two updated regressions. **14 pass**, ruff clean.
- Artifacts: `reports/diag_contact{,_gated}/diag_contact.json`. CORE.YAML: none.
