# R11.6B — Robustness-Seeking Generalization RL: conclusion

**Date:** 2026-08-05 (Mac)
**Verdict:** **`R11_6B_ROBUST_OBJECTIVE_STABLE_BUT_NO_GENERALIZATION_GAIN`** (3/3 seeds). The robust reward preserves the
train policy but does **not** lift unseen (dev) delivery over BC. **Not a failure** — a mechanism finding: a robustness
objective at the *RL* level cannot compensate for *narrow demonstrations*. The robustness must move to the **teacher**.

## Result (44 train / 7 dev / 5 test; A0/A1/A2 same panel; K=4, σ=1%, 3 seeds)
| arm | reward | mean train K6 | mean dev K6 | mean dev robust |
|---|---|---|---|---|
| **A0** | BC warm-start, no RL | (0.857 sub) | **0.286** | 0.286 |
| **A1** | anchored TD3, nominal (= v2.1) | 0.841 | 0.286 | 0.286 |
| **A2** | anchored TD3, **robust** | 0.841 | **0.286** | 0.286 |

A0 = A1 = A2 = dev **0.286**, all three identical across all seeds. `dev_gain_scenarios 0`, `seeds_with_dev_gain 0/3`,
train preserved (no collapse in the *selected* policy), 0 safety regression. The only variable changed between A1 and A2
was the reward semantics — so the null result is cleanly attributable to the robustness objective, not confounds.

## Mechanism — the decisive A2 training curve
The combined-robust selection score is `dev_nominal + dev_robust` while the train subset is preserved, else `-1`
(disqualified). The warm-start floor is `0.286 + 0.286 = 0.572`. Every A2 seed:
```
it 120: score 0.572 (warm-start, train preserved)
it 240 … 1000: score -1.0  (train DISQUALIFIED); train_recent -> 0.0 by it 600
```
The robust reward drove the greedy actor's train nominal to **zero** — *faster and harder* than R11.6A's nominal reward
(which oscillated 0.03–0.38). Combined selection correctly kept `best_val = update0` (the warm-start) because every trained
checkpoint after it 120 had collapsed train.

**Why:** the robust reward *penalizes* narrow θ (CVaR tail + the survival term), so it actively pushes the actor **off** the
narrow warm-start — but **TD3, from that narrow warm-start, in this action space and budget, did not *find* a wide-basin
policy that still delivers the training scenarios.** So the actor drifts to a non-delivering policy (train 0), and selection
falls back to the warm-start. **Scope of the claim (important):** this does NOT prove no reachable wide-basin policy exists —
only that RL did not reach one from a narrow base here. The clean falsification is teacher-side: search for wide-basin θ at
demonstration generation (R11.5R). If the *teacher* finds wide-basin θ for the same scenarios, the narrowness was an
artifact of nominal-K6 optimization, not an intrinsic limit.

## Interpretation (chain across R11.4B → R11.6A → R11.6B)
- **R11.4B:** narrow teacher θ ⇒ BC can memorize train but not generalize (0.29 held-out).
- **R11.6A:** narrow teacher θ ⇒ RL drifts off the warm-start; the anchor prevents forgetting but RL cannot improve the
  seen distribution (best = warm-start).
- **R11.6B:** a robustness *reward* cannot fix generalization either — it destabilizes the narrow warm-start without a
  reachable wide-basin replacement. **The bottleneck is the demonstrations, not the reward, the algorithm, or the anchor.**

## Next — R11.5R Robust Delivery Teacher Re-certification (the clean falsification)
Move the robustness from the RL reward to **demonstration generation** (R11.5R; R11.6C stays reserved for exact-zero
composition). The teacher optimized nominal K6, so it often *picked* narrow θ-basins; a teacher that searches directly for
local K6-survival may find **wider, more learnable, more transferable** delivery programs for the *same* scenarios. This is
the next clean falsification of "does a reachable wide-basin θ exist per scenario?".

Concretely: a two-arm teacher A/B (T0 nominal vs T1 robust) on the same scenarios/proposal-seeds, T1 scoring lexicographic
(safety → nominal K6 → survival 0.5% → 1% → CVaR-dtz → 2% stress) with a progressive-perturbation funnel; re-certify the
bank (`WIDE_RECERTIFIED` / `NARROW_ONLY` / `NO_NOMINAL_K6`, keeping the nominal fallback); then re-run the R11.4B BC harness
**at the same size** (B0 nominal vs B1 robust bank) to isolate the effect of demonstration *quality* on generalization —
only then RL from the wider warm-start. `WideBasinDeliveryCertificate` + `robust_rollout` are reused as the teacher's CEM
score. Not per-step RL, not more raw demos — *better-shaped* demos.

## Files touched (all non-core)
- `hymeko_rl/coin_delivery/theta_option/robust_delivery.py` (`RobustDeliveryReward`, `robust_rollout`,
  `WideBasinDeliveryCertificate`, `RobustCoinDeliveryEnv`), `hymeko_rl/experiments/r11_6b_robust_rl.py` (A0/A1/A2,
  wide-basin anchor, robust selection, gate, test panel), `hymeko_rl/tests/test_r11_6b_robust.py` (9 tests). Committed
  `9383233a`; result `reports/2026-08-05-r11-6b-robust-rl/`.
- **Read-only:** `option_rl/*`, the delivery engine, `delivery_bc/*` (incl. the basin audit), the demonstration bank.

## CORE.YAML items touched
**None.** New Python under `hymeko_rl/`; no dependency added.

## Test / gate / provenance
9 tests pass; ruff / mypy --strict / radon (no C+) clean. Wall ~6 h (≈36 min reconstruct 56 handoffs + A1 3×~20 min + A2
3×~100 min at 5× rollout cost); per-process RSS ≪ 16 GB. Mac, 48 GB, Apple Silicon, `OMP_NUM_THREADS=2`, deterministic
(fixed seeds; eval perturbations fixed-seed, training perturbations stochastic). Teacher θ from
`reports/2026-08-03-r11-4b-bc/dataset/`; basin labels from `reports/2026-08-03-r11-4b-bc/basin/`. §2 plan
`docs/plans/2026-08-05-r11-6b-robust-generalization/` (4-format). Energy diagnostic-only (frozen R11 contract → R11.8).

## Guards
Safety held (0 unsafe). Only the reward semantics changed vs v2.1 (A0/A1/A2 same panel isolate the effect). The negative is
honest and mechanism-grounded (robust *reward* can't fix narrow *demonstrations*) — it redirects to the robustness-aware
teacher, NOT per-step RL, NOT more raw demos. The test panel was correctly NOT run (no PASS), keeping it untouched for a
future robust-teacher attempt. `WideBasinDeliveryCertificate` is reusable as the teacher's CEM score.
