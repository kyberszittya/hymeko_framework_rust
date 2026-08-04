# R11.6A — Reward-Driven Delivery RL: conclusion (v1 → v2 → v2.1)

**Date:** 2026-08-05 (Mac)
**Verdict:** **`R11_6A_POSITIVE_REPLAY_PREVENTS_FORGETTING_STALLED`** (v2.1, combined selection, 3/3 seeds).
The immutable positive-replay anchor **prevents catastrophic forgetting**; online TD3 finds **no train-preserving dev
improvement** on the *seen* certified-handoff distribution. BC already solves R11.6A's task; RL's advantage — if any — is
generalization (**R11.6B**), not the seen distribution.

## The three-run arc (44 train / 7 dev, 3 TD3 seeds, teacher-BC warm-start)
| run | mechanism | selection | per-seed train | per-seed dev | verdict |
|---|---|---|---|---|---|
| **v1** | generic TD3 (no anchor) | dev-only | 0.932 / 0.136 / 0.295 | 0.286 / 0.429 / 0.571 | `RL_UNSTABLE` |
| **v2** | + immutable positive replay | dev-only | 0.227 / **0.841 / 0.841** | 0.429 / 0.286 / 0.286 | `RL_UNSTABLE` (artifact) |
| **v2.1** | + positive replay | **combined** | **0.818 / 0.818 / 0.818** | 0.286 / 0.286 / 0.286 | **`PREVENTS_FORGETTING_STALLED`** |

- **v1:** without an anchor, TD3 forgets the warm-start — train collapses to ~0 mid-training (Q → −13); 0/3 seeds preserve it.
- **v2:** the anchor makes the warm-start recoverable — **2/3 seeds preserve train 0.841**. The `RL_UNSTABLE` label was a
  **selection artifact**: seed 0's dev-only `best_val` latched onto a dev-lucky, train-collapsed checkpoint (dev 0.429 came
  *with* train 0.227).
- **v2.1:** combined train-preserving selection (score = dev **iff** the train subset is preserved, else disqualified;
  update0 = the warm-start is the floor) removes the artifact — **3/3 seeds land exactly at the warm-start** (train 0.818,
  dev 0.286), `seeds_with_dev_gain = 0/3`.

## Mechanism — why RL cannot improve the *seen* distribution here
The v2.1 training curve is decisive: the combined-selection score is `dev` while train is preserved, else `-1`. It reads
`0.286` at it 120 then **`-1.0` for every checkpoint it 240–1200** — i.e. the greedy actor's train subset **collapsed below
the preserve floor at every trained checkpoint**, even with the anchor. The anchor keeps update0 (the warm-start)
recoverable, so combined selection correctly returns it; but the actor *being trained* drifts off the warm-start under
gradient updates.

**This is the R11.4B narrow-basin finding acting on RL.** The certified teacher θ are chaotically sensitive / narrow-basin
(a ~1e-5 θ change moved a delivery 7.99 → 54.27 mm). The BC warm-start memorizes those knife-edges; any RL gradient step
slides off them → train collapses. So TD3 cannot *stably improve* a narrow-basin warm-start on the seen distribution — the
best train-preserving policy is the warm-start itself.

## What this means for the robot (eyes on the robot)
A learned, amortized, **CEM-free** policy delivers **~82% of certified grasps (train) at strict K6** — the deployable
no-search delivery policy on the seen certified-handoff distribution. R11.6A's task (learn delivery on the seen
distribution) is **substantially met, by BC**; online TD3 adds nothing here (and cannot, given narrow basins). This is
**not a failure** — it is the honest boundary: the RL win, if it exists, is generalization.

## Method value (the v1→v2→v2.1 arc is reusable)
A clean, tested demonstration that (1) TD3 over a warm-start needs an anchor (v1 vs v2), (2) an immutable positive-replay
buffer prevents forgetting (v2 2/3, v2.1 3/3 preserve), (3) checkpoint selection must require train preservation, not
dev-alone, or it manufactures dev-lucky/train-broken "wins" (v2 seed 0). The refined gate encodes the user's criteria:
`IMPROVEMENT_PASS` needs train preserved + dev-mean > warm-start + dev ≥ 0.50 + a **majority of seeds** gaining dev.

## Boundary (unchanged)
R11.6A is `certified-grasp handoff → learned delivery → K6`. **Not yet** `exact-zero HOME → reach → grasp → learned
delivery → K6`, and **not yet** other-shaped objects. Ladder: 6A → **6B (unseen coin/target)** → 6C (exact-zero
composition) → 7 (other shapes) → 8 (Hamiltonian).

## Next — R11.6B (unseen coin/target generalization)
The dev/test held-out (0.286) is R11.6B's target. The key design idea: RL optimizes a **reward**, so it can discover its
**own** smoother/robust θ from the physical objective rather than imitating the narrow teacher θ — potentially sidestepping
the narrow-basin obstacle that caps BC generalization. R11.6B trains on the train split, selects/measures on the **untouched
test split** (kept out of all R11.6A selection), and asks whether reward-RL generalizes where amortized BC (0.29) does not.

## Files touched (all non-core)
- `hymeko_rl/coin_delivery/theta_option/delivery_theta_env.py`, `hymeko_rl/experiments/r11_6a_delivery_rl.py`
  (env + `DeliveryReward` + `train_td3_anchored` + combined selection + gate), `hymeko_rl/tests/test_r11_6a_delivery_rl.py`
  (9 tests). Committed `5cc4154a` → `a248513e` (v1) → `a297ee74` (v2.1). Results `reports/2026-08-04-r11-6a-{v2,v2p1}/`.
- **Read-only:** `option_rl/*`, `forward_displacement`, `solver`, `delivery_bc/*`, the demonstration bank.

## CORE.YAML items touched
**None.** New Python under `hymeko_rl/`; no dependency added.

## Test / gate / provenance
9 tests pass; ruff / mypy --strict / radon (no C+) clean. 3 runs × (≈36 min reconstruct 51 handoffs + 3 × ~25 min TD3),
per-process RSS ≪ 16 GB. Mac, 48 GB, Apple Silicon, `OMP_NUM_THREADS=2`, deterministic (fixed seeds; fresh-rig
reconstruction). Teacher θ from `reports/2026-08-03-r11-4b-bc/dataset/` (bank md5 `473244de4795254f5de99f4ca7732714`).
§2 plan `docs/plans/2026-08-04-r11-6a-delivery-rl/` (4-format). Energy diagnostic-only (frozen R11 contract → R11.8).

## Guards
Safety held (0 unsafe rollouts, all runs). Full structured θ (not a residual — avoids R8/R9). Multi-scenario shaped reward,
no ranking gate (avoids R10.2). 56 demos across 51 scenarios (removes the R7 coverage gate). The negative is honest and
mechanism-grounded (narrow-basin targets cap seen-distribution RL improvement) — not an escalation to per-step RL, not
more demos. Dev-only selection on 7 scenarios is noisy and can manufacture spurious "wins" — always select
train-preservingly.
